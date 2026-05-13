import fitz
from pathlib import Path
from typing import Optional, List, Dict, Any
from models.ingestion_models import PDFDocument, IngestionStatus
from pymongo.errors import DuplicateKeyError
from zenml import step, log_metadata
from zenml.logger import get_logger
from artifacts.ingestion_artifacts import DedupedPDF, ParsedPDF, ParseFailure, PersistedPDF

logger = get_logger(__name__)



def _page_count(filepath: str) -> int:
    with fitz.open(filepath) as doc:
        return doc.page_count


def next_version(filename: str) -> int:
    latest = PDFDocument.find_sorted(sort=[("version", -1)], filename=filename)
    return (latest.version + 1) if latest else 1


def save_document(
    filepath: str,
    sha256: str,
    markdown: str,
    markdown_dir: Path,
    source: str = "local_folder",
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> PDFDocument:
    filename = Path(filepath).name
    stem = Path(filepath).stem

    markdown_path = markdown_dir / f"{stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    pages = _page_count(filepath)
    metadata: Dict[str, Any] = {"pages": pages, "source": source}
    if additional_metadata:
        metadata.update(additional_metadata)

    doc = PDFDocument(
        filename=filename,
        path=filepath,
        sha256=sha256,
        markdown_path=str(markdown_path),
        metadata=metadata,
        version=next_version(filename),
        status=IngestionStatus.INGESTED,
    )
    doc.save()
    return doc



@step(enable_cache=False)  # ← writes to Mongo + disk, never cache
def persist_pdfs(
        parsed_simple: List[ParsedPDF],
        parsed_complex: List[ParsedPDF],
        failures_simple: List[ParseFailure],
        failures_complex: List[ParseFailure],
        deduped: List[DedupedPDF],
        markdown_dir: str = "data/markdown",
        source: str = "data",
    ) -> List[PersistedPDF]:
    """Persist all parsed markdown to disk + Mongo, and emit one final
    record per file the pipeline saw (inserted / skipped / failed)."""
    md_dir = Path(markdown_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    results: List[PersistedPDF] = []

    # 1. Emit a record for every duplicate (skipped — already in DB)
    for d in deduped:
        if d.is_duplicate:
            results.append(PersistedPDF(
                filename=d.filename, status="skipped", pages=0,
            ))

    # 2. Persist every successful parse (simple + complex combined)
    for p in list(parsed_simple) + list(parsed_complex):
        try:
            doc = save_document(
                filepath=p.filepath,
                sha256=p.sha256,
                markdown=p.markdown,
                markdown_dir=md_dir,
                source=source,
            )
            results.append(PersistedPDF(
                filename=p.filename,
                status="inserted",
                pages=doc.metadata["pages"],
                doc_id=str(doc.id),
                parser=p.parser,
                version=doc.version,
            ))
            logger.info(
                f"[ok]     {p.filename}  ({p.parser}, "
                f"v{doc.version}, {doc.metadata['pages']}p)"
            )
        except DuplicateKeyError:
            # Race: another process inserted between dedup and persist
            results.append(PersistedPDF(
                filename=p.filename, status="skipped", pages=0, parser=p.parser,
            ))
            logger.warning(f"[skip]   {p.filename}  (duplicate key on insert)")
        except Exception as e:
            # Disk or Mongo write failed — don't crash, record it
            results.append(PersistedPDF(
                filename=p.filename, status="failed", pages=0,
                parser=p.parser, error=str(e),
            ))
            logger.error(f"[fail]   {p.filename}  on persist — {e}")

    # 3. Record parse failures (already failed upstream) as FAILED in Mongo
    for f in list(failures_simple) + list(failures_complex):
        try:
            PDFDocument(
                filename=f.filename,
                path=f.filepath,
                sha256=f.sha256,
                markdown_path="",
                metadata={"source": source, "error": f.error},
                version=next_version(f.filename),
                status=IngestionStatus.FAILED,
            ).save()
        except Exception as e:
            logger.warning(
                f"Could not record parse failure for {f.filename}: {e}"
            )
        results.append(PersistedPDF(
            filename=f.filename, status="failed", pages=0,
            parser=f.parser, error=f.error,
        ))

    # Counts for the dashboard
    inserted = sum(1 for r in results if r.status == "inserted")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    log_metadata(metadata={
        "inserted": inserted,
        "skipped": skipped,
        "failed": failed,
    })
    logger.info(
        f"Persist done — inserted={inserted}, skipped={skipped}, failed={failed}"
    )

    return results