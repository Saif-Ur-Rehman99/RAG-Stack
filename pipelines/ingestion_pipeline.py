import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pymongo.errors import DuplicateKeyError
from zenml import step, pipeline, log_metadata

from models.ingestion_models import PDFDocument, IngestionStatus
from uploader.deduplication import is_duplicate
from uploader.router import is_complex
from uploader import storage
from RAG.indexing.parser import PyMuPDFParser, LlamaParser

# fix the cleaning later
from uploader.load_files import discover_pdfs
from uploader.deduplication import deduplicate_pdfs
from uploader.router import route_pdfs
from uploader.storage import persist_pdfs
from RAG.indexing.parsing_step import llama_parse, pymupdf_parse
from artifacts.ingestion_artifacts import IngestionReport, PersistedPDF



MAX_RETRIES  = 3
RETRY_DELAY  = 2
MAX_WORKERS  = 4
MARKDOWN_DIR = Path("data/markdown")


def _select_parser(pdf_path: str):
    if is_complex(pdf_path):
        return LlamaParser() 
    return PyMuPDFParser()


def process_pdf(
    filepath: str,
    source: str = "local_folder",
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    filename = Path(filepath).name

    dup, sha = is_duplicate(filepath, MARKDOWN_DIR)
    if dup:
        tqdm.write(f"  [skip]   {filename}")
        return {"filename": filename, "status": "skipped", "pages": 0, "id": None}

    parser = _select_parser(filepath)
    parser_name = type(parser).__name__
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            markdown = parser.parse(filepath)
            doc = storage.save_document(filepath, sha, markdown, MARKDOWN_DIR, source, additional_metadata)
            tqdm.write(f"  [ok]     {filename}  ({parser_name}, v{doc.version}, {doc.metadata['pages']}p)")
            return {"filename": filename, "status": "inserted", "pages": doc.metadata["pages"], "id": str(doc.id)}

        except DuplicateKeyError:
            tqdm.write(f"  [skip]   {filename}  (duplicate key)")
            return {"filename": filename, "status": "skipped", "pages": 0, "id": None}

        except Exception as e:
            last_error = e
            tqdm.write(f"  [retry]  {filename}  attempt {attempt}/{MAX_RETRIES} — {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    tqdm.write(f"  [fail]   {filename}")
    try:
        PDFDocument(
            filename=filename,
            path=filepath,
            sha256=sha,
            markdown_path="",
            metadata={"source": source, "error": str(last_error)},
            version=storage.next_version(filename),
            status=IngestionStatus.FAILED,
        ).save()
    except Exception:
        pass

    return {"filename": filename, "status": "failed", "pages": 0, "id": None}


def ingest_directory(
    data_dir: str = "data/documents",
    source: str = "data",
    recursive: bool = True,
) -> Dict[str, Any]:
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise ValueError(f"Directory not found: {data_dir}")

    pdf_files = list(
        data_dir_path.rglob("*.pdf") if recursive else data_dir_path.glob("*.pdf")
    )
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    inserted, skipped, failed = 0, 0, 0
    documents: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pdf, str(p), source): p for p in pdf_files}
        for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Ingesting PDFs", unit="file"):
            result = future.result()
            documents.append(result)
            if result["status"] == "inserted":
                inserted += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1

    return {
        "total_found"       : len(pdf_files),
        "inserted"          : inserted,
        "skipped_duplicates": skipped,
        "failed"            : failed,
        "documents"         : documents,
    }


@step
def summarize(persisted: List[PersistedPDF]) -> IngestionReport:
    """Build the final IngestionReport and print a human-readable summary."""
    inserted = sum(1 for p in persisted if p.status == "inserted")
    skipped = sum(1 for p in persisted if p.status == "skipped")
    failed = sum(1 for p in persisted if p.status == "failed")

    report = IngestionReport(
        total_found=len(persisted),
        inserted=inserted,
        skipped_duplicates=skipped,
        failed=failed,
        documents=persisted,
    )

    print("\n─── Ingestion Summary ────────────────────────")
    print(f"  Total found : {report.total_found}")
    print(f"  Inserted    : {report.inserted}")
    print(f"  Skipped     : {report.skipped_duplicates}")
    print(f"  Failed      : {report.failed}")
    print(f"\n  {'Filename':<45} {'Pages':>6}  Status")
    print(f"  {'-' * 45} {'------':>6}  ------")
    for d in report.documents:
        print(f"  {d.filename:<45} {d.pages:>6}  {d.status}")
    print("──────────────────────────────────────────────\n")

    log_metadata(metadata={
        "inserted": inserted,
        "skipped": skipped,
        "failed": failed,
        "total": len(persisted),
    })
    return report


@pipeline(enable_cache=True)
def ingestion_pipeline(
    data_dir: str = "data/documents",
    source: str = "data",
    markdown_dir: str = "data/markdown",
    recursive: bool = True,
) -> None:
    candidates = discover_pdfs(data_dir=data_dir, recursive=recursive)
    deduped = deduplicate_pdfs(candidates=candidates, markdown_dir=markdown_dir)

    simple_pdfs, complex_pdfs = route_pdfs(deduped=deduped)

    parsed_simple, failures_simple = pymupdf_parse(simple_pdfs=simple_pdfs)
    parsed_complex, failures_complex = llama_parse(complex_pdfs=complex_pdfs)

    persisted = persist_pdfs(
        parsed_simple=parsed_simple,
        parsed_complex=parsed_complex,
        failures_simple=failures_simple,
        failures_complex=failures_complex,
        deduped=deduped,
        markdown_dir=markdown_dir,
        source=source,
    )

    summarize(persisted=persisted)

if __name__ == "__main__":
    ingestion_pipeline()