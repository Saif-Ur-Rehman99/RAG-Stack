import hashlib
from typing import List
from pathlib import Path
from config import settings
from db.mongo import MongoDatabaseConnector
from models.ingestion_models import PDFDocument
from zenml.logger import get_logger
from zenml import step, log_metadata
from artifacts.ingestion_artifacts import PDFCandidate, DedupedPDF


logger = get_logger(__name__)


def file_sha256(path: str, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_duplicate(filepath: str, markdown_dir: Path) -> tuple[bool, str]:
    """Return (duplicate, sha256).

    Also repairs corrupt DB records (sha matches but markdown missing/empty).
    """
    sha = file_sha256(filepath)
    stem = Path(filepath).stem

    existing = PDFDocument.find(sha256=sha)
    if existing:
        md = Path(existing.markdown_path)
        if existing.markdown_path and md.exists() and md.stat().st_size > 0:
            return True, sha
        # Corrupt record — remove so the file gets re-processed
        db = MongoDatabaseConnector()[settings.DATABASE_NAME]
        db[PDFDocument.get_collection_name()].delete_one({"sha256": sha})

    markdown_path = markdown_dir / f"{stem}.md"
    if markdown_path.exists() and markdown_path.stat().st_size > 0:
        return True, sha

    return False, sha



@step(enable_cache=False)  # side effects on Mongo, never cache
def deduplicate_pdfs(
        candidates: List[PDFCandidate],          # ARTIFACT from previous step
        markdown_dir: str = "data/markdown",     # PARAMETER (configurable, tracked)
    ) -> List[DedupedPDF]:                       # typed return = clean artifact
    """Hash each candidate file and check if it has already been ingested.

    For each PDF this:
      1. Computes its SHA-256
      2. Checks Mongo for an existing record
      3. Deletes any corrupt Mongo record (markdown missing/empty)
      4. Checks the local markdown dir for an existing file

    Returns one DedupedPDF per input, with is_duplicate=True when the
    file has already been fully ingested.
    """
    md_dir = Path(markdown_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    out: List[DedupedPDF] = []
    for c in candidates:
        dup, sha = is_duplicate(c.filepath, md_dir)
        out.append(DedupedPDF(
            filepath=c.filepath,
            filename=c.filename,
            sha256=sha,
            is_duplicate=dup,
        ))

    n_dup = sum(1 for d in out if d.is_duplicate)
    n_new = len(out) - n_dup
    logger.info(f"Dedup: {n_dup} duplicate(s), {n_new} new file(s)")

    # Show counts in the dashboard sidebar for this step run
    log_metadata(metadata={
        "total": len(out),
        "duplicates": n_dup,
        "new": n_new,
    })
    return out