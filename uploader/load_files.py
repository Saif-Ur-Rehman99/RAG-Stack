from pathlib import Path
from typing import List
from zenml import step, log_metadata
from zenml.logger import get_logger
from artifacts.ingestion_artifacts import PDFCandidate


logger = get_logger(__name__)


@step(enable_cache=False)  # filesystem changes aren't captured by ZenML's cache key
def discover_pdfs(
    data_dir: str = "data/documents",
    recursive: bool = True,
) -> List[PDFCandidate]:
    """Walk the directory and return every PDF found.

    Raises if data_dir is missing or not a directory. Returns an empty list
    if the directory exists but contains no PDFs (downstream steps handle
    that case cleanly).
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists() or not data_dir_path.is_dir():
        raise ValueError(f"Directory not found: {data_dir}")

    pdfs = list(
        data_dir_path.rglob("*.pdf") if recursive else data_dir_path.glob("*.pdf")
    )
    candidates = [PDFCandidate(filepath=str(p), filename=p.name) for p in pdfs]

    logger.info(f"Discovered {len(candidates)} PDF(s) in {data_dir}")
    log_metadata(metadata={
        "data_dir": data_dir,
        "recursive": recursive,
        "num_pdfs": len(candidates),
    })
    return candidates