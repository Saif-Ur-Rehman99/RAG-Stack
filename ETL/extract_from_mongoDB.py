import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from zenml import step, log_metadata

from models.ingestion_models import PDFDocument, IndexingStatus
from RAG.base.constants import DataCategory


_MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{4})"
)


def _extract_pdf_info(filename: str) -> Tuple[str, str, str]:
    """Extract bank_type, month, year from filename (case-sensitive)."""
    name = Path(filename).stem.replace("–", "-")

    if "Conventional" in name:
        bank_type = "Conventional"
    elif "Islamic" in name:
        bank_type = "Islamic"
    else:
        bank_type = "unknown"

    m = _MONTH_RE.search(name)
    month, year = (m.group(1), m.group(2)) if m else ("unknown", "unknown")

    return bank_type, month, year


@step
def query_data_warehouse() -> List[Document]:
    """Fetch un-indexed PDFs from MongoDB and return LangChain Documents."""
    records: List[PDFDocument] = PDFDocument.bulk_find(
        indexing_status=IndexingStatus.PENDING
    )

    langchain_docs: List[Document] = []
    skipped = 0

    for rec in records:
        md_path = Path(rec.markdown_path) if rec.markdown_path else None
        if not md_path or not md_path.is_file():
            print(f"  [skip] {rec.filename}: markdown missing at {rec.markdown_path!r}")
            skipped += 1
            continue

        bank_type, month, year = _extract_pdf_info(rec.filename)

        langchain_docs.append(Document(
            page_content=md_path.read_text(encoding="utf-8"),
            metadata={
                "filename" : rec.filename,
                "path"     : rec.path,
                "pages"    : rec.metadata.get("pages"),
                "source"   : rec.metadata.get("source"),
                "category" : DataCategory.PDF,
                "sha256"   : rec.sha256,
                "bank_type": bank_type,
                "month"    : month,
                "year"     : year,
            },
        ))

    log_metadata(metadata={
        "total_fetched" : len(records),
        "converted"     : len(langchain_docs),
        "skipped"       : skipped,
    })
    print(f"Fetched {len(records)} records → {len(langchain_docs)} documents ({skipped} skipped).")
    return langchain_docs


if __name__ == "__main__":
    docs = query_data_warehouse()
    print(docs[0].metadata)



