import fitz
from typing_extensions import List, Tuple, Annotated
from zenml import step, log_metadata
from zenml.logger import get_logger
from artifacts.ingestion_artifacts import DedupedPDF, RoutedPDF


logger = get_logger(__name__)

# Number of pages sampled for complexity detection
_SAMPLE_PAGES = 3


def is_complex(pdf_path: str) -> bool:
    """Return True if the PDF should be routed to LlamaParse.

    Signals checked (any match → complex):
      1. Tables detected on sampled pages (primary signal for financial docs)
      2. Embedded images present (charts, figures, scanned content)
    """
    with fitz.open(pdf_path) as doc:
        for i in range(min(_SAMPLE_PAGES, doc.page_count)):
            page = doc[i]

            try:
                if page.find_tables().tables:
                    return True
            except AttributeError:
                # PyMuPDF < 1.23 has no find_tables(); fall back to image check only
                pass

            if page.get_images():
                return True

    return False


@step  # ← cache enabled (the default)
def route_pdfs(
    deduped: List[DedupedPDF],
) -> Tuple[
    Annotated[List[RoutedPDF], "simple_pdfs"],
    Annotated[List[RoutedPDF], "complex_pdfs"],
]:
    """Split non-duplicate PDFs into two lists by complexity.

    Duplicates are dropped here — they don't need to be parsed.
    Returns (simple_pdfs, complex_pdfs); each downstream parser
    step consumes one of these.
    """
    simple: List[RoutedPDF] = []
    complex_: List[RoutedPDF] = []

    for d in deduped:
        if d.is_duplicate:
            continue

        if is_complex(d.filepath):
            complex_.append(RoutedPDF(
                filepath=d.filepath,
                filename=d.filename,
                sha256=d.sha256,
                parser="LlamaParser",
            ))
        else:
            simple.append(RoutedPDF(
                filepath=d.filepath,
                filename=d.filename,
                sha256=d.sha256,
                parser="PyMuPDFParser",
            ))

    logger.info(
        f"Routed → PyMuPDF: {len(simple)}, LlamaParse: {len(complex_)}"
        f" (skipped {sum(1 for d in deduped if d.is_duplicate)} duplicate(s))"
    )
    log_metadata(metadata={
        "pymupdf_count": len(simple),
        "llama_count": len(complex_),
        "duplicates_skipped": sum(1 for d in deduped if d.is_duplicate),
    })

    return simple, complex_