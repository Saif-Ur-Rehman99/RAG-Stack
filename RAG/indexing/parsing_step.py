import time
from typing_extensions import List, Tuple, Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed
from zenml import step, log_metadata
from zenml.logger import get_logger

from RAG.indexing.parser import PyMuPDFParser, LlamaParser, BaseParser
from artifacts.ingestion_artifacts import RoutedPDF, ParsedPDF, ParseFailure

logger = get_logger(__name__)



def _parse_one(
    parser: BaseParser,
    item: RoutedPDF,
    max_retries: int,
    retry_delay: int,
    ) -> ParsedPDF | ParseFailure:
    """Run a single PDF through the given parser, with retries.
    Returns ParsedPDF on success, ParseFailure on exhausted retries."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            md = parser.parse(item.filepath)
            return ParsedPDF(
                filepath=item.filepath,
                filename=item.filename,
                sha256=item.sha256,
                parser=item.parser,
                markdown=md,
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"[retry] {item.filename} ({item.parser}) "
                f"attempt {attempt}/{max_retries} — {e}"
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    return ParseFailure(
        filepath=item.filepath,
        filename=item.filename,
        sha256=item.sha256,
        parser=item.parser,
        error=str(last_error),
    )


def _run_parser_concurrent(
    parser: BaseParser,
    items: List[RoutedPDF],
    max_workers: int,
    max_retries: int,
    retry_delay: int,
    ) -> Tuple[List[ParsedPDF], List[ParseFailure]]:
    """Run a list of files through one parser, in parallel."""
    successes: List[ParsedPDF] = []
    failures: List[ParseFailure] = []

    if not items:
        return successes, failures

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_parse_one, parser, it, max_retries, retry_delay): it
            for it in items
        }
        for fut in as_completed(futures):
            result = fut.result()
            if isinstance(result, ParsedPDF):
                successes.append(result)
            else:
                failures.append(result)

    return successes, failures


# Step 4a — PyMuPDF (simple PDFs)
@step
def pymupdf_parse(
    simple_pdfs: List[RoutedPDF],
    max_workers: int = 4,
    max_retries: int = 2,       # cheap & local — fewer retries
    retry_delay: int = 1,
) -> Tuple[
    Annotated[List[ParsedPDF], "pymupdf_parsed"],
    Annotated[List[ParseFailure], "pymupdf_failures"],
]:
    """Parse simple PDFs locally with PyMuPDF (fast, cheap, deterministic)."""
    parser = PyMuPDFParser()
    successes, failures = _run_parser_concurrent(
        parser, simple_pdfs, max_workers, max_retries, retry_delay,
    )
    logger.info(
        f"PyMuPDF: {len(successes)} parsed, {len(failures)} failed "
        f"(of {len(simple_pdfs)} routed)"
    )
    log_metadata(metadata={
        "parsed": len(successes),
        "failed": len(failures),
        "total": len(simple_pdfs),
    })
    return successes, failures


# Step 4b — LlamaParse (complex PDFs) 
@step  # cached — same SHAs → same result, saves API $$$
def llama_parse(
    complex_pdfs: List[RoutedPDF],
    max_workers: int = 4,
    max_retries: int = 3,       # network-bound — more retries
    retry_delay: int = 2,
) -> Tuple[
    Annotated[List[ParsedPDF], "llama_parsed"],
    Annotated[List[ParseFailure], "llama_failures"],
]:
    """Parse complex PDFs (tables, images, scanned) with LlamaParse API."""
    if not complex_pdfs:
        # Skip the LlamaParse() constructor entirely if there's no work —
        # avoids loading the SDK and validating the API key for nothing.
        logger.info("LlamaParse: no complex PDFs routed, skipping.")
        log_metadata(metadata={"parsed": 0, "failed": 0, "total": 0})
        return [], []

    parser = LlamaParser()
    successes, failures = _run_parser_concurrent(
        parser, complex_pdfs, max_workers, max_retries, retry_delay,
    )
    logger.info(
        f"LlamaParse: {len(successes)} parsed, {len(failures)} failed "
        f"(of {len(complex_pdfs)} routed)"
    )
    log_metadata(metadata={
        "parsed": len(successes),
        "failed": len(failures),
        "total": len(complex_pdfs),
    })
    return successes, failures