from pathlib import Path
from abc import ABC, abstractmethod
import fitz
from config import settings


class BaseParser(ABC):
    @abstractmethod
    def parse(self, pdf_path: str) -> str:
        """Parse PDF and return full markdown string."""
        ...


class PyMuPDFParser(BaseParser):
    def parse(self, pdf_path: str) -> str:
        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("markdown")
                if text.strip():
                    pages.append(text)
        if not pages:
            raise RuntimeError(f"PyMuPDF returned empty content for: {pdf_path}")
        return "\n\n---\n\n".join(pages)


class LlamaParser(BaseParser):
    def __init__(self):
        from llama_parse import LlamaParse
        self._parser = LlamaParse(
            api_key=settings.LLAMAPARSE_API_KEY,
            parse_mode="parse_page_with_agent",
            model="anthropic-sonnet-4.0",
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
            result_type="markdown",
            verbose=False,
            language="en",
        )

    def parse(self, pdf_path: str) -> str:
        documents = self._parser.load_data(str(Path(pdf_path).resolve()))
        if not documents:
            raise RuntimeError(f"LlamaParse returned no documents for: {Path(pdf_path).name}")
        markdown = "\n\n---\n\n".join(doc.text for doc in documents)
        if not markdown.strip():
            raise RuntimeError(f"LlamaParse returned empty content for: {Path(pdf_path).name}")
        return markdown


