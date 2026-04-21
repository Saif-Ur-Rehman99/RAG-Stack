import re
from RAG.base.models import get_embedding_model

embedding_model = get_embedding_model()


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_markdown(text: str) -> str:
    """Normalize whitespace and remove HTML noise while preserving markdown structure."""
    text = re.sub(r"&#x[0-9a-fA-F]+;", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_markdown_by_heading(text: str) -> list[str]:
    """
    Split markdown into one chunk per heading (#, ##, ###, ...).
    Chunks that exceed the embedding model's token limit are further
    split by paragraph so no chunk is truncated at inference time,
    without mangling the original markdown text.
    """
    raw_chunks = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    max_tokens  = embedding_model.max_input_length
    tokenizer   = embedding_model.tokenizer

    def _token_count(s: str) -> int:
        return len(tokenizer.encode(s))

    _HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)

    final_chunks = []
    for chunk in chunks:
        if _token_count(chunk) <= max_tokens:
            final_chunks.append(chunk)
            continue

        # Oversized section: split by paragraph but always keep the heading
        # attached to the first content block — never emit a heading-only chunk.
        lines = chunk.split("\n", 1)
        heading = lines[0].strip() if _HEADING_RE.match(lines[0]) else ""
        body    = lines[1].strip() if len(lines) > 1 else (chunk if not heading else "")

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        # Seed current with the heading so it always bonds with the first paragraph
        current = heading
        first   = True
        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if first or _token_count(candidate) <= max_tokens:
                current = candidate
                first   = False
            else:
                final_chunks.append(current)
                current = para

        if current and current != heading:
            final_chunks.append(current)

    return final_chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    from langchain_text_splitters import ( 
        RecursiveCharacterTextSplitter, 
        SentenceTransformersTokenTextSplitter,
        )
    
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )

    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens

