import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing_extensions import Annotated
from zenml import get_step_context, step

from RAG.utilities import utils
from RAG.utilities.dispatcher import CleaningDispatcher, ChunkingDispatcher, EmbeddingDispatcher
from models.cleaning_models import CleanedDocument, PDFCleanedDocument
from models.chunking_models import Chunk
from models.embedding_models import EmbeddedChunk

_MAX_WORKERS = 5


def clean_documents(documents: Annotated[list, "raw_documents"]) -> Annotated[list, "cleaned_documents"]:
    """Clean raw documents using the appropriate handler (PDF, image, etc.)."""
    cleaned_documents = []

    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    # Log and store metadata for tracking
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents",
        metadata=_get_clean_metadata(cleaned_documents)
    )

    # print(cleaned_documents[0])
    return cleaned_documents


def chunk_and_embed(cleaned_documents: Annotated[list, "cleaned_documents"]) -> Annotated[list, "embedded_documents"]:
    """
    Step that performs:
    1. Chunking of cleaned documents into smaller text sections.
    2. Embedding each chunk into a numerical vector.
    """

    metadata = {
        "chunking": {},
        "embedding": {},
        "num_documents": len(cleaned_documents),
    }

    embedded_chunks: list[EmbeddedChunk] = []

    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        # 2. EMBEDDING LOOP
        for batched_chunks in utils.batch(chunks, size=10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    # 3. METADATA
    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    # 4. SAVE METADATA
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    # print(embedded_chunks[0])
    return embedded_chunks


def _process_single_document(document) -> tuple[list[EmbeddedChunk], Exception | None]:
    """Chunk and embed one document. Returns (embedded_chunks, error)."""
    try:
        m = document.metadata
        cleaned = PDFCleanedDocument(
            content   = document.page_content,
            filename  = m.get("filename"),
            path      = m.get("path"),
            sha256    = m.get("sha256"),
            bank_type = m.get("bank_type"),
            month     = m.get("month"),
            year      = m.get("year"),
        )
        chunks = ChunkingDispatcher.dispatch(cleaned)
        embedded: list[EmbeddedChunk] = []
        for batched in utils.batch(chunks, size=10):
            embedded.extend(EmbeddingDispatcher.dispatch(batched))
        return embedded, None
    except Exception as exc:
        return [], exc


@step
def chunking_and_embedding(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "embedded_documents"]:
    
    """Chunk, and Embed each document in parallel (max 5 workers)."""


    from RAG.base.models import EmbeddingModelSingleton
    EmbeddingModelSingleton()  # warm up singleton in main thread before workers start

    log = logging.getLogger(__name__)
    all_embedded: list[EmbeddedChunk] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_process_single_document, doc): doc for doc in documents}
        for future in as_completed(futures):
            embedded, error = future.result()
            if error:
                errors.append(str(error))
                log.warning("Document processing failed: %s", error)
            else:
                all_embedded.extend(embedded)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_documents",
        metadata={
            "num_documents": len(documents),
            "num_embedded_chunks": len(all_embedded),
            "num_errors": len(errors),
            "max_workers": _MAX_WORKERS,
        },
    )

    return all_embedded


# ---- Utiliy Functions --
def _get_clean_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    """Generate cleaning step metadata summary."""
    metadata = {
        "num_documents": len(cleaned_documents),
        "categories": {},
        "bank_types": set(),
        "months": set(),
        "years": set(),
    }

    for doc in cleaned_documents:
        category = getattr(doc, "category", "unknown")

        if category not in metadata["categories"]:
            metadata["categories"][category] = {"num_documents": 0}

        metadata["categories"][category]["num_documents"] += 1

        # Track domain-specific metadata
        bank_type = getattr(doc, "bank_type", None)
        month = getattr(doc, "month", None)
        year = getattr(doc, "year", None)

        if bank_type:
            metadata["bank_types"].add(bank_type)
        if month:
            metadata["months"].add(month)
        if year:
            metadata["years"].add(year)

    # Convert sets to lists (JSON serializable)
    metadata["bank_types"] = list(metadata["bank_types"])
    metadata["months"] = list(metadata["months"])
    metadata["years"] = list(metadata["years"])

    return metadata


def _add_chunks_metadata(chunks: list[Chunk], metadata: dict) -> dict:
    """Collect metadata related to chunking."""
    for chunk in chunks:
        category = chunk.get_category()
        if category not in metadata:
            metadata[category] = chunk.metadata.copy()

        metadata[category]["num_chunks"] = metadata[category].get("num_chunks", 0) + 1
        metadata[category].update({
            "filename": chunk.metadata.get("filename"),
            "sha256": chunk.metadata.get("sha256"),
            "bank_type": chunk.metadata.get("bank_type"),
            "month": chunk.metadata.get("month"),
            "year": chunk.metadata.get("year"),
            "chunking_version": chunk.metadata.get("chunking_version"),
            "chunking_strategy": chunk.metadata.get("chunking_strategy"),
            "chunking_size": chunk.metadata.get("chunking_size"),
            "chunking_overlap": chunk.metadata.get("chunking_overlap"),
            "chunking_status": chunk.metadata.get("chunking_status"),
        })

    return metadata


def _add_embeddings_metadata(embedded_chunks: list[EmbeddedChunk], metadata: dict) -> dict:
    """Collect metadata related to embeddings."""
    for embedded_chunk in embedded_chunks:
        category = embedded_chunk.get_category()
        if category not in metadata:
            metadata[category] = embedded_chunk.metadata.copy()

        metadata[category].update({
            "filename": embedded_chunk.metadata.get("filename"),
            "sha256": embedded_chunk.metadata.get("sha256"),
            "bank_type": embedded_chunk.metadata.get("bank_type"),
            "month": embedded_chunk.metadata.get("month"),
            "year": embedded_chunk.metadata.get("year"),
            "embedding_model": embedded_chunk.metadata.get("embedding_model"),
            "embedding_dimensions": embedded_chunk.metadata.get("embedding_dimensions"),
            "embedding_version": embedded_chunk.metadata.get("embedding_version"),
            "embedding_status": embedded_chunk.metadata.get("embedding_status"),
        })

    return metadata



