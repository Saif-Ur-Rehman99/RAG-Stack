from RAG.base.queries import Query, RAGPipeline
from RAG.prompt_templates import RAGTemplate
from models.embedding_models import EmbeddedChunk


class ContextAugmenter(RAGPipeline):
    def __init__(self, development: bool = False) -> None:
        super().__init__(development=development)
        self._template = RAGTemplate().create_template()

    def generate(self, query: Query, chunks: list[EmbeddedChunk], *args, **kwargs) -> str:
        if self._development:
            return f"[DEV] Augmented prompt for: {query.content}"

        context = EmbeddedChunk.to_context(chunks)
        return self._template.format(context=context, question=query.content)
