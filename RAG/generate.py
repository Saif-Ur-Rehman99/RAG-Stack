from typing import Callable
from config import settings
from utils.logger import logging as logger
from RAG.base.queries import Query, RAGPipeline


class RAGGenerator(RAGPipeline):
    def __init__(self, development: bool = False) -> None:
        super().__init__(development=development)
        if not development:
            self._llm: Callable[[str], str] = self._build_llm()

    @staticmethod
    def _build_llm() -> Callable[[str], str]:
        # if settings.GOOGLE_API_KEY:
        #     try:
        #         import google.generativeai as genai
        #         genai.configure(api_key=settings.GOOGLE_API_KEY)
        #         model = genai.GenerativeModel(settings.GOOGLE_MODEL_ID)
        #         return lambda prompt: model.generate_content(prompt).text
        #     except ImportError:
        #         pass

        # if settings.OPENAI_API_KEY:
        #     from openai import OpenAI
        #     client = OpenAI(api_key=settings.OPENAI_API_KEY)
        #     return lambda prompt: client.chat.completions.create(
        #         model=settings.OPENAI_MODEL_ID,
        #         messages=[{"role": "user", "content": prompt}],
        #     ).choices[0].message.content

        if settings.GROQ_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
            return lambda prompt: client.chat.completions.create(
                model=settings.GROQ_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content

        raise RuntimeError(
            "No LLM API key configured. Set GOOGLE_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in .env"
        )

    def generate(self, query: Query, prompt: str, *args, **kwargs) -> str:
        if self._development:
            return f"[DEV] Mock response for: {query.content}"

        answer = self._llm(prompt)
        logger.info(f"Response generated for: {query.content}")
        return answer
