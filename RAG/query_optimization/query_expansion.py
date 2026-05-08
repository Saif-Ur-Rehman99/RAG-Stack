# import opik
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from config import settings
from RAG.base.queries import Query, RAGPipeline
from RAG.prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGPipeline):
    # @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, n: int) -> list[Query]:
        assert n > 0, f"'expand_to_n' should be greater than 0. Got {n}."

        if self._development:
            return [query for _ in range(n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(n=n-1)
       
        model = ChatGroq(model=settings.GROQ_MODEL_ID, api_key=settings.GROQ_API_KEY, temperature=0)
        # model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)
        # model = ChatGoogleGenerativeAI(model=settings.GOOGLE_MODEL_ID, api_key=settings.GOOGLE_API_KEY, temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query})
        result = response.content

        raw_parts = result.strip().split(query_expansion_template.separator)[:n - 1]

        # Take the last non-empty line of each part to strip any LLM preamble
        queries_content = []
        for part in raw_parts:
            lines = [l.strip() for l in part.strip().splitlines() if l.strip()]
            if lines:
                queries_content.append(lines[-1])

        queries = [query]
        queries += [query.replace_content(content) for content in queries_content]

        return queries


if __name__ == "__main__":
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion(development=True)
    expanded_queries = query_expander.generate(query, n=2)
    for expanded_query in expanded_queries:
        print(expanded_query.content)