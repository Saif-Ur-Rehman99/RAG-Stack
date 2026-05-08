import opik
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from config import settings
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from RAG.base.queries import Query, RAGPipeline
from RAG.prompt_templates import SelfQueryTemplate



class MetadataParser(BaseModel):
    bank_type: Optional[str] = Field(None, description="Type of bank, e.g., 'Conventional' or 'Islamic'")
    month: Optional[str] = Field(None, description="Month mentioned in the query, e.g., 'March'")
    year: Optional[int] = Field(None, description="Year mentioned in the query, e.g., 2023")


class SelfQuery(RAGPipeline):
    # @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._development:
            return query

        parser = PydanticOutputParser(pydantic_object=MetadataParser)
        prompt = SelfQueryTemplate().create_template(parser=parser)

        # model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY, temperature=0)
        # model = ChatGoogleGenerativeAI(model=settings.GOOGLE_MODEL_ID, api_key=settings.GOOGLE_API_KEY, temperature=0)
        model = ChatGroq(model=settings.GROQ_MODEL_ID, api_key=settings.GROQ_API_KEY, temperature=0)

        chain = prompt | model
        response = chain.invoke({"question": query})

        try:
            # Use Pydantic parser to convert model output into MetadataParser instance
            metadata = parser.parse(response.content)
        except Exception as e:
            print(f"Failed to parse metadata: {e}")
            return query

        # Attach metadata to Query object
        query.metadata = {
            "bank_type": metadata.bank_type,
            "month": metadata.month,
            "year": metadata.year,
        }

        print(f"Extracted metadata: {query.metadata}")

        return query



if __name__ == "__main__":
    query = Query.from_str("Show me the Conventional fund summary for March 2023.")
    self_query = SelfQuery()
    query = self_query.generate(query)