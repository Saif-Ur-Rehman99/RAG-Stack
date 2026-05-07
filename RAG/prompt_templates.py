from langchain_core.prompts import PromptTemplate
from RAG.base.prompts import PromptTemplateFactory
from langchain_core.output_parsers import PydanticOutputParser


class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """
    You are an AI language model assistant. 
    Your task is to generate {n} different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}
    """

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "n": n,
            },
        )


class SelfQueryTemplate(PromptTemplateFactory):
    prompt: str = """
        You are an AI language model assistant. Your task is to extract specific information from a user query about investment funds.

        Extract the following fields:
        - bank_type (e.g., Conventional, Islamic)
        - month (e.g., January, February, March, etc.)
        - year (e.g., 2023, 2024, etc.)

        If a field is missing, set its value to null.

        Format your response exactly as described below.

        {format_instructions}

        ---

        User question: {question}
    """

    def create_template(self, parser) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )


class RAGTemplate(PromptTemplateFactory):
    prompt: str = """You are a financial assistant specializing in fund manager reports for Alfalah Investments.

Answer the user's question using ONLY the information provided in the context below.
Be precise and concise. If the context does not contain enough information to answer fully, say so explicitly.
Do not make up any figures, fund names, or performance numbers.

Context:
{context}

Question: {question}

Answer:"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["context", "question"],
        )