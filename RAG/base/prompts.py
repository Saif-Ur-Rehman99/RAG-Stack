from abc import ABC, abstractmethod
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# Prompt Template Factory
class PromptTemplateFactory(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass