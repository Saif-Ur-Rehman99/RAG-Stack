# Alfalah Investment Bot
A Production Ready RAG build with Optimizations and Scalability

## Project Setup
```
python -m venv venv
```
```
source venv/bin/activate
```
```
# Install Dependencies
pip install -r requirements.txt
```

#### get credentials for QDRANT
```
- create account
- get api_key
- get url
- paste in .env
```


## Run Ingestion Pipeline:

```
# Root Directory
python3 -m ETL.ingest_into_mongoDB
```

##### ZenML Dashboard 
- zenml login --local

## Run ETL Pipeline:
This extract all the documents from mongodb, make chunks and embeddings and load it into VectorDB
```
# Root Directory
python3 
```



from RAG.base.queries import Query, RAGPipeline, EmbeddedQuery
# from RAG.query_optimization.self_query import SelfQuery
# from RAG.query_optimization.query_expansion import QueryExpansion
# import random
from RAG.retrieval import ContextRetriever

# ---- 1. Define a mock RAG step ----
class MockRAGStep(RAGPipeline):
    def generate(self, query: Query, *args, **kwargs):
        if self._development:
            print("[DEV MODE] Using mock LLM output.")
            # Return a fake answer or processed query
            return f"Mock response for query: '{query.content}'"
        else:
            # In real mode, you would call an LLM or retrieval system here
            raise NotImplementedError("Real LLM generation not implemented yet.")


# ---- 2. Simulate pipeline ----
def main():

    # Step 1: Create a query
    # queries = [
    #     "Show me the Conventional fund summary for March 2023.",
    #     "Get Islamic fund report for January 2024.0",
    #     "Retrieve Conventional bank performance for 2022."
    # ]

    # Testing: Query Optimizations
    # user_input = "Show me the Conventional fund summary for March 2025."
    # query = Query.from_str(user_input)
    # print(f"Original Query: {query.content}")
    # print(f"Query ID: {query.id}")
    # print(f"Metadata: {query.metadata}\n")

    # # MetaData Extraction
    # _self_query = SelfQuery(development=True)
    # self_query = _self_query.generate(query)
    
    # # Query Expension
    # query_expander = QueryExpansion(development=True)
    # expanded_queries = query_expander.generate(query, n=2)
    
    # print("Query Expension\n")
    # for expanded_query in expanded_queries:
    #     print(expanded_query.content)
    
    # print(f"Self Query: {self_query}")


    # Testing: Retrieval
    retriever = ContextRetriever(development=False)
    query_text = "Show me the Conventional fund report for February 2026"

    results = retriever.search(query_text, k=3)

    print("RETRIEVAL RESULTS\n")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Filename: {doc.filename}")
        print(f"Bank Type: {doc.bank_type}")
        print(f"Month: {doc.month}")
        print(f"Year: {doc.year}")
        print(f"Content: {doc.content[:200]} ...")

    print("✅ Test completed successfully!")
    



if __name__ == "__main__":
    main()
