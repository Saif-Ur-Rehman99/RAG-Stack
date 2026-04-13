"""
This category helps your vector database layer, data storage, or RAG pipeline know what kind of data this model represents.
For example: You might have different collections/tables in your DB:
    - prompts
    - queries
    - articles
    - documents
"""

class DataCategory:
    PROMPT = "prompt"
    QUERIES = "queries"
    
    PDF = "pdf"
    IMAGES = "images"

    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"