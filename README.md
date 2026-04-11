# RAG Stack
A Production Ready RAG Framework build with Optimizations, Scalability and Easy Integrations

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
