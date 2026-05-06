from zenml import pipeline
from zenml.config import DockerSettings, PythonPackageInstaller
import builtins, sys, logging

from utils.logger import setup_logging
from ETL.extract_from_mongoDB import query_data_warehouse
from ETL.transform import chunking_and_embedding
from ETL.load_into_vectorDB import load_into_vectorDB

# Initialize global logging first
setup_logging(logging.INFO)

# Redirect all print statements safely
builtins.print = lambda *args, **kwargs: (
    sys.stdout.write(" ".join(map(str, args)) + "\n")
    if not kwargs else builtins.__dict__["print"](*args, **kwargs)
)

docker_settings = DockerSettings(
    python_package_installer=PythonPackageInstaller.PIP
)

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def etl_pipeline():
    raw_documents = query_data_warehouse()
    embedded_documents = chunking_and_embedding(raw_documents)
    load_into_vectorDB(embedded_documents)


if __name__ == "__main__":
    etl_pipeline()
