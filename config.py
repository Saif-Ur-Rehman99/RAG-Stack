from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MODEL API
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GROQ_API_KEY:   str | None = None
    
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    GOOGLE_MODEL_ID: str = "gemini-2.5-flash"
    GROQ_MODEL_ID:   str = "llama-3.3-70b-versatile"
    
    
    # Huggingface API (for embedding model)
    HF_TOKEN: str | None = None
    
    # Embedding MODEL
    TEXT_EMBEDDING_MODEL_ID:    str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_EMBEDDING_MODEL_ID:  str = "prithivida/Splade_PP_en_v1"                  # Hybrid Searching
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"  # Reranking
    
    # Document Parsing
    LLAMAPARSE_API_KEY: str | None = None
    
    RAG_MODEL_DEVICE: str = "cuda"


    # MongoDB Database
    DATABASE_HOST: str = "mongodb://127.0.0.1:27017/"
    DATABASE_NAME: str = "alfalah_investment"
    DATABASE_PORT: int = 27017


    # Qdrant VectorDB
    USE_QDRANT_CLOUD: bool = True
    QDRANT_API_KEY: str     | None = None
    QDRANT_CLOUD_URL: str   | None = None

    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333

    # vLLM embedding server (USE_VLLM_EMBEDDING=true to enable)
    USE_VLLM_EMBEDDING: bool = False
    VLLM_HOST: str = "localhost"
    VLLM_PORT: int = 8000
    VLLM_IMAGE: str = "vllm/vllm-openai:latest"
    VLLM_CONTAINER_NAME: str = "vllm-embedding-server"

    # TensorRT-LLM inference server
    TRTLLM_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"
    TRTLLM_HOST_PORT: int = 8000
    TRTLLM_IMAGE: str = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    TRTLLM_CONTAINER_NAME: str = "trtllm-server"
    TRTLLM_BACKEND: str = "tensorrt"            # "tensorrt" | "pytorch"
    TRTLLM_MAX_BATCH_SIZE: int = 32
    TRTLLM_MAX_NUM_TOKENS: int = 8192
    TRTLLM_MAX_SEQ_LEN: int = 4096

    # Comet ML (during training)
    # COMET_API_KEY: str | None = None
    # COMET_PROJECT: str = "twin"
    

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    # HF_MODEL_ID: str = "..."
    # GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    # SM_NUM_GPUS: int = 1
    # MAX_INPUT_LENGTH: int = 2048
    # MAX_TOTAL_TOKENS: int = 4096
    # MAX_BATCH_TOTAL_TOKENS: int = 4096
    # COPIES: int = 1  # Number of replicas
    # GPUS: int = 1  # Number of GPUs
    # CPUS: int = 2  # Number of CPU cores

    # SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    # SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    # TEMPERATURE_INFERENCE: float = 0.01
    # TOP_P_INFERENCE: float = 0.9
    # MAX_NEW_TOKENS_INFERENCE: int = 150


    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)
        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, 
        it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        try:
            print("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            print("Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file.")
            settings = Settings()

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            print(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )


settings = Settings.load_settings()