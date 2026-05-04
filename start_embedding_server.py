import logging
from utils.logger import setup_logging
from inference.vllm_server import VLLMEmbeddingServer

setup_logging(logging.INFO)

if __name__ == "__main__":
    server = VLLMEmbeddingServer()
    server.start()
    server.stream_logs()   # blocks here — Ctrl-C to stop cleanly
