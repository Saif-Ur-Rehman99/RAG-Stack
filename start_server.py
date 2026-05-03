import logging
from utils.logger import setup_logging
from inference.server import TRTLLMServer

setup_logging(logging.INFO)

if __name__ == "__main__":
    server = TRTLLMServer()
    server.start()
    server.stream_logs()   # blocks here — Ctrl-C to stop cleanly
