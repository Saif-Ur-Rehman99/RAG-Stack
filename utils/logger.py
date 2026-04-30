import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure clean, global logging for the entire project."""

    # Reset existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up stream handler for terminal output
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Set the base logging level
    root_logger.setLevel(level)

    # Silence noisy libraries
    logging.getLogger("zenml").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Optional: customize other third-party loggers if needed
    # logging.getLogger("some_other_lib").setLevel(logging.ERROR)

    logging.getLogger(__name__).info("✅ Logging initialized successfully.")
