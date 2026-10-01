import logging


DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = DEFAULT_LOG_LEVEL) -> None:
    """Configure application-wide logging."""
    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format=LOG_FORMAT,
        )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module or component name."""
    return logging.getLogger(name)