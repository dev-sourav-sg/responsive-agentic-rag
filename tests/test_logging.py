import logging

from responsive_agentic_rag.observability.logging import (
    configure_logging,
    get_logger,
)


def test_get_logger_returns_named_logger():
    logger = get_logger("test.component")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.component"


def test_configure_logging():
    configure_logging()

    logger = get_logger("test.configuration")

    assert logger.isEnabledFor(logging.INFO)