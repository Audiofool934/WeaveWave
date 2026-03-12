"""Centralized logging configuration for WeaveWave."""

import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    name: str | None = None,
) -> logging.Logger:
    """Configure and return a logger with a standard format.

    Args:
        level: Logging level (default ``logging.INFO``).
        name: Logger name.  When *None* the root logger is configured.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
