"""
Structured logger for the Modular RAG MCP Server.

This module provides a centralized logging facility with JSON-formatted output.
All logs are written to stderr to avoid interfering with MCP stdio communication.

Author: Modular RAG MCP Server Project
License: MIT
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> from src.observability.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "abc123"})
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def set_log_level(level: int) -> None:
    """
    Set the global logging level for all loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.basicConfig(level=level, stream=sys.stderr, force=True)
