"""Utility functions for setting up logging in the CDT project."""

import logging
import os
import sys
from pathlib import Path


def setup_logging(log_file: Path = None) -> logging.Logger:
    """Configure logging for both file and console output.

    Args:
        log_file: Path to log file. If None, only console logging is setup.
    """
    logger = logging.getLogger("cdt")
    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | Job:%(slurm_job)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # File handler if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    return logger


def add_slurm_job_id(logger: logging.Logger) -> None:
    """Add SLURM job ID to the logger context.

    Args:
        logger: Logger instance.
    """
    job_id = os.getenv("SLURM_JOB_ID", "local")
    logger = logging.LoggerAdapter(logger, {"slurm_job": job_id})
    logger.info(f"Running job with ID: {job_id}")
