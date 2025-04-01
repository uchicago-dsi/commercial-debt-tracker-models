"""Shared test configuration and fixtures.

This file is automatically discovered by pytest and its fixtures are made available
to all test files in this directory and subdirectories.
"""

import os
import pytest
import logging
from pathlib import Path
import pandas as pd


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests.
    
    This fixture runs automatically for every test (autouse=True).
    It ensures each test starts with a clean logging configuration.
    """
    # Remove any existing handlers
    logger = logging.getLogger("cdt")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Reset the logger level
    logger.setLevel(logging.INFO)
    
    yield
    
    # Clean up after tests
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment variables after tests.
    
    This fixture runs automatically for every test (autouse=True).
    It ensures environment variables are restored to their original state.
    """
    # Store original SLURM_JOB_ID if it exists
    original_job_id = os.environ.get("SLURM_JOB_ID")
    
    yield
    
    # Restore original SLURM_JOB_ID or remove it if it didn't exist
    if original_job_id is None:
        if "SLURM_JOB_ID" in os.environ:
            del os.environ["SLURM_JOB_ID"]
    else:
        os.environ["SLURM_JOB_ID"] = original_job_id


@pytest.fixture
def test_data():
    """Create a small test dataset.
    
    This fixture can be used by any test that needs sample data.
    """
    return pd.DataFrame({
        "id": [f"test_{i}" for i in range(5)],
        "text": [f"This is test text {i}" for i in range(5)]
    })


@pytest.fixture
def test_prompt_file(tmp_path):
    """Create a temporary prompt file.
    
    Uses pytest's built-in tmp_path fixture to create a temporary directory.
    """
    prompt_file = tmp_path / "test-prompt.md"
    prompt_file.write_text("This is a test prompt")
    return prompt_file


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary output directory.
    
    Uses pytest's built-in tmp_path fixture to create a temporary directory.
    """
    output_dir = tmp_path / "llm_outputs" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_log_file(tmp_path):
    """Create a temporary log file path.
    
    Uses pytest's built-in tmp_path fixture to create a temporary directory.
    """
    return tmp_path / "test_logging.log"

