"""Tests for logging functionality in the CDT project."""

import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from cdt.constants import BASE_DIR
from cdt.llm import run_model_on_data
from cdt.logging_utils import setup_logging, add_slurm_job_id


@pytest.fixture
def mock_model_and_tokenizer():
    """Mock the model and tokenizer to avoid loading actual models."""
    with patch("cdt.llm.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("cdt.llm.AutoTokenizer.from_pretrained") as mock_tokenizer:
        # Create mock objects
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Configure the mocks
        mock_model.return_value = model
        mock_tokenizer.return_value = tokenizer
        
        # Configure model
        model.device = "cuda"
        model.generate.return_value = [[1, 2, 3, 4, 5]]
        
        # Configure tokenizer
        tokenizer.apply_chat_template.return_value = "mock template"
        tokenizer.return_value = MagicMock(to=lambda x: MagicMock())
        tokenizer.batch_decode.return_value = ["mock response"]
        
        yield model, tokenizer


@pytest.fixture
def mock_model_output():
    """Mock output from run_model_with_output_validation."""
    return {
        "timestamp": "2024-04-01 09:05:53",
        "text_id": "test_id",
        "input_text": "test input",
        "response": "mock response",
        "inference_time": 1.0,
        "errors": [],
        "attempts": 1,
        "device": "test_device",
        "memory": "16GB",
        "max_allocated": "8GB",
        "reserved": "4GB",
        "clock_rate": "1.5GHz"
    }


@pytest.fixture
def mock_run_model(mock_model_output):
    """Mock the run_model_with_output_validation function."""
    with patch("cdt.llm.run_model_with_output_validation") as mock:
        mock.return_value = mock_model_output
        yield mock


def run_model_test(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model, expected_job_id):
    """Helper function to run model and verify logging."""
    # Set up logging
    logger = setup_logging(log_file=test_log_file)
    logger.info("Starting logging test")

    # Run the model
    output_file = test_output_dir / "test_model.csv"
    with patch("cdt.llm.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("cdt.llm.AutoTokenizer.from_pretrained") as mock_tokenizer:
        # Create mock objects
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Configure the mocks
        mock_model.return_value = model
        mock_tokenizer.return_value = tokenizer
        
        # Configure model
        model.device = "cuda"
        model.generate.return_value = [[1, 2, 3, 4, 5]]
        
        # Configure tokenizer
        tokenizer.apply_chat_template.return_value = "mock template"
        tokenizer.return_value = MagicMock(to=lambda x: MagicMock())
        tokenizer.batch_decode.return_value = ["mock response"]

        run_model_on_data(
            model_name="test-model",
            prompt_file=test_prompt_file,
            data=test_data,
            output_file=output_file,
            flush_every=1,
            max_retries=1,
        )

    # Read the log file and verify job ID
    log_content = test_log_file.read_text()
    assert f"[{expected_job_id}]" in log_content
    assert "Starting inference with model: test-model" in log_content
    assert "Finished processing all samples for test-model" in log_content

    return output_file


def test_logging_with_slurm_job_id(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model):
    """Test logging when SLURM_JOB_ID is set."""
    with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
        run_model_test(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model, "12345")


def test_logging_without_slurm_job_id(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model):
    """Test logging when SLURM_JOB_ID is not set."""
    # Ensure SLURM_JOB_ID is not set
    if "SLURM_JOB_ID" in os.environ:
        del os.environ["SLURM_JOB_ID"]
    
    run_model_test(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model, "local")


def test_logger_adapter_consistency():
    """Test that logger adapter consistently adds job ID context."""
    # Test with SLURM job ID
    with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
        logger = logging.getLogger("cdt")
        adapter = add_slurm_job_id(logger)
        assert isinstance(adapter, logging.LoggerAdapter)
        assert adapter.extra["job_id"] == "12345"

        # Test updating existing adapter
        adapter2 = add_slurm_job_id(adapter)
        assert adapter2 is adapter  # Should return same adapter
        assert adapter2.extra["job_id"] == "12345"

    # Test without SLURM job ID
    if "SLURM_JOB_ID" in os.environ:
        del os.environ["SLURM_JOB_ID"]
    logger = logging.getLogger("cdt")
    adapter = add_slurm_job_id(logger)
    assert isinstance(adapter, logging.LoggerAdapter)
    assert adapter.extra["job_id"] == "local"


def test_output_file_creation(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model):
    """Test that output files are created correctly."""
    with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
        output_file = run_model_test(test_data, test_prompt_file, test_output_dir, test_log_file, mock_run_model, "12345")

        # Verify output file exists and has correct content
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert len(df) == len(test_data)
        assert "prompt_commit" in df.columns
        assert "model" in df.columns
        assert "prompt_file" in df.columns
        assert all(df["model"] == "test-model") 