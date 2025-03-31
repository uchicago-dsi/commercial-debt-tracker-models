"""Run LLMs on data using submitit for batch processing"

Runs specified LLM on input data and saves results to file.
Tracks:
- GPU usage
- Compute Time
- Prompt used (file name and commit hash for traceability)
- Model used
- When the model was run
- Number of attempts

By default, outputs are saved:
    data/llm_outputs/<run_name>/<model_name>.csv
    logs/llm_outputs/<run_name>.log
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import submitit
from huggingface_hub import login

from cdt.constants import BASE_DIR
from cdt.llm import run_model_on_data
from cdt.logging_utils import setup_logging

login(token=os.environ["HF_TOKEN"])


models = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistralai/Ministral-8B-Instruct-2410",
    # "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # "google/gemma-3-27b-it",
    "google/gemma-3-4b-it",
    # "google/gemma-3-1b-it",
    "Qwen/QwQ-32B",
    #  "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]
ap = argparse.ArgumentParser()
ap.add_argument("--local", action="store_true")
ap.add_argument("--models", nargs="+", default=models)
ap.add_argument("--gpu", default=None)
ap.add_argument(
    "--input_data",
    default=BASE_DIR / "data" / "banktrack-8K-230501-01-v3.csv",
    help="Path to input data file or directory",
)
ap.add_argument(
    "--prompt_file", default=BASE_DIR / "data" / "prompts" / "combined-prompt.md"
)
ap.add_argument(
    "--run_name",
    help="Identifier for run. By default, outputs are saved to data/llm_outputs/<run_name> and logs/llm_outputs/<run_name>.log",
    default=None,
)
ap.add_argument("--output_dir", default=None)
ap.add_argument(
    "--log_file",
    default=None,
    help="Directory for log files",
)
args = ap.parse_args()

if not args.run_name:
    args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

if not args.output_dir:
    args.output_dir = BASE_DIR / "data" / "llm_outputs" / args.run_name
# Set up logging
if not args.log_file:
    if not (BASE_DIR / "logs" / "runs").exists():
        (BASE_DIR / "logs" / "runs").mkdir(parents=True, exist_ok=True)
    args.log_file = BASE_DIR / "logs" / "runs" / f"{args.run_name}.log"
log_file = Path(args.log_file)
logger = setup_logging(log_file=log_file)
# Set up input data
if Path(args.input_data).exists() and Path(args.input_data).is_file():
    input_data_files = [Path(args.input_data)]
elif Path(args.input_data).exists() and Path(args.input_data).is_dir():
    input_data_files = list(Path(args.input_data).glob("*.csv"))
else:
    raise ValueError(f"Input data path {args.input_data} is not valid")
# Set up output directory
if not Path(args.output_dir).exists():
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
output_dir = Path(args.output_dir)
# Set up prompt file
if not Path(args.prompt_file).exists():
    raise ValueError(f"Prompt file {args.prompt_file} does not exist")
prompt_file = Path(args.prompt_file)
input_df = pd.read_csv(args.input_date)

# Slurm setup
if not args.gpu:
    gres = "gpu:1"
    logger.info("Using any available GPU")
else:
    gres = f"gpu:{args.gpu}:1"
    logger.info(f"Using GPU: {args.gpu}")
executor = submitit.AutoExecutor(folder=BASE_DIR / "logs")
executor.update_parameters(
    slurm_partition="general",
    slurm_time="720:00",
    slurm_mem_per_cpu="256G",
    slurm_gres=gres,
    slurm_job_name="commercial-debt-tracker",
    slurm_array_parallelism=6,
)
# Run all models on all data in batches
with executor.batch():
    for input_data_file in input_data_files:
        input_df = pd.read_csv(input_data_file)
        for model_name in args.models:
            output_file = output_dir / f"{model_name.split('/')[-1]}.csv"
            # filter out rows that have already been processed
            if Path(output_file).exists():
                processed_df = pd.read_csv(output_file)
                input_df = input_df[~input_df["text_id"].isin(processed_df["text_id"])]
            if args.local:
                logger.info("Running in local mode")
                run_model_on_data(model_name, prompt_file, input_df, output_file)
            else:
                logger.info("Running in SLURM mode")
                executor.submit(
                    run_model_on_data, model_name, prompt_file, input_df, output_file
                )
