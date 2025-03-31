"""Run LLMs on data using submitit for batch processing."""

import argparse
import os
from pathlib import Path

import pandas as pd
import submitit
from huggingface_hub import login

from cdt.constants import BASE_DIR
from cdt.llm import run_model_on_data

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
with (BASE_DIR / "data" / "prompts" / "combined-prompt.md").open(
    "r", encoding="utf-8"
) as f:
    instruction_text = f.read()

# === Read input CSV file (assumes columns "id" and "text") ===
input_df = pd.read_csv(BASE_DIR / "data" / "banktrack-8K-230501-01-v3.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--models", nargs="+", default=models)
    ap.add_argument("--gpu", default=None)
    ap.add_argument(
        "--input_data",
        default=BASE_DIR / "data" / "banktrack-8K-230501-01-v3.csv",
    )
    ap.add_argument(
        "--prompt_file", default=BASE_DIR / "data" / "prompts" / "combined-prompt.md"
    )
    ap.add_argument(
        "--output_file",
        default=BASE_DIR / "data" / "llm_outputs" / "commercial-debt-tracker.csv",
    )
    args = ap.parse_args()

    if not args.gpu:
        gres = "gpu:1"
        print("Using any available GPU")
    else:
        gres = f"gpu:{args.gpu}:1"
        print(f"Using GPU: {args.gpu}")

    if Path(args.input_data).exists() and Path(args.input_data).is_file():
        input_data_files = [Path(args.input_data)]
    elif Path(args.input_data).exists() and Path(args.input_data).is_dir():
        input_data_files = list(Path(args.input_data).glob("*.csv"))
    else:
        raise ValueError(f"Input data path {args.input_data} is not valid")

    with Path(args.prompt_file).open("r", encoding="utf-8") as f:
        instruction_text = f.read()
    input_df = pd.read_csv(args.input_date)

    executor = submitit.AutoExecutor(folder=BASE_DIR / "logs")
    executor.update_parameters(
        slurm_partition="general",
        slurm_time="720:00",
        slurm_mem_per_cpu="256G",
        slurm_gres=gres,
        slurm_job_name="commercial-debt-tracker",
        slurm_array_parallelism=6,
    )

    with executor.batch():
        for input_data_file in input_data_files:
            input_df = pd.read_csv(input_data_file)
            for model in args.models:
                if args.local:
                    run_model_on_data(model, instruction_text, input_df)
                else:
                    executor.submit(
                        run_model_on_data,
                        model,
                        instruction_text,
                        input_df,
                    )
