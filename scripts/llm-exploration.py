import argparse
import os
import time
from pathlib import Path

import pandas as pd
import submitit
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from cdt.postprocess import get_llm_output_errors


login(token=os.environ["HF_TOKEN"])

BASE_DIR = Path(__file__).resolve().parent.parent
MAX_NEW_TOKENS = 16000

models = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # "mistralai/Ministral-8B-Instruct-2410",
    # "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # "google/gemma-3-27b-it",
    # "google/gemma-3-4b-it",
    # "google/gemma-3-1b-it",
    "Qwen/QwQ-32B",
    #  "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
with (BASE_DIR / "data" / "prompts" / "combined-prompt.md").open(
    "r", encoding="utf-8"
) as f:
    instruction_text = f.read()

# === Read input CSV file (assumes columns "id" and "text") ===
input_df = pd.read_csv(BASE_DIR / "data" / "banktrack-8K-230501-01-v3.csv")


def get_gpu_details() -> dict[str, str]:
    """Retrieve information of GPU"""
    torch.cuda.reset_peak_memory_stats(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        "max_allocated": f"{torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB",
        "reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        "clock_rate": f"{torch.cuda.clock_rate(0) / 1e3:.2f} GHz",
    }


def run_model_with_output_validation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: str,
    sample: pd.Series,
    max_retries: int,
) -> dict:
    """Run the model on a single sample and validate the output

    Args:
        model: The model from AutoModelForCausalLM
        tokenizer: The tokenizer to use
        instructions: The instructions to use
        sample: row of dataframe with 'id' and 'text' columns
        max_retries: Maximum number of retries for a given sample
    """
    sample_id = sample["id"]
    input_text = sample["text"]
    error_message = ""
    for attempt_no in range(max_retries):
        messages = [
            {
                "role": "user",
                "content": f"{instructions}\n{error_message}\n{input_text}",
            },
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        start_time = time.perf_counter()
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
        end_time = time.perf_counter()
        gpu_information = get_gpu_details()
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        errors = get_llm_output_errors(response, input_text)
        if errors:
            error_message = "\nYour previous attempt had the following errors:\n"
            error_message += "\n".join(errors)
            error_message += "\nPlease correct these errors and try again."
        else:
            break
    results = {
        "input_text_id": sample_id,
        "input_text": input_text,
        "response": response,
        "inference_time": end_time - start_time,
        "errors": errors,
        "attempts": attempt_no + 1,
    }
    results.update(gpu_information)
    return results


def run_model_on_data(
    model_name: str,
    instructions: str,
    data: pd.DataFrame,
    flush_every: int = 1,
    max_retries: int = 3,
    output_file: str = None,
) -> None:
    """Run specified model based on instructions, performing basic checks on output

    Args:
        model_name (str): The model name to use from huggingface model hub
        instructions: Detailed instructions to be passed to the model
        data: The input data to be processed. Must have columns "id" and "text"
        flush_every: Number of samples to process before flushing to CSV
        max_retries: Maximum number of retries for a given sample
        output_file: The output file to write results to. If None, defaults to
            {model_name}-results.csv
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    counter = 0
    if output_file is None:
        output_file = f"{model_name.split('/')[-1]}-results.csv"
    for _, row in data.iterrows():
        result = run_model_with_output_validation(
            model, tokenizer, instructions, row, max_retries
        )
        results.append(result)
        counter += 1
        if counter % flush_every == 0:
            pd.DataFrame(results).to_csv(
                output_file,
                mode="a",
                index=False,
                header=False,
            )
            print(f"Flushed {counter} samples to {output_file}")
            results = []
    if results:
        pd.DataFrame(results).to_csv(
            output_file,
            mode="a",
            index=False,
            header=False,
        )
        print(f"Finished processing all samples for {model_name}")
        print(f"Results written to {output_file}")
        print(f"Total samples processed: {counter}")
        print(
            f"Average number of attempts per sample: {sum(r['attempts'] for r in results) / len(results):.2f}"
        )
        print(
            f"Average inference time per sample: {sum(r['inference_time'] for r in results) / len(results):.2f}"
        )


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder=BASE_DIR / "logs")
    executor.update_parameters(
        slurm_partition="general",
        slurm_time="720:00",
        slurm_mem_per_cpu="256G",
        slurm_gres="gpu:1",
        slurm_job_name="commercial-debt-tracker",
        slurm_array_parallelism=6,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--models", nargs="+", default=models)
    args = ap.parse_args()

    with executor.batch():
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
