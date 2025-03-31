"""Functions for running and tracking LLM inference"""

from pathlib import Path
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cdt.postprocess import get_llm_output_errors
from cdt.utils import get_files_last_commit_hash

MAX_NEW_TOKENS = 16000


def get_gpu_details() -> dict[str, str]:
    """Retrieve information of GPU"""
    details = {
        "device": torch.cuda.get_device_name(0),
        "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        "max_allocated": f"{torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB",
        "reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        "clock_rate": f"{torch.cuda.clock_rate(0) / 1e3:.2f} GHz",
    }
    torch.cuda.reset_peak_memory_stats(0)
    return details


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
    messages = [
        {
            "role": "user",
            "content": f"{instructions}\n{input_text}",
        }
    ]

    for attempt_no in range(max_retries):  # noqa B007
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
            error_message = "Your previous response had these errors:\n" + "\n".join(
                errors
            )
            error_message += "\nPlease correct these errors and provide a new response."
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": error_message})
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


def append_results_to_csv(
    results: list[dict],
    output_file: str,
    prompt_commit: str,
    model_name: str,
    prompt_file: str,
) -> None:
    """Append results to CSV file

    Args:
        results: DataFrame containing results to append
        output_file: Path to the output CSV file
        prompt_commit: Commit hash of the prompt file
        model_name: Name of the model used
        prompt_file: Path to the prompt file
    """
    results = pd.DataFrame(results)
    results["prompt_commit"] = prompt_commit
    results["model"] = model_name
    results["prompt_file"] = prompt_file
    if not Path(output_file).exists():
        header = True
    else:
        header = False
    results.to_csv(
        output_file,
        mode="a",
        index=False,
        header=header,
    )
    print(f"Flushed {len(results)} samples to {output_file}")


def run_model_on_data(
    model_name: str,
    prompt_file: Path,
    data: pd.DataFrame,
    flush_every: int = 1,
    max_retries: int = 3,
    output_file: str = None,
) -> None:
    """Run specified model based on instructions, performing basic checks on output

    Args:
        model_name (str): The model name to use from huggingface model hub
        prompt_file: Path to Detailed instructions to be passed to the model
        data: The input data to be processed. Must have columns "id" and "text"
        flush_every: Number of samples to process before flushing to CSV
        max_retries: Maximum number of retries for a given sample
        output_file: The output file to write results to. If None, defaults to
            {model_name}-results.csv
    """
    with prompt_file.open("r", encoding="utf-8") as f:
        instructions = f.read()
    prompt_commit = get_files_last_commit_hash(prompt_file)
    dtype = "auto" if "gemma" not in model_name else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
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
            append_results_to_csv(
                results,
                output_file,
                prompt_commit,
                model_name,
                prompt_file,
            )
            results = []
    if results:
        append_results_to_csv(
            results,
            output_file,
            prompt_commit,
            model_name,
            prompt_file,
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
