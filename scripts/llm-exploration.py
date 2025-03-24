import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 16000

models = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistralai/Ministral-8B-Instruct-2410",
    # "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # "google/gemma-3-27b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
    # "Qwen/QwQ-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
with (BASE_DIR / "data" / "prompts" / "instructions.md").open(
    "r", encoding="utf-8"
) as f:
    instruction_text = f.read()

# === Read input CSV file (assumes columns "id" and "text") ===
input_df = pd.read_csv(BASE_DIR / "data" / "banktrack-8K-230501-01-v3.csv")


def run_auto_model_for_causalLM_on_prompt(
    model_name: str, instructions: str, data: pd.DataFrame, flush_every: int = 1
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    counter = 0

    # Process each row in the CSV
    for _, row in data.iterrows():
        sample_id = row["id"]
        input_text = row["text"]

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_text},
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Append result to our list
        results.append(
            {
                "input_text_id": sample_id,
                "input_text": input_text,
                "model_name": model_name,
                "response": response,
            }
        )

        counter += 1
        # After every N samples, write out the results to a CSV file.
        if counter % flush_every == 0:
            # Convert current results list into a DataFrame and append to CSV.
            pd.DataFrame(results).to_csv(
                "results.csv",
                mode="a",
                index=False,
                header=not pd.io.common.file_exists("results.csv"),
            )
            print(f"Flushed {counter} samples to results.csv")
            results = []  # clear the list after flushing

    # Write any remaining results after processing all rows
    if results:
        pd.DataFrame(results).to_csv(
            "results.csv",
            mode="a",
            index=False,
            header=not pd.io.common.file_exists("results.csv"),
        )
        print(f"Finished processing all samples for {model_name}")


if __name__ == "__main__":
    import submitit

    executor = submitit.AutoExecutor(folder=BASE_DIR / "logs")
    executor.update_parameters(
        slurm_partition="general",
        slurm_time="720:00",
        slurm_mem_per_cpu="256G",
        slurm_gres="gpu:1",
        slurm_job_name="commercial-debt-tracker",
        slurm_array_parallelism=6,
    )

    with executor.batch():
        for model in models:
            executor.submit(
                run_auto_model_for_causalLM_on_prompt, model, instruction_text, input_df
            )
