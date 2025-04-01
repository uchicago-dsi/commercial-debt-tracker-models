"""Evaluate LLM outputs across multiple runs and models.

This script processes LLM outputs from multiple directories, cleans and formats them,
and evaluates agreement between models and runs. It generates a report with metrics,
plots, and figures to summarize the results.

Steps:
1. Load CSVs from provided directories (each directory corresponds to a run).
2. Use postprocess.py to clean responses and discard invalid ones.
3. Use format.py to format valid responses into structured data.
4. Use evaluate.py to compare outputs pairwise for agreement.
5. Generate a report with metrics, confusion matrices, and other visualizations.

Usage:
    python evaluate-llms.py --dirs <dir1> <dir2> --output_dir <output_dir>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from cdt.evaluate import convert_dataset_to_seqeval, evaluate_model_agreement
from cdt.format import parse_annotated_text
from cdt.postprocess import get_llm_output_errors, standardize_llm_output


def process_and_format_outputs(directory: Path) -> tuple[dict, dict]:
    """Process and format LLM outputs from a directory.

    Args:
        directory (Path): Path to the directory containing CSV files.

    Returns:
        tuple[dict, dict]: Formatted outputs and performance metrics
    """
    formatted_outputs = {}
    performance_stats = {}

    for csv_file in directory.glob("*.csv"):
        model_name = csv_file.stem
        results_df = pd.read_csv(csv_file)
        valid_responses = []
        total_responses = len(results_df)
        rejected_responses = 0

        perf_metrics = {
            "inference_time": [],
            "attempts": [],
            "max_allocated": [],
        }

        for _, row in results_df.iterrows():
            original_text = row["input_text"]
            response = row["response"]

            # Track performance metrics
            for metric in ["inference_time", "attempts"]:
                if metric in row:
                    perf_metrics[metric].append(row[metric])
            perf_metrics["max_allocated"].append(float(row["max_allocated"].split()[0]))

            # Clean and validate response
            cleaned_response = standardize_llm_output(response)
            errors = get_llm_output_errors(cleaned_response, original_text)

            if not errors:
                # Parse and format valid response
                instruments_df, agreements_df, coref_map = parse_annotated_text(
                    cleaned_response, text_id=row["text_id"]
                )
                valid_responses.append(
                    {
                        "text_id": row["text_id"],
                        "instruments": instruments_df,
                        "agreements": agreements_df,
                        "coreferences": coref_map,
                    }
                )
            else:
                rejected_responses += 1

        formatted_outputs[model_name] = valid_responses

        # Compute performance statistics
        performance_stats[model_name] = {
            "total_responses": total_responses,
            "rejected_responses": rejected_responses,
            "acceptance_rate": (total_responses - rejected_responses) / total_responses,
            "avg_inference_time": np.mean(perf_metrics["inference_time"])
            if perf_metrics["inference_time"]
            else None,
            "std_inference_time": np.std(perf_metrics["inference_time"])
            if perf_metrics["inference_time"]
            else None,
            "avg_attempts": np.mean(perf_metrics["attempts"])
            if perf_metrics["attempts"]
            else None,
            "avg_memory": np.mean(perf_metrics["max_allocated"])
            if perf_metrics["max_allocated"]
            else None,
            "max_memory": np.max(perf_metrics["max_allocated"])
            if perf_metrics["max_allocated"]
            else None,
        }

    return formatted_outputs, performance_stats


def compare_models_across_runs(runs: dict) -> pd.DataFrame:
    """Compare models across runs for agreement.

    Args:
        runs (dict): A dictionary with run names as keys and formatted outputs as values.

    Returns:
        pd.DataFrame: A DataFrame summarizing pairwise agreement metrics.
    """
    results = []
    for run_name, models in runs.items():
        model_names = list(models.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                for response1, response2 in zip(
                    models[model1], models[model2]
                ):  # Pairwise comparison
                    agreement_metrics = evaluate_model_agreement(
                        (
                            response1["instruments"],
                            response1["agreements"],
                            response1["coreferences"],
                        ),
                        (
                            response2["instruments"],
                            response2["agreements"],
                            response2["coreferences"],
                        ),
                    )
                    results.append(
                        {
                            "run_name": run_name,
                            "model1": model1,
                            "model2": model2,
                            **agreement_metrics,
                        }
                    )
    return pd.DataFrame(results)


def generate_report(
    results: pd.DataFrame, performance_stats: dict, runs: dict, output_dir: Path
) -> None:
    """Generate a report with metrics and visualizations.

    Args:
        results (pd.DataFrame): DataFrame containing pairwise agreement metrics.
        performance_stats (dict): Dictionary containing performance statistics.
        runs (dict): Dictionary containing formatted outputs.
        output_dir (Path): Path to the directory where the report will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    results.to_csv(output_dir / "agreement_metrics.csv", index=False)

    # Generate confusion matrices for each pair of models
    for _, row in results.iterrows():
        model1, model2 = row["model1"], row["model2"]
        run_name = row["run_name"]
        confusion = row.get("confusion_matrix")
        if confusion is not None:
            cm = confusion_matrix(confusion["true"], confusion["pred"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix: {model1} vs {model2} ({run_name})")
            plt.savefig(output_dir / f"confusion_{model1}_vs_{model2}_{run_name}.png")
            plt.close()

    # Generate token-level evaluation metrics using seqeval
    token_metrics = {}
    for run_name, models in runs.items():
        model_names = list(models.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                true_tags, pred_tags = [], []
                for response1, response2 in zip(models[model1], models[model2]):
                    # Convert both models' outputs to seqeval format
                    seqeval_data1 = convert_dataset_to_seqeval(
                        response1["text_id"],
                        response1["instruments"],
                        response1["agreements"],
                        response1["coreferences"],
                    )
                    seqeval_data2 = convert_dataset_to_seqeval(
                        response2["text_id"],
                        response2["instruments"],
                        response2["agreements"],
                        response2["coreferences"],
                    )
                    true_tags.extend(seqeval_data1["tags"])
                    pred_tags.extend(seqeval_data2["tags"])

                # Compute seqeval metrics
                f1 = f1_score(true_tags, pred_tags)
                report = classification_report(true_tags, pred_tags, output_dict=True)
                token_metrics[f"{run_name}/{model1}_vs_{model2}"] = {
                    "f1": f1,
                    "report": report,
                }

    # Plot token-level F1 scores
    if token_metrics:
        f1_scores = {k: v["f1"] for k, v in token_metrics.items()}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
        plt.title("Token-level F1 Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "token_level_f1_scores.png")
        plt.close()

    # Save detailed token-level evaluation report
    with (output_dir / "token_evaluation_report.txt").open("w") as f:
        f.write("=== Token-level Evaluation Report ===\n\n")
        for model_pair, metrics in token_metrics.items():
            f.write(f"\n{model_pair}:\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write("Detailed Classification Report:\n")
            f.write(str(metrics["report"]))

    # Generate per-model performance report
    with (output_dir / "performance_report.txt").open("w") as f:
        f.write("=== Model Performance Report ===\n\n")

        for run_name, run_stats in performance_stats.items():
            f.write(f"\nRun: {run_name}\n")
            f.write("=" * (len(run_name) + 6) + "\n")

            for model, stats in run_stats.items():
                f.write(f"\nModel: {model}\n")
                f.write("-" * (len(model) + 8) + "\n")
                f.write(f"Total responses: {stats['total_responses']}\n")
                f.write(f"Rejected responses: {stats['rejected_responses']}\n")
                f.write(f"Acceptance rate: {stats['acceptance_rate']:.2%}\n")

                if stats["avg_inference_time"] is not None:
                    f.write(
                        f"Average inference time: {stats['avg_inference_time']:.2f}s\n"
                    )
                    f.write(f"Std inference time: {stats['std_inference_time']:.2f}s\n")

                if stats["avg_attempts"] is not None:
                    f.write(f"Average attempts: {stats['avg_attempts']:.2f}\n")

                if stats["avg_memory"] is not None:
                    f.write(
                        f"Average memory usage: {stats['avg_memory'] / 1e6:.2f}MB\n"
                    )
                    f.write(f"Peak memory usage: {stats['max_memory'] / 1e6:.2f}MB\n")

                f.write("\n")

        # Add existing pairwise comparison results
        f.write("\n=== Model Agreement Evaluation ===\n")
        f.write(results.describe().to_string())

    # Create performance visualization
    plt.figure(figsize=(12, 6))
    all_models = []
    inference_times = []
    memory_usage = []
    acceptance_rates = []

    for run_name, run_stats in performance_stats.items():
        for model, stats in run_stats.items():
            all_models.append(f"{run_name}/{model}")
            inference_times.append(stats["avg_inference_time"])
            memory_usage.append(stats["avg_memory"] / 1e6 if stats["avg_memory"] else 0)
            acceptance_rates.append(stats["acceptance_rate"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot inference times
    sns.barplot(x=inference_times, y=all_models, ax=ax1)
    ax1.set_title("Average Inference Time (s)")

    # Plot memory usage
    sns.barplot(x=memory_usage, y=all_models, ax=ax2)
    ax2.set_title("Average Memory Usage (MB)")

    # Plot acceptance rates
    sns.barplot(x=acceptance_rates, y=all_models, ax=ax3)
    ax3.set_title("Response Acceptance Rate")

    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png")
    plt.close()


parser = argparse.ArgumentParser(description="Evaluate LLM outputs across runs.")
parser.add_argument(
    "--dirs",
    nargs="+",
    required=True,
    help="List of directories containing LLM output CSVs.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory to save the evaluation report and results.",
)
args = parser.parse_args()


# Process each directory
runs = {}
performance_stats = {}
for dir_path in args.dirs:
    run_name = Path(dir_path).name
    formatted_outputs, perf_stats = process_and_format_outputs(Path(dir_path))
    runs[run_name] = formatted_outputs
    performance_stats[run_name] = perf_stats

# Compare models across runs
results = compare_models_across_runs(runs)

# Generate report
output_dir = Path(args.output_dir)
generate_report(results, performance_stats, runs, output_dir)
print(f"Evaluation report generated in {output_dir}")
