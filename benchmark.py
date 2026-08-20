import logging
import math
import random
from pathlib import Path

import polars as pl

from eval import evaluate
from iplom_llm_parser.config import Config, load_config
from iplom_llm_parser.llm_client import LLMClient
from iplom_llm_parser.pipeline import TemplatePipeline, write_config, write_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def scaled_chunk_size(
    n_rows: int,
    min_rows: int = 20_000,
    max_rows: int = 16_000_000,
    min_chunk: int = 3000,
    max_chunk: int = 40_000,
) -> int:
    n_rows = max(min_rows, min(max_rows, n_rows))
    frac = (math.log10(n_rows) - math.log10(min_rows)) / (
        math.log10(max_rows) - math.log10(min_rows)
    )
    chunk = min_chunk + frac * (max_chunk - min_chunk)
    return int(round(chunk / 500) * 500)


def run_all(config: Config):
    datasets = [
        "HDFS",
        "Hadoop",
        "Spark",
        "Zookeeper",
        "BGL",
        "HPC",
        # "Thunderbird",
        "Linux",
        "HealthApp",
        "Apache",
        "Proxifier",
        "OpenSSH",
        "OpenStack",
        "Mac",
    ]

    summary = []
    summary_df = None

    dataset_type = "full"
    parent_dir = "full-loghub-2.0"
    model_tag = config.llm.model.replace("/", "-").replace(":", "-")
    corr_tag = "cor" if config.pipeline.template_correction else "nocor"
    summary_path = f"results/summary_{corr_tag}_{model_tag}_{dataset_type}.csv"

    Path("results").mkdir(exist_ok=True)
    write_config(config, f"results/config_{corr_tag}_{model_tag}_{dataset_type}.json")

    with LLMClient(config.llm) as client:
        for dataset in datasets:
            random.seed(42)
            print(f"\n{'=' * 40}\n{dataset}\n{'=' * 40}")

            row: dict = {"dataset": dataset}
            out_path = (
                f"results/{dataset}_{dataset_type}_{corr_tag}_{model_tag}_output.csv"
            )

            try:
                df = pl.read_csv(
                    f"{parent_dir}/{dataset}/{dataset}_{dataset_type}.log_structured.csv"
                )
                df = df.with_row_index("row_nr")

                dataset_chunk_size = scaled_chunk_size(df.height)
                pipeline_config = config.pipeline.model_copy(
                    update={"chunk_size": dataset_chunk_size}
                )

                pipeline = TemplatePipeline(pipeline_config, config.iplom, client, df)
                result_df = pipeline.run()
                write_output(result_df, out_path)

                print(
                    f"Done in {pipeline.stats.total_time:.2f} - {pipeline.stats.matched} matched, {pipeline.stats.noise} noise"
                )

                row.update(
                    {
                        "chunk_size": dataset_chunk_size,
                        "out_path": out_path,
                        "matched": pipeline.stats.matched,
                        "noise": pipeline.stats.noise,
                        "llm_calls": pipeline.stats.llm_calls,
                        "input_tokens": pipeline.stats.input_tokens,
                        "output_tokens": pipeline.stats.output_tokens,
                        "total_time": f"{pipeline.stats.total_time:.4f}",
                    }
                )
            except Exception as e:
                print(f"Pipeline failed on {dataset}: {e}")
                row["error"] = f"pipeline: {e}"
                summary.append(row)
                summary_df = pl.DataFrame(summary)
                summary_df.write_csv(summary_path)
                continue  # can't eval without pipeline output

            try:
                metrics = evaluate(
                    f"full_loghub-2.0/{dataset}/{dataset}_full.log_structured.csv",
                    out_path,
                    template_correction=False,
                    verbose=False,
                )
                row.update(
                    {
                        "GA": metrics["GA"],
                        "FGA": metrics["FGA"],
                        "PA": metrics["PA"],
                        "n_gt_templates": metrics["n_gt_templates"],
                        "n_pred_templates": metrics["n_pred_templates"],
                        "exact_template_matches": metrics["exact_template_matches"],
                        "template_precision": metrics["template_precision"],
                        "template_recall": metrics["template_recall"],
                        "template_f1": metrics["template_f1"],
                    }
                )
            except Exception as e:
                print(f"Eval failed on {dataset}: {e}")
                row["error"] = f"eval: {e}"

            summary.append(row)

            summary_df = pl.DataFrame(summary)
            summary_df.write_csv(summary_path)

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    config = load_config("config.toml")
    run_all(config)
