import logging
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from iplom_llm_parser.cache import CacheEntry, TemplateCache
from iplom_llm_parser.config import Config, IPLoMConfig, PipelineConfig
from iplom_llm_parser.iplom import IPLoM, Partition
from iplom_llm_parser.llm_client import LLMClient, postprocess
from iplom_llm_parser.regex_comp import infer_slot_regexes

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_time: float
    matched: int
    noise: int


class TemplatePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        iplom_config: IPLoMConfig,
        client: LLMClient,
        full_df: pl.DataFrame,
    ):
        self.config = config
        self.iplom_config = iplom_config
        self.client = client
        self.full_df = full_df
        self.cache = TemplateCache()
        self.messages: pl.Series = full_df[config.content_col]
        self._stats: RunStats | None = None

        if "row_nr" not in full_df.columns:
            self.full_df = self.full_df.with_row_index("row_nr")

    @property
    def stats(self) -> RunStats:
        if self._stats is None:
            raise RuntimeError(
                "pipeline.stats is only available after run() has completed"
            )
        return self._stats

    def _batch_sample(self, row_nrs: list[int], n: int) -> list[str]:
        sampled = random.sample(row_nrs, min(n, len(row_nrs)))
        return [self.messages[rn] for rn in sampled]

    def _match_cache_rows(self, row_nrs: list[int]) -> tuple[list[tuple], list[int]]:
        matched, unmatched = [], []
        for rn in row_nrs:
            hit = self.cache.match_message(self.messages[rn])
            if hit is not None:
                entry, m = hit
                self.cache.increment(entry, 1)
                matched.append((rn, entry.template, list(m.groups()), entry.slot_types))
            else:
                unmatched.append(rn)
        return matched, unmatched

    def _validate(
        self, entry: CacheEntry, row_nrs: list[int]
    ) -> tuple[list[tuple[int, list[str]]], list[int]]:
        matched, unmatched = [], []
        for rn in row_nrs:
            m = entry.regex.fullmatch(self.messages[rn])
            if m:
                matched.append((rn, list(m.groups())))
            else:
                unmatched.append(rn)
        return matched, unmatched

    def _reconcile(
        self,
        row_nrs: list[int],
        template: str,
        slot_regexes: list[str] | None,
        slot_types: list[str],
    ) -> tuple[list[tuple], list[int]]:
        matched, unmatched = self._match_cache_rows(row_nrs)
        if not unmatched:
            return matched, []

        entry = self.cache.insert(template, slot_regexes, slot_types)
        new_matched, still_unmatched = self._validate(entry, unmatched)

        if new_matched:
            self.cache.increment(entry, len(new_matched))
            matched.extend(
                [
                    (rn, entry.template, params, entry.slot_types)
                    for rn, params in new_matched
                ]
            )
        else:
            self.cache.remove(entry)

        return matched, still_unmatched

    def _process_chunk(self, leaves: list[Partition]) -> tuple[list[tuple], list[int]]:
        chunk_results: list[tuple] = []
        chunk_unmatched: list[int] = []
        needs_llm: list[tuple[Partition, list[int]]] = []

        for p in leaves:
            matched, unmatched = self._match_cache_rows(p.row_nrs)
            chunk_results.extend(matched)
            if unmatched:
                needs_llm.append((p, unmatched))

        if not needs_llm:
            return chunk_results, chunk_unmatched

        logger.info(f"Making {len(needs_llm)} LLM call(s)")
        llm_samples = [
            self._batch_sample(unmatched_nrs, self.config.llm_sample_n)
            for _, unmatched_nrs in needs_llm
        ]
        llm_raw = self.client.query_batch(llm_samples)

        for (_, unmatched_nrs), raw_template in zip(needs_llm, llm_raw):
            template, slot_types = postprocess(
                raw_template,
                template_correction=self.config.template_correction,
            )

            if self.config.infer_slot_regexes:
                infer_samples = self._batch_sample(
                    unmatched_nrs, max(self.config.llm_sample_n, 50)
                )
                slot_regexes = infer_slot_regexes(template, infer_samples, slot_types)
            else:
                slot_regexes = None

            matched, still_unmatched = self._reconcile(
                unmatched_nrs, template, slot_regexes, slot_types
            )
            chunk_results.extend(matched)
            chunk_unmatched.extend(still_unmatched)

        return chunk_results, chunk_unmatched

    def _partition(self, df: pl.DataFrame) -> list[Partition]:
        start_t = time.perf_counter()
        iplom = IPLoM(
            df,
            content_col=self.config.content_col,
            CT=self.iplom_config.CT,
            lower_bound=self.iplom_config.lower_bound,
        )
        iplom.parse()

        leaves = iplom.collect_leaf_partitions(iplom.partitions)
        outliers = iplom.collect_leaf_partitions(iplom.outlier_partitions)
        parts = leaves + outliers

        end_t = time.perf_counter()
        logger.info(f"IPLoM partition time: {(end_t - start_t):.6f} seconds")
        return parts

    def _process_repool(self, unmatched: list[int]) -> tuple[list[tuple], list[int]]:
        if not unmatched:
            return [], []

        repool_df = self.full_df.filter(pl.col("row_nr").is_in(unmatched))
        leaves = self._partition(repool_df)
        return self._process_chunk(leaves)

    def _process_singleton(self, row_nrs: list[int]) -> tuple[list[tuple], list[int]]:
        results, still_unmatched = self._match_cache_rows(row_nrs)
        if not still_unmatched:
            return results, []

        logger.info(f"Singleton LLM fallback for {len(still_unmatched)} rows")
        llm_raw = self.client.query_batch(
            [[self.messages[rn]] for rn in still_unmatched]
        )

        remaining: list[int] = []
        for rn, raw_template in zip(still_unmatched, llm_raw):
            template, slot_types = postprocess(
                raw_template,
                template_correction=self.config.template_correction,
            )
            matched, unmatched = self._reconcile([rn], template, None, slot_types)
            if matched:
                results.extend(matched)
            else:
                logger.warning(
                    f"Singleton LLM template failed to self-match | row {rn}: {self.messages[rn][:120]}"
                )
                remaining.extend(unmatched)

        return results, remaining

    def _build_result_df(self, results: list[tuple]) -> pl.DataFrame:
        schema = {
            "row_nr": self.full_df["row_nr"].dtype,
            "EventTemplate": pl.String,
            "ParameterList": pl.List(pl.String),
            "SlotTypes": pl.List(pl.String),
        }

        if results:
            result_df = pl.DataFrame(results, schema=schema, orient="row")
            unique_templates = sorted(
                t
                for t in result_df["EventTemplate"].unique().to_list()
                if t is not None
            )
            event_id_map_df = pl.DataFrame(
                {
                    "EventTemplate": unique_templates,
                    "EventId": [f"E{i + 1}" for i in range(len(unique_templates))],
                },
                schema={"EventTemplate": pl.String, "EventId": pl.String},
            )
            result_df = result_df.join(event_id_map_df, on="EventTemplate", how="left")
        else:
            result_df = pl.DataFrame(schema={**schema, "EventId": pl.String})

        id_col = "LineId" if "LineId" in self.full_df.columns else "row_nr"

        return (
            self.full_df.select("row_nr", id_col, self.config.content_col)
            .join(result_df, on="row_nr", how="left")
            .select(
                pl.col(id_col).alias("LineId"),
                pl.col("EventId"),
                pl.col(self.config.content_col).alias("Content"),
                pl.col("EventTemplate"),
                pl.col("ParameterList"),
                pl.col("SlotTypes"),
            )
            .sort("LineId")
        )

    def _build_result_df_from_files(
        self, chunk_files: list[str], temp_dir: tempfile.TemporaryDirectory
    ) -> pl.DataFrame:
        id_col = "LineId" if "LineId" in self.full_df.columns else "row_nr"

        if chunk_files:
            results_lazy = pl.scan_parquet(chunk_files)

            unique_templates = (
                results_lazy.select("EventTemplate")
                .unique()
                .collect()["EventTemplate"]
                .drop_nulls()
                .sort()
                .to_list()
            )

            event_id_map_df = pl.DataFrame(
                {
                    "EventTemplate": unique_templates,
                    "EventId": [f"E{i + 1}" for i in range(len(unique_templates))],
                },
                schema={"EventTemplate": pl.String, "EventId": pl.String},
            )

            result_df = results_lazy.join(
                event_id_map_df.lazy(), on="EventTemplate", how="left"
            )

            final_df = (
                self.full_df.lazy()
                .select("row_nr", id_col, self.config.content_col)
                .join(result_df, on="row_nr", how="left")
                .select(
                    pl.col(id_col).alias("LineId"),
                    pl.col("EventId"),
                    pl.col(self.config.content_col).alias("Content"),
                    pl.col("EventTemplate"),
                    pl.col("ParameterList"),
                    pl.col("SlotTypes"),
                )
                .sort("LineId")
                .collect(engine="streaming")
            )
        else:
            final_df = self.full_df.select(
                pl.col(id_col).alias("LineId"),
                pl.lit(None, dtype=pl.String).alias("EventId"),
                pl.col(self.config.content_col).alias("Content"),
                pl.lit(None, dtype=pl.String).alias("EventTemplate"),
                pl.lit(None, dtype=pl.List(pl.String)).alias("ParameterList"),
                pl.lit(None, dtype=pl.List(pl.String)).alias("SlotTypes"),
            )

        temp_dir.cleanup()
        return final_df

    def run(self) -> pl.DataFrame:
        start_t = time.perf_counter()
        calls_before = self.client.llm_calls
        in_before = self.client.input_tokens
        out_before = self.client.output_tokens

        self.cache = TemplateCache()
        self.messages = self.full_df[self.config.content_col]

        temp_dir = tempfile.TemporaryDirectory()
        chunk_files = []

        results_count = 0
        noise: list[int] = []

        n_chunks = (
            len(self.full_df) + self.config.chunk_size - 1
        ) // self.config.chunk_size

        schema = {
            "row_nr": self.full_df["row_nr"].dtype,
            "EventTemplate": pl.String,
            "ParameterList": pl.List(pl.String),
            "SlotTypes": pl.List(pl.String),
        }

        for chunk_idx in range(n_chunks):
            start_chunk_t = time.perf_counter()

            chunk = self.full_df.slice(
                chunk_idx * self.config.chunk_size, self.config.chunk_size
            )
            logger.info(f"Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} rows)")

            leaves = self._partition(chunk)
            chunk_results, unmatched = self._process_chunk(leaves)

            for repool_pass in range(self.config.repool_passes):
                if not unmatched:
                    break
                repool_results, unmatched = self._process_repool(unmatched)
                chunk_results.extend(repool_results)

            if unmatched and self.config.singleton_fallback:
                singleton_results, unmatched = self._process_singleton(unmatched)
                chunk_results.extend(singleton_results)

            noise.extend(unmatched)

            if chunk_results:
                results_count += len(chunk_results)
                chunk_df = pl.DataFrame(chunk_results, schema=schema, orient="row")
                chunk_file = (
                    f"{temp_dir.name}/chunk_{chunk_idx}_{time.time_ns()}.parquet"
                )
                chunk_df.write_parquet(chunk_file)
                chunk_files.append(chunk_file)
                chunk_results.clear()

            end_chunk_t = time.perf_counter()
            logger.info(
                f"Matched: {results_count}, Noise: {len(noise)} | Chunk finished in {(end_chunk_t - start_chunk_t):.4f}s"
            )

        if noise:
            logger.info(
                f"Final cache-only check over {len(noise)} remaining noise rows"
            )
            recovered, noise = self._match_cache_rows(noise)
            if recovered:
                results_count += len(recovered)
                rec_df = pl.DataFrame(recovered, schema=schema, orient="row")
                rec_file = f"{temp_dir.name}/chunk_recovered.parquet"
                rec_df.write_parquet(rec_file)
                chunk_files.append(rec_file)
                recovered.clear()

        end_t = time.perf_counter()

        self._stats = RunStats(
            llm_calls=self.client.llm_calls - calls_before,
            input_tokens=self.client.input_tokens - in_before,
            output_tokens=self.client.output_tokens - out_before,
            total_time=end_t - start_t,
            matched=results_count,
            noise=len(noise),
        )

        return self._build_result_df_from_files(chunk_files, temp_dir)


def write_output_parquet(df_res: pl.DataFrame, output_path: str) -> None:
    df_res.write_parquet(output_path)
    logger.info(f"Output written to {output_path} ({len(df_res)} rows)")


def write_config(config: Config, output_path: str) -> None:
    path = Path(output_path)
    path.write_text(config.model_dump_json(indent=2))
    logger.info(f"Config written to {output_path}")


def write_output(df_res: pl.DataFrame, output_path: str) -> None:
    def _format_list_col(col_name: str) -> pl.Expr:
        joined = pl.col(col_name).list.join("', '")
        formatted = pl.concat_str([pl.lit("['"), joined, pl.lit("']")])
        return (
            pl.when(pl.col(col_name).is_not_null() & (pl.col(col_name).list.len() > 0))
            .then(formatted)
            .otherwise(None)
            .alias(col_name)
        )

    df_res.lazy().with_columns(
        _format_list_col("ParameterList"),
        _format_list_col("SlotTypes"),
    ).sink_csv(output_path)

    logger.info(f"Output written to {output_path} ({len(df_res)} rows)")
