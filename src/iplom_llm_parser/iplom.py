# Modified from LogLead
# https://github.com/EvoTestOps/LogLead/tree/main/loglead/parsers/pl_iplom

import logging
from collections import Counter

import polars as pl

logger = logging.getLogger(__name__)


class Partition:
    def __init__(self, df, token_len, split_trace, split_col=None, split_value=None):
        self.df = df
        self.subpartitions = []
        self.token_len = token_len
        self.split_trace = split_trace

        self.split_col = split_col
        self.split_value = split_value

    @property
    def fingerprint(self):
        return (self.token_len, self.split_col, self.split_value)

    @property
    def row_nrs(self):
        return self.df["row_nr"].to_list()


class IPLoM:
    def __init__(
        self,
        df,
        content_col="Content",
        PST=0.0,
        FST=0.0,
        CT=0.15,
        single_outlier_event=False,
        lower_bound=0.25,
    ):
        if "e_words" not in df.columns:
            # df = df.with_columns(pl.col(content_col).str.split(by=" ").alias("e_words"))
            df = df.with_columns(
                pl.col(content_col).str.extract_all(r"[^\s=:,]+").alias("e_words")
            )
            df = df.with_columns(e_words_len=pl.col("e_words").list.len())

        self.df = df
        self.content_col = content_col
        self.PST = PST
        self.FST = FST
        self.CT = CT
        self.single_outlier_event = single_outlier_event
        self.lower_bound = lower_bound
        self.partitions = []
        self.outlier_partitions = []

    def parse(self, output_path=None):
        self.s1_clust_by_message_length()

        for i in range(len(self.partitions)):
            self.s2_clust_by_token_pos(self.partitions[i])
            self.s3_clust_by_bijection(self.partitions[i])

        self.merge_partitions_to_dataframe()

        if output_path:
            self.write_partitions_to_file(output_path)

        return self.acc_df

    def s1_clust_by_message_length(self):
        df_aggre_s1 = self.df.select(pl.col("e_words_len")).unique()
        df_temp = (
            self.df.select("e_words_len", "e_words", "row_nr")
            .group_by("e_words_len")
            .agg(pl.col("e_words").alias("events"), pl.col("row_nr").alias("row_nr"))
        )
        df_aggre_s1 = df_aggre_s1.join(df_temp, on="e_words_len")
        df_temp = self.df.group_by("e_words_len").agg(pl.len().alias("part_len"))
        df_aggre_s1 = df_aggre_s1.join(df_temp, on="e_words_len")

        for i in range(len(df_aggre_s1)):
            df_part = df_aggre_s1[i]
            len_words = df_part["e_words_len"].item(0)
            df_part = (
                df_aggre_s1[i]
                .with_columns(pl.col("events", "row_nr"))
                .explode("events", "row_nr")
            )
            df_part = df_part.with_columns(
                pl.col("events").list.to_struct(upper_bound=len_words)
            ).unnest("events")
            df_part = df_part.drop("e_words_len", "part_len")
            self.add_partition(
                Partition(df=df_part, token_len=len_words, split_trace="S1 ")
            )

        return df_aggre_s1

    def s2_clust_by_token_pos(self, partition):
        cols = [col for col in partition.df.columns if col != "row_nr"]
        if not cols:
            return

        unique_counts = [len(partition.df[col].unique()) for col in cols]
        min_count = min(unique_counts)
        min_col = cols[unique_counts.index(min_count)]

        if min_count > 1:
            row_dict = partition.df.partition_by(min_col, as_dict=True)
            for dataframe in row_dict.values():
                split_value = dataframe[min_col][0]
                child = Partition(
                    dataframe,
                    token_len=partition.token_len,
                    split_trace=partition.split_trace + "S2 ",
                    split_col=min_col,
                    split_value=split_value,
                )
                self.add_partition(child, parent_partition=partition)
            partition.df = None
        else:
            if partition.split_col is None:
                partition.split_col = min_col
                partition.split_value = partition.df[min_col][0]

    def _get_p1_p2(self, unique_counts):
        if len(unique_counts) > 2:
            freqs = Counter(unique_counts)
            common_items = [
                (num, count) for num, count in freqs.most_common() if num != 1
            ]

            if len(common_items) >= 2 and common_items[0][1] == common_items[1][1]:
                two_smallest = sorted(common_items, key=lambda x: x[0])[:2]
                mf1, mf2 = two_smallest[0][0], two_smallest[1][0]
                positions1 = [i for i, n in enumerate(unique_counts) if n == mf1]
                positions2 = [i for i, n in enumerate(unique_counts) if n == mf2]
                if len(positions1) >= 2:
                    p1, p2 = positions1[0], positions1[1]
                else:
                    p1 = positions1[0] if positions1 else None
                    p2 = positions2[0] if positions2 else None
            elif common_items:
                most_freq_num = common_items[0][0]
                positions = [
                    i for i, n in enumerate(unique_counts) if n == most_freq_num
                ]
                p1, p2 = positions[0], positions[1]
            else:
                return -1, 0
        elif len(unique_counts) == 2:
            p1, p2 = 0, 1
        else:
            p1, p2 = -1, 0

        return p1, p2

    def _get_rank_position(self, length, card, one_to_m):
        dist = card / length
        if dist < self.lower_bound:
            return 2 if one_to_m else 1
        else:
            return 1 if one_to_m else 2

    def s3_clust_by_bijection(self, partition):
        if partition.subpartitions:
            for subpartition in partition.subpartitions:
                self.s3_clust_by_bijection(subpartition)
            return

        if partition.df is None:
            return

        part_df = partition.df

        cols = [c for c in part_df.columns if c != "row_nr"]
        unique_counts = (
            part_df.select(pl.col(cols).n_unique()).transpose().to_series().to_list()
        )

        number_of_ones = unique_counts.count(1)
        cluster_goodness = number_of_ones / len(unique_counts) if unique_counts else 1.0

        if cluster_goodness > self.CT:
            return

        p1, p2 = self._get_p1_p2(unique_counts)
        if p1 == -1:
            return

        col_p1 = cols[p1]
        col_p2 = cols[p2]

        p1_part_dict = {}
        p2_part_dict = {}

        unique_pairs = (
            part_df.select([col_p1, col_p2])
            .unique()
            .select(pl.concat_list(pl.col([col_p1, col_p2])).alias("unique_pairs"))
        )

        for row in unique_pairs.to_dicts():
            pair = row["unique_pairs"]
            value_p1, value_p2 = pair

            count_p1 = (
                part_df.filter(pl.col(col_p1) == value_p1)
                .select(pl.col(col_p2))
                .n_unique()
            )
            count_p2 = (
                part_df.filter(pl.col(col_p2) == value_p2)
                .select(pl.col(col_p1))
                .n_unique()
            )

            if count_p1 > 1 and count_p2 > 1:
                split_pos = 0
            elif count_p1 > 1:
                s_temp = part_df.filter(
                    (pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)
                ).select(col_p1)
                split_pos = self._get_rank_position(
                    s_temp.shape[0], s_temp.n_unique(), True
                )
            elif count_p2 > 1:
                s_temp = part_df.filter(
                    (pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)
                ).select(col_p2)
                split_pos = self._get_rank_position(
                    s_temp.shape[0], s_temp.n_unique(), False
                )
            else:
                split_pos = 1

            if split_pos == 1:
                new_df = part_df.filter(
                    (pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)
                )
                part_df = part_df.filter(
                    ~((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2))
                )
                if value_p1 in p1_part_dict:
                    p1_part_dict[value_p1] = pl.concat([p1_part_dict[value_p1], new_df])
                else:
                    p1_part_dict[value_p1] = new_df

            elif split_pos == 2:
                new_df = part_df.filter(
                    (pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)
                )
                part_df = part_df.filter(
                    ~((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2))
                )
                if value_p2 in p2_part_dict:
                    p2_part_dict[value_p2] = pl.concat([p2_part_dict[value_p2], new_df])
                else:
                    p2_part_dict[value_p2] = new_df

        for dataframe in p1_part_dict.values():
            self.add_partition(
                Partition(
                    dataframe,
                    token_len=partition.token_len,
                    split_trace=partition.split_trace + "S3 ",
                ),
                parent_partition=partition,
            )

        for dataframe in p2_part_dict.values():
            self.add_partition(
                Partition(
                    dataframe,
                    token_len=partition.token_len,
                    split_trace=partition.split_trace + "S3 ",
                ),
                parent_partition=partition,
            )

        partition.df = None if part_df.shape[0] == 0 else part_df

    def add_partition(self, partition, parent_partition=None):
        if self.FST > 0 and partition.df.shape[0] / self.df.shape[0] < self.FST:
            self.outlier_partitions.append(partition)
        else:
            if parent_partition:
                if (
                    self.PST > 0
                    and partition.df.shape[0] / parent_partition.df.shape[0] < self.PST
                ):
                    self.outlier_partitions.append(partition)
                else:
                    parent_partition.subpartitions.append(partition)
            else:
                self.partitions.append(partition)

    def merge_partitions_to_dataframe(self):
        df_list = []

        def traverse_and_concat(partitions, parent_id):
            for i, partition in enumerate(partitions, start=1):
                current_id = f"{parent_id}e{i}"
                if partition.df is not None:
                    df_parsed = partition.df.select("row_nr").with_columns(
                        [
                            pl.lit(current_id).alias("event_id"),
                            pl.lit(partition.token_len).alias("event_len"),
                        ]
                    )
                    df_list.append(df_parsed)
                if partition.subpartitions:
                    traverse_and_concat(partition.subpartitions, current_id)

        def process_outliers():
            for i, outlier in enumerate(self.outlier_partitions, start=1):
                current_id = (
                    "outlier_e" if self.single_outlier_event else f"outlier_e{i}"
                )
                if outlier.df is not None:
                    df_parsed = outlier.df.select("row_nr").with_columns(
                        [
                            pl.lit(current_id).alias("event_id"),
                            pl.lit(outlier.token_len).alias("event_len"),
                        ]
                    )
                    df_list.append(df_parsed)

        self.acc_df = pl.DataFrame()
        traverse_and_concat(self.partitions, "")
        process_outliers()
        if df_list:
            self.acc_df = pl.concat(df_list)
        return self.acc_df

    def print_cluster_info(self):
        self._print_cluster_info_recursive(self.partitions, depth=0)
        self._print_outlier_clusters()

    def _print_outlier_clusters(self):
        print("Outlier clusters:")
        for outlier in self.outlier_partitions:
            print(
                f"  len={outlier.token_len}  trace={outlier.split_trace}  rows={outlier.df.shape[0]} "
            )

    def _print_cluster_info_recursive(self, partitions, depth):
        prefix = "  " * depth
        for partition in partitions:
            if partition.df is None:
                print(
                    f"{prefix}[depth={depth}  len={partition.token_len}  trace={partition.split_trace}  df=None]"
                )
            else:
                print(
                    f"{prefix}[depth={depth}  len={partition.token_len}  trace={partition.split_trace}  rows={partition.df.shape[0]}"
                )
            if partition.subpartitions:
                print(f"{prefix}  {len(partition.subpartitions)} subpartitions:")
                self._print_cluster_info_recursive(partition.subpartitions, depth + 1)

    def write_partitions_to_file(self, output_path):
        raw_lookup = {}
        if self.content_col in self.df.columns:
            raw_lookup = dict(
                zip(self.df["row_nr"].to_list(), self.df[self.content_col].to_list())
            )

        with open(output_path, "w") as f:
            valid_partitions = self.collect_leaf_partitions(self.partitions)
            f.write(f"{'=' * 60}\n")
            f.write(f"Total partitions: {len(valid_partitions)}\n")
            f.write(f"Outlier partitions: {len(self.outlier_partitions)}\n")
            f.write(f"{'=' * 60}\n")

            for i, partition in enumerate(valid_partitions, start=1):
                f.write(
                    f"\nPartition {i}  |  trace={partition.split_trace}  |  token_len={partition.token_len}  |  rows={partition.df.shape[0]}  |  fingerprint={partition.fingerprint}\n"
                )
                f.write("-" * 60 + "\n")
                for row in partition.df.to_dicts():
                    if raw_lookup:
                        f.write("  " + raw_lookup.get(row["row_nr"], "N/A") + "\n")
                    else:
                        tokens = [str(v) for k, v in row.items() if k != "row_nr"]
                        f.write("  " + "  ".join(tokens) + "\n")

            if self.outlier_partitions:
                f.write(f"\n{'=' * 60}\n")
                f.write("OUTLIERS\n")
                f.write(f"{'=' * 60}\n")
                for i, outlier in enumerate(self.outlier_partitions, start=1):
                    f.write(
                        f"\nOutlier {i}  |  trace={outlier.split_trace}  |  token_len={outlier.token_len}  |  rows={outlier.df.shape[0]}\n"
                    )
                    f.write("-" * 60 + "\n")
                    for row in outlier.df.to_dicts():
                        if raw_lookup:
                            f.write("  " + raw_lookup.get(row["row_nr"], "N/A") + "\n")
                        else:
                            tokens = [str(v) for k, v in row.items() if k != "row_nr"]
                            f.write("  " + "  ".join(tokens) + "\n")

        logger.info(f"Partitions written to: {output_path}")

    def collect_leaf_partitions(self, partitions):
        leaves = []
        for partition in partitions:
            if partition.df is not None:
                leaves.append(partition)
            if partition.subpartitions:
                leaves.extend(self.collect_leaf_partitions(partition.subpartitions))
        return leaves
