import re

import polars as pl

from iplom_llm_parser.template_cor import correct_single_template


def load_groundtruth(path: str, template_correction: bool) -> pl.DataFrame:
    df = pl.read_csv(path)

    if template_correction:
        df = df.with_columns(
            pl.col("EventTemplate").map_elements(
                correct_single_template, return_dtype=pl.Utf8
            )
        )

    return df


def load_parsed(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def grouping_accuracy(gt: pl.DataFrame, parsed: pl.DataFrame) -> tuple[float, float]:
    merged = gt.select("LineId", pl.col("EventTemplate").alias("gt_template")).join(
        parsed.select("LineId", pl.col("EventTemplate").alias("pred_template")),
        on="LineId",
        how="inner",
    )
    total = len(merged)
    correctly_grouped = 0
    accurate_templates = 0

    for (gt_t,), group in merged.group_by("gt_template"):
        pred_templates = group["pred_template"].drop_nulls().unique()

        if group["pred_template"].null_count() > 0 or len(pred_templates) != 1:
            continue

        pred_t = pred_templates[0]
        pred_group_total = merged.filter(pl.col("pred_template") == pred_t)
        gt_total = merged.filter(pl.col("gt_template") == gt_t)
        if (
            len(group) == len(gt_total)
            and pred_group_total["gt_template"].n_unique() == 1
        ):
            correctly_grouped += len(group)
            accurate_templates += 1

    ga = correctly_grouped / total if total > 0 else 0.0

    n_pred_templates = merged["pred_template"].n_unique()
    n_gt_templates = merged["gt_template"].drop_nulls().n_unique()
    pga = accurate_templates / n_pred_templates if n_pred_templates > 0 else 0.0
    rga = accurate_templates / n_gt_templates if n_gt_templates > 0 else 0.0
    fga = 2 * (pga * rga) / (pga + rga) if (pga + rga) > 0 else 0.0

    return ga, fga


def _normalise(template: str) -> str:
    return re.sub(r"\s+", " ", template).strip()


def parsing_accuracy(gt: pl.DataFrame, parsed: pl.DataFrame) -> float:
    merged = gt.select("LineId", pl.col("EventTemplate").alias("gt_template")).join(
        parsed.select("LineId", pl.col("EventTemplate").alias("pred_template")),
        on="LineId",
        how="inner",
    )

    correct = sum(
        1
        for gt_t, pred_t in zip(
            merged["gt_template"].to_list(),
            merged["pred_template"].to_list(),
        )
        if pred_t is not None and _normalise(gt_t) == _normalise(pred_t)
    )

    return correct / len(merged) if len(merged) > 0 else 0.0


def template_metrics(gt: pl.DataFrame, parsed: pl.DataFrame) -> dict:
    merged = gt.select(
        "LineId", "Content", pl.col("EventTemplate").alias("gt_template")
    ).join(
        parsed.select("LineId", pl.col("EventTemplate").alias("pred_template")),
        on="LineId",
        how="inner",
    )

    gt_templates = set(merged["gt_template"].unique().to_list())
    pred_templates = set(
        t for t in merged["pred_template"].unique().to_list() if t is not None
    )

    exact_matches = gt_templates & pred_templates
    precision = len(exact_matches) / len(pred_templates) if pred_templates else 0.0
    recall = len(exact_matches) / len(gt_templates) if gt_templates else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "n_gt_templates": len(gt_templates),
        "n_pred_templates": len(pred_templates),
        "exact_template_matches": len(exact_matches),
        "template_precision": precision,
        "template_recall": recall,
        "template_f1": f1,
    }


def per_template_pa(gt: pl.DataFrame, parsed: pl.DataFrame) -> pl.DataFrame:
    merged = gt.select("LineId", pl.col("EventTemplate").alias("gt_template")).join(
        parsed.select("LineId", pl.col("EventTemplate").alias("pred_template")),
        on="LineId",
        how="inner",
    )

    rows = []
    for (gt_t,), group in merged.group_by("gt_template"):
        total = len(group)
        pred_templates = group["pred_template"].drop_nulls().unique().to_list()
        correct = sum(
            1
            for pred_t in group["pred_template"].to_list()
            if pred_t is not None and _normalise(gt_t) == _normalise(pred_t)
        )
        rows.append(
            {
                "gt_template": gt_t,
                "pred_templates": ", ".join(pred_templates),
                "total": total,
                "correct": correct,
                "pa": correct / total,
            }
        )

    return pl.DataFrame(rows).sort("pa")


def evaluate(
    gt_path: str, parsed_path: str, template_correction: bool, verbose: bool = True
) -> dict:
    gt = load_groundtruth(gt_path, template_correction)
    parsed = load_parsed(parsed_path)

    ga, fga = grouping_accuracy(gt, parsed)
    pa = parsing_accuracy(gt, parsed)
    tmpl = template_metrics(gt, parsed)

    results = {"GA": ga, "FGA": fga, "PA": pa, **tmpl}

    if verbose:
        print(f"Grouping Accuracy (GA):  {ga:.4f}")
        print(f"Grouping F1 (FGA):       {fga:.4f}")
        print(f"Parsing Accuracy (PA):   {pa:.4f}")
        print(f"GT templates:            {tmpl['n_gt_templates']}")
        print(f"Pred templates:          {tmpl['n_pred_templates']}")
        print(f"Exact template matches:  {tmpl['exact_template_matches']}")
        print(f"Template precision:      {tmpl['template_precision']:.4f}")
        print(f"Template recall:         {tmpl['template_recall']:.4f}")
        print(f"Template F1:             {tmpl['template_f1']:.4f}")

        breakdown = per_template_pa(gt, parsed)
        worst = breakdown.filter(pl.col("pa") < 1.0)
        if len(worst) > 0:
            print(f"\nTemplates with PA < 1.0 ({len(worst)}):")
            for row in worst.iter_rows(named=True):
                print(f"  [{row['correct']}/{row['total']}] {row['gt_template']}")
                print(f"        -> {row.get('pred_templates') or 'NaN'}")

    return results
