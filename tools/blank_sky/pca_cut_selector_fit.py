#!/usr/bin/env python3
"""Fit a cheap bounded PCA selector against oracle outputs.

This tool takes one or more `pca_cut_oracle_detailed.csv` files, uses the
oracle-selected `k` as the reference answer, and searches a small weight grid
for a cheaper runtime-friendly score built only from metrics that are practical
to compute inline after each candidate PCA subtraction.

The current selector score uses:

- `med_abs_corr` as the primary coherence term
- `log2(cm_low_mid_ratio)` for low-frequency common-mode leakage
- `tail4_binom_z` for non-Gaussian tails
- `top_mode_frac` for first-mode dominance
- a regularization term that penalizes moving away from the baseline cut
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHT_GRID = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
REG_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)


def _safe_log2_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    good = np.isfinite(x) & (x > 0.0)
    out[good] = np.log2(x[good])
    return out


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    if vmax > vmin:
        out[finite] = (values[finite] - vmin) / (vmax - vmin)
    else:
        out[finite] = 0.0
    return out


def _prepare_rows(df: pd.DataFrame, baseline_k: int, k_step: int) -> pd.DataFrame:
    out = df.copy()
    out["inline_corr_term"] = pd.to_numeric(out["med_abs_corr"], errors="coerce")
    out["inline_low_term"] = np.maximum(_safe_log2_pos(out["cm_low_mid_ratio"].to_numpy(dtype=float)), 0.0)
    out["inline_tail_term"] = np.maximum(pd.to_numeric(out["tail4_binom_z"], errors="coerce").to_numpy(dtype=float), 0.0)
    out["inline_topmode_term"] = pd.to_numeric(out["top_mode_frac"], errors="coerce")
    out["inline_reg_term"] = np.abs(pd.to_numeric(out["k"], errors="coerce").to_numpy(dtype=float) - float(baseline_k)) / max(float(k_step), 1.0)

    normed_groups: list[pd.DataFrame] = []
    for _, g in out.groupby(["obsnum", "scan", "network"], sort=False):
        gg = g.copy()
        for col in [
            "inline_corr_term",
            "inline_low_term",
            "inline_tail_term",
            "inline_topmode_term",
        ]:
            gg[f"{col}_norm"] = _normalize(gg[col].to_numpy(dtype=float))
        normed_groups.append(gg)
    return pd.concat(normed_groups, ignore_index=True)


def _prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = df.sort_values(["obsnum", "scan", "network", "k"]).reset_index(drop=True).copy()
    counts = df.groupby(["obsnum", "scan", "network"])["k"].transform("size")
    full = counts == counts.max()
    df = df.loc[full].reset_index(drop=True).copy()
    n_candidates = int(df.groupby(["obsnum", "scan", "network"])["k"].size().iloc[0])
    n_groups = int(len(df) // n_candidates)

    mats = {
        "corr": df["inline_corr_term_norm"].fillna(1.0).to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "low": df["inline_low_term_norm"].fillna(1.0).to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "tail": df["inline_tail_term_norm"].fillna(1.0).to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "topmode": df["inline_topmode_term_norm"].fillna(1.0).to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "reg": df["inline_reg_term"].fillna(0.0).to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "oracle": df["oracle_score"].to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "k": df["k"].to_numpy(dtype=float).reshape(n_groups, n_candidates),
        "group_obsnum": df["obsnum"].to_numpy().reshape(n_groups, n_candidates)[:, 0],
        "group_scan": df["scan"].to_numpy(dtype=int).reshape(n_groups, n_candidates)[:, 0],
        "group_network": df["network"].to_numpy(dtype=int).reshape(n_groups, n_candidates)[:, 0],
    }
    mats["oracle_idx"] = np.argmin(mats["oracle"], axis=1)
    mats["oracle_k"] = mats["k"][np.arange(n_groups), mats["oracle_idx"]]
    mats["oracle_best"] = mats["oracle"][np.arange(n_groups), mats["oracle_idx"]]
    return df, mats


def _evaluate_rule(mats: dict[str, np.ndarray], weights: dict[str, float]) -> tuple[dict[str, float], np.ndarray]:
    score = (
        1.0 * mats["corr"]
        + weights["low"] * mats["low"]
        + weights["tail"] * mats["tail"]
        + weights["topmode"] * mats["topmode"]
        + weights["reg"] * mats["reg"]
    )
    inline_idx = np.argmin(score, axis=1)
    inline_k = mats["k"][np.arange(score.shape[0]), inline_idx]
    inline_oracle = mats["oracle"][np.arange(score.shape[0]), inline_idx]
    k_abs_err = np.abs(inline_k - mats["oracle_k"])
    oracle_regret = inline_oracle - mats["oracle_best"]
    metrics = {
        "w_corr": 1.0,
        "w_low": float(weights["low"]),
        "w_tail": float(weights["tail"]),
        "w_topmode": float(weights["topmode"]),
        "w_reg": float(weights["reg"]),
        "n_rows": int(score.shape[0]),
        "exact_match_frac": float(np.mean(k_abs_err == 0)),
        "within_one_step_frac": float(np.mean(k_abs_err <= 2)),
        "mean_k_abs_err": float(np.mean(k_abs_err)),
        "median_k_abs_err": float(np.median(k_abs_err)),
        "mean_oracle_regret": float(np.mean(oracle_regret)),
        "median_oracle_regret": float(np.median(oracle_regret)),
    }
    summary = np.column_stack([
        mats["group_obsnum"],
        mats["group_scan"],
        mats["group_network"],
        mats["oracle_k"],
        inline_k,
        k_abs_err,
        mats["oracle_best"],
        inline_oracle,
        oracle_regret,
    ])
    return metrics, summary


def _summary_to_frame(summary: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obsnum": summary[:, 0],
            "scan": summary[:, 1].astype(int),
            "network": summary[:, 2].astype(int),
            "oracle_k": summary[:, 3].astype(int),
            "inline_k": summary[:, 4].astype(int),
            "k_abs_err": summary[:, 5].astype(int),
            "oracle_best_score": summary[:, 6].astype(float),
            "inline_oracle_score": summary[:, 7].astype(float),
            "oracle_regret": summary[:, 8].astype(float),
        }
    )


def _write_report(
    outpath: Path,
    csvs: list[Path],
    baseline_k: int,
    top_rules: pd.DataFrame,
    best_rows: pd.DataFrame,
) -> None:
    best = top_rules.iloc[0]
    lines = [
        "# PCA Selector Fit",
        "",
        "This note fits a cheap inline score to the oracle-selected PCA cuts.",
        "",
        "Inputs:",
    ]
    lines.extend([f"- `{p}`" for p in csvs])
    lines.extend(
        [
            f"- Baseline cut: `{baseline_k}`",
            "",
            "Cheap score terms:",
            "- `med_abs_corr`",
            "- `log2(cm_low_mid_ratio)` clipped at zero",
            "- `tail4_binom_z` clipped at zero",
            "- `top_mode_frac`",
            "- regularization away from baseline `k`",
            "",
            "## Best Rule",
            "",
            "- weights: "
            f"`corr={best['w_corr']:.2f}, low={best['w_low']:.2f}, tail={best['w_tail']:.2f}, "
            f"topmode={best['w_topmode']:.2f}, reg={best['w_reg']:.2f}`",
            f"- exact match fraction: `{best['exact_match_frac']:.3f}`",
            f"- within-one-step fraction: `{best['within_one_step_frac']:.3f}`",
            f"- mean |k error|: `{best['mean_k_abs_err']:.3f}`",
            f"- mean oracle regret: `{best['mean_oracle_regret']:.3f}`",
            "",
            "## Top Rules",
            "",
            "| w_low | w_tail | w_topmode | w_reg | exact | within step | mean |k err| | mean regret |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_rules.iterrows():
        lines.append(
            "| {w_low:.2f} | {w_tail:.2f} | {w_topmode:.2f} | {w_reg:.2f} | "
            "{exact_match_frac:.3f} | {within_one_step_frac:.3f} | {mean_k_abs_err:.3f} | {mean_oracle_regret:.3f} |".format(
                **row.to_dict()
            )
        )

    moved = best_rows[best_rows["inline_k"] != best_rows["oracle_k"]].copy()
    lines.extend(["", "## Largest Rule/Oracle Disagreements", ""])
    if moved.empty:
        lines.append("- none")
    else:
        moved = moved.sort_values(["oracle_regret", "k_abs_err"], ascending=[False, False]).head(12)
        for _, row in moved.iterrows():
            lines.append(
                "- obs={obs} scan={scan} nw={nw}: oracle `k={ok}` vs inline `k={ik}` "
                "(|Δk|={ke}), oracle regret `{reg:.3f}`".format(
                    obs=row["obsnum"],
                    scan=int(row["scan"]),
                    nw=int(row["network"]),
                    ok=int(row["oracle_k"]),
                    ik=int(row["inline_k"]),
                    ke=int(row["k_abs_err"]),
                    reg=float(row["oracle_regret"]),
                )
            )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-csv", action="append", required=True, help="Path to pca_cut_oracle_detailed.csv; repeatable")
    ap.add_argument("--baseline-k", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    csvs = [Path(os.path.expanduser(p)).resolve() for p in args.oracle_csv]
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    k_values = sorted(int(k) for k in pd.unique(df["k"]))
    k_step = min(abs(b - a) for a, b in zip(k_values[:-1], k_values[1:])) if len(k_values) > 1 else 1
    df = _prepare_rows(df, baseline_k=int(args.baseline_k), k_step=int(k_step))
    df, mats = _prepare_matrix(df)

    results: list[dict[str, float]] = []
    best_eval: pd.DataFrame | None = None
    best_key: tuple[float, float, float, float] | None = None

    for w_low, w_tail, w_top, w_reg in itertools.product(WEIGHT_GRID, WEIGHT_GRID, WEIGHT_GRID, REG_GRID):
        metrics, summary = _evaluate_rule(
            mats,
            weights={"low": w_low, "tail": w_tail, "topmode": w_top, "reg": w_reg},
        )
        results.append(metrics)
        key = (
            -metrics["exact_match_frac"],
            -metrics["within_one_step_frac"],
            metrics["mean_oracle_regret"],
            metrics["mean_k_abs_err"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_eval = _summary_to_frame(summary)

    results_df = pd.DataFrame(results).sort_values(
        ["exact_match_frac", "within_one_step_frac", "mean_oracle_regret", "mean_k_abs_err"],
        ascending=[False, False, True, True],
    )
    top_rules = results_df.head(int(args.top_n)).copy()

    if best_eval is None:
        raise RuntimeError("no selector-fit evaluation results")

    outdir = Path(os.path.expanduser(args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rules_csv = outdir / "pca_selector_fit_rules.csv"
    rows_csv = outdir / "pca_selector_fit_best_rule_rows.csv"
    report_md = outdir / "PCA_SELECTOR_FIT.md"
    top_rules.to_csv(rules_csv, index=False)
    best_eval.to_csv(rows_csv, index=False)
    _write_report(report_md, csvs, int(args.baseline_k), top_rules, best_eval)

    print(f"Wrote {rules_csv}")
    print(f"Wrote {rows_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
