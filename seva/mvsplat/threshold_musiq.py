#!/usr/bin/env python3
"""
MUSIQ analysis + curve-based x→k rules (with 0.95 threshold) + average steps.

Input CSV must contain:
  - 'scene/image', 'mvsplat', 'context_000', 'context_100'
  - numeric k columns: "0","5","10","15","20","25","30","35","40","45" (subset ok, must include "0")

Outputs (in --outdir):
  - musiq_ratios_and_thresholds.csv   # full table with ratios, flags, per-row k_decision & run_steps
  - musiq_compact_ratios.csv          # x and all y(k) ratios per row
  - musiq_binned_means.csv            # per-bin means used for curve plot & rule extraction
  - musiq_ratio_scatter.png           # scatter: x vs y(k) for all k>0
  - musiq_curves_plot.png             # line plot: binned x vs mean y(k) per k
  - decision_summary.txt              # curve-based x→k rules + averages
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Utilities ----------------

def extract_frame_pos(stem: str) -> int | None:
    """Pull last 3 digits from filename stem; keep only 000..003."""
    m = re.search(r"(\d{3})(?!.*\d)", stem)
    if not m:
        return None
    v = int(m.group(1))
    return v if v in (0, 1, 2, 3) else None


def compute_benchmark(df: pd.DataFrame,
                      context0_col: str = "context_000",
                      context1_col: str = "context_100") -> pd.Series:
    """
    Linear interpolation between contexts at t ∈ {1/5,2/5,3/5,4/5}
    for frame_pos {000,001,002,003}.
    """
    t_map = {0: 1/5, 1: 2/5, 2: 3/5, 3: 4/5}
    delta = df[context1_col] - df[context0_col]
    return df[context0_col] + df["frame_pos"].map(t_map) * delta


def decide_k_for_row(row, k_cols, thr=0.95,
                     x_col="x_ratio_mvsplat_over_benchmark",
                     k0_col="0",
                     serve_mvsplat_k=50):
    """
    Per-row decision (not used for the curve-based ranges, but useful to export):
      1) If x_ratio > 1: serve MVSplat (k=serve_mvsplat_k).
      2) Else choose largest k with score(k) >= thr * score(k=0).
      3) Else k=0.
    """
    if row[x_col] > 1.0:
        return serve_mvsplat_k
    k0_score = row[k0_col]
    eligible = []
    for k in k_cols:
        if k == k0_col:
            continue
        if row[k] >= thr * k0_score:
            eligible.append(int(k))
    return max(eligible) if eligible else int(k0_col)


def merge_bin_rules_to_ranges(bin_edges, best_k_per_bin, upper_cap=1.0):
    """
    Merge contiguous bins with the same best k into x ranges.
    Only include ranges with upper bound <= upper_cap (default 1.0).
    Returns a list of (x_start, x_end, k).
    """
    ranges = []
    current_k = None
    start_idx = None

    for i, k in enumerate(best_k_per_bin):
        if k != current_k:
            if current_k is not None:
                # close previous range
                x_start = bin_edges[start_idx]
                x_end = bin_edges[i]  # end edge of previous bin
                # clip to upper_cap
                x_end_clipped = min(x_end, upper_cap)
                if x_start < x_end_clipped:
                    ranges.append((x_start, x_end_clipped, current_k))
            current_k = k
            start_idx = i

    # close last
    if current_k is not None:
        x_start = bin_edges[start_idx]
        x_end = bin_edges[len(best_k_per_bin)]
        x_end_clipped = min(x_end, upper_cap)
        if x_start < x_end_clipped:
            ranges.append((x_start, x_end_clipped, current_k))

    # Remove zero-length or duplicate-adjacent ranges, and also drop ranges with k<0 sentinel
    cleaned = []
    for r in ranges:
        xs, xe, k = r
        if xs >= xe:
            continue
        if cleaned and cleaned[-1][2] == k and np.isclose(cleaned[-1][1], xs):
            # merge with previous
            prev = cleaned.pop()
            cleaned.append((prev[0], xe, k))
        else:
            cleaned.append(r)

    return cleaned


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Path to musiq_per_image CSV.")
    ap.add_argument("--outdir", type=Path, default=Path("./musiq_out"), help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.95, help="Curve threshold for mean_y(k) ≥ thr.")
    ap.add_argument("--num_bins", type=int, default=10, help="Equal-width bins for x in line plot/rules.")
    ap.add_argument("--total_steps", type=int, default=50, help="Total denoising steps of diffusion model.")
    ap.add_argument("--serve_mvsplat_k", type=int, default=50, help="Sentinel for 'serve MVSplat'.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    required = {"scene/image", "mvsplat", "context_000", "context_100"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # k columns (numeric strings)
    k_cols = sorted([c for c in df.columns if re.fullmatch(r"\d+", str(c))], key=lambda x: int(x))
    if "0" not in k_cols:
        raise ValueError("Expected a '0' column for k=0 baseline.")

    # Extract 000..003, compute benchmark & ratios
    stems = df["scene/image"].astype(str).apply(lambda p: Path(p).stem)
    df["frame_pos"] = stems.apply(extract_frame_pos)
    df = df[df["frame_pos"].isin([0, 1, 2, 3])].copy()
    if df.empty:
        raise ValueError("No rows with frame_pos in {000,001,002,003}.")

    df["benchmark"] = compute_benchmark(df)
    df["x_ratio_mvsplat_over_benchmark"] = df["mvsplat"] / df["benchmark"]
    for k in k_cols:
        df[f"y_ratio_k{k}_over_k0"] = df[k] / df["0"]
        df[f"k{k}_meets_thr_vs_k0"] = df[k] >= (args.threshold * df["0"])

    # Per-row decision & steps (export convenience)
    df["k_decision"] = df.apply(
        lambda r: decide_k_for_row(r, k_cols, thr=args.threshold,
                                   x_col="x_ratio_mvsplat_over_benchmark",
                                   k0_col="0",
                                   serve_mvsplat_k=args.serve_mvsplat_k),
        axis=1
    )
    df["run_steps"] = np.where(
        df["k_decision"] == args.serve_mvsplat_k,
        0,
        np.maximum(args.total_steps - df["k_decision"], 0)
    )

    # Save tables
    full_csv = args.outdir / "musiq_ratios_and_thresholds.csv"
    df.to_csv(full_csv, index=False)

    ratio_cols = ["scene/image", "frame_pos", "x_ratio_mvsplat_over_benchmark"] + \
                 [f"y_ratio_k{k}_over_k0" for k in k_cols]
    df[ratio_cols].to_csv(args.outdir / "musiq_compact_ratios.csv", index=False)

    # -------- Scatter plot --------
    plt.figure(figsize=(7.5, 5.0))
    x = df["x_ratio_mvsplat_over_benchmark"].to_numpy()
    for k in k_cols:
        if int(k) == 0:
            continue
        y = df[f"y_ratio_k{k}_over_k0"].to_numpy()
        plt.scatter(x, y, s=12, alpha=0.7, label=f"k={k}")
    plt.axhline(args.threshold, linestyle="--", linewidth=1.0)
    plt.xlabel("MVSplat / Benchmark (MUSIQ)")
    plt.ylabel("Score(k) / Score(k=0)")
    plt.title("Per-frame Ratios vs. Benchmark (MUSIQ)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(title="Skip steps", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.outdir / "musiq_ratio_scatter.png", dpi=200)
    plt.close()

    # -------- Binned curves (and rule extraction) --------
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x_ratio_mvsplat_over_benchmark"] + [f"y_ratio_k{k}_over_k0" for k in k_cols]
    ).copy()

    rules_text_lines = []
    gb = None

    if not valid.empty:
        x_min = valid["x_ratio_mvsplat_over_benchmark"].min()
        x_max = valid["x_ratio_mvsplat_over_benchmark"].max()
        if np.isclose(x_min, x_max):
            x_max = x_min + 1e-6

        # Bin edges and centers
        bin_edges = np.linspace(x_min, x_max, args.num_bins + 1)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        valid["x_bin"] = pd.cut(valid["x_ratio_mvsplat_over_benchmark"],
                                bins=bin_edges, include_lowest=True, labels=centers)

        # Aggregate per bin
        agg = {"count": ("x_ratio_mvsplat_over_benchmark", "size"),
               "x_center": ("x_bin", "first")}
        for k in k_cols:
            agg[f"mean_y_k{k}"] = (f"y_ratio_k{k}_over_k0", "mean")
        gb = valid.groupby("x_bin").agg(**agg).reset_index(drop=True)
        gb.to_csv(args.outdir / "musiq_binned_means.csv", index=False)

        # Plot curves
        plt.figure(figsize=(7.5, 5.0))
        xline = gb["x_center"].astype(float).to_numpy()
        for k in k_cols:
            if int(k) == 0:
                continue
            yline = gb[f"mean_y_k{k}"].to_numpy()
            plt.plot(xline, yline, marker="o", linewidth=1.5, label=f"k = {k}")
        plt.axhline(args.threshold, linestyle="--", linewidth=1.0)
        plt.xlabel("MVSplat / Benchmark (MUSIQ)")
        plt.ylabel("Score(k) / Score(k=0)")
        plt.title("Quality Factor vs. MVSplat Benchmark Ratio")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend(title="Skip steps", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.outdir / "musiq_curves_plot.png", dpi=200)
        plt.close()
        
        mask = xline < 1.05
        if mask.any():
            plt.figure(figsize=(7.5, 5.0))
            for k in k_cols:
                if int(k) == 0:
                    continue
                yline = gb[f"mean_y_k{k}"].to_numpy()
                plt.plot(xline[mask], yline[mask], marker="o", linewidth=1.5, label=f"k = {k}")
            plt.axhline(args.threshold, linestyle="--", linewidth=1.0)
            plt.xlabel("MVSplat / Benchmark (MUSIQ)")
            plt.ylabel("Score(k) / Score(k=0)")
            plt.title("Quality Factor vs. MVSplat Benchmark Ratio (x < 1.0)")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            plt.legend(title="Skip steps", fontsize=8)
            plt.tight_layout()
            plt.savefig(args.outdir / "musiq_curves_plot_xlt1.png", dpi=200)
            plt.close()
        # -------- Extract x→k rules from curves (for x < 1.0) --------
        # For each bin, choose largest k with mean_y_k >= threshold; else 0.
        best_k_per_bin = []
        for _, row in gb.iterrows():
            elig = []
            for k in k_cols:
                if int(k) == 0:
                    continue
                if row[f"mean_y_k{k}"] >= args.threshold:
                    elig.append(int(k))
            best_k_per_bin.append(max(elig) if elig else 0)

        # Merge contiguous bins to ranges, clipped to upper_cap=1.0
        ranges = merge_bin_rules_to_ranges(bin_edges, best_k_per_bin, upper_cap=1.0)

        # Human-readable rules
        rules_text_lines.append(f"Curve-based x→k rules (threshold = {args.threshold:.2f}):")
        if ranges:
            for (xs, xe, k) in ranges:
                rules_text_lines.append(f"- For x in [{xs:.3f}, {xe:.3f}): choose k = {k}")
        else:
            rules_text_lines.append("- (No x<1.0 bins met the threshold; fall back to k=0 there.)")
        rules_text_lines.append(f"- For x ≥ 1.000: choose k = {args.serve_mvsplat_k} (serve MVSplat directly)")
    else:
        rules_text_lines.append("No valid rows for binning/curves; cannot derive x→k rules from curves.")
        rules_text_lines.append(f"- Default rule: For x ≥ 1.000 → k = {args.serve_mvsplat_k} (serve MVSplat).")

    # -------- Aggregate stats (rows) --------
    avg_k = df["k_decision"].replace(args.serve_mvsplat_k, np.nan).dropna().astype(float).mean()
    frac_serve = (df["k_decision"] == args.serve_mvsplat_k).mean()
    avg_run_steps = df["run_steps"].mean()

    summary_lines = [
        "K Decision Logic (summary):",
        "1) Compute x = mvsplat / benchmark, benchmark via linear interp between contexts at t={1/5,2/5,3/5,4/5}.",
        "2) From binned curves of mean_y(k)=mean(score(k)/score(k=0)) vs x:",
        f"   - For x < 1.0, choose largest k with mean_y(k) ≥ {args.threshold:.2f} (merged into x ranges below).",
        f"   - For x ≥ 1.0, choose k = {args.serve_mvsplat_k} (serve MVSplat).",
        "",
        *rules_text_lines,
        "",
        "Aggregate results on rows (per-row decisions):",
        f"- Average selected k (excluding serve-MVSplat): {0.0 if np.isnan(avg_k) else avg_k:.2f}",
        f"- Fraction serving MVSplat directly: {100*frac_serve:.2f}%",
        f"- Average diffusion steps to run: {avg_run_steps:.2f}",
    ]
    summary_path = args.outdir / "decision_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Console print
    print("\n".join(summary_lines))
    print(f"\nSaved files to: {args.outdir}")
    print(f"- Full table:            {full_csv}")
    print(f"- Compact ratios:        {args.outdir / 'musiq_compact_ratios.csv'}")
    print(f"- Binned means:          {args.outdir / 'musiq_binned_means.csv'}")
    print(f"- Scatter plot:          {args.outdir / 'musiq_ratio_scatter.png'}")
    print(f"- Curves plot:           {args.outdir / 'musiq_curves_plot.png'}")
    print(f"- Decision summary:      {summary_path}")


if __name__ == "__main__":
    main()
