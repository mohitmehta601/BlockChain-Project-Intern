from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create graphs from sweep_summary.csv."
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def line_plot(
    attacks: pd.DataFrame,
    baseline_value: float,
    y_column: str,
    y_label: str,
    title: str,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for remove_pct, group in attacks.groupby("remove_pct"):
        group = group.sort_values("top_k")
        ax.plot(
            group["top_k"],
            group[y_column],
            marker="o",
            label=f"{remove_pct:g}% removed",
        )

    ax.axhline(
        baseline_value,
        linestyle="--",
        label="Baseline",
    )
    ax.set_xlabel("Number of Top-K EBC channel pairs attacked")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def heatmap(
    attacks: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_file: Path,
) -> None:
    pivot = attacks.pivot(
        index="remove_pct",
        columns="top_k",
        values=value_column,
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}%" for value in pivot.index])

    ax.set_xlabel("Number of Top-K EBC channel pairs attacked")
    ax.set_ylabel("Liquidity removed from each selected pair")
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.out_dir)
    summary = pd.read_csv(args.summary, low_memory=False)

    baseline_rows = summary[summary["experiment_type"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError(
            "sweep_summary.csv must contain exactly one baseline row."
        )

    baseline = baseline_rows.iloc[0]
    attacks = summary[summary["experiment_type"] == "ebc_attack"].copy()

    line_plot(
        attacks=attacks,
        baseline_value=float(baseline["failure_rate_pct"]),
        y_column="failure_rate_pct",
        y_label="Failed transactions (%)",
        title="Failure rate under Top-K EBC liquidity shocks",
        output_file=output / "failure_rate_vs_top_k.png",
    )

    line_plot(
        attacks=attacks,
        baseline_value=float(baseline["success_rate_pct"]),
        y_column="success_rate_pct",
        y_label="Successful transactions (%)",
        title="Payment success rate under Top-K EBC liquidity shocks",
        output_file=output / "success_rate_vs_top_k.png",
    )

    line_plot(
        attacks=attacks,
        baseline_value=float(baseline["total_depletion_events"]),
        y_column="total_depletion_events",
        y_label="Total directed-edge depletion events",
        title="Liquidity depletion events under Top-K EBC shocks",
        output_file=output / "depletion_events_vs_top_k.png",
    )

    line_plot(
        attacks=attacks,
        baseline_value=float(baseline["average_path_length"]),
        y_column="average_path_length",
        y_label="Average routed path length (hops)",
        title="Average successful-route path length under EBC shocks",
        output_file=output / "average_path_length_vs_top_k.png",
    )

    line_plot(
        attacks=attacks,
        baseline_value=float(baseline["average_routing_fee"]),
        y_column="average_routing_fee",
        y_label="Average routing fee (satoshis)",
        title="Average successful-payment routing fee under EBC shocks",
        output_file=output / "average_fee_vs_top_k.png",
    )

    heatmap(
        attacks=attacks,
        value_column="failure_rate_pct",
        title="Failure-rate heatmap for EBC liquidity shocks",
        colorbar_label="Failed transactions (%)",
        output_file=output / "failure_rate_heatmap.png",
    )

    heatmap(
        attacks=attacks,
        value_column="failed_transactions_change_pct",
        title="Relative increase in failed transactions compared with baseline",
        colorbar_label="Change compared with baseline (%)",
        output_file=output / "failed_transactions_change_heatmap.png",
    )

    print("\nGraphs saved in:")
    print(output)


if __name__ == "__main__":
    main()
