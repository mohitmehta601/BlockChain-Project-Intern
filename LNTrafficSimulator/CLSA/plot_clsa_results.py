"""
Generate graphs for the improved offline CLSA resilience experiment.

Place this file inside:
    LNTrafficSimulator/CLSA/plot_clsa_results.py

Run from the LNTrafficSimulator root directory:
    python .\\CLSA\\plot_clsa_results.py --input .\\output\\clsa_run_improved
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def numeric(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> np.ndarray:
    if column not in dataframe.columns:
        return np.full(
            len(dataframe),
            default,
            dtype=float,
        )

    return (
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(default)
        .to_numpy(dtype=float)
    )


def annotate_line(
    x_values,
    y_values,
) -> None:
    for x_value, y_value in zip(
        x_values,
        y_values,
    ):
        plt.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )


def annotate_bars(
    bars,
) -> None:
    for bar in bars:
        height = float(bar.get_height())

        plt.annotate(
            f"{height:.3f}",
            (
                bar.get_x()
                + bar.get_width() / 2.0,
                height,
            ),
            textcoords="offset points",
            xytext=(0, 5 if height >= 0 else -13),
            ha="center",
            fontsize=8,
        )


def load_results(
    input_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_df = pd.read_csv(
        input_directory
        / "baseline_metrics_summary.csv"
    )

    budget_df = pd.read_csv(
        input_directory
        / "clsa_budget_results.csv"
    )

    jam_ratio_df = pd.read_csv(
        input_directory
        / "clsa_jam_ratio_results.csv"
    )

    comparison_df = pd.read_csv(
        input_directory
        / "clsa_targeting_comparison.csv"
    )

    return (
        baseline_df,
        budget_df,
        jam_ratio_df,
        comparison_df,
    )


def plot_throughput_drop_vs_budget(
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "throughput_drop_percent_mean",
    )

    y_errors = numeric(
        dataframe,
        "throughput_drop_percent_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    plt.axhline(
        y=0.0,
        linewidth=1,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Throughput Drop (%)")
    plt.title("CLSA: Throughput Drop vs Attack Budget")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_throughput_drop_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_caf_vs_budget(
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "CAF_mean",
    )

    y_errors = numeric(
        dataframe,
        "CAF_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Cascade Amplification Factor (CAF)")
    plt.title("CLSA: Cascade Amplification vs Attack Budget")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_caf_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_success_ratio_vs_budget(
    baseline_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "success_ratio_mean",
    )

    y_errors = numeric(
        dataframe,
        "success_ratio_std",
    )

    baseline_ratio = float(
        baseline_df.iloc[0][
            "success_ratio_mean"
        ]
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
        label="Attacked network",
    )

    plt.axhline(
        y=baseline_ratio,
        linewidth=1,
        linestyle="--",
        label="Baseline",
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Payment Success Ratio")
    plt.title("CLSA: Payment Success Ratio Under Attack")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_success_ratio_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_throughput_drop_vs_jam_ratio(
    jam_ratio_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = jam_ratio_df.sort_values(
        "jam_ratio"
    )

    x_values = numeric(
        dataframe,
        "jam_ratio",
    )

    y_values = numeric(
        dataframe,
        "throughput_drop_percent_mean",
    )

    y_errors = numeric(
        dataframe,
        "throughput_drop_percent_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    plt.axhline(
        y=0.0,
        linewidth=1,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Jam Ratio")
    plt.ylabel("Mean Throughput Drop (%)")
    plt.title("CLSA: Partial and Full Liquidity Shocks")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_throughput_drop_vs_jam_ratio.png",
        dpi=180,
    )

    plt.close()


def plot_selector_comparison(
    comparison_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = comparison_df.sort_values(
        "throughput_drop_percent_mean",
        ascending=False,
    )

    labels = dataframe[
        "strategy"
    ].astype(str).tolist()

    y_values = numeric(
        dataframe,
        "throughput_drop_percent_mean",
    )

    y_errors = numeric(
        dataframe,
        "throughput_drop_percent_std",
    )

    plt.figure(figsize=(8.5, 4.8))

    bars = plt.bar(
        labels,
        y_values,
        yerr=y_errors,
        capsize=4,
    )

    plt.axhline(
        y=0.0,
        linewidth=1,
    )

    annotate_bars(
        bars,
    )

    plt.xlabel("Target-Selection Strategy")
    plt.ylabel("Mean Throughput Drop (%)")
    plt.title("CLSA: Comparison of Target-Selection Strategies")
    plt.xticks(
        rotation=18,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_selector_comparison.png",
        dpi=180,
    )

    plt.close()


def plot_path_length_vs_budget(
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "average_successful_path_length_mean",
    )

    y_errors = numeric(
        dataframe,
        "average_successful_path_length_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Successful Path Length")
    plt.title("CLSA: Route-Length Change Under Liquidity Shock")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_path_length_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_fee_vs_budget(
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "average_successful_fee_mean",
    )

    y_errors = numeric(
        dataframe,
        "average_successful_fee_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Fee of Successful Payments")
    plt.title("CLSA: Fee Change Under Liquidity Shock")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_fee_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_depletion_events_vs_budget(
    budget_df: pd.DataFrame,
    figures_directory: Path,
) -> None:
    dataframe = budget_df.sort_values("budget")

    x_values = numeric(
        dataframe,
        "budget",
    )

    y_values = numeric(
        dataframe,
        "depletion_events_mean",
    )

    y_errors = numeric(
        dataframe,
        "depletion_events_std",
    )

    plt.figure(figsize=(7.5, 4.8))

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    annotate_line(
        x_values,
        y_values,
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Mean Depletion Events")
    plt.title("CLSA: Directional Depletion Pressure")

    plt.tight_layout()

    plt.savefig(
        figures_directory
        / "clsa_depletion_events_vs_budget.png",
        dpi=180,
    )

    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate graphs from the improved CLSA CSV outputs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    figures_directory = (
        args.input
        / "figures"
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        baseline_df,
        budget_df,
        jam_ratio_df,
        comparison_df,
    ) = load_results(
        args.input
    )

    plot_throughput_drop_vs_budget(
        budget_df,
        figures_directory,
    )

    plot_caf_vs_budget(
        budget_df,
        figures_directory,
    )

    plot_success_ratio_vs_budget(
        baseline_df,
        budget_df,
        figures_directory,
    )

    plot_throughput_drop_vs_jam_ratio(
        jam_ratio_df,
        figures_directory,
    )

    plot_selector_comparison(
        comparison_df,
        figures_directory,
    )

    plot_path_length_vs_budget(
        budget_df,
        figures_directory,
    )

    plot_fee_vs_budget(
        budget_df,
        figures_directory,
    )

    plot_depletion_events_vs_budget(
        budget_df,
        figures_directory,
    )

    print("\nGraphs saved in:")
    print(figures_directory.resolve())


if __name__ == "__main__":
    main()
