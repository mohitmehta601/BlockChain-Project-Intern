"""
Generate CLSA graphs.

Place this file inside:
    LNTrafficSimulator/CLSA/plot_clsa_results.py

Run from the LNTrafficSimulator root directory:
    python .\\CLSA\\plot_clsa_results.py --input .\\output\\clsa_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_throughput_drop_vs_budget(
    budget_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))

    plt.bar(
        budget_df["budget"].astype(str),
        budget_df["throughput_drop_percent"],
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Throughput Drop (%)")
    plt.title("CLSA: Throughput Drop vs Attack Budget")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "clsa_throughput_drop_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_caf_vs_budget(
    budget_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        budget_df["budget"],
        budget_df["CAF"],
        marker="o",
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Cascade Amplification Factor (CAF)")
    plt.title("CLSA: Cascade Amplification vs Attack Budget")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "clsa_caf_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_success_ratio_vs_budget(
    budget_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        budget_df["budget"],
        budget_df["success_ratio"],
        marker="o",
    )

    plt.xlabel("Number of Jammed Channels (k)")
    plt.ylabel("Payment Success Ratio")
    plt.title("CLSA: Payment Success Ratio Under Attack")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "clsa_success_ratio_vs_budget.png",
        dpi=180,
    )

    plt.close()


def plot_throughput_drop_vs_jam_ratio(
    jam_ratio_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        jam_ratio_df["jam_ratio"],
        jam_ratio_df["throughput_drop_percent"],
        marker="o",
    )

    plt.xlabel("Jam Ratio")
    plt.ylabel("Throughput Drop (%)")
    plt.title("CLSA: Impact of Partial and Full Channel Jamming")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "clsa_throughput_drop_vs_jam_ratio.png",
        dpi=180,
    )

    plt.close()


def plot_centrality_vs_random(
    comparison_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))

    plt.bar(
        comparison_df["strategy"],
        comparison_df["mean_throughput_drop_percent"],
    )

    plt.xlabel("Target-Selection Strategy")
    plt.ylabel("Mean Throughput Drop (%)")
    plt.title("CLSA: Centrality-Based vs Random Targeting")

    plt.tight_layout()

    plt.savefig(
        figures_dir / "clsa_centrality_vs_random.png",
        dpi=180,
    )

    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graphs from CLSA CSV outputs."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the output directory created by run_clsa_attack.py.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    figures_dir = args.input / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    budget_df = pd.read_csv(
        args.input / "clsa_budget_results.csv"
    )

    jam_ratio_df = pd.read_csv(
        args.input / "clsa_jam_ratio_results.csv"
    )

    comparison_df = pd.read_csv(
        args.input / "clsa_targeting_comparison.csv"
    )

    plot_throughput_drop_vs_budget(
        budget_df,
        figures_dir,
    )

    plot_caf_vs_budget(
        budget_df,
        figures_dir,
    )

    plot_success_ratio_vs_budget(
        budget_df,
        figures_dir,
    )

    plot_throughput_drop_vs_jam_ratio(
        jam_ratio_df,
        figures_dir,
    )

    plot_centrality_vs_random(
        comparison_df,
        figures_dir,
    )

    print("\nGraphs saved in:")
    print(figures_dir.resolve())


if __name__ == "__main__":
    main()