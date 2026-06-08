"""
Improved offline CLSA resilience experiment for LNTrafficSimulator.

Place this file inside:
    LNTrafficSimulator/CLSA/run_clsa_attack.py

The script keeps the static liquidity-shock approximation, but improves:
1. fee-weighted target selection aligned with simulator routing,
2. baseline-usage-aware targeting,
3. a hybrid selector,
4. multiple random seeds,
5. selector comparison,
6. richer metrics.

Run from the LNTrafficSimulator root directory:
    python .\\CLSA\\run_clsa_attack.py --approx-k 500

Example multi-seed experiment:
    python .\\CLSA\\run_clsa_attack.py `
      --approx-k 500 `
      --seeds 20260601 20260602 20260603 20260604 20260605 `
      --budgets 5 10 20 50 100 200 `
      --jam-sweep-k 50 `
      --main-selector hybrid

This code is intended only for controlled, offline simulation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lnsimulator.ln_utils import preprocess_json_file
import lnsimulator.simulator.transaction_simulator as ts
from lnsimulator.simulator.graph_preprocessing import prepare_edges_for_simulation


SUPPORTED_SELECTORS = (
    "unweighted_betweenness",
    "fee_weighted_betweenness",
    "baseline_usage",
    "hybrid",
)


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        params = json.load(file)

    required_keys = {
        "amount",
        "count",
        "epsilon",
        "drop_disabled",
        "drop_low_cap",
        "with_depletion",
    }

    missing_keys = required_keys.difference(params.keys())

    if missing_keys:
        raise ValueError(
            f"Missing keys in {params_path}: {sorted(missing_keys)}"
        )

    return params


def load_providers(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        print(
            f"WARNING: {metadata_path} was not found.\n"
            "The merchant list will be empty and epsilon will be forced to 0.0."
        )
        return []

    metadata = pd.read_csv(metadata_path)

    if "pub_key" not in metadata.columns:
        raise ValueError(
            "The metadata file must contain a 'pub_key' column.\n"
            f"Columns found: {metadata.columns.tolist()}"
        )

    return (
        metadata["pub_key"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def load_directed_edges(network_json_path: Path) -> pd.DataFrame:
    directed_edges = preprocess_json_file(str(network_json_path)).copy()

    required_columns = {"src", "trg", "capacity"}
    missing_columns = required_columns.difference(directed_edges.columns)

    if missing_columns:
        raise ValueError(
            "The preprocessed graph does not contain the expected columns.\n"
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {directed_edges.columns.tolist()}"
        )

    directed_edges["src"] = directed_edges["src"].astype(str)
    directed_edges["trg"] = directed_edges["trg"].astype(str)

    directed_edges["capacity"] = (
        pd.to_numeric(directed_edges["capacity"], errors="coerce")
        .fillna(0.0)
    )

    print("\n======================================================")
    print("PREPROCESSED EDGE COLUMNS")
    print("======================================================")
    print(directed_edges.columns.tolist())

    print("\n======================================================")
    print("FIRST TWO DIRECTED EDGES")
    print("======================================================")
    print(directed_edges.head(2).to_string(index=False))

    return directed_edges


def save_workload(workload, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(workload, pd.DataFrame):
        workload.to_csv(output_path, index=False)
        return

    try:
        pd.DataFrame(workload).to_csv(output_path, index=False)
    except Exception:
        print("WARNING: Transaction workload could not be exported as CSV.")


def safe_depletion_count(total_depletions) -> int:
    if total_depletions is None:
        return 0

    if isinstance(total_depletions, dict):
        return int(sum(total_depletions.values()))

    if isinstance(total_depletions, (list, tuple, set)):
        return int(len(total_depletions))

    try:
        return int(total_depletions)
    except (TypeError, ValueError):
        return 0


def strip_pseudo_target(node: object) -> str:
    value = str(node)

    if value.endswith("_trg"):
        return value[:-4]

    return value


def canonical_pair(node_a: object, node_b: object) -> tuple[str, str]:
    a = strip_pseudo_target(node_a)
    b = strip_pseudo_target(node_b)

    return tuple(sorted((a, b)))


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)

    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if maximum <= minimum:
        return pd.Series(
            np.zeros(len(numeric)),
            index=numeric.index,
            dtype=float,
        )

    return (numeric - minimum) / (maximum - minimum)


# ---------------------------------------------------------
# Baseline simulation and metrics
# ---------------------------------------------------------

def calculate_simulation_metrics(
    shortest_paths: pd.DataFrame,
    total_depletions,
) -> dict:
    if "length" not in shortest_paths.columns:
        raise ValueError(
            "The simulator path output does not contain a 'length' column.\n"
            f"Columns found: {shortest_paths.columns.tolist()}"
        )

    success_mask = shortest_paths["length"] > -1

    total_transactions = int(len(shortest_paths))
    successful_transactions = int(success_mask.sum())
    failed_transactions = total_transactions - successful_transactions

    success_ratio = (
        successful_transactions / total_transactions
        if total_transactions
        else 0.0
    )

    average_fee = float("nan")

    if (
        "original_cost" in shortest_paths.columns
        and successful_transactions > 0
    ):
        average_fee = float(
            shortest_paths.loc[success_mask, "original_cost"].mean()
        )

    average_path_length = float("nan")

    if successful_transactions > 0:
        average_path_length = float(
            shortest_paths.loc[success_mask, "length"].mean()
        )

    return {
        "total_transactions": total_transactions,
        "successful_transactions": successful_transactions,
        "failed_transactions": failed_transactions,
        "success_ratio": success_ratio,
        "average_successful_fee": average_fee,
        "average_successful_path_length": average_path_length,
        "depletion_events": safe_depletion_count(total_depletions),
    }


def run_simulation(
    directed_edges: pd.DataFrame,
    providers: list[str],
    params: dict,
    output_directory: Path,
    seed: int,
    fixed_workload=None,
    max_threads: int = 2,
) -> tuple[pd.DataFrame, object, dict]:
    output_directory.mkdir(parents=True, exist_ok=True)

    epsilon = float(params["epsilon"]) if providers else 0.0

    set_random_seed(seed)

    simulator = ts.TransactionSimulator(
        directed_edges.copy(),
        providers,
        int(params["amount"]),
        int(params["count"]),
        epsilon=epsilon,
        drop_disabled=bool(params["drop_disabled"]),
        drop_low_cap=bool(params["drop_low_cap"]),
        with_depletion=bool(params["with_depletion"]),
    )

    if fixed_workload is not None:
        simulator.transactions = copy.deepcopy(fixed_workload)

    # The simulator randomly initializes directional liquidity.
    # Reusing the seed improves reproducibility across scenarios.
    set_random_seed(seed)

    shortest_paths, _, _, total_depletions = simulator.simulate(
        weight="total_fee",
        with_node_removals=False,
        max_threads=max_threads,
    )

    simulator.export(str(output_directory))

    shortest_paths.to_csv(
        output_directory / "paths.csv",
        index=False,
    )

    save_workload(
        simulator.transactions,
        output_directory / "transactions.csv",
    )

    metrics = calculate_simulation_metrics(
        shortest_paths,
        total_depletions,
    )

    pd.DataFrame([metrics]).to_csv(
        output_directory / "metrics.csv",
        index=False,
    )

    return (
        shortest_paths,
        copy.deepcopy(simulator.transactions),
        metrics,
    )


# ---------------------------------------------------------
# Channel ranking
# ---------------------------------------------------------

def prepare_ranking_edges(
    directed_edges: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Apply the same edge filtering and fee calculation used by the simulator.
    """
    prepared = prepare_edges_for_simulation(
        directed_edges.copy(),
        int(params["amount"]),
        bool(params["drop_disabled"]),
        bool(params["drop_low_cap"]),
        time_window=None,
        verbose=False,
    ).copy()

    if prepared.empty:
        raise RuntimeError(
            "No usable channels remain after simulator preprocessing."
        )

    prepared["src"] = prepared["src"].astype(str)
    prepared["trg"] = prepared["trg"].astype(str)

    prepared["total_fee"] = (
        pd.to_numeric(prepared["total_fee"], errors="coerce")
        .fillna(0.0)
    )

    # NetworkX treats edge weights as distances. Fees must be positive.
    prepared["routing_distance"] = prepared["total_fee"].clip(lower=1e-9)

    return prepared


def build_directed_ranking_graph(
    prepared_edges: pd.DataFrame,
) -> nx.DiGraph:
    return nx.from_pandas_edgelist(
        prepared_edges,
        source="src",
        target="trg",
        edge_attr=[
            "capacity",
            "total_fee",
            "routing_distance",
        ],
        create_using=nx.DiGraph(),
    )


def calculate_pair_betweenness(
    graph: nx.DiGraph,
    weight: str | None,
    approx_k: int,
    seed: int,
) -> pd.DataFrame:
    if approx_k > 0:
        sampled_nodes = min(
            int(approx_k),
            graph.number_of_nodes(),
        )

        directed_scores = nx.edge_betweenness_centrality(
            graph,
            k=sampled_nodes,
            normalized=True,
            weight=weight,
            seed=seed,
        )
    else:
        directed_scores = nx.edge_betweenness_centrality(
            graph,
            normalized=True,
            weight=weight,
        )

    records = []

    for (src, trg), score in directed_scores.items():
        u, v = canonical_pair(src, trg)

        records.append(
            {
                "u": u,
                "v": v,
                "directed_score": float(score),
            }
        )

    return (
        pd.DataFrame(records)
        .groupby(["u", "v"], as_index=False)["directed_score"]
        .sum()
    )


def count_baseline_channel_usage(
    shortest_paths: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count successful baseline routes that used each bidirectional channel.

    The final node in a simulator path is a pseudo target ending in '_trg'.
    It is mapped back to the real target node before counting channel usage.
    """
    usage_counter: Counter[tuple[str, str]] = Counter()

    successful_paths = shortest_paths[
        shortest_paths["length"] > -1
    ]

    for path in successful_paths["path"]:
        if not isinstance(path, (list, tuple)):
            continue

        for src, trg in zip(path[:-1], path[1:]):
            u, v = canonical_pair(src, trg)

            if u == v:
                continue

            usage_counter[(u, v)] += 1

    records = [
        {
            "u": u,
            "v": v,
            "baseline_usage": count,
        }
        for (u, v), count in usage_counter.items()
    ]

    if not records:
        return pd.DataFrame(
            columns=["u", "v", "baseline_usage"]
        )

    return pd.DataFrame(records)


def build_channel_rankings(
    prepared_edges: pd.DataFrame,
    baseline_paths: pd.DataFrame,
    approx_k: int,
    centrality_seed: int,
    hybrid_alpha: float,
) -> pd.DataFrame:
    """
    Build four rankings:
    1. unweighted betweenness,
    2. fee-weighted betweenness,
    3. baseline usage,
    4. hybrid score.

    hybrid_score =
        hybrid_alpha * normalized fee-weighted betweenness
        + (1 - hybrid_alpha) * normalized baseline usage
    """
    graph = build_directed_ranking_graph(prepared_edges)

    print("\n======================================================")
    print("CHANNEL RANKING")
    print("======================================================")
    print(f"Directed graph nodes: {graph.number_of_nodes()}")
    print(f"Directed graph edges: {graph.number_of_edges()}")

    if approx_k > 0:
        print(
            "Centrality mode: approximate calculation "
            f"with k={min(approx_k, graph.number_of_nodes())}"
        )
    else:
        print("Centrality mode: exact calculation")
        print("NOTE: Exact calculation can take time on a large graph.")

    base_pairs = prepared_edges[["src", "trg", "capacity"]].copy()

    base_pairs[["u", "v"]] = base_pairs.apply(
        lambda row: pd.Series(
            canonical_pair(row["src"], row["trg"])
        ),
        axis=1,
    )

    base_pairs = (
        base_pairs
        .groupby(["u", "v"], as_index=False)
        .agg(channel_capacity_sat=("capacity", "max"))
    )

    unweighted = calculate_pair_betweenness(
        graph=graph,
        weight=None,
        approx_k=approx_k,
        seed=centrality_seed,
    ).rename(
        columns={
            "directed_score": "unweighted_betweenness"
        }
    )

    fee_weighted = calculate_pair_betweenness(
        graph=graph,
        weight="routing_distance",
        approx_k=approx_k,
        seed=centrality_seed,
    ).rename(
        columns={
            "directed_score": "fee_weighted_betweenness"
        }
    )

    baseline_usage = count_baseline_channel_usage(
        baseline_paths
    )

    rankings = (
        base_pairs
        .merge(unweighted, on=["u", "v"], how="left")
        .merge(fee_weighted, on=["u", "v"], how="left")
        .merge(baseline_usage, on=["u", "v"], how="left")
        .fillna(
            {
                "unweighted_betweenness": 0.0,
                "fee_weighted_betweenness": 0.0,
                "baseline_usage": 0,
            }
        )
    )

    rankings["baseline_usage"] = (
        pd.to_numeric(
            rankings["baseline_usage"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    rankings["unweighted_betweenness_norm"] = (
        normalize_series(
            rankings["unweighted_betweenness"]
        )
    )

    rankings["fee_weighted_betweenness_norm"] = (
        normalize_series(
            rankings["fee_weighted_betweenness"]
        )
    )

    rankings["baseline_usage_norm"] = (
        normalize_series(
            rankings["baseline_usage"]
        )
    )

    rankings["hybrid_score"] = (
        float(hybrid_alpha)
        * rankings["fee_weighted_betweenness_norm"]
        + (1.0 - float(hybrid_alpha))
        * rankings["baseline_usage_norm"]
    )

    return rankings


def rank_for_selector(
    rankings: pd.DataFrame,
    selector: str,
) -> pd.DataFrame:
    selector_to_column = {
        "unweighted_betweenness": "unweighted_betweenness",
        "fee_weighted_betweenness": "fee_weighted_betweenness",
        "baseline_usage": "baseline_usage",
        "hybrid": "hybrid_score",
    }

    if selector not in selector_to_column:
        raise ValueError(
            f"Unsupported selector '{selector}'. "
            f"Choose from: {SUPPORTED_SELECTORS}"
        )

    score_column = selector_to_column[selector]

    ranked = (
        rankings
        .sort_values(
            [
                score_column,
                "baseline_usage",
                "fee_weighted_betweenness",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
        .copy()
    )

    ranked.insert(
        0,
        "rank",
        range(1, len(ranked) + 1),
    )

    ranked.insert(
        1,
        "selector",
        selector,
    )

    return ranked


def select_top_channels(
    ranked_channels: pd.DataFrame,
    number_of_channels: int,
) -> list[tuple[str, str]]:
    number_of_channels = min(
        max(1, int(number_of_channels)),
        len(ranked_channels),
    )

    return list(
        ranked_channels
        .head(number_of_channels)[["u", "v"]]
        .itertuples(index=False, name=None)
    )


def select_random_channels(
    rankings: pd.DataFrame,
    number_of_channels: int,
    seed: int,
) -> list[tuple[str, str]]:
    number_of_channels = min(
        max(1, int(number_of_channels)),
        len(rankings),
    )

    sampled = rankings.sample(
        n=number_of_channels,
        replace=False,
        random_state=seed,
    )

    return list(
        sampled[["u", "v"]]
        .itertuples(index=False, name=None)
    )


def save_selected_targets(
    selected_channels: Iterable[tuple[str, str]],
    rankings: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets_df = pd.DataFrame(
        list(selected_channels),
        columns=["u", "v"],
    )

    targets_df = targets_df.merge(
        rankings,
        on=["u", "v"],
        how="left",
    )

    targets_df.to_csv(
        output_path,
        index=False,
    )


# ---------------------------------------------------------
# Static CLSA approximation
# ---------------------------------------------------------

def inject_clsa(
    directed_edges: pd.DataFrame,
    target_channels: Iterable[tuple[str, str]],
    jam_ratio: float,
) -> tuple[pd.DataFrame, float]:
    """
    Static CLSA approximation.

    A jam_ratio of 0.8 means that 80% of the selected channels'
    capacity is treated as temporarily unavailable.

    Both directions of each selected bidirectional channel are reduced.
    """
    if not 0.0 <= jam_ratio <= 1.0:
        raise ValueError(
            "jam_ratio must be between 0.0 and 1.0."
        )

    attacked_edges = directed_edges.copy()

    target_set = {
        canonical_pair(u, v)
        for u, v in target_channels
    }

    row_pairs = [
        canonical_pair(src, trg)
        for src, trg in zip(
            attacked_edges["src"],
            attacked_edges["trg"],
        )
    ]

    target_mask = pd.Series(
        [
            pair in target_set
            for pair in row_pairs
        ],
        index=attacked_edges.index,
    )

    capacity_before = float(
        attacked_edges.loc[
            target_mask,
            "capacity",
        ].sum()
    )

    attacked_edges.loc[
        target_mask,
        "capacity",
    ] = np.floor(
        attacked_edges.loc[
            target_mask,
            "capacity",
        ].astype(float)
        * (1.0 - float(jam_ratio))
    )

    capacity_after = float(
        attacked_edges.loc[
            target_mask,
            "capacity",
        ].sum()
    )

    locked_capacity = (
        capacity_before - capacity_after
    )

    return attacked_edges, locked_capacity


def run_attack_scenario(
    original_edges: pd.DataFrame,
    providers: list[str],
    params: dict,
    baseline_workload,
    baseline_metrics: dict,
    target_channels: list[tuple[str, str]],
    jam_ratio: float,
    output_directory: Path,
    seed: int,
    max_threads: int,
) -> dict:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    attacked_edges, locked_capacity = inject_clsa(
        original_edges,
        target_channels,
        jam_ratio,
    )

    attacked_edges.to_csv(
        output_directory / "attacked_edges.csv",
        index=False,
    )

    _, _, attack_metrics = run_simulation(
        directed_edges=attacked_edges,
        providers=providers,
        params=params,
        output_directory=output_directory,
        seed=seed,
        fixed_workload=baseline_workload,
        max_threads=max_threads,
    )

    net_lost_successful_transactions = (
        int(baseline_metrics["successful_transactions"])
        - int(attack_metrics["successful_transactions"])
    )

    lost_successful_transactions = max(
        0,
        net_lost_successful_transactions,
    )

    throughput_drop = (
        float(baseline_metrics["success_ratio"])
        - float(attack_metrics["success_ratio"])
    )

    caf = (
        lost_successful_transactions
        / len(target_channels)
        if target_channels
        else 0.0
    )

    directed_capacity_total = float(
        pd.to_numeric(
            original_edges["capacity"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    locked_capacity_percent = (
        100.0
        * locked_capacity
        / directed_capacity_total
        if directed_capacity_total > 0
        else 0.0
    )

    attack_metrics.update(
        {
            "channels_jammed": len(target_channels),
            "jam_ratio": float(jam_ratio),
            "locked_capacity_sat": locked_capacity,
            "locked_capacity_percent": locked_capacity_percent,
            "net_lost_successful_transactions": (
                net_lost_successful_transactions
            ),
            "lost_successful_transactions": (
                lost_successful_transactions
            ),
            "throughput_drop": throughput_drop,
            "throughput_drop_percent": (
                throughput_drop * 100.0
            ),
            "CAF": caf,
        }
    )

    pd.DataFrame([attack_metrics]).to_csv(
        output_directory / "attack_metrics.csv",
        index=False,
    )

    return attack_metrics


# ---------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------

SUMMARY_METRICS = [
    "successful_transactions",
    "failed_transactions",
    "success_ratio",
    "average_successful_fee",
    "average_successful_path_length",
    "depletion_events",
    "locked_capacity_sat",
    "locked_capacity_percent",
    "net_lost_successful_transactions",
    "lost_successful_transactions",
    "throughput_drop_percent",
    "CAF",
]


def aggregate_results(
    raw_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    available_metrics = [
        metric
        for metric in SUMMARY_METRICS
        if metric in raw_df.columns
    ]

    if not available_metrics:
        raise ValueError(
            "No numeric result metrics were found for aggregation."
        )

    grouped = raw_df.groupby(
        group_columns,
        dropna=False,
    )[available_metrics]

    mean_df = (
        grouped
        .mean()
        .add_suffix("_mean")
        .reset_index()
    )

    std_df = (
        grouped
        .std(ddof=0)
        .fillna(0.0)
        .add_suffix("_std")
        .reset_index()
    )

    count_df = (
        grouped
        .size()
        .rename("runs")
        .reset_index()
    )

    return (
        mean_df
        .merge(
            std_df,
            on=group_columns,
            how="left",
        )
        .merge(
            count_df,
            on=group_columns,
            how="left",
        )
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an improved, offline CLSA resilience experiment "
            "using LNTrafficSimulator."
        )
    )

    parser.add_argument(
        "--network",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "sample.json",
    )

    parser.add_argument(
        "--meta",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "1ml_meta_data.csv",
    )

    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "params.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "clsa_run_improved",
    )

    parser.add_argument(
        "--main-selector",
        choices=SUPPORTED_SELECTORS,
        default="hybrid",
        help=(
            "Selector used for budget and jam-ratio sweeps. "
            "The selector comparison always evaluates all selectors."
        ),
    )

    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to normalized fee-weighted betweenness "
            "inside the hybrid score. The remaining weight is assigned "
            "to normalized baseline usage."
        ),
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50, 100, 200],
    )

    parser.add_argument(
        "--budget-jam-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--jam-ratios",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    parser.add_argument(
        "--jam-sweep-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--comparison-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--random-trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260606],
        help=(
            "Use multiple values for a statistically stronger experiment. "
            "Start with one seed to verify the workflow."
        ),
    )

    parser.add_argument(
        "--centrality-seed",
        type=int,
        default=20260606,
        help=(
            "Random seed used only when approximate betweenness is enabled."
        ),
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--approx-k",
        type=int,
        default=0,
        help=(
            "Approximate betweenness with this many sampled nodes. "
            "Use 0 for exact calculation."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.hybrid_alpha <= 1.0:
        parser.error("--hybrid-alpha must be between 0.0 and 1.0.")

    if not 0.0 <= args.budget_jam_ratio <= 1.0:
        parser.error("--budget-jam-ratio must be between 0.0 and 1.0.")

    for ratio in args.jam_ratios:
        if not 0.0 <= ratio <= 1.0:
            parser.error(
                "Each value passed to --jam-ratios "
                "must be between 0.0 and 1.0."
            )

    return args


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n======================================================")
    print("LOAD INPUTS")
    print("======================================================")

    params = load_params(args.params)

    original_edges = load_directed_edges(
        args.network
    )

    providers = load_providers(
        args.meta
    )

    if not providers:
        params["epsilon"] = 0.0

    print("\n======================================================")
    print("SIMULATOR PARAMETERS")
    print("======================================================")
    print(json.dumps(params, indent=2))
    print(f"Providers loaded: {len(providers)}")
    print(f"Seeds: {args.seeds}")
    print(f"Main selector: {args.main_selector}")

    prepared_ranking_edges = prepare_ranking_edges(
        original_edges,
        params,
    )

    budget_rows = []
    jam_ratio_rows = []
    comparison_rows = []
    baseline_rows = []

    first_seed_rankings: pd.DataFrame | None = None

    for seed in args.seeds:
        print("\n######################################################")
        print(f"SEED {seed}")
        print("######################################################")

        # -------------------------------------------------
        # Baseline
        # -------------------------------------------------

        print("\n======================================================")
        print("STEP 1: BASELINE SIMULATION")
        print("======================================================")

        baseline_directory = (
            args.output
            / "baseline"
            / f"seed_{seed}"
        )

        (
            baseline_paths,
            baseline_workload,
            baseline_metrics,
        ) = run_simulation(
            directed_edges=original_edges,
            providers=providers,
            params=params,
            output_directory=baseline_directory,
            seed=seed,
            max_threads=args.max_threads,
        )

        baseline_rows.append(
            {
                "seed": seed,
                **baseline_metrics,
            }
        )

        # -------------------------------------------------
        # Rankings
        # -------------------------------------------------

        print("\n======================================================")
        print("STEP 2: BUILD CHANNEL RANKINGS")
        print("======================================================")

        rankings = build_channel_rankings(
            prepared_edges=prepared_ranking_edges,
            baseline_paths=baseline_paths,
            approx_k=args.approx_k,
            centrality_seed=args.centrality_seed,
            hybrid_alpha=args.hybrid_alpha,
        )

        ranking_directory = (
            args.output
            / "rankings"
            / f"seed_{seed}"
        )

        ranking_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        rankings.to_csv(
            ranking_directory / "all_channel_scores.csv",
            index=False,
        )

        for selector in SUPPORTED_SELECTORS:
            ranked = rank_for_selector(
                rankings,
                selector,
            )

            ranked.to_csv(
                ranking_directory
                / f"ranking_{selector}.csv",
                index=False,
            )

            ranked.head(50).to_csv(
                ranking_directory
                / f"top50_{selector}.csv",
                index=False,
            )

        if first_seed_rankings is None:
            first_seed_rankings = rankings.copy()

        main_ranked = rank_for_selector(
            rankings,
            args.main_selector,
        )

        # -------------------------------------------------
        # Budget sweep
        # -------------------------------------------------

        print("\n======================================================")
        print("STEP 3: ATTACK-BUDGET SWEEP")
        print("======================================================")

        for budget in args.budgets:
            targets = select_top_channels(
                main_ranked,
                budget,
            )

            scenario_directory = (
                args.output
                / "budget_runs"
                / args.main_selector
                / f"seed_{seed}"
                / f"k_{len(targets)}"
            )

            save_selected_targets(
                targets,
                rankings,
                scenario_directory / "targets.csv",
            )

            print(
                f"\nBudget sweep: selector={args.main_selector}, "
                f"seed={seed}, k={len(targets)}, "
                f"jam_ratio={args.budget_jam_ratio}"
            )

            scenario_metrics = run_attack_scenario(
                original_edges=original_edges,
                providers=providers,
                params=params,
                baseline_workload=baseline_workload,
                baseline_metrics=baseline_metrics,
                target_channels=targets,
                jam_ratio=args.budget_jam_ratio,
                output_directory=scenario_directory,
                seed=seed,
                max_threads=args.max_threads,
            )

            budget_rows.append(
                {
                    "seed": seed,
                    "selector": args.main_selector,
                    "budget": len(targets),
                    **scenario_metrics,
                }
            )

        # -------------------------------------------------
        # Jam-ratio sweep
        # -------------------------------------------------

        print("\n======================================================")
        print("STEP 4: JAM-RATIO SWEEP")
        print("======================================================")

        jam_targets = select_top_channels(
            main_ranked,
            args.jam_sweep_k,
        )

        for jam_ratio in args.jam_ratios:
            scenario_directory = (
                args.output
                / "jam_ratio_runs"
                / args.main_selector
                / f"seed_{seed}"
                / f"jam_{jam_ratio:.2f}"
            )

            save_selected_targets(
                jam_targets,
                rankings,
                scenario_directory / "targets.csv",
            )

            print(
                f"\nJam-ratio sweep: selector={args.main_selector}, "
                f"seed={seed}, k={len(jam_targets)}, "
                f"jam_ratio={jam_ratio}"
            )

            scenario_metrics = run_attack_scenario(
                original_edges=original_edges,
                providers=providers,
                params=params,
                baseline_workload=baseline_workload,
                baseline_metrics=baseline_metrics,
                target_channels=jam_targets,
                jam_ratio=jam_ratio,
                output_directory=scenario_directory,
                seed=seed,
                max_threads=args.max_threads,
            )

            jam_ratio_rows.append(
                {
                    "seed": seed,
                    "selector": args.main_selector,
                    "budget": len(jam_targets),
                    **scenario_metrics,
                }
            )

        # -------------------------------------------------
        # Selector comparison
        # -------------------------------------------------

        print("\n======================================================")
        print("STEP 5: SELECTOR COMPARISON")
        print("======================================================")

        for selector in SUPPORTED_SELECTORS:
            ranked = rank_for_selector(
                rankings,
                selector,
            )

            targets = select_top_channels(
                ranked,
                args.comparison_k,
            )

            scenario_directory = (
                args.output
                / "selector_comparison"
                / selector
                / f"seed_{seed}"
            )

            save_selected_targets(
                targets,
                rankings,
                scenario_directory / "targets.csv",
            )

            print(
                f"\nSelector comparison: selector={selector}, "
                f"seed={seed}, k={len(targets)}, "
                f"jam_ratio={args.budget_jam_ratio}"
            )

            scenario_metrics = run_attack_scenario(
                original_edges=original_edges,
                providers=providers,
                params=params,
                baseline_workload=baseline_workload,
                baseline_metrics=baseline_metrics,
                target_channels=targets,
                jam_ratio=args.budget_jam_ratio,
                output_directory=scenario_directory,
                seed=seed,
                max_threads=args.max_threads,
            )

            comparison_rows.append(
                {
                    "seed": seed,
                    "trial": 1,
                    "strategy": selector,
                    "budget": len(targets),
                    **scenario_metrics,
                }
            )

        for trial in range(
            1,
            args.random_trials + 1,
        ):
            random_seed = (
                seed + 10_000 + trial
            )

            targets = select_random_channels(
                rankings,
                args.comparison_k,
                random_seed,
            )

            scenario_directory = (
                args.output
                / "selector_comparison"
                / "random"
                / f"seed_{seed}"
                / f"trial_{trial}"
            )

            save_selected_targets(
                targets,
                rankings,
                scenario_directory / "targets.csv",
            )

            print(
                f"\nSelector comparison: selector=random, "
                f"seed={seed}, trial={trial}, "
                f"k={len(targets)}, "
                f"jam_ratio={args.budget_jam_ratio}"
            )

            scenario_metrics = run_attack_scenario(
                original_edges=original_edges,
                providers=providers,
                params=params,
                baseline_workload=baseline_workload,
                baseline_metrics=baseline_metrics,
                target_channels=targets,
                jam_ratio=args.budget_jam_ratio,
                output_directory=scenario_directory,
                seed=seed,
                max_threads=args.max_threads,
            )

            comparison_rows.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "strategy": "random",
                    "budget": len(targets),
                    **scenario_metrics,
                }
            )

    # -----------------------------------------------------
    # Save raw results
    # -----------------------------------------------------

    baseline_raw_df = pd.DataFrame(
        baseline_rows
    )

    budget_raw_df = pd.DataFrame(
        budget_rows
    )

    jam_ratio_raw_df = pd.DataFrame(
        jam_ratio_rows
    )

    comparison_raw_df = pd.DataFrame(
        comparison_rows
    )

    baseline_raw_df.to_csv(
        args.output / "baseline_metrics_raw.csv",
        index=False,
    )

    budget_raw_df.to_csv(
        args.output / "clsa_budget_results_raw.csv",
        index=False,
    )

    jam_ratio_raw_df.to_csv(
        args.output / "clsa_jam_ratio_results_raw.csv",
        index=False,
    )

    comparison_raw_df.to_csv(
        args.output / "clsa_targeting_comparison_raw.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save summary results
    # -----------------------------------------------------

    baseline_summary_df = aggregate_results(
        baseline_raw_df,
        group_columns=[],
    ) if False else pd.DataFrame(
        [
            {
                **{
                    f"{column}_mean": float(
                        pd.to_numeric(
                            baseline_raw_df[column],
                            errors="coerce",
                        ).mean()
                    )
                    for column in [
                        "successful_transactions",
                        "failed_transactions",
                        "success_ratio",
                        "average_successful_fee",
                        "average_successful_path_length",
                        "depletion_events",
                    ]
                },
                **{
                    f"{column}_std": float(
                        pd.to_numeric(
                            baseline_raw_df[column],
                            errors="coerce",
                        ).std(ddof=0)
                    )
                    for column in [
                        "successful_transactions",
                        "failed_transactions",
                        "success_ratio",
                        "average_successful_fee",
                        "average_successful_path_length",
                        "depletion_events",
                    ]
                },
                "runs": len(baseline_raw_df),
            }
        ]
    )

    budget_summary_df = aggregate_results(
        budget_raw_df,
        group_columns=[
            "selector",
            "budget",
            "jam_ratio",
        ],
    )

    jam_ratio_summary_df = aggregate_results(
        jam_ratio_raw_df,
        group_columns=[
            "selector",
            "budget",
            "jam_ratio",
        ],
    )

    comparison_summary_df = aggregate_results(
        comparison_raw_df,
        group_columns=[
            "strategy",
            "budget",
            "jam_ratio",
        ],
    )

    baseline_summary_df.to_csv(
        args.output / "baseline_metrics_summary.csv",
        index=False,
    )

    budget_summary_df.to_csv(
        args.output / "clsa_budget_results.csv",
        index=False,
    )

    jam_ratio_summary_df.to_csv(
        args.output / "clsa_jam_ratio_results.csv",
        index=False,
    )

    comparison_summary_df.to_csv(
        args.output / "clsa_targeting_comparison.csv",
        index=False,
    )

    if first_seed_rankings is not None:
        first_seed_rankings.to_csv(
            args.output / "channel_rankings_first_seed.csv",
            index=False,
        )

        rank_for_selector(
            first_seed_rankings,
            args.main_selector,
        ).head(50).to_csv(
            args.output
            / f"top50_targets_{args.main_selector}.csv",
            index=False,
        )

    print("\n======================================================")
    print("BASELINE SUMMARY")
    print("======================================================")
    print(
        baseline_summary_df.to_string(index=False)
    )

    print("\n======================================================")
    print("BUDGET SUMMARY")
    print("======================================================")
    print(
        budget_summary_df[
            [
                "selector",
                "budget",
                "jam_ratio",
                "success_ratio_mean",
                "throughput_drop_percent_mean",
                "throughput_drop_percent_std",
                "CAF_mean",
                "average_successful_path_length_mean",
                "average_successful_fee_mean",
                "depletion_events_mean",
                "runs",
            ]
        ].to_string(index=False)
    )

    print("\n======================================================")
    print("SELECTOR COMPARISON")
    print("======================================================")
    print(
        comparison_summary_df[
            [
                "strategy",
                "budget",
                "jam_ratio",
                "success_ratio_mean",
                "throughput_drop_percent_mean",
                "throughput_drop_percent_std",
                "CAF_mean",
                "runs",
            ]
        ].to_string(index=False)
    )

    print("\n======================================================")
    print("DONE")
    print("======================================================")
    print(f"Results saved in:\n{args.output.resolve()}")

    print("\nGenerate graphs using:")
    print(
        "python .\\CLSA\\plot_clsa_results.py "
        "--input .\\output\\clsa_run_improved"
    )


if __name__ == "__main__":
    main()
