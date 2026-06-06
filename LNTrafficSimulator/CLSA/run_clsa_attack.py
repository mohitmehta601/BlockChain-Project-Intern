"""
Cascading Liquidity Shock Attack (CLSA)
for the LNTrafficSimulator project.

Place this file inside:
    LNTrafficSimulator/CLSA/run_clsa_attack.py

Run it from the LNTrafficSimulator root directory:
    python .\\CLSA\\run_clsa_attack.py

For a large graph, use approximate edge betweenness:
    python .\\CLSA\\run_clsa_attack.py --approx-k 500
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
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


# ---------------------------------------------------------
# General helper functions
# ---------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """
    Set seeds so that baseline and attack runs are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_params(params_path: Path) -> dict:
    """
    Load the existing LNTrafficSimulator configuration.
    """
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
    """
    Load merchant/provider nodes from 1ml_meta_data.csv.
    """
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

    providers = (
        metadata["pub_key"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return providers


def load_directed_edges(network_json_path: Path) -> pd.DataFrame:
    """
    Convert the raw LN JSON snapshot into the simulator's directed-edge format.
    """
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


def safe_depletion_count(total_depletions) -> int:
    """
    Convert simulator depletion information into a single count.
    """
    if total_depletions is None:
        return 0

    if isinstance(total_depletions, dict):
        return int(sum(total_depletions.values()))

    if isinstance(total_depletions, (list, tuple, set)):
        return len(total_depletions)

    try:
        return int(total_depletions)
    except (TypeError, ValueError):
        return 0


def save_workload(workload, output_path: Path) -> None:
    """
    Save the sampled source-target workload when possible.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(workload, pd.DataFrame):
        workload.to_csv(output_path, index=False)
        return

    try:
        pd.DataFrame(workload).to_csv(output_path, index=False)
    except Exception:
        print("WARNING: Transaction workload could not be exported as CSV.")


# ---------------------------------------------------------
# CLSA target selection
# ---------------------------------------------------------

def build_undirected_channel_graph(
    directed_edges: pd.DataFrame,
) -> nx.Graph:
    """
    Build an undirected graph for target ranking.

    A Lightning channel is represented by directed edges in the simulator.
    For attack targeting, both directions are treated as one bidirectional
    channel between nodes u and v.
    """
    graph = nx.Graph()

    for row in directed_edges.itertuples(index=False):
        src = str(row.src)
        trg = str(row.trg)
        capacity = float(row.capacity)

        if src == trg:
            continue

        if capacity <= 0:
            continue

        if graph.has_edge(src, trg):
            graph[src][trg]["capacity"] += capacity
        else:
            graph.add_edge(src, trg, capacity=capacity)

    return graph


def rank_channels_by_edge_betweenness(
    directed_edges: pd.DataFrame,
    seed: int,
    approx_k: int,
) -> pd.DataFrame:
    """
    Rank channels by edge betweenness centrality.

    High-betweenness channels appear on many shortest paths.
    Jamming them should affect more payments than randomly selected channels.

    approx_k:
        0    -> exact calculation
        > 0  -> approximate calculation using sampled source nodes
    """
    graph = build_undirected_channel_graph(directed_edges)

    print("\n======================================================")
    print("EDGE BETWEENNESS CENTRALITY")
    print("======================================================")
    print(f"Nodes:    {graph.number_of_nodes()}")
    print(f"Channels: {graph.number_of_edges()}")

    if approx_k > 0:
        sampled_nodes = min(approx_k, graph.number_of_nodes())

        print(
            "Mode: approximate calculation "
            f"using {sampled_nodes} sampled source nodes"
        )

        scores = nx.edge_betweenness_centrality(
            graph,
            k=sampled_nodes,
            normalized=True,
            weight=None,
            seed=seed,
        )
    else:
        print("Mode: exact calculation")
        print("NOTE: This may take time for a large network.")

        scores = nx.edge_betweenness_centrality(
            graph,
            normalized=True,
            weight=None,
        )

    records = []

    for (node_a, node_b), score in scores.items():
        u, v = sorted((str(node_a), str(node_b)))

        records.append(
            {
                "u": u,
                "v": v,
                "edge_betweenness": float(score),
            }
        )

    ranked_channels = (
        pd.DataFrame(records)
        .sort_values("edge_betweenness", ascending=False)
        .reset_index(drop=True)
    )

    ranked_channels.insert(
        0,
        "rank",
        range(1, len(ranked_channels) + 1),
    )

    return ranked_channels


def select_top_channels(
    ranked_channels: pd.DataFrame,
    number_of_channels: int,
) -> list[tuple[str, str]]:
    """
    Select the top-k highest-centrality channels.
    """
    number_of_channels = min(
        max(1, int(number_of_channels)),
        len(ranked_channels),
    )

    return list(
        ranked_channels
        .head(number_of_channels)[["u", "v"]]
        .itertuples(index=False, name=None)
    )


def save_selected_targets(
    selected_channels: Iterable[tuple[str, str]],
    ranked_channels: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the channels selected for an attack scenario.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets_df = pd.DataFrame(
        list(selected_channels),
        columns=["u", "v"],
    )

    targets_df = targets_df.merge(
        ranked_channels,
        on=["u", "v"],
        how="left",
    )

    targets_df.to_csv(output_path, index=False)


# ---------------------------------------------------------
# CLSA injection
# ---------------------------------------------------------

def inject_clsa(
    directed_edges: pd.DataFrame,
    target_channels: Iterable[tuple[str, str]],
    jam_ratio: float,
) -> tuple[pd.DataFrame, float]:
    """
    Reduce the capacity of selected channels.

    Example:
        jam_ratio = 0.8

        remaining capacity
        = original capacity * (1 - 0.8)
        = original capacity * 0.2

    Therefore, 80% of the capacity is treated as locked.

    Both directions of each selected channel are reduced.
    """
    if not 0.0 <= jam_ratio <= 1.0:
        raise ValueError("jam_ratio must be between 0.0 and 1.0.")

    attacked_edges = directed_edges.copy()

    target_set = {
        tuple(sorted((str(u), str(v))))
        for u, v in target_channels
    }

    row_pairs = [
        tuple(sorted((str(src), str(trg))))
        for src, trg in zip(
            attacked_edges["src"],
            attacked_edges["trg"],
        )
    ]

    target_mask = pd.Series(
        [pair in target_set for pair in row_pairs],
        index=attacked_edges.index,
    )

    capacity_before = float(
        attacked_edges.loc[target_mask, "capacity"].sum()
    )

    attacked_edges.loc[target_mask, "capacity"] = np.floor(
        attacked_edges.loc[target_mask, "capacity"].astype(float)
        * (1.0 - jam_ratio)
    )

    capacity_after = float(
        attacked_edges.loc[target_mask, "capacity"].sum()
    )

    locked_capacity = capacity_before - capacity_after

    return attacked_edges, locked_capacity


# ---------------------------------------------------------
# Simulator execution
# ---------------------------------------------------------

def calculate_simulation_metrics(
    shortest_paths: pd.DataFrame,
    total_depletions,
) -> dict:
    """
    Calculate network-level metrics for one simulation.
    """
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
            shortest_paths
            .loc[success_mask, "original_cost"]
            .mean()
        )

    average_path_length = float("nan")

    if successful_transactions > 0:
        average_path_length = float(
            shortest_paths
            .loc[success_mask, "length"]
            .mean()
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
    """
    Run one LNTrafficSimulator experiment.

    The same fixed workload is reused for baseline and attacked scenarios.
    """
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
    """
    Inject CLSA and run one attacked simulation.
    """
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

    lost_successful_transactions = max(
        0,
        int(baseline_metrics["successful_transactions"])
        - int(attack_metrics["successful_transactions"]),
    )

    throughput_drop = (
        float(baseline_metrics["success_ratio"])
        - float(attack_metrics["success_ratio"])
    )

    caf = (
        lost_successful_transactions / len(target_channels)
        if target_channels
        else 0.0
    )

    attack_metrics.update(
        {
            "channels_jammed": len(target_channels),
            "jam_ratio": jam_ratio,
            "locked_capacity_sat": locked_capacity,
            "lost_successful_transactions": lost_successful_transactions,
            "throughput_drop": throughput_drop,
            "throughput_drop_percent": throughput_drop * 100.0,
            "CAF": caf,
        }
    )

    pd.DataFrame([attack_metrics]).to_csv(
        output_directory / "attack_metrics.csv",
        index=False,
    )

    return attack_metrics


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform CLSA liquidity-jamming experiments "
            "using LNTrafficSimulator."
        )
    )

    parser.add_argument(
        "--network",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "sample.json",
        help="Path to the raw Lightning Network JSON snapshot.",
    )

    parser.add_argument(
        "--meta",
        type=Path,
        default=PROJECT_ROOT / "sample_data" / "1ml_meta_data.csv",
        help="Path to the merchant metadata CSV file.",
    )

    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "params.json",
        help="Path to the existing simulator params.json file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "clsa_run",
        help="Directory where CLSA outputs will be stored.",
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50],
        help="Numbers of highest-centrality channels to jam.",
    )

    parser.add_argument(
        "--budget-jam-ratio",
        type=float,
        default=0.8,
        help="Jam ratio used for the attack-budget experiment.",
    )

    parser.add_argument(
        "--jam-ratios",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Jam ratios used for the sensitivity experiment.",
    )

    parser.add_argument(
        "--jam-sweep-k",
        type=int,
        default=10,
        help="Number of channels jammed during the jam-ratio sweep.",
    )

    parser.add_argument(
        "--random-k",
        type=int,
        default=10,
        help="Number of randomly selected channels per random trial.",
    )

    parser.add_argument(
        "--random-trials",
        type=int,
        default=5,
        help="Number of random-targeting trials.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260606,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=2,
        help="Maximum number of simulator threads.",
    )

    parser.add_argument(
        "--approx-k",
        type=int,
        default=0,
        help=(
            "Use approximate edge betweenness with this number "
            "of sampled nodes. Use 0 for exact calculation."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    args.output.mkdir(parents=True, exist_ok=True)

    print("\n======================================================")
    print("LOAD CLSA INPUTS")
    print("======================================================")

    params = load_params(args.params)

    directed_edges = load_directed_edges(args.network)

    providers = load_providers(args.meta)

    if not providers:
        params["epsilon"] = 0.0

    print("\n======================================================")
    print("SIMULATOR PARAMETERS")
    print("======================================================")
    print(json.dumps(params, indent=2))
    print(f"Merchant/provider nodes loaded: {len(providers)}")

    # -----------------------------------------------------
    # Step 1: Baseline simulation
    # -----------------------------------------------------

    print("\n======================================================")
    print("STEP 1: RUN BASELINE SIMULATION")
    print("======================================================")

    baseline_directory = args.output / "baseline"

    _, baseline_workload, baseline_metrics = run_simulation(
        directed_edges=directed_edges,
        providers=providers,
        params=params,
        output_directory=baseline_directory,
        seed=args.seed,
        max_threads=args.max_threads,
    )

    pd.DataFrame([baseline_metrics]).to_csv(
        args.output / "baseline_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Step 2: Rank attack targets
    # -----------------------------------------------------

    print("\n======================================================")
    print("STEP 2: RANK CHANNELS BY EDGE BETWEENNESS")
    print("======================================================")

    ranked_channels = rank_channels_by_edge_betweenness(
        directed_edges=directed_edges,
        seed=args.seed,
        approx_k=args.approx_k,
    )

    ranked_channels.to_csv(
        args.output / "edge_betweenness_all_channels.csv",
        index=False,
    )

    ranked_channels.head(50).to_csv(
        args.output / "top50_targets.csv",
        index=False,
    )

    print("\nTop 10 central channels:")
    print(ranked_channels.head(10).to_string(index=False))

    # -----------------------------------------------------
    # Step 3: Vary attack budget
    # -----------------------------------------------------

    print("\n======================================================")
    print("STEP 3: CLSA ATTACK-BUDGET EXPERIMENT")
    print("======================================================")

    budget_results = []

    for budget in args.budgets:
        targets = select_top_channels(
            ranked_channels,
            budget,
        )

        scenario_directory = (
            args.output
            / "budget_runs"
            / f"k_{len(targets)}"
        )

        scenario_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_selected_targets(
            targets,
            ranked_channels,
            scenario_directory / "targets.csv",
        )

        print(
            f"\nRunning attack: k={len(targets)}, "
            f"jam_ratio={args.budget_jam_ratio}"
        )

        scenario_metrics = run_attack_scenario(
            original_edges=directed_edges,
            providers=providers,
            params=params,
            baseline_workload=baseline_workload,
            baseline_metrics=baseline_metrics,
            target_channels=targets,
            jam_ratio=args.budget_jam_ratio,
            output_directory=scenario_directory,
            seed=args.seed,
            max_threads=args.max_threads,
        )

        budget_results.append(
            {
                "budget": len(targets),
                **scenario_metrics,
            }
        )

    budget_results_df = pd.DataFrame(budget_results)

    budget_results_df.to_csv(
        args.output / "clsa_budget_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Step 4: Vary jam ratio
    # -----------------------------------------------------

    print("\n======================================================")
    print("STEP 4: CLSA JAM-RATIO SENSITIVITY EXPERIMENT")
    print("======================================================")

    jam_ratio_targets = select_top_channels(
        ranked_channels,
        args.jam_sweep_k,
    )

    jam_ratio_results = []

    for jam_ratio in args.jam_ratios:
        scenario_directory = (
            args.output
            / "jam_ratio_runs"
            / f"jam_{jam_ratio:.2f}"
        )

        scenario_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_selected_targets(
            jam_ratio_targets,
            ranked_channels,
            scenario_directory / "targets.csv",
        )

        print(
            f"\nRunning attack: k={len(jam_ratio_targets)}, "
            f"jam_ratio={jam_ratio}"
        )

        scenario_metrics = run_attack_scenario(
            original_edges=directed_edges,
            providers=providers,
            params=params,
            baseline_workload=baseline_workload,
            baseline_metrics=baseline_metrics,
            target_channels=jam_ratio_targets,
            jam_ratio=jam_ratio,
            output_directory=scenario_directory,
            seed=args.seed,
            max_threads=args.max_threads,
        )

        jam_ratio_results.append(scenario_metrics)

    jam_ratio_results_df = pd.DataFrame(jam_ratio_results)

    jam_ratio_results_df.to_csv(
        args.output / "clsa_jam_ratio_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Step 5: Random vs centrality-based targeting
    # -----------------------------------------------------

    print("\n======================================================")
    print("STEP 5: RANDOM VS CENTRALITY TARGETING")
    print("======================================================")

    random_results = []

    random_k = min(
        max(1, args.random_k),
        len(ranked_channels),
    )

    for trial in range(1, args.random_trials + 1):
        trial_seed = args.seed + trial

        random_targets_df = ranked_channels.sample(
            n=random_k,
            replace=False,
            random_state=trial_seed,
        )

        random_targets = list(
            random_targets_df[["u", "v"]]
            .itertuples(index=False, name=None)
        )

        scenario_directory = (
            args.output
            / "random_runs"
            / f"trial_{trial}"
        )

        scenario_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_selected_targets(
            random_targets,
            ranked_channels,
            scenario_directory / "targets.csv",
        )

        print(
            f"\nRandom trial {trial}: "
            f"k={random_k}, "
            f"jam_ratio={args.budget_jam_ratio}"
        )

        scenario_metrics = run_attack_scenario(
            original_edges=directed_edges,
            providers=providers,
            params=params,
            baseline_workload=baseline_workload,
            baseline_metrics=baseline_metrics,
            target_channels=random_targets,
            jam_ratio=args.budget_jam_ratio,
            output_directory=scenario_directory,
            seed=args.seed,
            max_threads=args.max_threads,
        )

        random_results.append(
            {
                "trial": trial,
                **scenario_metrics,
            }
        )

    random_results_df = pd.DataFrame(random_results)

    random_results_df.to_csv(
        args.output / "clsa_random_targeting_results.csv",
        index=False,
    )

    centrality_targets = select_top_channels(
        ranked_channels,
        random_k,
    )

    centrality_directory = (
        args.output
        / "centrality_comparison_run"
    )

    centrality_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_selected_targets(
        centrality_targets,
        ranked_channels,
        centrality_directory / "targets.csv",
    )

    centrality_metrics = run_attack_scenario(
        original_edges=directed_edges,
        providers=providers,
        params=params,
        baseline_workload=baseline_workload,
        baseline_metrics=baseline_metrics,
        target_channels=centrality_targets,
        jam_ratio=args.budget_jam_ratio,
        output_directory=centrality_directory,
        seed=args.seed,
        max_threads=args.max_threads,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "strategy": "edge_betweenness",
                "channels_jammed": random_k,
                "jam_ratio": args.budget_jam_ratio,
                "mean_success_ratio": centrality_metrics[
                    "success_ratio"
                ],
                "mean_throughput_drop_percent": centrality_metrics[
                    "throughput_drop_percent"
                ],
                "mean_CAF": centrality_metrics["CAF"],
                "trials": 1,
            },
            {
                "strategy": "random",
                "channels_jammed": random_k,
                "jam_ratio": args.budget_jam_ratio,
                "mean_success_ratio": random_results_df[
                    "success_ratio"
                ].mean(),
                "mean_throughput_drop_percent": random_results_df[
                    "throughput_drop_percent"
                ].mean(),
                "mean_CAF": random_results_df["CAF"].mean(),
                "trials": len(random_results_df),
            },
        ]
    )

    comparison_df.to_csv(
        args.output / "clsa_targeting_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Step 6: Final summary table
    # -----------------------------------------------------

    baseline_summary = {
        "scenario": "baseline",
        "budget": 0,
        "channels_jammed": 0,
        "jam_ratio": 0.0,
        "successful_transactions": baseline_metrics[
            "successful_transactions"
        ],
        "failed_transactions": baseline_metrics[
            "failed_transactions"
        ],
        "success_ratio": baseline_metrics["success_ratio"],
        "throughput_drop_percent": 0.0,
        "CAF": 0.0,
    }

    attack_summary = budget_results_df[
        [
            "budget",
            "channels_jammed",
            "jam_ratio",
            "successful_transactions",
            "failed_transactions",
            "success_ratio",
            "throughput_drop_percent",
            "CAF",
        ]
    ].copy()

    attack_summary.insert(
        0,
        "scenario",
        "CLSA",
    )

    final_summary_df = pd.concat(
        [
            pd.DataFrame([baseline_summary]),
            attack_summary,
        ],
        ignore_index=True,
    )

    final_summary_df.to_csv(
        args.output / "clsa_final_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n======================================================")
    print("BASELINE METRICS")
    print("======================================================")
    print(
        pd.DataFrame([baseline_metrics])
        .to_string(index=False)
    )

    print("\n======================================================")
    print("CLSA ATTACK-BUDGET RESULTS")
    print("======================================================")

    display_columns = [
        "budget",
        "jam_ratio",
        "successful_transactions",
        "failed_transactions",
        "success_ratio",
        "throughput_drop_percent",
        "CAF",
        "locked_capacity_sat",
    ]

    print(
        budget_results_df[display_columns]
        .to_string(index=False)
    )

    print("\n======================================================")
    print("CENTRALITY VS RANDOM TARGETING")
    print("======================================================")
    print(comparison_df.to_string(index=False))

    print("\n======================================================")
    print("DONE")
    print("======================================================")
    print(f"Results saved in:\n{args.output.resolve()}")

    print("\nGenerate graphs using:")
    print(
        "python .\\CLSA\\plot_clsa_results.py "
        "--input .\\output\\clsa_run"
    )


if __name__ == "__main__":
    main()