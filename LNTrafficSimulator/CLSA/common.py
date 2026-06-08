from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_EDGE_COLUMNS = {
    "src",
    "trg",
    "capacity",
    "disabled",
    "fee_base_msat",
    "fee_rate_milli_msat",
}


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def json_dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert CSV values such as True, False, 1, 0, 'true', and 'false'
    into a reliable Boolean pandas Series.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    lowered = series.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def normalize_node_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["src"] = result["src"].astype(str)
    result["trg"] = result["trg"].astype(str)
    return result


def read_snapshot(edges_file: str | Path, snapshot_id: int) -> pd.DataFrame:
    """
    Read one complete snapshot from ln_edges.csv.

    All other snapshots are ignored by the experiment. This prevents rows
    from different points in time from being mixed together.
    """
    edges_file = Path(edges_file)
    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")

    edges = pd.read_csv(edges_file, low_memory=False)

    if "snapshot_id" not in edges.columns:
        raise ValueError(
            "The CSV does not contain a 'snapshot_id' column. "
            "Use the preprocessed ln_edges.csv file."
        )

    missing = REQUIRED_EDGE_COLUMNS.difference(edges.columns)
    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    snapshot = edges[edges["snapshot_id"] == snapshot_id].copy()
    if snapshot.empty:
        available = sorted(edges["snapshot_id"].dropna().unique().tolist())
        raise ValueError(
            f"No rows were found for snapshot_id={snapshot_id}. "
            f"Available IDs include: {available[:30]}"
        )

    if "Unnamed: 0" in snapshot.columns:
        snapshot = snapshot.drop(columns=["Unnamed: 0"])

    snapshot = normalize_node_ids(snapshot)
    snapshot["disabled"] = parse_bool_series(snapshot["disabled"])
    snapshot["capacity"] = pd.to_numeric(
        snapshot["capacity"], errors="coerce"
    ).fillna(0.0)
    snapshot["fee_base_msat"] = pd.to_numeric(
        snapshot["fee_base_msat"], errors="coerce"
    ).fillna(0.0)
    snapshot["fee_rate_milli_msat"] = pd.to_numeric(
        snapshot["fee_rate_milli_msat"], errors="coerce"
    ).fillna(0.0)

    return snapshot.reset_index(drop=True)


def calculate_total_fee(edges: pd.DataFrame, amount_sat: int) -> pd.Series:
    """
    Match the fee calculation used by LNTrafficSimulator:
      base fee in millisatoshis -> satoshis
      proportional fee rate -> satoshis for the payment amount
    """
    return (
        edges["fee_base_msat"] / 1000.0
        + amount_sat * edges["fee_rate_milli_msat"] / 10.0**6
    )


def prepare_policy_edges(
    snapshot: pd.DataFrame,
    amount_sat: int,
    drop_disabled: bool = True,
    drop_low_cap: bool = True,
) -> pd.DataFrame:
    """
    Prepare the directed routing graph in the same general form used by the
    simulator: filter policies, calculate fees, and aggregate parallel
    directed edges between the same node pair.
    """
    edges = normalize_node_ids(snapshot)

    if drop_low_cap:
        edges = edges[edges["capacity"] >= amount_sat].copy()

    if drop_disabled:
        edges = edges[~edges["disabled"]].copy()

    if edges.empty:
        raise ValueError(
            "No usable channel policies remain after filtering. "
            "Check the snapshot, amount, and disabled-channel settings."
        )

    edges["total_fee"] = calculate_total_fee(edges, amount_sat)

    prepared = (
        edges.groupby(["src", "trg"], as_index=False)
        .agg(
            capacity=("capacity", "sum"),
            total_fee=("total_fee", "mean"),
        )
        .reset_index(drop=True)
    )

    return prepared


def canonical_pair(src: str, trg: str) -> tuple[str, str]:
    return (src, trg) if src <= trg else (trg, src)


def add_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pairs = [canonical_pair(str(s), str(t)) for s, t in zip(result["src"], result["trg"])]
    result["node_a"] = [pair[0] for pair in pairs]
    result["node_b"] = [pair[1] for pair in pairs]
    return result


def calculate_ebc_rankings(
    snapshot: pd.DataFrame,
    amount_sat: int,
    metric: str = "hops",
    k_sources: int = 200,
    seed: int = 42,
    drop_disabled: bool = True,
    drop_low_cap: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Calculate directed and physical-pair EBC rankings.

    metric='hops':
        Every usable edge has equal cost.

    metric='fee':
        EBC is calculated over the lower-fee routes. NetworkX requires
        strictly positive weights, so zero-fee policies receive a tiny
        positive epsilon.

    k_sources=0:
        Exact EBC calculation over all source nodes.

    k_sources>0:
        Approximate EBC using that many sampled source nodes.
    """
    if metric not in {"hops", "fee"}:
        raise ValueError("metric must be either 'hops' or 'fee'")

    prepared = prepare_policy_edges(
        snapshot=snapshot,
        amount_sat=amount_sat,
        drop_disabled=drop_disabled,
        drop_low_cap=drop_low_cap,
    )

    graph = nx.from_pandas_edgelist(
        prepared,
        source="src",
        target="trg",
        edge_attr=["capacity", "total_fee"],
        create_using=nx.DiGraph(),
    )

    weight_column: str | None = None
    if metric == "fee":
        for src, trg, data in graph.edges(data=True):
            data["ebc_weight"] = max(float(data["total_fee"]), 1e-12)
        weight_column = "ebc_weight"

    requested_k = int(k_sources)
    if requested_k <= 0:
        effective_k = None
    else:
        effective_k = min(requested_k, graph.number_of_nodes())

    started = time.perf_counter()
    centrality = nx.edge_betweenness_centrality(
        graph,
        k=effective_k,
        normalized=True,
        weight=weight_column,
        seed=seed,
    )
    elapsed = time.perf_counter() - started

    directed = pd.DataFrame(
        [
            {
                "src": src,
                "trg": trg,
                "edge_betweenness": float(score),
            }
            for (src, trg), score in centrality.items()
        ]
    )

    directed = directed.merge(
        prepared,
        on=["src", "trg"],
        how="left",
    )

    directed = (
        directed.sort_values(
            ["edge_betweenness", "src", "trg"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    directed.insert(0, "directed_rank", range(1, len(directed) + 1))

    with_pairs = add_pair_columns(directed)
    pairs = (
        with_pairs.groupby(["node_a", "node_b"], as_index=False)
        .agg(
            pair_edge_betweenness=("edge_betweenness", "sum"),
            directions_present=("edge_betweenness", "count"),
            aggregated_capacity=("capacity", "max"),
        )
        .sort_values(
            ["pair_edge_betweenness", "node_a", "node_b"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    pairs.insert(0, "pair_rank", range(1, len(pairs) + 1))

    metadata = {
        "metric": metric,
        "requested_k_sources": requested_k,
        "effective_k_sources": (
            "all_nodes_exact" if effective_k is None else effective_k
        ),
        "seed": seed,
        "amount_sat": amount_sat,
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "physical_node_pairs": len(pairs),
        "calculation_seconds": elapsed,
    }

    return directed, pairs, metadata


def apply_pair_shock(
    snapshot: pd.DataFrame,
    pair_ranking: pd.DataFrame,
    top_k: int,
    remove_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reduce capacity for all raw rows belonging to the selected Top-K
    physical node pairs.

    If several parallel public channels connect the same two nodes, all of
    those raw rows are reduced. This matches the EBC aggregation level.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not 0 <= remove_pct <= 100:
        raise ValueError("remove_pct must be between 0 and 100")

    if top_k > len(pair_ranking):
        raise ValueError(
            f"Requested top_k={top_k}, but the EBC file contains only "
            f"{len(pair_ranking)} physical channel pairs."
        )

    selected = pair_ranking.head(top_k).copy()
    selected_pairs = set(
        zip(selected["node_a"].astype(str), selected["node_b"].astype(str))
    )

    attacked = add_pair_columns(snapshot)
    attacked["is_selected_pair"] = [
        (a, b) in selected_pairs
        for a, b in zip(attacked["node_a"], attacked["node_b"])
    ]

    attacked["original_capacity"] = attacked["capacity"].astype(float)

    multiplier = 1.0 - (float(remove_pct) / 100.0)
    attacked.loc[attacked["is_selected_pair"], "capacity"] = np.floor(
        attacked.loc[attacked["is_selected_pair"], "capacity"] * multiplier
    )

    changed = attacked[attacked["is_selected_pair"]].copy()
    changed["new_capacity"] = changed["capacity"]
    changed["removed_capacity"] = (
        changed["original_capacity"] - changed["new_capacity"]
    )
    changed["remove_pct"] = float(remove_pct)

    selected = selected.copy()
    selected["top_k"] = int(top_k)
    selected["remove_pct"] = float(remove_pct)

    raw_columns_to_drop = [
        "node_a",
        "node_b",
        "is_selected_pair",
        "original_capacity",
    ]
    attacked_snapshot = attacked.drop(
        columns=[
            column
            for column in raw_columns_to_drop
            if column in attacked.columns
        ]
    )

    log_columns = [
        column
        for column in [
            "snapshot_id",
            "channel_id",
            "src",
            "trg",
            "node_a",
            "node_b",
            "original_capacity",
            "new_capacity",
            "removed_capacity",
            "remove_pct",
        ]
        if column in changed.columns
    ]

    return (
        attacked_snapshot.reset_index(drop=True),
        selected.reset_index(drop=True),
        changed[log_columns].reset_index(drop=True),
    )


def load_providers(meta_file: str | Path | None) -> list[str]:
    if meta_file is None:
        return []

    meta_file = Path(meta_file)
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    metadata = pd.read_csv(meta_file, low_memory=False)
    if "pub_key" not in metadata.columns:
        raise ValueError(
            "Metadata file does not contain the required 'pub_key' column."
        )

    return metadata["pub_key"].dropna().astype(str).drop_duplicates().tolist()


def generate_fixed_workload(
    prepared_edges: pd.DataFrame,
    amount_sat: int,
    count: int,
    epsilon: float,
    providers: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """
    Generate a fixed workload following the original simulator's broad
    endpoint-sampling approach:
      - sources are sampled from active nodes
      - epsilon fraction of targets are sampled from active providers
      - provider sampling is weighted by graph degree
      - self-payments are removed
    """
    graph = nx.from_pandas_edgelist(
        prepared_edges,
        source="src",
        target="trg",
        edge_attr=["capacity"],
        create_using=nx.DiGraph(),
    )

    nodes = sorted(str(node) for node in graph.nodes())
    if not nodes:
        raise ValueError("No active nodes are available for workload generation.")

    rng = np.random.default_rng(seed)
    sources = rng.choice(nodes, size=count, replace=True)

    active_providers = sorted(set(str(p) for p in providers).intersection(nodes))
    num_provider_targets = int(float(epsilon) * count)

    targets: list[str]
    if num_provider_targets > 0 and active_providers:
        provider_degrees = np.array(
            [float(graph.degree(provider)) for provider in active_providers],
            dtype=float,
        )
        if provider_degrees.sum() <= 0:
            provider_probabilities = None
        else:
            provider_probabilities = provider_degrees / provider_degrees.sum()

        provider_targets = rng.choice(
            active_providers,
            size=num_provider_targets,
            replace=True,
            p=provider_probabilities,
        )
        random_targets = rng.choice(
            nodes,
            size=count - num_provider_targets,
            replace=True,
        )
        targets_array = np.concatenate([provider_targets, random_targets])
        rng.shuffle(targets_array)
        targets = targets_array.tolist()
    else:
        targets = rng.choice(nodes, size=count, replace=True).tolist()

    workload = pd.DataFrame(
        {
            "transaction_id": range(count),
            "source": sources.astype(str),
            "target": pd.Series(targets, dtype=str),
            "amount_SAT": int(amount_sat),
        }
    )

    workload = workload[workload["source"] != workload["target"]].copy()
    workload = workload.reset_index(drop=True)
    return workload


def read_workload(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workload file not found: {path}")

    workload = pd.read_csv(path, low_memory=False)
    required = {"transaction_id", "source", "target", "amount_SAT"}
    missing = required.difference(workload.columns)
    if missing:
        raise ValueError(
            "Workload is missing required columns: "
            + ", ".join(sorted(missing))
        )

    workload["source"] = workload["source"].astype(str)
    workload["target"] = workload["target"].astype(str)
    workload["amount_SAT"] = pd.to_numeric(
        workload["amount_SAT"], errors="raise"
    ).astype(int)

    return workload


@dataclass
class SimulationResult:
    metrics: pd.DataFrame
    transaction_log: pd.DataFrame
    edge_usage: pd.DataFrame
    depletion_events: pd.DataFrame
    router_incomes: pd.DataFrame
    lengths_distribution: pd.DataFrame
    initial_liquidity: pd.DataFrame
    final_liquidity: pd.DataFrame


class SequentialLiquiditySimulator:
    """
    Sequential Lightning Network routing simulation.

    The implementation follows the original LNTrafficSimulator model:
      - directed policies are aggregated by source and target
      - directional liquidity is initialized by splitting pair capacity
      - payments route over lowest-cost currently usable paths
      - successful forwarding decreases forward liquidity
      - successful forwarding replenishes reverse liquidity
      - edges are removed or restored as their usable balance crosses the
        payment-amount threshold
      - the recipient does not charge a routing fee
    """

    def __init__(
        self,
        prepared_edges: pd.DataFrame,
        workload: pd.DataFrame,
        amount_sat: int,
        liquidity_seed: int,
        route_metric: str = "fee",
        with_depletion: bool = True,
    ) -> None:
        if route_metric not in {"fee", "hops"}:
            raise ValueError("route_metric must be either 'fee' or 'hops'")

        self.prepared_edges = prepared_edges.copy()
        self.workload = workload.copy()
        self.amount_sat = int(amount_sat)
        self.route_metric = route_metric
        self.with_depletion = bool(with_depletion)
        self.rng = np.random.default_rng(liquidity_seed)

        self.target_nodes = set(self.workload["target"].astype(str))
        self.policy_map: dict[tuple[str, str], dict] = {}
        self.balances: dict[tuple[str, str], float] = {}
        self.initial_balances: dict[tuple[str, str], float] = {}
        self.graph = nx.DiGraph()

        self._initialize_state()

    def _initialize_state(self) -> None:
        for row in self.prepared_edges.itertuples(index=False):
            key = (str(row.src), str(row.trg))
            self.policy_map[key] = {
                "total_capacity": float(row.capacity),
                "total_fee": float(row.total_fee),
            }

        pair_keys = sorted(
            {
                canonical_pair(src, trg)
                for src, trg in self.policy_map
            }
        )

        for node_a, node_b in pair_keys:
            forward = (node_a, node_b)
            reverse = (node_b, node_a)
            has_forward = forward in self.policy_map
            has_reverse = reverse in self.policy_map

            if has_forward and has_reverse:
                pair_capacity = max(
                    self.policy_map[forward]["total_capacity"],
                    self.policy_map[reverse]["total_capacity"],
                )
                split = float(self.rng.random())
                self.balances[forward] = pair_capacity * split
                self.balances[reverse] = pair_capacity * (1.0 - split)
            elif has_forward:
                self.balances[forward] = self.policy_map[forward]["total_capacity"]
            elif has_reverse:
                self.balances[reverse] = self.policy_map[reverse]["total_capacity"]

        self.initial_balances = dict(self.balances)

        for edge in sorted(self.policy_map):
            if self.balances.get(edge, 0.0) >= self.amount_sat:
                self._activate_edge(*edge)

    def _weight(self, src: str, trg: str) -> float:
        if self.route_metric == "hops":
            return 1.0
        return float(self.policy_map[(src, trg)]["total_fee"])

    def _pseudo_target(self, node: str) -> str:
        return f"{node}_trg"

    def _activate_edge(self, src: str, trg: str) -> None:
        if (src, trg) not in self.policy_map:
            return

        self.graph.add_edge(
            src,
            trg,
            weight=self._weight(src, trg),
            total_fee=float(self.policy_map[(src, trg)]["total_fee"]),
        )

        if trg in self.target_nodes:
            self.graph.add_edge(
                src,
                self._pseudo_target(trg),
                weight=0.0,
                total_fee=0.0,
            )

    def _deactivate_edge(self, src: str, trg: str) -> None:
        if self.graph.has_edge(src, trg):
            self.graph.remove_edge(src, trg)

        pseudo_target = self._pseudo_target(trg)
        if self.graph.has_edge(src, pseudo_target):
            self.graph.remove_edge(src, pseudo_target)

    def _forward(
        self,
        src: str,
        trg: str,
        transaction_id: int,
        depletion_records: list[dict],
    ) -> None:
        edge = (src, trg)
        current = self.balances[edge]

        if current < self.amount_sat:
            raise RuntimeError(
                f"Insufficient simulated balance on edge {src}->{trg}: "
                f"{current} < {self.amount_sat}"
            )

        new_balance = current - self.amount_sat
        self.balances[edge] = new_balance

        if (
            self.with_depletion
            and current >= self.amount_sat
            and new_balance < self.amount_sat
        ):
            self._deactivate_edge(src, trg)
            depletion_records.append(
                {
                    "transaction_id": transaction_id,
                    "src": src,
                    "trg": trg,
                    "balance_before": current,
                    "balance_after": new_balance,
                }
            )

    def _backward(self, src: str, trg: str) -> None:
        edge = (src, trg)
        if edge not in self.balances:
            return

        current = self.balances[edge]
        new_balance = current + self.amount_sat
        self.balances[edge] = new_balance

        if (
            self.with_depletion
            and current < self.amount_sat
            and new_balance >= self.amount_sat
        ):
            self._activate_edge(src, trg)

    def _liquidity_frame(self, balances: dict[tuple[str, str], float]) -> pd.DataFrame:
        records = []
        for (src, trg), balance in sorted(balances.items()):
            policy = self.policy_map[(src, trg)]
            records.append(
                {
                    "src": src,
                    "trg": trg,
                    "available_liquidity": float(balance),
                    "public_capacity": float(policy["total_capacity"]),
                    "total_fee": float(policy["total_fee"]),
                    "is_usable_for_payment": bool(balance >= self.amount_sat),
                }
            )
        return pd.DataFrame(records)

    def run(self) -> SimulationResult:
        started = time.perf_counter()

        transaction_records: list[dict] = []
        depletion_records: list[dict] = []
        edge_usage_counts: dict[tuple[str, str], int] = {}
        router_income: dict[str, float] = {}
        router_counts: dict[str, int] = {}

        for tx in self.workload.itertuples(index=False):
            transaction_id = int(tx.transaction_id)
            source = str(tx.source)
            target = str(tx.target)
            amount = int(tx.amount_SAT)

            if amount != self.amount_sat:
                raise ValueError(
                    "All workload transaction amounts must equal the amount "
                    "used to initialize the simulator."
                )

            destination = self._pseudo_target(target)
            success = False
            route_nodes: list[str] = []
            routing_fee = math.nan
            failure_reason = ""

            if source not in self.graph:
                failure_reason = "source_not_in_active_graph"
            elif destination not in self.graph:
                failure_reason = "target_not_in_active_graph"
            else:
                try:
                    pseudo_path = nx.shortest_path(
                        self.graph,
                        source=source,
                        target=destination,
                        weight="weight",
                    )

                    route_nodes = [
                        target if node == destination else str(node)
                        for node in pseudo_path
                    ]

                    actual_edges: list[tuple[str, str]] = []
                    for index in range(len(pseudo_path) - 1):
                        src = str(pseudo_path[index])
                        next_node = str(pseudo_path[index + 1])

                        if next_node == destination:
                            trg = target
                        else:
                            trg = next_node

                        actual_edges.append((src, trg))

                    # Recipient does not earn a routing fee.
                    intermediate_edges = actual_edges[:-1]
                    routing_fee = sum(
                        float(self.policy_map[edge]["total_fee"])
                        for edge in intermediate_edges
                    )

                    for src, trg in actual_edges:
                        self._forward(
                            src=src,
                            trg=trg,
                            transaction_id=transaction_id,
                            depletion_records=depletion_records,
                        )
                        self._backward(src=trg, trg=src)

                        edge = (src, trg)
                        edge_usage_counts[edge] = edge_usage_counts.get(edge, 0) + 1

                    for index, edge in enumerate(intermediate_edges):
                        router = edge[1]
                        fee = float(self.policy_map[edge]["total_fee"])
                        router_income[router] = router_income.get(router, 0.0) + fee
                        router_counts[router] = router_counts.get(router, 0) + 1

                    success = True

                except nx.NetworkXNoPath:
                    failure_reason = "no_liquidity_feasible_path"

            transaction_records.append(
                {
                    "transaction_id": transaction_id,
                    "source": source,
                    "target": target,
                    "amount_SAT": amount,
                    "success": success,
                    "path_length": (
                        len(route_nodes) - 1 if success else math.nan
                    ),
                    "routing_fee": routing_fee,
                    "route": "|".join(route_nodes),
                    "failure_reason": failure_reason,
                }
            )

        elapsed = time.perf_counter() - started

        transaction_log = pd.DataFrame(transaction_records)
        depletion_events = pd.DataFrame(depletion_records)

        edge_usage = pd.DataFrame(
            [
                {
                    "src": src,
                    "trg": trg,
                    "usage_count": count,
                    "forwarded_satoshis": count * self.amount_sat,
                }
                for (src, trg), count in edge_usage_counts.items()
            ]
        )
        if not edge_usage.empty:
            edge_usage = edge_usage.sort_values(
                ["usage_count", "src", "trg"],
                ascending=[False, True, True],
            ).reset_index(drop=True)

        router_incomes = pd.DataFrame(
            [
                {
                    "node": node,
                    "fee": income,
                    "num_trans": router_counts.get(node, 0),
                }
                for node, income in router_income.items()
            ]
        )
        if not router_incomes.empty:
            router_incomes = router_incomes.sort_values(
                ["fee", "node"],
                ascending=[False, True],
            ).reset_index(drop=True)

        successful = transaction_log[transaction_log["success"]].copy()
        lengths_distribution = (
            successful["path_length"]
            .value_counts()
            .sort_index()
            .rename_axis("path_length")
            .reset_index(name="successful_transactions")
        )
        if not lengths_distribution.empty:
            lengths_distribution["share_of_successful_routes_pct"] = (
                100.0
                * lengths_distribution["successful_transactions"]
                / lengths_distribution["successful_transactions"].sum()
            )

        total_transactions = len(transaction_log)
        successful_transactions = int(transaction_log["success"].sum())
        failed_transactions = total_transactions - successful_transactions

        if depletion_events.empty:
            nodes_with_depletion = 0
        else:
            nodes_with_depletion = int(
                depletion_events["trg"].astype(str).nunique()
            )

        metric_values = {
            "total_transactions": total_transactions,
            "successful_transactions": successful_transactions,
            "failed_transactions": failed_transactions,
            "success_rate_pct": (
                100.0 * successful_transactions / total_transactions
                if total_transactions
                else math.nan
            ),
            "failure_rate_pct": (
                100.0 * failed_transactions / total_transactions
                if total_transactions
                else math.nan
            ),
            "average_path_length": (
                float(successful["path_length"].mean())
                if not successful.empty
                else math.nan
            ),
            "average_routing_fee": (
                float(successful["routing_fee"].mean())
                if not successful.empty
                else math.nan
            ),
            "total_depletion_events": len(depletion_events),
            "nodes_with_depletion_events": nodes_with_depletion,
            "initial_active_directed_edges": int(
                sum(
                    balance >= self.amount_sat
                    for balance in self.initial_balances.values()
                )
            ),
            "final_active_directed_edges": int(
                sum(
                    balance >= self.amount_sat
                    for balance in self.balances.values()
                )
            ),
            "runtime_seconds": elapsed,
        }

        metrics = pd.DataFrame(
            [
                {"metric": metric, "value": value}
                for metric, value in metric_values.items()
            ]
        )

        return SimulationResult(
            metrics=metrics,
            transaction_log=transaction_log,
            edge_usage=edge_usage,
            depletion_events=depletion_events,
            router_incomes=router_incomes,
            lengths_distribution=lengths_distribution,
            initial_liquidity=self._liquidity_frame(self.initial_balances),
            final_liquidity=self._liquidity_frame(self.balances),
        )


def save_simulation_result(
    result: SimulationResult,
    output_dir: str | Path,
    config: dict,
) -> None:
    output = ensure_dir(output_dir)

    result.metrics.to_csv(output / "metrics.csv", index=False)
    result.transaction_log.to_csv(output / "transaction_log.csv", index=False)
    result.edge_usage.to_csv(output / "edge_usage.csv", index=False)
    result.depletion_events.to_csv(output / "depletion_events.csv", index=False)
    result.router_incomes.to_csv(output / "router_incomes.csv", index=False)
    result.lengths_distribution.to_csv(
        output / "lengths_distrib.csv", index=False
    )
    result.initial_liquidity.to_csv(
        output / "initial_liquidity.csv", index=False
    )
    result.final_liquidity.to_csv(
        output / "final_liquidity.csv", index=False
    )
    json_dump(output / "experiment_config.json", config)


def metrics_to_dict(metrics: pd.DataFrame) -> dict[str, float]:
    return dict(zip(metrics["metric"], metrics["value"]))


def calculate_pair_capacity(snapshot: pd.DataFrame) -> float:
    """
    Estimate snapshot physical capacity without double-counting directions.
    Uses channel_id when available; otherwise uses maximum directional
    capacity per unordered node pair.
    """
    frame = add_pair_columns(snapshot)

    if "channel_id" in frame.columns:
        channel_level = (
            frame.groupby(["node_a", "node_b", "channel_id"], as_index=False)
            .agg(capacity=("capacity", "max"))
        )
    else:
        channel_level = (
            frame.groupby(["node_a", "node_b"], as_index=False)
            .agg(capacity=("capacity", "max"))
        )

    return float(channel_level["capacity"].sum())
