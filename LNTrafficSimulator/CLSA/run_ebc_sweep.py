from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from common import (
    SequentialLiquiditySimulator,
    apply_pair_shock,
    calculate_ebc_rankings,
    calculate_pair_capacity,
    ensure_dir,
    generate_fixed_workload,
    json_dump,
    load_providers,
    metrics_to_dict,
    prepare_policy_edges,
    read_snapshot,
    save_simulation_result,
)


DEFAULT_TOP_K = [5, 10, 20, 30, 40, 50, 100, 200]
DEFAULT_REMOVE_PCT = [50, 60, 70, 80, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full EBC-targeted Cascading Liquidity Shock Attack "
            "experiment sweep."
        )
    )

    parser.add_argument("--edges", required=True, help="Path to ln_edges.csv")
    parser.add_argument("--snapshot-id", type=int, default=0)
    parser.add_argument("--params", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=DEFAULT_TOP_K,
        help="Top-K physical channel-pair values to attack.",
    )
    parser.add_argument(
        "--remove-pct",
        type=float,
        nargs="+",
        default=DEFAULT_REMOVE_PCT,
        help="Liquidity-removal percentages.",
    )

    parser.add_argument(
        "--ebc-metric",
        choices=["hops", "fee"],
        default="hops",
    )
    parser.add_argument(
        "--k-sources",
        type=int,
        default=200,
        help=(
            "Approximate EBC source count. "
            "Use 0 for exact all-node EBC."
        ),
    )
    parser.add_argument("--ebc-seed", type=int, default=42)
    parser.add_argument("--transaction-seed", type=int, default=42)
    parser.add_argument("--liquidity-seed", type=int, default=42)
    parser.add_argument(
        "--route-metric",
        choices=["fee", "hops"],
        default="fee",
    )
    parser.add_argument(
        "--save-shocked-edges",
        action="store_true",
        help=(
            "Save the complete shocked snapshot for every attack. "
            "This uses additional disk space."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not generate PNG comparison graphs.",
    )

    return parser.parse_args()


def relative_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return 100.0 * (current - baseline) / baseline


def flatten_metrics(
    experiment_type: str,
    experiment_id: str,
    metric_values: dict,
    top_k: int | None = None,
    remove_pct: float | None = None,
    baseline_values: dict | None = None,
    selected_pairs: int = 0,
    changed_raw_rows: int = 0,
    removed_capacity_sat: float = 0.0,
) -> dict:
    record = {
        "experiment_type": experiment_type,
        "experiment_id": experiment_id,
        "top_k": top_k,
        "remove_pct": remove_pct,
        "selected_channel_pairs": selected_pairs,
        "changed_raw_rows": changed_raw_rows,
        "removed_capacity_sat": removed_capacity_sat,
    }
    record.update(metric_values)

    if baseline_values is not None:
        for metric in [
            "successful_transactions",
            "failed_transactions",
            "success_rate_pct",
            "failure_rate_pct",
            "average_path_length",
            "average_routing_fee",
            "total_depletion_events",
            "nodes_with_depletion_events",
        ]:
            record[f"{metric}_change_pct"] = relative_change(
                float(metric_values[metric]),
                float(baseline_values[metric]),
            )

    return record


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.out)
    ebc_dir = ensure_dir(root / "ebc")
    baseline_dir = ensure_dir(root / "baseline")
    attacks_dir = ensure_dir(root / "attacks")
    plots_dir = ensure_dir(root / "plots")

    with Path(args.params).open("r", encoding="utf-8") as file:
        params = json.load(file)

    amount = int(params["amount"])
    count = int(params["count"])
    epsilon = float(params["epsilon"])
    drop_disabled = bool(params.get("drop_disabled", True))
    drop_low_cap = bool(params.get("drop_low_cap", True))
    with_depletion = bool(params.get("with_depletion", True))

    print("\n# 1. Loading the selected complete LN snapshot")
    snapshot = read_snapshot(args.edges, args.snapshot_id)
    original_pair_capacity = calculate_pair_capacity(snapshot)

    print("\n# 2. Calculating EBC ranking once")
    directed_ebc, pair_ebc, ebc_metadata = calculate_ebc_rankings(
        snapshot=snapshot,
        amount_sat=amount,
        metric=args.ebc_metric,
        k_sources=args.k_sources,
        seed=args.ebc_seed,
        drop_disabled=drop_disabled,
        drop_low_cap=drop_low_cap,
    )
    directed_ebc.to_csv(
        ebc_dir / "edge_betweenness_directed.csv",
        index=False,
    )
    pair_ebc.to_csv(
        ebc_dir / "edge_betweenness_channel_pairs.csv",
        index=False,
    )
    json_dump(ebc_dir / "ebc_metadata.json", ebc_metadata)

    print("\n# 3. Creating one fixed transaction workload")
    prepared_baseline = prepare_policy_edges(
        snapshot=snapshot,
        amount_sat=amount,
        drop_disabled=drop_disabled,
        drop_low_cap=drop_low_cap,
    )
    providers = load_providers(args.meta)
    workload = generate_fixed_workload(
        prepared_edges=prepared_baseline,
        amount_sat=amount,
        count=count,
        epsilon=epsilon,
        providers=providers,
        seed=args.transaction_seed,
    )
    workload_file = root / "fixed_transaction_workload.csv"
    workload.to_csv(workload_file, index=False)

    print("\n# 4. Running the baseline simulation")
    baseline_simulator = SequentialLiquiditySimulator(
        prepared_edges=prepared_baseline,
        workload=workload,
        amount_sat=amount,
        liquidity_seed=args.liquidity_seed,
        route_metric=args.route_metric,
        with_depletion=with_depletion,
    )
    baseline_result = baseline_simulator.run()
    baseline_config = {
        "experiment_type": "baseline",
        "edges": str(Path(args.edges)),
        "snapshot_id": args.snapshot_id,
        "params": params,
        "transaction_seed": args.transaction_seed,
        "liquidity_seed": args.liquidity_seed,
        "route_metric": args.route_metric,
        "original_physical_capacity_sat": original_pair_capacity,
    }
    save_simulation_result(
        result=baseline_result,
        output_dir=baseline_dir,
        config=baseline_config,
    )
    baseline_values = metrics_to_dict(baseline_result.metrics)

    summary_records = [
        flatten_metrics(
            experiment_type="baseline",
            experiment_id="baseline",
            metric_values=baseline_values,
        )
    ]

    top_k_values = sorted(set(int(value) for value in args.top_k))
    remove_values = sorted(set(float(value) for value in args.remove_pct))

    total_experiments = len(top_k_values) * len(remove_values)
    completed = 0

    print(
        f"\n# 5. Running {total_experiments} EBC-targeted "
        "liquidity-shock simulations"
    )

    for remove_pct in remove_values:
        for top_k in top_k_values:
            completed += 1
            experiment_id = f"ebc_top{top_k}_remove{remove_pct:g}"
            experiment_dir = ensure_dir(attacks_dir / experiment_id)

            print(
                f"\n[{completed}/{total_experiments}] "
                f"Running {experiment_id}"
            )

            attacked_snapshot, selected, changed = apply_pair_shock(
                snapshot=snapshot,
                pair_ranking=pair_ebc,
                top_k=top_k,
                remove_pct=remove_pct,
            )

            selected.to_csv(
                experiment_dir / "selected_channel_pairs.csv",
                index=False,
            )
            changed.to_csv(
                experiment_dir / "shocked_raw_channel_rows.csv",
                index=False,
            )

            if args.save_shocked_edges:
                attacked_snapshot.to_csv(
                    experiment_dir / "ln_edges_shocked_snapshot.csv",
                    index=False,
                )

            prepared_attack = prepare_policy_edges(
                snapshot=attacked_snapshot,
                amount_sat=amount,
                drop_disabled=drop_disabled,
                drop_low_cap=drop_low_cap,
            )

            attack_simulator = SequentialLiquiditySimulator(
                prepared_edges=prepared_attack,
                workload=workload,
                amount_sat=amount,
                liquidity_seed=args.liquidity_seed,
                route_metric=args.route_metric,
                with_depletion=with_depletion,
            )
            attack_result = attack_simulator.run()

            removed_capacity = float(changed["removed_capacity"].sum())
            attacked_pair_capacity = calculate_pair_capacity(attacked_snapshot)

            attack_config = {
                "experiment_type": "ebc_attack",
                "experiment_id": experiment_id,
                "edges": str(Path(args.edges)),
                "snapshot_id": args.snapshot_id,
                "params": params,
                "ebc_metric": args.ebc_metric,
                "k_sources": args.k_sources,
                "ebc_seed": args.ebc_seed,
                "transaction_seed": args.transaction_seed,
                "liquidity_seed": args.liquidity_seed,
                "route_metric": args.route_metric,
                "top_k": top_k,
                "remove_pct": remove_pct,
                "selected_channel_pairs": len(selected),
                "changed_raw_rows": len(changed),
                "removed_capacity_sat_from_directed_rows": removed_capacity,
                "original_physical_capacity_sat": original_pair_capacity,
                "attacked_physical_capacity_sat": attacked_pair_capacity,
            }

            save_simulation_result(
                result=attack_result,
                output_dir=experiment_dir,
                config=attack_config,
            )

            attack_values = metrics_to_dict(attack_result.metrics)

            summary_records.append(
                flatten_metrics(
                    experiment_type="ebc_attack",
                    experiment_id=experiment_id,
                    metric_values=attack_values,
                    top_k=top_k,
                    remove_pct=remove_pct,
                    baseline_values=baseline_values,
                    selected_pairs=len(selected),
                    changed_raw_rows=len(changed),
                    removed_capacity_sat=removed_capacity,
                )
            )

            pd.DataFrame(summary_records).to_csv(
                root / "sweep_summary.csv",
                index=False,
            )

    sweep_summary = pd.DataFrame(summary_records)
    sweep_summary.to_csv(root / "sweep_summary.csv", index=False)

    sweep_config = {
        "edges": str(Path(args.edges)),
        "snapshot_id": args.snapshot_id,
        "params": params,
        "top_k": top_k_values,
        "remove_pct": remove_values,
        "ebc_metric": args.ebc_metric,
        "k_sources": args.k_sources,
        "ebc_seed": args.ebc_seed,
        "transaction_seed": args.transaction_seed,
        "liquidity_seed": args.liquidity_seed,
        "route_metric": args.route_metric,
        "save_shocked_edges": args.save_shocked_edges,
        "number_of_attack_experiments": total_experiments,
    }
    json_dump(root / "sweep_config.json", sweep_config)

    print("\n# 6. Sweep completed")
    print(f"Summary saved to: {root / 'sweep_summary.csv'}")

    if not args.skip_plots:
        print("\n# 7. Generating comparison graphs")
        plot_script = Path(__file__).resolve().parent / "plot_ebc_sweep.py"
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--summary",
                str(root / "sweep_summary.csv"),
                "--out-dir",
                str(plots_dir),
            ],
            check=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
