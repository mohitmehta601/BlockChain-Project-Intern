from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    SequentialLiquiditySimulator,
    ensure_dir,
    generate_fixed_workload,
    json_dump,
    load_providers,
    prepare_policy_edges,
    read_snapshot,
    read_workload,
    save_simulation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one reproducible liquidity-depletion simulation "
            "from an ln_edges.csv snapshot."
        )
    )
    parser.add_argument("--edges", required=True)
    parser.add_argument("--snapshot-id", type=int, default=0)
    parser.add_argument("--params", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workload-in", default=None)
    parser.add_argument("--workload-out", default=None)
    parser.add_argument("--transaction-seed", type=int, default=42)
    parser.add_argument("--liquidity-seed", type=int, default=42)
    parser.add_argument(
        "--route-metric",
        choices=["fee", "hops"],
        default="fee",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.out)

    with Path(args.params).open("r", encoding="utf-8") as file:
        params = json.load(file)

    amount = int(params["amount"])
    count = int(params["count"])
    epsilon = float(params["epsilon"])
    drop_disabled = bool(params.get("drop_disabled", True))
    drop_low_cap = bool(params.get("drop_low_cap", True))
    with_depletion = bool(params.get("with_depletion", True))

    snapshot = read_snapshot(args.edges, args.snapshot_id)
    prepared_edges = prepare_policy_edges(
        snapshot=snapshot,
        amount_sat=amount,
        drop_disabled=drop_disabled,
        drop_low_cap=drop_low_cap,
    )

    if args.workload_in:
        workload = read_workload(args.workload_in)
    else:
        providers = load_providers(args.meta)
        workload = generate_fixed_workload(
            prepared_edges=prepared_edges,
            amount_sat=amount,
            count=count,
            epsilon=epsilon,
            providers=providers,
            seed=args.transaction_seed,
        )

    if args.workload_out:
        workload_file = Path(args.workload_out)
        workload_file.parent.mkdir(parents=True, exist_ok=True)
        workload.to_csv(workload_file, index=False)

    simulator = SequentialLiquiditySimulator(
        prepared_edges=prepared_edges,
        workload=workload,
        amount_sat=amount,
        liquidity_seed=args.liquidity_seed,
        route_metric=args.route_metric,
        with_depletion=with_depletion,
    )

    result = simulator.run()

    config = {
        "edges": str(Path(args.edges)),
        "snapshot_id": args.snapshot_id,
        "params": params,
        "workload_in": args.workload_in,
        "workload_out": args.workload_out,
        "transaction_seed": args.transaction_seed,
        "liquidity_seed": args.liquidity_seed,
        "route_metric": args.route_metric,
    }

    save_simulation_result(result, output, config)

    print("\nSimulation completed.")
    print(result.metrics.to_string(index=False))
    print("\nSaved results in:")
    print(output)


if __name__ == "__main__":
    main()
