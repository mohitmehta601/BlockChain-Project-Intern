from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    calculate_ebc_rankings,
    ensure_dir,
    json_dump,
    read_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Lightning Network edge betweenness centrality "
            "rankings for one ln_edges.csv snapshot."
        )
    )
    parser.add_argument("--edges", required=True, help="Path to ln_edges.csv")
    parser.add_argument("--snapshot-id", type=int, default=0)
    parser.add_argument("--amount", type=int, default=10000)
    parser.add_argument(
        "--metric",
        choices=["hops", "fee"],
        default="hops",
        help="Use minimum-hop or lower-fee paths for EBC.",
    )
    parser.add_argument(
        "--k-sources",
        type=int,
        default=200,
        help=(
            "Approximate EBC source sample count. "
            "Use 0 for exact all-node EBC."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.out_dir)

    snapshot = read_snapshot(args.edges, args.snapshot_id)

    directed, pairs, metadata = calculate_ebc_rankings(
        snapshot=snapshot,
        amount_sat=args.amount,
        metric=args.metric,
        k_sources=args.k_sources,
        seed=args.seed,
    )

    directed.to_csv(
        output / "edge_betweenness_directed.csv",
        index=False,
    )
    pairs.to_csv(
        output / "edge_betweenness_channel_pairs.csv",
        index=False,
    )
    json_dump(output / "ebc_metadata.json", metadata)

    print("\nTop 10 critical physical channel pairs:")
    print(pairs.head(10).to_string(index=False))
    print("\nSaved:")
    print(output / "edge_betweenness_directed.csv")
    print(output / "edge_betweenness_channel_pairs.csv")
    print(output / "ebc_metadata.json")


if __name__ == "__main__":
    main()
