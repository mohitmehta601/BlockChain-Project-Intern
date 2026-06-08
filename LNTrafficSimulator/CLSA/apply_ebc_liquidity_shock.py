from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    apply_pair_shock,
    ensure_dir,
    read_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce capacity on Top-K EBC-ranked physical node pairs "
            "for one Lightning Network snapshot."
        )
    )
    parser.add_argument("--edges", required=True, help="Path to ln_edges.csv")
    parser.add_argument("--snapshot-id", type=int, default=0)
    parser.add_argument(
        "--ebc-pairs",
        required=True,
        help="Path to edge_betweenness_channel_pairs.csv",
    )
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--remove-pct", type=float, required=True)
    parser.add_argument(
        "--out-edges",
        required=True,
        help="Output CSV containing the shocked selected snapshot.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.out_dir)

    snapshot = read_snapshot(args.edges, args.snapshot_id)
    pair_ranking = pd.read_csv(args.ebc_pairs, low_memory=False)

    attacked, selected, changed = apply_pair_shock(
        snapshot=snapshot,
        pair_ranking=pair_ranking,
        top_k=args.top_k,
        remove_pct=args.remove_pct,
    )

    out_edges = Path(args.out_edges)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    attacked.to_csv(out_edges, index=False)

    selected.to_csv(output / "selected_channel_pairs.csv", index=False)
    changed.to_csv(output / "shocked_raw_channel_rows.csv", index=False)

    print("\nShock applied successfully.")
    print(f"Selected physical channel pairs: {len(selected)}")
    print(f"Changed raw directed rows: {len(changed)}")
    print(f"Liquidity removed from each selected pair: {args.remove_pct}%")
    print("\nSaved:")
    print(out_edges)
    print(output / "selected_channel_pairs.csv")
    print(output / "shocked_raw_channel_rows.csv")


if __name__ == "__main__":
    main()
