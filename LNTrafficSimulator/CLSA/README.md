# CLSA: EBC-Targeted Cascading Liquidity Shock Attack

This folder adds a reproducible Cascading Liquidity Shock Attack experiment
pipeline to an existing `LNTrafficSimulator` project without modifying the
original `lnsimulator` package.

## Default sweep

The full sweep runs:

- Top-K EBC channel pairs: `5 10 20 30 40 50 100 200`
- Liquidity removed from every selected pair: `50 60 70 80 90`
- Total attacks: `8 × 5 = 40`
- Plus one baseline simulation

The same LN snapshot, transaction workload, payment amount, workload seed,
and directional-liquidity seed are used for every run.

## Expected project structure

Copy the complete `CLSA` folder into your cloned project root:

```text
LNTrafficSimulator/
│
├── CLSA/
│   ├── __init__.py
│   ├── common.py
│   ├── compute_ebc.py
│   ├── apply_ebc_liquidity_shock.py
│   ├── run_simulation_from_csv.py
│   ├── run_ebc_sweep.py
│   ├── plot_ebc_sweep.py
│   ├── params_clsa.json
│   ├── requirements.txt
│   └── README.md
│
├── ln_data/
│   ├── ln_edges.csv
│   └── 1ml_meta_data.csv
│
├── lnsimulator/
├── scripts/
└── output/
```

`sample.json` and `ln.tsv` are not required for this sweep.

## Install packages

From the project root in PowerShell:

```powershell
pip install -r .\CLSA\requirements.txt
```

## Recommended first run: approximate EBC

This runs the complete 40-attack sweep on the full selected snapshot.
Only EBC calculation is approximated using 200 source nodes.

```powershell
python .\CLSA\run_ebc_sweep.py `
  --edges .\ln_data\ln_edges.csv `
  --snapshot-id 0 `
  --params .\CLSA\params_clsa.json `
  --meta .\ln_data\1ml_meta_data.csv `
  --out .\output\clsa_ebc_sweep `
  --top-k 5 10 20 30 40 50 100 200 `
  --remove-pct 50 60 70 80 90 `
  --ebc-metric hops `
  --k-sources 200 `
  --transaction-seed 42 `
  --liquidity-seed 42
```

## Exact EBC run

To calculate EBC using every source node, replace:

```powershell
--k-sources 200
```

with:

```powershell
--k-sources 0
```

The attack simulations always use the complete network snapshot. The
`--k-sources` argument only controls whether EBC ranking is approximate or
exact. Exact EBC can take considerably longer.

## Save every shocked graph

By default, each attack folder saves the selected channels and changed raw
rows, but not a complete duplicate CSV snapshot. To save every shocked
snapshot too, append:

```powershell
--save-shocked-edges
```

This uses additional disk space.

## Output structure

```text
output/clsa_ebc_sweep/
│
├── fixed_transaction_workload.csv
├── sweep_config.json
├── sweep_summary.csv
│
├── ebc/
│   ├── edge_betweenness_directed.csv
│   ├── edge_betweenness_channel_pairs.csv
│   └── ebc_metadata.json
│
├── baseline/
│   ├── metrics.csv
│   ├── transaction_log.csv
│   ├── edge_usage.csv
│   ├── depletion_events.csv
│   ├── router_incomes.csv
│   ├── lengths_distrib.csv
│   ├── initial_liquidity.csv
│   ├── final_liquidity.csv
│   └── experiment_config.json
│
├── attacks/
│   ├── ebc_top5_remove50/
│   ├── ebc_top10_remove50/
│   ├── ...
│   └── ebc_top200_remove90/
│
└── plots/
    ├── failure_rate_vs_top_k.png
    ├── success_rate_vs_top_k.png
    ├── depletion_events_vs_top_k.png
    ├── average_path_length_vs_top_k.png
    ├── average_fee_vs_top_k.png
    ├── failure_rate_heatmap.png
    └── failed_transactions_change_heatmap.png
```

## Run only one baseline manually

```powershell
python .\CLSA\run_simulation_from_csv.py `
  --edges .\ln_data\ln_edges.csv `
  --snapshot-id 0 `
  --params .\CLSA\params_clsa.json `
  --meta .\ln_data\1ml_meta_data.csv `
  --out .\output\clsa_manual_baseline `
  --workload-out .\output\fixed_transaction_workload.csv `
  --transaction-seed 42 `
  --liquidity-seed 42
```

## Calculate EBC only

```powershell
python .\CLSA\compute_ebc.py `
  --edges .\ln_data\ln_edges.csv `
  --snapshot-id 0 `
  --amount 10000 `
  --metric hops `
  --k-sources 200 `
  --seed 42 `
  --out-dir .\output\clsa_manual_ebc
```

## Apply one shock manually

```powershell
python .\CLSA\apply_ebc_liquidity_shock.py `
  --edges .\ln_data\ln_edges.csv `
  --snapshot-id 0 `
  --ebc-pairs .\output\clsa_manual_ebc\edge_betweenness_channel_pairs.csv `
  --top-k 10 `
  --remove-pct 70 `
  --out-edges .\ln_data\ln_edges_ebc_top10_remove70.csv `
  --out-dir .\output\clsa_manual_top10_remove70
```

## Run one shocked snapshot manually

The standalone shock script writes one selected snapshot CSV. Its rows retain
their original `snapshot_id`, so use the same ID:

```powershell
python .\CLSA\run_simulation_from_csv.py `
  --edges .\ln_data\ln_edges_ebc_top10_remove70.csv `
  --snapshot-id 0 `
  --params .\CLSA\params_clsa.json `
  --meta .\ln_data\1ml_meta_data.csv `
  --out .\output\clsa_manual_attack_top10_remove70 `
  --workload-in .\output\fixed_transaction_workload.csv `
  --liquidity-seed 42
```
