# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.4621 | 0.4621 | 0.7358 | 0.4400 | 0.0 | 0.000 | 140.18 | 0.00 |
| Reuse-only | -0.5349 | 0.5349 | 0.7775 | 0.4600 | 43322245.1 | 1.000 | 45.48 | -94.71 |
| Patch-only | -0.5794 | 0.5794 | 0.7667 | 0.4400 | 43322245.1 | 1.000 | 159.47 | 19.28 |
