# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.4621 | 0.4621 | 0.7358 | 0.4400 | 0.0 | 0.000 | 99.05 | 0.00 |
| Reuse-only | -0.5349 | 0.5349 | 0.7775 | 0.4600 | 43322245.1 | 1.000 | 58.11 | -40.94 |
| Patch-only | -0.5782 | 0.5782 | 0.7547 | 0.4200 | 43322245.1 | 1.000 | 63.96 | -35.09 |
