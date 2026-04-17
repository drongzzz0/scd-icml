# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7765 | 0.7765 | 0.7829 | 0.6000 | 0.0 | 0.000 | 122.50 | 0.00 |
| Reuse-only | -6.1226 | 6.1226 | 0.0212 | 0.0000 | 76656148.5 | 1.000 | 33.60 | -88.90 |
| Patch-only | -0.7342 | 0.7342 | 0.7528 | 0.5000 | 153312297.0 | 0.500 | 75.94 | -46.56 |
