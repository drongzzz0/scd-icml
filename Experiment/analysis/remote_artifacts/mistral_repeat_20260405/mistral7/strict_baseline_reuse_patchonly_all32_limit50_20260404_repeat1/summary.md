# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7056 | 0.7056 | 0.6185 | 0.3600 | 0.0 | 0.000 | 129.72 | 0.00 |
| Reuse-only | -0.7048 | 0.7048 | 0.6486 | 0.3600 | 76656148.5 | 1.000 | 45.64 | -84.08 |
| Patch-only | -0.7045 | 0.7045 | 0.6185 | 0.3600 | 153312297.0 | 0.500 | 118.77 | -10.95 |
