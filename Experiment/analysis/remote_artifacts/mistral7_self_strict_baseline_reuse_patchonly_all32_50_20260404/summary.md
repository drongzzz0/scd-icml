# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7056 | 0.7056 | 0.6185 | 0.3600 | 0.0 | 0.000 | 129.91 | 0.00 |
| Reuse-only | -0.7048 | 0.7048 | 0.6486 | 0.3600 | 76656148.5 | 1.000 | 35.01 | -94.89 |
| Patch-only | -0.7046 | 0.7046 | 0.6385 | 0.3800 | 153312297.0 | 0.500 | 47.10 | -82.81 |
