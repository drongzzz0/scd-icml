# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7056 | 0.7056 | 0.6185 | 0.3600 | 0.0 | 0.000 | 130.72 | 0.00 |
| Reuse-only | -0.7048 | 0.7048 | 0.6486 | 0.3600 | 76656148.5 | 1.000 | 37.97 | -92.74 |
| Patch-only | -0.7045 | 0.7045 | 0.6185 | 0.3600 | 153312297.0 | 0.500 | 57.28 | -73.44 |
