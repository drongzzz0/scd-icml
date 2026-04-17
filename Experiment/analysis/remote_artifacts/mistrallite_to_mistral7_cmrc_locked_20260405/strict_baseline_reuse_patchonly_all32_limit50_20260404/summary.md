# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7056 | 0.7056 | 0.6185 | 0.3600 | 0.0 | 0.000 | 127.86 | 0.00 |
| Reuse-only | -7.4975 | 7.4975 | 0.0117 | 0.0000 | 76656148.5 | 1.000 | 33.39 | -94.47 |
| Patch-only | -0.6742 | 0.6742 | 0.8072 | 0.5400 | 153312297.0 | 0.500 | 121.48 | -6.37 |
