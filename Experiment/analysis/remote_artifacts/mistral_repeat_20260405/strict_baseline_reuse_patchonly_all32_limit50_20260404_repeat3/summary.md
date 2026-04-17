# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7765 | 0.7765 | 0.7829 | 0.6000 | 0.0 | 0.000 | 113.59 | 0.00 |
| Reuse-only | -0.8033 | 0.8033 | 0.7686 | 0.5800 | 76656148.5 | 1.000 | 46.16 | -67.42 |
| Patch-only | -0.7779 | 0.7779 | 0.7829 | 0.6000 | 153312297.0 | 0.500 | 68.00 | -45.58 |
