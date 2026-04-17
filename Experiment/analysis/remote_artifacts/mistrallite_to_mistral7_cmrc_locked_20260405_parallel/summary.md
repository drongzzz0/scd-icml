# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.6858 | 0.6858 | 0.6911 | 0.4427 | 0.0 | 0.000 | 133.39 | 0.00 |
| Reuse-only | -7.9743 | 7.9743 | 0.0109 | 0.0000 | 81896546.3 | 1.000 | 38.49 | -94.90 |
| Patch-only | -0.6651 | 0.6651 | 0.7898 | 0.5769 | 163793092.6 | 0.500 | 67.37 | -66.02 |
