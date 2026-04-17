# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.4916 | 0.4916 | 0.5969 | 0.2500 | 0.0 | 0.000 | 103.78 | 0.00 |
| Reuse-only | -7.8040 | 7.8040 | 0.0141 | 0.0000 | 79695052.8 | 1.000 | 33.41 | -70.37 |
