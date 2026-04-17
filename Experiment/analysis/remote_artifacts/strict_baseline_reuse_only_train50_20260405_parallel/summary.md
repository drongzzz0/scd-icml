# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.4017 | 0.4017 | 0.6364 | 0.2200 | 0.0 | 0.000 | 128.56 | 0.00 |
| Reuse-only | -7.6187 | 7.6187 | 0.0161 | 0.0000 | 80554229.8 | 1.000 | 52.77 | -75.79 |
