# Strict Runtime Comparison

| mode | avg_logp | avg_nll | f1 | em | bytes | compression | ttft_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | -0.7022 | 0.7022 | 0.7375 | 0.3800 | 0.0 | 0.000 | 50.30 | 0.00 |
| Reuse-only | -0.7006 | 0.7006 | 0.7372 | 0.3800 | 43322245.1 | 1.000 | 88.33 | 38.03 |
| Reuse + Patch | -0.6638 | 0.6638 | 0.7700 | 0.4600 | 86644490.2 | 0.500 | 99.14 | 48.84 |
