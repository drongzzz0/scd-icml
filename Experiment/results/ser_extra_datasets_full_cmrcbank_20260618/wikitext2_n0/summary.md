# Strict Runtime Comparison: wikitext2

| mode | avg_nll | ppl | f1 | em | payload_mb | compression | ttft_ms | total_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | 1.9839 | 7.2709 | 0.2213 | 0.0000 | 0.0000 | 0.000 | 60.32 | 709.85 | 0.00 |
| Reuse-only | 6.8323 | 927.3553 | 0.0893 | 0.0000 | 15.9299 | 1.000 | 24.67 | 667.12 | -35.65 |
| SCD | 3.1681 | 23.7634 | 0.1386 | 0.0000 | 22.4014 | 0.711 | 30.46 | 652.34 | -29.86 |
