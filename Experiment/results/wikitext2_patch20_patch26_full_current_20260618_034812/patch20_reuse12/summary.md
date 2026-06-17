# Strict Runtime Comparison: wikitext2

| mode | avg_nll | ppl | f1 | em | payload_mb | compression | ttft_ms | total_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | 1.9839 | 7.2709 | 0.2213 | 0.0000 | 0.0000 | 0.000 | 62.40 | 735.87 | 0.00 |
| SCD | 2.3817 | 10.8228 | 0.1885 | 0.0000 | 25.8861 | 0.615 | 35.84 | 686.13 | -26.56 |
