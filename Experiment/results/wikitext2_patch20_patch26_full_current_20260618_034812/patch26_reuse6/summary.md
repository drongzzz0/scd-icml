# Strict Runtime Comparison: wikitext2

| mode | avg_nll | ppl | f1 | em | payload_mb | compression | ttft_ms | total_ms | delta_ttft_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Consumer | 1.9839 | 7.2709 | 0.2213 | 0.0000 | 0.0000 | 0.000 | 61.30 | 715.47 | 0.00 |
| SCD | 2.2067 | 9.0860 | 0.2008 | 0.0000 | 28.8729 | 0.552 | 37.34 | 665.14 | -23.96 |
