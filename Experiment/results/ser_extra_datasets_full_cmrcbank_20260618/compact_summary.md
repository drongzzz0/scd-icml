# ser extra-dataset full CMRC-bank diagnostics

Route: CMRC-trained reuse19/patch13. Full-context HotpotQA is diagnostic and not claim-facing.

## HotpotQA full-context

| mode | n | F1 | EM | NLL | PPL | TTFT ms | Total ms | Payload MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_consumer | 7405 | 0.353979 | 0.219851 | 0.990936 | 2.693756 | 215.327 | 870.941 | 0.000000 |
| reuse_only | 7405 | 0.008499 | 0.000000 | 9.083990 | 8813.063540 | 54.480 | 714.147 | 205.752346 |
| scd_full | 7405 | 0.103724 | 0.077920 | 1.524896 | 4.594665 | 60.865 | 132.157 | 289.339237 |

## WikiText-2

| mode | n | F1 | EM | NLL | PPL | TTFT ms | Total ms | Payload MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_consumer | 763 | 0.221258 | 0.000000 | 1.983874 | 7.270854 | 60.318 | 709.852 | 0.000000 |
| reuse_only | 763 | 0.089332 | 0.000000 | 6.832337 | 927.355309 | 24.672 | 667.117 | 15.929882 |
| scd_full | 763 | 0.138638 | 0.000000 | 3.168148 | 23.763423 | 30.460 | 652.337 | 22.401397 |
