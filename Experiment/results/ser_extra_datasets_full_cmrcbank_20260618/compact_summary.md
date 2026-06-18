# ser CMRC-bank full-dataset diagnostic summary

Remote run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_datasets_full_cmrcbank_20260618_121640`.

| Dataset | Method | n | F1 | EM | NLL | PPL | TTFT ms | Total ms | Payload MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMRC2018 dev full | Full Consumer | 3219 | 0.744253 | 0.477788 | 0.550784 | 1.734612 | 105.262038 | 762.813012 | 0.000000 |
| CMRC2018 dev full | Reuse-only | 3219 | 0.008128 | 0.000000 | 8.138158 | 3422.607202 | 34.947930 | 690.761903 | 78.997554 |
| CMRC2018 dev full | SCD | 3219 | 0.671571 | 0.411929 | 0.905659 | 2.473561 | 40.321946 | 285.183295 | 111.090310 |
| HotpotQA full-context | Full Consumer | 7405 | 0.353979 | 0.219851 | 0.990936 | 2.693756 | 215.327408 | 870.941145 | 0.000000 |
| HotpotQA full-context | Reuse-only | 7405 | 0.008499 | 0.000000 | 9.083990 | 8813.063540 | 54.480396 | 714.146742 | 205.752346 |
| HotpotQA full-context | SCD | 7405 | 0.103724 | 0.077920 | 1.524896 | 4.594665 | 60.864799 | 132.156827 | 289.339237 |
| WikiText-2 test | Full Consumer | 763 | 0.221258 | 0.000000 | 1.983874 | 7.270854 | 60.317945 | 709.851824 | 0.000000 |
| WikiText-2 test | Reuse-only | 763 | 0.089332 | 0.000000 | 6.832337 | 927.355309 | 24.672482 | 667.116861 | 15.929882 |
| WikiText-2 test | SCD | 763 | 0.138638 | 0.000000 | 3.168148 | 23.763423 | 30.460461 | 652.337162 | 22.401397 |

Interpretation: this is a full-dataset diagnostic for the CMRC2018-trained MistralLite-to-Mistral-7B reuse19/patch13 route. It is useful as transfer/risk evidence, not as claim-facing HotpotQA or WikiText-2 quality evidence.
