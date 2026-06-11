# Evaluation Readme

- mode: synthetic_smoke
- run_id: qwen35_close_smoke__scd_original__cmrc2018__seed1
- pair_id: qwen35_close_smoke
- pair_bucket: close
- method: scd_original
- claim_anchor: table1_qwen32b_primary
- patch_layers: [7, 15]
- selection_trace: [{'step': 1, 'selected_layer': 7, 'marginal_gain': 0.063}, {'step': 2, 'selected_layer': 15, 'marginal_gain': 0.03}]
- significance_note: single-seed outputs are descriptive rather than confirmatory.
- route_rescue_summary.csv is intentionally empty until the original baseline fails and route A/B/C are run on the same matched pair.
