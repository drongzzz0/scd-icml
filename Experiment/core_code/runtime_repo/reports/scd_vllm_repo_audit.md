# SCD vLLM Repo Audit

- repo_root: `.`
- python: `3.13.9`

## Existing entry points

- `build-route` -> `scd_repro.cli.build_route:main`
- `capture-pair` -> `scd_repro.cli.capture_pair:main`
- `eval` -> `scd_repro.cli.eval_cmd:main`
- `eval-text-strict` -> `scd_repro.cli.eval_text_strict:main`
- `infer` -> `scd_repro.cli.infer:main`
- `profile-layers` -> `scd_repro.cli.profile_layers:main`
- `report` -> `scd_repro.cli.report_cmd:main`
- `summarize-mistral-repro` -> `scd_repro.cli.summarize_mistral_repro:main`
- `train-patch` -> `scd_repro.cli.train_patch:main`
- `train-reuse` -> `scd_repro.cli.train_reuse:main`

## Existing model/config files

- `configs/data.yaml`
- `configs/data_a800.yaml`
- `configs/data_local_smoke.yaml`
- `configs/data_ser.yaml`
- `configs/diag_routes_qwen17/reuse_layer_0.json`
- `configs/diag_routes_qwen17/reuse_layer_1.json`
- `configs/diag_routes_qwen17/reuse_layer_2.json`
- `configs/diag_routes_qwen17/reuse_layer_3.json`
- `configs/diag_routes_qwen17/reuse_layer_6.json`
- `configs/diag_routes_qwen17/reuse_layer_7.json`
- `configs/eval.yaml`
- `configs/eval_baseline_only_train50_20260405.json`
- `configs/eval_baseline_reuse_only_50_20260405.json`
- `configs/eval_baseline_reuse_only_fulldev3219_20260405.json`
- `configs/eval_baseline_reuse_patchonly_50_20260404.json`
- `configs/eval_full_consumer_only.json`
- `configs/eval_oracle_diag.yaml`
- `configs/eval_patch_only_50_20260405.json`
- `configs/eval_patch_only_fulldev3219_20260405.json`
- `configs/eval_patch_only_train50_20260405.json`
- `configs/eval_patchonly_hotqids_20260405.json`
- `configs/eval_q2_mistral7_to_mistrallite_fix7_20260407.json`
- `configs/eval_qwen17_route_a_state_native.json`
- `configs/eval_qwen17_route_a_state_native_nofallback.json`
- `configs/eval_qwen17_route_b_direct_kv.json`
- `configs/eval_qwen17_route_b_direct_kv_aligned.json`
- `configs/eval_qwen17_route_b_direct_kv_nofallback.json`
- `configs/eval_qwen17_route_c_state_recompute.json`
- `configs/eval_qwen17_route_c_state_recompute_aligned.json`
- `configs/eval_qwen17_route_c_state_recompute_nofallback.json`
- `configs/eval_reuse_fallback_diag50.json`
- `configs/eval_reuse_fallback_main200.json`
- `configs/eval_reuse_only_train50_20260405.json`
- `configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`
- `configs/eval_scd_icml_main_methods.json`
- `configs/eval_scd_icml_no_fallback_no_recompute.json`
- `configs/eval_scd_icml_qwen32_no_fallback_no_recompute.json`
- `configs/eval_stepc_fix6_current.yaml`
- `configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json`
- `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json`
- `configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json`
- `configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_paper_k6_e1_h2048_r512_b2_20260404.json`
- `configs/models.yaml`
- `configs/models_a800.yaml`
- `configs/models_a800_mistrallite_to_mistral7_20260405.json`
- `configs/models_a800_qwen25_32b_base_to_instruct_template.json`
- `configs/models_local_smoke.yaml`
- `configs/models_mistrallite_to_mistral7_template.json`
- `configs/models_ser.yaml`
- `configs/models_ser_mistral7_self_20260404.json`
- `configs/models_ser_mistral7_to_mistrallite_20260405.json`
- `configs/models_ser_mistral7_to_mistrallite_mql.json`
- `configs/models_ser_mistral_instruct.yaml`
- `configs/models_ser_mistrallite_self_20260404.json`
- `configs/models_ser_mistrallite_to_mistral7_20260405.json`
- `configs/models_ser_mistrallite_to_mistral7_mql.json`
- `configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json`
- `configs/models_ser_qwen17_bf16.yaml`
- `configs/models_ser_qwen17_gray.yaml`
- `configs/models_ser_qwen17_self.yaml`
- `configs/qwen17_route_a_state_native_eval.json`
- `configs/qwen17_route_b_direct_kv_eval.json`
- `configs/qwen17_route_c_state_recompute_eval.json`
- `configs/qwen17_self_allreuse_eval_aligned.json`
- `configs/qwen17_self_patch_xnorm_nativekv_residual_mlp_alllayers_approx_subset12_e2_h2048_b2_20260403.json`
- `configs/qwen17_self_patch_xnorm_nativekv_residual_mlp_alllayers_identity_train.json`
- `configs/qwen17_self_patch_xnorm_nativekv_residual_mlp_alllayers_train.json`
- `configs/qwen17_to_bf16_patch_xnorm_nativekv_residual_mlp_alllayers_e2_h2048_b2_20260403.json`
- `configs/qwen17_to_bf16_patch_xnorm_nativekv_residual_mlp_alllayers_subset12_e1_h2048_b2_20260403.json`
- `configs/qwen17_to_bf16_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403.json`
- `configs/qwen17_to_gray_patch_xnorm_nativekv_shared_residual_mlp_alllayers_subset12_e1_h2048_r512_b2_20260403.json`
- `configs/qwen32_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b1_20260614.json`
- `configs/route_reuse_patchonly_all32_20260404.json`
- `configs/route_selection.yaml`
- `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`
- `configs/routes/ser_qwen17_reuse_layers_2_6_7.json`
- `configs/routes/ser_qwen17_reuse_layers_7_6.json`
- `configs/routes/ser_qwen17_self_all_reuse.json`
- `configs/run_ser_mistral_instruct_pair.sh`
- `configs/scd_vllm_mvp.yaml`

## Existing route files

- `configs/eval_qwen17_route_a_state_native.json`
- `configs/eval_qwen17_route_a_state_native_nofallback.json`
- `configs/eval_qwen17_route_b_direct_kv.json`
- `configs/eval_qwen17_route_b_direct_kv_aligned.json`
- `configs/eval_qwen17_route_b_direct_kv_nofallback.json`
- `configs/eval_qwen17_route_c_state_recompute.json`
- `configs/eval_qwen17_route_c_state_recompute_aligned.json`
- `configs/eval_qwen17_route_c_state_recompute_nofallback.json`
- `configs/qwen17_route_a_state_native_eval.json`
- `configs/qwen17_route_b_direct_kv_eval.json`
- `configs/qwen17_route_c_state_recompute_eval.json`
- `configs/route_reuse_patchonly_all32_20260404.json`
- `configs/route_selection.yaml`
- `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`
- `configs/routes/ser_qwen17_reuse_layers_2_6_7.json`
- `configs/routes/ser_qwen17_reuse_layers_7_6.json`
- `configs/routes/ser_qwen17_self_all_reuse.json`
- `configs/train_qwen17_route_a_state_native.json`
- `configs/train_qwen17_route_b_direct_kv.json`

## Existing KV capture/export candidates

- `src/scd_repro/cli/capture_pair.py`

## Existing Reuse/Patch candidates

### Reuse

- `src/scd_repro/cli/train_reuse.py`
- `src/scd_repro/reuse.py`
- `scripts/__pycache__/generate_conditional_reuse_routes.cpython-313.pyc`
- `scripts/__pycache__/select_patch_reuse_route.cpython-313.pyc`
- `scripts/generate_conditional_reuse_routes.py`
- `scripts/run_ser_qwen17_self_reuse_diag.sh`
- `scripts/run_ser_qwen17_self_reuse_group4_diag.sh`
- `scripts/run_ser_qwen17_self_reuse_group4_xkvstyle_diag.sh`
- `scripts/run_ser_qwen17_self_reuse_layerwise_diag.sh`
- `scripts/select_patch_reuse_route.py`

### Patch

- `src/scd_repro/cli/train_patch.py`
- `src/scd_repro/direct_kv_residual_patch.py`
- `src/scd_repro/patch.py`
- `scripts/__pycache__/make_patch_meta_route.cpython-313.pyc`
- `scripts/__pycache__/merge_patch_ckpts.cpython-313.pyc`
- `scripts/__pycache__/select_patch_reuse_route.cpython-313.pyc`
- `scripts/make_nopatch_route.py`
- `scripts/make_patch_meta_route.py`
- `scripts/merge_patch_ckpts.py`
- `scripts/resume_ser_mistral_instruct_from_patch.sh`
- `scripts/run_ser_scd_icml_parallel_patch_shards.sh`
- `scripts/run_ser_scd_icml_patchonly_anchor.sh`
- `scripts/select_patch_reuse_route.py`

## Existing metrics/eval candidates

- `src/scd_repro/cli/eval_cmd.py`
- `src/scd_repro/cli/eval_cmrc_strict.py`
- `src/scd_repro/cli/eval_text_strict.py`
- `src/scd_repro/cli/merge_eval_cmrc_strict.py`
- `src/scd_repro/cli_eval_cmrc_strict.py`
- `scripts/__pycache__/collect_route_eval_metrics.cpython-313.pyc`
- `scripts/__pycache__/evaluate_selected_route.cpython-313.pyc`
- `scripts/collect_route_eval_metrics.py`
- `scripts/evaluate_selected_route.py`
- `scripts/filter_strict_eval_by_qids.py`
- `scripts/run_paper_eval.py`
- `scripts/run_ser_qwen17_route_eval.sh`
- `scripts/run_ser_scd_icml_eval_existing_route.sh`
- `scripts/run_ser_scd_icml_postprocess_candidate_evals.sh`

## Existing tests

- `tests/test_analysis_figures_script.py`
- `tests/test_cmrc_split.py`
- `tests/test_filter_strict_eval_by_qids_script.py`
- `tests/test_finalize_full7405_bundle_script.py`
- `tests/test_layer_mapping.py`
- `tests/test_metrics.py`
- `tests/test_neighboring_baseline_table_script.py`
- `tests/test_offline_cost_table_script.py`
- `tests/test_paper_artifact_bundle_script.py`
- `tests/test_paper_artifact_check_script.py`
- `tests/test_quality_gated_route_script.py`
- `tests/test_result_tables_script.py`
- `tests/test_route_serialization.py`
- `tests/test_scd_vllm_bundle.py`
- `tests/test_scd_vllm_hf_export.py`
- `tests/test_scd_vllm_kv_spec.py`
- `tests/test_scd_vllm_metrics.py`
- `tests/test_scd_vllm_truekv_oracle.py`
- `tests/test_scd_vllm_vllm_baseline.py`
- `tests/test_selector.py`
- `tests/test_stack_viz_input_extractor.py`
- `tests/test_stack_viz_script.py`
- `tests/test_text_benchmarks.py`

## Missing pieces

- `src/scd_vllm/scd_injection.py`

## Proposed mapping to scd_vllm modules

- `src/scd_vllm/kv_spec.py`: model/runtime KV compatibility checks.
- `src/scd_vllm/bundle.py`: standard SCDCacheBundle manifest and hash helpers.
- `src/scd_vllm/env_probe.py`: local/A800 dependency, model, and vLLM API probe.
- `src/scd_vllm/hf_export.py`: HF full-consumer reference bundle exporter.
- `src/scd_vllm/vllm_baseline.py`: plain official vLLM baseline latency runner.
- `src/scd_vllm/truekv_oracle.py`: vLLM TrueKV API inspection and future save/load correctness oracle.
- `src/scd_vllm/scd_injection.py`: future SCD reconstructed KV injection path, gated on TrueKV oracle pass.

## Risks before implementation

- vLLM runtime phases require a Linux CUDA host; Windows local work should stay static/unit-test only.
- vLLM KVConnector APIs are version-sensitive; Phase 4 must inspect the installed vLLM source before coding against method names.
- Current SCD evidence uses dataset-specific routes; the vLLM MVP must not assume a universal route.
- TrueKV oracle is the main blocker. SCD injection must remain skipped until TrueKV save/load is proven or fails loudly.
