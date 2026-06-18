# Experiment Log

## 2026-05-25 - synthetic_p0_smoke

Experiment id: synthetic_p0_smoke

Purpose: verify the implementation path for the locked mainline before running real Qwen paired traces.

Method:

- Generate synthetic paired small/large KV traces with a known Large-to-Small layer correspondence.
- Discover the Large-to-Small mapping using offline ridge probes.
- Train a target-layer-conditioned Small-to-Large Patch network using the discovered mapping as scaffold.
- Calibrate an offline oracle reuse / patch / recompute route from reconstruction error.

Parameters planned:

- epochs: 5
- samples: 512
- sequence length: 24
- small layers: 4
- large layers: 8
- small KV dim: 16
- large KV dim: 32
- device: auto, prefer CUDA if available

Inputs: synthetic traces only.

Expected outputs:

- Experiment/core_code/checkpoints/model_final.pth
- experiments/synthetic_p0/results.json
- Experiment/core_code/logs/machine_learning_agent_iter_submit.json

Analysis status: pending run.

## 2026-05-25 - synthetic_p0_smoke result

Execution steps:

- Inspected GPU availability with nvidia-smi; GPUs were idle.
- Ran python3 -m compileall scd_tpami run_training_testing.py.
- Ran CUDA_VISIBLE_DEVICES=0 python3 run_training_testing.py --epochs 5 --device auto --samples 512 --seq-len 24.

Outputs:

- Experiment/core_code/checkpoints/model_final.pth
- experiments/synthetic_p0/results.json
- Experiment/core_code/logs/machine_learning_agent_iter_submit.json

Results:

- device: cuda:0
- mapping accuracy: 1.0000
- validation MSE: 0.063816
- validation normalized RMSE: 0.403020
- validation mean cosine: 0.920449
- route fractions: reuse=0.5000, patch=0.3500, recompute=0.1500
- runtime seconds: 5.870

Analysis:

The synthetic smoke run validates the software path and artifact contract: paired traces, Large-to-Small mapping discovery, compression-guided Patch training, route calibration, checkpoint writing, and submit log writing. This is not real-model evidence; the next experiment must replace synthetic traces with Qwen2.5 paired traces and replace oracle error routing with online-available risk features.

Next actions:

1. Copy or implement minimal Qwen2.5 paired trace capture.
2. Run 1.5B -> 7B trace capture on a small prompt set.
3. Reuse this scaffold for real mapping heatmaps and KV-native Patch training.

## 2026-05-26 - qwen25_real_p0_preflight

Experiment id: qwen25_real_p0_preflight

Purpose: prepare the scaffold for real Qwen2.5-1.5B -> Qwen2.5-7B paired KV trace capture and downstream mapping/Patch training.

Preflight findings:

- GPUs 0-3 were idle by `nvidia-smi` at preflight time.
- Existing project code has only synthetic paired traces and no real model capture entrypoint.
- Qwen2.5-1.5B and Qwen2.5-7B were not found under `/disk_n/c101-3-mql/models` or `/home/c101-3-mql/models`.
- A Qwen2.5-3B-Instruct asset exists under `/disk_n/c101-3-mql/gaojie/models`, but it is not the requested 1.5B -> 7B pair.

Implementation update:

- Added a real paired-trace capture module and CLI.
- Added a real mapping/Patch training CLI that saves `mapping_heatmap.png`, `real_mapping_patch_results.json`, and `model_final_real_qwen25.pth`.

Next actions:

1. Verify the new real-code entrypoints with `compileall`.
2. Make the requested Qwen2.5-1.5B and Qwen2.5-7B checkpoints available.
3. Run a small capture smoke with 2-4 prompts and max length 64.
4. Run `run_real_mapping_patch.py` on the captured artifact.

## 2026-05-26 - qwen25_1p5b_to_7b_real_p0 result

Experiment id: qwen25_1p5b_to_7b_real_p0

Purpose: produce the first real-model P0 evidence for the locked post-ICML direction, using Qwen2.5-1.5B -> Qwen2.5-7B paired KV traces to generate a Large-to-Small mapping heatmap and train the KV-native Patch scaffold.

Inputs and environment:

- Remote workspace: `/disk_n/c101-3-mql/projects/scd-tpami`
- Model pair: `/disk_n/c101-3-mql/models/Qwen2.5-1.5B` -> `/disk_n/c101-3-mql/models/Qwen2.5-7B`
- Runtime environment: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
- GPU placement: `CUDA_VISIBLE_DEVICES=0`
- Prompt count: 12
- Maximum length: 64
- Dtype: bf16

Execution steps:

- Downloaded Qwen2.5-1.5B and Qwen2.5-7B to `/disk_n/c101-3-mql/models` through `HF_ENDPOINT=https://hf-mirror.com` after direct Hugging Face download timed out.
- Ran `python -m compileall scd_tpami run_training_testing.py run_qwen25_capture.py run_real_mapping_patch.py`.
- Fixed Qwen2.5 `DynamicCache` extraction by supporting `cache.layers[*].keys` and `cache.layers[*].values`.
- Fixed the real-trace ridge projection path by regularizing the bias term and adding an `lstsq` fallback for singular normal equations.
- Captured paired traces with `python run_qwen25_capture.py --small-model /disk_n/c101-3-mql/models/Qwen2.5-1.5B --large-model /disk_n/c101-3-mql/models/Qwen2.5-7B --limit-prompts 12 --max-length 64 --batch-size 1 --dtype bf16 --device cuda:0`.
- Trained the mapping-conditioned Patch with `python run_real_mapping_patch.py --trace ../../experiments/qwen25_1p5b_to_7b/paired_traces.pt --epochs 10 --batch-size 2 --hidden-dim 512 --device cuda:0`.

Outputs:

- `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b/paired_traces.pt`
- `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b/paired_traces.manifest.json`
- `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b/mapping_heatmap.png`
- `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b/real_mapping_patch_results.json`
- `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/checkpoints/model_final_real_qwen25.pth`

Results:

- small KV shape: `[12, 28, 64, 512]`
- large KV shape: `[12, 28, 64, 1024]`
- discovered Large-to-Small mapping: `[3, 13, 23, 8, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 22, 14, 14, 14, 22, 14, 14, 23, 23, 23, 27, 27, 27]`
- final validation MSE: `12.567176818847656`
- final validation RMSE: `3.545021413030908`
- final validation normalized RMSE: `0.5234374617913784`
- final validation mean cosine: `0.32736536860466003`
- route fractions from offline error calibration: reuse=`0.5`, patch=`0.349888414144516`, recompute=`0.1501116156578064`
- capture runtime seconds: `18.98934245109558`
- mapping/Patch runtime seconds: `4.111410140991211`

Analysis:

This is the first real Qwen2.5 paired-trace result for the TPAMI extension scaffold. It verifies that the project can capture aligned small/large KV tensors, compute a non-synthetic Large-to-Small mapping score matrix, render the mapping heatmap, and train the KV-native Patch model on real traces. The run is still a P0 sanity experiment: it uses only 12 prompts and 64 tokens, and the routing fractions are offline oracle-style calibration from reconstruction error rather than an online risk predictor.

Next actions:

1. Increase prompt count and use a fixed real corpus split for train/validation/test.
2. Add stability diagnostics for the mapping matrix across prompt subsets and token positions.
3. Replace offline error-based routing with online-available risk features.
4. Run the same artifact contract on the next larger capture once the P0 metrics are stable.

## 2026-05-26 - qwen25_1p5b_to_7b result analysis

Artifacts found:

- experiments/qwen25_1p5b_to_7b/paired_traces.pt
- experiments/qwen25_1p5b_to_7b/real_mapping_patch_results.json
- experiments/qwen25_1p5b_to_7b/mapping_heatmap.png
- Experiment/core_code/checkpoints/model_final_real_qwen25.pth
- Experiment/core_code/logs/real_qwen25_mapping_patch_submit.json

Captured data:

- pair: Qwen2.5-1.5B -> Qwen2.5-7B
- prompts: 12
- max length: 64
- valid token fraction: 0.203125
- small KV shape: [12, 28, 64, 512]
- large KV shape: [12, 28, 64, 1024]

Submitted run metrics:

- training: 10 epochs, unmasked MSE over all 64 positions including padding
- unmasked validation MSE: 12.567176818847656
- unmasked validation mean cosine: 0.32736536860466003
- masked valid-token validation MSE: 13.745871289816602
- masked valid-token validation mean cosine: 0.28083109855651855
- route fractions: reuse=0.5, patch=0.349888414144516, recompute=0.1501116156578064

Baseline check on valid tokens:

- Patch MSE/cosine: 13.745871289816602 / 0.28083109855651855
- zero baseline MSE/cosine: 45.960779138513516 / 0.0
- train mean per layer-feature MSE/cosine: 1.483650119148166 / 0.6632699966430664
- train mean per layer-token-feature MSE/cosine: 1.0757123439008205 / 0.7973717451095581

Diagnostic rerun, not checkpointed:

- masked-loss 120 epoch existing-mapping Patch: val MSE 1.3314659089195222, val cosine 0.7505466938018799
- masked-loss 120 epoch identity-mapping Patch: val MSE 1.2228987410261825, val cosine 0.7748220562934875

Conclusion:

The real pipeline completed, but the submitted 10-epoch unmasked run is not a valid positive result. It underfits and loses badly to train-mean baselines. The in-memory masked-loss diagnostic indicates that the architecture can learn a useful mapping on this tiny trace set, but the next official run must use attention-mask-aware mapping/training/evaluation and include trivial mean and identity-mapping baselines.

## 2026-05-26 - qwen25_1p5b_to_7b_masked submit and judge

Implementation update: added attention-mask-aware Large-to-Small mapping, Patch training loss, validation metrics, and mandatory baselines.

Artifacts:

- experiments/qwen25_1p5b_to_7b_masked/real_mapping_patch_results.json
- experiments/qwen25_1p5b_to_7b_masked/analysis_summary.json
- experiments/qwen25_1p5b_to_7b_masked/mapping_heatmap.png
- Experiment/core_code/checkpoints/model_final_real_qwen25_masked.pth
- Experiment/core_code/checkpoints/model_final.pth

Key valid-token metrics:

- Patch discovered: MSE 1.080214, nRMSE 0.153307, cosine 0.770672
- Patch identity: MSE 1.246869, nRMSE 0.164709, cosine 0.776665
- Patch uniform: MSE 1.246869, nRMSE 0.164709, cosine 0.776665
- Baseline zero: MSE 45.960777, nRMSE 1.000000, cosine 0.000000
- Baseline train_mean_layer_feature: MSE 1.483650, nRMSE 0.179669, cosine 0.663270
- Baseline train_mean_layer_token_feature: MSE 1.075712, nRMSE 0.152987, cosine 0.797372

Judge conclusion: artifact contract is correct and the implementation signal is real, but this is not yet a final positive scientific result because the token-position mean baseline remains too strong on the tiny 12-prompt trace set.

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_masked

- Purpose: scale the real paired-trace Patch experiment to 512 prompts and max_length=128.
- Method: capture Qwen2.5-1.5B and Qwen2.5-7B KV traces, train mask-aware Small-to-Large Patch networks with discovered, identity, and uniform layer scaffolds.
- Environment: ser, /disk_n/c101-3-mql/projects/scd-tpami, GPU cuda:0, bf16 trace capture, 10 training epochs, batch_size=8, eval_batch_size=16.
- Inputs: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128/paired_traces.pt.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_masked/real_mapping_patch_results.json, /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_masked/analysis_summary.json, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/checkpoints/model_final_real_qwen25_masked.pth, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/checkpoints/model_final.pth.
- Result: best strategy identity; masked val MSE=0.723858, nRMSE=0.125410, cosine=0.824176. Strong token-position mean baseline MSE=1.049478, nRMSE=0.151005, cosine=0.804745.
- Analysis: identity/uniform Patch beats the token-position mean baseline; discovered mapping is weaker than identity/uniform, so compression-path discovery remains the current bottleneck.
- Next actions: soft/constrained mapping, prompt-disjoint longer-context validation, large-model decode/perplexity evaluation, then risk-router calibration.

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_dual_source

- Purpose: test whether compression-path scaffold provides incremental information beyond identity Patch.
- Method: dual-source Patch input concatenates identity depth source with discovered/local constrained compression source.
- Environment: ser, cuda:0, hidden_dim=768, epochs=10, batch_size=8, eval_batch_size=16.
- Result: best strategy identity_plus_local_hard; masked val MSE=0.549358, nRMSE=0.109253, cosine=0.868318. Same-run identity MSE=0.595306, nRMSE=0.113730, cosine=0.850867.
- Analysis: dual-source reduces MSE by 7.72% vs identity and improves cosine by 0.017451. It also reduces MSE by 47.65% vs token-position mean baseline.
- Conclusion: current implementation supports compression-guided Patch as auxiliary cross-layer context; next gate is decode/perplexity evaluation and robustness splits.
- Artifacts: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_dual_source/real_mapping_patch_results.json; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_dual_source/analysis_summary.json; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_dual_source/analysis_summary.md; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/checkpoints/model_final_real_qwen25_masked.pth

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_random_ablation

- Purpose: test whether dual-source gains are caused only by doubled input dimensionality.
- Method: compare identity_plus_local_hard against identity_plus_random_local and identity_plus_random_global under the same 512x128 trace, hidden_dim=768, epochs=10.
- Result: identity_plus_local_hard MSE=0.549358, nRMSE=0.109253, cosine=0.868318. Random-local MSE=0.570836, cosine=0.861201; random-global MSE=0.576047, cosine=0.862250.
- Analysis: compression-guided auxiliary source beats random-local by 3.76% MSE and random-global by 4.63% MSE. This supports structured compression-path contribution beyond capacity.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_random_ablation/analysis_summary.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_random_multiseed_ablation

- Purpose: repeat random auxiliary controls over seeds [1001, 1002, 1003, 1004, 1005].
- Result: fixed identity_plus_local_hard MSE=0.549358, cosine=0.868318. Random-local mean MSE=0.581602, max cosine=0.864195; random-global mean MSE=0.571623, max cosine=0.866120.
- Analysis: compression-guided local_hard beats all random-local and random-global seeds on both MSE and cosine.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_random_multiseed_ablation/analysis_summary.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_decode_eval

- Purpose: evaluate whether generated KV drafts are decode-safe under Qwen2.5-7B.
- Method: use predicted prefix KV as past_key_values, score 8 suffix tokens on 16 validation prompts. True-cache and full-recompute paths are included as plumbing controls.
- Result: full recompute NLL=4.782392; true cache NLL=4.782594; predicted identity NLL=6.425801; predicted identity_plus_local_hard NLL=7.284038.
- Analysis: true cache matches full recompute, validating the evaluation. Current predicted caches are not decode-safe; local_hard improves KV reconstruction but worsens suffix NLL vs identity.
- Conclusion: next implementation must optimize decode/logit quality or use risk routing / selective recomputation.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_decode_eval/analysis_summary.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_partial_recompute_calibrated

- Purpose: test whether partial exact-KV replacement can recover decode quality and identify a useful routing granularity.
- Method: evaluate local_hard predicted KV mixed with true KV under multiple replacement policies; additionally compute a fixed high-risk layer order from 64 training prompts and test calibrated top-k layer replacement on 16 validation prompts.
- Result: full recompute NLL=4.782392; predicted cache NLL=7.284038; oracle top14 layer NLL=4.881981; calibrated top14 layer NLL=4.705880; fixed last14 layer NLL=6.943402.
- Analysis: layer-level replacement is stable; token/layer-token mixing is unstable. Calibrated non-contiguous high-risk layers recover decode NLL, but this is an upper-bound unless a practical partial large-model execution path exists.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_partial_recompute/analysis_summary_calibrated.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_decode_routing_budget

- Purpose: refine the decode gate after full predicted-cache decode failed.
- Method: test layer replacement budgets and a simple high-risk layer-weighted KV loss.
- Result: calibrated top10 layer replacement NLL=4.688350 at 0.357 layer budget; full recompute NLL=4.782392; predicted cache NLL=7.284038. Weighted top14 factor4 Patch decode NLL=7.839099.
- Analysis: calibrated layer routing is promising as an upper bound; per-prompt MSE oracle and layer-weighted MSE do not directly solve decode.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_partial_recompute/analysis_summary_budget_curve.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_contiguous_policy

- Purpose: test deployable contiguous layer-block partial recompute policies after arbitrary layer oracle showed decode recovery.
- Method: mix Patch-generated KV with true contiguous prefix, suffix, middle, and best-window layer blocks; evaluate suffix NLL on 16 validation prompts.
- Result: predicted cache NLL=7.284038; prefix-8 NLL=4.756209 at 0.286 layer budget; prefix-10 NLL=4.732270 at 0.357 layer budget; suffix-10 NLL=7.018043. Best contiguous window starts at layer 0 for all widths and all 16 prompts.
- Analysis: prefix-block recompute is the practical routing direction. It can be implemented as early-stop large-model prefill for the first k layers, then Patch KV for upper layers.
- Artifact: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_contiguous_policy/analysis_summary.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_online_prefix

- Purpose: validate a deployable online prefix-k policy after contiguous layer analysis identified lower-layer prefix recompute as the practical route.
- Method: run Qwen2.5-7B embedding plus the first k decoder layers during prefill, combine exact lower-layer KV with compression-guided Patch upper-layer KV, and score suffix NLL on 16 validation prompts.
- Result: full recompute mean NLL=4.782392; Patch-only predicted cache NLL=7.284038; online_prefix_8 NLL=4.755569 at 8/28 exact large layers; online_prefix_10 NLL=4.730057 at 10/28 exact large layers.
- Analysis: prefix-8 and prefix-10 recover decode quality while avoiding exact large-model prefill for upper 20 or 18 layers. This is an online sequential-transformer implementation, not an arbitrary offline true-KV replacement upper bound.
- Limitations: 16 prompts, short suffix scoring, and rough timing only; controlled latency and longer-context validation remain open.
- Artifacts: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_online_prefix/analysis_summary.json; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_online_prefix/analysis_summary.md; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/logs/online_prefix_policy_submit.json

## 2026-05-26 - qwen25_1p5b_to_7b_512x128_online_prefix_latency

- Purpose: measure the latency side of the online prefix-k route after prefix-8/10 recovered decode NLL.
- Method: CUDA-event microbenchmark with model load excluded. Compare full Qwen2.5-7B prefill against serial small prefill + KV feature pack + Patch + first-k Qwen2.5-7B layers. Run both actual short validation prefixes and forced 128/256/512 token timing-only prefixes.
- Result: actual short prefixes and forced-128 are not faster. Forced-256 is near break-even for prefix-8 and positive for prefix-4. Forced-512 gives mean serial savings: prefix-4=43.4%, prefix-8=30.0%, prefix-10=23.5%.
- Analysis: prefix-k policy is not a short-context accelerator in the current implementation, but has a plausible long-context latency regime. Prefix-8 currently has the best quality/speed tradeoff: near-full-recompute NLL and clear forced-512 timing savings.
- Limitations: forced-length timing repeats token IDs and is not a quality benchmark; real long-context paired traces are still required.
- Artifacts: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_online_prefix/latency_analysis_summary.json; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_512x128_online_prefix/latency_analysis_summary.md


## 2026-05-26 LongBench 128x512 diagnostic update

- Purpose: test whether the fixed mainline, offline Large-to-Small compression path plus compression-guided Small-to-Large Patch, survives a real long-context LongBench setting.
- Data: 128 LongBench cached prompts, selected from rows with token_len >= 512 and captured at max_length=512. Trace: experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt.
- Models: Qwen2.5-1.5B small model and Qwen2.5-7B large model.
- Reconstruction:
  - e10 identity_plus_local_hard: MSE 1.154364, nRMSE 0.158056, cosine 0.782774.
  - e30 identity_plus_local_hard: MSE 0.998853, nRMSE 0.147025, cosine 0.812751.
  - e30 upper20 weight4: MSE 1.029124, nRMSE 0.149236, cosine 0.808869.
- Online NLL, limit-val=16, eval-tokens=8:
  - e10 k8/k10/k16/k24/k26 deltas: +2.0810, +1.7047, +0.9443, +0.2678, +0.2358.
  - e30 k8/k10/k16/k24/k26 deltas: +1.9302, +1.5614, +1.0406, +0.2851, +0.2528.
  - upper20w4 k8/k10/k16/k24/k26 deltas: +2.0726, +1.6791, +1.1590, +0.4100, +0.3615.
- Latency: full large prefill mean 0.057779s; k8/k10/k12/k16 saved fractions are 30.53%, 24.00%, 17.38%, 3.81% respectively, while k20+ is slower than full prefill.
- Conclusion: current LongBench result is diagnostic rather than a success result. KV reconstruction improves, but online NLL remains poor in the speed-positive region. Next implementation should add decode-aware or attention-output-aware training loss.
- Artifacts: Experiment/core_code/logs/longbench_128x512_diagnostic_summary.json and Experiment/core_code/logs/longbench_128x512_diagnostic_summary.md.


## 2026-05-26 Decode-aware Patch iteration

- Implemented: Experiment/core_code/run_decode_aware_patch.py. The Large-to-Small compression mapping is kept fixed as the Patch scaffold; only Patch parameters are fine-tuned with mixed-KV decode losses.
- Smoke test: passed on 1 sample / 1 epoch / eval_tokens=2, proving gradients flow from large-model decode logits through mixed KV into the Patch.
- Main runs:
  - decodeaware_t24_e2_kl05: train_limit=24, epochs=2, prefix set 8/10/12/16, kv=0.1, kl=0.5, ce=0.0.
  - decodeaware_t48_e3_kl05: train_limit=48, epochs=3, prefix set 8/10/12/16, kv=0.1, kl=0.5, ce=0.0.
  - decodeaware_t48_e3_k8k10_kl05: train_limit=48, epochs=3, prefix set 8/10, kv=0.1, kl=0.5, ce=0.0.
  - decodeaware_t48_e3_kl03_ce02: train_limit=48, epochs=3, prefix set 8/10/12/16, kv=0.1, kl=0.3, ce=0.2.
- Best general run: decodeaware_t48_e3_kl03_ce02. Online NLL deltas vs full recompute on 16 validation prompts: k8 +1.3930, k10 +0.9224, k12 +0.7560, k16 +0.5849, k20 +0.3623, k24 +0.0873, k26 +0.0575.
- Compared with e30 pure MSE: k8 +1.9302, k10 +1.5614, k12 +1.4869, k16 +1.0406. Decode-aware clearly improves the speed-relevant region.
- Remaining issue: k8-k12 still have too much NLL loss for a strong speed-quality claim; k24/k26 are close to full recompute but slower than full prefill in serial timing.
- Artifacts: Experiment/core_code/logs/decode_aware_patch_summary.json and Experiment/core_code/logs/decode_aware_patch_summary.md.


## 2026-05-26 Decode-aware full split scale-up

- Scale-up runs used all 102 train prompts and all 26 validation prompts from the LongBench 128x512 trace.
- t102_e5_kl03ce02: k8 +1.0214, k10 +0.6699, k12 +0.5562, k16 +0.4133.
- t102_e8_kl03ce02: k8 +0.9783, k10 +0.6799, k12 +0.5732, k16 +0.4304. More epochs helped k8 but hurt k10-k16.
- t102_e5_kv005_kl03ce02 is the current best general checkpoint: k10 +0.6632, k12 +0.5486, k16 +0.4076. It passes the k10 provisional threshold but not k12/k16.
- Conclusion: continue from low-KV decode-aware training, but the next improvement should change the objective by adding suffix hidden-state or attention-output distillation rather than only adding epochs.
- Artifacts: Experiment/core_code/logs/decode_aware_scaleup_summary.json and Experiment/core_code/logs/decode_aware_scaleup_summary.md.


## 2026-05-27 Hidden-state distillation test

- Implemented optional hidden-state distillation in Experiment/core_code/run_decode_aware_patch.py via --hidden-weight and --hidden-layer. Default hidden_weight=0 keeps all previous experiments unchanged.
- Smoke test passed with hidden_weight=0.05, hidden_layer=-1.
- Full split tests from the current best low-KV checkpoint:
  - hidden_hw005_last: k8 +1.0453, k10 +0.6978, k12 +0.5704, k16 +0.4332.
  - hidden_hw001_last: k8 +1.0307, k10 +0.6851, k12 +0.5586, k16 +0.4084.
  - hidden_hw001_penultimate: k8 +1.0268, k10 +0.6878, k12 +0.5643, k16 +0.4251.
- Baseline current best remains t102_e5_kv005_kl03ce02: k8 +1.0265, k10 +0.6632, k12 +0.5486, k16 +0.4076.
- Conclusion: suffix hidden-state MSE is a negative branch. Next method should be attention-output distillation or risk routing, not more hidden-state MSE.
- Artifacts: Experiment/core_code/logs/hidden_distill_summary.json and Experiment/core_code/logs/hidden_distill_summary.md.


## 2026-05-27 - oracle-routing-diagnostic

- Purpose: decide whether prompt-level oracle routing has enough headroom over fixed-k prefix-layer mixing before training a real risk scorer.
- Method: no retraining; reuse the existing online prefix-k mixed-KV evaluation path on the same 26 validation prompts; evaluate per-prompt NLL for k in {0,4,8,10,12,14,16,18,20,22,24,26,28}, then compute oracle prompt-level choices for tau in {0.3,0.5,0.7,1.0}.
- Inputs: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/logs/decode_aware_scaleup_summary.md.
- Environment: ser, cuda:0, dtype=bf16, seed=17; deterministic NLL scoring, eval_tokens=8; no training or weight updates.
- Exact commands run:
  - `python run_oracle_prompt_routing_diagnostic.py --latency-audit-json /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/latency_audit_remeasure_512_b1.json --latency-audit-command 'python run_online_prefix_latency_benchmark.py --trace /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt --checkpoint /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth --device cuda:0 --dtype bf16 --limit-val 1 --warmup 1 --eval-tokens 8 --force-prefix-len 512 --prefix-layers 4,8,10,12,14,16,18,20,22,24,26 --output /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/latency_audit_remeasure_512_b1.json' --hardware-summary 'NVIDIA GeForce RTX 4090 D, single GPU cuda:0 on ser'`
  - `python run_online_prefix_latency_benchmark.py --trace /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt --checkpoint /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth --device cuda:0 --dtype bf16 --limit-val 1 --warmup 1 --eval-tokens 8 --force-prefix-len 512 --prefix-layers 4,8,10,12,14,16,18,20,22,24,26 --output /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/latency_audit_remeasure_512_b1.json`
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/per_prompt_per_k_nll.csv; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/pareto_oracle_vs_fixed.png; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/oracle_routing_summary.json; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle-routing-diagnostic.md.
- Result table: fixed k=10: delta=0.6632, mean_k=10.00, saved=24.01%; fixed k=12: delta=0.5486, mean_k=12.00, saved=17.39%; fixed k=16: delta=0.4076, mean_k=16.00, saved=3.81%; oracle-router(tau=0.3): delta=0.1657, mean_k=15.69, saved=5.95%; oracle-router(tau=0.5): delta=0.2972, mean_k=13.85, saved=12.30%; oracle-router(tau=0.7): delta=0.4073, mean_k=12.00, saved=17.09%; oracle-router(tau=1.0): delta=0.6195, mean_k=9.38, saved=26.06%.
- Decision rule: A=0.4459, B=0.2972, A-B=0.1487.
- Final decision: train a real prompt-level risk scorer using {small_entropy, small_margin, prompt_length, patch_residual_norm} as features.

## 2026-05-27 - oracle-routing-expanded-feature-diagnostic

- Purpose: scale oracle prompt-level routing diagnostic to a clean held-out router validation set and identify candidate routing features without training a router.
- Method: captured 100 tokenizer-hash-deduplicated LongBench-family prompts at max_len=512; reused current Patch and online prefix-k diagnostic; combined original 26 validation prompts with the new set; extracted small-model entropy/margin and Patch residual diagnostics; computed Pearson/Spearman correlations.
- Inputs: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/per_prompt_per_k_nll.csv; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/extra_per_prompt_per_k_nll.csv.
- Environment: ser, cuda:0, dtype=bf16; no Patch retraining, no router training, max_len=512.
- Exact command run: `python run_router_feature_correlation.py --original-trace /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt --extra-trace /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt --checkpoint /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth --original-csv /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/per_prompt_per_k_nll.csv --extra-csv /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/extra_per_prompt_per_k_nll.csv --extra-prompt-manifest /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/prompt_manifest.json --output-dir /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing --markdown /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle-routing-diagnostic.md --experiment-log /disk_n/c101-3-mql/projects/scd-tpami/docs/experiment-log.md --device cuda:0 --dtype bf16`
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_correlations.csv; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_val_features.csv; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_correlation_summary.json; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_small_mean_entropy.png, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_small_mean_top1_margin.png, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_small_p95_entropy.png, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_prompt_length.png, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_patch_residual_kv_l2.png, /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/router_feature_patch_residual_kv_l2_late.png; /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle-routing-diagnostic.md.
- Result: combined N=126; original iso-k12 gap=0.1413; combined iso-k12 gap=0.1946; abs diff=0.0533; N=26 result not_reproduced.
- Top features by Spearman hardness: patch_residual_kv_l2_late, patch_residual_kv_l2, small_mean_top1_margin.
- Recommendation: use patch_residual_kv_l2_late, patch_residual_kv_l2, small_mean_top1_margin, small_mean_entropy; target continuous score (NLL_delta); keep Patch residual features as post-Patch/offline diagnostics or replace with an online proxy before deployment.


## 2026-05-27 - noncontiguous-oracle-and-ridge-baseline

- Purpose: test whether prefix-k exact-layer routing is a topology bottleneck, and test whether a closed-form ridge baseline can replace or initialize the current MLP Patch before training a ridge+residual variant.
- Method Task A: no retraining; fixed current best Patch; N=126 router-val set; prefix-k sanity checked against existing CSV; greedy non-contiguous per-prompt exact-layer selection up to k=18 using the same suffix NLL cache-evaluation path.
- Method Task B: fit closed-form ridge on the original 102 train prompts only using the same selected_small_kv feature consumed by MappingConditionedPatch; evaluate ridge-only on the fixed N=126 router-val set at k={0,4,8,12,16}; apply the pre-registered early-stop rule before any residual training.
- Inputs: /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt; /disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth; Experiment/analysis/oracle_routing/per_prompt_per_k_nll.csv; Experiment/analysis/oracle_routing/extra_per_prompt_per_k_nll.csv.
- Environment: ser, RTX 4090 D GPUs; Task A sharded over cuda:0-3, dtype=bf16, seed=17; Task B ridge fit/eval on cuda:0, dtype=bf16 for evaluation; no new prompt capture, no trace schema change, no router training.
- Exact commands run:
  - python Experiment/core_code/run_noncontiguous_oracle_eval.py --mode shard --device cuda:0 --num-shards 4 --shard-index <0..3> --candidate-batch-size 8 --output-dir Experiment/analysis/oracle_routing with CUDA_VISIBLE_DEVICES=<0..3>.
  - python Experiment/core_code/run_noncontiguous_oracle_eval.py --mode combine --num-shards 4 --output-dir Experiment/analysis/oracle_routing.
  - python Experiment/core_code/run_ridge_residual_patch_experiment.py --mode fit_eval --device cuda:0 --force-refit.
- Outputs: Experiment/analysis/oracle_routing/noncontiguous_per_prompt_per_k_nll.csv; Experiment/analysis/oracle_routing/noncontiguous_vs_prefix_summary.csv; Experiment/analysis/oracle_routing/noncontiguous_oracle_diagnostic.md; Experiment/analysis/oracle_routing/noncontiguous_oracle_summary.json; Experiment/analysis/ridge_residual/ridge_baseline.pt; Experiment/analysis/ridge_residual/ridge_only_eval.csv; Experiment/analysis/ridge_residual/ridge_only_summary.json; Experiment/analysis/ridge_residual/early_stop_finding.md; Experiment/analysis/codex_2026_05_27_summary.md.
- Result Task A: max_gap=1.1221; k8 gap=1.1221, k12 gap=0.9546, k16 gap=0.8746; non-contiguous strict-better fraction is 0.9841/0.9921/1.0000 at k8/k12/k16.
- Decision Task A: Non-contiguous routing is structurally advantageous. Recommend redesigning deployment route to per-layer selection.
- Result Task B: ridge_only_NLL_delta@12=0.5326; current_best_NLL_delta@12=0.7016; early-stop threshold=0.7516.
- Decision Task B: early stop triggered; skip ridge+residual B.3-B.5 training by rule. Ridge-only is already better than the current best at k=12.
- Next action: make closed-form ridge the new Patch baseline in comparisons, then design a deployable approximation for the non-contiguous/per-layer oracle route.


## 2026-05-27 - noncontiguous-oracle-sanity-checks

- Purpose: validate the unusually large non-contiguous oracle gain before attempting a deployable static layer-set policy.
- Method: analysis-only sanity task; reused Task A CSV and evaluated only missing k=0/k=28 closure/equality values with the same Patch checkpoint and suffix decode cache path. No Patch training, no checkpoint regeneration, no trace modification, no Task D static policy evaluation.
- Inputs: Experiment/analysis/oracle_routing/noncontiguous_per_prompt_per_k_nll.csv; Experiment/analysis/oracle_routing/noncontiguous_vs_prefix_summary.csv; Experiment/analysis/oracle_routing/noncontiguous_oracle_diagnostic.md; Experiment/analysis/oracle_routing/extra_per_prompt_per_k_nll.csv.
- Environment: ser, cuda:0, dtype=bf16, seed=17.
- Exact command run: python Experiment/core_code/run_static_layer_set_policy.py --mode sanity --device cuda:0.
- Outputs: Experiment/analysis/oracle_routing/sanity_checks.md. Task A CSV was augmented only with missing k=0/k=28 sanity values.
- Result C.1: FAIL. mean prefix k28 delta=-0.0002931; mean noncontig k28 delta=-0.0002931; max abs prefix/noncontig k28 delta=0.0505016, above the 1e-5 threshold. Top violating prompts: 99, 67, 81, 104, 16.
- Result C.2: PASS. max k0 prefix-vs-noncontig diff=4.99e-11; mean diff=2.84e-11.
- Result C.3: k8 top-8 frequency sum=3.8889 (weak), pairwise Jaccard=0.2498; k12 top-12 frequency sum=7.0238 (medium), pairwise Jaccard=0.3351.
- Final verdict: FAIL: C.1 k=28 closure violated. Per task rule, Task D was not run.
- Suspected bug location: exact-cache closure path / DynamicCache conversion or comparison baseline for full recompute versus cached suffix replay; investigate before trusting the non-contiguous oracle magnitude.


## 2026-05-27 - k28-closure-failure-diagnosis

- Purpose: diagnose why k=28 all-exact cache replay failed closure against the full-forward NLL baseline after the non-contiguous oracle sanity check.
- Method: diagnosis-only; no pipeline fix, no Task A/C rerun, no training, no trace modification. Ran four targeted tests on prompt 99 plus cross-prompt triangulation on the top five violators and five near-zero prompts.
- Inputs: Experiment/analysis/oracle_routing/noncontiguous_per_prompt_per_k_nll.csv; experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt; experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt; current Qwen2.5-7B model path from traces.
- Environment: ser, cuda:0, dtype=bf16, seed=17.
- Exact command run: python Experiment/core_code/debug_k28_closure.py.
- Outputs: Experiment/analysis/oracle_routing/debug_k28_closure/test1_kv_bitlevel_prompt99.csv; Experiment/analysis/oracle_routing/debug_k28_closure/test2_nll_triangulation_prompt99.json; Experiment/analysis/oracle_routing/debug_k28_closure/test3_position_mask_audit.txt; Experiment/analysis/oracle_routing/debug_k28_closure/test4_cross_prompt_pattern.csv; Experiment/analysis/oracle_routing/debug_k28_closure/diagnosis_report.md.
- Result Test 1: cached trace KV and fresh prefix KV for prompt 99 have max absolute difference 0 across layers. Cached dtype is torch.float32 holding bf16 values; fresh prefix cache dtype is torch.bfloat16.
- Result Test 2: prompt 99 full fresh NLL=2.9810574055; cached replay NLL=2.9305558205; eval baseline NLL=2.9810574055; A-B=0.0505015850; A-C=0; fresh-prefix-cache replay matches cached-trace replay.
- Result Test 3: route replay uses past_seen_tokens=past_len and suffix positions start at past_len; full-forward suffix positions are equivalent; attention length/sum are equivalent for all-valid 512-token prompts.
- Result Test 4: mean abs(A-B) for top violators=0.0352990627; near-zero prompts=0.0000607412. A-C remains zero and B-C equals the Task-A k=28 residual.
- Verdict: H5 plus cache-replay numerical-path drift. The baseline denominator is fresh full forward while route NLL is cached suffix replay; these paths are not numerically closed under bf16/Qwen2.5 attention. H1 stored-cache precision loss, H2 position mismatch, H3 RoPE mismatch, and H4 attention-mask mismatch are not supported.
- Implication: the 1.12 non-contiguous-vs-prefix gap is likely robust as a relative route-vs-route result, but absolute NLL deltas versus the fresh full-forward baseline should be recomputed with an all-exact cache-replay denominator if strict k=28 closure is required.
- Recommended fix, not implemented: use k=28 all-exact cached replay NLL as the zero baseline for cache-replay policies and keep fresh full-forward NLL as a separate diagnostic column.

## 2026-05-27 16:47 - closure-fix-ridge-oracle-per-layer-r2

- Purpose: repair k=28 closure baseline, rerun MLP v2 summaries, evaluate ridge non-contiguous oracle, and measure per-layer ridge fit quality.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/noncontiguous_per_prompt_per_k_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single-GPU shard jobs on RTX 4090 D.
- Commands: `python run_closure_fix_ridge_oracle.py --mode f_baseline/f_update/g_shard/g_combine/h_metrics/write_summary`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/baseline_cache_replay_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/noncontiguous_vs_prefix_summary_v2.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/ridge_vs_mlp_oracle_summary.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/ridge_oracle_diagnostic.md`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_per_layer_metrics.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_per_layer_diagnostic.md`.
- Results: F=pass; G max_gap_ridge=0.8292540745; H n_low=2.
- Conclusion: Non-contiguous + ridge is the new headline; next step is to validate a static set. Targeted MLP residual on the low-R2 pairs is the next step.

## 2026-05-27 17:22 - static-set-eval-targeted-residual-2-14

- Purpose: distinguish deployable static non-contiguous routing from greedy oracle artifact, and test a small residual on low-R2 layers 2 and 14.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/noncontiguous_per_prompt_per_k_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/ridge_per_prompt_per_k_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/baseline_cache_replay_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single-GPU eval shards on RTX 4090 D.
- Commands: `python run_static_targeted_residual.py --mode derive_static/static_shard/static_combine/train_residual/eval_residual/residual_report/write_summary`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/static_layer_sets.json`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/static_per_prompt_per_k_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/static_five_way_summary.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/static_set_diagnostic.md`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_ridge_plus_residual_2_14_e5/residual_2_14.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_plus_targeted_residual_eval.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/targeted_residual_diagnostic.md`.
- Results: static gap k12=0.3979280988; static advantage k12=0.3854092303; residual improvement k12=0.0314061191.
- Conclusion: Static loses part of greedy advantage but remains better than prefix. Use ridge + static as the main line. Targeted residual gives only marginal improvement; ridge-only remains the practical default.

## 2026-05-27 19:54 - task-k-decode-aware-loss-ablation

- Purpose: zero-training compare ridge, targeted residual, and decode-aware MLP before deciding whether to train decode-aware residuals.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_ridge_plus_residual_2_14_e5/residual_2_14.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/baseline_cache_replay_nll.csv`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single GPU RTX 4090 D.
- Commands: `python run_task_k_decode_aware_ablation.py --mode k1_eval/k1_verdict/k3_update`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/decode_aware/k1_existing_checkpoint_compare.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/decode_aware/k1_oracle_headroom.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/decode_aware/k1_verdict.md`.
- Results: g_kl=0.2004173722; g_res=-0.0314061191; verdict=HALT; locked recipe=ridge + static non-contiguous set from Task I; targeted residual is marginal.

## 2026-05-27 20:31 - task-l-deployable-prompt-level-router

- Purpose: train deployable small-side prompt-level router for locked ridge + static non-contiguous Patch recipe.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/oracle_routing/static_layer_sets.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, sklearn, single RTX 4090 D.
- Commands: `python run_task_l_prompt_router.py --mode extract_features/eval_labels/train_router/headline/feature_ablation/update_docs`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/router_features_n126.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/per_prompt_per_k_nll_locked_ridge.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/L2_results.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/L3_verdict.md`.
- Results: verdict=FAIL; dNLL=0.0333088404; recovery=0.2899831549; locked=ridge baseline + static non-contiguous set; router not locked.

## 2026-05-28 10:24 - task-m-within-prompt-oracle-routing

- Purpose: diagnose whether within-prompt block routing has more oracle headroom than Task L prompt-level routing.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_extra100x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/locked_ridge_static_sets_for_router.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, sklearn/numpy, single RTX 4090 D.
- Commands: `python run_task_m_within_prompt_oracle.py --mode extract_features/eval_blocks/analyze/update_docs`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/block_per_k_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/oracle_within_prompt.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/M2_headline.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/M2_verdict.md`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/block_feature_corr.csv`.
- Results: verdict=WEAK; ratio=1.7374351579; selected_B=4; locked recipe unchanged.

## 2026-05-28 10:44 - task-n-within-prompt-routing-cost-benefit

- Purpose: decide whether within-prompt routing quality headroom is worth runtime overhead.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/L3_headline.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/M2_headline.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single RTX 4090 D.
- Commands: `python run_task_n_within_prompt_cost_benefit.py --mode all`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N1_latency_breakdown.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N1_routing_overhead.md`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N2_quality_projection.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N3_pareto.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N4_verdict.md`.
- Results: verdict=DECLINE; d_q=0.0265621722; d_t_pct=0.0094548492.

## 2026-05-28 10:45 - task-n-within-prompt-routing-cost-benefit

- Purpose: decide whether within-prompt routing quality headroom is worth runtime overhead.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/L3_headline.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/M2_headline.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single RTX 4090 D.
- Commands: `python run_task_n_within_prompt_cost_benefit.py --mode all`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N1_latency_breakdown.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N1_routing_overhead.md`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N2_quality_projection.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N3_pareto.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/within_prompt/N4_verdict.md`.
- Results: verdict=DECLINE; d_q=0.0265621722; d_t_pct=0.0084275048.

## 2026-05-28 11:40 - task-p-long-context-validation-8k

- Purpose: validate locked ridge + static non-contiguous recipe at 8k context.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_30x8k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/router/locked_ridge_static_sets_for_router.json`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single RTX 4090 D.
- Commands: `python run_task_p_long_context.py --mode prepare/capture/eval_locked/transfer/latency/verdict/update_docs`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P2_fixed_k_8k_locked.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P3_layer_set_transfer.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P4_latency_curve_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P5_verdict.md`.
- Results: verdict=REGRESSION; NLL_8k_lockedI=1.4751339048; saving_8k_k12=0.3256007520.

## 2026-05-28 - task-q-long-context-regression-root-cause-diagnosis

- Purpose: diagnose why Task P long-context 8k validation regressed without new training or new trace capture.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_30x8k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_ridge_plus_residual_2_14_e5/residual_2_14.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P3_layer_set_overlap.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single RTX 4090 D.
- Commands: `python run_task_q_long_context_diagnosis.py --mode q1/q2/q3/q4/q5/q6`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q1_residual_at_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q2_per_position_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q3_per_layer_error_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q4_bootstrap_ci.json`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q5_kv_stats_512_vs_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q6_diagnosis_summary.md`.
- Results: H1=REFUTED; H3=H3_CONFIRMED_MONOTONIC_POSITION_EFFECT; H4=CONFIRMED; H5=REFUTED; repair=R2: position-aware Patch.

## 2026-05-28 - task-q-long-context-regression-root-cause-diagnosis

- Purpose: diagnose why Task P long-context 8k validation regressed without new training or new trace capture.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_30x8k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_128x512_ridge_plus_residual_2_14_e5/residual_2_14.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P3_layer_set_overlap.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, single RTX 4090 D.
- Commands: `python run_task_q_long_context_diagnosis.py --mode q1/q2/q3/q4/q5/q6`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q1_residual_at_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q2_per_position_nll.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q3_per_layer_error_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q4_bootstrap_ci.json`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q5_kv_stats_512_vs_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Q6_diagnosis_summary.md`.
- Results: H1=REFUTED; H3=H3_CONFIRMED_MONOTONIC_POSITION_EFFECT; H4=CONFIRMED; H5=REFUTED; repair=R2: position-aware Patch.

## 2026-05-28 - task-r-position-aware-patch-repair

- Purpose: repair Task P/Q 8k regression using position-aware or capacity-added Patch variants without new trace capture.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_30x8k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/ridge_residual/ridge_baseline.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P3_layer_set_overlap.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, PyTorch, single RTX 4090 D.
- Commands: `python run_task_r_position_aware_patch.py --mode r1/r3_onehot/r3_sincos/r2_train/r4`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R1_per_block_ridge.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R2_residual_mlp_at_8k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R3_position_feature.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R4_comparison.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R4_verdict.md`.
- Results: verdict=REPAIR_FOUND; best_variant=R1_per_block_ridge; best_R=0.1664818183.

## 2026-05-28 17:38 - task-s-long-context-pareto-16k-validation

- Purpose: complete 8k per-block Pareto, choose block count, capture/evaluate 16k traces, and decide scaling verdict.
- Inputs: `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_30x8k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/R1_per_block_ridge.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/P3_layer_set_overlap.json`.
- Environment: ser, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, HF Transformers, PyTorch, single GPU.
- Commands: `python run_task_s_long_context.py --mode s1/s2/prepare16/capture16/s3/s4_s5_docs`.
- Outputs: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/S1_pareto_8k_per_block.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/S2_block_count_ablation.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/experiments/qwen25_1p5b_to_7b_longbench_20x16k/paired_traces.pt`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/S3_pareto_16k.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/S4_context_scaling.csv`, `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/S5_verdict.md`.
- Results: verdict=16K_REGRESSION; NLL_16k=0.1329666015; saving_16k=0.2602586038; best_B=8.


## 2026-05-31 - Patch Method Repair and 200-sample QA Evaluation

- Task: patch-method-repair-final-safe6-200
- Purpose: 修正 Patch 方法并在 10% F1 损失门槛内尽量加速，最终用 200 条 QA-complete 16k 样本验证。
- Inputs: V_200x16k_eval_per_sample.csv, W_qa_per_block_ridge_train40_B8.pt, Qwen2.5-1.5B/7B local models.
- Methods: QA-complete per-block ridge refit; single-layer Patch sensitivity scan; targeted zero-init residual MLP with KV-MSE and KV-MSE+cosine; final safe6 route patch layers [4,6,7,21,25,26].
- Environment: ser, /disk_n/c101-3-mql/projects/scd-tpami, venv /disk_n/c101-3-mql/venvs/A800, 4x RTX 4090D.
- Exact commands: see Experiment/analysis/long_context/W_launch_final_safe6_200.sh and W_launch_residual_train.sh.
- Outputs: Experiment/analysis/long_context/W_patch_method_repair_final_summary.md; W_patch_route_repair_final_safe6_200_metrics.json; W_patch_route_repair_final_safe6_200_per_sample.csv; W_patch_method_route_candidates.csv.
- Results all_200: full F1 0.4739, mixed F1 0.4883, full EM 0.3450, mixed EM 0.3600, full 2110.62 ms, mixed 2449.72 ms, saving -21.99%.
- Results holdout_40_199: full F1 0.4921, mixed F1 0.5073, saving -23.62%.
- Conclusion: quality gate PASS, speed gate FAIL. Current ridge/per-block/residual Patch method cannot accelerate QA generation under <=10% F1 loss because quality-preserving route still requires 22 exact layers.
- Next action: pivot Patch construction to cross-layer/decode-aware training for unsafe layers or reduce exact-layer recompute overhead before making speed claims.


## 2026-05-31 - Task W: Patch Optimization Benchmark

- Purpose: targeted Patch-method benchmark for unsafe 16k QA layers, with fixed 200 QA-complete evaluation set.
- Inputs: V_200x16k QA-complete per-sample baseline, W_qa_per_block_ridge_train40_B8.pt, residual MSE adapters, safe route from prior Task W repair.
- Methods: per-block ridge extension, dense cross-layer ridge local_3/local_5/local_7, layer-specific direct-KV residual adapter, schema check for xnorm/native-KV, and final 200-sample safe+layer24 route.
- Outputs: W_patch_optimization_summary.md, W_patch_optimization_quick_screen.csv, W_patch_optimization_200eval.csv, W_patch_optimization_per_sample.csv, W_patch_optimization_best_route.json, W_patch_optimization_debug_samples.md.
- Result: PASS-B. Layer 24 can be patched under per-block ridge while preserving F1; layers 5/20/22/23/27 remain exact. Best route F1=0.4783, EM=0.3400, total=2589.53 ms, saving=-28.54%, exact_count=21.
- Conclusion: quality gate passes but speed remains negative; continue targeted compact cross-layer/decode-aware Patch for unsafe layers.


## 2026-05-31 - Task W: Patch Optimization Benchmark

- Purpose: targeted Patch-method benchmark for unsafe 16k QA layers, with fixed 200 QA-complete evaluation set.
- Inputs: V_200x16k QA-complete per-sample baseline, W_qa_per_block_ridge_train40_B8.pt, residual MSE adapters, safe route from prior Task W repair.
- Methods: per-block ridge extension, dense cross-layer ridge local_3/local_5/local_7, layer-specific direct-KV residual adapter, schema check for xnorm/native-KV, and final 200-sample safe+layer24 route.
- Outputs: W_patch_optimization_summary.md, W_patch_optimization_quick_screen.csv, W_patch_optimization_200eval.csv, W_patch_optimization_per_sample.csv, W_patch_optimization_best_route.json, W_patch_optimization_debug_samples.md.
- Result: FAIL. Layer 24 can be patched under per-block ridge while preserving F1; layers 5/20/22/23/27 remain exact. Best route F1=0.4783, EM=0.3400, total=2589.53 ms, saving=-28.54%, exact_count=21.
- Conclusion: quality gate passes but speed remains negative; continue targeted compact cross-layer/decode-aware Patch for unsafe layers.


## 2026-05-31 - Task X compact pooled cross-layer residual Patch follow-up

- Purpose: test whether lightweight pooled cross-layer residual adapters can repair unsafe layers 5, 20, 22, and 23 for 16k QA-complete evaluation.
- Inputs: W QA per-block ridge cache; 200 QA-complete 16k records; checkpoint `X_pooled_residual_local3_5_20_22_23_train40.pt`.
- Method: local_3 pooled source windows, zero-init residual MLP, train_n=40, epochs=2, quick screen n=10.
- Result: all four candidates failed quality. Mixed F1: layer5=0.0000, layer20=0.0011, layer22=0.0000, layer23=0.0011; full-large F1 on same subset=0.5105.
- Conclusion: pooled cross-layer residual does not repair these unsafe layers. Keep them exact/recompute; only layer 24 remains newly patchable from Task W.
- Outputs: `Experiment/analysis/long_context/X_compact_patch_followup_summary.md`, `X_pooled_local3_quick_n10_summary.csv`, `X_pooled_local3_quick_n10_per_sample.csv`.


## 2026-05-31 23:21:39 - task-y-xnorm-to-kv-patch-followup

- Purpose: test a structural Patch variant using small xnorm hidden features to repair unsafe 16k QA layers beyond the safe per-block ridge route.
- Inputs: 200 QA-complete 16k evaluation set; base per-block ridge cache `Experiment/analysis/long_context/W_qa_per_block_ridge_train40_B8.pt`; xnorm-to-KV ridge cache `Experiment/analysis/long_context/Y_xnorm_to_kv_ridge_5_20_22_23_train40.pt`.
- Method: quick-screen combined xnorm routes, then evaluate best route `Y_24_5_22` on all 200 QA samples using two GPU shards.
- Best route: exact layers `[0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,23,27]`; patch layers `[4,5,6,7,21,22,24,25,26]`; xnorm-to-KV layers `[5,22]`; base per-block ridge layers `[4,6,7,21,24,25,26]`.
- Results: full large F1/EM/total = 0.4739 / 0.3450 / 2110.62 ms; best mixed F1/EM/total = 0.4439 / 0.3200 / 2195.72 ms; F1 retention = 93.68%; saving vs full large = -10.35%.
- Verdict: PASS-B. Quality gate passes and exact layer count drops from 22 to 19, but wall-clock speed is still negative.
- Outputs: `Experiment/analysis/long_context/Y_xnorm_to_kv_Y24522_200_summary.md`, `Experiment/analysis/long_context/Y_xnorm_to_kv_Y24522_200_main_table.csv`, `Experiment/analysis/long_context/Y_xnorm_to_kv_Y24522_200_per_sample.csv`, `Experiment/analysis/long_context/Y_xnorm_to_kv_Y24522_200_best_route.json`.
- Next action: do not promote this as the final speed recipe; either optimize hidden-state/xnorm extraction overhead or move to a cheaper targeted decode-aware/native projection adapter before another 200-sample run.


## 2026-05-31 23:56:20 - task-z-four-gpu-targeted-patch-followup

- Purpose: use four GPUs to test cheaper targeted direct-KV residual adapters for unsafe layers after Task Y passed quality but not speed.
- Method: train/evaluate layer-specific residuals for layers 20, 23, 27, and 22 using the QA-complete 16k setup; gate on n=40 before any 200-sample promotion.
- Results: layer 20 diverged during training; layer 23 and layer 22 collapsed during early n=40 eval; layer 27 completed n=40 but failed quality with F1 retention 0.7184.
- Decision: no direct-residual candidate promoted to 200; best available remains Task Y xnorm-to-KV route with quality pass but negative speed.
- Output: `/disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/Z_four_gpu_targeted_patch_followup_summary.md`.


## 2026-06-01 - AA native-projection targeted Patch gate

- Purpose: test whether predicting large xnorm and using native Qwen2.5-7B k_proj/v_proj+RoPE can safely patch unsafe layers.
- Inputs: 200 QA-complete 16k sample pipeline; first 40 samples for gate; base per-block ridge; targets [5,20,22,23,24,27].
- Commands: run_task_aa_native_projection_patch.py fit/eval on four GPUs, split into [5,20,22] and [23,24,27].
- Outputs: Experiment/analysis/long_context/AA_native_projection_quick_screen_summary.csv, Experiment/analysis/long_context/AA_native_projection_targeted_patch_summary.md, AA_native_projection_A/B_*, AA_native_projection_layer5/23_*.
- Results: quality passed for layers 5/20/22/23/24; layer 27 failed. Accurate single-layer timing remained slower than full large: layer5 saving -14.38%, layer23 saving -16.84%.
- Conclusion: native projection is stable but too expensive as currently implemented; do not run 200-sample final for this branch. Next experiment should use cheaper direct xnorm-to-KV or compact cross-layer ridge.


## 2026-06-01 - AB Patch runtime path audit

- Purpose: diagnose why all Patch variants appeared slow on 16k QA-complete eval.
- Result: confirmed harness issue. Sparse routes compute all 28 Patch layers and materialize all 28 predicted KV layers before overwriting exact layers. Microbenchmark on one 11719-token sample: Patch compute 180.67 ms -> 36.49 ms with subset compute; materialization 33.99 ms -> 8.59 ms.
- Additional issue: non-contiguous exact/recompute speed currently uses len(exact_layers) as prefix depth, which is not a deployable model for anchors like [0,14,22,27].
- Output: Experiment/analysis/long_context/AB_patch_runtime_path_audit.md
- Next: repair evaluation harness before making speed claims.


## 2026-06-01 - AC corrected sparse Patch n=40 eval

- Purpose: rerun sparse Patch speed/quality after fixing full-28 Patch overcompute.
- Inputs: first 40 QA-complete 16k samples; routes safe and ybest.
- Outputs: Experiment/analysis/long_context/AC_corrected_sparse_n40_per_sample.csv, Experiment/analysis/long_context/AC_corrected_sparse_n40_summary.csv, Experiment/analysis/long_context/AC_corrected_sparse_patch_eval_summary.md.
- Result: subset Patch reduces Patch compute, but deployable conservative recompute remains negative because exact layers include high anchors up to 27. Count-proxy is not a deployable speed claim.
- Next: revise route constraint or implement real selective recompute before 200-sample final.


## 2026-06-01 - AC prefix-deployable early stop

- Purpose: test deployable prefix-exact routes after sparse Patch compute fix.
- Routes: prefix12_base, prefix12_y, prefix16_y, prefix20_y.
- Result: early samples had near-zero F1 across all prefix routes; jobs stopped early.
- Output: Experiment/analysis/long_context/AC_prefix_deployable_early_stop.md
- Conclusion: current Patch cannot replace broad upper-layer suffixes under prefix-exact deployment. Non-contiguous high-layer anchors are quality-critical but require runtime support.

## 2026-06-01 - AD prefix-deployable xnorm ridge screening

- Purpose: test whether direct small-xnorm to large-KV per-block ridge over layers 12-27 can make prefix-deployable routes viable.
- Inputs: AD_prefix_xnorm_12_19_train40.pt, AD_prefix_xnorm_20_27_train40.pt, merged into AD_prefix_xnorm_12_27_train40.pt; QA-complete 16k samples 0-39.
- Commands: fit jobs ran on GPU pairs 0/1 and 2/3; evaluation ran run_task_ac_corrected_sparse_patch_eval.py with routes prefix12/16/20 xnorm-all.
- Outputs: AD_prefix_xnorm_deployable_n40_per_sample.csv, AD_prefix_xnorm_deployable_n40_summary.csv, AD_prefix_xnorm_deployable_summary.md.
- Results:
  - prefix12_xall: F1 retention=0.1165, mixed F1=0.0467, EM=0.0000, deploy saving=-74.59%.
  - prefix16_xall: F1 retention=0.0320, mixed F1=0.0128, EM=0.0000, deploy saving=-60.30%.
  - prefix20_xall: F1 retention=0.2493, mixed F1=0.0999, EM=0.0500, deploy saving=-39.76%.
- Conclusion: FAIL screening; do not proceed to 200 samples for this prefix-deployable xnorm-ridge Patch route.

## 2026-06-01 - AB combined native-projection route screen

- Purpose: test whether individually passing native-projection Patch layers 5/20/22/23/24/27 compose into a route-level mixed-KV policy.
- Inputs: AA_native_projection_A_5_20_22_train40.pt, AA_native_projection_B_23_24_27_train40.pt; QA-complete 16k samples 0-39.
- Outputs: AB_native_combined_n40_per_sample.csv, AB_native_combined_n40_summary.csv, AB_native_projection_combined_summary.md.
- Result AA_5_20_22_23_24: F1 retention=0.8137, mixed F1=0.3261, EM=0.2000, deploy saving=-53.41%.
- Result AA_5_20_22_23_24_27: F1 retention=0.3643, mixed F1=0.1460, EM=0.0500, deploy saving=-57.72%.
- Conclusion: FAIL screening; do not run 200 samples for combined native-projection route.

## 2026-06-01 - AB greedy native-projection combination screen

- Purpose: isolate whether smaller native-projection layer combinations compose safely after the all-layer combination failed.
- Inputs: AA native-projection payloads; QA-complete 16k samples 0-39.
- Outputs: AB_native_greedycombo_n40_per_sample.csv, AB_native_greedycombo_n40_summary.csv, AB_native_projection_greedycombo_summary.md.
- Result AB_g2_5_22: F1 retention=1.0516, mixed F1=0.4215, EM=0.3250, deploy saving=-40.60%.
- Result AB_g4_5_20_22_23: F1 retention=0.8601, mixed F1=0.3447, EM=0.2250, deploy saving=-53.43%.
- Result AB_g2_22_23: F1 retention=0.8588, mixed F1=0.3442, EM=0.2500, deploy saving=-44.30%.
- Result AB_g3_5_22_23: F1 retention=0.8283, mixed F1=0.3320, EM=0.2250, deploy saving=-43.31%.
- Result AB_g3_20_22_23: F1 retention=0.8193, mixed F1=0.3284, EM=0.2000, deploy saving=-56.11%.
- Conclusion: FAIL screening; do not run 200 samples for native-projection combinations.

## 2026-06-01 - AE layer27 targeted native-projection repair

- Purpose: test whether larger held-out closed-form native-projection fits for layer 27 remove the final exact anchor.
- Method: fit layer 27 on QA-complete samples 40-119 with tokens_per_block=512 and ridge lambda in {0.1, 1.0}; evaluate n=40 on samples 0-39 for 27-only and 5+22+27 routes.
- Outputs: AE_native27_lam01_n40_summary.csv, AE_native27_lam1_n40_summary.csv, AE_layer27_targeted_repair_summary.md.
- Result AE_27only_lam01: retention=0.7485, mixed F1=0.3000, EM=0.2000, deploy saving=-43.98%.
- Result AE_5_22_27_lam01: retention=0.8113, mixed F1=0.3252, EM=0.2000, deploy saving=-38.36%.
- Result AE_27only_lam1: retention=0.7485, mixed F1=0.3000, EM=0.2000, deploy saving=-45.70%.
- Result AE_5_22_27_lam1: retention=0.8112, mixed F1=0.3251, EM=0.2000, deploy saving=-40.05%.
- Conclusion: FAIL; do not run 200 samples for this layer27 repair.
## 2026-06-01 - AF layer27 decode-aware adapter quick screen

- Task: AF layer27 decode-aware adapter after W/AA/AB/AE showed layer 27 is the deploy-speed bottleneck.
- Purpose: test whether a small trained residual on top of native-projection ridge can make layer 27 patchable without running a full 200-sample evaluation.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: 4x available; training/eval run in two parallel 2-GPU jobs.
- Inputs:
  - QA-complete 16k data and baseline: `Experiment/analysis/long_context/V_200x16k_eval_per_sample.csv`
  - Native ridge payloads: `AA_native_projection_A_5_20_22_train40.pt`, `AE_native27_s40_n80_tok512_lam01.pt`
  - Corrected sparse evaluation path: `run_task_ab_native_projection_combined_eval.py`
- Method:
  - Implemented `run_task_af_layer27_decode_adapter.py`.
  - Adapter: zero-init xnorm residual for layer 27.
  - Loss: `KL + 0.1 * first-answer-token CE + 0.01 * sampled KV MSE`.
  - Trained on sample ids 40-59, evaluated on sample ids 0-39.
- Commands:
  - Smoke: `CUDA_VISIBLE_DEVICES=0,1 python Experiment/core_code/run_task_af_layer27_decode_adapter.py train --device cuda:0 --small-device cuda:1 --start 40 --train-n 1 --epochs 1 --out Experiment/analysis/long_context/AF_layer27_decode_adapter_smoke.pt`
  - e1 train: `CUDA_VISIBLE_DEVICES=0,1 python Experiment/core_code/run_task_af_layer27_decode_adapter.py train --device cuda:0 --small-device cuda:1 --start 40 --train-n 20 --epochs 1 --out Experiment/analysis/long_context/AF_layer27_decode_adapter_s40_n20_e1.pt`
  - e2 train: `CUDA_VISIBLE_DEVICES=2,3 python Experiment/core_code/run_task_af_layer27_decode_adapter.py train --device cuda:0 --small-device cuda:1 --start 40 --train-n 20 --epochs 2 --out Experiment/analysis/long_context/AF_layer27_decode_adapter_s40_n20_e2.pt`
  - n=40 eval: `python Experiment/core_code/run_task_ab_native_projection_combined_eval.py --start 0 --n 40 --native-patches AA_native_projection_A_5_20_22_train40.pt,AE_native27_s40_n80_tok512_lam01.pt --af-adapters <adapter.pt> --routes <AF_27only;AF_5_22_27> --tag AF_decode_adapter_<e1|e2>_n40`
- Results:
  - Best route: e1 `AF_5_22_27`.
  - Full large F1: 0.4008.
  - Mixed F1: 0.3232.
  - F1 retention: 0.8063.
  - EM: 0.200.
  - Deploy total: 2916.81 ms.
  - Deploy saving: -39.13%.
- Decision:
  - FAIL quick-screen gate.
  - No 200-sample evaluation.
  - Layer 27 remains unsafe under this adapter; current high-layer-anchor non-contiguous route remains non-deployable for wall-clock speed.
- Outputs:
  - `Experiment/analysis/long_context/AF_layer27_decode_adapter_summary.md`
  - `Experiment/analysis/long_context/AF_decode_adapter_e1_n40_summary.csv`
  - `Experiment/analysis/long_context/AF_decode_adapter_e1_n40_per_sample.csv`
  - `Experiment/analysis/long_context/AF_decode_adapter_e2_n40_summary.csv`
  - `Experiment/analysis/long_context/AF_decode_adapter_e2_n40_per_sample.csv`
## 2026-06-01 - AG prefix-depth constrained route benchmark

- Task: AG prefix-depth constrained route benchmark after AF layer27 adapter failed.
- Purpose: test whether wall-clock speed can be recovered by forcing lower deploy-prefix depth, instead of allowing high-layer exact anchors.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: 4x available; native27 and AF27 variants run in parallel 2-GPU jobs.
- Inputs:
  - QA-complete 16k baseline: `Experiment/analysis/long_context/V_200x16k_eval_per_sample.csv`
  - Native payloads: `AA_native_projection_A_5_20_22_train40.pt`, `AA_native_projection_B_23_24_27_train40.pt`, `AE_native27_s40_n80_tok512_lam01.pt`
  - AF adapter: `AF_layer27_decode_adapter_s40_n20_e1.pt`
- Routes:
  - `AG_cap24`: deploy prefix depth 24.
  - `AG_cap23`: deploy prefix depth 23.
  - `AG_cap21`: deploy prefix depth 21.
  - `AG_cap20`: deploy prefix depth 20.
  - `AG_cap18`: deploy prefix depth 18.
- Results:
  - Best native route: `AG_cap24`, F1 0.3285, EM 0.225, F1 retention 0.8195, deploy total 3358.96 ms, saving -75.73%.
  - Best AF route: `AG_cap24`, F1 0.3126, EM 0.200, F1 retention 0.7800, deploy total 3380.69 ms, saving -75.86%.
  - Lower caps reduce deploy depth but collapse quality: `AG_cap20` native retention 0.4487; `AG_cap18` native retention 0.0073.
- Decision:
  - FAIL quick-screen gate.
  - No 200-sample evaluation.
  - Route search under deploy-prefix-depth constraints confirms the remaining bottleneck is Patch construction for upper/high-risk layers, not just route selection.
- Outputs:
  - `Experiment/analysis/long_context/AG_prefix_depth_constrained_summary.md`
  - `Experiment/analysis/long_context/AG_prefix_depth_native27_n40_summary.csv`
  - `Experiment/analysis/long_context/AG_prefix_depth_native27_n40_per_sample.csv`
  - `Experiment/analysis/long_context/AG_prefix_depth_af27_n40_summary.csv`
  - `Experiment/analysis/long_context/AG_prefix_depth_af27_n40_per_sample.csv`
## 2026-06-01 - AH upper-layer joint decode-aware adapters

- Task: AH upper-layer joint decode-aware adapters after AG prefix-depth constrained routes failed.
- Purpose: test whether jointly trained residual adapters for upper/high-risk layers can preserve QA quality under prefix-depth constrained routes.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: 4x available; e1/e2 trained and evaluated in parallel 2-GPU jobs.
- Inputs:
  - QA-complete 16k baseline: `Experiment/analysis/long_context/V_200x16k_eval_per_sample.csv`
  - Native payloads: `AA_native_projection_A_5_20_22_train40.pt`, `AA_native_projection_B_23_24_27_train40.pt`, `AE_native27_s40_n80_tok512_lam01.pt`
  - Corrected sparse mixed-cache evaluation path: `run_task_ab_native_projection_combined_eval.py`
- Method:
  - Implemented `run_task_ah_upper_joint_decode_adapters.py`.
  - Target layers: `[20, 22, 23, 24, 27]`.
  - Adapter: per-layer zero-init xnorm residual.
  - Joint loss: `KL + 0.1 * first-answer-token CE + 0.01 * sampled KV MSE`.
  - Training samples: ids 40-59. Eval samples: ids 0-39.
- Results:
  - Best route: `AH_cap24`, e2.
  - Full large F1: 0.4008.
  - Mixed F1: 0.3398.
  - EM: 0.200.
  - F1 retention: 0.8477.
  - Deploy total: 3113.28 ms.
  - Deploy saving: -62.95%.
  - Lower-depth routes collapsed: e2 `AH_cap21` retention 0.1136, `AH_cap20` retention 0.1997.
- Decision:
  - FAIL quick-screen gate.
  - No 200-sample evaluation.
  - Joint shallow xnorm residuals improve slightly over AG/AF but remain non-deployable.
- Outputs:
  - `Experiment/analysis/long_context/AH_upper_joint_decode_adapter_summary.md`
  - `Experiment/analysis/long_context/AH_upper_joint_e1_n40_summary.csv`
  - `Experiment/analysis/long_context/AH_upper_joint_e1_n40_per_sample.csv`
  - `Experiment/analysis/long_context/AH_upper_joint_e2_n40_summary.csv`
  - `Experiment/analysis/long_context/AH_upper_joint_e2_n40_per_sample.csv`

## 2026-06-01 - AI cross-layer xnorm native Patch quick screen

- Task: AI cross-layer xnorm native Patch quick screen after AF/AG/AH failures.
- Purpose: test whether cross-layer small xnorm features can improve unsafe-layer Patch construction enough to reduce exact/recompute depth.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: 4x available; radius-1 and radius-2 variants evaluated in parallel 2-GPU jobs.
- Inputs:
  - QA-complete 16k baseline: `Experiment/analysis/long_context/V_200x16k_eval_per_sample.csv`
  - Script: `Experiment/core_code/run_task_ai_cross_layer_native_patch.py`
- Method:
  - Fitted cross-layer xnorm-native ridge caches on samples 40-59.
  - `local_3_r1`: radius-1 small-layer xnorm pooling.
  - `local_5_r2`: radius-2 small-layer xnorm pooling.
  - Evaluated deploy-prefix constrained routes `AI_cap24`, `AI_cap23`, `AI_cap21`, `AI_cap20`, `AI_cap18` on samples 0-39.
- Results:
  - Best retention: `local_3_r1 / AI_cap24`, mixed F1 0.1594, EM 0.075, retention 0.3976, deploy saving -106.53%.
  - Best deploy saving: `local_5_r2 / AI_cap23`, saving -95.47%, mixed F1 0.1581, retention 0.3945.
  - All variants failed F1 retention >= 0.90 and all remained slower than full large on deploy wall-clock.
- Decision:
  - FAIL quick-screen gate.
  - No 200-sample evaluation.
  - Cross-layer pooled xnorm ridge is not sufficient; next patch attempt should change the structural predictor rather than adding more shallow per-layer variants.
- Outputs:
  - `Experiment/analysis/long_context/AI_cross_layer_native_patch_summary.md`
  - `Experiment/analysis/long_context/AI_cross_layer_r1_n40_summary.csv`
  - `Experiment/analysis/long_context/AI_cross_layer_r1_n40_per_sample.csv`
  - `Experiment/analysis/long_context/AI_cross_layer_r2_n40_summary.csv`
  - `Experiment/analysis/long_context/AI_cross_layer_r2_n40_per_sample.csv`

## 2026-06-01 - AJ sparse Patch runtime audit

- Task: AJ sparse Patch runtime audit after repeated negative wall-clock results.
- Purpose: test whether mixed-route speed failures are partly caused by computing all Patch layers instead of only route-specific patch layers.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: large model on GPU0, small/Patch on GPU2; GPU1 had an unrelated ATIS job and was avoided.
- Inputs:
  - QA-complete 16k baseline: `Experiment/analysis/long_context/V_200x16k_eval_per_sample.csv`
  - Patch cache: `Experiment/analysis/long_context/W_qa_per_block_ridge_train40_B8.pt`
- Method:
  - Implemented `run_task_aj_sparse_patch_benchmark.py`.
  - Evaluated `safe_sparse` and `wbest_sparse` on samples 0-39.
  - Patch was computed only for each route’s `patch_layers`, not all 28 layers.
- Results:
  - `safe_sparse`: F1 0.4123, EM 0.325, retention 1.0287, total 2649.09 ms, saving -24.01%, patch 82.87 ms, recompute 1741.14 ms.
  - `wbest_sparse`: F1 0.4039, EM 0.325, retention 1.0076, total 2657.57 ms, saving -24.39%, patch 89.71 ms, recompute 1742.38 ms.
- Decision:
  - FAIL speed gate.
  - Sparse Patch compute reduces patch cost but does not solve speed because exact layer 27 forces deploy-prefix depth 28.
  - Next useful experiment must repair final/boundary layers, especially layer 27, or change serving to avoid prefix-depth recompute for non-contiguous exact anchors.
- Outputs:
  - `Experiment/analysis/long_context/AJ_sparse_patch_runtime_audit_summary.md`
  - `Experiment/analysis/long_context/AJ_sparse_patch_n40_summary.csv`
  - `Experiment/analysis/long_context/AJ_sparse_patch_n40_per_sample.csv`

## 2026-06-01 - AM boundary joint adapter quick screen

- Task: AM boundary joint adapter after AK/AL showed layer 27 alone can be patched but speed remains negative.
- Purpose: test whether jointly patching boundary layers `[23,24,27]` can lower deploy-prefix depth to 23/24 while preserving QA quality.
- Environment:
  - Project root: `/disk_n/c101-3-mql/projects/scd-tpami`
  - Env: `/disk_n/c101-3-mql/venvs/A800/bin/activate`
  - GPUs: h512 and h1024 trained/evaluated in parallel 2-GPU jobs.
- Method:
  - Trained h512 and h1024 joint adapters for layers 23/24/27 on samples 40-79 for 3 epochs.
  - Evaluated `AM_24_27` and `AM_23_24_27` on samples 0-39.
- Results:
  - h512 `AM_24_27`: F1 0.3178, EM 0.225, retention 0.7930, total 2661.25 ms, saving -24.97%.
  - h512 `AM_23_24_27`: F1 0.3220, EM 0.225, retention 0.8034, total 2547.99 ms, saving -20.88%.
  - h1024 `AM_24_27`: F1 0.3003, EM 0.200, retention 0.7492, total 2764.62 ms, saving -31.53%.
  - h1024 `AM_23_24_27`: F1 0.3334, EM 0.250, retention 0.8319, total 2759.19 ms, saving -27.92%.
- Decision:
  - FAIL quality gate; no 200-sample evaluation.
  - Boundary joint adapter does not safely replace layers 23/24/27.
  - Next useful direction is runtime support for non-contiguous exact anchors or a structurally stronger sequence/block Patch, not another shallow boundary adapter.
- Outputs:
  - `Experiment/analysis/long_context/AM_boundary_joint_adapter_summary.md`
  - `Experiment/analysis/long_context/AM_boundary_h512_n40_summary.csv`
  - `Experiment/analysis/long_context/AM_boundary_h512_n40_per_sample.csv`
  - `Experiment/analysis/long_context/AM_boundary_h1024_n40_summary.csv`
  - `Experiment/analysis/long_context/AM_boundary_h1024_n40_per_sample.csv`


## 2026-06-01 - AO xnorm-to-KV layer27 deploy-aware gate
- Purpose: test whether layer27 can be patched by direct small-xnorm to large-KV per-block ridge, enabling lower deploy prefix depth.
- Inputs: QA-complete 16k n=40 gate; Task Y xnorm payload for [5,20,22,23]; new layer27 ridge payloads with lambda 0.1 and 1.0.
- Commands: run_task_y_xnorm_ridge_quick.py fit --target-layers 27; run_task_aj_sparse_patch_benchmark.py --xnorm-kv-patches.
- Result: best route AO_lam01_24_27 F1=0.3290, retention=0.8210, deploy saving=-55.68%.
- Conclusion: FAIL; no AO route meets the 0.90 F1 retention gate or positive deploy saving. Do not run 200 samples.
- Outputs: Experiment/analysis/long_context/AO_xnorm_layer27_deploy_summary.md, AO_lam01_xnorm27_deploy_n40_summary.csv, AO_lam1_xnorm27_deploy_n40_summary.csv.


## 2026-06-01 - AP temporal-pooled xnorm layer27 Patch gate
- Purpose: test whether local token context fixes layer27 Patch enough to lower deploy prefix depth.
- Method: per-block ridge from concat(center small xnorm, local average small xnorm) to large KV for layer27; radii 8 and 32.
- Inputs: QA-complete 16k samples 0-39 for eval, 40-119 for fit; Task Y direct xnorm payload for optional layers 5/22/23.
- Result: best route AP_r8_24_23_27 F1=0.3190, retention=0.7959, deploy saving=-39.61%.
- Conclusion: FAIL; temporal pooled xnorm does not meet quality or speed gate. Do not run 200 samples.
- Outputs: Experiment/analysis/long_context/AP_temporal_xnorm_layer27_summary.md, AP_temporal_r8_deploy_n40_summary.csv, AP_temporal_r32_deploy_n40_summary.csv.


## 2026-06-01 - AR hidden-bridge Patch smoke
- Purpose: evaluate whether predicting a large bridge hidden state can repair upper/final unsafe layers without running a full 200-sample benchmark.
- Inputs: QA-complete 16k records, V baseline CSV, Qwen2.5-1.5B/7B, existing scoring/decode path.
- Commands: run_task_ar_hidden_bridge_patch.py fit_bridge/eval_bridge for bridge20 hidden, bridge20 xnorm, bridge24 xnorm, plus bridge20 oracle-hidden closure.
- Results: oracle-hidden bridge20 retention 1.0000; learned bridge20 hidden retention 0.2590; learned bridge20 xnorm retention 0.3584; learned bridge24 xnorm retention 0.4966.
- Decision: stop AR hidden-bridge Patch at smoke scale; do not expand to n=40/200.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/AR_hidden_bridge_smoke_summary.md.


## 2026-06-02 - AS sequence adapter for layers 24/27
- Purpose: test whether a targeted sequence residual adapter can patch boundary layers 24 and 27 while preserving QA F1 and reducing recompute depth.
- Method: xnorm Conv1D residual adapter, zero-init output, base = QA per-block ridge, target layers [24,27], train n=20, epochs=2, chunk_len=512, hidden=512, kernel=9.
- n=40 result: AS_safe retention 1.0287, saving -22.77%, deploy depth 28; AS_plus24_27 retention 0.9083, saving -27.57%, deploy depth 24.
- Analysis: recompute reduced by 251.12 ms, but patch compute increased by 43.08 ms and decode increased by 357.79 ms; mean decode length doubled from 16.90 to 34.35.
- Decision: do not expand AS_plus24_27 to 200 under current speed objective; it is quality-positive but speed-negative.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/AS_sequence_adapter_24_27_summary.md.


## 2026-06-02 - AT decode-control ablation
- Purpose: test whether decode latency changes should also affect baseline and whether stop-policy control rescues +24+27 sequence adapter speed.
- Methods: full_large, mixed_safe, mixed_plus24_27 on n=40 with eos32/eos64/eos128/newline128/fixed64.
- Results: fixed64 raises full_large total to 3040.71 ms, confirming the baseline also grows under fixed decode. Under eos128, +24+27 F1 0.3640, EM 0.2750, total 2421.60 ms, saving -7.22%; safe mixed total 2552.78 ms. Under newline128, +24+27 saving -12.95%.
- Decision: +24+27 is best quality-preserving Patch repair so far, but still not a positive-speed route; do not run 200 as a speed claim yet.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/AT_decode_control_ablation_summary.md.


## 2026-06-02 - AU overhead reduction depth-21 diagnostic
- Purpose: test whether +24+27 can become positive-speed by additionally patching layers 22/23 to reduce deploy depth.
- Method: expanded sequence residual adapter on [22,23,24,27], then base residual scale diagnostic.
- Result: training diverged around 1e15 loss. Base per-block ridge predictions for layers 20/22/23 are numerically exploded: layer20 pred std 1.38e6, layer22 8.63e4, layer23 8.55e7, while target std is ~1.4-1.7. Layers 24/27 are sane.
- Decision: stop residual-on-ridge depth21 route; further speed work requires direct/native predictor for 20/22/23 or non-Patch runtime reductions.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/AU_overhead_reduction_depth21_diagnostic.md.


## 2026-06-02 - AV direct sequence predictor for 20/22/23
- Purpose: replace exploded ridge base for layers 20/22/23 with direct xnorm sequence-to-KV predictor and test depth21/depth20 routes.
- Method: Conv1D sequence direct KV adapter, target layers [20,22,23], train n=20, epochs=2; combine with AS residual adapter for [24,27].
- Training: stable, mean loss about 1.47 after epoch 2.
- Results: AV_22_23_24_27 n=10 retention 0.1119, saving -39.81%; AV_20_22_23_24_27 retention 0.0, saving -31.52%.
- Decision: stop AV before n=40/200; layers 22/23 remain unpatchable with current direct/native sequence predictors.
- Outputs: /disk_n/c101-3-mql/projects/scd-tpami/Experiment/analysis/long_context/AV_direct_sequence_20_22_23_summary.md.


## 2026-06-02 - AW/W final 200 QA-complete Patch optimization confirmation
- Purpose: confirm on 200 QA-complete 16k samples whether the current best Patch-optimized route improves speed while preserving F1/EM.
- Methods: full_large, small_direct from V baseline, safe-layer mixed route, and sequence-adapter +24+27 mixed route under eos128 decode control.
- Inputs: V_200x16k QA-complete records; AS_seqres_24_27_train20_e2_h512_k9.pt; safe route patch layers [4, 6, 7, 21, 25, 26]; +24+27 route patch layers [4, 6, 7, 21, 24, 25, 26, 27].
- Results: full_large F1=0.4742, EM=0.345, total=2109.25 ms; safe_mixed F1=0.4883, EM=0.360, total=2608.51 ms, saving=-23.67%; plus24_27 F1=0.3576, EM=0.245, total=2586.82 ms, saving=-22.64%.
- Decision: FAIL for deployable Patch optimization. +24+27 lowers exact count but fails F1 retention; safe route remains quality-valid but speed-negative.
- Outputs: Experiment/analysis/long_context/W_patch_optimization_summary.md, W_patch_optimization_200eval.csv, W_patch_optimization_per_sample.csv, W_patch_optimization_best_route.json, W_patch_optimization_debug_samples.md, AW_final_200_eos128_summary.csv.


## 2026-06-02 - AX joint native-sequence Patch for layers 22/23
- Purpose: test a stronger cross-layer native-state Patch for unsafe layers 22/23 after per-block ridge, residual-on-ridge, direct sequence KV, and cross-layer ridge failures.
- Method: concat small xnorm source layers [20, 21, 22, 23, 24, 25] -> Conv1D adapter -> predicted large xnorm for layers [22, 23] -> native Qwen2.5-7B k_proj/v_proj. Train n=20, eval n=40 QA-complete 16k.
- Results: AX_22 F1=0.2944, retention=0.7346; AX_23 F1=0.4532, retention=1.1307; AX_22_23 F1=0.1893, retention=0.4724. Deploy-relevant AX23_AS24_27 F1=0.3022, retention=0.7540.
- Decision: FAIL. Layer 23 can be patched in isolation, but layer 22 fails, and the AX23+AS24/27 combination fails the 0.90 F1-retention gate.
- Note: AX eval was quality-only because the 24GB GPU cannot hold AX tensors plus exact recompute timing; no wall-clock claim is made for AX.
- Outputs: Experiment/analysis/long_context/AX_joint_native_sequence_22_23_summary.md, AX_joint_native_seq_22_23_n40_quality_summary.csv, AX23_AS24_27_n40_quality_summary.csv.


## 2026-06-02 - AY layer22 sequence hidden-bridge Patch

- Purpose: test whether changing Patch construction to a sequence hidden-bridge predictor can repair unsafe large layer 22.
- Inputs: 200 QA-complete 16k dataset; train samples 40-59; eval samples 0-39; Qwen2.5-1.5B -> Qwen2.5-7B.
- Method: small xnorm layer 22 -> sequence adapter -> predicted large layer-22 hidden input; hook large model at layer 22 for quality smoke.
- Command: python Experiment/core_code/run_task_ay_layer22_sequence_bridge.py train ...; python Experiment/core_code/run_task_ay_layer22_sequence_bridge.py eval ...
- Result: n=40 full_large_f1=0.4008, mixed_f1=0.02875, mixed_em=0.0, f1_retention=0.0717, mean_decode_length=99.175.
- Conclusion: FAIL. Layer 22 remains unsafe; do not expand AY to 200 samples.
- Outputs: Experiment/analysis/long_context/AY_layer22_sequence_bridge_summary.md, AY_layer22_seq_bridge_xnorm_train20_n40_summary.csv, AY_layer22_seq_bridge_xnorm_train20_n40_per_sample.csv.


## 2026-06-03 - Task X targeted NativeKV / compact cross-layer Patch repair

- Purpose: test whether NativeKV/xnorm and compact cross-layer Patch variants can repair unsafe 16k QA layers and reduce exact/recompute layers.
- Inputs: QA-complete 200x16k records, Task W safe per-block ridge, Task Y/AA/AB/AX/AY/AZ quick-screen and 200-sample artifacts.
- Method: no new training in this assembly step; standardized existing compact residual, native projection, joint native sequence, layer22 bridge, and xnorm-to-KV results into Task X output format.
- Result: best route is Y_24_5_22 with patch layers [4,5,6,7,21,22,24,25,26], F1=0.4439, EM=0.3200, F1 retention=0.9368, total=2195.72 ms, saving=-10.35% vs full large.
- Conclusion: PASS-B local repair only. Layers 5 and 22 can be patched under the F1 gate, but layers 20/23/27 remain exact and no positive wall-clock saving is achieved.
- Outputs: Experiment/analysis/long_context/X_nativekv_patch_summary.md, X_nativekv_patch_quick_screen.csv, X_nativekv_patch_200eval.csv, X_nativekv_patch_per_sample.csv, X_nativekv_patch_best_route.json, X_nativekv_patch_debug_samples.md, X_nativekv_patch_trace_audit.md.


## 2026-06-04 - YB sequence hidden bridge smoke

- Purpose: continue Direction B after YA by replacing the failed linear upper-block hidden bridge with a nonlinear sequence residual adapter for bridge layer 24.
- Method: trained zero-init Conv1D sequence residual adapters on top of YA per-block bridge, using small layer-22 hidden and xnorm features separately; train samples 40-59 (n=20), epochs=3, hidden=512, kernel=9; eval samples 0-9 (n=10).
- Results: hidden feature F1 retention=0.4601, xnorm feature F1 retention=0.4835; both are below the n=10 promotion gate 0.80 and essentially match the YA linear bridge results.
- Conclusion: FAIL smoke gate. Do not promote YB to n=40 or n=200. Direction B requires a much larger teacher-student native-state phase with behavioral supervision if continued; otherwise proceed with speed-gate revision / negative-result framing.
- Outputs: Experiment/analysis/long_context/YB_sequence_hidden_bridge_summary.md; YB_bridge24_hidden_seqres_train20_n10_summary.csv; YB_bridge24_xnorm_seqres_train20_n10_summary.csv.

## 2026-06-04 - Task Y upper-block / route-level joint Patch with behavior loss

- Purpose: test whether joint behavior training can solve the non-compositional failure where single unsafe layers can sometimes be patched, but multi-layer routes collapse and late exact anchors keep runtime near full large.
- Scope: no prefix-k redesign, no prompt-level router, no dense concat cross-layer predictor, and no full-layer decode-aware MLP. The tested routes used repaired QA-complete 16k samples and behavior losses instead of KV-only reconstruction.
- Data and trace status: no stale missing-question traces were used. Hidden/xnorm/KV were obtained from online full forward with `output_hidden_states=True` and `use_cache=True`; teacher behavior came from full-large online logits.
- Upper-block evidence: reused AH upper-layer behavior-loss adapters and YB hidden/native-state bridge smoke. AH best n=40 route `AH_cap24` reached F1 retention 0.8477, below the 0.90 gate; YB bridge24 hidden/xnorm reached 0.4601/0.4835, below the 0.80 smoke gate.
- Route-level joint training: trained a route-level residual bank on previous-best NativeKV plus late anchors [20,23,27], with next-token KL, first-answer-token CE, top-k teacher-set loss, and sampled KV MSE. The suffix_len=16 variant OOMed, so the executable behavior-loss version used next-token teacher behavior plus sampled KV.
- n=40 results: `Y_prevbest_joint` retention 0.9611 but total 4022.86 ms and saving -101.11%; `Y_plus27_joint` retention 0.8812 and saving -73.38%; `Y_plus23_27_joint` retention 0.7122 and saving -60.70%; `Y_plus20_23_27_joint` retention 0.7236 and saving -55.23%.
- Decision: FAIL. Task Y did not repair layer20, layer23, or layer27 under the quick-screen gates, and no route reached robust positive saving. Do not promote Task Y to 200-sample expansion or full-layer stress testing.
- Outputs: `Experiment/analysis/long_context/Y_joint_patch_summary.md`, `Y_upper_block_quick_screen.csv`, `Y_route_level_quick_screen.csv`, `Y_joint_patch_per_sample.csv`, `Y_joint_patch_best_route.json`, `Y_joint_patch_debug_samples.md`, and `Y_joint_patch_negative_results.md`.
- Next action: stop the current lightweight Patch family for the 16k speed claim. Either write the speed-gate revision / negative-result section, or explicitly start a new larger teacher-student native-state Patch phase with more QA-complete traces, deeper architecture, and multi-token behavioral supervision.

## 2026-06-04 - Task Z full-layer joint Patch-only smoke

- Purpose: test the final-form 0-exact route: small full prefill, joint Patch generates all 28 Qwen2.5-7B native-KV layers, then large decode, with behavior loss rather than KV-only local loss.
- Scope: no ridge route search, no single-layer adapter tuning, no prompt-level router. Stage 0 is a debug/smoke run only and is not used as a formal quality conclusion.
- Implementation: added `remote_scripts/run_task_z_full_layer_joint_patch.py` and deployed it to `ser:/disk_n/c101-3-mql/projects/scd-tpami/Experiment/core_code/run_task_z_full_layer_joint_patch.py`. The Patch path is small xnorm -> shared compact trunk + layer-specific heads -> predicted large xnorm -> Qwen2.5-7B native k_proj/v_proj + RoPE -> full 28-layer KV cache.
- Compute note: full 16k route-level behavior-loss backprop OOMed on a 24GB 4090D even after removing the full true-KV cache from training. Stage 0 therefore used a 2048-token prompt-tail training cap while preserving the Question/Answer tail; full 16k inference was still attempted for eval.
- Data availability: the current QA-complete 16k builder exposes 200 valid records. The pre-registered Stage 1 split train=200/val=40/test=40 requires at least 280 de-duplicated QA-complete records, so Stage 1 was marked not runnable.
- Stage 0 setup: train samples 40-55 (n=16), eval samples 0-9 (n=10), epochs=2, rank=32, suffix_len=8, max_new_tokens=128, exact layer count=0, patch layer count=28.
- Results: training loss decreased from 4.0275 to 2.8073, but eval full-layer Patch-only F1=0.0000 and EM=0.0000. Full large on the same 10 eval samples was F1=0.5105 / EM=0.4000 / 2055.76 ms. Full-layer Patch-only total runtime averaged 2956.47 ms, saving=-99.82% versus full large.
- Decision: FAIL smoke. The optimization path can run and decrease loss, but it does not produce a usable generation signal and is slower than full large under the current implementation. Do not expand to Stage 2. Stage 1 requires additional QA-complete records and a memory-feasible behavior-loss training design before it can be fairly attempted.
- Outputs: `Experiment/analysis/long_context/Z_full_layer_joint_patch_summary.md`, `Z_full_layer_joint_patch_train_log.csv`, `Z_full_layer_joint_patch_eval_stage0.csv`, `Z_full_layer_joint_patch_eval_stage1.csv`, `Z_full_layer_joint_patch_eval_stage2.csv`, `Z_full_layer_joint_patch_per_sample.csv`, `Z_full_layer_joint_patch_best_checkpoint.json`, and `Z_full_layer_joint_patch_debug_samples.md`.


## 2026-06-04 - CUT cut-point cross-scale KV reuse gate

- Purpose: test a new cut-point scheme that predicts the Qwen2.5-7B layer-20 input hidden state, then would run upper layers from that hidden instead of recomputing a deep exact prefix.
- Method: implemented `run_task_cut_point.py` with modes `sanity_oracle`, `fit_base_bridge`, `train_predictor`, and `eval_quality_upperbound`. Reused `run_task_ar_hidden_bridge_patch.py` for oracle hidden capture and full-forward quality proxy.
- Compute preflight: `ser` reachable; GPU0/2/3 were idle at launch, GPU1 was occupied. Experiments used GPU0 for large model and GPU2 as visible small-model GPU.
- G0 result: PASS. c=20 oracle hidden n=10 reached full_large F1=0.5105, cut F1=0.5105, retention=1.0000.
- Base bridge: PASS. Fitted c=20 xnorm per-block bridge with train_n=150, tokens_per_block=512, ridge_lambda=0.1.
- Predictor training: STOP. The requested 112,351,744-parameter behavior-first predictor OOMed before the first optimization step on a 24GB RTX 4090 D.
- Conclusion: cut mechanism is sane, but the requested full-strength predictor training is not feasible on the current 24GB card without a memory redesign or larger GPU. G1/G2/G3 were not run because no predictor checkpoint exists.
- Outputs: `Experiment/analysis/long_context/CUT_cutpoint_summary.md`, `CUT_sanity_oracle_summary.*`, `CUT_sanity_oracle_c20_per_sample.csv`, `CUT_fit_base_bridge_summary.*`, `CUT_train_predictor_summary.*`, `CUT_train_predictor_log.csv`.


## 2026-06-05 - A800 cut-point deployment setup

- Purpose: migrate the Cut-point experiment runtime from `ser` to A800 so the 112M-parameter predictor can be trained on an 80GB GPU rather than the 24GB 4090D.
- Target path: `A800:/home/student_t/maqianli/scd-tpami`.
- Files copied: `Experiment/core_code`, `.pipeline`, `docs`, `instance.json`, selected CUT/V analysis artifacts, `data/json-test.arrow`, the required mapping checkpoint under `experiments/qwen25_1p5b_to_7b_longbench_128x512_decodeaware_e30_t102_e5_kv005_kl03ce02/model_final.pth`, and local copies of `Qwen2.5-1.5B` and `Qwen2.5-7B`.
- Storage: copied project footprint is about 18GB; `/home` on A800 had about 139GB free after copying.
- Environment: use `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`; verified `torch 2.10.0+cu128`, `transformers 5.3.0`, `datasets 4.8.5`.
- Path patching: remote `run_task_*.py` constants were rewritten from `/disk_n/c101-3-mql/...` to `/home/student_t/maqianli/scd-tpami/...` for the project root, Qwen model paths, and QA arrow path.
- Smoke validation: ran `CUDA_VISIBLE_DEVICES=5 python run_task_cut_point.py sanity_oracle --cut-layer 20 --n 1 --device cuda --small-device cuda --dtype bf16`; result PASS with full-large F1=1.0000, cut F1=1.0000, retention=1.0000. This is only an A800 deployment smoke check, not a replacement for the ser n=10 G0 result.
- GPU note: GPU5 was free before smoke, but after the run it was occupied by other users' Python processes; future training should recheck GPU availability before launch.

## 2026-06-07 - SCD-ICML core-code audit smoke

- Purpose: inspect the restored GitHub `new` core-code layer and verify that the active `Experiment/core_code` entry scripts and runtime package can execute a minimal schema-level pipeline.
- Inputs: `Experiment/core_code/configs/smoke_pair_screening.json`, `Experiment/core_code/configs/smoke_original_scd_baseline.json`, `runtime_repo/configs/models_local_smoke.yaml`, `runtime_repo/configs/data_local_smoke.yaml`, `runtime_repo/configs/train_reuse_qwen17_self_group4.yaml`, `runtime_repo/configs/train_qwen17_route_b_direct_kv.json`, and `runtime_repo/configs/eval.yaml`.
- Environment: local Windows Python 3.13.9, torch 2.11.0+cu128, CUDA available on GPU0 (`NVIDIA GeForce RTX 5070 Ti`, 4352/16303 MB used, 12% utilization before runtime smoke).
- Method: ran pair-screening smoke, original-SCD-baseline smoke, no-write-bytecode compile/config parsing checks, and a synthetic runtime chain: `capture_pair --backend synthetic`, `train_reuse`, `profile_layers`, `build_route`, `train_patch` with `direct_kv_v1`, `infer --method scd_full`, `eval_cmd`, and `report_cmd`.
- Outputs: temporary audit artifacts under `tmp_code_audit_outputs/`, including `pair_screen_summary.csv`, `original_baseline/*`, and `runtime_smoke/*`.
- Results: 72 Python files compiled in memory; 87 JSON files and 6 JSONL files parsed; outer validation smoke produced 3 pair rows and original baseline summaries; runtime synthetic chain completed end to end and wrote predictions, oracle, metrics, route, reuse, patch, and report artifacts.
- Observed issues: runtime `infer/eval/report` is a synthetic/surrogate path, not a true model-serving evaluation; payload and speed metrics are not yet tied to actual checkpoint size or measured serving cost; Matplotlib report generation emitted missing-CJK-glyph warnings for Chinese labels.
- Conclusion: PASS for local schema/CLI executability only. Do not use the synthetic runtime metrics as claim evidence; prioritize repairing real runtime evaluation, payload accounting, and report font handling before code cleanup for publication.

## 2026-06-07 - SCD-ICML Mistral reproduction setup on ser

- Purpose: prepare the current codebase to reproduce or closely reproduce the SCD-ICML Mistral results on `ser`, while explicitly skipping Qwen-32B.
- Environment preflight: `ser` host `leadtek-3`; `/disk_n/c101-3-mql/` exists; `/disk_n/c101-3-mql/venvs/A800/bin/activate` is usable; four RTX 4090 D GPUs were idle at preflight; MistralLite, Mistral-7B-v0.1, and Mistral-7B-Instruct-v0.1 directories exist under `/disk_n/c101-3-mql/models/`.
- Data preflight: CMRC2018 and WikiText-2 exist under `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data`; HotpotQA official dev file exists under the same snapshot tree in `tmp/hotpotqa_official/`.
- Method: added `scd_repro.text_benchmarks` loaders and `scd_repro.cli.eval_text_strict` to run strict real-model evaluation for CMRC2018, HotpotQA, and WikiText-2. Added `configs/eval_scd_icml_main_methods.json` to expose paper-table methods: full consumer, raw KV, 4-bit quantized KV, selective recompute, reuse-only, and SCD.
- Remote wrapper: added `scripts/run_ser_scd_icml_mistral_repro.sh`, which trains reuse/patch on CMRC2018 calibration prompts and evaluates the selected Mistral direction(s) over selected datasets using the strict transformers runtime.
- Smoke result: synced to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` and ran a real-model smoke with `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `DIRECTIONS=lite_to_7b`, `DATASETS=cmrc2018`, `MAX_LENGTH=256`, and `MAX_NEW_TOKENS=8`. The smoke completed capture, reuse training, profiling, patch training, patch-meta route generation, and strict eval.
- Smoke output: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_smoke_20260607_1544/scd_icml_mistral_summary.md`. This is only an executability check; the one-sample F1 values are not claim evidence.
- Fixes discovered during smoke: removed the UTF-8 BOM from `pyproject.toml` because Python 3.12 `tomllib`/pip rejected it; added route fields to `eval_scd_icml_main_methods.json`; changed the wrapper default route generation to `patch_meta`, because MistralLite produces an 8-layer patch checkpoint and all32 routes are invalid for the heterogenous MistralLite-to-Mistral-7B direction.
- Active run: launched the 200-sample Mistral matrix in background at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`, PID `561642`, with `TRAIN_LIMIT=200`, `EVAL_LIMIT=200`, `DIRECTIONS=lite_to_7b,7b_to_lite`, `DATASETS=cmrc2018,hotpotqa,wikitext2`, `MAX_LENGTH=1024`, and `MAX_NEW_TOKENS=32`.

## 2026-06-07 - SCD-ICML Mistral 200-sample run progress

- Purpose: record the active 200-sample Mistral reproduction run status while the long offline calibration and strict evaluation matrix continues on `ser`.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- PID: `561642`.
- Scope: `TRAIN_LIMIT=200`, `EVAL_LIMIT=200`, `DIRECTIONS=lite_to_7b,7b_to_lite`, `DATASETS=cmrc2018,hotpotqa,wikitext2`, `MAX_LENGTH=1024`, `MAX_NEW_TOKENS=32`; Qwen-32B remains intentionally skipped.
- Paper alignment: follows the ICML camera-ready TeX setup for the Mistral reproduction slice: 200 CMRC2018 calibration prefixes, QA evaluation on CMRC2018/HotpotQA, WikiText-2 PPL/NLL, and strict comparison against Oracle/full-consumer, raw KV, 4-bit KV, selective recompute, reuse-only, and full SCD.
- Progress at approximately 2026-06-07 16:40 CST: `lite_to_7b_capture_pair` completed successfully in 708 seconds; `lite_to_7b_train_reuse` completed successfully in 1080 seconds; `lite_to_7b_profile_layers` completed its first 200/200 pass and entered patch-KV profiling.
- Current status: run is still active. No failure has been observed yet. Final claim comparison remains pending until both Mistral directions and all selected datasets produce summaries.

## 2026-06-07 - SCD-ICML Mistral Patch calibration progress

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Progress at approximately 2026-06-07 17:50 CST: `lite_to_7b_profile_layers` completed successfully in 2272 seconds.
- Active stage: `lite_to_7b_train_patch`.
- Patch route selected from the profile: `[16, 15, 17, 13, 18, 10, 12, 11]`.
- Patch training progress: the 200/200 shared-stat pass completed; linear-base solving has completed for layer 16 and layer 15. The process is still active and CPU-bound/IO-bound, with no failure status written.
- Runtime note: linear-base solving is taking roughly 10-12 minutes per selected layer on the current `ser` load. This is consistent with Patch calibration being the expensive offline stage and should not yet be treated as a hang.

## 2026-06-07 - SCD-ICML Mistral resume after GPU contention OOM

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Failure: `lite_to_7b_train_patch` failed with CUDA OOM after completing linear-base solves for layers 16, 15, and 17. The failure was caused by concurrent jobs occupying GPU0, leaving only a few MB free before a 20 MB allocation.
- Evidence: failed status was written at approximately 2026-06-07 18:05 CST with `exit_code=1`; failed logs were preserved at `logs/lite_to_7b_train_patch_failed_oom_20260607_1805`.
- Code robustness update: updated `scripts/run_ser_scd_icml_mistral_repro.sh` with configurable device variables and optional GPU-free-memory gates. Successful stages are now skipped before waiting under `RESUME=1`, so resume does not block on already-completed stages.
- Resume launch: synced the updated wrapper to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` and relaunched with `RESUME=1`, `WAIT_TRAIN_FREE_MB=6000`, `WAIT_EVAL_FREE_MB=18000`, `WAIT_CAPTURE_FREE_MB=18000`, and PID `715084`.
- Resume status: the wrapper skipped `lite_to_7b_capture_pair`, `lite_to_7b_train_reuse`, and `lite_to_7b_profile_layers`, then restarted `lite_to_7b_train_patch`. Later strict-eval and reverse-direction capture will wait for two GPUs with at least 18 GB free each.

## 2026-06-07 - SCD-ICML Mistral patch checkpoint recovered

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Recovered stage: `lite_to_7b_train_patch` completed successfully after resume. Status written at `2026-06-07T20:15:39+0800`, wall clock 7585 seconds.
- Patch route: `[16, 15, 17, 13, 18, 10, 12, 11]`. All eight linear-base solves completed, followed by one shared residual epoch.
- Checkpoint outputs: `lite_to_7b/ckpt_patch/patch_cmrc2018_seed42/metrics.json`, `meta.json`, and `weights.pt`.
- Next stage: `lite_to_7b_route_patch_meta` completed successfully immediately after patch training.
- Runtime adjustment: the resumed wrapper initially waited for physical GPU0 and GPU1, but GPU3 was the only free card. The waiting wrapper was stopped and relaunched with `CUDA_VISIBLE_DEVICES=3,1` and `RESUME=1`, preserving completed stages and rebinding eval to GPU3 plus GPU1.
- Current status: the run is waiting for GPU1 to free at least 18 GB before strict eval starts. No strict-eval claim metrics have been produced yet.

## 2026-06-07 - SCD-ICML Mistral CMRC strict-eval result

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Completed stage: `lite_to_7b_eval_cmrc2018`, status success at `2026-06-07T20:56:58+0800`, wall clock 1356 seconds.
- Output: `lite_to_7b/strict_eval/cmrc2018/summary.md`, `summary.json`, `summary.csv`, `per_mode_summary.json`, and per-sample logs.
- CMRC2018 n=200 result at 200 Gbps:
  - Full Consumer: F1 0.6573, EM 0.3650, TTFT 101.38 ms.
  - Raw KV BF16: F1 0.0918, EM 0.0000, TTFT 29.90 ms.
  - 4-bit KV: F1 0.1097, EM 0.0000, TTFT 52.82 ms.
  - Selective Recompute: F1 0.5223, EM 0.2150, TTFT 99.27 ms.
  - Reuse-only: F1 0.0089, EM 0.0000, TTFT 34.37 ms.
  - SCD: F1 0.7347, EM 0.4250, TTFT 85.23 ms.
- Paper comparison note: the SCD CMRC F1 is about 0.05 absolute below the paper's canonical MistralLite-to-Mistral-7B CMRC SCD result (0.7850), while preserving the core trend that raw/4-bit KV collapse and SCD recovers high QA quality. The current Full Consumer F1 is lower than the paper table, so later open-source documentation must state the exact model paths, split, sample count, and decoding settings used for this reproduced run.
- Current status: `lite_to_7b_eval_hotpotqa` started automatically after CMRC and is running.

## 2026-06-07 - SCD-ICML Mistral canonical strict-eval matrix completed

- Purpose: record the completed paper-facing Mistral reproduction slice for the open-source SCD-ICML code release.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Direction: `lite_to_7b` (`MistralLite -> Mistral-7B`), matching the canonical Mistral direction in the SCD-ICML paper. Qwen-32B was intentionally skipped, and the reverse `7b_to_lite` direction was not treated as claim-facing.
- Calibration/eval settings: `TRAIN_LIMIT=200`, `EVAL_LIMIT=200`, seed 42, `MAX_LENGTH=1024`, `MAX_NEW_TOKENS=32`, 200 Gbps bandwidth setting, CMRC2018 calibration prompts, strict transformers runtime.
- Completed statuses: `lite_to_7b_eval_cmrc2018` success at `2026-06-07T20:56:58+0800`; `lite_to_7b_eval_hotpotqa` success at `2026-06-07T21:13:02+0800`; `lite_to_7b_eval_wikitext2` success at `2026-06-07T21:42:47+0800`.
- Aggregated outputs: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605/scd_icml_mistral_summary.json` and `scd_icml_mistral_summary.md`, verified with 18 rows covering 3 datasets x 6 methods.
- CMRC2018 n=200 result: Full Consumer F1 0.6573 / TTFT 101.38 ms; Raw KV F1 0.0918 / TTFT 29.90 ms; 4-bit KV F1 0.1097 / TTFT 52.82 ms; Selective Recompute F1 0.5223 / TTFT 99.27 ms; Reuse-only F1 0.0089 / TTFT 34.37 ms; SCD F1 0.7347 / TTFT 85.23 ms.
- HotpotQA n=200 result: Full Consumer F1 0.0343 / TTFT 140.14 ms; Raw KV F1 0.0133 / TTFT 32.97 ms; 4-bit KV F1 0.0151 / TTFT 56.52 ms; Selective Recompute F1 0.0324 / TTFT 139.63 ms; Reuse-only F1 0.0079 / TTFT 42.40 ms; SCD F1 0.0419 / TTFT 116.40 ms.
- WikiText-2 n=200 result: Full Consumer PPL 7.5840 / TTFT 64.11 ms; Raw KV PPL 10.6300 / TTFT 25.17 ms; 4-bit KV PPL 10.7175 / TTFT 48.41 ms; Selective Recompute PPL 7.8385 / TTFT 62.21 ms; Reuse-only PPL 778.8231 / TTFT 26.50 ms; SCD PPL 7.6356 / TTFT 56.49 ms.
- Interpretation: CMRC and WikiText-2 are strong enough for an approximate open-source reproduction claim: SCD recovers high QA quality on CMRC and near-full-consumer PPL on WikiText-2, while reuse-only collapses. HotpotQA absolute F1 is too low across all methods under the current model/prompt slice and should be described only as a sanity check.
- Release boundary: present these numbers as a strict-runtime reproduction trend, not an exact paper-table replication. The public documentation must include model paths/checkpoint variants, sample limits, decoding settings, and the fact that Qwen-32B is skipped on this machine.

## 2026-06-07 - SCD-ICML Mistral release tooling validation

- Purpose: convert the verified strict-eval run into a reusable open-source workflow rather than a one-off shell-script footer.
- Code changes: added `src/scd_repro/cli/summarize_mistral_repro.py`, registered the `summarize-mistral-repro` console script, and changed `scripts/run_ser_scd_icml_mistral_repro.sh` to call `python -m scd_repro.cli.summarize_mistral_repro` after dataset evaluations finish.
- Metadata cleanup: rebuilt `Experiment/core_code/runtime_repo/pyproject.toml` with ASCII package description/author metadata and the new summary entry point.
- Documentation: added `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md` with smoke command, 200-sample command, verified result table, and public claim boundary. Updated `README.md` to link this release-facing reproduction note and to show the standalone summary-regeneration command.
- Local verification: `python -m py_compile Experiment\core_code\runtime_repo\src\scd_repro\cli\summarize_mistral_repro.py` passed; `tomllib` parsed `pyproject.toml` and resolved `summarize-mistral-repro = scd_repro.cli.summarize_mistral_repro:main`.
- Remote deployment: synced the new CLI, wrapper, `pyproject.toml`, README, and `REPRODUCING_SCD_ICML.md` to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Remote verification command: from the synced runtime repo, ran `python -m py_compile src/scd_repro/cli/summarize_mistral_repro.py`, `bash -n scripts/run_ser_scd_icml_mistral_repro.sh`, and `PYTHONPATH=src python -m scd_repro.cli.summarize_mistral_repro --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605 --eval-config configs/eval_scd_icml_main_methods.json --expected-directions lite_to_7b --expected-datasets cmrc2018,hotpotqa,wikitext2`.
- Remote result: summary regeneration succeeded with 18 rows, covering 3 datasets x 6 methods for `lite_to_7b`.
- Remaining release work: prepare the final minimal GitHub tree and check whether data/model download instructions need to be generalized beyond local `/disk_n/c101-3-mql/` paths.

## 2026-06-07 - SCD-ICML portable release setup validation

- Purpose: make the verified Mistral reproduction path suitable for external users who do not have the local `/disk_n/c101-3-mql/` model and data layout.
- Code changes: added `configs/models_mistrallite_to_mistral7_template.json`; updated `scripts/run_ser_scd_icml_mistral_repro.sh` so `MODELS_CONFIG_LITE_TO_7B` and `MODELS_CONFIG_7B_TO_LITE` can override the local verified model configs.
- Documentation changes: added `DATA_PREPARATION.md` with accepted CMRC2018, HotpotQA, and WikiText-2 directory layouts; added `OPEN_SOURCE_RELEASE_MANIFEST.md` to identify required runtime code, configs, scripts, and docs for the claim-facing release; updated README and `REPRODUCING_SCD_ICML.md` to point users to portable data/model setup.
- Readiness gate: added `scripts/check_release_readiness.py`, which checks required files, JSON/TOML parsing, key Python compile targets, wrapper defaults, eval-config method flags, model-template fields, documentation snippets, and optional aggregate run-root rows.
- Local verification: `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed. JSON parsing also passed for `configs/models_mistrallite_to_mistral7_template.json` and `configs/eval_scd_icml_main_methods.json`.
- Remote deployment: synced the new template, docs, wrapper, and readiness script to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`; removed accidental root-level duplicate script copies after sync.
- Remote verification: from the synced runtime repo, ran `bash -n scripts/run_ser_scd_icml_mistral_repro.sh` and `python scripts/check_release_readiness.py --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`; both passed, and the run-root check verified the expected 18 aggregate rows for `lite_to_7b`.
- Current release status: the main Mistral reproduction path now has a documented portable setup, a verified 200-sample result table, and a release self-check. Remaining work is final pruning/packaging of historical files before publishing.

## 2026-06-07 - SCD-ICML release preview package validation

- Purpose: produce a non-destructive open-source release preview for the paper-facing Mistral reproduction code while keeping historical scripts/configs in the working tree.
- Code changes: added `scripts/build_release_tree.py`; updated `OPEN_SOURCE_RELEASE_MANIFEST.md` to document the preview builder; updated `scripts/check_release_readiness.py` to include the builder and to compile Python files into a temporary directory instead of leaving `__pycache__` in release previews.
- Local build command: `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force`.
- Local output: `Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release`; readiness check passed; 60 files; 0 `__pycache__` directories.
- Local report: `Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release\RELEASE_BUILD_REPORT.md`. The report marks 6 root files, 71 configs, and 34 scripts as excluded historical material, not deleted.
- Remote deployment: synced the updated builder, readiness script, README, and release manifest to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Remote verification command: from the synced runtime repo, ran `python scripts/check_release_readiness.py --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`, `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`, and `python release_preview/scd_icml_runtime_release/scripts/check_release_readiness.py --root release_preview/scd_icml_runtime_release --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
- Remote output: `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release`; source and preview readiness checks passed; 60 files; 0 `__pycache__` directories; verified the 18 expected aggregate rows for `lite_to_7b`.
- Documentation: added `docs/scd-icml-open-source-release-preview-2026-06-07.md` summarizing preview paths, included release surface, excluded historical categories, and verification commands.
- Current release status: a concrete candidate release tree now exists locally and on `ser`. Remaining work is review/approval of whether to replace the broader runtime repo with this preview tree for GitHub publication.

## 2026-06-07 - SCD-ICML release preview GPU gate validation

- Purpose: prevent the public release wrapper from failing with avoidable CUDA OOM when run on a shared machine with busy GPUs.
- Observed GPU state on `ser` at 2026-06-07 22:22-22:25 CST: GPU0 used about 20GB, GPU1 used about 16GB, GPU2 used about 17GB, and GPU3 was idle. Only one GPU had enough free memory for the Mistral pair path.
- Decision: do not launch the release-preview real-model smoke while only one GPU is sufficiently free. The Mistral pair smoke requires two GPUs; using busy GPUs would violate the project GPU-default rule and risk OOM.
- Code change: updated `scripts/run_ser_scd_icml_mistral_repro.sh` defaults to `WAIT_CAPTURE_FREE_MB=18000`, `WAIT_TRAIN_FREE_MB=6000`, and `WAIT_EVAL_FREE_MB=18000`.
- Readiness update: added these default gate snippets to `scripts/check_release_readiness.py`, so future release checks fail if the wrapper reverts to unsafe no-wait defaults.
- Local verification: source runtime readiness passed; release preview rebuilt with `scripts/build_release_tree.py`; preview readiness passed.
- Remote verification: synced the updated wrapper/readiness/builder/docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`; ran `bash -n scripts/run_ser_scd_icml_mistral_repro.sh`, source readiness with the verified 200-sample run root, release preview rebuild, and preview readiness with the same run root. All passed; preview `__pycache__` count remained 0.
- Evidence status: the release preview has not yet rerun a fresh real-model smoke after packaging, but it independently validates the existing 18-row `lite_to_7b` aggregate result table and is now guarded against busy-GPU OOM by default. Run the documented `TRAIN_LIMIT=2 EVAL_LIMIT=1` smoke once at least two GPUs have about 18GB free.

## 2026-06-07 - SCD-ICML release preview autogpu smoke queued

- Purpose: launch the packaged release preview itself through the real-model smoke path, while respecting shared-GPU availability.
- First attempt: launched `scd_icml_release_preview_smoke_20260607_223010`, which installed the release package and waited on `CUDA_VISIBLE_DEVICES=3,1`. This was stopped because it was fixed to GPU1, which remained busy.
- Code improvement: updated `scripts/run_ser_scd_icml_mistral_repro.sh` with `pick_free_gpus`, `AUTO_SELECTED_CUDA_VISIBLE_DEVICES`, and capture/eval re-selection. Updated `scripts/check_release_readiness.py` so these snippets are release-gated. Updated README and `REPRODUCING_SCD_ICML.md` to describe automatic re-selection.
- Verification after improvement: local source readiness passed; local release preview rebuild passed; remote `bash -n`, source readiness with the verified run root, release preview rebuild, and preview readiness with the verified run root all passed. Remote preview script contains the new automatic re-selection logic.
- Active smoke: stopped the pre-reselection waiting smoke and launched PID `1022105` from `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release`.
- Active smoke run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_20260607_223858_autogpu_v2`.
- Active smoke command scope: `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `DIRECTIONS=lite_to_7b`, `DATASETS=cmrc2018`, `MAX_LENGTH=256`, `MAX_NEW_TOKENS=8`, `WAIT_GPU_POLL_SEC=300`.
- Current status at 2026-06-07 22:39 CST: package install succeeded, 2 CMRC calibration prompts were written, no stage status files or failures exist yet, and the wrapper is waiting at the 18GB GPU capture gate because only one GPU is sufficiently free.
- Next verification: when a second GPU frees, the smoke should proceed through capture, reuse training, profiling, patch training, patch-meta route generation, CMRC strict eval, and aggregate summary generation. Then run `python scripts/check_release_readiness.py --run-root <smoke-run-root>` inside the release preview.

## 2026-06-07 - SCD-ICML release smoke status checker validation

- Purpose: add a deterministic way to inspect queued or completed release-preview smoke runs without manually parsing logs.
- Code changes: added `scripts/check_release_smoke_status.py`; updated `scripts/build_release_tree.py`, `scripts/check_release_readiness.py`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` so the status checker is included and compiled in the release preview.
- Local verification: `python -m py_compile` passed for the status checker, release readiness script, and release builder; source readiness passed; local release preview rebuild and readiness passed.
- Remote verification: synced the new status checker/readiness/builder/manifest to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`; remote source readiness and release preview rebuild/readiness passed.
- Status command: inside the remote release preview, ran `python scripts/check_release_smoke_status.py --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_20260607_223858_autogpu_v2 --pid 1022105`.
- Status result: `state=waiting_gpu`; `pid_running=true`; `status_count=0`; expected stages are capture, reuse train, profile, patch train, patch-meta route, and CMRC eval; summary files do not exist yet; log tail shows the wrapper waiting for `WAIT_CAPTURE_FREE_MB=18000`.
- Interpretation: the active release-preview smoke has not failed or completed. It remains safely queued until a second GPU reaches the 18GB free-memory gate.

## 2026-06-07 - SCD-ICML release-preview single-GPU smoke completed

- Purpose: validate the packaged release preview itself on currently available `ser` hardware without occupying busy GPUs.
- Initial condition: only GPU3 had about 24GB free; GPUs 0/1/2 were busy. The previous two-GPU release smoke PID `1022105` was safely waiting for a second 18GB-free GPU and was stopped before launching the single-GPU validation.
- Code changes:
  - Added `configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json`.
  - Updated `scripts/run_ser_scd_icml_mistral_repro.sh` to infer automatic GPU count from configured CUDA devices and to wait only on the unique logical devices used by capture/eval.
  - Added same-device sequential capture in `src/scd_repro/cli/capture_pair.py`: producer capture is moved to CPU before loading the consumer.
  - Added same-device strict eval staging in `src/scd_repro/cli/eval_text_strict.py`: producer prefills are moved to CPU, the producer model is released, and the consumer is loaded afterward.
  - Fixed `TransformersRuntimeSession.close()` and `TransformersCaptureSession.close()` so retained `layers` and `tokenizer` references do not keep model weights alive.
  - Added `configs/data_ser.yaml` and the single-GPU config to the release builder and readiness checks.
- Failed attempts and root causes:
  - `scd_icml_release_preview_smoke_single_gpu_20260607_225332` failed because release preview did not include `configs/data_ser.yaml`.
  - `scd_icml_release_preview_smoke_single_gpu_20260607_225539` failed because same-device capture loaded MistralLite and Mistral-7B simultaneously and OOMed.
  - First resume of `scd_icml_release_preview_smoke_single_gpu_20260607_230105` failed because strict eval loaded producer and consumer simultaneously on one GPU.
  - Second resume failed because `close()` retained layer references, so producer weights were not fully released before consumer load.
- Final successful run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_single_gpu_20260607_230105`.
- Final command scope: release preview package, `RESUME=1` after successful capture/train/profile/route stages, `MODELS_CONFIG_LITE_TO_7B=configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json`, all logical devices set to `cuda:0`, `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `DIRECTIONS=lite_to_7b`, `DATASETS=cmrc2018`, `MAX_LENGTH=256`, `MAX_NEW_TOKENS=8`.
- Completed stage statuses:
  - `lite_to_7b_capture_pair`: success, wall clock 66 s.
  - `lite_to_7b_train_reuse`: success, wall clock 4 s.
  - `lite_to_7b_profile_layers`: success, wall clock 6 s.
  - `lite_to_7b_train_patch`: success, wall clock 27 s.
  - `lite_to_7b_route_patch_meta`: success.
  - `lite_to_7b_eval_cmrc2018`: success on final resume, wall clock 16 s.
- Smoke summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_single_gpu_20260607_230105/scd_icml_mistral_summary.md` and `.json`; 6 rows for CMRC2018 x 6 methods. Single-sample F1 values are all 0.0000 and are not claim evidence.
- Verification commands:
  - Remote release preview readiness with verified 200-sample run root passed: `python scripts/check_release_readiness.py --root /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
  - Smoke status passed with `state=complete`, all expected stages successful, and both aggregate summary files present: `python scripts/check_release_smoke_status.py --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_single_gpu_20260607_230105`.
- Conclusion: the release preview now has two layers of evidence: the full 200-sample `lite_to_7b` result table for claim-facing reproduction and a completed packaged single-GPU smoke for external wiring validation. The single-GPU smoke does not replace the two-GPU/full-run recommendation for reproducing paper-like numbers.

## 2026-06-07 - SCD-ICML final release-readiness audit

- Purpose: confirm the current open-source candidate still satisfies the Mistral-only reproduction target after the release-preview smoke fixes and documentation update.
- Local verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force` rebuilt the local release preview and its internal readiness check passed.
- Remote verification:
  - From `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`, `python scripts/check_release_readiness.py --root "$ROOT" --run-root "$RUN"` passed with `RUN=/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605`.
  - Remote `python scripts/build_release_tree.py --root "$ROOT" --out "$ROOT/release_preview/scd_icml_runtime_release" --force` rebuilt the release preview.
  - Remote preview readiness passed with `python "$ROOT/release_preview/scd_icml_runtime_release/scripts/check_release_readiness.py" --root "$ROOT/release_preview/scd_icml_runtime_release" --run-root "$RUN"`.
- Evidence checked:
  - Full run summary remains 18 rows: `lite_to_7b` across CMRC2018, HotpotQA, and WikiText-2, with 6 methods per dataset.
  - Release preview build report includes the paper-facing runtime, configs, docs, wrapper, builder, readiness checker, smoke status checker, and `src/scd_repro/`; historical Qwen-1.7B, reverse-direction, and one-off diagnostic scripts are excluded from the default release preview.
  - `docs/scd-icml-open-source-release-preview-2026-06-07.md` now records that the old two-GPU waiting smoke PID `1022105` was stopped and that the packaged single-GPU smoke completed.
- Conclusion: the current release candidate is ready as a MistralLite-to-Mistral-7B open-source reproduction package. It supports the paper-facing 200-sample Mistral reproduction and a real-model packaged smoke test, while explicitly excluding Qwen-32B from the current machine-verified scope.

## 2026-06-08 - SCD-ICML no-fallback/no-recompute correction

- Purpose: correct the claim-facing SCD implementation after discovering that the previous `scd_full` route used consumer fallback layers for non-patch layers.
- Invalidated evidence: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605` remains useful as a diagnostic run, but its `scd_full` row is not valid no-fallback/no-recompute SCD evidence because the generated route placed non-patch layers in `fallback_layers`.
- GitHub baseline check: `new-main/new-main/Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md` records a strict no-fallback MistralLite-to-Mistral-7B full-dev comparison with `patch-only` F1 0.7898 / EM 0.5769 / TTFT 67.37 ms and `reuse-only` F1 0.0109 / TTFT 38.49 ms.
- Code changes:
  - `scripts/make_patch_meta_route.py` now emits `fallback_layers=[]`, `selected_recompute_layers=[]`, reuse on non-patch layers, and patch on selected layers.
  - `src/scd_repro/strict_runtime.py` now rejects `scd_full` routes with fallback layers and allows `reuse_layers=[]` only when patch layers cover all layers.
  - `src/scd_repro/direct_kv_residual_patch.py` now honors `force_residual_layers_as_selected` in the shared residual patch trainer.
  - `configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json` now sets `force_residual_layers_as_selected=true`.
  - `configs/eval_scd_icml_no_fallback_no_recompute.json` is the default claim-facing eval config and disables selective recompute, reuse fallback, patch-only separate mode, reuse+patch+recompute, and short recompute.
- Documentation changes: `README.md`, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` now state that claim-facing SCD is no-fallback/no-recompute and that the old 18-row result is legacy diagnostic evidence, not final SCD claim evidence.
- Local verification: `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed after the route/runtime/training/doc changes.
- Remote verification: synced the touched files to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`; remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
- Smoke 1: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_smoke_nofallback_20260607_2355` completed with 5 aggregate rows and route `fallback_layers=[]`, but it used a sparse 8-layer patch because the all-layer selection bug had not yet been fixed.
- Smoke 2: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_smoke_nofallback_20260608_0010` completed after the all-layer trainer fix. Its route has `patch_layers=0..31`, `reuse_layers=[]`, `fallback_layers=[]`, and no recompute map; status checker reports `state=complete` with all expected stages successful. This is wiring evidence only because it uses `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `MAX_LENGTH=256`, and `MAX_NEW_TOKENS=8`.
- Full-run queue: launched `/disk_n/c101-3-mql/projects/SCD-ICML/runs/launch_true_scd_allpatch_20260608_0020.sh` as PID `1136660`. The run root is `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_allpatch_repro_20260608_0020`; it is waiting at the GPU gate for a second GPU with at least 18GB free.
- Current conclusion: the code path is now aligned with the user's constraint, but the final open-source quality claim is not complete until the queued 200-sample no-fallback/no-recompute run finishes and passes `check_release_readiness.py --run-root`.

## 2026-06-08 - Release-preview no-fallback hygiene check

- Purpose: remove remaining release-preview ambiguity after confirming the GitHub baseline is strict `reuse-only` plus `patch-only`, not fallback-hybrid SCD.
- Code/config changes:
  - Removed `configs/eval_scd_icml_main_methods.json` from the release builder and readiness required-file lists. It is no longer packaged in `release_preview/scd_icml_runtime_release`.
  - Updated `OPEN_SOURCE_RELEASE_MANIFEST.md` and `README.md` to state that older selective-recompute/fallback diagnostic configs are kept only for traceability and excluded from the release preview.
  - Set `runtime.cpu_fallback=false` in `configs/eval_scd_icml_no_fallback_no_recompute.json`, `configs/train_reuse_mistral_self_layerwise_20260404.json`, and `configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json`.
  - Extended `scripts/check_release_readiness.py` to reject CPU fallback in the release-facing eval, reuse-train, and all-layer patch-train configs.
- Local verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force` passed and rebuilt the local preview.
  - `Test-Path Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release\configs\eval_scd_icml_main_methods.json` returned `False`; the build report lists it only under `Not Included`.
- Remote verification:
  - Synced the changed release scripts/configs/docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
  - Remote `python scripts/build_release_tree.py --root "$ROOT" --out "$ROOT/release_preview/scd_icml_runtime_release" --force` passed.
  - Remote release preview reports `main_methods_absent`; `sed` inspection confirmed `enable_short_recompute=false`, `recompute_segment_lengths=[]`, `run_selective_recompute=false`, `run_reuse_fallback=false`, `run_reuse_patch_recompute=false`, `default_recompute_segment_length=0`, and `runtime.cpu_fallback=false`.
- Current conclusion: the release-preview package is now stricter than the working tree: it packages only the no-fallback/no-recompute Mistral path, while legacy fallback/recompute configs remain outside the package for audit history.

## 2026-06-08 - Explicit patch-only release-table update

- Purpose: align the release-facing aggregate table with the GitHub reference, which reports strict `reuse-only` and strict `patch-only` directly.
- Code/config changes:
  - Set `runtime_eval.run_patch_only=true` in `configs/eval_scd_icml_no_fallback_no_recompute.json`.
  - Updated `scripts/check_release_readiness.py` so expected methods are now `full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, `patch_only`, and `scd_full`; expected aggregate rows are now 18.
  - Added a run-root guard requiring all layers to be in `patch_layers` when the release table includes `patch_only`.
  - Updated `OPEN_SOURCE_RELEASE_MANIFEST.md` and `REPRODUCING_SCD_ICML.md` to document the 18-row table and explicit `patch_only` anchor.
- Verification:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local release preview rebuild passed with `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force`.
  - Remote source readiness and release-preview rebuild/readiness passed after syncing the changed files.
  - The queued run root temporary eval config `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_allpatch_repro_20260608_0020/tmp_cfg/eval_main_lite_to_7b.json` was updated before capture/training/eval to contain `run_patch_only=true`.
- Current conclusion: the next full run, once it obtains two sufficiently free GPUs, should produce a no-fallback/no-recompute 18-row table with an explicit patch-only quality anchor.

## 2026-06-08 - Mixed no-fallback route generator smoke

- Purpose: add a concrete open-source path for true mixed SCD routing, not just all-layer patch-only.
- Code changes:
  - `scripts/make_patch_meta_route.py` now parses `--patch-layers` layer specs with ids and ranges. Requested layers must be present in the trained patch checkpoint metadata.
  - `scripts/run_ser_scd_icml_mistral_repro.sh` now accepts `ROUTE_PATCH_LAYERS` and passes it through to route generation when `ROUTE_MODE=patch_meta`; when this subset route is requested, the generated eval config disables the all-layer `patch_only` row.
  - `scripts/check_release_readiness.py` now includes a semantic smoke for explicit mixed route generation.
  - `scripts/validate_scd_route.py` was added as a standalone route validator for no-fallback/no-recompute routes.
- Local verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force` passed.
  - A fake 4-layer route smoke with `--patch-layers 1,3` produced `reuse_layers=[0,2]`, `patch_layers=[1,3]`, `fallback_layers=[]`, `selected_recompute_layers=[]`, and `recompute_segment_length_by_layer={}`.
  - `validate_scd_route.py` accepted the good fake mixed route and rejected a bad fake route containing `fallback_layers=[0]`.
  - A remote `bash -n` check on a temporary copy of the local wrapper returned `bash_n_exit=0`.
- Operational note: the remote source wrapper is not overwritten while the queued 200-sample run is still executing from that file. The local workspace and local release preview contain the mixed-route support; remote sync should happen after PID `1136660` exits or is intentionally stopped.

## 2026-06-08 - Remote source sync and queued-run restart

- Purpose: make the remote source tree consistent with the new mixed-route release preview without modifying a wrapper while it is being interpreted by bash.
- Remote process handling:
  - The previous queued process `1136660`/`1136661` was only waiting at the GPU gate and had not started capture, training, route generation, or evaluation.
  - It was stopped before syncing the new wrapper.
  - The same launch script and run root were relaunched under PID `1181263`.
- Remote sync and verification:
  - Synced `scripts/run_ser_scd_icml_mistral_repro.sh`, `scripts/make_patch_meta_route.py`, `scripts/validate_scd_route.py`, `scripts/check_release_readiness.py`, `scripts/build_release_tree.py`, and the updated release docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
  - Remote `python scripts/build_release_tree.py --root "$ROOT" --out "$ROOT/release_preview/scd_icml_runtime_release" --force` passed.
  - Remote source and preview both contain `scripts/validate_scd_route.py`; the preview still excludes `configs/eval_scd_icml_main_methods.json`.
- Relaunched run state:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_allpatch_repro_20260608_0020`.
  - PID: `1181263`.
  - `run_config.json` records `route_mode="patch_meta"` and `route_patch_layers=""`, so this run is still the all-layer Patch anchor rather than a subset mixed-route run.
  - The generated eval config has `run_patch_only=true`, `run_scd_full=true`, `run_selective_recompute=false`, `run_reuse_fallback=false`, `run_reuse_patch_recompute=false`, and `default_recompute_segment_length=0`.
  - Current state remains GPU-gate waiting; no model stage has started.

## 2026-06-08 - Strict-stop hybrid search release wrapper

- Purpose: correct the open-source mixed-route path after confirming that the GitHub reference treats `hybrid` as a separately validated strict-stop search result, not as a manually chosen layer subset.
- Code changes:
  - Added `scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh`.
  - The wrapper requires `BASE_RUN_ROOT`, reuses `<run-root>/lite_to_7b/ckpt_reuse/cmrc2018_seed42`, `<run-root>/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42`, and the generated all-layer route template.
  - It runs `python -m scd_repro.cli.search_blockwise_hybrid` with default `BLOCK_SIZE=1`, `REPLAY_TOP_K=3`, and `REPLAY_DECISION_MAX_F1_DROP_RATIO=0.05`.
  - It validates the selected route with `python scripts/validate_scd_route.py --route <search-out>/selected_hybrid_route.json`.
  - Added `scripts/validate_strict_hybrid_search.py` to validate `search_report.json`, selected-route metrics, replay F1 gate settings, accepted-block records, and route-level no-fallback/no-recompute constraints. The `--require-mixed` flag is the claim gate for saying a mixed route was actually found.
  - The strict hybrid wrapper now runs both route validation and search-report validation before returning.
  - The wrapper resolves relative `MODELS_CONFIG`, `PATCH_EVAL_CONFIG`, and `ROUTE_TEMPLATE` against `ROOT`, so it can be launched from outside the repository root when needed.
  - `scripts/build_release_tree.py` now includes the strict hybrid wrapper and strict search validator in the release preview.
  - `scripts/check_release_readiness.py` now requires the wrapper, compiles `src/scd_repro/cli/search_blockwise_hybrid.py`, checks the `search-blockwise-hybrid` console entry point, and runs a fake good/bad strict-search artifact smoke.
- Documentation changes:
  - `README.md`, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` now describe `ROUTE_PATCH_LAYERS` as a manual ablation switch only.
  - The canonical mixed-route procedure is documented as strict-stop hybrid search: patch-only anchor, greedy Reuse replacement, replay F1 gate, `fallback_layers=[]`, and no recomputation.
  - `docs/scd-icml-open-source-release-status-2026-06-08.md` was updated with the same boundary.
- Verification status:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed after adding the wrapper.
  - Local `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force` passed and rebuilt the preview.
  - Preview inspection confirmed `scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh` is included and `configs/eval_scd_icml_main_methods.json` is still excluded.
  - Synced the strict hybrid wrapper and updated release docs/checkers to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `bash -n "$ROOT/scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh"` passed.
  - Remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
  - Remote `python scripts/build_release_tree.py --root "$ROOT" --out "$ROOT/release_preview/scd_icml_runtime_release" --force` passed.
  - After the path-resolution fix, local source readiness, local release-preview readiness, remote `bash -n`, remote source readiness, and remote release-preview readiness all passed again.
  - After adding `validate_strict_hybrid_search.py`, local source readiness, local release-preview readiness, remote wrapper `bash -n`, remote validator `py_compile`, remote source readiness, and remote release-preview readiness all passed again; remote preview contains `scripts/validate_strict_hybrid_search.py`.
  - Remote preview inspection confirmed `scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh` is included and `configs/eval_scd_icml_main_methods.json` is absent.
  - No new strict hybrid search has been launched yet because the current 200-sample all-layer anchor run is still waiting at the GPU gate.
  - The current anchor PID `1181263` remains `bash -> bash -> sleep`, and the log tail still shows waiting for two GPUs with at least 18GB free.

## 2026-06-08 - Add MIT license to release preview

- Purpose: complete the open-source packaging surface. `pyproject.toml` already declared MIT, but the release preview lacked an actual `LICENSE` file.
- Code/package changes:
  - Added `Experiment/core_code/runtime_repo/LICENSE` with the MIT license text.
  - Added `LICENSE` to `scripts/build_release_tree.py` root files.
  - Added `LICENSE` to `scripts/check_release_readiness.py` required files and doc snippet checks.
  - Updated `README.md` and `OPEN_SOURCE_RELEASE_MANIFEST.md` to mention the license file.
- Local verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - `python Experiment\core_code\runtime_repo\scripts\build_release_tree.py --root Experiment\core_code\runtime_repo --out Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release --force` passed.
  - `Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release\LICENSE` exists and contains `MIT License`.
- Remote verification:
  - Synced `LICENSE`, updated release docs, and updated release scripts to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
  - Remote `python scripts/build_release_tree.py --root "$ROOT" --out "$ROOT/release_preview/scd_icml_runtime_release" --force` passed.
  - Remote preview contains `LICENSE`; `grep -n MIT .../LICENSE` returned `1:MIT License`.

## 2026-06-08 - Release preview import smoke

- Purpose: verify that the release preview is importable as a Python package, not just structurally present on disk.
- Check added:
  - `scripts/check_release_readiness.py` now reads `[project.scripts]` from `pyproject.toml`, prepends `<root>/src` to `sys.path`, and imports each declared console-script module.
- Manual local evidence before integrating:
  - A direct import smoke against `Experiment/core_code/runtime_repo/release_preview/scd_icml_runtime_release` returned `imports_ok`.
  - The inspected console scripts were `build-route`, `capture-pair`, `eval`, `eval-text-strict`, `infer`, `profile-layers`, `report`, `search-blockwise-hybrid`, `summarize-mistral-repro`, `train-patch`, and `train-reuse`.
- Verification:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local release preview rebuild/readiness passed.
  - Synced the updated readiness checker to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py --root "$ROOT"` passed.
  - Remote release preview rebuild/readiness passed, and `preview_core_files_ok` confirmed `LICENSE` plus `scripts/validate_strict_hybrid_search.py` are present in the remote preview.

## 2026-06-08 - Paper mixed Reuse + Patch release correction

- Purpose: align the open-source reproduction with the SCD-ICML paper rather than the incomplete GitHub seed code.
- Method:
  - Treat paper SCD as sparse Patch on selected transition layers plus Reuse on every non-patch layer.
  - Use `scripts/make_paper_scd_route.py` to generate `route_semantics=paper_scd_sparse_patch_reuse_all_other_layers`.
  - Use `route.patch_budget=6`, `ROUTE_MODE=paper_profile`, and `ROUTE_PATCH_BUDGET=6`.
  - Disable fallback, recompute, and default `patch_only` evaluation rows.
  - Keep Qwen-32B out of this validation pass.
- Local code/config updates:
  - `configs/eval_scd_icml_no_fallback_no_recompute.json` and `configs/eval_scd_icml_main_methods.json` now set `route.allow_patch_only=false`.
  - `scripts/build_release_tree.py` no longer includes `scripts/make_patch_meta_route.py` in the default preview.
  - `pyproject.toml` no longer exposes `search-blockwise-hybrid`, and `scripts/build_release_tree.py` excludes `src/scd_repro/cli/search_blockwise_hybrid.py` from the default preview.
  - `scripts/check_release_readiness.py` no longer requires or smokes `make_patch_meta_route.py` or strict-search validators for the default paper release.
  - `docs/scd-icml-open-source-release-status-2026-06-08.md` now records the paper mixed route as the release candidate.
  - `OPEN_SOURCE_RELEASE_MANIFEST.md` no longer lists `scripts/make_patch_meta_route.py` as a required release script.
- Local verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed after disabling `patch_only` in the paper-facing eval config.
- Remote run status to verify next:
  - Current intended run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
  - Current intended PID: `1248354`.
  - Expected run config: `route_mode=paper_profile`, `route_patch_budget=6`, `route_patch_layers=""`, `TRAIN_LIMIT=200`, `EVAL_LIMIT=200`, `DIRECTIONS=lite_to_7b`.
- Remaining gate:
  - Rebuild local/remote release previews after the release-scope cleanup.
  - Sync the latest local paper-scope cleanup to remote.
  - Wait for the 200-sample paper-mixed run to complete, then validate with `python scripts/check_release_readiness.py --root "$ROOT" --run-root <run-root>`.

## 2026-06-08 - Remote paper release resync verification

- Purpose: verify that the remote source and preview match the local paper-facing release scope after removing GitHub-style entry points from the default package.
- Sync:
  - Pushed `Experiment/core_code/runtime_repo` to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` with Aline.
- Remote verification:
  - `python scripts/check_release_readiness.py --root /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` passed.
  - `python scripts/build_release_tree.py --root /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo --out /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release --force` passed.
  - Remote preview contains `scripts/make_paper_scd_route.py`.
  - Remote preview does not contain `scripts/make_patch_meta_route.py`, `scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh`, `scripts/validate_strict_hybrid_search.py`, or `src/scd_repro/cli/search_blockwise_hybrid.py`.
  - Remote preview `pyproject.toml` does not expose `search-blockwise-hybrid`.
- Current 200-sample run status:
  - PID `1248354` is still running as `bash scripts/run_ser_scd_icml_mistral_repro.sh`.
  - Run root `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851` has no `scd_icml_mistral_summary.json` yet.
  - Log tail still shows GPU-gate waiting and automatic reselection among busy GPUs.
  - `run_config.json` records `route_mode=paper_profile`, `route_patch_budget=6`, `route_patch_layers=""`, `DIRECTIONS=lite_to_7b`, `TRAIN_LIMIT=200`, and `EVAL_LIMIT=200`.

## 2026-06-08 - Paper run status checker update

- Purpose: make the packaged status checker accurate for the paper mixed route.
- Code change:
  - `scripts/check_release_smoke_status.py` now describes itself as inspecting release smoke or reproduction runs.
  - `expected_stages()` now maps `route_mode=paper_profile` to `route_paper_scd`.
- Verification:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local release preview rebuild passed.
  - Synced the updated runtime tree to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py --root /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` passed.
  - Remote release preview rebuild passed.
  - Remote status command `python scripts/check_release_smoke_status.py --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851 --pid 1248354 --tail-lines 18` returned `state=waiting_gpu`, `pid_running=true`, expected stages including `lite_to_7b_route_paper_scd`, and `summary_json_exists=false`.
- GPU note:
  - All four GPUs are currently occupied by unrelated Python jobs under other project paths, so the paper run remains queued without starting model stages. No unrelated GPU process was killed.

## 2026-06-08 - Paper release audit artifact

- Purpose: preserve a concise audit of what is already verified and what remains unverified for the paper-facing release.
- Artifact:
  - `docs/scd-icml-paper-release-audit-2026-06-08.md`
- Contents:
  - Objective boundary: paper SCD, mixed Reuse + Patch, no fallback, no recompute, Qwen-32B deferred.
  - Code/package evidence: local source and preview readiness commands.
  - Release surface: included and excluded scripts/configs.
  - Eval and route contracts.
  - Remote status: run root, PID, `waiting_gpu`, no summary files.
  - Open gate: final `check_release_readiness.py --run-root` command after the 200-sample run completes.

## 2026-06-08 - No-fallback mixed k=6 fast evaluation

- Purpose: verify the paper-profile mixed route without fallback or recomputation after GPUs became available.
- Method:
  - Used the completed reuse/profile artifacts from `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
  - Used the existing patch checkpoint from `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mistral_repro_20260607_1605/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42`.
  - Generated route in `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_fast_mixed_oldpatch_20260608_131847`.
  - Route contract: `reuse_n=26`, `patch_n=6`, `fallback_n=0`, `recompute_n=0`, `route_semantics=paper_scd_sparse_patch_reuse_all_other_layers`.
  - Ran 200-sample evals for `cmrc2018`, `hotpotqa`, and `wikitext2` with methods `full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, and `scd_full`.
- Results:
  - Summary path: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_fast_mixed_oldpatch_20260608_131847/scd_icml_mistral_summary.json`.
  - CMRC2018: `full_consumer` F1 0.6573 / TTFT 129.27 ms; `reuse_only` F1 0.0089 / TTFT 52.11 ms; `scd_full` F1 0.0177 / TTFT 39.81 ms.
  - HotpotQA: `full_consumer` F1 0.0343 / TTFT 169.38 ms; `reuse_only` F1 0.0079 / TTFT 57.35 ms; `scd_full` F1 0.0079 / TTFT 50.55 ms.
  - WikiText-2: `full_consumer` PPL 7.5840 / TTFT 70.65 ms; `reuse_only` PPL 778.8231 / TTFT 31.08 ms; `scd_full` PPL 707.1308 / TTFT 30.84 ms.
- Conclusion:
  - The strict no-fallback k=6 mixed route is not a usable reproduction result: quality is close to reuse-only and far below full consumer.
  - The earlier strong `scd_full` result was not paper-faithful evidence because its route used fallback for non-patch layers.

## 2026-06-08 - Patch layer selection fix and corrected high-coverage patch training

- Purpose: repair the Mistral patch training path so high patch budgets actually train enough layers for no-fallback mixed SCD.
- Finding:
  - `configs/mistral_self_patch_xnorm_nativekv_shared_residual_mlp_alllayers_e1_h2048_r512_b2_20260404.json` had `patch_budget=32`.
  - `train_patch.py` stopped at `profile.recommended_patch_layers` when that list existed, so the previous "alllayers" checkpoint actually trained only layers `10,11,12,13,15,16,17,18`.
- Code change:
  - Updated `src/scd_repro/cli/train_patch.py::_select_layers` to use recommended layers first, then fill the remaining budget by descending `expected_gain`.
  - Applied the same correction to `src/scd_repro/direct_kv_residual_patch.py::_select_layers`, which is the selector used by the shared residual patch fit mode.
- Verification:
  - Local `python -m py_compile Experiment\core_code\runtime_repo\src\scd_repro\cli\train_patch.py` passed.
  - Local selection smoke returned `[2, 1, 3, 0]` for a synthetic profile with two recommended layers and `patch_budget=4`.
  - Synced the fixed file to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python -m py_compile src/scd_repro/cli/train_patch.py` and the same selection smoke passed.
  - Local and remote `python -m py_compile src/scd_repro/direct_kv_residual_patch.py` plus the same selector smoke passed.
- Current corrected run:
  - Stopped the obsolete k=6 patch process from `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
  - Launched `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mixed_patch31_reuse0_train_20260608_143847` with route `patch_layers=1..31`, `reuse_layers=[0]`, no fallback, and no recompute.
  - Training log confirms the run chose layers `0..31`; this run was launched before the shared residual selector was corrected, so it trains one extra patch layer. The eventual evaluation route will still use `1..31` patch plus layer 0 reuse to remain mixed.

## 2026-06-08 - Parallel non-shared patch shard training

- Purpose: use the newly freed GPUs to train a mergeable high-coverage Patch checkpoint faster, while leaving the slower shared-trunk all-layer Patch run on GPU0 as a backup.
- Code/config changes:
  - Added `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json` using `fit_mode=xnorm_nativekv_residual_mlp_v1`, `patch_budget=32`, `patch_num_epochs=1`, `patch_batch_size=128`, and `runtime.cpu_fallback=false`.
  - Added `scripts/merge_patch_ckpts.py` to merge independently trained per-layer patch checkpoint shards and reject non-layer-local shared-trunk keys.
  - Added the new config and merge script to `scripts/check_release_readiness.py` and `scripts/build_release_tree.py`.
- Verification:
  - Local `python -m py_compile Experiment\core_code\runtime_repo\scripts\merge_patch_ckpts.py Experiment\core_code\runtime_repo\scripts\check_release_readiness.py Experiment\core_code\runtime_repo\scripts\build_release_tree.py` passed.
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Synced the updated scripts/config to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python -m py_compile scripts/merge_patch_ckpts.py scripts/check_release_readiness.py scripts/build_release_tree.py` passed.
  - Remote `python scripts/check_release_readiness.py --root .` passed from the runtime repo.
- Run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_2gpu_20260608_152928`.
  - Source pairs/profile: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851/lite_to_7b`.
  - Target route after merge: `reuse_layers=[0]`, `patch_layers=1..31`, `fallback_layers=[]`, `selected_recompute_layers=[]`.
  - Shard `shard_01_16` launched on physical GPU1 with PID root `644505` and Python child `644542`.
  - Shard `shard_17_31` launched on physical GPU2 with PID root `644508` and Python child `644543`.
- Startup evidence:
  - GPU query showed GPU1 and GPU2 occupied by the two shard Python children, each using roughly 2.6-2.8 GB after model load.
  - Logs confirm selected layers `[1..16]` and `[17..31]` respectively.
  - Both jobs entered `xnorm native-kv stats progress` at chunk `1/200`; no checkpoint was complete at the time of this record.

### 2026-06-08 progress update

- Both parallel shards completed the 200/200 native-KV statistics pass.
- First linear-base layer evidence:
  - `shard_01_16`: `2026-06-08 15:58:27`, `linear base solved: layer=1`, `linear_base_kv_rel_mse=0.005274`.
  - `shard_17_31`: `2026-06-08 15:58:27`, `linear base solved: layer=17`, `linear_base_kv_rel_mse=0.050138`.
- No shard checkpoint files had been written yet at this checkpoint.
- Backup shared-trunk run `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mixed_patch31_reuse0_train_20260608_143847` was also still running and had reached `linear base solved: layer=5` by `2026-06-08 15:58:27`.
- Operational conclusion: the route is running correctly, but full checkpoint completion is still hours-scale because the current implementation performs per-layer 200-chunk error scans before residual training and does not save partial layer checkpoints.

### 2026-06-08 4-GPU restart

- Reason: the 2-GPU shard run was correct but each shard had 15-16 layers and would remain hours-scale. GPU3 was idle, and the GPU0 shared-trunk backup was not safely mergeable from shards.
- Action:
  - Stopped only the processes whose command lines matched:
    - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_2gpu_20260608_152928`
    - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mixed_patch31_reuse0_train_20260608_143847`
  - Confirmed all four GPUs were free afterward.
- New open-source helper:
  - Added `scripts/run_ser_scd_icml_parallel_patch_shards.sh`.
  - The helper creates shard routes, launches per-GPU shard training, and can optionally wait, merge checkpoints, generate the final paper-SCD route, and validate it.
  - Added the helper to `scripts/check_release_readiness.py` and `scripts/build_release_tree.py`.
- Verification:
  - Local `python -m py_compile Experiment\core_code\runtime_repo\scripts\check_release_readiness.py Experiment\core_code\runtime_repo\scripts\build_release_tree.py` passed.
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Synced the helper and release scripts to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `bash -n scripts/run_ser_scd_icml_parallel_patch_shards.sh` passed.
  - Remote `python scripts/check_release_readiness.py --root .` passed.
- Current 4-GPU run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331`.
  - Shards:
    - GPU0: `shard_01_08`, layers `1..8`, PID root `674376`, Python child `674504`.
    - GPU1: `shard_09_16`, layers `9..16`, PID root `674380`, Python child `674503`.
    - GPU2: `shard_17_24`, layers `17..24`, PID root `674388`, Python child `674502`.
    - GPU3: `shard_25_31`, layers `25..31`, PID root `674398`, Python child `674510`.
  - Startup logs confirm all four shard layer ranges.
  - GPU process query showed all four physical GPUs occupied by the four shard Python children.
  - All four shards entered `xnorm native-kv stats progress`; latest checked progress was `chunk 15/200` for each shard, with no Traceback or RuntimeError.
- Documentation/release sync:
  - Updated `README.md`, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` to document the high-coverage parallel Patch shard workflow.
  - Local source readiness and release-preview readiness passed after the documentation update.
  - Remote `bash -n scripts/run_ser_scd_icml_parallel_patch_shards.sh`, source readiness, and release-preview rebuild/readiness passed after syncing.
  - Verified the remote runtime root does not contain accidental root-level copies of the new script or release-check scripts.

## 2026-06-08 - Objective correction after GPU availability update

- Purpose: align the ongoing reproduction with the ICML paper objective and the user's clarification that the final route should be Reuse+Patch, not a hand-designed patch31 route.
- Correction:
  - `patch_layers=1..31` is retained only as a high-coverage Patch candidate pool / upper-bound diagnostic.
  - The final route objective is `maximize_reuse_subject_to_quality_gate`: generate fixed sparse top-k routes from the offline profile and evaluate them with the no-fallback/no-recompute strict runtime.
- Implementation updates:
  - `scripts/make_paper_scd_route.py` now fills profile ranking beyond `recommended_patch_layers` when a larger Patch budget is requested.
  - Added `scripts/make_paper_scd_candidate_routes.py` to emit fixed top-k candidate routes from a merged Patch pool.
  - Added `scripts/run_ser_scd_icml_eval_existing_route.sh` to evaluate an existing route with existing Reuse/Patch checkpoints without recapturing pairs or retraining.
  - Relaxed run-root release readiness from fixed `k=6` to a variable mixed paper-SCD route while still rejecting fallback, recompute, and all-patch routes.
- Current remote status:
  - Active run remains `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331`.
  - Shards are still in the statistics/training phase; no merged checkpoint or candidate evaluation result exists yet.

### 2026-06-08 remote sync and postprocess monitor

- Remote sync:
  - Synced updated route scripts, candidate route script, existing-route eval wrapper, release scripts, and release docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `bash -n scripts/run_ser_scd_icml_parallel_patch_shards.sh` passed.
  - Remote `bash -n scripts/run_ser_scd_icml_eval_existing_route.sh` passed.
  - Remote `python -m py_compile scripts/make_paper_scd_route.py scripts/make_paper_scd_candidate_routes.py scripts/check_release_readiness.py scripts/build_release_tree.py` passed.
  - Remote `python scripts/check_release_readiness.py --root .` passed.
  - Remote release preview rebuild passed.
- Current shard status:
  - All four shard logs reached `xnorm native-kv stats progress: chunk 200/200`.
  - No shard checkpoint file has been written yet.
  - Processes remain alive and CPU-active in the post-stat linear solve stage; no Python exception is visible.
- Background postprocess monitor:
  - PID: `713848`.
  - Script: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331/logs/postprocess_quality_gate/postprocess_quality_gate.sh`.
  - Log: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331/logs/postprocess_quality_gate/postprocess.log`.
  - Behavior: wait for all four shard `meta.json` files, merge the Patch pool, generate candidate routes for budgets `5,6,8,12,16,24,31`, validate them, then run CMRC2018 50-sample probes for `k=31` and `k=16` on disjoint GPU pairs.

### 2026-06-08 quality-gated selector utility

- Purpose: make the final route selection explicit instead of manually comparing candidate eval summaries.
- Added `scripts/select_quality_gated_scd_route.py`.
- Local verification:
  - `python -m py_compile Experiment\core_code\runtime_repo\scripts\select_quality_gated_scd_route.py Experiment\core_code\runtime_repo\scripts\check_release_readiness.py Experiment\core_code\runtime_repo\scripts\build_release_tree.py` passed.
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local release preview rebuild passed.
- Remote verification:
  - Synced selector, release scripts, and release docs to the remote runtime.
  - Remote `python -m py_compile scripts/select_quality_gated_scd_route.py scripts/check_release_readiness.py scripts/build_release_tree.py` passed.
  - Remote `python scripts/check_release_readiness.py --root .` passed.
  - Remote release preview rebuild passed.
- Background selector monitor:
  - PID: `719541`.
  - Log: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331/logs/postprocess_quality_gate/select_after_probes.log`.
  - Behavior: wait for candidate eval summaries, then generate route-selection reports for CMRC2018 F1 retention thresholds 0.95 and 0.90.

### 2026-06-08 17:15 remote training progress

- All four shard processes remain alive.
- First post-stat linear-base evidence appeared at `2026-06-08 17:11:32`:
  - `shard_01_08`: layer 1, `linear_base_kv_rel_mse=0.005274`.
  - `shard_09_16`: layer 9, `linear_base_kv_rel_mse=0.045118`.
  - `shard_17_24`: layer 17, `linear_base_kv_rel_mse=0.050138`.
  - `shard_25_31`: layer 25, `linear_base_kv_rel_mse=0.035458`.
- Interpretation: the training is not deadlocked. It has finished the all-chunk native-KV statistics pass and is now in per-layer linear-base/error/residual work. No shard checkpoint exists yet, so merge and candidate evals are still pending.

### 2026-06-08 early middle-shard sparse candidate monitor

- Purpose: avoid waiting for all layers `1..31` if a much smaller sparse Patch set already passes the quality gate.
- Rationale: profile top layers for `k=8` and `k=11` are contained in the middle shard checkpoints `shard_09_16` and `shard_17_24`; these can be merged and evaluated before `shard_01_08` and `shard_25_31` finish.
- Background monitor:
  - PID: `726484`.
  - Log: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331/logs/postprocess_quality_gate/early_top8_top11_probe.log`.
  - Behavior: wait for `shard_09_16/meta.json` and `shard_17_24/meta.json`, merge a middle-layer Patch pool, generate candidate routes for budgets `5,6,8,10,11`, run CMRC2018 50-sample probes for `k=8` and `k=11`, then write 0.95 and 0.90 F1-retention route-selection reports.

### 2026-06-08 linear-only Patch fallback support

- Purpose: provide a faster fallback Patch-pool path if the residual MLP shard run proves too slow or fails.
- Code change:
  - `src/scd_repro/direct_kv_residual_patch.py` now supports `patch.xnorm_nativekv_linear_only=true` for `fit_mode=xnorm_nativekv_residual_mlp_v1`.
  - In this mode, training saves `direct_wk.<layer>` and `direct_wv.<layer>` after direct xnorm-to-native-KV solving and skips residual MLP/error refinement.
  - Runtime compatibility is preserved because `residual_layers=[]` makes the existing xnorm-native-KV runtime use the direct linear Patch path.
- Config:
  - Added `configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json`.
- Verification:
  - Local py-compile, release readiness, and release preview rebuild passed.
  - Synced the code/config/docs to the remote runtime.
  - Remote py-compile, release readiness, and release preview rebuild passed.
- Operational decision:
  - Did not launch the linear-only fallback yet. At `2026-06-08 17:37`, the remote load average was about `104/124/134`, and the active residual shard processes were still alive. Launching another training run would likely slow the active path.

### 2026-06-08 corrected 2-GPU linear top11 backup launch

- Purpose: use two newly available GPUs to run a faster linear-only Patch diagnostic for the profile top11 layer set while the main residual candidate-pool run continues.
- Correction:
  - The first linear-only code insertion had been placed in `train_direct_kv_knorm_residual_patch`, which was the wrong training function for the xnorm-native-KV config.
  - The corrected implementation puts `xnorm_nativekv_linear_only` inside `train_xnorm_nativekv_residual_patch`.
  - Local assertion confirmed `xnorm_nativekv_linear_only` is absent from the direct-knorm block and present in the xnorm-native-KV block.
- Verification:
  - Local `python -m py_compile Experiment/core_code/runtime_repo/src/scd_repro/direct_kv_residual_patch.py ...` passed.
  - Local `python Experiment/core_code/runtime_repo/scripts/check_release_readiness.py --root Experiment/core_code/runtime_repo` passed.
  - Remote `python -m py_compile src/scd_repro/direct_kv_residual_patch.py` passed.
  - Remote assertions confirmed `direct_has_linear_only=False`, `xnorm_has_linear_only=True`, `xnorm_has_save_log=True`, and config `linear_only=True`.
  - Remote `python scripts/check_release_readiness.py --root .` passed.
- Run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_linear_top11_2gpu_20260608_175651`.
  - Source run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
  - Config: `configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json`.
  - Shards: `shard_10_15` on GPU3 and `shard_16_20` on GPU0.
  - Launch evidence: `shard_10_15` started with `selected=[10, 11, 12, 13, 14, 15] residual_layers=[]`; `shard_16_20` started with `selected=[16, 17, 18, 19, 20] residual_layers=[]`.
- Background postprocess:
  - PID: `750655`.
  - Log: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_linear_top11_2gpu_20260608_175651/logs/postprocess_linear_top11/stdout.log`.
  - Behavior: wait for both shard `meta.json` files, merge Patch checkpoints, generate candidate routes for k=`5,6,8,10,11`, validate routes, run CMRC2018 50-sample probes, and write 0.95/0.90 F1-retention selection reports.
- Concurrent main residual status:
  - Main run `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331` still has no shard `meta.json`.
  - Latest checked residual logs show each shard has solved at least two post-stat linear-base layers with no visible Python exception.
  - Existing main postprocess monitors continue waiting; no candidate eval summary exists yet.

### 2026-06-08 18:01 linear top11 progress check

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_linear_top11_2gpu_20260608_175651`.
- Both shards finished loading weights and entered the xnorm native-KV statistics pass.
- Progress evidence:
  - `shard_10_15`: `xnorm native-kv stats progress: chunk 5/200 elapsed=35.8s eta=1395.1s`.
  - `shard_16_20`: `xnorm native-kv stats progress: chunk 5/200 elapsed=35.5s eta=1385.5s`.
- No shard `meta.json` exists yet.
- Postprocess PID `750655` is still waiting for shard checkpoints.

### 2026-06-08 18:27 linear top11 result and stop decision

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_linear_top11_2gpu_20260608_175651`.
- Checkpoint status:
  - `shard_10_15` and `shard_16_20` both reached `chunk 200/200`.
  - Both saved `xnorm native-kv linear-only patch checkpoint` at about `2026-06-08 18:13:18`.
  - The merged linear Patch checkpoint was written to `lite_to_7b/ckpt_patch/patch_cmrc2018_seed42`.
- CMRC2018 50-sample probes:
  - `k=5`, patch layers `[13, 15, 16, 17, 18]`: full-consumer F1 `0.618514853953628`, reuse-only F1 `0.009670731337398004`, SCD F1 `0.013126948680429155`, F1 retention `0.021223336184280746`.
  - `k=11`, patch layers `[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`: full-consumer F1 `0.618514853953628`, reuse-only F1 `0.009670731337398004`, SCD F1 `0.01153836434233078`, F1 retention `0.018654951079308837`.
- Interpretation:
  - Linear-only Patch is not a viable final path for the strict Reuse+Patch paper route.
  - The largest trained linear candidate `k=11` fails badly, so smaller linear candidates cannot satisfy the quality-retention gate.
- Action:
  - Stopped the remaining linear-only postprocess and eval processes for this run, including in-progress `k=6` and `k=8` probes.

### 2026-06-08 18:30 stop broad patch31 residual run

- Stopped run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331`.
- Reason:
  - No shard `meta.json`, candidate route, or summary artifact existed after more than two hours.
  - Four Python shard children were each consuming roughly 17 CPU cores.
  - The run trained many layers outside the likely sparse top-k route, while the current objective is maximum Reuse subject to quality.
- Evidence before stop:
  - `MAIN ARTIFACTS BEFORE STOP` was empty.
  - `load average` was about `105/117/114` near stop time.
- Evidence after stop:
  - No processes remained matching the patch31 run root.
  - All four GPUs returned to about `18 MiB` used.

### 2026-06-08 focused residual top11 launch

- Purpose: train a residual-MMLP Patch pool only for profile top11 layers, which directly supports paper-SCD sparse candidates while avoiding patch31 overhead.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_4gpu_20260608_183110`.
- Source run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
- Config: `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json`.
- Shards:
  - GPU0 `shard_10_12`, selected/residual layers `[10, 11, 12]`, root PID `776234`.
  - GPU1 `shard_13_15`, selected/residual layers `[13, 14, 15]`, root PID `776236`.
  - GPU2 `shard_16_18`, selected/residual layers `[16, 17, 18]`, root PID `776240`.
  - GPU3 `shard_19_20`, selected/residual layers `[19, 20]`, root PID `776248`.
- Background postprocess:
  - PID: `776263`.
  - Log: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_4gpu_20260608_183110/logs/postprocess_residual_top11/stdout.log`.
  - Behavior: wait for all shard `meta.json` files, merge Patch checkpoints, generate k=`5,6,8,10,11` paper-SCD routes, validate them, run CMRC2018 50-sample probes in two-GPU batches, and write 0.95/0.90 F1-retention selection reports.

### 2026-06-08 eval wrapper install guard

- Purpose: reduce overhead for multi-candidate evaluations in the residual top11 postprocess.
- Change: `scripts/run_ser_scd_icml_eval_existing_route.sh` now has `INSTALL_EDITABLE=${INSTALL_EDITABLE:-0}` and only runs `python -m pip install -e .` when `INSTALL_EDITABLE=1`.
- Rationale: the wrapper already exports `PYTHONPATH=${ROOT}/src`, so editable install is not required for normal source-tree execution.
- Verification:
  - Local release readiness passed.
  - Remote `bash -n scripts/run_ser_scd_icml_eval_existing_route.sh` passed.
  - Remote assertions confirmed the install guard and conditional install are present.
  - Remote release readiness passed.

### 2026-06-08 18:39 residual top11 progress check

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_4gpu_20260608_183110`.
- All four shards are alive and reached `xnorm native-kv stats progress: chunk 100/200`.
- No shard `meta.json` exists yet.
- The focused run is behaving as expected and is much lighter than the stopped patch31 run; load average dropped from about `105/117/114` near the stop decision to about `5.6/39/79` during this check.

### 2026-06-08 skip linear-base error stats and relaunch residual top11

- Motivation:
  - After stats reached `chunk 200/200`, the first focused residual top11 run entered the same slow per-layer direct-linear error-statistics phase that made patch31 inefficient.
  - Code inspection showed this phase only populates `linear_base_*` diagnostics before residual training. The residual trainer itself computes `base_h_rel_mse`, `base_k_rel_mse`, and `base_v_rel_mse` in its init/eval pass.
- Code change:
  - Added `patch.skip_linear_base_error_stats` handling in `train_xnorm_nativekv_residual_patch`.
  - When enabled, the code initializes per-layer metric dictionaries, logs that the linear-base error stats are skipped, and proceeds directly to residual layer training.
  - Added `skip_linear_base_error_stats: true` to `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json`.
- Verification:
  - Local `python -m py_compile Experiment/core_code/runtime_repo/src/scd_repro/direct_kv_residual_patch.py` passed.
  - Local release readiness passed.
  - Remote `python -m py_compile src/scd_repro/direct_kv_residual_patch.py` passed.
  - Remote assertions confirmed the config flag and skip log string are present.
  - Remote release readiness passed.
- Stopped superseded run:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_4gpu_20260608_183110`.
  - It had no shard `meta.json` or summary artifacts before stop.
- Relaunched run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_skiplin_4gpu_20260608_185457`.
  - Shards: GPU0 `10..12`, GPU1 `13..15`, GPU2 `16..18`, GPU3 `19..20`.
  - Postprocess PID: `793134`.
  - Launch evidence confirms all four shards start in residual mode, e.g. `selected=[10, 11, 12] residual_layers=[10, 11, 12]`.
  - Remote config check confirms `skip_linear_base_error_stats=True`.

### 2026-06-08 residual top11 skiplin quality-gate result and correction

- Completed run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_skiplin_4gpu_20260608_185457`.
- Route contract:
  - Candidate routes k=`5,6,8,10,11` all passed mixed-route validation.
  - Every candidate had `fallback_layers=[]` and `selected_recompute_layers=[]`.
- CMRC2018 50-sample results:
  - `k=5`, patch_n `5`, reuse_n `27`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.0158405693388955`, retention `0.025610653062962844`.
  - `k=6`, patch_n `6`, reuse_n `26`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.013341182251744622`, retention `0.021569703890643912`.
  - `k=8`, patch_n `8`, reuse_n `24`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.02027632609985551`, retention `0.03278227834019923`.
  - `k=10`, patch_n `10`, reuse_n `22`: full F1 `0.618514853953628`, SCD F1 `0.025071170136348277`, retention `0.040534467322959296`.
  - `k=11`, patch_n `11`, reuse_n `21`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.024266119030749017`, retention `0.03923287997957818`.
- Analysis:
  - The run is a valid strict Reuse+Patch control, but it fails the quality gate.
  - Inspection of residual training logs showed each layer reported `base_* -> base_*` after training.
  - Code inspection found `best_state` was chosen by total loss including delta regularization. Since the identity initialization had `delta_reg=0`, the checkpoint kept the identity state even when one epoch slightly improved h/k/v reconstruction.
- Code/config correction:
  - `direct_kv_residual_patch.py`: residual best-state selection now uses `quality_loss` rather than regularized `total_loss`.
  - `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json`: `patch_num_epochs=3`, `lambda_delta=1e-5`.
  - Verification: local `py_compile` passed; remote `py_compile` passed; remote config and source assertions confirmed.
- Relaunched corrected run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_4gpu_20260608_201408`.
  - Shards: GPU0 `10..12`, GPU1 `13..15`, GPU2 `16..18`, GPU3 `19..20`.
  - Postprocess PID: `856782`.
  - Initial status at `2026-06-08T20:15:56+08:00`: all four shards alive and reached `xnorm native-kv stats progress: chunk 1/200`; postprocess is waiting for shard `meta.json` files.

### 2026-06-08 teacher-aligned route-selection implementation

- Purpose: replace fixed-k/top-k layer selection as the main route-selection method while preserving it as a baseline.
- Method:
  - Added route schema with per-consumer-layer `{action, source_layer}` records, plus legacy route conversion for the existing strict runtime.
  - Added source-target layer mapping with same-index and proportional strategies.
  - Added teacher objective utilities for KV drift, hidden drift, logit KL, PPL delta, and cost.
  - Added `ConditionalReusePruner`, which starts from a high-quality route and accepts Patch-to-Reuse changes only when route-level teacher metrics remain within constraints.
  - Added pairwise interaction calculation and a proxy beam selector scaffold.
- New files:
  - `Experiment/core_code/runtime_repo/src/scd_repro/route_selection/`
  - `Experiment/core_code/runtime_repo/scripts/estimate_layer_mapping.py`
  - `Experiment/core_code/runtime_repo/scripts/profile_routes.py`
  - `Experiment/core_code/runtime_repo/scripts/select_patch_reuse_route.py`
  - `Experiment/core_code/runtime_repo/scripts/evaluate_selected_route.py`
  - `Experiment/core_code/runtime_repo/configs/route_selection.yaml`
  - `Experiment/core_code/runtime_repo/tests/test_route_serialization.py`
  - `Experiment/core_code/runtime_repo/tests/test_layer_mapping.py`
  - `Experiment/core_code/runtime_repo/tests/test_metrics.py`
  - `Experiment/core_code/runtime_repo/tests/test_selector.py`
- Verification:
  - Local py-compile for route-selection modules and scripts passed.
  - Local `python -m unittest discover -s Experiment/core_code/runtime_repo/tests -v` passed 10 tests.
  - Local `python Experiment/core_code/runtime_repo/scripts/check_release_readiness.py --root Experiment/core_code/runtime_repo` passed.
  - Remote sync to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo` completed.
  - Remote py-compile, remote 10-test unittest run, remote release readiness, and remote layer-mapping smoke passed.
- Result boundary:
  - This is a code/schema/selector implementation result, not a new paper metric.
  - Missing route-level teacher metrics remain missing; the selector writes pending routes instead of inventing values.

### 2026-06-08 residual top11 qualitybest gpupreload threads2 progress

- Purpose: verify the corrected residual best-state selection under GPU-side preload and low CPU thread caps.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_gpupreload_threads2_4gpu_20260608_211359`.
- Source run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
- Config: `configs/mistral_self_patch_xnorm_nativekv_residual_mlp_parallel_e1_h2048_b2_20260404.json` with `patch_num_epochs=3`, `lambda_delta=1e-5`, and `skip_linear_base_error_stats=true`.
- Shards:
  - GPU0 `shard_10_12`
  - GPU1 `shard_13_15`
  - GPU2 `shard_16_18`
  - GPU3 `shard_19_20`
- Status:
  - All shards reached `xnorm native-kv stats progress: chunk 200/200`.
  - All shards skipped linear-base diagnostics.
  - First residual layer per shard completed with trained metrics lower than base metrics:
    - Layer 10: `h/k/v` rel-MSE `0.194485/0.068328/0.220374 -> 0.184139/0.056349/0.177670`.
    - Layer 13: `0.249286/0.091023/0.321349 -> 0.237137/0.076298/0.260241`.
    - Layer 16: `0.267039/0.126224/0.283977 -> 0.256916/0.112838/0.249165`.
    - Layer 19: `0.206658/0.096698/0.157953 -> 0.200206/0.089301/0.145065`.
- Conclusion:
  - The previous identity-checkpoint failure is fixed for the first trained layers.
  - The run is still in progress; no candidate eval summary or quality-gate result exists yet.

### 2026-06-08 residual top11 qualitybest gpupreload threads2 final result

- Completed run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_gpupreload_threads2_4gpu_20260608_211359`.
- Source run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
- Route contract:
  - Candidate routes k=`5,6,8,10,11` all passed strict mixed-route validation.
  - Every candidate used `fallback_layers=[]` and `selected_recompute_layers=[]`.
- CMRC2018 50-sample results:
  - `k=5`, patch layers `[13,15,16,17,18]`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.01706453941473071`, retention `0.0275895385626585`.
  - `k=6`, patch layers `[10,13,15,16,17,18]`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.009785339346314956`, retention `0.01582070225761885`.
  - `k=8`, patch layers `[10,11,12,13,15,16,17,18]`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.018854279380595168`, retention `0.030483147268131304`.
  - `k=10`, patch layers `[10,11,12,13,14,15,16,17,18,19]`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.028996433933126977`, retention `0.04688073980402891`.
  - `k=11`, patch layers `[10,11,12,13,14,15,16,17,18,19,20]`: full F1 `0.618514853953628`, reuse F1 `0.009670731337398004`, SCD F1 `0.02215067504574479`, retention `0.035812680817857115`.
- Quality gate:
  - No candidate passed the 95% or 90% full-consumer F1 retention gate.
  - Highest observed candidate was `k=10`, but retention was only `4.69%`.
- Analysis:
  - The result is a valid strict Reuse+Patch negative result, not caused by fallback or recompute leakage.
  - Increasing k within the trained top11 patch set does not recover quality.
  - Next diagnostic should test a real all-layer patch-only baseline before further mixed route search, because the current failure could be caused by insufficient patch coverage or by the patch model itself rather than by the route budget alone.

### 2026-06-08 all-layer patch-only diagnostic launch

- Purpose: test whether Patch itself can recover full-consumer quality before continuing mixed Patch+Reuse route search.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_residual_e3_d1e5_2gpu_20260608_223357`.
- Source run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`.
- Method:
  - Train all 32 consumer layers with `xnorm_nativekv_residual_mlp_v1`.
  - Use `TRAIN_LIMIT=200`, `patch_num_epochs=3`, `lambda_delta=1e-5`, `skip_linear_base_error_stats=true`.
  - Two-GPU sharding:
    - GPU0: layers `0..15`, shard `shard_00_15`.
    - GPU1: layers `16..31`, shard `shard_16_31`.
- Initial status:
  - Both shards reached `xnorm native-kv stats progress: chunk 200/200`.
  - Both shards entered residual training.
  - First completed layers improved reconstruction:
    - Layer 0: `h/k/v` rel-MSE `0.000028/0.000016/0.000058 -> 0.000013/0.000003/0.000078`.
    - Layer 16: `0.267039/0.126224/0.283977 -> 0.256923/0.112838/0.249151`.
    - Layer 1: `0.016270/0.008201/0.029382 -> 0.011221/0.003637/0.008382`.
    - Layer 17: `0.224687/0.080854/0.201413 -> 0.216787/0.073159/0.179751`.
- Postprocess:
  - Watcher PID `989167` waits for both shard `meta.json` files.
  - After merge, it will evaluate an all-patch diagnostic route on CMRC2018 50 samples with `full_consumer`, `reuse_only`, `patch_only`, and `scd_full`.
- Result boundary:
  - No patch-only quality result exists yet for this run.

### 2026-06-09 all-layer patch-only diagnostic final result

- Completed run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_residual_e3_d1e5_2gpu_20260608_223357`.
- Evaluation output: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_residual_e3_d1e5_2gpu_20260608_223357/lite_to_7b/eval_allpatch_patchonly_cmrc50_streaming_20260609_022458`.
- Runtime fix:
  - Patched `src/scd_repro/cli/eval_cmrc_strict.py` and `src/scd_repro/cli/eval_text_strict.py`.
  - Added runtime eval controls for `patch_ckpt_device`, `runtime_patch_window_size`, `runtime_patch_persist_shards`, and `runtime_patch_persist_modules`.
  - The successful run used `patch_ckpt_device=cpu`, `runtime_patch_window_size=1`, `runtime_patch_persist_shards=false`, and `runtime_patch_persist_modules=false`.
  - This avoided the earlier prewarm/all-layer GPU residency OOM by using the existing runtime patch window path.
- Verification:
  - Local py-compile for the two patched CLI files passed.
  - Local `python -m unittest discover -s Experiment/core_code/runtime_repo/tests -v` passed 10 tests.
  - Local `python Experiment/core_code/runtime_repo/scripts/check_release_readiness.py --root Experiment/core_code/runtime_repo` passed.
  - Remote py-compile for the two patched CLI files passed.
  - Remote 3-sample smoke test completed without OOM.
  - Remote 50-sample run completed with exit status `0`; wall time `26:16.36`; maximum RSS `20223808` KB.
- CMRC2018 50-sample results:
  - `full_consumer`: F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`, TTFT `129.34527330100536` ms, total `877.8296566661447` ms.
  - `reuse_only`: F1 `0.009670731337398004`, EM `0.0`, avg NLL `7.625489065008026`, TTFT `46.43352407030761` ms, total `732.5903952866793` ms.
  - `patch_only`: F1 `0.788977574686301`, EM `0.52`, avg NLL `0.6808870162091324`, TTFT `27668.041981831193` ms, total `27901.00043475628` ms, patch infer `27644.523684075102` ms.
- Analysis:
  - Patch-only quality is strong on CMRC50 and exceeds the full-consumer baseline in F1, EM, and NLL for this sample window.
  - The previous strict mixed Reuse+Patch negative result is not explained by patch model incapacity alone; it is more likely caused by route coverage/interaction when many non-patch layers are supplied by reuse.
  - The current all-patch runtime is diagnostic-only and too slow for serving because it streams 32 residual patch modules from CPU each sample.
  - Next route-search step should use teacher-aligned route selection over Patch vs Reuse decisions, with patch-only/all-patch as an upper-bound diagnostic and strict Reuse+Patch as the target serving contract.

### 2026-06-09 conditional Reuse pruning prefix screen

- Purpose: implement and exercise the new route-selection direction from a high-quality all-Patch route toward more Reuse, instead of fixed-k selection from an all-Reuse baseline.
- Code updates:
  - Added/updated `src/scd_repro/route_selection/` route schema, mapping, metrics, conditional pruner, pairwise interaction helper, and proxy beam selector.
  - Added/updated CLI entry points: `estimate_layer_mapping.py`, `profile_routes.py`, `select_patch_reuse_route.py`, `evaluate_selected_route.py`.
  - Updated `collect_route_eval_metrics.py` and `select_quality_gated_scd_route.py` so strict eval outputs can be gated by real `f1_retention`, `avg_nll_delta`, and TTFT proxy fields when KV/hidden teacher drift has not yet been profiled.
  - Updated `configs/route_selection.yaml`, README, and `REPRODUCING_SCD_ICML.md` to mark fixed-k as baseline/diagnostic rather than the main selector.
- Verification:
  - Local `python -m unittest discover -s Experiment/core_code/runtime_repo/tests -v` passed 12 tests.
  - Local `python Experiment/core_code/runtime_repo/scripts/check_release_readiness.py --root Experiment/core_code/runtime_repo` passed.
  - Synced updated files to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote 12-test unittest and release readiness both passed.
- Prefix screen:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_prefix_cmrc10_20260609_033528`.
  - Base route: all-Patch route from `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_residual_e3_d1e5_2gpu_20260608_223357/lite_to_7b/route/route_allpatch_cmrc2018_seed42.json`.
  - Patch checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_residual_e3_d1e5_2gpu_20260608_223357/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42`.
  - Candidate prefix budgets: Reuse counts `4,8,12,16,20,24,28,31`, with all remaining layers Patch and no fallback/recompute.
  - Eval: CMRC2018 10 samples, full-consumer plus strict `scd_full` only.
- CMRC2018 10-sample results:
  - Reuse 4 / Patch 28: F1 retention `1.116125`, avg NLL delta `-0.040500`, SCD TTFT `23930.2` ms.
  - Reuse 8 / Patch 24: F1 retention `0.960733`, avg NLL delta `0.020474`, SCD TTFT `20947.9` ms.
  - Reuse 12 / Patch 20: F1 retention `0.905411`, avg NLL delta `0.072887`, SCD TTFT `17150.2` ms.
  - Reuse 16 / Patch 16: F1 retention `0.940596`, avg NLL delta `0.201217`, SCD TTFT `13722.3` ms.
  - Reuse 20 / Patch 12: F1 retention `0.464258`, avg NLL delta `1.554404`, SCD TTFT `10303.1` ms.
  - Reuse 24 / Patch 8: F1 retention `0.438243`, avg NLL delta `2.642855`, SCD TTFT `6969.5` ms.
  - Reuse 28 / Patch 4: F1 retention `0.022326`, avg NLL delta `6.320401`, SCD TTFT `3487.9` ms.
  - Reuse 31 / Patch 1: F1 retention `0.028964`, avg NLL delta `6.547634`, SCD TTFT `170.6` ms.
- Selection:
  - Gate: F1 retention `>=0.95`, avg NLL delta `<=0.05`.
  - Selected route: Reuse 8 / Patch 24.
  - Selection report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_prefix_cmrc10_20260609_033528/lite_to_7b/conditional_report/quality_gate_f1ret095_nll005.json`.
  - Copied selected route: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_prefix_cmrc10_20260609_033528/lite_to_7b/conditional_report/selected_route_f1ret095_nll005.json`.
- Follow-up launched:
  - Parent run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_quality_gate_cmrc50_20260609_0350`.
  - Reuse 8 / Patch 24 CMRC50: PID `1175765`, GPUs `0,1`.
  - Reuse 4 / Patch 28 CMRC50: PID `1175767`, GPUs `2,3`.
  - Both runs use no fallback/recompute, `patch_ckpt_device=cpu`, `runtime_patch_window_size=1`, and 50 CMRC2018 samples.

### 2026-06-09 conditional route CMRC50 gate result

- Parent run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_quality_gate_cmrc50_20260609_0350`.
- Completed routes:
  - Reuse 8 / Patch 24:
    - Reuse layers `[1,8,9,11,13,21,24,25]`.
    - Patch layers `[0,2,3,4,5,6,7,10,12,14,15,16,17,18,19,20,22,23,26,27,28,29,30,31]`.
    - Full-consumer F1 `0.618514853953628`; SCD F1 `0.6426122791569092`; F1 retention `1.0389601398402113`.
    - Full avg NLL `0.7055898962204568`; SCD avg NLL `1.4437272491271385`; avg NLL delta `0.7381373529066817`.
    - Full TTFT `130.28515726327896` ms; SCD TTFT `20546.695250924677` ms.
  - Reuse 4 / Patch 28:
    - Reuse layers `[1,8,13,25]`.
    - Patch layers `[0,2,3,4,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31]`.
    - Full-consumer F1 `0.618514853953628`; SCD F1 `0.7185760200299643`; F1 retention `1.16177649645232`.
    - Full avg NLL `0.7055898962204568`; SCD avg NLL `1.2893416688300634`; avg NLL delta `0.5837517726096066`.
    - Full TTFT `148.20410958491266` ms; SCD TTFT `24103.794098114595` ms.
- Quality-gate result:
  - Gate: F1 retention `>=0.95`, avg NLL delta `<=0.05`.
  - No CMRC50 candidate passed because both routes strongly worsened avg NLL despite high F1.
  - Report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_quality_gate_cmrc50_20260609_0350/quality_gate_f1ret095_nll005.json`.
- Analysis:
  - 10-sample prefix screening overestimated NLL stability; CMRC50 shows that even 4-8 reused layers create substantial likelihood drift.
  - F1 alone is not sufficient for the teacher-aligned route objective.
  - Current strict CPU-window runtime remains slow for any high-patch route; this run is quality-selection evidence, not a final latency result.
- Follow-up launched:
  - Low-Reuse CMRC50 screen root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418`.
  - Prefix budgets: Reuse counts `1,2,3`.
  - PID `1193848`; Reuse 1 and Reuse 2 started in parallel, Reuse 3 queued.

### 2026-06-09 low-Reuse CMRC50 gate result

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418`.
- Eval: CMRC2018 50 samples, no fallback/recompute, full-consumer plus strict `scd_full`.
- Results:
  - Reuse 1 / Patch 31:
    - Reuse layers `[8]`.
    - SCD F1 `0.8263109183531486`; full F1 `0.618514853953628`; retention `1.3359596993851657`.
    - SCD avg NLL `0.666978841225944`; full avg NLL `0.7055898962204568`; avg NLL delta `-0.03861105499451278`.
    - SCD TTFT `26522.2902063746` ms.
  - Reuse 2 / Patch 30:
    - Reuse layers `[8,25]`.
    - SCD F1 `0.8130814875493471`; full F1 `0.618514853953628`; retention `1.3145706725586679`.
    - SCD avg NLL `0.658217633134098`; full avg NLL `0.7055898962204568`; avg NLL delta `-0.04737226308635878`.
    - SCD TTFT `25633.967208201066` ms.
  - Reuse 3 / Patch 29:
    - Reuse layers `[1,8,25]`.
    - SCD F1 `0.747605874091`; full F1 `0.618514853954`; retention `1.208711269119`.
    - SCD avg NLL `1.251411486208`; full avg NLL `0.705589896220`; avg NLL delta `0.545821589987`.
    - SCD TTFT `25082.767` ms.
- Quality-gate selection:
  - Gate: F1 retention `>=0.95`, avg NLL delta `<=0.05`.
  - Selected route: Reuse 2 / Patch 30.
  - Selected route path: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418/lite_to_7b/conditional_report/selected_route_f1ret095_nll005.json`.
  - Selection report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418/lite_to_7b/conditional_report/quality_gate_f1ret095_nll005.json`.
- Analysis:
  - Current CMRC50 evidence supports only two safe Reuse layers under the F1+NLL gate: layers `8` and `25`.
  - Adding layer `1` as the third Reuse layer causes a large avg NLL jump, so independent/single-layer ranking is not sufficient.
  - The selected route is quality-safe by the current proxy gate but still slow because it uses 30 Patch layers with CPU-window streaming.
  - Next route work should focus on interaction checks around layers `1`, `8`, and `25`, plus runtime optimization for many-Patch routes if this high-patch route remains the only quality-safe path.

### 2026-06-09 pairwise interaction check for layers 1, 8, 25

- Purpose: test whether layer `1` fails by itself or mainly through interaction with the currently safe Reuse set `[8,25]`.
- Code updates:
  - `scripts/generate_conditional_reuse_routes.py` now supports `--reuse-sets`, a semicolon-separated explicit set list such as `1;25;1,8;1,25`.
  - `scripts/run_ser_scd_icml_conditional_route_screen.sh` now passes `REUSE_SETS` through to the generator.
  - Added `scripts/report_pairwise_interactions.py` to compute pairwise interaction tables from completed route metrics without imputing missing routes.
  - Updated README, `REPRODUCING_SCD_ICML.md`, `OPEN_SOURCE_RELEASE_MANIFEST.md`, release preview builder, and release readiness checks.
- Verification:
  - Local py-compile for changed/new scripts passed.
  - Local unittest passed 12 tests.
  - Local release readiness passed.
  - Remote `bash -n`, py-compile, release readiness, and `REUSE_SETS` smoke passed.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_pairwise_1_8_25_cmrc50_20260609_0525`.
- New CMRC50 routes:
  - Reuse `[1]`: F1 retention `1.183420`, avg NLL delta `0.331580`, TTFT `26427.1` ms.
  - Reuse `[25]`: F1 retention `1.267685`, avg NLL delta `-0.034370`, TTFT `26410.8` ms.
  - Reuse `[1,8]`: F1 retention `1.193779`, avg NLL delta `0.522831`, TTFT `25620.7` ms.
  - Reuse `[1,25]`: F1 retention `1.196565`, avg NLL delta `0.337742`, TTFT `25614.0` ms.
- Reused existing CMRC50 routes:
  - Reuse `[8]`: avg NLL delta `-0.038611`.
  - Reuse `[8,25]`: avg NLL delta `-0.047372`.
  - Reuse `[1,8,25]`: avg NLL delta `0.545822`.
- Pairwise interaction report:
  - Output JSON: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_pairwise_1_8_25_cmrc50_20260609_0525/lite_to_7b/conditional_report/pairwise_interactions_avg_nll.json`.
  - Output CSV: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_pairwise_1_8_25_cmrc50_20260609_0525/lite_to_7b/conditional_report/pairwise_interactions_avg_nll.csv`.
  - Base objective: all-Patch avg NLL delta `-0.0247028792453244`.
  - Interaction `(1,8)`: `+0.20515969917119706`, conflict under threshold `0.05`.
  - Interaction `(1,25)`: `+0.015830123023312253`, not a pairwise conflict by threshold, but layer `1` alone already fails the NLL gate.
  - Interaction `(8,25)`: `+0.0009062659497809759`, safe pair.
- Analysis:
  - Layer `25` is individually safe, and layer `8` plus `25` is also safe.
  - Layer `1` is not safe under the CMRC50 NLL gate even alone; it also has a clear positive interaction with layer `8`.
  - The current selected Reuse set `[8,25]` is supported by both conditional pruning and pairwise interaction evidence.

### 2026-06-09 Patch runtime window sweep for selected route

- Purpose: reduce TTFT for the current quality-safe Reuse `[8,25]` / Patch 30 route without fallback or recompute.
- Route: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418/lite_to_7b/conditional_report/selected_route_f1ret095_nll005.json`.
- Sweep root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_runtime_window_sweep_cmrc5_20260609_0615`.
- Eval: CMRC2018 5 samples, same route and checkpoints, `PATCH_CKPT_DEVICE=cpu`, persist shards/modules disabled.
- Results:
  - Window 2: SCD TTFT `25500.482` ms, total `25736.075` ms, max RSS `20222444` KB, wall time `2:49.25`.
  - Window 4: SCD TTFT `22627.609` ms, total `22870.345` ms, max RSS `20225812` KB, wall time `2:34.64`.
  - Window 8: SCD TTFT `21856.312` ms, total `22089.663` ms, max RSS `20224740` KB, wall time `2:33.81`.
  - Window 16: SCD TTFT `21910.812` ms, total `22133.010` ms, max RSS `20224208` KB, wall time `2:31.34`.
  - Window 32: failed with CUDA OOM while loading residual Patch modules; GPU 1 had only `11.12` MiB free.
- Code/default update:
  - `scripts/run_ser_scd_icml_eval_existing_route.sh` now defaults to `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, `RUNTIME_PATCH_PERSIST_SHARDS=false`, and `RUNTIME_PATCH_PERSIST_MODULES=false`.
  - `scripts/run_ser_scd_icml_conditional_route_screen.sh` now defaults to `RUNTIME_PATCH_WINDOW_SIZE=8`.
  - README and `REPRODUCING_SCD_ICML.md` document window 8 as the current safe Patch-heavy default and note that window 32 OOMed.
- Default smoke:
  - Run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_default_window8_smoke_cmrc1_20260609_0630`.
  - Generated config confirmed default `patch_ckpt_device='cpu'`, `runtime_patch_window_size=8`, persist flags `false`.
  - The 1-sample eval completed and wrote `scd_icml_mistral_summary.json`.
- Analysis:
  - Window 8 is the best current TTFT setting among successful windows and improves TTFT by about `14.3%` versus window 2 on the 5-sample smoke.
  - The selected route is still much slower than full consumer because it uses 30 Patch layers, but the open-source runtime default is now safer and faster than window 1/2 streaming.

### 2026-06-09 conditional Reuse expansion beyond `[8,25]`

- Purpose: continue teacher-aligned conditional route search from the quality-safe Reuse `[8,25]` / Patch 30 route, excluding layer `1` because CMRC50 showed it is unsafe alone and conflicts with layer `8`.
- Runtime constraints: no fallback, no recompute, `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, persistence disabled, two GPUs (`CUDA_VISIBLE_DEVICES=0,1`).
- Single-add CMRC20 screen:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_from_8_25_cmrc20_20260609_0915`.
  - Tested explicit Reuse sets `[8,25,x]` for `x in {13,9,24,21,11,22}`.
  - All six passed the proxy gate on CMRC2018 20 samples.
  - Best NLL candidate: Reuse `[8,24,25]`, Patch 29, F1 retention `1.230718820123373`, avg NLL delta `-0.08436752425299748`, TTFT `20127.66595967114` ms.
  - Fastest among the best single-add routes: Reuse `[8,9,25]`, Patch 29, F1 retention `1.220518388285736`, avg NLL delta `-0.05470009026704009`, TTFT `20142.859945748933` ms.
- Combination CMRC20 interaction screen:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_combo_cmrc20_20260609_0730`.
  - Tested `[8,21,24,25]`, `[8,9,24,25]`, `[8,11,24,25]`, `[8,13,24,25]`, `[8,9,21,24,25]`, `[8,9,11,13,21,24,25]`, and `[8,9,11,13,21,22,24,25]`.
  - All seven passed the CMRC20 proxy gate.
  - Best efficiency candidate: Reuse `[8,9,11,13,21,22,24,25]`, Patch 24, F1 retention `1.1557219335744036`, avg NLL delta `-0.09403756327099272`, TTFT `17449.470412847586` ms.
  - Key analysis: the previous Reuse 8 / Patch 24 CMRC50 failure used layer `1` (`[1,8,9,11,13,21,24,25]`). Excluding layer `1` allowed 7-8 Reuse layers to remain stable on CMRC20.
- CMRC50 Pareto validation:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830`.
  - Metrics: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830/lite_to_7b/conditional_report/route_metrics.json`.
  - Reuse `[8,9,11,13,21,22,24,25]`, Patch 24:
    - SCD F1 `0.7565720072590736`; full F1 `0.618514853953628`; F1 retention `1.2232074984504675`.
    - SCD avg NLL `0.6004365290340221`; full avg NLL `0.7055898962204568`; avg NLL delta `-0.10515336718643464`.
    - SCD TTFT `17121.27815722488` ms; payload `127.93375` MB.
  - Reuse `[8,9,11,13,21,24,25]`, Patch 25:
    - SCD F1 `0.771428719174609`; F1 retention `1.2472274743986105`.
    - SCD avg NLL `0.6108021005198814`; avg NLL delta `-0.09478779570057538`.
    - SCD TTFT `18031.21856189333` ms; payload `130.21828125` MB.
  - Reuse `[8,9,21,24,25]`, Patch 27:
    - SCD F1 `0.7897491142444539`; F1 retention `1.2768474503017575`.
    - SCD avg NLL `0.6338935294465116`; avg NLL delta `-0.0716963667739452`.
    - SCD TTFT `19260.57493634522` ms; payload `134.78734375` MB.
- Quality-gate selection:
  - Gate: F1 retention `>=0.95`, avg NLL delta `<=0.05`, SCD TTFT `<=18000` ms.
  - Selected route: Reuse `[8,9,11,13,21,22,24,25]`, Patch 24.
  - Selected route path: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_ttft18s.json`.
  - Quality-gate report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830/lite_to_7b/conditional_report/quality_gate_f1ret095_nll005_ttft18s.json`.
- Analysis:
  - Current selected route improves from the previous quality-safe Reuse `[8,25]` / Patch 30 route to Reuse 8 / Patch 24 without violating the CMRC50 F1+NLL gate.
  - TTFT improves from the previous CMRC50 `[8,25]` route (`25633.967208201066` ms) to `17121.27815722488` ms, a reduction of about `33.2%`.
  - The result supports the pasted teacher-aligned conditional-search direction: single-layer candidates were used only for screening, and the promoted route required conditional combination validation plus CMRC50 verification.
  - No fabricated metrics were introduced; all reported numbers come from strict eval summaries under the listed run roots.

### 2026-06-09 max-Reuse route search with work-queue scheduler

- Purpose: follow the updated objective to maximize the number of Reuse layers while keeping quality loss small. TTFT is recorded but not used for selection in this pass.
- Scheduler change:
  - Script: `scripts/run_ser_scd_icml_conditional_route_screen.sh`.
  - Implementation: candidate route files are written to `logs/route_queue.txt`; one worker is started per `GPU_PAIRS` entry; each worker uses `flock` on `logs/route_queue.lock` and dequeues the next route as soon as its current route finishes.
  - Verification: remote `bash -n scripts/run_ser_scd_icml_conditional_route_screen.sh` passed and `/usr/bin/flock` exists on `ser`.
- CMRC20 expansion run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse16_18_cmrc20_workqueue_20260609_1425`.
  - Eval: CMRC2018 20 samples, no fallback/recompute, `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, two GPU pairs `0,1;2,3`.
  - Highest tested Reuse counts:
    - Reuse19 `[5,6,8,9,10,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`: F1 retention `1.040958`, avg NLL delta `0.171696`; failed NLL gate.
    - Reuse18 `[5,8,9,10,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`: F1 retention `1.059472`, avg NLL delta `0.103336`; failed NLL gate.
    - Reuse17 `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`: F1 retention `1.157562`, avg NLL delta `0.036072`; passed CMRC20 gate.
    - Reuse17 `[5,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`: F1 retention `1.120527`, avg NLL delta `0.048988`; passed CMRC20 gate but was close to the NLL limit.
    - Best Reuse16 backup `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`: F1 retention `1.148598`, avg NLL delta `0.005587`.
- CMRC50 validation run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510`.
  - Eval: CMRC2018 50 samples, same strict runtime and GPU-pair work queue.
  - Work-queue evidence: worker 0 started Reuse17 on GPUs `0,1` at `2026-06-09T15:05:28+08:00`, finished at `15:20:00`, and immediately started the queued Reuse16 route on GPUs `0,1`. Worker 1 ran the other Reuse17 route on GPUs `2,3`, finished at `15:20:09`, and idled because the queue was empty.
  - Reuse17 `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`: SCD F1 `0.690546`, F1 retention `1.116458`, avg NLL `0.763543`, avg NLL delta `0.057953`; failed NLL gate.
  - Reuse17 `[5,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`: SCD F1 `0.692733`, F1 retention `1.119994`, avg NLL `0.784201`, avg NLL delta `0.078611`; failed NLL gate.
  - Reuse16 `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`: SCD F1 `0.6956317099725124`, full F1 `0.618514853953628`, F1 retention `1.124680685558226`, SCD avg NLL `0.7289511036145745`, avg NLL delta `0.023361207394117733`; passed.
- Selection:
  - Command gate: `--retention-threshold 0.95 --max-avg-nll-delta 0.05` with no `--max-scd-ttft-ms`.
  - Selected route: Reuse16 / Patch16.
  - Reuse layers: `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`.
  - Patch layers: `[0,1,2,3,4,6,7,10,12,15,16,17,18,19,26,31]`.
  - Selection report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_no_ttft.json`.
  - Copied selected route: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_no_ttft.route.json`.
- Analysis:
  - Under the current CMRC50 F1+NLL quality gate, Reuse16 is the highest confirmed Reuse count.
  - Reuse17 is plausible on CMRC20 but does not survive CMRC50 because avg NLL delta exceeds `0.05`.
  - The work-queue scheduler eliminates the previous batch-level idle wait; newly launched screens now keep a free GPU pair busy whenever more route candidates remain in the queue.

### 2026-06-09 Reuse17 CMRC200 validation

- Purpose: rerun the stronger Reuse17 candidate on CMRC2018 200 samples and report full-consumer and SCD F1.
- Resource note: GPU 1 was occupied by an external `/disk_n/gaojie/miniconda3/envs/lab/bin/python3.10` multiprocessing process, so this run used the free GPU pair `2,3` instead of four GPUs.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse17_best_cmrc200_g23_20260609_1550`.
- Route: Reuse layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`; Patch 15.
- Eval: CMRC2018 200 samples, no fallback/recompute, `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, persistence disabled.
- Runtime: screen `scd_reuse17_cmrc200_g23_1550`, started `2026-06-09T15:50:22+08:00`, finished `2026-06-09T16:48:20+08:00`.
- Results:
  - Full-consumer F1: `0.657307250029563`; EM `0.365`; avg NLL `0.5692842822281626`.
  - SCD F1: `0.6325964374391636`; EM `0.37`; avg NLL `0.6572635681813226`.
  - F1 retention: `0.9624059941689552`.
  - Avg NLL delta: `0.08797928595315996`.
- Artifacts:
  - Metrics JSON: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse17_best_cmrc200_g23_20260609_1550/lite_to_7b/conditional_report/route_metrics.json`.
  - Raw summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse17_best_cmrc200_g23_20260609_1550/lite_to_7b/conditional_evals/conditional_reuse_reuse_l05_l08_l09_l11_l13_l14_l20_l21_l22_l23_l24_l25_l27_l28_l29_l30_l31_patch15/scd_icml_mistral_summary.json`.
- Analysis:
  - On CMRC200, this Reuse17 candidate keeps F1 above the `0.95` retention gate (`0.9624`), but avg NLL delta worsens to `0.08798`, so it still fails the current F1+NLL quality gate.

### 2026-06-09 CMRC train/dev leakage fix

- Purpose: remove the risk that Patch training/calibration prompts and strict eval are built from the same CMRC split.
- Root cause:
  - `scripts/run_ser_scd_icml_mistral_repro.sh` built train prompts via `load_text_benchmark_samples("cmrc2018", DATA_ROOT, limit=TRAIN_LIMIT)`.
  - The old CMRC loader defaulted to dev-path candidates before train-path candidates, so a data root containing `cmrc2018_dev.json` could supply dev samples to train-prompt generation.
  - This means route-search results produced before the fix may have train/eval overlap risk.
- Code changes:
  - `src/scd_repro/cmrc.py` now requires a `split` argument with valid values `train` or `dev`; default is `dev` for eval compatibility.
  - `src/scd_repro/text_benchmarks.py` forwards `cmrc_split`.
  - `src/scd_repro/cli/eval_text_strict.py` and `src/scd_repro/cli/eval_cmrc_strict.py` accept `--cmrc-split` and record `cmrc_split` in result metadata.
  - `scripts/run_ser_scd_icml_mistral_repro.sh` defaults to `TRAIN_CMRC_SPLIT=train` and `EVAL_CMRC_SPLIT=dev`; generated train prompts now include a sibling `*.meta.json` with qids and split metadata.
  - `scripts/run_ser_scd_icml_eval_existing_route.sh` and `scripts/run_ser_scd_icml_conditional_route_screen.sh` default eval to `CMRC_SPLIT=dev`.
  - `scripts/check_cmrc_loader_remote.py` gained `--split {dev,train}` for remote split smoke tests.
  - `tests/test_cmrc_split.py` verifies default dev loading, explicit train loading, and no dev-to-train fallback.
- Local verification:
  - `python -m py_compile Experiment\core_code\runtime_repo\src\scd_repro\cmrc.py Experiment\core_code\runtime_repo\src\scd_repro\text_benchmarks.py Experiment\core_code\runtime_repo\src\scd_repro\cli\eval_text_strict.py Experiment\core_code\runtime_repo\src\scd_repro\cli\eval_cmrc_strict.py`
  - `$env:PYTHONPATH='C:\Users\mql16\Desktop\SCD-ICML\Experiment\core_code\runtime_repo\src'; python -m unittest discover -s Experiment\core_code\runtime_repo\tests -v`
  - Result: 15 tests passed.
- Remote verification on `ser`:
  - `bash -n scripts/run_ser_scd_icml_eval_existing_route.sh; bash -n scripts/run_ser_scd_icml_conditional_route_screen.sh; bash -n scripts/run_ser_scd_icml_mistral_repro.sh`
  - `python -m py_compile src/scd_repro/cmrc.py src/scd_repro/text_benchmarks.py src/scd_repro/cli/eval_text_strict.py src/scd_repro/cli/eval_cmrc_strict.py tests/test_cmrc_split.py`
  - `python -m unittest discover -s tests -v`
  - `python scripts/check_cmrc_loader_remote.py --data-root /disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854 --split train --limit 2`
  - `python scripts/check_cmrc_loader_remote.py --data-root /disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854 --split dev --limit 2`
  - Result: shell syntax check passed, 15 tests passed, train smoke reported first qid `TRAIN_186_QUERY_0`, and dev smoke reported first qid `DEV_0_QUERY_0`.
- Result status:
  - Pre-fix Reuse16/Reuse17 route-search runs, including CMRC50 and CMRC200 validations, are diagnostic only.
  - They should not be used as claim-safe final paper evidence until the Patch prompt/checkpoint path is regenerated from the train split and eval is rerun on the dev split.
- Next action:
  - Launch a clean retrain/calibration plus eval run for the current best route family once the desired GPU allocation is available.

### 2026-06-09 Clean Reuse16/Patch16 CMRC50 after train/dev split fix

- Purpose: rerun the selected Reuse16/Patch16 route with Patch prompt/capture from CMRC train split and strict eval on CMRC dev split after fixing the leakage risk.
- Code update before run:
  - `src/scd_repro/io_utils.py`: added optional `torch.load(..., mmap=True)` bundle loading with fallback.
  - `src/scd_repro/direct_kv_residual_patch.py`: changed shared residual Patch training to a chunk-first streaming pass that combines linear error stats and residual training over the same loaded per-chunk arrays.
- Local verification:
  - `python -m py_compile Experiment/core_code/runtime_repo/src/scd_repro/io_utils.py Experiment/core_code/runtime_repo/src/scd_repro/direct_kv_residual_patch.py`.
  - Synthetic CPU smoke for `_train_xnorm_nativekv_shared_residual_patch_streaming` passed.
- Remote verification:
  - Synced changed files to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/src/scd_repro/`.
  - Remote py-compile and synthetic CPU smoke passed under `/disk_n/c101-3-mql/venvs/A800`.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723`.
- Route:
  - Reuse layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`.
  - Patch layers `[0,1,2,3,4,6,7,10,12,15,16,17,18,19,26,31]`.
- Data split:
  - Patch capture/train prompt split: CMRC2018 train, `n=200`.
  - Strict eval split: CMRC2018 dev, `n=50`, first eval qid `DEV_0_QUERY_0`.
- Resource note:
  - GPU 1 was occupied by an external process, so Patch train used GPU 0 and strict eval used physical GPUs 0 and 2 via `CUDA_VISIBLE_DEVICES=0,2`.
- Stage timings:
  - `lite_to_7b_capture_pair`: success, `801` sec.
  - `lite_to_7b_train_reuse`: success, `626` sec.
  - `lite_to_7b_profile_layers`: success, `1720` sec.
  - `lite_to_7b_train_patch_patch16`: success, `2390` sec.
  - `lite_to_7b_route_patch16`: success, `0` sec.
  - `lite_to_7b_eval_cmrc2018_patch16`: success, `488` sec.
- Patch training observations:
  - Direct stats pass completed `200/200` chunks in `696.8` sec.
  - New streaming pass completed `200/200` chunks in `1564.4` sec.
  - Shared residual chose identity fallback because streaming epoch mean total loss `0.377293` was slightly worse than identity `0.376494`; direct Patch weights were still saved and used.
- CMRC50 dev results:
  - Full Consumer: F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`, TTFT `126.8715` ms.
  - Reuse-only: F1 `0.0065764186193433384`, EM `0.0`, avg NLL `7.539012339486356`, TTFT `55.3845` ms.
  - SCD: F1 `0.6687529784411572`, EM `0.44`, avg NLL `0.7342439769359117`, TTFT `43.5927` ms.
  - SCD vs Full: F1 retention `1.081221149964047`, avg NLL delta `0.028654080715454944`, TTFT delta `-83.2788` ms.
- Artifacts:
  - Eval summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/cmrc2018_reuse16_patch16_clean_dev50/summary.json`.
  - Compact summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/scd_icml_mistral_summary.json`.
  - Patch metrics: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_patch16/metrics.json`.
- Conclusion:
  - This is the first clean post-split Reuse16/Patch16 CMRC50 result in this run root.
  - Under the current quality gate, Reuse16/Patch16 passes CMRC50 dev: F1 is above full consumer and avg NLL delta remains below `0.05`.

### 2026-06-09 Clean Reuse17/Patch15 and Reuse18 boundary after train/dev split fix

- Purpose: continue maximizing Reuse layer count after the clean CMRC train/dev split fix, using Reuse layer count as the current speed proxy and retaining the quality gate F1 retention `>=0.95` plus avg NLL delta `<=0.05`.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723`.
- Baseline for deltas:
  - Full Consumer CMRC50 dev: F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`.
  - Clean Reuse16/Patch16 SCD: F1 `0.6687529784411572`, EM `0.44`, avg NLL `0.7342439769359117`, avg NLL delta `0.028654080715454944`.
- Reuse17 candidate adding layer `31`:
  - Route: Reuse `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30,31]`; Patch `[0,1,2,3,4,6,7,10,12,15,16,17,18,19,26]`.
  - SCD CMRC50 dev: F1 `0.7077913554592317`, EM `0.44`, avg NLL `0.7711014678734646`.
  - SCD vs Full: F1 retention `1.1443401091096463`, avg NLL delta `0.0655115716530078`.
  - Status: rejected because avg NLL delta exceeds `0.05`.
  - Artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/cmrc2018_reuse17_patch15_clean_dev50/summary.json`.
- Reuse17 single-add scan from the clean Reuse16 baseline:
  - Scan artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/reuse17_single_add_scdonly_clean_dev50/summary_reuse17_single_add_scdonly.json`.
  - Passing candidate: add layer `3`.
  - Passing SCD metrics for add layer `3`: F1 `0.7178651651494944`, EM `0.44`, avg NLL `0.7506309336299498`, F1 retention `1.1606272033093565`, avg NLL delta `0.04504103740949306`.
- Full eval for selected Reuse17(add3)/Patch15:
  - Route: Reuse `[3,5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`; Patch `[0,1,2,4,6,7,10,12,15,16,17,18,19,26,31]`.
  - Full Consumer: F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`, TTFT `128.34175441414118` ms.
  - SCD: F1 `0.7178651651494944`, EM `0.44`, avg NLL `0.7506309336299498`, TTFT `41.16989119909704` ms, payload `107.37296875` MB, compression ratio `0.6808510638297872`.
  - SCD vs Full: F1 retention `1.1606272033093565`, avg NLL delta `0.04504103740949306`, TTFT delta `-87.17186321504414` ms.
  - Status: passes the current clean CMRC50 quality gate.
  - Route artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/route/route_cmrc2018_seed42_reuse17_add3_patch15_clean.json`.
  - Eval artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/cmrc2018_reuse17_add3_patch15_clean_dev50/summary.json`.
- Reuse18 scan from Reuse17(add3):
  - Scan artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/reuse18_from_add3_scdonly_clean_dev50/summary_reuse18_from_add3_scdonly.json`.
  - Result: all 15 second-add candidates failed the avg NLL gate.
  - Best F1 candidate `[3,6]`: F1 `0.7318751531840901`, EM `0.42`, avg NLL `0.9395432755422822`, F1 retention `1.1832782163691755`, avg NLL delta `0.23395337932182547`.
  - Lowest-near-boundary candidate among the top F1 group still exceeded the NLL gate; e.g. `[3,31]` had avg NLL delta `0.08388605450741937`.
- Resource/status check after completion:
  - `screen -ls` on `ser` reported no active sockets.
  - GPU memory status was GPU 0 `18/24564` MiB, GPU 1 `20653/24564` MiB from an external process, GPU 2 `18/24564` MiB, GPU 3 `18/24564` MiB.
- Conclusion:
  - Under the current clean CMRC50 gate, the maximum confirmed Reuse count is `17`.
  - The selected clean route is Reuse17(add3)/Patch15. Continuing to Reuse18 is not justified unless the quality gate is relaxed, the Patch/reuse implementation improves, or the candidate is revalidated under a different objective.

### 2026-06-09 Clean CMRC200 validation for Reuse17(add3) and Reuse16

- Purpose: validate whether the clean CMRC50 selected routes remain quality-safe on a larger CMRC2018 dev subset.
- Quality gate used for this check: F1 retention `>=0.95` and avg NLL delta `<=0.05` against Full Consumer on the same 200 dev samples.
- Shared setup:
  - Patch capture/train prompt split: CMRC2018 train, `n=200`.
  - Eval split: CMRC2018 dev, `n=200`.
  - Data root: `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854`.
  - Reuse checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_reuse/cmrc2018_seed42`.
  - Patch checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_patch16`.
  - Runtime: `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, no persistent Patch shards/modules.
- Reuse17(add3)/Patch15 CMRC200:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse17_add3_cmrc200_dev_20260609_214840`.
  - Route: Reuse `[3,5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`; Patch `[0,1,2,4,6,7,10,12,15,16,17,18,19,26,31]`.
  - Full Consumer: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, TTFT `129.8858017101884` ms.
  - SCD: F1 `0.6715361993886206`, EM `0.38`, avg NLL `0.6668955393563621`, TTFT `41.92783338949084` ms, payload `108.0173828125` MB.
  - SCD vs Full: F1 retention `1.0216473336608072`, avg NLL delta `0.09761125712819951`, TTFT delta `-87.95796832069755` ms.
  - Status: rejected under the current F1+NLL gate because avg NLL delta exceeds `0.05`.
  - Artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse17_add3_cmrc200_dev_20260609_214840/lite_to_7b/strict_eval/cmrc2018/summary.json`.
- Reuse16/Patch16 CMRC200:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc200_dev_20260609_220744`.
  - Route: Reuse `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`; Patch `[0,1,2,3,4,6,7,10,12,15,16,17,18,19,26,31]`.
  - Full Consumer: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, TTFT `129.84369908925146` ms.
  - SCD: F1 `0.6447946591903111`, EM `0.395`, avg NLL `0.6225295365171866`, TTFT `44.8479413962923` ms, payload `110.315625` MB.
  - SCD vs Full: F1 retention `0.980963863035606`, avg NLL delta `0.053245254289023936`, TTFT delta `-84.99575769295916` ms.
  - Status: narrowly rejected under the current F1+NLL gate because avg NLL delta is `0.0032452542890239336` above the `0.05` threshold.
  - Artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc200_dev_20260609_220744/lite_to_7b/strict_eval/cmrc2018/summary.json`.
- Local code verification during this run:
  - `python -m py_compile Experiment\core_code\runtime_repo\src\scd_repro\io_utils.py Experiment\core_code\runtime_repo\src\scd_repro\direct_kv_residual_patch.py Experiment\core_code\runtime_repo\src\scd_repro\cmrc.py Experiment\core_code\runtime_repo\src\scd_repro\text_benchmarks.py Experiment\core_code\runtime_repo\src\scd_repro\cli\eval_text_strict.py`.
  - `$env:PYTHONPATH='C:\Users\mql16\Desktop\SCD-ICML\Experiment\core_code\runtime_repo\src'; python -m unittest discover -s Experiment\core_code\runtime_repo\tests -v`.
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo`.
  - Result: py-compile passed, 15 unit tests passed, and release readiness passed.
- Follow-up action launched:
  - Reason: current clean Patch16 checkpoint cannot evaluate Reuse15 candidates made by converting one current Reuse layer to Patch, because no Patch weights exist for those Reuse16 layers.
  - Launched 3-shard Patch training for additional layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]` using the same `xnorm_nativekv_shared_residual_mlp_v1` config.
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706`.
  - Shards: GPU 0 `extra_05_14=[5,8,9,11,13,14]`; GPU 2 `extra_20_24=[20,21,22,23,24]`; GPU 3 `extra_25_30=[25,27,28,29,30]`; GPU 1 avoided because it was occupied by an external process.
  - Status at first logging time: all three shards were in the residual streaming phase after completing shared stats chunk `200/200`.
  - Final shard status: all three shards completed and saved checkpoints:
    - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch_shards/extra_05_14`.
    - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch_shards/extra_20_24`.
    - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch_shards/extra_25_30`.
  - Merge note: direct `merge_patch_ckpts.py` rejected these shared-residual checkpoints because each shard contains non-layer-local `xnorm_nativekv_shared_residual_patch.*` weights. This is correct; independently trained shared residual MLPs should not be silently concatenated.
  - A direct-only merged pool was created for runtime probing at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_all32_direct_only_clean`, with `fit_mode=xnorm_nativekv_residual_mlp_v1` and `residual_layers=[]`. A 1-sample smoke with Reuse16 route completed, but the single-sample NLL was very poor (`11.631944444444445`), so this direct-only pool should not be used as the main quality-selection evidence.
  - A single all-32 shared-residual Patch training run was launched instead, using a temporary config with `patch_budget=32`:
    - Screen: `scd_clean_patch_all32_shared_232004`.
    - Output: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_all32_shared_clean`.
    - Config: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/tmp_cfg/train_patch_all32_shared.json`.
    - Startup verification: selected layers and residual layers were both `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]`.
- Conclusion:
  - CMRC50 overestimated route safety for Reuse17(add3) and slightly overestimated Reuse16.
  - Under the strict `0.05` NLL gate on CMRC200, no current clean route with 16 or 17 Reuse layers is confirmed passing.
  - The next evidence-producing step is to merge the extra Patch shards with the existing clean Patch16 checkpoint, then scan Reuse15 or adjusted Reuse16 routes using the expanded Patch pool.

### 2026-06-10 F1/EM-primary Reuse boundary scan on CMRC200

- Purpose: follow the corrected objective where the primary route-selection metric is answer quality (F1/EM) while maximizing Reuse layers; avg NLL is reported as an auxiliary risk signal rather than used as a hard veto.
- Shared setup:
  - Dataset: CMRC2018 dev, `n=200`.
  - Direction: MistralLite producer to Mistral-7B consumer (`lite_to_7b`).
  - Existing clean Reuse checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_reuse/cmrc2018_seed42`.
  - Existing clean Patch checkpoint for the forward Reuse-increase scans: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_patch16`.
  - Full Consumer reference from the final selected-route run: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, TTFT `100.85939839249477` ms.
- Metric decision:
  - Primary: maximize Reuse count while keeping F1/EM close to Full Consumer.
  - Working quality boundary used for this scan: F1 retention around `0.95` or higher, with EM not showing a large collapse.
  - Auxiliary: avg NLL is recorded as calibration/risk evidence only.
- Reuse18 scan:
  - Partial scan run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_primary_f1em_reuse18_cmrc200_20260609_232740`.
  - Tested high-priority candidates after reordering by prior CMRC50 F1/EM.
  - `Reuse18(add3,add6)/Patch14`: F1 `0.666828`, EM `0.370`, avg NLL `0.816394`, payload `105.719` MB, TTFT `42.404` ms. This exceeds Full Consumer F1/EM and was selected as the Reuse18 base for the next scan.
  - `Reuse18(add3,add17)/Patch14`: F1 `0.661765`, EM `0.405`, avg NLL `0.740469`, payload `105.719` MB, TTFT `42.807` ms. This is a stable alternate base with better EM and lower NLL than add6.
  - `add0` and `add1` were poor and rejected by F1/EM.
- Reuse19 scan:
  - Partial scan run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_primary_f1em_reuse19_from_add6_cmrc200_20260609_235150`.
  - Base route: Reuse18(add3,add6)/Patch14.
  - Best selected candidate: `Reuse19(add3,add6,add17)/Patch13`.
  - SCD-only scan metrics for the selected candidate: F1 `0.642751`, EM `0.350`, avg NLL `0.934572`, payload `103.421` MB, TTFT `48.125` ms.
  - Retention against Full Consumer: F1 retention `0.9779`, EM retention `0.9589`, EM delta `-0.015`.
  - Alternate tested candidate `Reuse19(add3,add6,add26)/Patch13`: F1 `0.624669`, EM `0.355`, F1 retention `0.9503`, EM retention `0.9726`.
- Reuse20 boundary:
  - Partial scan run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_primary_f1em_reuse20_from_add6_add17_cmrc200_20260610_001042`.
  - Base route: Reuse19(add3,add6,add17)/Patch13.
  - Tested six high-priority candidates: add `26`, `10`, `31`, `7`, `4`, and `16`.
  - Best Reuse20 candidate among the tested set was add `26`: F1 `0.611265`, EM `0.360`, F1 retention `0.9300`, EM retention `0.9863`, avg NLL `1.030351`.
  - The next five candidates all had lower F1 retention (`0.9088`, `0.8918`, `0.8837`, `0.8804`, `0.6844`), so Reuse20 was stopped as not quality-safe under the F1/EM-primary boundary.
- Full+SCD confirmation for the selected Reuse19 route:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_full_scd_cmrc200_20260610_004450`.
  - Route: Reuse `[3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`; Patch `[0,1,2,4,7,10,12,15,16,18,19,26,31]`.
  - Methods run: `full_consumer` and `scd_full` only; no fallback, no recompute, no raw/4bit side methods.
  - Full Consumer: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, TTFT `100.85939839249477` ms.
  - SCD: F1 `0.6427511605994372`, EM `0.35`, avg NLL `0.9345715692188692`, TTFT `40.80477777868509` ms, payload `103.4208984375` MB.
  - SCD vs Full: F1 retention `0.9778549689974191`, EM retention `0.958904109589041`, EM delta `-0.015000000000000013`, avg NLL delta `0.36528728699070656`, TTFT delta `-60.054620613809675` ms.
- Full all-32 shared Patch checkpoint status:
  - Output: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_all32_shared_clean`.
  - Verified `meta.json` has selected layers and residual layers `[0..31]`.
  - Verified `weights.pt` contains 32 `direct_wk`, 32 `direct_wv`, and the shared residual MLP arrays.
- All-32 Patch probe for the best Reuse20 candidate:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_probe_reuse20_add26_all32patch_cmrc200_20260610_005549`.
  - Route: Reuse20(add3,add6,add17,add26)/Patch12.
  - Patch checkpoint: full all-32 shared Patch checkpoint above.
  - SCD result: F1 `0.61126515351457`, EM `0.36`, avg NLL `1.0303513656672685`, TTFT `41.04385194834322` ms, payload `101.12265625` MB.
  - Known Full Consumer comparison: F1 retention `0.9299534631438764`, EM retention `0.9863013698630136`, EM delta `-0.0050000000000000044`.
  - Decision: all-32 Patch does not rescue Reuse20; the Reuse20 boundary remains below the working F1 retention target.
- Conclusion:
  - Under the F1/EM-primary objective, the current highest verified Reuse count is `19`.
  - Reuse20 was initially unsupported by the high-priority CMRC200 evidence because F1 retention dropped to at best `0.9300`; the later remaining-candidate scan below completes the Reuse20 one-layer extension check.
  - The selected route is Reuse19(add3,add6,add17)/Patch13. NLL is substantially worse than Full Consumer and should be described as an auxiliary calibration risk, not as the primary route-selection blocker.

### 2026-06-10 Selected Reuse19 route on HotpotQA and WikiText-2

- Purpose: test whether the CMRC-selected Reuse19/Patch13 route transfers to the other paper-facing diagnostic datasets under the same no-fallback/no-recompute runtime.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_full_scd_hotpot_wiki_20260610_011000`.
- Route: `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Patch checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_patch16`.
- Reuse checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/ckpt_reuse/cmrc2018_seed42`.
- Eval config: `configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`.
- Methods run: `full_consumer` and `scd_full` only; no fallback, no recompute, no raw/4-bit/reuse-only side methods.
- HotpotQA dev `n=200`:
  - Full Consumer: F1 `0.034348380586727865`, EM `0.01`, avg NLL `4.657311402155862`, PPL `105.35245067026526`, TTFT `140.94537812285125` ms.
  - SCD Reuse19: F1 `0.04486385345024311`, EM `0.015`, avg NLL `4.090771353932278`, PPL `59.78599008132167`, TTFT `50.51531478064135` ms, payload `178.7044921875` MB.
  - Delta: F1 retention `1.3061417360554808`, EM retention `1.5`, avg NLL delta `-0.5665400482235841`, TTFT delta `-90.4300633422099` ms.
  - Interpretation: SCD does not harm this HotpotQA slice, but absolute F1/EM are very low for both methods, so this should remain a sanity check rather than a strong task-quality claim.
- WikiText-2 `n=200`:
  - Full Consumer: avg NLL `2.0260445468127726`, PPL `7.584028681680868`, TTFT `62.887440412305295` ms.
  - SCD Reuse19: avg NLL `3.274238524995744`, PPL `26.423097289857097`, TTFT `31.559598443564028` ms, payload `22.8181640625` MB.
  - Delta: avg NLL delta `1.2481939781829716`, TTFT delta `-31.327841968741268` ms.
  - Interpretation: the CMRC-selected Reuse19/Patch13 route is not quality-safe for WikiText-2. A separate WikiText-2 boundary search or a more conservative route is required before making broad language-modeling claims.
- Conclusion:
  - Reuse19/Patch13 remains the current CMRC F1/EM-primary boundary.
  - It should not be advertised as a universal route for all datasets.
  - Next evidence-producing step: run WikiText-2 SCD-only probes for more conservative routes, starting from Reuse16/Reuse17/Reuse18 and comparing against the known Full Consumer NLL.

### 2026-06-10 Selection metric correction in runtime tooling

- Purpose: align route-selection code with the clarified objective: maximize Reuse layers while preserving QA F1/EM; use avg NLL only as an auxiliary risk signal.
- Changed local script:
  - `Experiment/core_code/runtime_repo/scripts/select_quality_gated_scd_route.py`.
- Method update:
  - The selected route is now sorted by maximum Reuse count first, then by the configured primary metric retention, with avg NLL only as a later tie-break/risk field.
  - `--max-avg-nll-delta` is advisory by default.
  - `--enforce-avg-nll-delta` was added for diagnostic runs that intentionally want strict NLL gating.
  - Reports now include F1 and EM values plus `f1_retention` and `em_retention` even when the primary metric argument is only one of them.
- Documentation update:
  - Updated `README.md`, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` so the release-facing guidance says F1/EM are primary for QA route selection and NLL is an auxiliary risk note.
- Current interpretation:
  - The CMRC200 Reuse19/Patch13 route remains the current maximum verified QA route under the F1/EM-primary objective.
  - WikiText-2 NLL scans should not replace the QA route-selection criterion; they are auxiliary language-modeling diagnostics.

### 2026-06-10 CMRC200 Reuse20 remaining-candidate F1/EM scan

- Purpose: make the Reuse20 boundary more complete under the F1/EM-primary objective. Earlier Reuse20 evidence tested only six high-priority additions; this run covers the remaining untested current Patch layers.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_remaining_cmrc200_f1em_20260610_032352`.
- Base route: `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Candidate layers: `[0,1,2,12,15,18,19]`, each converted from Patch to Reuse on top of the Reuse19/Patch13 base.
- Dataset and split: CMRC2018 dev, `n=200`.
- Eval mode: SCD-only no fallback/no recompute, using the fixed Full Consumer baseline from the selected-route run for later F1/EM retention calculation.
- GPU schedule: work queue over GPU pairs `0,1` and `2,3`.
- Screen: `scd_cmrc_reuse20_remaining`.
- Output summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_remaining_cmrc200_f1em_20260610_032352/lite_to_7b/conditional_report/qa_f1em_baseline_summary.json`.
- Results against the fixed Full Consumer baseline F1 `0.657307250029563`, EM `0.365`:
  - Add layer `2`: F1 `0.546818`, F1 retention `0.8319`, EM `0.290`, EM retention `0.7945`, avg NLL delta `1.062505`.
  - Add layer `19`: F1 `0.531610`, F1 retention `0.8088`, EM `0.290`, EM retention `0.7945`, avg NLL delta `0.543506`.
  - Add layer `12`: F1 `0.490987`, F1 retention `0.7470`, EM `0.260`, EM retention `0.7123`, avg NLL delta `1.447099`.
  - Add layer `1`: F1 `0.458611`, F1 retention `0.6977`, EM `0.210`, EM retention `0.5753`, avg NLL delta `1.558706`.
  - Add layer `0`: F1 `0.405715`, F1 retention `0.6172`, EM `0.130`, EM retention `0.3562`, avg NLL delta `1.385350`.
  - Add layer `15`: F1 `0.397806`, F1 retention `0.6052`, EM `0.190`, EM retention `0.5205`, avg NLL delta `1.483259`.
  - Add layer `18`: F1 `0.370308`, F1 retention `0.5634`, EM `0.145`, EM retention `0.3973`, avg NLL delta `1.538846`.
- Combined Reuse20 conclusion:
  - Previously tested additions: `26`, `10`, `31`, `7`, `4`, and `16`; best was add `26` with F1 retention `0.9300`.
  - This run tested the remaining additions: `0`, `1`, `2`, `12`, `15`, `18`, and `19`; best was add `2` with F1 retention `0.8319`.
  - Therefore all 13 one-layer Reuse20 extensions from Reuse19/Patch13 have been tested on CMRC200, and none preserves F1/EM under the current objective. The maximum verified CMRC200 route remains Reuse19/Patch13.

### 2026-06-10 WikiText-2 all-patch single-layer Reuse scan completed

- Purpose: auxiliary language-modeling risk diagnostic only. This scan should not override QA F1/EM route selection.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_wikitext2_single_reuse_scan_all32patch_20260610_015328`.
- Setup: start from all-32 Patch and convert one layer at a time to Reuse; WikiText-2 `n=200`; SCD-only no fallback/no recompute.
- Full Consumer reference: avg NLL `2.0260445468127726`, PPL `7.584028681680868`.
- Completed candidates: `32/32`.
- Best NLL-risk single-layer Reuse candidates:
  - Layer `13`: avg NLL `2.079837`, delta `0.053793`, PPL `8.003168`.
  - Layer `28`: avg NLL `2.080062`, delta `0.054018`, PPL `8.004968`.
  - Layer `27`: avg NLL `2.081191`, delta `0.055147`, PPL `8.014011`.
  - Layer `16`: avg NLL `2.081376`, delta `0.055332`, PPL `8.015494`.
  - Layer `24`: avg NLL `2.081428`, delta `0.055383`, PPL `8.015905`.
- Highest-risk single-layer Reuse candidates:
  - Layer `0`: avg NLL `2.832320`, delta `0.806275`, PPL `16.984820`.
  - Layer `3`: avg NLL `2.158392`, delta `0.132347`, PPL `8.657202`.
  - Layer `2`: avg NLL `2.142174`, delta `0.116130`, PPL `8.517937`.
- Interpretation: single-layer WikiText NLL risk favors mid/late layers and strongly disfavors layer `0`, but this remains auxiliary because the current primary objective is QA F1/EM retention under maximum Reuse.

### 2026-06-10 Selected Reuse19 5-method aggregate run launched

- Purpose: produce the release-verifiable claim-facing aggregate evidence: 3 datasets times 5 methods under the selected Reuse19/Patch13 route, with no fallback and no recomputation.
- Aggregate run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
- Runtime root: `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Route: `configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Eval config: `configs/eval_scd_icml_no_fallback_no_recompute.json`.
- Methods expected per dataset: `full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, `scd_full`.
- Subruns:
  - CMRC2018 dev `n=200`: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/cmrc2018_run`, screen `scd_agg_cmrc`, GPUs `0,1`.
  - HotpotQA + WikiText-2 `n=200`: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/hotpot_wiki_run`, screen `scd_agg_hw`, GPUs `2,3`.
- Completion criterion: merge the subrun summaries into the aggregate run root and pass `scripts/check_release_readiness.py --root . --run-root <aggregate-run-root>`.

### 2026-06-10 Selected Reuse19 5-method aggregate run completed

- Aggregate run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
- Output files:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/scd_icml_mistral_summary.json`.
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/scd_icml_mistral_summary.md`.
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/lite_to_7b/route/route_cmrc2018_seed42.json`.
- Verification:
  - Remote `python scripts/check_release_readiness.py --root . --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426` passed.
  - Screen status after completion: no active screens.
  - GPU status after completion: all four GPUs idle with only `18 MiB` used each.
- CMRC2018 dev `n=200`:
  - Full Consumer: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, PPL `1.7670019240707435`, TTFT `100.603853398934` ms.
  - Raw KV: F1 `0.09182391909171182`, EM `0.0`, avg NLL `0.7153963975012197`, PPL `2.044997153375181`, TTFT `31.22283098520711` ms.
  - Quantized 4-bit: F1 `0.109653290936564`, EM `0.0`, avg NLL `0.7387463788478611`, PPL `2.093309651662363`, TTFT `54.029793911613524` ms.
  - Reuse-only: F1 `0.010625235235162449`, EM `0.0`, avg NLL `7.833419159763835`, PPL `2523.543036094375`, TTFT `36.459266904275864` ms.
  - SCD Reuse19/Patch13: F1 `0.6427511605994372`, EM `0.35`, avg NLL `0.9345715692188692`, PPL `2.546122386828472`, TTFT `41.59756151493639` ms.
- HotpotQA `n=200`:
  - Full Consumer: F1 `0.034348380586727865`, EM `0.01`, avg NLL `4.657311402155862`, PPL `105.35245067026526`, TTFT `140.07910876302049` ms.
  - Raw KV: F1 `0.01332896590525499`, EM `0.0`, avg NLL `2.92195471793705`, PPL `18.577565892420555`, TTFT `31.090989247895777` ms.
  - Quantized 4-bit: F1 `0.015100750763580448`, EM `0.0`, avg NLL `2.926937440533892`, PPL `18.670363750784933`, TTFT `54.35235824435949` ms.
  - Reuse-only: F1 `0.006231822725205078`, EM `0.0`, avg NLL `9.033460282916213`, PPL `8378.802495077993`, TTFT `39.87283800728619` ms.
  - SCD Reuse19/Patch13: F1 `0.04486385345024311`, EM `0.015`, avg NLL `4.090771353932278`, PPL `59.78599008132167`, TTFT `44.89055946469307` ms.
- WikiText-2 `n=200`:
  - Full Consumer: avg NLL `2.0260445468127726`, PPL `7.584028681680868`, TTFT `62.36824941821396` ms.
  - Raw KV: avg NLL `2.363679577652365`, PPL `10.629993465725152`, TTFT `24.11868618335575` ms.
  - Quantized 4-bit: avg NLL `2.3718748496472837`, PPL `10.717467099011241`, TTFT `47.31429586187005` ms.
  - Reuse-only: avg NLL `6.732487837076187`, PPL `839.2325448020879`, TTFT `25.13807148206979` ms.
  - SCD Reuse19/Patch13: avg NLL `3.274238524995744`, PPL `26.423097289857097`, TTFT `31.00419892463833` ms.
- Interpretation:
  - CMRC2018 supports Reuse19/Patch13 under the F1/EM-primary objective: F1 retention `0.9778549689974191`, EM retention `0.958904109589041`, with much lower TTFT than Full Consumer.
  - HotpotQA remains a sanity check because both Full Consumer and SCD have low absolute F1/EM under this prompt/model slice.
  - WikiText-2 remains an auxiliary language-modeling risk signal: the selected CMRC route is faster than Full Consumer but has substantially worse NLL/PPL.

### 2026-06-10 Reuse20-to-Reuse32 quality degradation curve launched

- Purpose: measure quality degradation as Reuse layer count increases beyond the selected Reuse19/Patch13 boundary.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604`.
- Base route: selected Reuse19/Patch13 route.
- Added-layer order, chosen by Reuse20 single-layer CMRC200 F1 ranking: `[26,10,31,4,7,2,19,12,1,16,0,15,18]`.
- Candidate route sequence:
  - Reuse20 through Reuse31: generated as nested mixed Patch/Reuse routes with prefix budgets `1-12`.
  - Reuse32: all layers are Reuse and therefore is not a mixed route; use the existing `reuse_only` rows from the verified aggregate run as the Reuse32 endpoint.
- Datasets: CMRC2018 dev, HotpotQA, WikiText-2, each `n=200`.
- Eval mode: SCD-only no fallback/no recompute for Reuse20-31.
- GPU schedule: work queue over GPU pairs `0,1` and `2,3`, screen `scd_curve_r20_r32`.

### 2026-06-10 Reuse20-to-Reuse32 quality degradation curve completed

- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604`.
- Output artifacts:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604/lite_to_7b/conditional_report/reuse20_to_32_quality_curve_with_reuse19_baseline.json`.
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604/lite_to_7b/conditional_report/reuse20_to_32_quality_curve_with_reuse19_baseline.csv`.
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604/lite_to_7b/conditional_report/reuse20_to_32_quality_curve_with_reuse19_baseline.md`.
- Completion status: Reuse20 through Reuse31 mixed routes completed on CMRC2018, HotpotQA, and WikiText-2. Reuse32 was filled from the verified aggregate run's `reuse_only` rows because all-32 Reuse has no Patch layers and is not a mixed route.
- Added-layer order after Reuse19: `[26,10,31,4,7,2,19,12,1,16,0,15,18]`.
- Remote status after completion: no active screen sessions; all four GPUs idle with only `18 MiB` used each.
- Quality curve summary:

| Reuse | Patch | Added layers | CMRC F1 | CMRC EM | CMRC F1 retention | Hotpot F1 | Hotpot EM | WikiText NLL | WikiText PPL |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 19 | 13 | - | 0.642751 | 0.350 | 0.977855 | 0.044864 | 0.015 | 3.274239 | 26.423097 |
| 20 | 12 | 26 | 0.611265 | 0.360 | 0.929953 | 0.029607 | 0.010 | 3.347518 | 28.432069 |
| 21 | 11 | 26,10 | 0.554634 | 0.310 | 0.843797 | 0.039627 | 0.015 | 3.449950 | 31.498827 |
| 22 | 10 | 26,10,31 | 0.496437 | 0.255 | 0.755259 | 0.044340 | 0.020 | 3.491113 | 32.822474 |
| 23 | 9 | 26,10,31,4 | 0.443754 | 0.230 | 0.675109 | 0.042036 | 0.020 | 3.611806 | 37.032858 |
| 24 | 8 | 26,10,31,4,7 | 0.440554 | 0.200 | 0.670241 | 0.047018 | 0.020 | 3.662087 | 38.942522 |
| 25 | 7 | 26,10,31,4,7,2 | 0.390004 | 0.190 | 0.593337 | 0.043239 | 0.015 | 4.148354 | 63.329680 |
| 26 | 6 | 26,10,31,4,7,2,19 | 0.288780 | 0.135 | 0.439338 | 0.047450 | 0.020 | 4.124323 | 61.825939 |
| 27 | 5 | 26,10,31,4,7,2,19,12 | 0.220660 | 0.075 | 0.335704 | 0.036366 | 0.010 | 4.239160 | 69.349571 |
| 28 | 4 | 26,10,31,4,7,2,19,12,1 | 0.212624 | 0.070 | 0.323478 | 0.027826 | 0.005 | 4.693250 | 109.207507 |
| 29 | 3 | 26,10,31,4,7,2,19,12,1,16 | 0.077045 | 0.010 | 0.117213 | 0.021953 | 0.000 | 4.751767 | 115.788712 |
| 30 | 2 | 26,10,31,4,7,2,19,12,1,16,0 | 0.009595 | 0.000 | 0.014597 | 0.006113 | 0.000 | 6.673624 | 791.257672 |
| 31 | 1 | 26,10,31,4,7,2,19,12,1,16,0,15 | 0.007903 | 0.000 | 0.012023 | 0.006280 | 0.000 | 6.687496 | 802.310561 |
| 32 | 0 | 26,10,31,4,7,2,19,12,1,16,0,15,18 | 0.010625 | 0.000 | 0.016165 | 0.006232 | 0.000 | 6.732488 | 839.232545 |

- Interpretation:
  - The F1/EM-primary boundary remains Reuse19/Patch13. Reuse20 already falls to CMRC F1 retention `0.929953`, below the `0.95` target.
  - Reuse21 and above show steep degradation on CMRC2018, reaching near-zero F1 by Reuse30.
  - HotpotQA remains noisy and low-absolute-F1 across all high-Reuse routes, so it should not drive route selection.
  - WikiText-2 NLL/PPL worsens monotonically enough to support the same risk conclusion: higher Reuse counts are not safe for language modeling quality.

### 2026-06-10 HotpotQA baseline prompt repair and rerun

- Purpose: fix the abnormal HotpotQA Full Consumer baseline, where the prior n=200 aggregate had Full Consumer F1 `0.034348380586727865` and EM `0.01`.
- Root cause found:
  - Full HotpotQA distractor context was concatenated before `Question/Answer`.
  - With `max_length=1024`, prompts could be right-truncated before the question and answer marker.
  - Existing sample evidence showed `prompt_tokens=1024` and Full Consumer was continuing evidence text rather than answering.
- Code changes:
  - `Experiment/core_code/runtime_repo/src/scd_repro/text_benchmarks.py` now builds a shorter HotpotQA prompt.
  - It prioritizes `supporting_facts` titles when available, falls back to question-token paragraph ranking, and keeps `Question ... Answer:` at the prompt tail.
  - `Experiment/core_code/runtime_repo/tests/test_text_benchmarks.py` adds regression coverage for relevant paragraph selection, supporting-title retention, and prompt tail preservation.
- Local verification:
  - `cd Experiment/core_code/runtime_repo; $env:PYTHONPATH='src'; python -m unittest discover -s tests` passed with 18 tests.
  - `python scripts/check_release_readiness.py --root .` passed.
  - `python scripts/build_release_tree.py --force` rebuilt the release preview and its readiness check passed.
- Remote verification:
  - Runtime repo: `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote unit tests passed with 18 tests.
  - Prompt inspection on real HotpotQA examples showed prompt lengths below the 1024-token truncation limit in sampled cases and retained supporting titles such as `[Shirley Temple]`.
- Small validation run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_baseline_fix_fullonly_20260610_114138`.
  - Command path: `logs/launcher.stdout.log` and `logs/lite_to_7b_eval_hotpotqa/stderr.log`.
  - Dataset/methods: HotpotQA n=50, Full Consumer only.
  - Result: Full Consumer F1 `0.46888161596857253`, EM `0.36`, avg NLL `0.9992654905632627`, TTFT `103.24004753492773` ms.
  - Sample check: first prediction changed to `Yes` for gold `yes`, with `prompt_tokens=865`.
- Formal rerun:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_baseline_fix_full_scd_20260610_114430`.
  - Dataset/methods: HotpotQA n=200, Full Consumer and SCD Reuse19/Patch13, no fallback/no recompute.
  - Full Consumer: F1 `0.46895380394210323`, EM `0.325`, avg NLL `0.8669146422592677`, TTFT `125.31250928295776` ms.
  - SCD Reuse19/Patch13: F1 `0.1665496154172625`, EM `0.12`, avg NLL `1.1563555703977766`, TTFT `48.30418422818184` ms.
  - Relative note: SCD is much faster on HotpotQA under this prompt, but quality retention is poor compared with the fixed Full baseline.
- Conclusion:
  - The HotpotQA Full baseline abnormality is fixed.
  - Prior HotpotQA rows from `scd_icml_selected_reuse19_5method_aggregate_20260610_034426` are stale and should not be used for HotpotQA claims.
  - HotpotQA now indicates poor cross-dataset quality retention for the CMRC-selected Reuse19/Patch13 route, while CMRC2018 remains the primary route-selection evidence.
  - Remote GPUs were idle after completion: all four GPUs showed `18 MiB` used and `0%` utilization.

### 2026-06-10 HotpotQA full-context correction

- Correction: the preceding HotpotQA prompt repair used gold-evidence-trimmed context and is invalid for the intended HotpotQA evaluation.
- Required evaluation semantics:
  - Use complete HotpotQA distractor context from each dataset row.
  - Do not filter by `supporting_facts`.
  - Do not apply the prior `1024` token truncation.
- Code changes:
  - Reverted `Experiment/core_code/runtime_repo/src/scd_repro/text_benchmarks.py` to preserve full HotpotQA context.
  - Kept the prompt structure that repeats the question before and after the evidence.
  - Updated tests in `Experiment/core_code/runtime_repo/tests/test_text_benchmarks.py` to assert full context preservation.
  - Changed default `MAX_LENGTH` from `1024` to `0` in:
    - `Experiment/core_code/runtime_repo/scripts/run_ser_scd_icml_eval_existing_route.sh`.
    - `Experiment/core_code/runtime_repo/scripts/run_ser_scd_icml_mistral_repro.sh`.
- Verification:
  - Local `PYTHONPATH=src python -m unittest discover -s tests` passed with 17 tests.
  - Local `python scripts/check_release_readiness.py --root .` passed.
  - Local `python scripts/build_release_tree.py --force` passed and rebuilt the release preview.
  - Remote unit tests passed with 17 tests.
- Small full-context validation run:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_fullonly20_20260610_115823`.
  - Dataset/methods: HotpotQA n=20, Full Consumer only, `MAX_LENGTH=0`.
  - Result: Full Consumer F1 `0.49638278388278384`, EM `0.35`, avg NLL `0.972549374183912`, TTFT `211.1234124749899` ms.
  - Evidence of no truncation: first sampled prompt token counts included `1423`, `1571`, `1797`, `1282`, and `1568`, all above the old `1024` cap.
- Formal full-context rerun:
  - Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`.
  - Dataset/methods: HotpotQA n=200, Full Consumer and SCD Reuse19/Patch13, no fallback/no recompute, `MAX_LENGTH=0`.
  - Full Consumer: F1 `0.4004526764135436`, EM `0.265`, avg NLL `0.9124961560906212`, TTFT `216.15322473226115` ms.
  - SCD Reuse19/Patch13: F1 `0.13827166297754534`, EM `0.105`, avg NLL `1.4230384913570084`, TTFT `67.06018289318308` ms.
  - Sample check: first five prompt token counts were `1423`, `1571`, `1797`, `1282`, and `1568`; these confirm the run used full context rather than the previous 1024-token truncation.
- Conclusion:
  - The valid repaired HotpotQA baseline is the full-context run above, not the gold-evidence-trimmed run.
  - Full-context HotpotQA baseline is now non-abnormal relative to the previous broken `0.034` F1, but it remains lower than the trimmed-context run because the task includes full distractor evidence.
  - The CMRC-selected Reuse19/Patch13 route has poor HotpotQA full-context retention: F1 retention approximately `0.3453` and EM retention approximately `0.3962`.

### 2026-06-10 HotpotQA retrain with CMRC-selected route layers

- Purpose: retrain Reuse and Patch on HotpotQA itself while keeping the CMRC-selected route layers fixed, then evaluate on a disjoint HotpotQA heldout split.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944`.
- Runtime repo: `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Route:
  - Reuse layers: `[3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`.
  - Patch layers: `[0,1,2,4,7,10,12,15,16,18,19,26,31]`.
- Data split:
  - Full-context HotpotQA, `MAX_LENGTH=0`.
  - Train prompts: first 200 samples.
  - Heldout eval qids: next 200 samples.
  - Launcher evidence: `wrote_hotpotqa_split train=200 eval=200 overlap=0`.
- Training artifacts:
  - Pair artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/pairs/hotpotqa_train_seed42_n200`.
  - Reuse checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/ckpt_reuse/hotpotqa_seed42`.
  - Patch checkpoint: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/ckpt_patch/patch_hotpotqa_seed42_cmrc_layers`.
  - Heldout eval summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/strict_eval/hotpotqa/summary.csv`.
- Heldout result:

| Method | n | F1 | EM | Avg NLL | TTFT ms |
|---|---:|---:|---:|---:|---:|
| Full Consumer | 200 | 0.383759 | 0.235 | 1.063052 | 217.594 |
| SCD | 200 | 0.092718 | 0.065 | 1.629119 | 116.679 |

- Analysis:
  - HotpotQA-specific Reuse/Patch training on the fixed CMRC route still gives poor full-context HotpotQA retention: F1 retention approximately `0.2416`, EM retention approximately `0.2766`.
  - The SCD path is faster in TTFT on this run, but the answer-quality loss is too large for HotpotQA claims.
  - The route remains CMRC-selected; this experiment does not justify using the same layer split as a HotpotQA quality route.
- Engineering follow-up:
  - `remote_scripts/run_hotpotqa_train_selected_route.sh` now defaults to `SKIP_FIXED_ROUTE_PROFILE=1`.
  - For fixed route runs, the script writes a minimal profile JSON and lets `train_patch --route` choose the patch layers directly.
  - Full `profile_layers` can still be forced by setting `SKIP_FIXED_ROUTE_PROFILE=0`.
  - The run above used the older full-profile path because it had already progressed past profile before the fast-path patch was applied.

### 2026-06-10 HotpotQA full-bank route composition

- Purpose: train a full patch-only bank and evaluate a reuse+patch mixed route by selecting layers from full reuse and full patch banks.
- Method:
  - Reuse bank source: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/ckpt_reuse/hotpotqa_seed42`.
  - Pair artifact source: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/pairs/hotpotqa_train_seed42_n200`.
  - Full patch bank route: train patch layers `[0,1,2,...,31]` with no reuse layers.
  - Mixed evaluation route: CMRC-selected Reuse19/Patch13 route.
  - Heldout split: same disjoint HotpotQA split as the preceding run, with train first 200 full-context samples and eval next 200 full-context samples.
- Script:
  - Local: `remote_scripts/run_hotpotqa_full_banks_from_existing.sh`.
  - Remote target: `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/scripts/run_hotpotqa_full_banks_from_existing.sh`.
- Status:
  - Completed.
- Run root:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`.
- Verification:
  - Full patch bank checkpoint wrote `selected_layers` and `residual_layers` for all layers `[0,1,2,...,31]`.
  - Mixed evaluation route remained the CMRC-selected Reuse19/Patch13 route.
  - Train patch bank wall time: `1:57:14`.
  - Heldout eval wall time: `9:09.48`.
- Heldout full-context HotpotQA result:

| Method | n | F1 | EM | Avg NLL | TTFT ms |
|---|---:|---:|---:|---:|---:|
| Full Consumer | 200 | 0.38375940307679435 | 0.235 | 1.0630523037492183 | 217.08079910371453 |
| SCD mixed from full banks | 200 | 0.09200354146235153 | 0.065 | 1.6303061479713485 | 116.00739535642788 |

- Analysis:
  - Training a full patch bank and composing Reuse19/Patch13 by layer did not improve HotpotQA full-context quality versus route-specific patch training.
  - The SCD mixed-from-banks F1/EM is essentially unchanged from the prior route-specific HotpotQA-trained patch result (`0.09271782717663724` / `0.065`).
  - The bottleneck is therefore not merely that the patch checkpoint was trained only on selected patch layers; the CMRC-selected Reuse19/Patch13 split itself is not a good full-context HotpotQA quality route.
  - The older GitHub snapshot's stronger HotpotQA patch-only result should be treated as a separate compressed-prompt benchmark, not as evidence for the current full-context HotpotQA setting.

### 2026-06-10 Old snapshot HotpotQA patch-only review

- Purpose: inspect the older GitHub snapshot because it reported stronger HotpotQA patch-only quality.
- Snapshot runtime:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runtime_repo`.
  - Core files inspected:
    - `src/scd_repro/hotpotqa.py`.
    - `src/scd_repro/cli/eval_hotpotqa_strict.py`.
- Result directory:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`.
- Recorded results:
  - Full 7405 eval, Patch-only: F1 `0.3516585390945674`, EM `0.2521269412559082`, avg NLL `2.450744741980222`, TTFT `114.33006256144647` ms.
  - Full 7405 eval, Full Consumer: F1 `0.19715099982161022`, EM `0.09898717083051992`, avg NLL `2.296655631430877`, TTFT `184.5385026596713` ms.
  - Heldout100 eval, Patch-only: F1 `0.2724804905239688`, EM `0.21`.
  - Heldout100 eval, Full Consumer: F1 `0.18491259889535752`, EM `0.12`.
- Important semantic differences from current full-context HotpotQA:
  - The old `hotpotqa.py` compresses evidence with question-aware passage and sentence ranking.
  - The old loader limits evidence to approximately 900 characters, at most 4 passages, and at most 3 selected sentences per passage.
  - The old eval commands used `--max-length 512`.
  - The patch checkpoint was trained from 100 compressed HotpotQA prompts. The retained train split uses prompt-hash qids rather than original HotpotQA `_id` values, so direct qid overlap against the 7405 eval is not meaningful.
  - A follow-up question-text overlap check found that all 100 train questions are present in the 7405 eval. Removing those 100 questions leaves the remaining 7305 samples at Full Consumer F1 `0.19743815119789307`, EM `0.09911019849418207`, and Patch-only F1 `0.35144349416338316`, EM `0.2516084873374401`, so the strong 7405 Patch-only result is not mainly explained by those 100 train questions.
  - The retained training capture prompts use an older Chinese short-answer template, while `src/scd_repro/hotpotqa.py` uses an English short-answer prompt for this full7405 eval; this is another reason to keep the old result scoped as its own compressed-prompt benchmark.
  - The current HotpotQA path intentionally uses complete distractor context with `MAX_LENGTH=0`; therefore these old compressed-prompt results are not directly comparable to the current full-context results.
- Per-sample diagnosis:
  - In the 7405 paired comparison, Patch-only had higher F1 than Full Consumer on 2293 samples, Full Consumer was higher on 812 samples, and 4300 samples tied.
  - Patch-only produced an exact match where Full Consumer did not on 1260 samples; the reverse happened on 126 samples.
  - Patch-only and Full Consumer produced exactly the same normalized prediction text on 1854 samples.
  - Patch-only outputs were much shorter than Full Consumer outputs: median 2 generated words versus median 11 generated words. This is consistent with the old prompt and patch-only runtime favoring short answer spans.
  - Patch-only still beat Full Consumer on the retained heldout100 split: F1 `0.2724804905239688`, EM `0.21`, versus Full Consumer F1 `0.18491259889535752`, EM `0.12`.
- Interpretation:
  - The old patch-only result is useful as a separate compressed-prompt reproduction target.
  - It should not replace the current full-context HotpotQA benchmark unless the experiment is explicitly scoped to the old compressed setting.

### 2026-06-10 Main-result reproducibility audit

- Purpose: audit whether the current paper/open-source SCD-ICML Mistral result
  appears runnable and non-fabricated, with Qwen32B excluded from scope.
- Static checks:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo\release_preview\scd_icml_runtime_release` passed.
  - Remote `python scripts/check_release_readiness.py --root .` in
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`
    exited with code 0.
  - Remote `check_release_readiness.py --run-root` exited with code 0 for the
    selected Reuse19 aggregate, old Mistral repro, paper-mixed root, and
    Reuse20-32 quality-curve root. This is structural evidence only, not a
    quality-retention gate.
- Main evidence retained:
  - Selected Reuse19 aggregate root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
  - CMRC2018 dev `n=200`: Full Consumer F1 `0.657307250029563`, EM `0.365`;
    SCD Reuse19/Patch13 F1 `0.6427511605994372`, EM `0.35`.
  - F1 retention `0.9778549689974191`; EM retention `0.958904109589041`.
- Results excluded from positive main claims:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`
    has training/profile/pair artifacts but no completed strict-eval summary
    rows found during this audit.
  - Full-context HotpotQA with the CMRC-selected route is negative: run
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952`
    reported Full Consumer F1 `0.4004526764135436`, EM `0.265`, while SCD
    reported F1 `0.13827166297754534`, EM `0.105`.
  - HotpotQA full-bank composition also failed to recover quality:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156`
    reported Full Consumer F1 `0.38375940307679435`, EM `0.235`, while SCD
    mixed from banks reported F1 `0.09200354146235153`, EM `0.065`.
  - WikiText-2 from the selected aggregate is an auxiliary risk:
    Full Consumer PPL `7.584028681680868`, SCD PPL `26.423097289857097`.
- Old snapshot interpretation:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`
    has real patch-only HotpotQA evidence, but under compressed-prompt
    semantics: Patch-only F1 `0.3516585390945674`, EM `0.2521269412559082`;
    Full Consumer F1 `0.19715099982161022`, EM `0.09898717083051992`.
  - It should be treated as a separate compressed HotpotQA reproduction target,
    not as current full-context evidence.
- Artifact:
  - Added `docs/scd-main-result-reproducibility-audit-2026-06-10.md`.
- Conclusion:
  - No current evidence suggests the Mistral CMRC2018 main result is fabricated
    or unrunnable.
  - The safe positive claim is narrower than three-dataset quality preservation:
    CMRC2018 supports Reuse19/Patch13; HotpotQA and WikiText-2 should be
    presented as limitations/diagnostics unless separately revalidated.

### 2026-06-10 Release default wrapper smoke

- Purpose: confirm that the release default wrapper path runs end-to-end after
  changing defaults to the selected CMRC2018 Reuse19/Patch13 route.
- Run root:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_default_smoke_20260610_selected_cmrc`.
- Command profile:
  - `TRAIN_LIMIT=2`, `EVAL_LIMIT=2`, `DATASETS=cmrc2018`, `MAX_LENGTH=0`,
    `MAX_NEW_TOKENS=2`, `CUDA_VISIBLE_DEVICES=0,1`.
  - Wrapper default route mode used the selected template route with 19 reuse
    layers and 13 patch layers.
- Status:
  - No remaining remote process for this run was found.
  - `scd_icml_mistral_summary.json` and `strict_eval/cmrc2018/summary.json`
    were generated.
- Smoke metrics:
  - Full Consumer n=2: F1 `0.5666666666666667`, EM `0.0`, avg NLL
    `2.89990234375`, TTFT `189.3705613911152` ms.
  - SCD n=2: F1 `0.5666666666666667`, EM `0.0`, avg NLL `3.107421875`,
    TTFT `55.47775258310139` ms.
- Interpretation:
  - This is a structural smoke only because n=2 and `MAX_NEW_TOKENS=2`.
  - It supports that the release wrapper now reaches the intended default route
    and emits comparable Full/SCD outputs, but it is not a quality-retention
    substitute for the n=200 CMRC2018 result.

### 2026-06-10 Paper claim-to-evidence audit and release manifest correction

- Purpose: map the archived ICML camera-ready paper claims to current verified
  Mistral release evidence.
- Artifact:
  - Added `docs/scd-paper-claim-to-evidence-audit-2026-06-10.md`.
- Main finding:
  - The archived ICML TeX still contains historical broad claims that are not
    the current release evidence, including Mistral F1 `0.807 -> 0.769`,
    Mistral SCD F1 `0.7850`, WikiText-2 SCD PPL `4.65`, and Qwen-32B rows.
  - The current verified positive claim is narrower: CMRC2018 dev `n=200`,
    Reuse19/Patch13, Full Consumer F1 `0.657307250029563`, EM `0.365`, versus
    SCD F1 `0.6427511605994372`, EM `0.35`.
  - HotpotQA full-context and WikiText-2 remain limitations or diagnostics for
    the current selected route.
- Documentation updates:
  - Added superseded notes to `docs/scd-icml-paper-release-audit-2026-06-08.md`
    and `docs/scd-icml-open-source-release-status-2026-06-08.md`.
  - Updated `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`
    to include the all-layer Patch bank config required by the current default
    wrapper and to point the run-root check at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
- Verification:
  - Local `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - Local `python scripts\build_release_tree.py --force` in the runtime repo
    rebuilt `release_preview/scd_icml_runtime_release`.
  - Local `python scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release` passed.
  - Synced the updated manifest and docs to `ser`.
  - Remote `python scripts/build_release_tree.py --force`,
    `python scripts/check_release_readiness.py --root .`, and
    `python scripts/check_release_readiness.py --root release_preview/scd_icml_runtime_release`
    exited with code 0 in
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.

### 2026-06-10 Compressed HotpotQA boundary propagation

- Purpose: make the release-facing docs and paper-results fragment reflect the
  refined diagnosis of the old GitHub HotpotQA Patch-only result.
- Inputs:
  - Old snapshot result root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`.
  - Old compressed-prompt result: Patch-only F1 `0.3516585390945674`, EM
    `0.2521269412559082` on 7405 examples.
  - Question-overlap audit result: after removing the 100 training questions,
    the remaining 7305 examples retain Patch-only F1 `0.35144349416338316`, EM
    `0.2516084873374401`.
- Method:
  - Updated release docs and the TeX-ready current-results fragment to state
    that the old result is not primarily explained by the small train/eval
    overlap, but remains a separate compressed-prompt benchmark because it uses
    question-aware evidence compression and `max_length=512`.
- Files updated:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - `Publication/paper/scd-icml-current-results-revision-2026-06-10.tex`.
  - `docs/scd-paper-claim-to-evidence-audit-2026-06-10.md`.
- Verification:
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo` passed.
  - `python -m py_compile Experiment\core_code\runtime_repo\scripts\check_release_readiness.py Experiment\core_code\runtime_repo\scripts\build_release_tree.py` passed.
  - `python scripts\build_release_tree.py --force` rebuilt
    `release_preview/scd_icml_runtime_release`.
  - `python scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release` passed.
  - `rg` confirmed the question-text overlap note appears in source docs and in
    the rebuilt release preview.

### 2026-06-10 Current claim index and historical artifact warning pass

- Purpose: make the current reproducibility claim boundary easy to find and
  reduce accidental reuse of stale paper/package artifacts.
- Inputs inspected:
  - `docs/scd-icml-mistral-reproduction-2026-06-07.md`.
  - `archive/icml-completed-2026-05-24/camera-ready-and-final/final/README.md`.
  - `archive/icml-completed-2026-05-24/camera-ready-and-final/final/evidence/verified-results.md`.
  - Historical TeX sources under
    `archive/icml-completed-2026-05-24/camera-ready-and-final/`.
- Actions:
  - Added `docs/scd-current-claim-index-2026-06-10.md`.
  - Marked `docs/scd-icml-mistral-reproduction-2026-06-07.md` as superseded.
  - Added a superseded notice to the archive final package README.
  - Added
    `archive/icml-completed-2026-05-24/camera-ready-and-final/final/evidence/CURRENT_CLAIM_NOTICE.md`
    because the adjacent historical evidence file has non-normal text encoding
    and should not be rewritten directly.
  - Added non-rendering TeX comments to the two historical `example_paper.tex`
    copies so source readers see the current-claim warning without changing PDF
    output.
- Verification:
  - `rg` found `CURRENT-CLAIM NOTICE`, `Superseded Current-Claim Notice`, and
    `scd-current-claim-index` in the intended files.
  - `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py --root Experiment\core_code\runtime_repo`
    passed after these changes.

### 2026-06-10 Historical HotpotQA Patch-only code/result inspection

- Purpose: inspect the old GitHub snapshot that produced the strong HotpotQA
  Patch-only result and determine how it relates to the current release tree.
- Inputs:
  - Remote code root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runtime_repo`.
  - HotpotQA-specific files:
    `src/scd_repro/hotpotqa.py` and
    `src/scd_repro/cli/eval_hotpotqa_strict.py`.
  - Result root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423`.
- Findings:
  - The old `hotpotqa.py` does question-aware evidence compression: it ranks
    passages/sentences by question terms, numbers, and extracted entity-like
    phrases, keeps at most 4 passages, and caps evidence at 900 characters.
    This is not the same as the current full-context HotpotQA path.
  - The current local release tree does not contain `hotpotqa.py` or
    `eval_hotpotqa_strict.py`; it uses the generic `text_benchmarks.py` /
    `eval_text_strict.py` path, whose HotpotQA loader renders the full context
    instead of the old question-aware compressed evidence.
  - The full7405 old run used HotpotQA-trained checkpoints from
    `strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422` and route
    `configs/route_patch_only_all32.json`, where all 32 layers are selected for
    Patch-only and there is no fallback or recompute.
  - Full7405 old-result metrics:
    - Full Consumer: F1 `0.19715099982161022`, EM `0.09898717083051992`,
      avg NLL `2.296655631430877`, TTFT `184.5385026596713` ms.
    - Patch-only: F1 `0.3516585390945674`, EM `0.2521269412559082`,
      avg NLL `2.450744741980222`, TTFT `114.33006256144647` ms.
  - The training split manifest for the checkpoint records `train_samples=100`,
    `eval_offset=100`, `eval_samples=100`, and `load_limit=200`. The heldout100
    old run uses a qid allowlist from that heldout split.
  - Heldout100 old-result metrics:
    - Full Consumer: F1 `0.18491259889535752`, EM `0.12`, avg NLL
      `3.0921409197573393`, TTFT `103.0296441540122` ms.
    - Reuse-only: F1 `0.0009`, EM `0.0`, avg NLL `9.6628`, TTFT `79.14` ms.
    - Patch-only: F1 `0.2724804905239688`, EM `0.21`, avg NLL
      `3.5179453627118527`, TTFT `83.48573919385672` ms.
- Interpretation:
  - The old Patch-only result is a real result under the old compressed-prompt
    HotpotQA protocol, and the heldout100 result argues against explaining it
    purely as direct train/eval leakage.
  - It should not be merged with the current full-context HotpotQA baseline:
    the baseline, prompt, context length, and loader semantics differ. The
    immediate code action, if HotpotQA is kept as a benchmark, is to restore the
    old HotpotQA-specific compressed-evidence loader/eval path as a separate
    named protocol rather than silently comparing it with the full-context path.

### 2026-06-10 HotpotQA compressed protocol restored as named runtime path

- Purpose: make the older strong HotpotQA Patch-only result reproducible as its
  own protocol without mixing it with the current full-context HotpotQA
  limitation rows.
- Code changes:
  - Added `hotpotqa_compressed` to
    `Experiment/core_code/runtime_repo/src/scd_repro/text_benchmarks.py`.
  - The loader ports the old question-aware evidence compression behavior:
    entity-like question phrase matching, question-term/numeric overlap scoring,
    at most 4 passages, at most 3 sentences per passage, and a 900-character
    evidence cap.
  - Added CLI support through
    `Experiment/core_code/runtime_repo/src/scd_repro/cli/eval_text_strict.py`
    with `--dataset hotpotqa_compressed`.
  - Added tests in
    `Experiment/core_code/runtime_repo/tests/test_text_benchmarks.py` to keep
    full-context HotpotQA behavior unchanged while checking the compressed
    prompt text, evidence cap, and question-relevant passage retention.
- Documentation changes:
  - Updated runtime `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` to state that the old HotpotQA Patch-only
    result is a separate compressed-prompt benchmark exposed as
    `--dataset hotpotqa_compressed`.
  - Rebuilt the safe paper PDF after simplifying the non-CMRC diagnostics table
    labels; no result numbers were changed.
- Verification:
  - `$env:PYTHONPATH='src'; python -m pytest tests\test_text_benchmarks.py`
    passed with 4 tests.
  - `$env:PYTHONPATH='src'; python -m pytest tests` passed with 19 tests.
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --force` rebuilt
    `release_preview/scd_icml_runtime_release` and the embedded readiness check
    passed.
  - `python scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release`
    passed.
  - `rg -n hotpotqa_compressed release_preview\scd_icml_runtime_release`
    confirmed that the code and docs are present in the release preview.
  - `pdflatex` rebuilt
    `Publication/paper/current_safe_build/example_paper.pdf`; the final PDF was
    copied to `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`.
  - `pdftotext` plus stale-claim `rg` found no old broad-claim strings in the
    rebuilt PDF text, and the current key numbers were present.

### 2026-06-10 Release preview self-test and evidence-file audit

- Purpose: reduce open-source reproduction risk beyond static readiness by
  ensuring the release preview contains model-free tests and that referenced
  result roots have concrete summary files.
- Issue found:
  - `README.md` still had a generic Quick Start that referenced missing
    placeholder paths: `configs/models.yaml`, `configs/data.yaml`, and
    `configs/train.yaml`. A user following that first block would fail before
    reaching the real Mistral reproduction path.
- Code/documentation changes:
  - Replaced the generic Quick Start with install, readiness, unittest, and
    Mistral smoke-test commands.
  - Updated `scripts/build_release_tree.py` to include `tests/` in the release
    preview.
  - Updated `scripts/check_release_readiness.py` to require all lightweight test
    files and to fail if `README.md` references missing quick-start config
    paths.
  - Updated `OPEN_SOURCE_RELEASE_MANIFEST.md` to list `tests/` as a required
    lightweight test artifact.
  - Updated `docs/scd-current-claim-index-2026-06-10.md` with concrete evidence
    summary file paths and model-free verification commands.
  - Updated the current safe paper and replacement fragment to state that the
    historical compressed HotpotQA protocol is exposed as
    `hotpotqa_compressed`.
- Verification:
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed with
    19 tests.
  - `$env:PYTHONPATH='src'; python -m pytest tests` passed with 19 tests.
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --force` rebuilt the release preview
    and its embedded readiness check passed.
  - From `release_preview/scd_icml_runtime_release`,
    `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed with
    19 tests.
  - From `release_preview/scd_icml_runtime_release`,
    `python scripts\check_release_readiness.py --root .` passed.
  - `rg` confirmed the preview build report includes `tests/`, the README
    contains `python -m unittest discover -s tests`, and no invalid
    `configs/models.yaml`, `configs/data.yaml`, or `configs/train.yaml` quick
    start references remain.
  - The safe paper PDF was rebuilt and copied to
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`.
    `pdftotext` plus stale-claim `rg` found no old broad-claim strings, and the
    text contains the current CMRC, HotpotQA full-context, compressed HotpotQA,
    and Qwen-deferred boundary statements.
- Remote evidence check:
  - The aggregate evidence file exists at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426/scd_icml_mistral_summary.json`
    and reports CMRC2018 Full Consumer F1/EM `0.657307250029563`/`0.365` and
    SCD F1/EM `0.6427511605994372`/`0.35`.
  - The full-context HotpotQA evidence file exists at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_fullcontext_full_scd_20260610_115952/scd_icml_mistral_summary.json`
    and reports Full Consumer F1/EM `0.4004526764135436`/`0.265` and SCD F1/EM
    `0.13827166297754534`/`0.105`.
  - The HotpotQA full-bank heldout evidence file exists at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_full_banks_20260610_192156/summary.json`
    and reports Full Consumer F1/EM `0.38375940307679435`/`0.235` and SCD
    F1/EM `0.09200354146235153`/`0.065`.
  - The historical compressed HotpotQA evidence files exist under
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_full7405_baseline_patchonly_20260423/reports/hotpotqa/`
    and report Full Consumer F1/EM `0.19715099982161022`/`0.09898717083051992`
    plus Patch-only F1/EM `0.3516585390945674`/`0.2521269412559082`.

### 2026-06-10 Current release-preview remote smoke test

- Purpose: verify that the current local release preview, after the README and
  `tests/` packaging fixes, can be copied to `ser`, installed, and run through
  the actual model/data wrapper on a minimal CMRC smoke configuration.
- Remote code copy:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/release_preview_from_codex_20260610_2328/runtime_repo`.
- Remote run root:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/release_preview_smoke_20260610_2328`.
- Command shape:
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `TRAIN_LIMIT=2`
  - `EVAL_LIMIT=1`
  - `DIRECTIONS=lite_to_7b`
  - `DATASETS=cmrc2018`
  - `MAX_LENGTH=256`
  - `MAX_NEW_TOKENS=8`
  - `WAIT_CAPTURE_FREE_MB=0`
  - `WAIT_TRAIN_FREE_MB=0`
  - `WAIT_EVAL_FREE_MB=0`
  - `bash scripts/run_ser_scd_icml_mistral_repro.sh`
- Pre-run GPU state:
  - `ser` GPUs 0-3 were idle with approximately 24075 MiB free each.
- Model-free checks on the remote copied release:
  - `python -m pip install -e .` completed.
  - `PYTHONPATH=src python -m unittest discover -s tests` exited with code 0.
  - `python scripts/check_release_readiness.py --root .` exited with code 0.
- Smoke result:
  - `scripts/check_release_smoke_status.py --run-root ... --pid 3011221`
    reported state `complete`.
  - Successful stages:
    `lite_to_7b_capture_pair`, `lite_to_7b_train_reuse`,
    `lite_to_7b_profile_layers`, `lite_to_7b_train_patch`,
    `lite_to_7b_route_template`, and `lite_to_7b_eval_cmrc2018`.
  - Aggregate summaries exist:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/release_preview_smoke_20260610_2328/scd_icml_mistral_summary.json`
    and `.md`.
  - The smoke wrote 2 aggregate rows, which is expected for
    `DATASETS=cmrc2018` with the selected Full+SCD eval config.
- Post-run GPU state:
  - GPUs 0-3 returned to idle with approximately 24075 MiB free each.
## 2026-06-10 - Non-Qwen main-result reproduction audit and Patch-only anchor integration

- Experiment id: `main_result_reproduction_audit_patchonly_anchor_20260610`
- Purpose: answer whether the SCD-ICML main results are currently reproducible and establish a clean open-source entry point for the older GitHub/new-main Patch-only trend.
- Method:
  - Parsed the archived camera-ready TeX tables and figures under `archive/icml-completed-2026-05-24/camera-ready-and-final/final/paper/icml2026-camera-ready-source/example_paper.tex`.
  - Compared those claims with current strict Reuse19/Patch13 evidence and historical GitHub snapshot evidence under `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854`.
  - Added `Experiment/core_code/runtime_repo/scripts/run_ser_scd_icml_patchonly_anchor.sh`, which evaluates Full Consumer, Reuse-only, and Patch-only with dataset-specific historical Reuse/Patch banks for CMRC2018, HotpotQA compressed prompting, and WikiText-2.
  - Added the anchor script and required all32 route / full-dev eval configs to release readiness and release preview build lists.
- Results:
  - Current strict Reuse19/Patch13 route supports only a narrow CMRC trend: Full F1/EM `0.657307/0.365`, SCD `0.642751/0.350`.
  - Current full-context HotpotQA contradicts the broad preserved-quality claim for that route: Full F1/EM `0.400453/0.265`, SCD `0.138272/0.105`.
  - Historical forward Patch-only snapshot `strict_full_bidir_mistral_full_20260424` is a strong trend anchor:
    - CMRC2018: Full F1/EM `0.691118/0.442684`, Reuse-only `0.009836/0.0`, Patch-only `0.789799/0.576887`, TTFT `118.30 -> 48.77 ms`.
    - HotpotQA historical compressed protocol: Full F1/EM `0.274911/0.201621`, Reuse-only `0.001225/0.0`, Patch-only `0.336180/0.252802`, TTFT `80.89 -> 46.07 ms`.
    - WikiText-2: Full PPL `7.671660`, Reuse-only PPL `11.525732`, Patch-only PPL `7.791208`, TTFT `60.02 -> 39.70 ms`.
  - Local `python scripts/check_release_readiness.py` passed.
  - Release preview rebuilt at `Experiment/core_code/runtime_repo/release_preview/scd_icml_runtime_release` and passed readiness.
  - Local and release-preview unit tests passed with `PYTHONPATH=src python -m unittest discover -s tests` (`19` tests).
  - A manual remote CMRC patch-only smoke with old ckpt and current runtime was launched as PID `3029982` but produced no files after several minutes; it was stopped to free GPU0/1. This failed smoke should not be counted as evidence.
  - The new patch-only anchor wrapper was copied to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/scripts/run_ser_scd_icml_patchonly_anchor.sh` and passed remote `bash -n`.
- Analysis:
  - The answer to "can the main results be reproduced?" is currently "not fully under one clean release workflow."
  - The old GitHub Patch-only path does support the main trend across the three non-Qwen datasets, but the current mixed Reuse+Patch route needs more optimization before it can replace Patch-only as the paper-facing route.
  - The next route search should maximize Reuse layer count subject to F1/EM retention, treating NLL/PPL as auxiliary except for WikiText-2-specific claims.
- Next actions:
  - Run `scripts/run_ser_scd_icml_patchonly_anchor.sh` with `EVAL_LIMIT=1 MAX_LENGTH=256` first to validate runtime compatibility with logs, then full `EVAL_LIMIT=0` after the smoke writes summaries.
  - Rebuild mixed Reuse+Patch routes from full Reuse/Patch banks and compare them against the Patch-only anchor, dataset by dataset.
  - Regenerate patch-budget, rank-sweep, bandwidth, TTFT, and layer-selection figures from saved JSON/CSV instead of relying on old static assets.

## 2026-06-11 - Current-runtime Patch-only anchor rerun on three non-Qwen datasets

- Experiment id: `patchonly_anchor_200_three_dataset_20260611`
- Purpose:
  - Recover the GitHub/new-main Patch-only trend under the current
    `runtime_repo` entry point.
  - Establish a fresh current-runtime anchor before optimizing the final mixed
    Reuse+Patch route.
- Code and environment:
  - Local source: `Experiment/core_code/runtime_repo`.
  - Remote source:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote data/checkpoint snapshot:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854`.
  - Python environment: `/disk_n/c101-3-mql/venvs/A800`.
  - Wrapper: `scripts/run_ser_scd_icml_patchonly_anchor.sh`.
- Commands:
  - CMRC2018:
    `CUDA_VISIBLE_DEVICES=0,1 DATASETS=cmrc2018 EVAL_LIMIT=200 MAX_LENGTH=1024 MAX_NEW_TOKENS=32 WAIT_EVAL_FREE_MB=0 RUNTIME_PATCH_WINDOW_SIZE=1 PATCH_CKPT_DEVICE=cpu bash scripts/run_ser_scd_icml_patchonly_anchor.sh`
  - HotpotQA-compressed + WikiText-2:
    `CUDA_VISIBLE_DEVICES=2,3 DATASETS=hotpotqa_compressed,wikitext2 EVAL_LIMIT=200 MAX_LENGTH=1024 MAX_NEW_TOKENS=32 WAIT_EVAL_FREE_MB=0 RUNTIME_PATCH_WINDOW_SIZE=1 PATCH_CKPT_DEVICE=cpu bash scripts/run_ser_scd_icml_patchonly_anchor.sh`
- Artifacts:
  - CMRC2018:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/patchonly_anchor_cmrc200_20260611_001/patchonly_anchor_summary.md`
  - HotpotQA-compressed + WikiText-2:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/patchonly_anchor_hotpot_wiki200_20260611_002/patchonly_anchor_summary.md`
- Results:
  - CMRC2018: Full F1/EM `0.657307/0.365000`, Reuse-only
    `0.012848/0.000000`, Patch-only `0.774237/0.515000`; TTFT
    `100.20 -> 52.32 ms`.
  - HotpotQA-compressed: Full F1/EM `0.330552/0.250000`, Reuse-only
    `0.007903/0.000000`, Patch-only `0.372672/0.305000`; TTFT
    `78.94 -> 44.77 ms`.
  - WikiText-2: Full PPL `7.584029`, Reuse-only PPL `13.631409`,
    Patch-only PPL `7.913751`; TTFT `62.21 -> 40.66 ms`.
- Code changes made after the run:
  - Added optional `RESUME=1` behavior to
    `scripts/run_ser_scd_icml_patchonly_anchor.sh`, skipping completed
    dataset/stage outputs that already contain `per_mode_summary.json`.
  - Documented `RESUME=1` in `README.md` and `REPRODUCING_SCD_ICML.md`.
  - Synchronized the updated wrapper and docs to the remote runtime repo after
    the running anchor completed.
- Verification:
  - Local `python scripts/check_release_readiness.py` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed
    with `19` tests.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    rebuilt the release preview and its embedded readiness check passed.
  - From `release_preview/scd_icml_runtime_release`,
    `python scripts\check_release_readiness.py --root .` passed.
  - From `release_preview/scd_icml_runtime_release`,
    `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed with
    `19` tests.
  - Remote temporary `bash -n` and final remote
    `bash -n scripts/run_ser_scd_icml_patchonly_anchor.sh` passed.
  - Local and remote wrapper hashes matched after the corrected sync:
    `418ed5a766ceb0f214a26a3939ab169789e48647d0d91a49d1999e4bcf764158`.
  - The accidental duplicate remote wrapper copied to the runtime root was
    removed; the authoritative file is under `scripts/`.
- Analysis:
  - The three-dataset Patch-only anchor now has fresh current-runtime evidence
    and supports the old GitHub/new-main main trend.
  - This does not complete the paper-facing SCD result: Patch-only is the
    quality anchor, while the final objective still requires a mixed
    Reuse+Patch route with as many Reuse layers as possible and small F1/EM or
    PPL loss.
- Next actions:
  - Use `scripts/run_ser_scd_icml_conditional_route_screen.sh` as the work-queue
    route-screening entry point; it launches one worker per GPU pair and each
    worker dequeues the next route as soon as its pair is free.
  - Compare candidate mixed routes against the new three-dataset Patch-only
    anchor, with F1/EM primary for QA datasets and PPL primary for WikiText-2.
  - Regenerate paper tables/figures only after the accepted mixed route is
    selected and validated across the required datasets.

## 2026-06-11 - CMRC2018 snapshot-bank mixed Reuse+Patch prefix curve

- Experiment id: `snapshot_bank_prefix_curve_cmrc200_20260611`
- Purpose:
  - Find the maximum number of Reuse layers that preserves CMRC2018 F1/EM when
    mixing dataset-specific snapshot Reuse/Patch banks.
  - Use F1/EM as the primary quality gate and NLL/PPL as auxiliary risk signals.
- Code and environment:
  - Local source: `Experiment/core_code/runtime_repo`.
  - Remote source:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001`.
  - Reuse bank:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_reuse/cmrc2018_seed42`.
  - Patch bank:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/mistral_strict_default_gpu1_200_b2_20260421/ckpt_patch/patch_cmrc2018_seed42`.
- Method:
  - Used `scripts/run_ser_scd_icml_conditional_route_screen.sh` with
    `GPU_PAIRS=0,1;2,3`, `DATASETS=cmrc2018`, `EVAL_LIMIT=200`,
    `MAX_LENGTH=1024`, and `PREFIX_BUDGETS=8,12,16,19,20,21,22,23,24`.
  - Candidate order:
    `3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30,26,10,4,7,12,15,16,18,19,31,0,1,2`.
  - Fixed the route metric collector after the run so `route_metrics.json`
    also records selected/full EM, EM retention, EM delta, and selected TTFT.
- Results:
  - Full Consumer reference: F1/EM `0.657307/0.365000`, PPL `1.767002`.
  - `reuse08_patch24`: F1/EM `0.668018/0.360000`, F1 retention `1.0163`,
    PPL `1.969394`, TTFT `43.05 ms`.
  - `reuse12_patch20`: F1/EM `0.637175/0.355000`, F1 retention `0.9694`,
    PPL `2.137821`, TTFT `42.01 ms`.
  - `reuse16_patch16`: F1/EM `0.697745/0.425000`, F1 retention `1.0615`,
    PPL `2.176618`, TTFT `43.04 ms`.
  - `reuse19_patch13`: F1/EM `0.636821/0.340000`, F1 retention `0.9688`,
    PPL `2.464036`, TTFT `37.93 ms`.
  - `reuse20_patch12`: F1/EM `0.623676/0.360000`, F1 retention `0.9488`,
    PPL `2.694328`, TTFT `37.31 ms`.
  - `reuse21_patch11`: F1/EM `0.571623/0.335000`, F1 retention `0.8696`,
    PPL `3.556393`, TTFT `40.57 ms`.
  - `reuse22_patch10`: F1/EM `0.504905/0.245000`, F1 retention `0.7681`,
    PPL `6.886768`, TTFT `36.73 ms`.
  - `reuse23_patch09`: F1/EM `0.485349/0.250000`, F1 retention `0.7384`,
    PPL `7.793904`, TTFT `39.60 ms`.
  - `reuse24_patch08`: F1/EM `0.387813/0.185000`, F1 retention `0.5900`,
    PPL `16.259786`, TTFT `36.32 ms`.
- Artifacts:
  - Summary document:
    `docs/scd-icml-snapshot-bank-cmrc200-prefix-curve-2026-06-11.md`.
  - Remote metrics:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001/lite_to_7b/conditional_report/route_metrics.json`.
- Analysis:
  - The quality cliff begins after `reuse20_patch12`.
  - Under an F1 retention gate of `0.95`, `reuse19_patch13` is the conservative
    CMRC maximum.
  - `reuse20_patch12` is borderline: F1 retention is `0.9488`, but EM is almost
    unchanged relative to Full Consumer (`0.360` vs `0.365`).
  - `reuse21_patch11` and higher should be rejected for CMRC2018.
- Next actions:
  - Validate `reuse16/19/20` on full-context HotpotQA and WikiText-2 with
    dataset-specific snapshot banks before choosing the final paper-facing route.

## 2026-06-11 - HotpotQA full-context and WikiText-2 mixed-route boundary runs started

- Experiment id: `boundary_hotpotqa_wikitext2_200_20260611`
- Purpose:
  - Test whether the CMRC-selected boundary routes also preserve quality on
    HotpotQA and WikiText-2.
  - Use full-context `hotpotqa` loader rather than the compressed HotpotQA
    prompt protocol.
- Smoke runs:
  - HotpotQA full-context smoke:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpot_full_smoke_20260611_001`.
  - WikiText-2 smoke:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_smoke_20260611_001`.
  - Both smoke runs used `EVAL_LIMIT=5`, `PREFIX_BUDGETS=19`, and produced
    `scd_icml_mistral_summary.json` plus `route_metrics.json`.
- Full 200-sample boundary runs:
  - HotpotQA full-context run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_fullctx200_20260611_001`.
  - WikiText-2 run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001`.
  - HotpotQA uses GPU pair `0,1`; WikiText-2 uses GPU pair `2,3`.
  - Both use `PREFIX_BUDGETS=16,19,20`, `EVAL_LIMIT=200`,
    `MAX_LENGTH=1024`, `MAX_NEW_TOKENS=32`, `PATCH_CKPT_DEVICE=cpu`, and
    `RUNTIME_PATCH_WINDOW_SIZE=1`.
- Status at entry time:
  - Both 200-sample runs had launched and entered `reuse16_patch16`.
  - GPU memory/utilization showed all four GPUs occupied.
- Next actions:
  - Monitor route completion and collect F1/EM for HotpotQA plus PPL for
    WikiText-2.
  - If the 200-sample full-context HotpotQA boundary looks acceptable, run the
    selected route on full HotpotQA dev with `EVAL_LIMIT=0`.

## 2026-06-11 - HotpotQA protocol correction and WikiText-2 mixed-route results

- Experiment id: `boundary_hotpotqa_wikitext2_200_20260611_update`
- Purpose:
  - Decide whether the current non-Qwen main results can be treated as
    reproduced, and identify which dataset constrains the maximum reusable
    layer count.
- Protocol correction:
  - The full-context `hotpotqa` loader is not paper-comparable for the current
    reproduction target. Its 200-sample Full Consumer baseline was abnormal:
    F1/EM `0.032208/0.005000` at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_fullctx200_20260611_001`.
  - The GitHub snapshot's `full7405` HotpotQA result means all 7405 dev
    samples under the compressed evidence prompt protocol, not uncompressed
    long-context HotpotQA.
  - Paper-facing HotpotQA screening therefore uses `hotpotqa_compressed`,
    `MAX_LENGTH=512`, and the dataset-specific snapshot Reuse/Patch banks.
- HotpotQA-compressed 200-sample boundary:
  - Low-budget run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_compressed_low200_20260611_001`.
  - Full Consumer reference: F1/EM `0.330552/0.250000`, PPL `5.032427`.
  - `reuse04_patch28`: F1/EM `0.377345/0.290000`, F1 retention `1.1416`,
    PPL `6.765110`, TTFT `43.26 ms`.
  - `reuse08_patch24`: F1/EM `0.305450/0.235000`, F1 retention `0.9241`,
    F1 delta `-0.025102`, PPL `7.407202`, TTFT `41.26 ms`.
  - `reuse12_patch20`: F1/EM `0.250531/0.200000`, F1 retention `0.7579`,
    PPL `8.018260`, TTFT `39.29 ms`.
  - Mid-budget run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001`.
  - `reuse09_patch23`: F1/EM `0.249338/0.190000`, F1 retention `0.7543`,
    PPL `7.816221`, TTFT `40.44 ms`.
  - `reuse10_patch22` and `reuse11_patch21` were still running/pending at this
    log entry.
- WikiText-2 200-sample boundary:
  - Run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001`.
  - Full Consumer reference: PPL `7.584029`, avg NLL `2.026045`.
  - `reuse16_patch16`: PPL `8.236739`, relative PPL increase `8.61%`,
    avg NLL `2.108605`, TTFT `32.01 ms`.
  - `reuse19_patch13`: PPL `8.335916`, relative PPL increase `9.91%`,
    avg NLL `2.120573`, TTFT `30.63 ms`.
  - `reuse20_patch12`: PPL `8.329555`, relative PPL increase `9.83%`,
    avg NLL `2.119810`, TTFT `30.50 ms`.
- Analysis:
  - Patch-only anchors are recovered across CMRC2018, HotpotQA-compressed, and
    WikiText-2, but that is not sufficient to claim all mixed Reuse+Patch main
    results are fully reproduced.
  - CMRC2018 tolerates `reuse19_patch13` conservatively and `reuse20_patch12`
    as a borderline route.
  - WikiText-2 tolerates at least `reuse20_patch12` under a 10% relative PPL
    increase gate.
  - HotpotQA-compressed is the limiting dataset: `reuse08_patch24` has a small
    absolute F1 drop, but `reuse09_patch23` and `reuse12_patch20` are clear
    failures under F1 retention.
- Next actions:
  - Finish the pending HotpotQA-compressed `reuse10/11` jobs only for curve
    completeness; they are unlikely to change the accepted boundary because
    `reuse09_patch23` already failed.
  - Select whether the public result should use dataset-specific routes or a
    common conservative route.
  - Run the selected HotpotQA-compressed route on the full 7405-sample protocol
    before claiming full HotpotQA main-result reproduction.

## 2026-06-11 - HotpotQA-compressed full7405 mixed-route validation started

- Experiment id: `hotpotqa_compressed_full7405_reuse04_reuse08_20260611`
- Purpose:
  - Validate the paper-facing HotpotQA protocol on all 7405 dev samples after
    the 200-sample boundary showed that HotpotQA constrains the reusable-layer
    budget.
  - Test the maximum currently plausible route (`reuse08_patch24`) and a safer
    quality route (`reuse04_patch28`).
- Method:
  - Runtime:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Dataset/protocol: `hotpotqa_compressed`, `EVAL_LIMIT=0`, `MAX_LENGTH=512`,
    `MAX_NEW_TOKENS=32`.
  - Reuse bank:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422/ckpt_reuse/hotpotqa_seed42`.
  - Patch bank:
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422/ckpt_patch/patch_hotpotqa_seed42`.
- Launched runs:
  - `reuse08_patch24` on GPU pair `2,3`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse08_20260611_001`.
    The run reached live progress at `63/7405` shortly after launch.
  - `reuse04_patch28` on GPU pair `0,1`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse04_20260611_001`.
    This was scheduled with a lightweight waiter to start after the 200-sample
    mid-curve job writes `conditional route screen finished`.
- Status at entry time:
  - `reuse08_patch24` was running.
  - `reuse04_patch28` was queued behind the remaining 200-sample
    `reuse11_patch21` curve point.
  - No full7405 result should be treated as complete until
    `scd_icml_mistral_summary.json` and route metrics are present.
- Next actions:
  - Monitor both full7405 runs and collect F1/EM, PPL, TTFT, and route metrics.
  - If `reuse08_patch24` fails on full7405 while `reuse04_patch28` passes, use
    `reuse04_patch28` as the HotpotQA-safe route and report `reuse08_patch24`
    as the boundary failure.

## 2026-06-11 - Paper-table postprocessing helper added

- Change:
  - Added `scripts/build_scd_icml_result_tables.py` to the runtime release
    surface.
  - The script converts completed `patchonly_anchor_summary.json` and
    `route_metrics.json` artifacts into Markdown/CSV tables:
    `patch_anchor_table.*` and `mixed_route_boundary_table.*`.
- Purpose:
  - Make table regeneration depend on saved JSON artifacts instead of manual
    copying from terminal output.
  - Support the final paper-facing table pass after HotpotQA full7405 finishes.
- Local verification:
  - `python -m py_compile scripts\build_scd_icml_result_tables.py scripts\check_release_readiness.py scripts\build_release_tree.py`
    passed.
  - `python scripts\build_scd_icml_result_tables.py --out-dir tmp\result_tables_smoke`
    produced the expected empty smoke tables.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed
    (`19` tests).
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed, including release-preview readiness.
- Remote sync:
  - Synced the new script plus updated release check/build scripts and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Verified the new script exists under `runtime_repo/scripts/` and no
    accidental root-level copy remains.
- Real-artifact smoke:
  - Ran the script on completed CMRC2018, HotpotQA-compressed 200-sample, and
    WikiText-2 artifacts.
  - Output directory:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_001`.
  - Generated:
    `patch_anchor_table.csv`, `patch_anchor_table.md`,
    `mixed_route_boundary_table.csv`, `mixed_route_boundary_table.md`, and
    `scd_icml_result_tables_manifest.json`.
- Figure-output extension:
  - Extended the same script to emit per-dataset route-curve CSV/Markdown/SVG
    files from `route_metrics.json`.
  - Updated `scripts/collect_route_eval_metrics.py` so newly generated route
    metric rows include `direction` and `dataset` metadata.
  - Regenerated the partial real-artifact bundle at:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_002`.
  - New figure/data outputs include:
    `route_curve_cmrc2018.*`, `route_curve_hotpotqa_compressed.*`, and
    `route_curve_wikitext2.*`.
  - Copied this partial bundle into the local workspace at
    `Experiment/results/scd_icml_result_tables_partial_20260611_002` and added
    a local `README.md` marking it as a partial, pre-full7405 bundle.
  - Local `python scripts\check_release_readiness.py --root .` and
    `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed after the figure-output extension.

## 2026-06-11 - Full7405 postprocess waiter installed

- Experiment id: `hotpotqa_full7405_postprocess_20260611_001`
- Purpose:
  - Avoid manual result collection after the two long HotpotQA-compressed
    full7405 runs finish.
  - Automatically collect route metrics, run the HotpotQA full-dev selection
    gate, and regenerate the result-table/curve bundle with full7405 rows.
- Waiter script:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_and_postprocess.sh`.
- Waiter log:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/postprocess.log`.
- Trigger condition:
  - Wait until both full7405 summaries exist:
    - `scd_icml_hotpotqa_compressed_full7405_reuse04_20260611_001/.../conditional_reuse_reuse04_patch28/scd_icml_mistral_summary.json`.
    - `scd_icml_hotpotqa_compressed_full7405_reuse08_20260611_001/.../conditional_reuse_reuse08_patch24/scd_icml_mistral_summary.json`.
- Automatic outputs after trigger:
  - `full7405_route_metrics.json` and `.jsonl`.
  - `hotpotqa_full7405_f1_drop003_selection.json`.
  - `result_tables_with_full7405/` with Markdown/CSV tables and route-curve
    SVGs that include the full7405 HotpotQA rows.
- Status at entry time:
  - Waiter was running and polling every 300 seconds.
  - Full7405 `reuse04_patch28` was around `631/7405`.
  - Full7405 `reuse08_patch24` was around `845/7405`.
- Verification:
  - Local `python -m py_compile scripts\collect_route_eval_metrics.py scripts\build_scd_icml_result_tables.py`
    passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests`
    passed with `21` tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed.

## 2026-06-11 - Quality-gated route selection generalized for QA and WikiText-2

- Change:
  - Extended `scripts/select_quality_gated_scd_route.py` beyond F1/EM
    retention:
    - `--metric ppl` now selects lower-is-better WikiText-2 routes using a hard
      PPL relative-increase gate.
    - `--max-metric-drop` now supports absolute F1/EM loss gates for QA
      datasets with low Full Consumer baselines.
- Verification:
  - Added `tests/test_quality_gated_route_script.py`.
  - Local `python -m py_compile ...` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed
    with `21` tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed.
  - Synced the selector, test, and updated docs to the active `ser` runtime.
- Real-artifact selection reports:
  - CMRC2018 F1 retention `>=0.95` selected `reuse19_patch13`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_snapshot_bank_prefix_curve_cmrc200_20260611_001/lite_to_7b/conditional_report/cmrc_f1_retention095_selection.json`.
  - HotpotQA-compressed absolute F1 drop `<=0.03` selected `reuse08_patch24`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001/lite_to_7b/conditional_report/hotpotqa_f1_drop003_selection.json`.
  - WikiText-2 PPL relative increase `<=0.10` selected `reuse20_patch12`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001/lite_to_7b/conditional_report/wikitext2_ppl_gate_selection.json`.
- Analysis:
  - The current evidence supports dataset-specific maximum-Reuse routes:
    CMRC2018 `reuse19`, HotpotQA-compressed `reuse08`, and WikiText-2
    `reuse20`.
  - The common-route choice remains unresolved until HotpotQA full7405 confirms
    whether `reuse08_patch24` holds on the full dev protocol.

## 2026-06-11 - HotpotQA-compressed 200-sample mid-curve completed

- Experiment id: `boundary_hotpotqa_compressed_mid200_20260611_complete`
- Run root:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_compressed_mid200_20260611_001`.
- Results:
  - Full Consumer reference: F1/EM `0.330552/0.250000`.
  - `reuse09_patch23`: F1/EM `0.249338/0.190000`, F1 retention `0.7543`,
    PPL `7.816221`, TTFT `40.44 ms`.
  - `reuse10_patch22`: F1/EM `0.250413/0.200000`, F1 retention `0.7576`,
    PPL `7.952435`, TTFT `40.11 ms`.
  - `reuse11_patch21`: F1/EM `0.242163/0.190000`, F1 retention `0.7326`,
    PPL `7.946812`, TTFT `39.77 ms`.
- Analysis:
  - These mid-curve points confirm that the HotpotQA-compressed quality cliff
    begins before `reuse09_patch23`.
  - The only currently plausible high-Reuse HotpotQA candidate is
    `reuse08_patch24`; `reuse04_patch28` remains the safer quality fallback.
- Follow-up:
  - The full7405 `reuse08_patch24` and `reuse04_patch28` runs are now the
    authoritative HotpotQA validation jobs.

## 2026-06-11 - Local reproduction evidence index added

- Artifact:
  - `Experiment/results/README.md`.
- Purpose:
  - Record the current local result bundle, the completed remote selector
    reports, and the still-pending HotpotQA-compressed full7405 evidence in one
    place.
- Current local bundle:
  - `Experiment/results/scd_icml_result_tables_partial_20260611_002/`.
- Full7405 status at latest check:
  - `reuse04_patch28`: running, approximately `868/7405`.
  - `reuse08_patch24`: running, approximately `1071/7405`.
  - Waiter process:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_and_postprocess.sh`.
- Analysis:
  - The complete main-result reproduction is still unproven because the
    full7405 HotpotQA F1/EM rows are missing.
  - CMRC2018 and WikiText-2 route evidence is already complete enough for the
    current non-Qwen scope; HotpotQA remains the gating dataset.

## 2026-06-11 - Paper/result artifact coverage audit

- Artifact:
  - `docs/scd-icml-reproduction-coverage-audit-2026-06-11.md`.
- Findings:
  - The current paper source still contains stale CMRC boundary values and old
    full-context HotpotQA diagnostic wording.
  - The current paper-comparable HotpotQA path is `hotpotqa_compressed`,
    `MAX_LENGTH=512`, full validation size `7405`, not the old full-context
    diagnostic table.
  - Generated route-curve SVGs now cover CMRC2018, HotpotQA-compressed
    200-sample, and WikiText-2 partial boundary curves.
  - Static non-conceptual figures `patch_error1.pdf`, `layer_selection.pdf`,
    and `stack_viz.pdf` do not yet have an identified reproducible generator in
    the open-source runtime.
- Consequence:
  - Completion requires both the full7405 HotpotQA rows and a paper/release
    sync pass, not just waiting for the eval jobs.

## 2026-06-11 - Runtime release docs updated for HotpotQA protocol boundary

- Files updated:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Change:
  - Added explicit notes that full-context HotpotQA rows are diagnostics.
  - Marked the current paper-facing HotpotQA check as
    `hotpotqa_compressed`, `MAX_LENGTH=512`, full7405.
  - Kept completion open until full7405 selected-route rows are available and
    tables are regenerated.
- Verification:
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed, including readiness inside the preview tree.
  - Synced the three docs to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/`.
  - Remote `python scripts/check_release_readiness.py --root .` returned exit
    code `0`.
- Full7405 status after sync:
  - `reuse04_patch28`: approximately `990/7405`.
  - `reuse08_patch24`: approximately `1192/7405`.

## 2026-06-11 - Profile-based analysis figure generator added

- Code added:
  - `Experiment/core_code/runtime_repo/scripts/build_scd_icml_analysis_figures.py`.
  - `Experiment/core_code/runtime_repo/tests/test_analysis_figures_script.py`.
- Purpose:
  - Generate reproducible layer-profile CSV/Markdown/SVG figures from
    `profile_cmrc2018_seed42.json`.
  - Covers Reuse-vs-Patch error, expected Patch gain by layer, and cumulative
    Patch gain curves.
  - Does not generate t-SNE alignment figures because profile JSON does not
    contain representation samples.
- Local generated bundle:
  - `Experiment/results/profile_analysis_figures_20260611_001/`.
  - Source profile:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/route/profile_cmrc2018_seed42.json`.
- Release integration:
  - Added the script to release manifest, release tree builder, readiness check,
    README, and reproduction guide.
  - Synced script/test/docs to the active `ser` runtime.
- Verification:
  - Local `python -m py_compile ...` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests` passed
    with `22` tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed.
  - Remote `PYTHONPATH=src python -m unittest discover -s tests` passed with
    `22` tests.
  - Remote `python scripts/check_release_readiness.py --root .` returned exit
    code `0`.
- Full7405 status after this update:
  - `reuse04_patch28`: approximately `1261/7405`.
  - `reuse08_patch24`: approximately `1448/7405`.

## 2026-06-11 - Archived t-SNE alignment evidence regenerated

- Artifact:
  - `Experiment/results/tsne_alignment_evidence_20260611_001/`.
- Source:
  - Copied real archived Figure 8 inputs from
    `archive/icml-completed-2026-05-24/camera-ready-and-final/final/evidence/figure8-inputs/`.
  - Confirmed corresponding remote snapshot t-SNE CSV at
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/paper_missing_mistral_forward_20260426/analysis/cmrc2018_tsne_kv_alignment.csv`.
- Files:
  - Inputs: `cmrc2018_kv_distance_distribution.csv`,
    `cmrc2018_kv_distance_distribution_meta.json`,
    `cmrc2018_tsne_kv_alignment.csv`,
    `cmrc2018_tsne_kv_alignment_meta.json`.
  - Script: `plot_stack_viz.py`.
  - Outputs: `stack_viz.pdf`, `stack_viz.png`, `stack_viz.svg`.
- Verification:
  - `python plot_stack_viz.py` completed and generated all three outputs.
  - Visual inspection of `stack_viz.png` confirmed a nonblank, readable
    distance-distribution plus t-SNE figure.
  - `python -m py_compile Experiment\results\tsne_alignment_evidence_20260611_001\plot_stack_viz.py`
    passed.
- Analysis:
  - This covers the current t-SNE/stack visualization from archived real input
    data, but it is not a fresh current-runtime rerun.
  - If the final release requires current-runtime-only provenance, fresh KV
    representation extraction is still needed.

## 2026-06-11 - Offline calibration/training cost evidence generated

- Code added:
  - `Experiment/core_code/runtime_repo/scripts/build_offline_cost_table.py`.
  - `Experiment/core_code/runtime_repo/tests/test_offline_cost_table_script.py`.
- Artifact:
  - `Experiment/results/offline_cost_evidence_20260611_001/`.
- Inputs:
  - Raw `/usr/bin/time -v` logs copied from:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851/logs/`.
  - Raw `/usr/bin/time -v` logs copied from:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/logs/`.
- Generated outputs:
  - `offline_cost_table.csv`.
  - `offline_cost_table.md`.
  - `offline_cost_summary.json`.
  - `offline_cost_wall_time.svg`.
  - `raw_time_logs/*.time.txt`.
- Measured rows:
  - Paper mixed capture pair: `12:36`, peak RSS `14.492 GiB`.
  - Paper mixed profile layers: `28:30`, peak RSS `2.279 GiB`.
  - Paper mixed train reuse: `12:24`, peak RSS `2.156 GiB`.
  - Clean reuse16 capture pair: `13:20`, peak RSS `14.494 GiB`.
  - Clean reuse16 profile layers: `28:40`, peak RSS `2.309 GiB`.
  - Clean reuse16 train reuse: `10:26`, peak RSS `2.289 GiB`.
  - Clean reuse16 train patch patch16: `39:51`, peak RSS `41.161 GiB`.
- Limitation:
  - `paper_mixed_train_patch.time.txt` exists but is empty, so the generated
    table preserves it as `status=no_time_fields`; it must not be cited as a
    measured Patch-training cost.
- Verification:
  - `python -m py_compile scripts\build_offline_cost_table.py tests\test_offline_cost_table_script.py` passed.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v` passed
    with `25` tests.

## 2026-06-11 - Release-facing stack/t-SNE plot generator added

- Code added:
  - `Experiment/core_code/runtime_repo/scripts/build_stack_viz.py`.
  - `Experiment/core_code/runtime_repo/tests/test_stack_viz_script.py`.
- Artifact regenerated:
  - `Experiment/results/tsne_alignment_evidence_20260611_001/stack_viz.pdf`.
  - `Experiment/results/tsne_alignment_evidence_20260611_001/stack_viz.png`.
  - `Experiment/results/tsne_alignment_evidence_20260611_001/stack_viz.svg`.
- Inputs:
  - `Experiment/results/tsne_alignment_evidence_20260611_001/cmrc2018_kv_distance_distribution.csv`.
  - `Experiment/results/tsne_alignment_evidence_20260611_001/cmrc2018_kv_distance_distribution_meta.json`.
  - `Experiment/results/tsne_alignment_evidence_20260611_001/cmrc2018_tsne_kv_alignment.csv`.
  - `Experiment/results/tsne_alignment_evidence_20260611_001/cmrc2018_tsne_kv_alignment_meta.json`.
- Verification:
  - `$env:PYTHONPATH='src'; python -m unittest tests.test_stack_viz_script -v` passed.
  - `python -m py_compile scripts\build_stack_viz.py tests\test_stack_viz_script.py scripts\check_release_readiness.py scripts\build_release_tree.py` passed.
  - The real archived CSV/meta inputs regenerated non-empty PDF/PNG/SVG outputs
    through the new runtime script.
- Limitation:
  - This adds a release-facing plotter for stored artifacts; it does not
    perform a fresh current-runtime representation extraction.

## 2026-06-11 - Fresh current-runtime stack/t-SNE extraction completed

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/extract_stack_viz_inputs.py`.
  - `Experiment/core_code/runtime_repo/tests/test_stack_viz_input_extractor.py`.
  - Added optional `analysis` dependency group with `scikit-learn>=1.4.0` in
    `pyproject.toml`.
- Remote source run:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/`.
- Remote extraction output:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_stack_viz_extraction_20260611_001/analysis/`.
- Local artifact:
  - `Experiment/results/stack_viz_current_extraction_20260611_001/`.
- Method:
  - Extracted sampled source/target/translated KV vectors from
    `pairs/cmrc2018_train_seed42_n200`, `ckpt_reuse/cmrc2018_seed42`, and
    `route/profile_cmrc2018_seed42.json`.
  - Saved `cmrc2018_stack_viz_vectors.npz` on the server and copied it locally.
  - Re-ran `extract_stack_viz_inputs.py --vectors-npz` locally where
    `scikit-learn 1.8.0` is available, producing true t-SNE CSV/meta.
  - Rendered `stack_viz_current_tsne.pdf/png/svg` with `build_stack_viz.py`.
- Result:
  - Layer `16`.
  - Points per series `160`.
  - Reducer `tsne`.
  - Perplexity `50`.
  - t-SNE centroid gap ratio `0.044567`.
- Verification:
  - `$env:PYTHONPATH='src'; python -m unittest tests.test_stack_viz_input_extractor -v` passed with `5` tests.
  - `python -m py_compile scripts\extract_stack_viz_inputs.py tests\test_stack_viz_input_extractor.py` passed.
- Analysis:
  - `stack_viz` now has current-runtime extraction evidence, not just archived
    CSV provenance.

## 2026-06-11 - Result-table LaTeX output and release checks verified

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/build_scd_icml_result_tables.py`.
  - `Experiment/core_code/runtime_repo/tests/test_result_tables_script.py`.
  - `Experiment/core_code/runtime_repo/scripts/check_release_readiness.py`.
- Documentation updated:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Purpose:
  - Make paper-facing result-table regeneration produce LaTeX as well as
    CSV/Markdown.
  - Remove stale wording that implied full-context HotpotQA was the final
    paper-facing protocol.
- Verification:
  - Remote regenerated
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_003`
    from the completed patch-anchor and route-metric JSON artifacts.
  - Local copy added at
    `Experiment/results/scd_icml_result_tables_partial_20260611_003/`,
    including `patch_anchor_table.tex` and
    `mixed_route_boundary_table.tex`.
  - Local `python -m py_compile scripts\build_scd_icml_result_tables.py
    tests\test_result_tables_script.py` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest
    tests.test_result_tables_script -v` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 32 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Remote `PYTHONPATH=src python -m unittest tests.test_result_tables_script
    -v` passed.
  - Remote full unittest log `/tmp/scd_icml_full_unittest_latest.log` reports
    32 tests passed in 2.967s.
  - Remote release-tree log `/tmp/scd_icml_release_tree_latest.log` reports
    release readiness passed and produced
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release/RELEASE_BUILD_REPORT.md`.
- HotpotQA full7405 status at this check:
  - `reuse04_patch28`: still running, last observed around 2349/7405.
  - `reuse08_patch24`: still running, last observed around 2487/7405.
- Conclusion:
  - The release-facing code path for table generation is stronger, but the
    overall reproduction goal remains open until HotpotQA full7405 and final
    paper/table synchronization finish.

## 2026-06-11 - Current-safe paper synchronized and compiled

- File updated:
  - `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`.
- Source evidence:
  - `Experiment/results/scd_icml_result_tables_partial_20260611_003/`.
- Changes:
  - Updated CMRC selected-route values to Full F1/EM `0.657307/0.365` and SCD
    Reuse19/Patch13 F1/EM `0.636821/0.340`.
  - Updated CMRC boundary rows through Reuse24/Patch08 from the current route
    metrics.
  - Replaced stale non-CMRC diagnostics with HotpotQA-compressed 200-sample
    screen rows and WikiText-2 Reuse20/Patch12 boundary rows.
  - Clarified that HotpotQA-compressed full7405 is still pending and
    full-context HotpotQA is only a diagnostic protocol.
- Build verification:
  - `latexmk` could not be used because MiKTeX reported missing Perl.
  - `pdflatex -interaction=nonstopmode -halt-on-error
    -output-directory=current_safe_build
    scd-icml-current-safe-paper-2026-06-10.tex` passed.
  - `bibtex scd-icml-current-safe-paper-2026-06-10` passed from
    `Publication/paper/current_safe_build`.
  - Final `pdflatex` pass produced
    `Publication/paper/current_safe_build/scd-icml-current-safe-paper-2026-06-10.pdf`
    at 899980 bytes.
  - Log check found no overfull boxes and no undefined citation/reference
    warnings; remaining warnings were routine hyperref notices.
- Conclusion:
  - The current-safe paper is now aligned with the current partial artifact
    bundle, but final HotpotQA full7405 rows are still required before claiming
    complete non-Qwen paper reproduction.

## 2026-06-11 - Paper artifact bundle generator and current bundle

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/build_paper_artifact_bundle.py`.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_bundle_script.py`.
  - `Experiment/core_code/runtime_repo/scripts/check_release_readiness.py`.
  - `Experiment/core_code/runtime_repo/scripts/build_release_tree.py`.
- Documentation updated:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Purpose:
  - Assemble generated result tables, profile figures, stack/t-SNE outputs, and
    offline-cost artifacts into one paper-facing artifact bundle without
    recomputing or inventing metrics.
- Artifact generated:
  - `Experiment/results/paper_artifact_bundle_current_20260611_001/`.
  - `paper_artifact_manifest.json` reports `status=ok`, `copied=29`,
    `missing=0`, and `required_missing=0`.
- Inputs:
  - `Experiment/results/scd_icml_result_tables_partial_20260611_003/`.
  - `Experiment/results/profile_analysis_figures_20260611_001/`.
  - `Experiment/results/stack_viz_current_extraction_20260611_001/`.
  - `Experiment/results/offline_cost_evidence_20260611_001/`.
- Verification:
  - `python -m py_compile scripts\build_paper_artifact_bundle.py
    tests\test_paper_artifact_bundle_script.py` passed.
  - `$env:PYTHONPATH='src'; python -m unittest
    tests.test_paper_artifact_bundle_script -v` passed.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v` passed
    with 34 tests.
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced the changed runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `PYTHONPATH=src /disk_n/c101-3-mql/venvs/A800/bin/python -m unittest
    discover -s tests -v` passed with 34 tests; latest log is
    `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote `scripts/build_release_tree.py --out
    release_preview/scd_icml_runtime_release --force` passed and produced
    `RELEASE_BUILD_REPORT.md`.
- Conclusion:
  - Current non-Qwen paper artifacts now have a single assembled local bundle,
    but the main result remains incomplete until HotpotQA-compressed full7405 is
    available and the bundle is regenerated.

## 2026-06-11 - HotpotQA-compressed full7405 progress refresh

- Remote jobs:
  - `scd_icml_hotpotqa_compressed_full7405_reuse04_20260611_001`.
  - `scd_icml_hotpotqa_compressed_full7405_reuse08_20260611_001`.
  - Postprocess waiter:
    `scd_icml_full7405_postprocess_20260611_001/wait_and_postprocess.sh`.
- Latest observed progress:
  - `reuse04_patch28`: `2954/7405` at `2026-06-11 06:04:01`.
  - `reuse08_patch24`: `3056/7405` at `2026-06-11 06:04:00`.
  - Waiter log at `2026-06-11T06:03:22+08:00` still reports
    `summary04=no summary08=no`.
- GPU/process state:
  - Eval processes remained alive for both routes.
  - GPU query reported memory/utilization rows:
    `0,14577,10`; `1,15479,82`; `2,14839,0`; `3,15789,79`.
- Conclusion:
  - Main HotpotQA full7405 evidence is still pending; do not close the full
    non-Qwen reproduction objective yet.

## 2026-06-11 - HotpotQA-compressed full7405 progress refresh 2

- Latest observed progress:
  - `reuse04_patch28`: `3153/7405` at `2026-06-11 06:12:52`.
  - `reuse08_patch24`: `3246/7405` at `2026-06-11 06:12:53`.
  - Waiter log at `2026-06-11T06:08:22+08:00` still reports
    `summary04=no summary08=no`.
- GPU state:
  - `0,14839,7`; `1,15787,83`; `2,14569,0`; `3,15451,83`.
- Conclusion:
  - Both long evaluations remain alive; no full7405 summary is available yet.

## 2026-06-11 - HotpotQA-compressed full7405 progress refresh 3

- Latest observed progress:
  - `reuse04_patch28`: `3399/7405` at `2026-06-11 06:23:42`.
  - `reuse08_patch24`: `3481/7405` at `2026-06-11 06:23:41`.
  - Waiter log at `2026-06-11T06:23:22+08:00` still reports
    `summary04=no summary08=no`.
- Summary files:
  - `summary04=no`.
  - `summary08=no`.
  - `full7405_route_metrics.json=no`.
- Conclusion:
  - Both long evaluations remain alive; no full7405 summary is available yet.

## 2026-06-11 - TTFT and latency-quality paper artifacts

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/build_scd_icml_result_tables.py`.
  - `Experiment/core_code/runtime_repo/scripts/build_paper_artifact_bundle.py`.
  - `Experiment/core_code/runtime_repo/tests/test_result_tables_script.py`.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_bundle_script.py`.
- Purpose:
  - Cover remaining latency/TTFT/Pareto-style paper artifacts using completed
    patch-anchor and mixed-route summary JSON files.
- Remote artifact generated:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_004`.
- Local artifacts generated:
  - `Experiment/results/scd_icml_result_tables_partial_20260611_004/`.
  - `Experiment/results/paper_artifact_bundle_current_20260611_002/`.
- New outputs:
  - `ttft_breakdown_table.csv/md/tex`.
  - `latency_quality_pareto.csv/md`.
  - `latency_quality_pareto_cmrc2018.svg`.
  - `latency_quality_pareto_hotpotqa_compressed.svg`.
  - `latency_quality_pareto_wikitext2.svg`.
- Verification:
  - `python -m py_compile scripts\build_scd_icml_result_tables.py
    scripts\build_paper_artifact_bundle.py tests\test_result_tables_script.py
    tests\test_paper_artifact_bundle_script.py` passed.
  - `$env:PYTHONPATH='src'; python -m unittest tests.test_result_tables_script
    tests.test_paper_artifact_bundle_script -v` passed.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v` passed
    with 34 tests.
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced updated runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted tests for result tables and artifact bundle passed.
  - Remote full unittest passed with 34 tests; latest log is
    `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview regenerated at `2026-06-11 06:11:38 +0800` and
    produced `release_preview/scd_icml_runtime_release/RELEASE_BUILD_REPORT.md`.
- Conclusion:
  - Current TTFT/latency/Pareto artifact coverage is stronger, but the outputs
    are still pre-full7405 for HotpotQA and must be regenerated after the long
    runs finish.

## 2026-06-11 - Generated figures wired into current-safe paper

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/build_scd_icml_analysis_figures.py`.
  - `Experiment/core_code/runtime_repo/scripts/build_paper_artifact_bundle.py`.
  - `Experiment/core_code/runtime_repo/tests/test_analysis_figures_script.py`.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_bundle_script.py`.
- Artifacts generated:
  - `Experiment/results/profile_analysis_figures_20260611_002/`.
  - `Experiment/results/paper_artifact_bundle_current_20260611_003/`.
  - `Publication/paper/generated_figures/`.
- Paper changes:
  - Replaced `patch_error1.pdf` with
    `generated_figures/layerwise_xnorm_replacement.pdf`.
  - Replaced the old `layer_selection.pdf` figure with generated profile-gain
    and patch-budget PDF panels.
  - Replaced `stack_viz.pdf` with `generated_figures/stack_viz.pdf`.
  - Replaced the unsupported `9.74 h` offline-cost total with measured rows
    from `offline_cost_evidence_20260611_001/offline_cost_table.csv`, including
    an explicit `no time fields` row for the paper-mixed Patch log.
- Verification:
  - `python -m py_compile scripts\build_scd_icml_analysis_figures.py
    scripts\build_paper_artifact_bundle.py tests\test_analysis_figures_script.py
    tests\test_paper_artifact_bundle_script.py` passed.
  - `$env:PYTHONPATH='src'; python -m unittest tests.test_analysis_figures_script
    tests.test_paper_artifact_bundle_script -v` passed.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v` passed
    with 34 tests.
  - `python scripts\check_release_readiness.py --root .` passed.
  - `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced updated runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted tests for analysis figures and artifact bundle passed.
  - Remote full unittest passed with 34 tests; latest log is
    `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview regenerated at `2026-06-11 06:22:23 +0800` and
    produced `release_preview/scd_icml_runtime_release/RELEASE_BUILD_REPORT.md`.
  - `pdflatex` built
    `Publication/paper/current_safe_build/scd-icml-current-safe-paper-2026-06-10.pdf`
    at 883298 bytes.
  - The final LaTeX log check found no missing figures, LaTeX errors,
    overfull boxes, or undefined citation/reference warnings.
- Conclusion:
  - The current-safe paper now uses generated figure/cost artifacts for the
    non-conceptual figures and cost table that are available pre-full7405.
    Full completion still waits on HotpotQA-compressed full7405.

## 2026-06-11 - Paper artifact guard and full7405 progress refresh 4

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/check_paper_artifacts.py`.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_check_script.py`.
  - Release/readiness documentation and release-tree scripts now include the
    paper artifact guard.
- Artifact fixes:
  - Copied `Publication/paper/current_safe_build/fig1.pdf` to
    `Publication/paper/fig1.pdf`.
  - Copied `Publication/paper/current_safe_build/fig2.pdf` to
    `Publication/paper/fig2.pdf`.
- Paper artifact check:
  - Command: `python scripts\check_paper_artifacts.py --paper-tex
    ..\..\..\Publication\paper\scd-icml-current-safe-paper-2026-06-10.tex
    --base-dir ..\..\..\Publication\paper --bundle-dir
    ..\..\results\paper_artifact_bundle_current_20260611_003 ...`.
  - Result: passed.
  - Report:
    `Experiment/results/paper_artifact_bundle_current_20260611_003/paper_artifact_check_report.json`.
- Verification:
  - Local targeted paper artifact guard tests passed.
  - Local full unittest passed with 36 tests.
  - Local release readiness and release preview passed.
  - Remote targeted paper artifact guard tests passed.
  - Remote full unittest passed with 36 tests; `/tmp/scd_icml_full_unittest_latest.log`
    reports `Ran 36 tests in 3.344s` and `OK`.
  - Remote release preview report exists and includes
    `scripts/check_paper_artifacts.py`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked at `2026-06-11 06:32:11+0800` on `ser`.
  - `summary04=no`.
  - `summary08=no`.
  - `reuse04_patch28`: latest progress `3591/7405` at
    `2026-06-11 06:32:09`.
  - `reuse08_patch24`: latest progress `3663/7405` at
    `2026-06-11 06:32:08`.
- Conclusion:
  - The open-source runtime and current paper artifact guards are in good
    shape. The main reproduction claim remains open until HotpotQA-compressed
    full7405 final F1/EM is available and the paper-facing tables are
    regenerated.

## 2026-06-11 - Claim-boundary text guard and full7405 progress refresh 5

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/check_paper_artifacts.py` now
    supports `--require-text` and `--forbid-text` in addition to figure and
    bundle checks.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_check_script.py`
    now covers the text-boundary checks.
  - `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` document the text-boundary guard.
- Purpose:
  - Turn the current safe-paper treatment of neighboring compression baselines
    into a machine-checked release boundary. The paper must explicitly state
    that earlier adapted neighboring baseline rows are not part of the current
    verified release evidence and are not used as a claim-facing comparison.
- Paper artifact check:
  - Passed against
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`.
  - Updated report:
    `Experiment/results/paper_artifact_bundle_current_20260611_003/paper_artifact_check_report.json`.
  - The report has `missing_images=[]`, `bundle_status=ok`, required text
    entries present, and `errors=[]`.
- Verification:
  - Local targeted paper artifact guard tests passed with 3 tests.
  - Local full unittest passed with 37 tests.
  - Local release readiness and release preview passed.
  - Remote targeted paper artifact guard tests passed with 3 tests.
  - Remote full unittest passed with 37 tests; `/tmp/scd_icml_full_unittest_latest.log`
    reports `Ran 37 tests in 3.541s` and `OK`.
  - Remote release preview report exists and includes
    `scripts/check_paper_artifacts.py`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked at `2026-06-11 06:39:25+0800` on `ser`.
  - `summary04=no`.
  - `summary08=no`.
  - `reuse04_patch28`: latest progress `3751/7405` at
    `2026-06-11 06:39:23`.
  - `reuse08_patch24`: latest progress `3818/7405` at
    `2026-06-11 06:39:24`.
- Conclusion:
  - The unsupported neighboring-baseline table is now excluded by a checked
    claim boundary rather than left as an ambiguous unsupported artifact.
    Full completion still waits on HotpotQA-compressed full7405 F1/EM.

## 2026-06-11 - Patch-budget table included in paper artifact bundle

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/build_paper_artifact_bundle.py`
    now copies `patch_gain_curve.md` into `tables/patch_gain_curve.md`.
  - `Experiment/core_code/runtime_repo/tests/test_paper_artifact_bundle_script.py`
    now creates and checks the Markdown patch-gain table.
- Artifact generated:
  - `Experiment/results/paper_artifact_bundle_current_20260611_004/`.
- Result:
  - `paper_artifact_manifest.json`: `status=ok`, `copied=44`, `missing=0`.
  - `tables/patch_gain_curve.md` is present.
  - `paper_artifact_check_report.json` passed with `errors=[]`.
- Verification:
  - Targeted bundle tests passed.
  - The paper artifact check against
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` passed with
    `paper_artifact_bundle_current_20260611_004`.
  - Local full unittest passed with 37 tests.
  - Local release readiness and release preview passed.
  - Remote targeted bundle tests passed.
  - Remote full unittest passed with 37 tests; `/tmp/scd_icml_full_unittest_latest.log`
    reports `Ran 37 tests in 3.983s` and `OK`.
  - Remote release preview passed and its report includes
    `scripts/build_paper_artifact_bundle.py` and
    `scripts/check_paper_artifacts.py`.
  - `pdflatex -interaction=nonstopmode -halt-on-error
    -output-directory=current_safe_build
    scd-icml-current-safe-paper-2026-06-10.tex` passed after updating the
    paper evidence root to bundle `004`; the output PDF is 15 pages and
    883161 bytes.
  - The final LaTeX log grep found no missing figures, LaTeX errors, overfull
    boxes, or undefined citation/reference warnings.
- HotpotQA-compressed full7405 latest refresh:
  - Checked at `2026-06-11 06:47:00+0800` on `ser`.
  - `summary04=no` and `summary08=no`.
  - `reuse04_patch28`: latest progress `3918/7405` at
    `2026-06-11 06:46:58`.
  - `reuse08_patch24`: latest progress `3986/7405` at
    `2026-06-11 06:47:11`.
- Conclusion:
  - The patch-budget / patch-ablation evidence is now included as both readable
    table and generated figure in the current paper artifact bundle. It remains
    a current replacement, not a claim that the exact historical `k=6` table has
    been rerun.

## 2026-06-11 - Full7405 finalizer wrapper added

- Code added:
  - `Experiment/core_code/runtime_repo/scripts/finalize_scd_icml_full7405_bundle.py`.
  - `Experiment/core_code/runtime_repo/tests/test_finalize_full7405_bundle_script.py`.
- Purpose:
  - Once the remote waiter writes `result_tables_with_full7405/`, one command
    can assemble the final paper artifact bundle and run the current paper
    artifact gate against the paper source.
  - The wrapper calls existing artifact scripts only; it does not compute new
    metrics, invent missing rows, or edit the paper source.
- Release integration:
  - Added the script and test to `check_release_readiness.py`.
  - Added the script to `build_release_tree.py`.
  - Documented the command in `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Verification:
  - Local targeted finalizer test passed.
  - Local full unittest passed with 38 tests.
  - Local release readiness and release preview passed.
  - Synced finalizer files and docs to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted finalizer test passed.
  - Remote full unittest passed with 38 tests; `/tmp/scd_icml_full_unittest_latest.log`
    reports `Ran 38 tests in 4.670s` and `OK`.
  - Remote release preview passed; its report includes
    `scripts/finalize_scd_icml_full7405_bundle.py`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked at `2026-06-11 06:56:14+0800` on `ser`.
  - `summary04=no` and `summary08=no`.
  - `reuse04_patch28`: latest progress `4130/7405` at
    `2026-06-11 06:56:11`.
  - `reuse08_patch24`: latest progress `4190/7405` at
    `2026-06-11 06:56:28`.
- Conclusion:
  - The final post-full7405 packaging path is now scripted and release-tested.
    Completion still waits on the two full7405 summaries and final table/bundle
    regeneration.

## 2026-06-11 - Full7405 finalizer now rejects pre-full7405 tables

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/finalize_scd_icml_full7405_bundle.py`.
  - `Experiment/core_code/runtime_repo/tests/test_finalize_full7405_bundle_script.py`.
- Purpose:
  - Prevent the finalizer from accidentally packaging the current partial
    result table as if it were post-full7405 evidence.
- Implementation:
  - The finalizer now checks `mixed_route_boundary_table.csv` before bundling.
  - By default, at least one `source` field must contain `full7405`.
  - The requirement can be overridden or repeated with
    `--require-result-source-substring`, but the default final path is
    protected.
- Verification:
  - Local targeted finalizer tests passed with 2 tests.
  - Local full unittest passed with 39 tests.
  - Local release readiness and release preview passed.
  - Remote targeted finalizer tests passed with 2 tests.
  - Remote full unittest passed with 39 tests; `/tmp/scd_icml_full_unittest_latest.log`
    reports `Ran 39 tests in 3.869s` and `OK`.
  - Remote release preview passed and includes
    `scripts/finalize_scd_icml_full7405_bundle.py`.
  - Negative smoke on `scd_icml_result_tables_partial_20260611_004` returned
    `rc=1` and reported `mixed route table does not contain required source
    substring(s): full7405`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked at `2026-06-11 06:59:23+0800` on `ser`.
  - `summary04=no` and `summary08=no`.
  - `reuse04_patch28`: latest progress `4202/7405` at
    `2026-06-11 06:59:22`.
  - `reuse08_patch24`: latest progress `4252/7405` at
    `2026-06-11 06:59:22`.
- Conclusion:
  - The release path now has a guard against premature final packaging. The
    only remaining main-result blocker is still the full7405 summaries and
    final table/bundle regeneration.

## 2026-06-11 - Full7405 wait-finalize wrapper synced and running

- Code/documentation updated:
  - Added `Experiment/core_code/runtime_repo/scripts/wait_finalize_scd_icml_full7405_bundle.sh`.
  - Added the wait wrapper to `scripts/build_release_tree.py`,
    `scripts/check_release_readiness.py`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - Documented the post-full7405 wait/finalize command in `README.md` and
    `REPRODUCING_SCD_ICML.md`.
  - Updated the docs to invoke the wrapper with `bash
    scripts/wait_finalize_scd_icml_full7405_bundle.sh`, because files copied by
    `scp` or zip extraction may not preserve executable bits.
- Purpose:
  - Keep the final paper-bundle step automatic while the two HotpotQA-compressed
    full7405 evaluations finish.
  - The wrapper waits for
    `result_tables_with_full7405/mixed_route_boundary_table.csv` to exist and
    contain `full7405`, then invokes
    `scripts/finalize_scd_icml_full7405_bundle.py`.
- Local verification:
  - `python -m py_compile scripts\finalize_scd_icml_full7405_bundle.py
    scripts\check_release_readiness.py scripts\build_release_tree.py` passed.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed with
    `Ran 39 tests in 5.634s` and `OK`.
  - `python scripts\check_release_readiness.py` passed.
  - `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed, and the release
    report includes `scripts/wait_finalize_scd_icml_full7405_bundle.sh`.
  - The local wait wrapper has LF line endings (`crlf=0`).
- Remote sync and verification:
  - Synced the wait wrapper, finalizer, release scripts, finalizer test, and
    docs to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `bash -n scripts/wait_finalize_scd_icml_full7405_bundle.sh` and
    py_compile returned `syntax_ok`.
  - Remote `PYTHONPATH=src ... -m unittest discover -s tests -v` passed with
    `Ran 39 tests in 3.640s` and `OK`.
  - Remote release preview report includes
    `scripts/wait_finalize_scd_icml_full7405_bundle.sh`, and the copied README
    / reproducing guide contain the `bash scripts/...` invocation.
- Remote watcher:
  - First direct launch failed with `Permission denied` because the copied shell
    script was not executable.
  - Relaunched with `bash scripts/wait_finalize_scd_icml_full7405_bundle.sh`.
  - Active wrapper process observed on `ser`: PID `3296816`, with sleep child
    PID `3296823`.
  - Watcher log:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_finalize_bundle.log`.
  - Latest watcher state at `2026-06-11T07:22:23+08:00`:
    `waiting ... table_ready=no`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 07:26 +0800`.
  - `reuse04_patch28`: latest progress `4805/7405` at
    `2026-06-11 07:26:34`.
  - `reuse08_patch24`: latest progress `4830/7405` at
    `2026-06-11 07:26:34`.
  - Both eval processes were still alive:
    PID `3169031` for `reuse04_patch28`, PID `3160974` for `reuse08_patch24`.
  - No full7405 summaries or final metrics files existed yet.
- Conclusion:
  - The final packaging path is now both guarded and waiting automatically.
    The main reproduction claim remains open until the full7405 summaries,
    route selection, and final paper artifact bundle are produced.

## 2026-06-11 - Full7405 postprocess wrapper added and launched

- Code/documentation updated:
  - Added
    `Experiment/core_code/runtime_repo/scripts/wait_postprocess_scd_icml_full7405_tables.sh`.
  - Added the script to `scripts/build_release_tree.py`,
    `scripts/check_release_readiness.py`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - Documented it in `README.md` and `REPRODUCING_SCD_ICML.md` before the final
    artifact-bundle step.
- Purpose:
  - Move the ad hoc remote `wait_and_postprocess.sh` behavior into the open
    runtime release.
  - The wrapper waits for each full7405 eval root to contain
    `scd_icml_mistral_summary.json`, then writes:
    - `full7405_route_metrics.json`
    - `full7405_route_metrics.jsonl`
    - `hotpotqa_full7405_f1_drop003_selection.json`
    - `result_tables_with_full7405/`
- Implementation note:
  - The first version used space-separated env lists, which was fragile in
    remote `env ... nohup ...` commands.
  - The script now uses colon-separated lists for `EVAL_ROOTS`,
    `PATCH_ANCHOR_JSONS`, and `BASE_ROUTE_METRICS_JSONS`; `LIST_SEPARATOR` can
    override the delimiter.
- Verification:
  - Local `PYTHONPATH=src python -m unittest discover -s tests -v` passed with
    `Ran 39 tests in 5.698s` and `OK`.
  - Local release readiness and release preview passed.
  - Remote `bash -n scripts/wait_postprocess_scd_icml_full7405_tables.sh`
    passed.
  - Remote release preview was rebuilt and contains the colon-separated README
    command plus `scripts/wait_postprocess_scd_icml_full7405_tables.sh`.
- Remote watcher state:
  - Replaced the old ad hoc postprocess waiter PID `3185445`.
  - Launched the release-script watcher under
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_postprocess_tables.log`.
  - Latest log line at `2026-06-11T07:37:02+08:00`:
    `waiting ... summaries_ready=0/2`.
  - The final bundle watcher remains active in
    `wait_finalize_bundle.log` and is waiting with `table_ready=no`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 07:38 +0800`.
  - `reuse04_patch28`: latest progress `5091/7405` at
    `2026-06-11 07:38:57`.
  - `reuse08_patch24`: latest progress `5102/7405` at
    `2026-06-11 07:38:56`.
  - `full7405_route_metrics.json`,
    `hotpotqa_full7405_f1_drop003_selection.json`, and
    `result_tables_with_full7405/mixed_route_boundary_table.csv` do not exist
    yet.
- Conclusion:
  - The postprocess and final-bundle stages are now both represented by release
    scripts and active remote watchers. The reproduction remains incomplete
    until the two full7405 evals finish and the generated HotpotQA F1/EM row is
    inspected.

## 2026-06-11 - Release readiness now validates shell wrappers

- Code updated:
  - `Experiment/core_code/runtime_repo/scripts/check_release_readiness.py`.
- Purpose:
  - Ensure the open-source release check covers shell wrappers, not only Python
    files and documentation snippets.
  - This specifically protects the new full7405 postprocess/finalization
    wrappers from broken line endings, missing bash shebangs, missing strict
    mode, or syntax errors on Linux.
- Implementation:
  - Added a `SHELL_FILES` list covering the ser launchers and the two full7405
    wait wrappers.
  - The readiness check now enforces LF line endings, `#!/usr/bin/env bash`,
    and `set -euo pipefail`.
  - If `bash` is available in the current environment, readiness also runs
    `bash -n` for every listed shell script.
- Verification:
  - Local `python scripts\check_release_readiness.py` passed.
  - Local `PYTHONPATH=src python -m unittest discover -s tests -v` passed with
    `Ran 39 tests in 5.767s` and `OK`.
  - Local release preview passed.
  - Synced the strengthened readiness script to `ser`.
  - Remote readiness and release preview returned success after the shell checks
    were added.
  - Remote release preview README contains the colon-separated `EVAL_ROOTS`
    example, and the copied postprocess wrapper contains `LIST_SEPARATOR`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 07:45 +0800`.
  - `reuse04_patch28`: latest progress `5243/7405` at
    `2026-06-11 07:45:51`.
  - `reuse08_patch24`: latest progress `5248/7405` at
    `2026-06-11 07:45:51`.
  - Neither eval root had `scd_icml_mistral_summary.json` yet.
- Conclusion:
  - Open-source readiness is stronger, but the main reproduction remains
    incomplete until full7405 summaries and final F1/EM are produced.

## 2026-06-11 - Payload-based bandwidth sensitivity artifact added

- Experiment id: scd_icml_bandwidth_sensitivity_artifact_20260611_001
- Purpose:
  - Reduce the remaining paper-figure ambiguity around the bandwidth/latency
    tradeoff by making the result-table generator export an explicit
    payload-based bandwidth sensitivity table and SVG.
- Method:
  - Updated `scripts/build_scd_icml_result_tables.py`.
  - For rows with measured `payload_mb` and TTFT, estimate transfer time as
    `payload_mb * 8 / bandwidth_gbps` ms.
  - Use the current eval reference bandwidth `200 Gbps` to subtract the
    reference transfer component from measured TTFT, then re-add transfer time
    for `10, 25, 50, 100, 200 Gbps`.
  - Rows without `payload_mb` are deliberately excluded rather than filled with
    invented payload values. Current mixed-route rows have no payload bytes in
    the route metrics, so their latency evidence remains the direct TTFT curve.
- Outputs added by the generator:
  - `bandwidth_sensitivity.csv`
  - `bandwidth_sensitivity.md`
  - `bandwidth_sensitivity.svg`
- Release integration:
  - Added the three bandwidth-sensitivity outputs to
    `scripts/build_paper_artifact_bundle.py`.
  - Updated result-table, paper-bundle, and final-full7405-bundle regression
    tests to require the new outputs.
  - Synced the updated scripts/tests to `ser`, so the active full7405
    postprocess and finalizer watchers will use the new artifact contract when
    they reach table generation.
- Verification:
  - Local targeted tests passed:
    `PYTHONPATH=src python -m unittest tests.test_result_tables_script tests.test_paper_artifact_bundle_script tests.test_finalize_full7405_bundle_script -v`.
  - Local `python scripts\check_release_readiness.py` passed.
  - Local release preview passed.
  - Remote targeted tests passed with `Ran 5 tests in 0.442s` and `OK`.
  - Local full unittest passed with `Ran 39 tests in 5.601s` and `OK`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 07:59 +0800`.
  - `reuse04_patch28`: latest progress `5556/7405` at
    `2026-06-11 07:59:40`.
  - `reuse08_patch24`: latest progress `5549/7405` at
    `2026-06-11 07:59:42`.
  - `wait_postprocess_tables.log` still reports `summaries_ready=0/2`; no
    final full7405 summary files existed yet.
- Conclusion:
  - This closes a code/release artifact gap for bandwidth-sensitivity evidence
    without fabricating payloads. The main reproduction remains open until
    HotpotQA-compressed full7405 produces accepted F1/EM and the final tables
    and bundle are generated.

## 2026-06-11 - Bandwidth artifact documented in release surface

- Experiment id: scd_icml_bandwidth_docs_release_sync_20260611_001
- Purpose:
  - Keep the open-source release documentation aligned with the new
    `bandwidth_sensitivity.*` table/figure outputs.
- Changes:
  - Updated runtime `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` to describe the payload-based
    bandwidth-sensitivity artifacts and the no-imputation boundary for rows
    without measured payload bytes.
  - Updated `scripts/check_release_readiness.py` so documentation checks require
    the `bandwidth_sensitivity` snippet in the release docs.
- Verification:
  - Local `python scripts\check_release_readiness.py` passed.
  - Local targeted tests for result tables, paper artifact bundle, and
    full7405 finalizer passed with `Ran 5 tests in 0.780s` and `OK`.
  - Local release preview passed.
  - Synced the updated docs and readiness script to `ser`.
  - Remote readiness and remote release-preview commands returned success.
  - Remote release-preview grep confirmed `bandwidth_sensitivity` appears in
    `README.md`, `REPRODUCING_SCD_ICML.md`,
    `OPEN_SOURCE_RELEASE_MANIFEST.md`, and
    `scripts/check_release_readiness.py`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 08:06 +0800`.
  - `reuse04_patch28`: latest progress `5713/7405`.
  - `reuse08_patch24`: latest progress `5700/7405`.
  - Both `scd_icml_mistral_summary.json` files are still absent.
- Conclusion:
  - The code and release docs are now consistent for the final table contract.
    The remaining blocking evidence is still the HotpotQA-compressed full7405
    F1/EM and the post-full7405 bundle.

## 2026-06-11 - Current partial bundle refreshed with bandwidth artifacts

- Experiment id: scd_icml_partial_bundle_current_20260611_005
- Purpose:
  - Keep the local pre-full7405 evidence bundle aligned with the current
    result-table contract while waiting for HotpotQA-compressed full7405.
- Method:
  - Rebuilt partial result tables on `ser` with the current
    `scripts/build_scd_icml_result_tables.py`, using the existing patch-anchor
    and route-metric JSON inputs from partial bundle `004`.
  - Copied the generated directory back to:
    `Experiment/results/scd_icml_result_tables_partial_20260611_005`.
  - Built local paper artifact bundle:
    `Experiment/results/paper_artifact_bundle_current_20260611_005`.
- Outputs:
  - `scd_icml_result_tables_partial_20260611_005/bandwidth_sensitivity.csv`
  - `scd_icml_result_tables_partial_20260611_005/bandwidth_sensitivity.md`
  - `scd_icml_result_tables_partial_20260611_005/bandwidth_sensitivity.svg`
  - `paper_artifact_bundle_current_20260611_005/tables/bandwidth_sensitivity.md`
  - `paper_artifact_bundle_current_20260611_005/figures/bandwidth_sensitivity.svg`
- Verification:
  - Remote table generation printed:
    `result_tables=/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_005`.
  - Remote manifest contains `bandwidth_sensitivity.csv/md/svg`.
  - Local `scripts/build_paper_artifact_bundle.py` produced
    `paper_artifact_bundle_current_20260611_005`.
  - Local `python scripts\check_release_readiness.py` passed.
  - Local targeted unittest passed with `Ran 5 tests in 0.774s` and `OK`.
  - Local `scripts/check_paper_artifacts.py` passed for the current-safe paper
    against bundle `005`, including required
    `tables/bandwidth_sensitivity.md`.
  - Bundle `005` manifest reports `status=ok` and `missing=[]`.
- HotpotQA-compressed full7405 latest refresh:
  - Checked on `ser` around `2026-06-11 08:09 +0800`.
  - `reuse04_patch28`: latest progress `5774/7405`.
  - `reuse08_patch24`: latest progress `5758/7405`.
  - Summary files and `full7405_route_metrics.json` are still absent.
- Conclusion:
  - The local pre-full7405 artifact bundle is now consistent with the latest
    release contract. The main reproduction still waits on full7405 F1/EM and
    the final post-full7405 bundle.

## 2026-06-11 - HotpotQA full7405 progress refresh 08:15

- Purpose:
  - Record the latest long-running full7405 status after refreshing the current
    partial artifact bundle.
- Remote status:
  - Checked on `ser` around `2026-06-11 08:15 +0800`.
  - `reuse04_patch28`: latest progress `5900/7405`.
  - `reuse08_patch24`: latest progress `5879/7405`.
  - Both `scd_icml_mistral_summary.json` files are still absent.
  - `wait_postprocess_tables.log` reports `summaries_ready=0/2`.
  - `wait_finalize_bundle.log` reports `table_ready=no`.
- Conclusion:
  - Jobs remain healthy and progressing; final HotpotQA F1/EM is still pending.

## 2026-06-11 - HotpotQA full7405 progress refresh 08:19

- Purpose:
  - Record the latest long-running full7405 status after local full-test and
    release-preview verification.
- Verification:
  - Local full unittest passed with `Ran 39 tests in 10.887s` and `OK`.
  - Local release preview passed:
    `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`.
- Remote status:
  - Checked on `ser` around `2026-06-11 08:19 +0800`.
  - `reuse04_patch28`: latest progress `5990/7405`.
  - `reuse08_patch24`: latest progress `5968/7405`.
  - Both `scd_icml_mistral_summary.json` files are still absent.
- Conclusion:
  - Code/release checks remain green; final HotpotQA full7405 metrics are still
    pending.

## 2026-06-11 - HotpotQA full7405 progress refresh 08:23

- Purpose:
  - Record the latest full7405 polling state while waiting for final summaries.
- Remote status:
  - Checked on `ser` around `2026-06-11 08:23 +0800` after a short polling
    interval.
  - `reuse04_patch28`: latest progress `6081/7405`.
  - `reuse08_patch24`: latest progress `6050/7405`.
  - Both `scd_icml_mistral_summary.json` files are still absent.
  - Postprocess and finalizer watchers remain active; no duplicate launch is
    needed.
- Conclusion:
  - Jobs are still healthy and advancing. Final HotpotQA full7405 F1/EM remains
    the only unresolved main-result gate.

## 2026-06-11 - HotpotQA full7405 progress refresh 08:24

- Purpose:
  - Record the latest full7405 polling state before the final summaries exist.
- Remote status:
  - Checked on `ser` around `2026-06-11 08:24 +0800`.
  - `reuse04_patch28`: latest progress `6116/7405`.
  - `reuse08_patch24`: latest progress `6084/7405`.
  - Both `scd_icml_mistral_summary.json` files are still absent.
  - `wait_postprocess_tables.log` reports `summaries_ready=0/2` at
    `2026-06-11T08:22:02+08:00`.
  - `wait_finalize_bundle.log` reports `table_ready=no` at
    `2026-06-11T08:22:23+08:00`.
- Conclusion:
  - The existing eval jobs and waiters are healthy. The remaining work is still
    to inspect full7405 F1/EM after both summaries land.

## 2026-06-11 - HotpotQA full7405 progress refresh 08:33

- Purpose:
  - Record the latest full7405 polling state after resuming the active
    reproduction goal.
- Local code verification:
  - Full unittest passed with `Ran 39 tests in 6.403s` and `OK`.
  - `python scripts\check_release_readiness.py` passed in
    `Experiment/core_code/runtime_repo`.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed and the generated release preview also passed readiness checks.
- Remote status:
  - Checked on `ser` around `2026-06-11 08:33 +0800`.
  - `reuse04_patch28`: latest progress `6294/7405`.
  - `reuse08_patch24`: latest progress `6265/7405`.
  - Both eval processes remain live, and no error/traceback lines were found in
    the checked `reuse08_patch24` stderr window.
  - Both `scd_icml_mistral_summary.json` files are still absent.
  - `wait_postprocess_tables.log` reports `summaries_ready=0/2` at
    `2026-06-11T08:32:02+08:00`.
  - `wait_finalize_bundle.log` reports `table_ready=no` at
    `2026-06-11T08:32:23+08:00`.
- Conclusion:
  - The open-source code path remains green. The remaining main-result gate is
    still the full7405 HotpotQA F1/EM summary and the final post-full7405
    artifact bundle.

## 2026-06-11 - HotpotQA full7405 finalization

- Purpose:
  - Close the HotpotQA-compressed full7405 gate and regenerate the final
    release-facing paper artifact bundle.
- Remote completion:
  - `reuse04_patch28` completed `7405/7405` at `2026-06-11 09:21:40 +0800`.
  - `reuse08_patch24` completed `7405/7405` at `2026-06-11 09:25:46 +0800`.
  - The first postprocess watcher failed because the shell wrapper did not
    export `PYTHONPATH`, producing `ModuleNotFoundError: No module named
    'scd_repro'`.
  - Fixed `scripts/wait_postprocess_scd_icml_full7405_tables.sh` and
    `scripts/wait_finalize_scd_icml_full7405_bundle.sh` to export
    `PYTHONPATH="${RUNTIME}/src${PYTHONPATH:+:${PYTHONPATH}}"`.
  - Resynced the scripts to `ser`, verified them with remote `bash -n`, and
    relaunched postprocess/finalizer retry watchers.
- Full7405 results:
  - Full Consumer: F1/EM `0.291855/0.206077`.
  - `reuse04_patch28`: SCD F1/EM `0.328284/0.242944`, F1 retention
    `1.124820`; selected by the full7405 absolute F1-drop gate.
  - `reuse08_patch24`: SCD F1/EM `0.242740/0.169885`, F1 drop `0.049115`;
    rejected by the full7405 gate.
- Generated artifacts:
  - Remote:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/full7405_route_metrics.json`.
  - Remote:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/hotpotqa_full7405_f1_drop003_selection.json`.
  - Local:
    `Experiment/results/scd_icml_result_tables_full7405_20260611_001`.
  - Local:
    `Experiment/results/paper_artifact_bundle_with_full7405_20260611_001`.
  - Finalizer report: `status=ok`, `bundle_status=ok`, `copied=47`,
    `missing=0`, and a 20-row full7405-guarded
    `mixed_route_boundary_table.csv`.
- Verification:
  - `python scripts\check_paper_artifacts.py ... --bundle-dir
    C:\Users\mql16\Desktop\SCD-ICML\Experiment\results\paper_artifact_bundle_with_full7405_20260611_001`
    passed and wrote the final local `paper_artifact_check_report.json`.
  - The current-safe paper was updated from full7405-pending text to the
    `reuse04_patch28` full7405 result.
  - `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=current_safe_build
    scd-icml-current-safe-paper-2026-06-10.tex` passed twice and produced
    `Publication/paper/current_safe_build/scd-icml-current-safe-paper-2026-06-10.pdf`.
  - Full unittest passed with `Ran 39 tests in 5.961s` and `OK`.
  - `python scripts\check_release_readiness.py` passed.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed, and the generated release preview passed readiness checks.
- Conclusion:
  - The current non-Qwen release-facing evidence is closed for CMRC2018,
    WikiText-2, and HotpotQA-compressed full7405 under dataset-specific route
    gates. HotpotQA-compressed is the limiting dataset and accepts only
    `reuse04_patch28` among the full7405 candidates evaluated here.

## 2026-06-11 - HotpotQA full7405 boundary reuse05-07 launch

- Purpose:
  - Resolve the remaining maximum-reuse boundary for HotpotQA-compressed
    full7405 after `reuse04_patch28` passed and `reuse08_patch24` failed.
- Method:
  - Launched conditional full7405 evals for `reuse05_patch27`,
    `reuse06_patch26`, and `reuse07_patch25`.
  - Used the existing work-queue scheduler with GPU pairs `0,1` and `2,3` on
    `ser`.
  - Kept the same HotpotQA-trained reuse and patch banks, `EVAL_LIMIT=0`,
    `MAX_LENGTH=512`, `MAX_NEW_TOKENS=32`, and `RUNTIME_PATCH_WINDOW_SIZE=1`.
- Remote run:
  - Run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001`.
  - Launcher PID: `3380160`.
  - Initial status: `reuse05_patch27` and `reuse06_patch26` started at
    `2026-06-11T09:45:00+08:00`; `reuse07_patch25` remains queued until one
    worker finishes.
- Scheduler update:
  - A manual attempt to start `reuse07_patch25` early on physical GPUs `0,2`
    exited during model load with CUDA OOM on GPU 0. This produced no valid
    summary and must not be treated as a route-quality result.
  - A second immediate fast-path attempt also exited with CUDA OOM before any
    valid progress rows were produced.
  - To avoid both duplicate all-method evaluation and unsafe oversubscription,
    `reuse07_patch25` was removed from the original route queue and handed to a
    wait-then-run helper. The helper waits for `reuse05` or `reuse06` to finish
    and then runs `reuse07_patch25` with the existing Full+SCD-only eval config
    (`configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`), which is
    sufficient for the F1/EM selection gate.
  - A boundary postprocess watcher was launched for the five full7405 roots
    (`reuse04` through `reuse08`) with postprocess PID `3384419`; a matching
    finalizer watcher was launched with PID `3384420`.
  - The wait-then-run helper PID is `3392102`; it writes
    `logs/conditional_reuse_reuse07_patch25.wait_then_fast_full_scd.log`.
- Next action:
  - Monitor summaries and combine `reuse04` through `reuse08` before updating
    the final route selection and paper bundle.

## 2026-06-11 - HotpotQA full7405 boundary progress 10:13

- Purpose:
  - Record live status for the `reuse05`/`reuse06`/`reuse07` full7405 boundary
    sweep.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:13 +0800`.
  - `reuse05_patch27`: `618/7405`, estimated rate `0.371` sample/s, ETA
    around `2026-06-11 15:18`.
  - `reuse06_patch26`: `607/7405`, estimated rate `0.364` sample/s, ETA
    around `2026-06-11 15:24`.
  - `reuse07_patch25`: no valid progress rows yet; the wait-then-run helper is
    still waiting for either `reuse05` or `reuse06` to finish and release a GPU
    pair.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`; finalizer
    still reports `table_ready=no`.
- Conclusion:
  - The active full7405 boundary run is healthy but still far from summary
    generation. No additional oversubscription attempts should be made before
    one active eval finishes.

## 2026-06-11 - HotpotQA full7405 boundary fast-path switch

- Purpose:
  - Reduce unnecessary runtime for the `reuse05`/`reuse06`/`reuse07` full7405
    boundary sweep while preserving the F1/EM evidence needed for selecting the
    maximum acceptable reuse depth.
- Change:
  - Stopped the in-progress all-method `reuse05_patch27` and `reuse06_patch26`
    evals, which were running Full Consumer, Raw KV, 4-bit KV, Reuse-only, and
    SCD.
  - Relaunched `reuse05_patch27` and `reuse06_patch26` with
    `configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`, which runs
    only Full Consumer and SCD. These are the only methods required by
    `select_quality_gated_scd_route.py` for the HotpotQA F1-drop gate.
  - Preserved the boundary postprocess watcher and the `reuse07_patch25`
    wait-then-run helper.
- Remote evidence:
  - Switch log:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001/logs/full_scd_switch.log`.
  - Relaunched PIDs:
    - `reuse05_patch27`: `3397409`.
    - `reuse06_patch26`: `3397412`.
  - Verified the generated `tmp_cfg/eval_lite_to_7b.json` files set
    `run_raw_kv=false`, `run_quantized_4bit=false`, `run_reuse_only=false`,
    and `run_scd_full=true`.
- Current status:
  - At `2026-06-11 10:20 +0800`, `reuse05_patch27` reached `150/7405` at
    about `0.788` sample/s, ETA around `12:54`.
  - At the same check, `reuse06_patch26` reached `147/7405` at about `0.777`
    sample/s, ETA around `12:56`.
  - `reuse07_patch25` remains queued through the wait-then-run helper and will
    use the same Full+SCD-only path after one GPU pair is released.
- Conclusion:
  - This switch preserves the primary F1/EM boundary evidence and cuts the
    expected wait by more than two hours versus continuing the all-method
    `reuse05`/`reuse06` run.

## 2026-06-11 - HotpotQA full7405 boundary fast-path progress 10:27

- Purpose:
  - Record post-switch stability for the Full+SCD-only boundary evals.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:27 +0800`.
  - `reuse05_patch27`: `493/7405`, estimated rate `0.795` sample/s, ETA
    around `2026-06-11 12:53`.
  - `reuse06_patch26`: `484/7405`, estimated rate `0.780` sample/s, ETA
    around `2026-06-11 12:56`.
  - `reuse07_patch25`: still waiting for one of the first two runs to produce
    a summary and release a GPU pair.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`.
- Conclusion:
  - The Full+SCD-only fast path is stable. Further reducing to SCD-only would
    require injecting Full Consumer rows into per-route summaries and changing
    watcher assumptions, so it is not worth the added risk while the current
    path is already substantially faster.

## 2026-06-11 - HotpotQA full7405 boundary fast-path progress 10:40

- Purpose:
  - Record continued health of the Full+SCD-only HotpotQA full7405 boundary
    evals after the fast-path switch.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:40 +0800`.
  - `reuse05_patch27`: `1104/7405`, estimated rate `0.794` sample/s, ETA
    around `2026-06-11 12:53`.
  - `reuse06_patch26`: `1091/7405`, estimated rate `0.785` sample/s, ETA
    around `2026-06-11 12:55`.
  - `reuse07_patch25`: still waiting for `reuse05` or `reuse06` to finish and
    free a GPU pair.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`; finalizer
    remains at `table_ready=no`.
- Conclusion:
  - The fast-path evals are progressing normally. No additional scheduling
    change is needed before the first summary lands.

## 2026-06-11 - HotpotQA full7405 boundary fast-path progress 10:45

- Purpose:
  - Record continued health after a local SSH polling command returned no
    useful output.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:45 +0800`.
  - `reuse05_patch27`: `1357/7405`.
  - `reuse06_patch26`: `1342/7405`.
  - `reuse07_patch25`: wait-then-run helper remains active and waiting for
    either `reuse05` or `reuse06` to produce a summary.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`; finalizer
    remains at `table_ready=no`.
- Conclusion:
  - The prior missing polling output was local/SSH-side noise, not a remote job
    failure. The active fast-path evals continue to make progress.

## 2026-06-11 - HotpotQA full7405 boundary progress and finalizer hardening 10:50

- Purpose:
  - Continue monitoring the Full+SCD-only HotpotQA full7405 boundary sweep and
    close a release-check gap in the paper artifact finalizer.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:50 +0800`.
  - `reuse05_patch27`: `1595/7405`, estimated rate `0.797` sample/s, ETA
    around `2026-06-11 12:52`.
  - `reuse06_patch26`: `1575/7405`, estimated rate `0.787` sample/s, ETA
    around `2026-06-11 12:54`.
  - `reuse07_patch25`: no valid progress rows yet; wait-then-run helper remains
    active and is waiting for either `reuse05` or `reuse06` to produce a
    summary and free a GPU pair.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`; finalizer
    remains at `table_ready=no`.
- Code change:
  - Strengthened `scripts/finalize_scd_icml_full7405_bundle.py` so its default
    paper check requires the HotpotQA full7405 selected-route text and forbids
    stale pending-status text such as `full7405 remains pending`.
  - Extended the same finalizer gate to require the current safe paper's four
    claim-facing table labels:
    `tab:current_cmrc_selected_route`, `tab:current_reuse_boundary`,
    `tab:current_non_cmrc_diagnostics`, and `tab:offline_cost`.
  - Added
    `tests.test_finalize_full7405_bundle_script.test_cli_rejects_stale_full7405_pending_paper_text`.
- Verification:
  - Local targeted test command:
    `PYTHONPATH=src python -m unittest tests.test_finalize_full7405_bundle_script tests.test_paper_artifact_check_script -v`.
  - Result: 6 tests passed locally.
  - Local full verification after the finalizer changes:
    `PYTHONPATH=src python -m unittest discover -s tests -v` passed 40 tests,
    and `python scripts/check_release_readiness.py` passed.
  - Local release preview was rebuilt with
    `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`,
    and the rebuilt preview passed
    `python release_preview\scd_icml_runtime_release\scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release`.
  - The stricter paper artifact check passed against the current safe paper and
    `Experiment/results/paper_artifact_bundle_with_full7405_20260611_001`,
    writing
    `paper_artifact_check_report_strict_20260611_boundary_guard.json`.
  - Synced the updated finalizer script and test to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Ran the same targeted command on `ser` after each sync; 6 tests passed
    remotely after the stale-text guard and again after the table-label guard.
  - A later remote `check_release_readiness.py` invocation exited with status
    `0` but returned no stdout through the SSH wrapper, so it should be rerun
    as part of the final release-preview gate rather than used as the only
    release-readiness evidence.
- Conclusion:
  - The currently running finalizer watcher has not yet invoked the finalizer,
    so the synced remote runtime will apply the stronger stale-paper guard when
    the boundary result table becomes ready.

## 2026-06-11 - HotpotQA full7405 boundary fast-path progress 10:58

- Purpose:
  - Record the next health check after finalizer hardening and full local
    verification.
- Remote status:
  - Checked on `ser` at `2026-06-11 10:58 +0800`.
  - `reuse05_patch27`: `1969/7405`, estimated rate `0.795` sample/s, ETA
    around `2026-06-11 12:52`.
  - `reuse06_patch26`: `1945/7405`, estimated rate `0.785` sample/s, ETA
    around `2026-06-11 12:54`.
  - `reuse07_patch25`: no progress rows yet because the wait-then-run helper is
    still waiting for one of the first two summaries.
  - Boundary postprocess watcher remains at `summaries_ready=2/5`; finalizer
    remains at `table_ready=no`.
- Conclusion:
  - The active fast-path evals continue normally. No new launch or scheduler
    change is needed before the first summary lands.

## 2026-06-11 - Release docs full7405 state refresh

- Purpose:
  - Remove stale open-source documentation that still described the
    HotpotQA-compressed full7405 route table as pending.
- Change:
  - Updated `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` under `Experiment/core_code/runtime_repo`.
  - The docs now state that the first full7405 passing/failing bracket is
    closed: `reuse04_patch28` passes and `reuse08_patch24` fails.
  - The docs now frame `reuse05_patch27`, `reuse06_patch26`, and
    `reuse07_patch25` as maximum-reuse boundary refinement, not as the initial
    full7405 closure.
  - The paper artifact check examples now include the stricter full7405 text,
    table-label, and stale-pending guards.
- Verification:
  - Stale phrase scan over the release docs only returns the intended
    `--forbid-text` examples.
  - `python scripts\check_release_readiness.py` passed.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed, and the rebuilt preview passed
    `python release_preview\scd_icml_runtime_release\scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release`.
  - Synced the updated release docs to the active `ser` runtime directory.
- Conclusion:
  - The open-source docs no longer contradict the completed full7405 bracket,
    while still warning that the exact maximum HotpotQA reuse count awaits
    the running 05/06/07 refinement.

## 2026-06-11 - Release readiness stale-doc guard

- Purpose:
  - Make the release readiness gate catch stale HotpotQA full7405 documentation
    rather than relying on manual grep.
- Change:
  - Updated `scripts/check_release_readiness.py`.
  - The doc check now requires the release docs to mention `reuse04_patch28`
    and `reuse08_patch24` in the current HotpotQA full7405 context.
  - The doc check now rejects stale phrases such as `HotpotQA main row is not
    closed`, `pending compressed-protocol full7405 route table`, and `being
    closed with full7405 evaluation`.
- Verification:
  - `python scripts\check_release_readiness.py` passed locally.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed 40 tests.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    passed, and the rebuilt release preview passed its own readiness check.
  - Synced the updated readiness script to the active `ser` runtime directory.
  - Remote readiness in that `ser` runtime also passed with explicit output:
    `release readiness check passed:
    /disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Conclusion:
  - Future release-preview checks now fail if the open-source docs regress to
    the old full7405-pending state.

## 2026-06-11 - Remote checked finalizer watcher

- Purpose:
  - Ensure the remote boundary postprocess produces a paper-checked artifact
    bundle, not only a table/bundle-checked bundle.
- Issue found:
  - The existing `wait_finalize_scd_icml_full7405_bundle.sh` process uses the
    wrapper default `SKIP_PAPER_CHECK=1`, which is appropriate for a remote
    host without paper sources but does not validate stale paper text.
- Change:
  - Synced the current safe paper source and required PDFs to
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/paper_source_for_check`.
  - Verified the remote paper source with `scripts/check_paper_artifacts.py`
    using the stricter full7405/table-label/stale-text constraints.
  - Started a second finalizer watcher with `SKIP_PAPER_CHECK=0`, PID
    `3429341`, log
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/wait_finalize_boundary_checked.log`.
  - Its output directory is
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/paper_artifact_bundle_with_full7405_boundary_checked`.
- Conclusion:
  - When the boundary table becomes ready, the remote run should now produce
    both the original bundle and a checked bundle that validates the current
    paper source.

## 2026-06-11 - HotpotQA full7405 boundary progress 11:16

- Purpose:
  - Record health after adding the checked finalizer watcher.
- Remote status:
  - Checked on `ser` at `2026-06-11 11:16 +0800`.
  - `reuse05_patch27`: latest progress `2833/7405`.
  - `reuse06_patch26`: latest progress `2786/7405`.
  - `reuse07_patch25`: still waiting; the OOM text in its stderr is the stale
    earlier failed manual launch, not the queued wait-then-run helper.
  - Watchers:
    - Postprocess watcher remains at `summaries_ready=2/5`.
    - Original finalizer remains at `table_ready=no`.
    - Checked finalizer PID `3429341` is running and waiting for the same
      table.
- Conclusion:
  - The boundary evals and both finalizer watchers are healthy. No intervention
    is needed until the first of `reuse05` or `reuse06` produces a summary.

## 2026-06-11 - Finalizer selected-route paper-text guard

- Purpose:
  - Prevent a boundary-refined result from selecting `reuse05`, `reuse06`, or
    `reuse07` while the paper source still states the older
    `Reuse04/Patch28` route.
- Change:
  - Updated `scripts/finalize_scd_icml_full7405_bundle.py`.
  - If `hotpotqa_full7405_f1_drop003_selection.json` is present next to the
    result-table directory, the finalizer now reads `selected.reuse_n` and
    `selected.patch_n` and requires the paper source to contain
    `ReuseXX/PatchYY` for that selected route.
  - Added
    `tests.test_finalize_full7405_bundle_script.test_cli_rejects_paper_route_text_mismatch_when_selector_exists`.
- Verification:
  - Local targeted finalizer/checker tests passed with 7 tests.
  - Local full unittest passed with 41 tests.
  - Local release readiness passed.
  - The release preview was rebuilt and passed readiness.
  - Synced the updated finalizer and test to the active `ser` runtime; remote
    targeted finalizer/checker tests passed with 7 tests.
- Conclusion:
  - The checked boundary finalizer should now fail if the selected full7405
    HotpotQA route changes but the paper text is not updated to match.

## 2026-06-11 - Release docs for selector-route paper guard

- Purpose:
  - Document the finalizer behavior that ties the selected HotpotQA route in
    `hotpotqa_full7405_f1_drop003_selection.json` to the paper text.
- Change:
  - Updated `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - The docs now say that, when the selector report exists, the finalizer
    requires the paper source to contain the selected `ReuseXX/PatchYY` token.
- Verification:
  - Local `python scripts\check_release_readiness.py` passed.
  - Local release preview rebuild passed and the rebuilt preview passed
    readiness.
  - Synced the docs to the active `ser` runtime and remote readiness passed.
- Conclusion:
  - The open-source docs now describe the route-text guard that prevents stale
    HotpotQA route claims after boundary refinement.

## 2026-06-11 - HotpotQA full7405 boundary progress 11:29

- Purpose:
  - Record live status after selector-route paper guard changes and release-doc
    synchronization.
- Remote status:
  - Checked on `ser` at `2026-06-11 11:29 +0800`.
  - `reuse05_patch27`: latest progress `3426/7405`, estimated rate `0.794`
    sample/s, ETA around `2026-06-11 12:52`.
  - `reuse06_patch26`: latest progress `3368/7405`, estimated rate `0.781`
    sample/s, ETA around `2026-06-11 12:55`.
  - `reuse07_patch25`: still waiting for either `reuse05` or `reuse06` to
    produce a summary and free a GPU pair.
  - Postprocess watcher remains at `summaries_ready=2/5`.
  - Checked finalizer watcher remains active and waiting for
    `result_tables_with_full7405`.
- Conclusion:
  - The boundary evals are healthy and still compute-bound. No intervention is
    needed before the first summary appears.

## 2026-06-11 - Wait finalizer paper-check default

- Purpose:
  - Make the long-running finalizer wrapper safer when a paper source is
    available, without breaking remote hosts that do not have paper files.
- Change:
  - Updated `scripts/wait_finalize_scd_icml_full7405_bundle.sh`.
  - If `PAPER_TEX` is set and `SKIP_PAPER_CHECK` is not set, the wrapper now
    enables paper checking by default.
  - If no `PAPER_TEX` is set, the wrapper keeps the old remote-friendly default
    and skips paper checking.
  - Updated `README.md` and `REPRODUCING_SCD_ICML.md` to document this behavior.
- Verification:
  - Local `python scripts\check_release_readiness.py` passed.
  - Local release preview rebuild passed and the rebuilt preview passed
    readiness.
  - Synced the wrapper and docs to the active `ser` runtime; remote readiness
    passed there as well.
- Conclusion:
  - Future remote or local finalizer waits are less likely to accidentally skip
    paper checks when a paper source path has already been provided.

## 2026-06-11 - Local verification while HotpotQA boundary runs continue

- Purpose:
  - Keep the open-source runtime and paper artifacts verified while waiting for
    the full7405 `reuse05`/`reuse06`/`reuse07` boundary rows.
- Local code verification:
  - Ran `python -m unittest discover -s tests -v` in
    `Experiment/core_code/runtime_repo` with `PYTHONPATH=src`.
  - Result: 41 tests passed.
  - Ran `python scripts\check_release_readiness.py`.
  - Result: release readiness passed.
  - Rebuilt the release preview with
    `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`.
  - Result: the rebuilt preview passed release readiness.
  - Ran `python scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release`.
  - Result: release-preview readiness passed.
- Paper verification:
  - First direct `pdflatex` run from `Publication/paper` failed because
    `icml2026.sty` is stored under `Publication/paper/current_safe_build`.
  - Re-ran with `pdflatex -interaction=nonstopmode -halt-on-error -include-directory=current_safe_build scd-icml-current-safe-paper-2026-06-10.tex`.
  - Result: two passes completed and produced
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`.
- Remote status:
  - Checked `ser` at `2026-06-11 11:39 +0800`.
  - `reuse05_patch27`: `3891/7405`, ETA around `2026-06-11 12:53`.
  - `reuse06_patch26`: `3821/7405`, ETA around `2026-06-11 12:55`.
  - `reuse07_patch25`: waiting for the first completed 05/06 summary.
  - GPU/process inspection showed `reuse05` uses CUDA `0,1` and `reuse06`
    uses CUDA `2,3`, so all four cards are occupied by the two full7405 evals.
- Conclusion:
  - Local code, release preview, and current paper source are healthy. The
    remaining open gate is the final HotpotQA boundary table and the checked
    finalizer after 05/06/07 complete.

## 2026-06-11 - HotpotQA full7405 no-train-overlap filtering

- Purpose:
  - Address the leakage risk from HotpotQA-trained checkpoints whose training
    questions may overlap with the full7405 dev evaluation file.
- Audit:
  - Inspected
    `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/runs/strict_hotpotqa_200_total_heldout100_promptfix_max512_20260422/tmp_cfg/data_hotpot_train100_max512.json`.
  - Extracted training questions from the stored capture prompts using the
    Unicode `问题：` / `证据：` prompt delimiters.
  - Compared by normalized question text against
    `data/HotpotQA/raw/hotpot_dev_distractor_v1.json`.
  - Found `100` overlapping training questions in the full `7405` HotpotQA dev
    rows.
- Code change:
  - Added `scripts/filter_strict_eval_by_qids.py` to recompute strict-eval
    summaries from saved `per_sample_metrics.jsonl` after include/exclude qid
    filtering.
  - Added `tests/test_filter_strict_eval_by_qids_script.py`.
  - Added the script/test to release build and readiness checks, and documented
    the workflow in `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Verification:
  - Local targeted test passed.
  - Local release readiness passed.
  - Synced the script/test/docs to the active `ser` runtime.
  - Remote targeted test and remote release readiness passed.
- Filtered HotpotQA evidence:
  - Created
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_train100_overlap_qids.txt`.
  - Recomputed filtered strict summaries for completed `reuse04_patch28` and
    `reuse08_patch24` after excluding the 100 overlapping qids.
  - Filtered `reuse04_patch28`: Full F1/EM `0.292006/0.205886`; SCD F1/EM
    `0.328620/0.242710`; selected under F1-drop `<= 0.03`.
  - Filtered `reuse08_patch24`: Full F1/EM `0.292006/0.205886`; SCD F1/EM
    `0.242895/0.169884`; fails with F1 drop `0.049111`.
- Conclusion:
  - The current 04-vs-08 HotpotQA conclusion is stable after removing the 100
    train-overlap questions. The same filtering must be applied to
    `reuse05`/`reuse06`/`reuse07` after those full7405 evals finish.

## 2026-06-11 - HotpotQA full7405 filtered boundary update for reuse05/reuse06

- Purpose:
  - Extend the no-train-overlap HotpotQA boundary check to the newly completed
    `reuse05_patch27` and `reuse06_patch26` full7405 evaluations while
    `reuse07_patch25` is still running.
- Inputs:
  - Evaluation run:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001`.
  - Excluded-qid list:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_train100_overlap_qids.txt`.
  - Active runtime:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
- Method:
  - Ran `scripts/filter_strict_eval_by_qids.py` on the saved strict-eval
    `per_sample_metrics.jsonl` for `reuse05_patch27` and `reuse06_patch26`.
  - Excluded the 100 train-overlap HotpotQA qids, leaving `7305` held-out
    examples and `14610` per-mode metric rows per route.
  - Copied each route JSON into
    `filtered_eval_roots/<route>_no_train_overlap/` and regenerated route-level
    summaries for selector compatibility.
  - Re-ran `collect_route_eval_metrics.py` and
    `select_quality_gated_scd_route.py` over filtered
    `reuse04_patch28`, `reuse05_patch27`, `reuse06_patch26`, and
    `reuse08_patch24`.
- Results:
  - Filtered Full baseline: F1/EM `0.292006/0.205886`.
  - Filtered `reuse04_patch28`: SCD F1/EM `0.328620/0.242710`, passes.
  - Filtered `reuse05_patch27`: SCD F1/EM `0.323243/0.237509`, passes.
  - Filtered `reuse06_patch26`: SCD F1/EM `0.304488/0.221492`, passes.
  - Filtered `reuse08_patch24`: SCD F1/EM `0.242895/0.169884`, fails with
    F1 drop `0.049111`.
  - Among the completed filtered boundary points, the current maximum passing
    route is `reuse06_patch26`.
- Outputs:
  - Partial filtered metrics:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/full7405_route_metrics.no_train_overlap.partial_04_05_06_08.json`.
  - Partial filtered selection:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_f1_drop003_selection.no_train_overlap.partial_04_05_06_08.json`.
- Next action:
  - Wait for `reuse07_patch25` to finish, apply the same qid filtering to it,
    and then regenerate the final filtered and unfiltered HotpotQA boundary
    selection reports.

## 2026-06-11 - HotpotQA reuse07 full7405 sharded acceleration

- Purpose:
  - Reduce wall time for the remaining `reuse07_patch25` full7405 HotpotQA
    strict evaluation and use all four available GPUs.
- Decision:
  - The original single-process `reuse07_patch25` strict evaluation had reached
    roughly `385/7405` samples and was projected to finish around
    `2026-06-11 15:34 +0800`.
  - Because `eval_text_strict.py` already supports `--qid-file`, the full qid
    list was split into two disjoint shards rather than changing model logic.
- Method:
  - Generated two qid files from the full `hotpotqa_compressed` loader:
    `3703` qids for shard0 and `3702` qids for shard1.
  - Preserved the interrupted single-process strict-eval directory by moving it
    aside before writing the final merged output path.
  - Terminated only the active `reuse07_patch25` single-process tree.
  - Launched shard0 with `CUDA_VISIBLE_DEVICES=0,1` and shard1 with
    `CUDA_VISIBLE_DEVICES=2,3`.
  - Started a merge watcher that waits for both shard `per_sample_metrics.jsonl`
    files, concatenates the per-sample rows, writes a merged manifest, and
    re-aggregates summaries with `scripts/filter_strict_eval_by_qids.py` into
    the original `reuse07_patch25/scd_icml_mistral_summary.json` expected by
    the full7405 postprocess watcher.
- Launch evidence:
  - shard0 PID: `3485651`.
  - shard1 PID: `3485658`.
  - merge watcher PID: `3485664`.
  - First status check at `2026-06-11 13:08 +0800` showed shard0 at
    `52/3703` and shard1 at `50/3702`; both shard logs were growing and the
    merge watcher was alive.
- Next action:
  - Monitor both shards until merge completes, then apply the no-train-overlap
    qid filter to the merged `reuse07_patch25` output and regenerate the final
    HotpotQA full7405 boundary reports.

## 2026-06-11 - HotpotQA full7405 boundary final selection

- Purpose:
  - Close the HotpotQA-compressed full7405 maximum-Reuse boundary after
    completing `reuse05_patch27`, `reuse06_patch26`, and qid-sharded
    `reuse07_patch25`.
- Completed run evidence:
  - `reuse07_patch25` shard0 completed `3703/3703`.
  - `reuse07_patch25` shard1 completed `3702/3702`.
  - The merge watcher combined `14810` per-mode rows over `7405` unique qids
    and wrote
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001/lite_to_7b/conditional_evals/conditional_reuse_reuse07_patch25/scd_icml_mistral_summary.json`.
- Unfiltered full7405 selector:
  - Report:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_f1_drop003_selection.json`.
  - Full baseline F1/EM: `0.291855/0.206077`.
  - `reuse04_patch28`: SCD F1/EM `0.328284/0.242944`, passes.
  - `reuse05_patch27`: SCD F1/EM `0.322962/0.237812`, passes.
  - `reuse06_patch26`: SCD F1/EM `0.304212/0.221742`, passes and is the
    maximum passing route.
  - `reuse07_patch25`: SCD F1/EM `0.257260/0.181769`, fails with F1 drop
    `0.034595`.
  - `reuse08_patch24`: SCD F1/EM `0.242740/0.169885`, fails with F1 drop
    `0.049115`.
- No-train-overlap selector:
  - Excluded the 100 overlapping HotpotQA training qids, leaving `7305`
    held-out examples.
  - Report:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/hotpotqa_full7405_f1_drop003_selection.no_train_overlap.json`.
  - Full baseline F1/EM: `0.292006/0.205886`.
  - `reuse06_patch26` remains selected with SCD F1/EM
    `0.304488/0.221492`.
  - `reuse07_patch25` still fails with F1 drop `0.034507`.
- Paper/bundle synchronization:
  - Updated `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`
    from `Reuse04/Patch28` to `Reuse06/Patch26` for the HotpotQA-compressed
    full7405 selected route.
  - Synced the updated paper source to the remote `paper_source_for_check`.
  - Ran `scripts/finalize_scd_icml_full7405_bundle.py` against the final
    boundary tables and updated paper source.
  - Finalizer report:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/paper_artifact_bundle_with_full7405_boundary_checked/finalize_report.json`.
  - Result: `status=ok`, `bundle_status=ok`, selected paper-text requirement
    `Reuse06/Patch26`, `paper_check=true`.
  - `paper_artifact_check_report.json` has `errors=[]` and `missing_images=[]`.
- Local artifact copies:
  - `Experiment/results/scd_icml_result_tables_full7405_boundary_20260611_001/`.
  - `Experiment/results/paper_artifact_bundle_with_full7405_boundary_checked_20260611_001/`.
  - `Experiment/results/hotpotqa_full7405_boundary_reports_20260611_001/`.
- Conclusion:
  - HotpotQA-compressed full7405 is now closed for the current non-Qwen release
    path. The maximum F1-gated route is `reuse06_patch26`; `reuse07` and
    `reuse08` are quality-failing boundary points.

## 2026-06-11 - Final local verification after boundary synchronization

- Purpose:
  - Verify the open-source runtime, final result bundle, and updated paper
    source after switching the HotpotQA-compressed full7405 selected route to
    `Reuse06/Patch26`.
- Runtime verification:
  - Ran `python -m unittest discover -s tests -v` in
    `Experiment/core_code/runtime_repo` with `PYTHONPATH=src`.
  - Result: `42` tests passed.
  - Ran `python scripts\check_release_readiness.py`.
  - Result: release readiness passed for the runtime root.
  - Rebuilt the release preview with
    `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`.
  - Ran
    `python scripts\check_release_readiness.py --root release_preview\scd_icml_runtime_release`.
  - Result: release-preview readiness passed.
- Paper artifact verification:
  - Ran `scripts\check_paper_artifacts.py` against local
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` and
    `Experiment/results/paper_artifact_bundle_with_full7405_boundary_checked_20260611_001`.
  - Required text included `Reuse06/Patch26`; forbidden stale full7405-pending
    text was checked.
  - Result:
    `Experiment/results/paper_artifact_bundle_with_full7405_boundary_checked_20260611_001/paper_artifact_check_report.local_final.json`
    has `errors=[]` and `missing_images=[]`.
- Paper compilation:
  - Ran `pdflatex -interaction=nonstopmode -halt-on-error -include-directory=current_safe_build scd-icml-current-safe-paper-2026-06-10.tex` twice from `Publication/paper`.
  - Result: PDF compiled successfully:
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`, `16` pages,
    `884477` bytes.
  - MiKTeX emitted its standard update warning and LaTeX underfull box
    warnings; no compile error occurred.
- Sanity search:
  - Searched current paper/docs/result README for stale claim-facing text such
    as `Reuse04/Patch28 route`, `full7405 remains pending`, and
    `boundary refinement ... running`.
  - Remaining matches are historical experiment/plan log records, not current
    paper-facing claims.
- Conclusion:
  - The current local non-Qwen release path is verified after the final
    HotpotQA boundary update.

## 2026-06-11 - Neighboring-baseline open rerun path and smoke

- Purpose:
  - Move the original appendix `neighboring_baselines` item from a checked
    exclusion-only state toward full reproducibility by adding public current
    runtime entry points and a table consolidator.
- Code changes:
  - Added `Experiment/core_code/runtime_repo/scripts/run_svdllm_style_kv_cmrc.py`.
  - Added `Experiment/core_code/runtime_repo/scripts/build_neighboring_baseline_table.py`.
  - Included the SVD-LLM-style, SliceGPT-style, TensorLLM-style, and neighboring
    table scripts in `scripts/build_release_tree.py` and
    `scripts/check_release_readiness.py`.
  - Added `tests/test_neighboring_baseline_table_script.py`.
  - Updated `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` so the appendix baseline scripts and the
    final HotpotQA `reuse06_patch26` boundary are documented.
  - Fixed `scripts/filter_strict_eval_by_qids.py` to prefer the local `src/`
    package path before importing `scd_repro`.
- Local verification:
  - `python -m unittest tests.test_neighboring_baseline_table_script -v` passed.
  - `python scripts\check_release_readiness.py` passed.
  - `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v` passed with
    `43` tests.
  - `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`
    rebuilt the preview and its readiness check passed.
- Remote verification on `ser`:
  - Before smoke, GPUs `0,1,2,3` all showed `18 MiB` used and `0%`
    utilization.
  - Synced the updated scripts/docs/test into
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py` passed.
  - Remote `PYTHONPATH=src python -m unittest tests.test_neighboring_baseline_table_script -v`
    passed.
  - Ran `limit=2`, `max_length=128`, `max_new_tokens=8` smoke jobs:
    - SVD-LLM-style:
      `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_smoke_20260611_001/svdllm/summary.json`.
    - TensorLLM-style:
      `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_smoke_20260611_001/tensorllm/summary.json`.
    - SliceGPT-style:
      `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_smoke_20260611_001/slicegpt/summary.json`.
  - Ran `scripts/build_neighboring_baseline_table.py`; output table:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_smoke_20260611_001/table/neighboring_baselines.md`.
- Local result copy:
  - Copied the smoke run to
    `Experiment/results/neighboring_baselines_smoke_20260611_001/`.
- Smoke table values:
  - SVD-LLM-style: F1 `0.0556`, TTFT `1053.94 ms`.
  - SliceGPT-style: F1 `0.0833`, TTFT `47.09 ms`.
  - TensorLLM-style: F1 `0.0909`, TTFT `763.21 ms`.
  - SCD reference row in the smoke table uses the existing CMRC selected-route
    summary: F1 `0.6428`, TTFT `40.80 ms`.
- Interpretation:
  - This is execution and table-generation evidence only. Because the three
    baseline smoke rows use `limit=2`, their quality retention is not a
    paper-quality appendix result and should not be cited as reproduced
    neighboring-baseline numbers.
- Next action:
  - To restore original appendix Table `neighboring_baselines` as a reproduced
    non-Qwen paper artifact, run full-sample CMRC reruns for the three adapted
    neighboring baselines and rebuild the table from those summaries.

## 2026-06-11 - Neighboring-baseline CMRC200 full rerun launched

- Purpose:
  - Start the full-sample CMRC neighboring-baseline rerun needed to restore the
    original appendix `neighboring_baselines` table as current-runtime numeric
    evidence.
- Command path:
  - Remote runtime:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Wrapper:
    `scripts/run_ser_neighboring_baselines_cmrc.sh`.
  - Run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001`.
- Launch:
  - Started with `LIMIT=200`.
  - Wrapper PID observed: `3565103`.
  - SVD-LLM-style child PID observed: `3565119`.
  - TensorLLM-style child PID observed: `3565120`.
  - Initial GPU assignment: SVD-LLM-style on visible `0,1`, TensorLLM-style on
    visible `2,3`; SliceGPT-style is configured to run after both finish.
- Status after launch:
  - Remote readiness passed before launch.
  - GPUs were idle before launch: each showed about `18 MiB` used and `0%`
    utilization.
  - After launch, SVD/Tensor processes entered model-loading/disk-I/O state and
    opened the intended GPU devices.
  - At the latest check, GPU0 and GPU2 each held about `12691 MiB`; GPU1 and
    GPU3 had not yet loaded the consumer model.
  - Logs currently show model-loading warnings and completed first weight-load
    progress bars, with no Python traceback.
- Logs:
  - Wrapper log:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001/wrapper.nohup.log`.
  - SVD log:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001/logs/svdllm.log`.
  - TensorLLM log:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001/logs/tensorllm.log`.
- Next action:
  - Monitor until both parallel jobs finish, then confirm the wrapper starts
    SliceGPT-style and finally writes
    `table/neighboring_baselines.{csv,md,tex}`.

## 2026-06-11 - Neighboring-baseline CMRC200 full rerun completed; SCD256 alignment launched

- Purpose:
  - Move appendix `neighboring_baselines` from smoke-only evidence toward a
    current-runtime numeric reproduction, while preventing mixed-protocol rows
    from becoming claim-facing evidence.
- Completed adapted-baseline rerun:
  - Remote run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001`.
  - Local copy:
    `Experiment/results/neighboring_baselines_cmrc200_20260611_001/`.
  - Outputs:
    `svdllm/summary.json`, `slicegpt/summary.json`,
    `tensorllm/summary.json`, and
    `table/neighboring_baselines.{csv,md,tex}`.
  - Table values from the first generated table:
    SVD-LLM-style retention `7.2%`, speedup `0.17x`, F1 `0.0055`;
    SliceGPT-style retention `58.1%`, speedup `0.87x`, F1 `0.0444`;
    TensorLLM-style retention `2.5%`, speedup `0.10x`, F1 `0.0019`;
    SCD retention `97.8%`, speedup `2.47x`, F1 `0.6428`.
- Protocol issue:
  - The adapted baselines used `max_length=256`, while the first SCD table row
    used the existing selected-route summary from `max_length=1024`.
  - The table supports the broad trend but is not yet a same-protocol
    claim-facing appendix row.
- Code update:
  - Updated
    `Experiment/core_code/runtime_repo/scripts/build_neighboring_baseline_table.py`
    to record source protocols and emit warnings when protocol fields or
    Full-consumer F1 values differ across sources.
  - Synced the updated table builder to the remote runtime repo.
- New alignment run:
  - Launched same-protocol SCD route evaluation on `ser`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_cmrc200_maxlen256_for_neighboring_20260611_001`.
  - Route: selected CMRC Reuse19/Patch13.
  - Settings: `DATASETS=cmrc2018`, `EVAL_LIMIT=200`,
    `MAX_LENGTH=256`, `MAX_NEW_TOKENS=32`,
    `CUDA_VISIBLE_DEVICES=0,1`, `PRODUCER_DEVICE=cuda:0`,
    `CONSUMER_DEVICE=cuda:1`.
  - GPU check before launch: GPU0/1/3 were idle; GPU2 was occupied by another
    process using about `20 GiB`, so the run avoided GPU2.
  - Latest successful status check: script PID `3619765` with Python child
    PID `3619791`; GPU0 used about `14.2 GiB`, GPU1 about `6.0 GiB`.
  - Subsequent SSH checks were closed by the remote/proxy before a summary
    could be copied back.
- Verification:
  - `PYTHONPATH=src python -m unittest tests.test_neighboring_baseline_table_script -v`
    passed.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    (`44` tests).
  - `python scripts/check_release_readiness.py` passed.
  - `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`
    rebuilt the release preview and its readiness check passed.
- Next action:
  - When SSH access to `ser` recovers, check the SCD256 summary, regenerate
    `neighboring_baselines.{csv,md,tex}` using the SCD256 summary, copy the
    aligned table locally, and rerun release readiness.

## 2026-06-11 - Neighboring-baseline wrapper protocol-safety fix

- Purpose:
  - Prevent future current-runtime neighboring-baseline reruns from silently
    producing a mixed-protocol appendix table.
- Change:
  - Updated
    `Experiment/core_code/runtime_repo/scripts/run_ser_neighboring_baselines_cmrc.sh`
    so `SCD_SUMMARY` has no fixed default.
  - If `SCD_SUMMARY` is unset, the wrapper still runs SVD-LLM-style,
    TensorLLM-style, and SliceGPT-style adapted baselines, but writes a
    pending-table note to `logs/build_table.log` instead of generating
    `neighboring_baselines.{csv,md,tex}`.
  - Added a release-readiness guard in
    `scripts/check_release_readiness.py` requiring this protocol-safety
    behavior and rejecting a fixed prior-run `SCD_SUMMARY` default.
  - Updated `REPRODUCING_SCD_ICML.md` and
    `OPEN_SOURCE_RELEASE_MANIFEST.md` to require a same-protocol SCD summary
    for appendix table consolidation.
- Verification:
  - `python scripts/check_release_readiness.py` passed.
  - `PYTHONPATH=src python -m unittest tests.test_neighboring_baseline_table_script -v`
    passed.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    (`44` tests).
  - `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`
    rebuilt the release preview and its readiness check passed.
- Remote status:
  - `ssh ser` still closes via the remote/proxy before command execution, so
    the latest wrapper safety change and docs have not yet been synced to the
    remote runtime repo.

## 2026-06-11 - Neighboring-baseline table hard protocol gate

- Purpose:
  - Make the appendix neighboring-baseline table builder fail closed for
    paper-facing consolidation when source protocols disagree.
- Change:
  - Added `--require-compatible-protocol` to
    `Experiment/core_code/runtime_repo/scripts/build_neighboring_baseline_table.py`.
  - With the flag enabled, any protocol warning, including mismatched
    `max_length` or a large Full-consumer F1 discrepancy across sources,
    exits nonzero before writing table outputs.
  - Updated `run_ser_neighboring_baselines_cmrc.sh` to pass this flag whenever
    `SCD_SUMMARY` is provided.
  - Updated `REPRODUCING_SCD_ICML.md`,
    `OPEN_SOURCE_RELEASE_MANIFEST.md`, and the release-readiness guard to make
    this hard gate part of the release path.
- Verification:
  - `PYTHONPATH=src python -m unittest tests.test_neighboring_baseline_table_script -v`
    passed with the new mismatch-failure test.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    (`45` tests).
  - `python scripts/check_release_readiness.py` passed.
  - `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`
    rebuilt the release preview and its readiness check passed.
- Remote status:
  - `ssh ser` still closes before command execution, so the hard-gate update
    remains local until `ser` SSH recovers.

## 2026-06-11 - Neighboring-baseline CMRC200 max_length1024 rerun completed

- Purpose:
  - Restore the neighboring-baseline appendix as a current-runtime,
    same-protocol numeric table that matches the main SCD selected-route
    protocol closely enough for paper-facing trend claims.
- Why max_length1024 was needed:
  - The first full adapted-baseline rerun used `max_length=256`.
  - A same-protocol SCD256 table was generated successfully with empty
    protocol warnings, but its Full Consumer baseline was weak
    (Full F1 `0.072135`) and SCD retention was only `81.0%`, so it was not a
    suitable replacement for the paper-style `95%+` neighboring-baseline claim.
- Execution:
  - Synced the latest hard-gate scripts and docs to the remote runtime repo:
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote release readiness passed.
  - Remote targeted neighboring-table tests passed.
  - Launched a sequential GPU0/1 rerun to avoid GPU2, which was occupied by
    another process:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_maxlen1024_20260611_001`.
  - Ran SVD-LLM-style, TensorLLM-style, then SliceGPT-style with
    `LIMIT=200`, `MAX_LENGTH=1024`, `MAX_NEW_TOKENS=32`,
    `BANDWIDTH_GBPS=200`.
  - Consolidated with selected Reuse19/Patch13 SCD summary using
    `build_neighboring_baseline_table.py --require-compatible-protocol`.
- Outputs:
  - Local copy:
    `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/`.
  - Final table:
    `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines.md`.
  - Manifest:
    `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines_manifest.json`.
- Results:
  - Manifest `status=ok`, `warnings=[]`.
  - SVD-LLM-style: quality retention `2.4%`, TTFT speedup `0.30x`, F1
    `0.0158`, TTFT `241.35 ms`.
  - SliceGPT-style adapted KV transfer: retention `2.8%`, speedup `1.43x`,
    F1 `0.0186`, TTFT `51.45 ms`.
  - TensorLLM-style adapted KV transfer: retention `0.3%`, speedup `0.08x`,
    F1 `0.0016`, TTFT `955.31 ms`.
  - SCD: retention `97.8%`, speedup `2.47x`, F1 `0.6428`, TTFT `40.80 ms`.
- Conclusion:
  - The neighboring-baseline appendix now has a current-runtime,
    same-protocol non-Qwen reproduction with trend and magnitude aligned to the
    paper claim: adapted neighboring compression collapses quality, while SCD
    keeps high quality and improves TTFT.
## 2026-06-11 - Neighboring-baseline appendix wired into paper and final bundle

- Purpose: move the now-reproduced neighboring-baseline appendix table from
  result-only evidence into the current paper source and checked paper artifact
  bundle, and close the remaining patch-only anchor table gap.
- Inputs:
  - `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines.{csv,md,tex}`.
  - `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines_manifest.json`.
  - `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`.
  - `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/tables/patch_anchor_table.{csv,md,tex}`.
- Method:
  - Replaced the prior safe-paper neighboring-baseline exclusion paragraph with
    Table `neighboring_baselines`, using the protocol-aligned CMRC200 rerun.
  - Added Table `patchonly_anchors` to the current paper source so Full,
    Patch-only, and Reuse-only anchors across CMRC2018, HotpotQA-compressed,
    and WikiText-2 are represented in the paper text, not only the bundle.
  - Added `--neighboring-baselines-dir` to the paper bundle builder/finalizer
    path so the appendix table CSV/Markdown/LaTeX and manifest are copied into
    the final bundle.
  - Updated final paper artifact checks to require Table
    `neighboring_baselines`, Table `patchonly_anchors`, and forbid the old
    unsupported-neighboring exclusion text.
- Outputs:
  - Updated paper source:
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`.
  - Recompiled paper PDF:
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`
    (`886613` bytes, 16 pages).
  - Final checked bundle:
    `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001/`.
  - Bundle manifest copied `51` files with `missing=0`; paper artifact check
    report has `errors=[]`.
- Verification:
  - `PYTHONPATH=src python -m unittest tests.test_paper_artifact_bundle_script tests.test_finalize_full7405_bundle_script tests.test_paper_artifact_check_script -v` passed.
  - `PYTHONPATH=src python -m unittest discover -s tests -v` passed locally
    with 45 tests.
  - `python scripts/check_release_readiness.py` passed locally.
  - `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force` passed locally and the generated release preview passed readiness.
  - `pdflatex/bibtex/pdflatex/pdflatex` passed in
    `Publication/paper/current_safe_build`; final log has no undefined
    references.
  - Synced the updated runtime scripts/tests/docs to `ser` at
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `python scripts/check_release_readiness.py` passed.
  - Remote `PYTHONPATH=src python -m unittest tests.test_paper_artifact_bundle_script tests.test_finalize_full7405_bundle_script tests.test_paper_artifact_check_script -v` passed.
- Conclusion:
  - The neighboring-baseline appendix is now part of the checked paper-facing
    artifact set, not only a separate result directory.
  - The patch-only/Reuse-only/Full anchor table is now also part of the checked
    paper-facing artifact set through Table `patchonly_anchors`.
  - A final requirement-by-requirement completion audit was added at
    `docs/scd-icml-final-nonqwen-completion-audit-2026-06-11.md`.

## 2026-06-11 - Payload-bytes bandwidth sensitivity simulation

- Purpose: close the bandwidth-sensitivity gap where mixed reuse+patch routes
  were omitted from `bandwidth_sensitivity` because `payload_mb` was blank.
- Method:
  - Extended `build_scd_icml_result_tables.py` to infer mixed-route payload
    from same-dataset Patch-only and Reuse-only anchors:
    `mixed_payload = patch_n * patch_only_payload / total_layers +
    reuse_n * reuse_only_payload / total_layers`.
  - Added explicit `payload_bytes` and `payload_source` fields to
    latency-quality and bandwidth-sensitivity artifacts.
  - Reconstructed minimal local JSON inputs from the existing checked
    `patch_anchor_table.csv` and `mixed_route_boundary_table.csv` because the
    original manifest points to ser absolute paths not mounted in this Windows
    workspace.
- Key selected-route payload estimates:
  - CMRC2018 Reuse19/Patch13: `103.421 MB`, `108444778` bytes; simulated TTFT
    at `10/25/50/100/200 Gbps` is `116.531/66.889/50.342/42.068/37.931 ms`.
  - HotpotQA-compressed full7405 Reuse06/Patch26: `78.343 MB`, `82148590`
    bytes; simulated TTFT at `10/25/50/100/200 Gbps` is
    `103.316/65.711/53.176/46.909/43.775 ms`.
  - WikiText-2 Reuse20/Patch12: `22.311 MB`, `23394779` bytes; simulated TTFT
    at `10/25/50/100/200 Gbps` is `47.454/36.745/33.175/31.390/30.498 ms`.
- Outputs:
  - Reconstructed local inputs:
    `Experiment/results/payload_bandwidth_reconstructed_inputs_20260611_001/`.
  - New result tables and SVG:
    `Experiment/results/scd_icml_result_tables_payload_bytes_20260611_001/`.
  - New checked paper artifact bundle:
    `Experiment/results/paper_artifact_bundle_final_payload_bytes_checked_20260611_001/`.
  - `bandwidth_sensitivity.csv` now has `145` rows, including `115` mixed-route
    rows.
- Verification:
  - `D:\python\python.exe -m unittest Experiment.core_code.runtime_repo.tests.test_result_tables_script -v` passed.
  - `D:\python\python.exe scripts/check_paper_artifacts.py ...` passed for the
    new payload-bytes bundle.
  - `D:\python\python.exe Experiment/core_code/runtime_repo/scripts/check_release_readiness.py` passed.
  - `$env:PYTHONPATH="src;."; D:\python\python.exe -m unittest discover -s tests -v`
    passed in `Experiment/core_code/runtime_repo` with 45 tests.
- Conclusion:
  - The bandwidth-sensitivity artifact now simulates the actual selected and
    boundary SCD reuse+patch routes using payload bytes, instead of only the
    Patch-only/Reuse-only anchors.

## 2026-06-14 - Qwen32B A800 guarded preflight

- Experiment id: `qwen32_a800_preflight_20260614_001`
- Purpose: prepare the final-paper Qwen-32B reproduction path and verify that
  the runner refuses to launch unless the model pair, datasets, and GPU
  availability are suitable.
- Method:
  - Added Qwen2.5-32B base-to-instruct config templates and guarded A800
    wrapper scripts in `Experiment/core_code/runtime_repo`.
  - Copied the runtime subset to A800 at
    `/home/student_t/maqianli/SCD-ICML/qwen32_runtime_20260614/runtime_repo`.
  - Ran static validation locally and remotely.
  - Ran a 60-second A800 GPU monitor preflight with `required_gpus=2`,
    `required_free_mb=70000`, and `sample_interval=10`.
- Commands:
  - Local static compile:
    `D:\python\python.exe -m py_compile scripts\preflight_qwen32_a800.py src\scd_repro\cli\train_patch.py`.
  - Remote static validation:
    `ssh A800 'cd /home/student_t/maqianli/SCD-ICML/qwen32_runtime_20260614/runtime_repo && bash -n scripts/run_a800_scd_icml_qwen32_repro.sh && python3 -m py_compile scripts/preflight_qwen32_a800.py src/scd_repro/cli/train_patch.py'`.
  - Remote preflight:
    `ssh A800 'cd /home/student_t/maqianli/SCD-ICML/qwen32_runtime_20260614/runtime_repo && python3 scripts/preflight_qwen32_a800.py --models-config configs/models_a800_qwen25_32b_base_to_instruct_template.json --data-root /home/student_t/maqianli/no4/data --datasets cmrc2018 --required-gpus 2 --required-free-mb 70000 --monitor-seconds 60 --sample-interval 10 --skip-autoconfig --out /home/student_t/maqianli/SCD-ICML/runs/qwen32_preflight_20260614_001/preflight.json --write-env /home/student_t/maqianli/SCD-ICML/runs/qwen32_preflight_20260614_001/selected_gpus.env'`.
  - Remote wrapper smoke test:
    `ssh A800 'cd /home/student_t/maqianli/SCD-ICML/qwen32_runtime_20260614/runtime_repo && CONDA_ENV=cache GPU_MONITOR_SECONDS=0 REQUIRED_FREE_MB=70000 bash scripts/run_a800_scd_icml_qwen32_repro.sh'`.
- Inputs:
  - Model config:
    `configs/models_a800_qwen25_32b_base_to_instruct_template.json`.
  - Data root: `/home/student_t/maqianli/no4/data`.
  - Dataset preflight: `cmrc2018`.
  - Environment: A800 host `gpu010`, conda environment `cache`.
- Results:
  - Static validation passed locally and remotely.
  - The wrapper activated the `cache` conda environment and installed the
    runtime package editably before preflight.
  - Preflight failed closed before any capture, training, or evaluation.
  - Blocking model checks:
    `model_a path does not exist: /home/student_t/maqianli/models/Qwen2.5-32B`;
    `model_b path does not exist: /home/student_t/maqianli/models/Qwen2.5-32B-Instruct`.
  - Blocking GPU check: no two GPUs simultaneously satisfied sustained zero
    utilization and at least 70 GB free memory during the monitored window.
  - CMRC2018 train/dev files were present under the configured data root.
- Analysis:
  - The current A800 workspace is ready to reject invalid Qwen-32B evidence, but
    it cannot yet produce Qwen-32B paper-row results.
  - The available `/home/cuihanshuai/nas/models/Qwen2.5-VL-32B-Instruct`
    checkpoint is a visual-language checkpoint and must not be used as the
    text CausalLM Qwen-32B pair for this claim.
  - No Qwen-32B quality, latency, or speedup number was generated by this run.
- Conclusion:
  - The correct next step is to materialize or point to the actual text
    Qwen2.5-32B base/instruct checkpoints, then rerun the guarded wrapper when
    at least two A800 GPUs remain idle with sufficient free memory.

## 2026-06-18 - Qwen3-1.7B Gray Full Strict Evaluation

- Experiment id:
  `extra_model_qwen17_gray_full_strict_20260617_223703`.
- Purpose:
  Add a strict no-fallback/no-recompute extra-model check for
  Qwen3-1.7B -> Qwen3-1.7B-gray on full evaluation datasets.
- Data split discipline:
  CMRC2018 train samples were used for calibration/training. Evaluation used
  CMRC2018 dev, HotpotQA dev, and WikiText-2 test.
- Route:
  `route_qwen17_gray_patch6_seed42.json`, with patch layers
  `[0, 22, 24, 25, 26, 27]` and reuse layers
  `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]`.
  The route has no fallback layers and no recompute layers.
- CMRC2018 dev (`3219` samples):
  Full Consumer F1/EM `0.748358 / 0.440509`; SCD reuse+patch F1/EM
  `0.782770 / 0.504815`; total speedup `0.950069`; TTFT speedup
  `0.591745`.
- HotpotQA dev (`7405` samples):
  Full Consumer F1/EM `0.033071 / 0.007292`; SCD reuse+patch F1/EM
  `0.038549 / 0.010533`; total speedup `0.801846`; TTFT speedup
  `0.267766`.
- WikiText-2 test (`763` samples):
  Full Consumer F1-like/EM `0.245261 / 0.000000`; SCD reuse+patch
  F1-like/EM `0.236718 / 0.000000`; total speedup `1.007005`; TTFT speedup
  `1.122838`.
- Interpretation:
  This is not release-facing speedup evidence. It is a strictness and risk
  result: QA F1/EM is preserved or improved, but QA latency is worse and
  NLL/PPL deteriorates. Only WikiText-2 shows a small speed win with a small
  quality drop.
- Artifacts:
  `Experiment/results/qwen17_gray_full_strict_microsharded_20260617/`.

## 2026-06-18 - A800 SCD-vLLM Bundle Doctor and Injection Smoke

- Experiment id:
  `a800_scd_vllm_bundle_doctor_smoke_20260618_024236`.
- Purpose:
  Validate the publishable plug-and-play SCD-vLLM path on A800 using the
  `cache` conda environment, without relying on a NAS virtual environment.
- Environment:
  Host `A800mql`; working directory
  `/home/c101-800-mql/maqianli/SCD-ICML`; environment
  `conda activate cache`; Python
  `/home/c101-800-mql/.conda/envs/cache/bin/python`; vLLM `0.18.0`.
- Method:
  Synced the current runtime package to
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_ecbb130`.
  Added and exercised `scd-vllm doctor --bundle-dir`, then validated two
  reconstructed SCD bundles and converted them to vLLM ExampleConnector
  storage. Ran a single-GPU vLLM smoke test for `patch20_reuse12` on GPU 0.
- Inputs:
  Consumer model path
  `/home/c101-800-mql/nas/mql/models/Mistral-7B-expanded-for-MistralLite`.
  Reconstructed bundles:
  `reuse19_patch13_cmrc200` and `patch20_reuse12`, both under
  `/home/c101-800-mql/nas/mql/SCD-ICML/runtime_repo_open_release_20260617/runs/scd_vllm/`.
- Results:
  - `reuse19_patch13`: `scd-vllm doctor` passed. Route has patch/reuse
    `13/19`, fallback `0`, recompute `0`; bundle manifest validates
    `32` layer files, `229` tokens, route hash match, payload bytes
    `29,889,280`; ExampleConnector conversion payload bytes `29,362,944`.
  - `patch20_reuse12`: `scd-vllm doctor` passed. Route has patch/reuse
    `20/12`, fallback `0`, recompute `0`; bundle manifest validates
    `32` layer files, `229` tokens, route hash match, payload bytes
    `29,889,280`; ExampleConnector conversion payload bytes `29,362,944`.
  - vLLM smoke for `patch20_reuse12`: `adapter_smoke_pass=True`,
    `scd_injection_pass=True`, `reconstructed_scd=True`,
    `loaded_generation_ok=True`, `bundle_token_match=True`,
    `same_token_ids=True`, aligned tokens `224`, layers `32`, payload bytes
    `29,362,944`.
  - Greedy token equality is `False` for both 1-token and n-token outputs.
    This is acceptable for reconstructed approximate SCD and is not treated as
    an adapter failure.
- Analysis:
  This run strengthens the open-source vLLM integration evidence: the package
  can validate existing SCD bundles, convert them into ExampleConnector
  storage, and execute a real vLLM cache-load smoke path under the corrected
  A800 `cache` environment. It does not replace dataset-level F1/EM/PPL
  quality evidence.
- Artifacts:
  `Experiment/results/a800_scd_vllm_bundle_doctor_smoke_20260618_024236/`.

## 2026-06-18 - A800 SCD-vLLM Raw-Bank End-to-End Smoke

- Experiment id:
  `a800_scd_vllm_raw_bank_run_20260618_025602`.
- Purpose:
  Validate the full plug-and-play path from raw CMRC-trained reuse/patch banks
  to a reconstructed SCD bundle, ExampleConnector storage conversion, and a
  real vLLM cache-load smoke run.
- Environment:
  Host `A800mql`; working directory
  `/home/c101-800-mql/maqianli/SCD-ICML`; environment
  `conda activate cache`; runtime
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_ecbb130`; vLLM
  `0.18.0`; `CUDA_VISIBLE_DEVICES=3,4`.
- Inputs:
  Producer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/MistralLite`;
  consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`;
  reuse bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`;
  patch bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`;
  route
  `scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
- Method:
  Ran `scd-vllm doctor` against the raw reuse and patch banks, exported a
  reconstructed strict reuse+patch bundle, converted it to vLLM
  ExampleConnector storage, then executed the plugin smoke pipeline on the
  consumer model.
- Results:
  - Raw-bank doctor passed with `failure_count=0`, `warning_count=0`.
  - Route strictness passed: `patch_count=13`, `reuse_count=19`,
    `fallback_count=0`, `recompute_count=0`, and `num_layers=32`.
  - Bundle manifest reports `bundle_type=reuse_patch`, `num_tokens=229`,
    `num_layers=32`, route hash
    `8c0ba4475f4c194cdfe1fc64e6b845b093306273a04168b2ca5c2aaebb0dad3f`.
  - ExampleConnector conversion reports `aligned_tokens=224` and
    `payload_bytes=29,362,944`.
  - vLLM plugin smoke reports `pass=True`, `adapter_smoke_pass=True`,
    `scd_injection_pass=True`, `reconstructed_scd=True`,
    `loaded_generation_ok=True`, `bundle_token_match=True`, and
    `same_token_ids=True`.
  - Greedy output equality remains `False` for 1-token and n-token checks.
    This is treated as an approximate-route quality risk, not as a vLLM
    adapter or cache-injection failure.
- Analysis:
  This run proves the current open-source plugin can consume raw trained
  reuse/patch banks and drive the complete export-convert-inject path under the
  corrected A800 `cache` conda environment. It is release-facing integration
  evidence, while dataset-level F1/EM/PPL evaluations remain the source of
  quality claims.
- Artifacts:
  `Experiment/results/a800_scd_vllm_raw_bank_run_20260618_025602/`.

## 2026-06-18 - A800 SCD-vLLM Raw-Bank E2E Smoke for patch20/reuse12

- Experiment id:
  `a800_scd_vllm_raw_bank_patch20_run_20260618_032217`.
- Purpose:
  Validate the current vLLM demo route from raw CMRC-trained reuse/patch banks
  through reconstructed bundle export, ExampleConnector storage conversion, and
  real vLLM cache-load smoke.
- Environment:
  Host `A800mql`; working directory
  `/home/c101-800-mql/maqianli/SCD-ICML`; environment
  `conda activate cache`; runtime
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_ecbb130`; vLLM
  `0.18.0`; `CUDA_VISIBLE_DEVICES=0,1`. GPUs 0 and 1 were selected only after
  a 60-second zero-utilization check.
- Inputs:
  Producer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/MistralLite`;
  consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`;
  reuse bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`;
  patch bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`;
  route
  `scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`.
- Method:
  Ran `scd-vllm doctor` against the raw reuse and patch banks, exported a
  reconstructed strict reuse+patch bundle, converted it to vLLM
  ExampleConnector storage, then executed the plugin smoke pipeline on the
  consumer model.
- Results:
  - Raw-bank doctor passed with `failure_count=0`, `warning_count=0`.
  - Route strictness passed: `patch_count=20`, `reuse_count=12`,
    `fallback_count=0`, `recompute_count=0`, and `num_layers=32`.
  - Bundle manifest reports `bundle_type=reuse_patch`, `num_tokens=229`,
    and `num_layers=32`.
  - ExampleConnector conversion reports `aligned_tokens=224` and
    `payload_bytes=29,362,944`.
  - vLLM plugin smoke reports `pass=True`, `adapter_smoke_pass=True`,
    `scd_injection_pass=True`, `reconstructed_scd=True`,
    `loaded_generation_ok=True`, `bundle_token_match=True`, and
    `same_token_ids=True`.
  - Greedy output equality passed for both checks:
    `greedy_1tok_match=True`, `greedy_ntok_match=True`.
- Analysis:
  This is the strongest current plug-and-play evidence for the open-source
  plugin path because it starts from raw trained reuse/patch banks and reaches
  deterministic vLLM injection agreement for the checked prompt. It complements
  dataset-level CMRC2018 and HotpotQA F1/EM evidence for the same
  `patch20_reuse12` route.
- Artifact note:
  The remote script exited non-zero after successful plugin execution because a
  collection step looked for the old file name `injection_metrics.json`. The
  actual file was `injection_smoke/metrics.json`; the collected artifact set
  was rebuilt with stable local names.
- Artifacts:
  `Experiment/results/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/`.

## 2026-06-18 - Full WikiText-2 Strict Evaluation for patch20/reuse12 and patch26/reuse6

- Experiment id:
  `wikitext2_patch20_patch26_full_current_20260618_034812`.
- Purpose:
  Complete the MistralLite -> Mistral-7B three-dataset evidence matrix by
  evaluating strict reuse+patch routes on full WikiText-2 test data.
- Environment:
  Host `ser`; runtime directory
  `/disk_n/c101-3-mql/projects/SCD-ICML/runtime_repo_codex_current_20260618`;
  environment activated through
  `/disk_n/c101-3-mql/venvs/A800/bin/activate`; GPUs `0,1,2,3` were sampled
  for 60 seconds with zero utilization before launch. `patch20_reuse12` used
  GPUs `0,1`; `patch26_reuse6` used GPUs `2,3`.
- Inputs:
  WikiText-2 data
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data/wikitext-2/wiki.test.txt`;
  reuse bank
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222/ckpt_reuse/layerwise_seed42`;
  patch bank
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222/ckpt_patch/native_alllayers_seed42`.
- Method:
  Ran `scripts/run_ser_scd_icml_eval_existing_route.sh` twice in parallel with
  `DATASETS=wikitext2`, `EVAL_LIMIT=0`,
  `configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`, and strict
  fixed routes. The route validator passed before launch for both routes.
- Results:
  - Both runs completed with exit code `0`.
  - Samples: `763` for both routes.
  - `patch20_reuse12`: Full NLL/PPL `1.983874 / 7.270854`; SCD NLL/PPL
    `2.381655 / 10.822796`; delta NLL `+0.397781`; TTFT speedup `1.741x`;
    total speedup `1.072x`; compression `0.615385`.
  - `patch26_reuse6`: Full NLL/PPL `1.983874 / 7.270854`; SCD NLL/PPL
    `2.206730 / 9.085953`; delta NLL `+0.222856`; TTFT speedup `1.642x`;
    total speedup `1.076x`; compression `0.551724`.
- Analysis:
  WikiText-2 uses NLL/PPL as the relevant language-modeling quality metrics.
  Both strict routes speed up TTFT but degrade PPL versus Full Consumer.
  `patch26_reuse6` is the better quality route on WikiText-2, while
  `patch20_reuse12` remains the faster and higher-Reuse vLLM demo route.
- Artifacts:
  `Experiment/results/wikitext2_patch20_patch26_full_current_20260618_034812/`.

## 2026-06-18 - A800 SCD-vLLM Raw-Bank E2E Smoke for patch26/reuse6

- Experiment id:
  `a800_scd_vllm_raw_bank_patch26_run_20260618_044521`.
- Purpose:
  Close the release-facing integration gap for the quality-first
  `patch26_reuse6` route by validating the same raw-bank-to-vLLM path already
  used by `patch20_reuse12`.
- Environment:
  Host `A800mql`; runtime
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_ecbb130`;
  activation `conda activate cache`; Python
  `/home/c101-800-mql/.conda/envs/cache/bin/python`; vLLM `0.18.0`;
  `CUDA_VISIBLE_DEVICES=0,1`.
- GPU selection:
  GPUs `0,1` were sampled for 60 seconds before launch; all six samples showed
  utilization `0`.
- Inputs:
  Producer `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/MistralLite`;
  consumer
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`;
  Reuse bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/layerwise_seed42`;
  Patch bank
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/banks/open_build_native_fixedroute_cmrc200_20260616_133222/native_alllayers_seed42`;
  route
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json`.
- Method:
  Ran `python -m scd_vllm_plugin.cli doctor` followed by
  `python -m scd_vllm_plugin.cli run` from raw trained Reuse/Patch banks,
  exporting a reconstructed `reuse_patch` bundle, converting it to
  ExampleConnector-compatible storage, and running the vLLM injection smoke.
- Outputs:
  Local artifact directory
  `Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`.
  Remote run root
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch26_run_20260618_044521`.
- Results:
  `doctor_pass=True`, `plugin_pass=True`, `adapter_smoke_pass=True`,
  `scd_injection_pass=True`, `reconstructed_scd=True`,
  `bundle_token_match=True`, `same_token_ids=True`,
  `greedy_1tok_match=True`, `greedy_ntok_match=True`,
  `aligned_tokens=224`, `payload_bytes=29362944`.
- Analysis:
  `patch26_reuse6` is now both the quality-first route for CMRC2018/WikiText-2
  and a raw-bank-to-vLLM validated route. `patch20_reuse12` remains the
  higher-Reuse/speed demo route.
- ResearchKB:
  Harvested the local result directory with
  `D:\ResearchKB\rk-harvest.cmd --project "KV Cache Reuse"`; 7 JSON records
  were recorded.

## 2026-06-18 - A800 cache Env Probe and vLLM Status Packaging Gate

- Experiment id:
  `a800_env_probe_mistrallite_mistral7_20260618`.
- Purpose:
  Record the exact A800mql `conda activate cache` runtime environment used for
  current vLLM evidence and bind that evidence into the release-facing status
  report and release preview gate.
- Environment:
  Host `A800mql`; activation `conda activate cache`; Python
  `/home/c101-800-mql/.conda/envs/cache/bin/python`; PyTorch `2.10.0+cu128`;
  Transformers `5.3.0`; vLLM `0.18.0`; CUDA available on 8 x
  `NVIDIA A800-SXM4-80GB`.
- Inputs:
  Producer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/MistralLite`;
  consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`;
  dtype `bfloat16`.
- Method:
  Ran `python -m scd_vllm.env_probe` on A800mql, copied the generated
  `env_probe.json` and Markdown status to the local release evidence tree, then
  refreshed `reports/scd_vllm_status.md` and release preview/readiness checks.
- Outputs:
  `Experiment/results/a800_env_probe_mistrallite_mistral7_20260618/env_probe.json`
  and `Experiment/results/a800_env_probe_mistrallite_mistral7_20260618/status.md`.
- Results:
  KV compatibility check passed with `compatible=True`; both producer and
  consumer report 32 layers, 32 attention heads, 8 KV heads, head dim 128, and
  hidden size 4096. vLLM `0.18.0` imports in the active cache environment.
- Analysis:
  The probe supports the current MistralLite to Mistral-7B vLLM smoke evidence
  boundary. It is environment and compatibility evidence, not a dataset-quality
  or throughput benchmark.
- ResearchKB:
  Recorded the experiment run as `run_c8494235f16247e7`; direct harvest found
  no additional parseable JSON experiment records beyond the explicit probe
  artifact.

## 2026-06-18 - vLLM Phase-Script Release Surface Verification

- Experiment id:
  `release_vllm_phase_script_surface_20260618`.
- Purpose:
  Verify that the release preview is self-contained for the documented vLLM
  integration phases, not only for the plugin Python modules.
- Environment:
  Windows local release tree verification in
  `Experiment/core_code/runtime_repo`; static checks only. GPU runtime behavior
  remains covered by the A800 `conda activate cache` smoke artifacts.
- Inputs:
  `scripts/scd_vllm/run_00_repo_audit.sh` through
  `scripts/scd_vllm/run_06_report.sh`, `configs/scd_vllm_mvp.yaml`,
  `reports/scd_vllm_repo_audit.md`, and `reports/scd_vllm_status.md`.
- Method:
  Expanded `build_release_tree.py` and `check_release_readiness.py` so the
  vLLM phase wrappers, MVP config, complete vLLM modules, and vLLM tests are
  part of the required release surface. Regenerated the repo audit after fixing
  stale `scd_injection.py` naming and filtering local cache files.
- Verification commands:
  `python scripts\check_release_readiness.py`;
  `python scripts\build_release_tree.py --out release_preview\scd_icml_runtime_release --force`;
  `Get-ChildItem release_preview\scd_icml_runtime_release\scripts\scd_vllm`;
  `$env:PYTHONPATH='src'; python -m pytest tests -q`.
- Results:
  Source readiness passed; release-preview readiness passed; all eight vLLM
  phase wrappers are present in the preview; `configs/scd_vllm_mvp.yaml` is
  present; preview `__pycache__` count is `0`; full tests report `94 passed`.
- Analysis:
  The release preview is now consistent with the vLLM status report's
  reproduction commands. This is a packaging and traceability check, not a new
  quality or latency benchmark.

## 2026-06-18 - A800 Qwen3-1.7B Plain vLLM Compatibility Baseline

- Experiment id:
  `a800_qwen17_vllm_baseline_20260618_0530`.
- Purpose:
  Add a second-model-family vLLM compatibility signal for the open-source SCD
  release without claiming Qwen SCD speedup or route quality.
- Environment:
  Host `A800mql`; activation `conda activate cache`; code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_f9f07b5`;
  Python `/home/c101-800-mql/.conda/envs/cache/bin/python`; vLLM `0.18.0`;
  `CUDA_VISIBLE_DEVICES=0`.
- GPU selection:
  GPUs `0,1,2,5` were sampled every 10 seconds for 60 seconds before launch;
  all samples reported utilization `0`.
- Inputs:
  Qwen3-1.7B model
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`;
  prompt `Summarize Semantic Cache Distillation in one sentence.`;
  dtype `bfloat16`; max model length `512`; max new tokens `16`;
  `enforce_eager=True`; `trust_remote_code=True`.
- Method:
  Ran `python -m scd_vllm.env_probe` with Qwen3-1.7B as both producer and
  consumer, then ran `python -m scd_vllm.vllm_baseline` as a plain official
  vLLM generation baseline. The final shell `find | sort` listing failed due
  to a CRLF quoting issue in the SSH stdin script after the metrics were
  already written; no GPU rerun was needed.
- Outputs:
  `Experiment/results/a800_qwen17_vllm_baseline_20260618_0530/`.
- Results:
  KV compatibility `True`; Qwen KV spec has `28` layers, `16` attention heads,
  `8` KV heads, and head dim `128`. Plain vLLM baseline loaded and generated
  successfully. Mean 16-token generation time was `0.207544s`; mean throughput
  was `77.092046` tokens/s; 1-token proxy time was `0.013509s`; prompt token
  count was `11`.
- Analysis:
  This supports multi-model vLLM runtime compatibility for the release, but it
  does not use SCD Reuse/Patch artifacts and must not be reported as a Qwen SCD
  route-quality or speedup result.
- ResearchKB:
  Harvest recorded `harvest_04e0bf84b52a3611`; structured run recorded
  `run_92cfa0ed515947c1`.

## 2026-06-18 - A800 Qwen3-1.7B Gray SCD-vLLM Plugin Smoke

- Experiment id:
  `a800_qwen17_scd_vllm_plugin_smoke_20260618_cache`.
- Purpose:
  Verify that the external SCD-vLLM adapter can export and load a reconstructed
  `reuse_patch` KV bundle for Qwen3-1.7B -> Qwen3-1.7B-gray, using the already
  trained strict Qwen reuse/patch banks.
- Environment:
  Host `A800mql`; activation `conda activate cache`; code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_f9f07b5`;
  Python `/home/c101-800-mql/.conda/envs/cache/bin/python`; PyTorch `2.10.0`;
  Transformers `5.3.0`; vLLM `0.18.0`; `CUDA_VISIBLE_DEVICES=0,1`.
- GPU selection:
  All A800 GPUs were sampled every 10 seconds for 60 seconds before launch.
  Every sample reported utilization `0`; GPUs `0,1` had the lowest resident
  memory among the available cards and were selected.
- Inputs:
  Producer model
  `/home/c101-800-mql/maqianli/2/models/Qwen/Qwen3-1___7B`; consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`; reuse
  bank `ckpt_reuse_layerwise_seed42`; patch bank
  `ckpt_patch_xnorm_alllayers_seed42`; route
  `route_qwen17_gray_patch6_seed42.json`.
- Method:
  Copied the gray model and compact reuse/patch artifacts from ser to the A800
  local project disk, ran `scd-vllm doctor --require-local-models`, then ran
  `scd-vllm run` with a 27-token Chinese QA-style prefix. The first smoke run
  failed because vLLM `ExampleConnector` does not support HMA; rerunning with
  `--disable-hybrid-kv-cache-manager` passed.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_doctor_20260618_cache/` and
  `Experiment/results/a800_qwen17_scd_vllm_plugin_smoke_20260618_cache/`.
- Results:
  Doctor passed with failure count `0`. Route strictness passed with
  `num_layers=28`, patch/reuse `6/22`, fallback/recompute `0/0`. Plugin smoke
  passed with `adapter_smoke_pass=True`, `scd_injection_pass=True`,
  `bundle_token_match=True`, `same_token_ids=True`, and
  `loaded_generation_ok=True`. Export payload bytes were `2,981,888`;
  ExampleConnector loaded `16` aligned tokens with payload bytes `1,837,472`.
- Analysis:
  This closes the earlier Qwen plugin-evidence gap and supports the adapter's
  portability to a second model family. It remains a smoke/integration result,
  not a Qwen speedup or full-quality claim.
- ResearchKB:
  Harvest recorded `harvest_c6fa1d1d15daaf1b`,
  `harvest_c11d19bd735cb841`, and `harvest_82d02c54a125e7ad`;
  structured run recorded `run_aec7378b4554436b`.

## 2026-06-18 - A800 Qwen3-1.7B Gray SCD-vLLM Batch Benchmark

- Experiment id:
  `a800_qwen17_scd_vllm_batchbench_20260618_cache`.
- Purpose:
  Validate the public `scd-vllm batch-benchmark` command and the batched
  repeated-prefix `ExampleConnector` load path for the Qwen3-1.7B ->
  Qwen3-1.7B-gray strict SCD bundle.
- Environment:
  Host `A800mql`; activation `conda activate cache`; code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_batchbench_20260618`;
  vLLM `0.18.0`; `CUDA_VISIBLE_DEVICES=0`.
- GPU selection:
  A800 GPUs `3,4,6,7` were sampled for 60 seconds and remained busy. GPUs
  `0,1,2,5` stayed at `0` utilization, so GPU `0` was selected.
- Inputs:
  Consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`; bundle
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/reuse_patch_qwen17_gray_cmrc_route`;
  storage
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/example_connector_storage`.
- Method:
  Ran `python -m scd_vllm_plugin.cli batch-benchmark` with
  `batch_size=128`, `num_batches=3`, `warmup_batches=1`,
  `max_new_tokens=16`, `dtype=bfloat16`, `max_model_len=256`,
  `gpu_memory_utilization=0.35`, `enforce_eager=True`, and
  `--disable-hybrid-kv-cache-manager`. The command measured a plain vLLM
  baseline stage and an SCD load stage in separate subprocesses.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_batchbench_20260618_cache/`.
- Results:
  `benchmark_pass=True`; `route_id=paper_scd_reuse_patch_k6`; patch/reuse
  `6/22`; expected requests `384`; `load_generation_ok=True`;
  `bundle_token_match=True`; `storage_ready=True`. SCD load throughput was
  `2000.185735` tokens/s and `125.011608` requests/s. Plain vLLM baseline
  throughput was `6514.906148` tokens/s and `407.181634` requests/s. The
  load/baseline ratio was `0.307017`.
- Analysis:
  The batch path functions and reports positive throughput, but the current
  disk-backed `ExampleConnector` path is slower than plain vLLM in this
  repeated-prefix microbenchmark. Treat this as integration and overhead-risk
  evidence, not as a Qwen speedup result.
- ResearchKB:
  Harvest recorded `harvest_b686e6617ed38a60`; structured run recorded
  `run_46116bf79b924d40`.

## 2026-06-18 - A800 Qwen3-1.7B Gray SCD-vLLM Prefix-Cache Control Benchmark

- Experiment id:
  `a800_qwen17_scd_vllm_prefixbench_20260618_cache`.
- Purpose:
  Add the missing vLLM native prefix-cache control for the repeated-prefix
  Qwen SCD-vLLM benchmark, so release evidence distinguishes plain vLLM,
  official vLLM prefix caching, and SCD `ExampleConnector` loading.
- Environment:
  Host `A800mql`; activation `conda activate cache`; code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_prefixbench_20260618_0616`;
  vLLM `0.18.0`; `CUDA_VISIBLE_DEVICES=7`.
- GPU selection:
  A800 GPUs `3,4,6,7` were sampled for 60 seconds and stayed at utilization
  `0`; GPU `7` had the lowest resident memory among those cards and was used.
- Inputs:
  Consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`; bundle
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/reuse_patch_qwen17_gray_cmrc_route`;
  storage
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/example_connector_storage`.
- Method:
  Ran `python -m scd_vllm_plugin.cli batch-benchmark` with
  `--include-prefix-cache-baseline`, `batch_size=128`, `num_batches=3`,
  `warmup_batches=1`, `max_new_tokens=16`, `dtype=bfloat16`,
  `max_model_len=256`, `gpu_memory_utilization=0.35`, `enforce_eager=True`,
  and `--disable-hybrid-kv-cache-manager`. The command measured plain vLLM,
  vLLM prefix caching, and SCD load stages in separate subprocesses.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_prefixbench_20260618_cache/`.
- Results:
  `benchmark_pass=True`; `route_id=paper_scd_reuse_patch_k6`; patch/reuse
  `6/22`; expected requests `384`; `load_generation_ok=True`;
  `prefix_cache_generation_ok=True`; `bundle_token_match=True`;
  `storage_ready=True`. Plain vLLM throughput was `6189.640815` tokens/s;
  prefix-cache vLLM throughput was `7016.957157` tokens/s; SCD load throughput
  was `1968.496077` tokens/s. Prefix-cache/plain ratio was `1.133661`;
  SCD load/plain ratio was `0.318031`; SCD load/prefix-cache ratio was
  `0.280534`.
- Analysis:
  Prefix caching is the stronger built-in baseline for this repeated-prefix
  workload and outperforms plain vLLM. The current disk-backed SCD load path is
  much slower than both vLLM baselines. This result makes the release boundary
  sharper: the current plugin is functional integration evidence, not a
  throughput-improving serving connector.
- ResearchKB:
  Harvest recorded `harvest_d7b3f06d675c5b85`; structured run recorded
  `run_695cfcac2dbc4528`.

## 2026-06-18 - A800 Qwen3-1.7B Gray SCDPreloadedConnector Benchmark

- Experiment id:
  `a800_qwen17_scd_vllm_preloadedbench_20260618_cache`.
- Purpose:
  Test the new repository-provided external vLLM connector
  `SCDPreloadedConnector`, which uses vLLM's `kv_connector_module_path`
  mechanism and caches loaded SCD KV tensors by filename inside each worker.
- Environment:
  Host `A800mql`; activation `conda activate cache`; code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_preloaded_20260618_0634`;
  vLLM `0.18.0`; `CUDA_VISIBLE_DEVICES=7`.
- GPU selection:
  A800 GPUs `3,4,6,7` were sampled for 60 seconds and stayed at utilization
  `0`; GPU `7` had the lowest resident memory among those cards and was used.
- Inputs:
  Consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`; bundle
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/reuse_patch_qwen17_gray_cmrc_route`;
  storage
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/example_connector_storage`.
- Method:
  Ran `python -m scd_vllm_plugin.cli batch-benchmark` with
  `--include-prefix-cache-baseline`, `--kv-connector SCDPreloadedConnector`,
  `--kv-connector-module-path scd_vllm.preloaded_connector`,
  `--connector-cache-device cuda`, `batch_size=128`, `num_batches=3`,
  `warmup_batches=1`, `max_new_tokens=16`, `dtype=bfloat16`,
  `max_model_len=256`, `gpu_memory_utilization=0.35`, `enforce_eager=True`,
  and `--disable-hybrid-kv-cache-manager`.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_preloadedbench_20260618_cache/`.
- Results:
  `benchmark_pass=True`; `kv_connector=SCDPreloadedConnector`;
  `kv_connector_module_path=scd_vllm.preloaded_connector`;
  `route_id=paper_scd_reuse_patch_k6`; patch/reuse `6/22`; expected requests
  `384`; `load_generation_ok=True`; `prefix_cache_generation_ok=True`;
  `bundle_token_match=True`; `storage_ready=True`. Plain vLLM throughput was
  `6194.909625` tokens/s; vLLM prefix-cache throughput was `6786.831734`
  tokens/s; SCDPreloaded load throughput was `4294.997427` tokens/s.
  Prefix-cache/plain ratio was `1.095550`; SCDPreloaded/plain ratio was
  `0.693311`; SCDPreloaded/prefix-cache ratio was `0.632843`.
- Analysis:
  The external connector mechanism works from the repository without modifying
  the installed vLLM package. Caching loaded tensors materially improves the
  SCD load path relative to the previous disk-backed ExampleConnector run
  (`1968.496077` tokens/s), but it remains below both vLLM baselines. The
  release can claim an experimental external connector prototype and measured
  overhead reduction, not a throughput speedup.
- ResearchKB:
  Harvest recorded `harvest_7c7032d9feccb303`; structured run recorded
  `run_ce95bb2097b74b55`.

## 2026-06-18 - A800 Mistral SCDPreloadedConnector Batch Benchmarks

- Experiment ids:
  `a800_mistral_patch20_scd_vllm_preloadedbench_20260618_cache` and
  `a800_mistral_patch26_scd_vllm_preloadedbench_20260618_cache`.
- Purpose:
  Validate the repository-provided external `SCDPreloadedConnector` on the main
  release-facing MistralLite -> Mistral-7B routes, not only on the auxiliary
  Qwen route. The benchmark compares plain vLLM, official vLLM prefix caching,
  and SCDPreloaded load under the same repeated-prefix trusted batch shape.
- Environment:
  Host `A800mql`; activation
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`;
  code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_preloaded_20260618_0634`;
  vLLM `0.18.0`.
- GPU selection:
  A800 GPUs `0` and `1` were sampled three times over 40 seconds and stayed at
  utilization `0`. `patch20_reuse12` used `CUDA_VISIBLE_DEVICES=0`;
  `patch26_reuse6` used `CUDA_VISIBLE_DEVICES=1`.
- Inputs:
  Consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`.
  `patch20_reuse12` bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/plugin_run_patch20_reuse12/`.
  `patch26_reuse6` bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/plugin_run_patch26_reuse6/`.
- Method:
  Ran `python -m scd_vllm_plugin.cli batch-benchmark` with
  `--include-prefix-cache-baseline`, `--kv-connector SCDPreloadedConnector`,
  `--kv-connector-module-path scd_vllm.preloaded_connector`,
  `--connector-cache-device cuda`, `batch_size=128`, `num_batches=3`,
  `warmup_batches=1`, `max_new_tokens=16`, `dtype=bfloat16`,
  `max_model_len=512`, `gpu_memory_utilization=0.35`, `enforce_eager=True`,
  and `--disable-hybrid-kv-cache-manager`.
- Outputs:
  `Experiment/results/a800_mistral_patch20_scd_vllm_preloadedbench_20260618_cache/`
  and
  `Experiment/results/a800_mistral_patch26_scd_vllm_preloadedbench_20260618_cache/`.
- Results:
  Both runs reported `benchmark_pass=True`,
  `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `load_generation_ok=True`, `prefix_cache_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
  `patch20_reuse12`: plain vLLM `953.216251` tok/s, vLLM prefix-cache
  `4083.576373` tok/s, SCDPreloaded `2870.665243` tok/s,
  SCD/plain `3.011557`, SCD/prefix `0.702978`.
  `patch26_reuse6`: plain vLLM `963.863263` tok/s, vLLM prefix-cache
  `4327.769903` tok/s, SCDPreloaded `2823.938822` tok/s,
  SCD/plain `2.929813`, SCD/prefix `0.652516`.
- Analysis:
  This is now the main Mistral vLLM connector benchmark gate. It supports a
  plug-in external connector and a repeated-prefix throughput gain over plain
  vLLM recomputation. It must still be framed below native vLLM prefix caching
  for static repeated prefixes, so it is not a general serving-speed claim.
- ResearchKB:
  Harvest recorded `harvest_c45b6b2452fec2a4`,
  `harvest_ab0ff9e6fc68db88`, `harvest_0a9274a41472e576`, and
  `harvest_24d53dcae704e4ae`; structured runs recorded
  `run_f8e2ff751fbc48cc` and `run_e51eba0f3920471b`.

## 2026-06-18 - A800 Mistral SCDPreloadedConnector Cache Stats Benchmarks

- Experiment ids:
  `a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache` and
  `a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache`.
- Purpose:
  Strengthen the Mistral SCD-vLLM benchmark from timing-only evidence to
  direct worker-side cache-reuse evidence. Also verify that the benchmark code
  can run `SCDPreloadedConnector` without requiring users to manually pass the
  vLLM HMA-disable flag.
- Environment:
  Host `A800mql`; activation
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`;
  code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_stats_20260618_0717`;
  vLLM `0.18.0`.
- GPU selection:
  A800 GPUs `0` and `1` were sampled for 60 seconds and stayed at utilization
  `0`. Both had resident low-priority python server memory, but enough free
  memory for the Mistral run. `patch20_reuse12` used
  `CUDA_VISIBLE_DEVICES=0`; `patch26_reuse6` used `CUDA_VISIBLE_DEVICES=1`.
- Inputs:
  Consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`.
  `patch20_reuse12` bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch20_run_20260618_032217/plugin_run_patch20_reuse12/`.
  `patch26_reuse6` bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/plugin_run_patch26_reuse6/`.
- Method:
  First ran the full batch benchmark with the new stats-enabled connector. The
  load stage failed with vLLM `ValueError: Connector SCDPreloadedConnector
  does not support HMA but HMA is enabled`. Updated the code so the load stage
  automatically disables `disable_hybrid_kv_cache_manager` whenever
  `SCDPreloadedConnector` is selected. Then reran only the failed load stages
  and merged the already completed baseline/prefix-cache records with the new
  load and stats records.
- Outputs:
  `Experiment/results/a800_mistral_patch20_scd_vllm_preloadedstats_20260618_cache/`
  and
  `Experiment/results/a800_mistral_patch26_scd_vllm_preloadedstats_20260618_cache/`.
- Results:
  Both runs reported `benchmark_pass=True`,
  `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `load_generation_ok=True`, `prefix_cache_generation_ok=True`,
  `bundle_token_match=True`, `storage_ready=True`,
  `preloaded_connector_worker_count=1`, `cache_hits=16352`,
  `cache_misses=32`, `cache_events=16384`, `cache_hit_rate=0.998047`,
  `cached_files=32`, and `cached_bytes=29360128`.
  `patch20_reuse12`: plain vLLM `969.093202` tok/s, vLLM prefix-cache
  `4103.775535` tok/s, SCDPreloaded `2773.109399` tok/s,
  SCD/plain `2.861551`, SCD/prefix `0.675746`.
  `patch26_reuse6`: plain vLLM `988.035039` tok/s, vLLM prefix-cache
  `4233.686187` tok/s, SCDPreloaded `2657.446443` tok/s,
  SCD/plain `2.689628`, SCD/prefix `0.627691`.
- Analysis:
  The new stats confirm actual connector-side tensor reuse after the first
  per-layer load: `32` layer files are loaded once, then reused for the
  repeated-prefix batches. The evidence supports plug-in integration,
  cache-reuse behavior, and a repeated-prefix gain over plain vLLM
  recomputation. It still must be framed below native vLLM prefix caching and
  is not production multi-tenant serving evidence.
- ResearchKB:
  Harvest recorded `harvest_d5bbf3b43d13f3f7`,
  `harvest_1d60c78b5ce98e8b`, `harvest_106d8c86e09bb862`, and
  `harvest_af30d0edb4c7fa8a`; structured runs recorded
  `run_1e09f65397f648ab` and `run_bdaec275945942e6`.

## 2026-06-18 - SCD-vLLM Official Launch Spec API

- Experiment id:
  `local_scd_vllm_launch_spec_api_20260618`.
- Purpose:
  Convert the working external connector path into a more plug-and-play
  integration contract for official vLLM, without forking vLLM or expanding
  the claim to production multi-tenant serving.
- Method:
  Added `build_preloaded_launch_spec` and `scd-vllm launch-spec`. The launch
  spec reads the SCD bundle manifest, checks ExampleConnector-compatible
  storage, emits `vllm.LLM(**llm_kwargs)` inputs with
  `SCDPreloadedConnector`, automatically sets
  `disable_hybrid_kv_cache_manager=True` for the preloaded connector, and
  optionally emits `SCD_PRELOADED_STATS_DIR`.
- Inputs:
  Unit-test synthetic `reuse_patch` bundle manifest and dummy
  `.safetensors` storage artifact. No GPU was required.
- Outputs:
  Updated `src/scd_vllm_plugin/engine.py`, `src/scd_vllm_plugin/cli.py`,
  `src/scd_vllm_plugin/__init__.py`, `docs/scd_vllm_plugin_mvp.md`,
  `OPEN_SOURCE_RELEASE_MANIFEST.md`, and `reports/scd_vllm_status.md`.
- Results:
  Targeted tests passed:
  `PYTHONPATH=.;src pytest tests/test_scd_vllm_plugin.py tests/test_scd_vllm_batch_benchmark.py -q`
  reported `12 passed`. Release readiness also passed.
- Analysis:
  This does not add new throughput evidence, but it improves publishability:
  users can now generate an auditable JSON launch contract for official vLLM
  instead of copying benchmark internals. The artifact remains explicitly
  single-tenant and trusted-prefix only.
- ResearchKB:
  Structured run recorded `run_59ff8dacd10f4ffa`.

## 2026-06-18 - A800 Official vLLM Launch-Smoke for Mistral Patch26/Reuse6

- Experiment id:
  `a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache`.
- Purpose:
  Verify that the `scd-vllm launch-spec` JSON is not only inspectable but can
  be consumed by official vLLM through `vllm.LLM(**llm_kwargs)` with the
  external `SCDPreloadedConnector`.
- Environment:
  Host `A800`; activation
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`;
  code snapshot
  `/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_launchsmoke_20260618_081014`;
  vLLM `0.18.0`. The runtime snapshot was placed under the account home
  directory, not NAS.
- GPU selection:
  GPUs `3/4/6/7` became busy with unrelated workloads and were not used. GPUs
  `0/1/2` were sampled for 60 seconds and stayed at utilization `0`; the final
  run used `CUDA_VISIBLE_DEVICES=2`.
- Inputs:
  Consumer model
  `/home/c101-800-mql/nas/mql/scd_icml_ser_assets/models/Mistral-7B-v0.1`.
  The SCD bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/plugin_run_patch26_reuse6/`.
- Method:
  Generated a launch spec with `scd-vllm launch-spec` using
  `SCDPreloadedConnector`, `max_model_len=512`, and
  `gpu_memory_utilization=0.35`. Then ran `scd-vllm launch-smoke` with
  `batch_size=8`, `warmup_batches=1`, and `max_new_tokens=16`. During the
  first attempt, stats embedded in `launch_smoke.json` were collected before
  the vLLM EngineCore child exited. The CLI was updated to wrap the vLLM smoke
  in a child process and recollect final connector stats in the parent.
- Outputs:
  `Experiment/results/a800_mistral_patch26_scd_vllm_launchsmoke_20260618_cache/`
  containing `launch_spec.json`, `launch_smoke.json`, stdout/stderr logs, and
  `preloaded_connector_stats/scd_preloaded_connector_546561.json`.
- Results:
  `launch_smoke_pass=True`, `generation_ok=True`,
  `bundle_token_match=True`, `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `disable_hybrid_kv_cache_manager=True`, `request_count=8`,
  `output_tokens=128`, `prompt_token_count=229`,
  `tokens_per_second=596.609039`, and `requests_per_second=37.288065`.
  Final connector stats: `cache_hits=480`, `cache_misses=32`,
  `cache_events=512`, `cache_hit_rate=0.9375`, `cached_files=32`,
  `cached_bytes=29360128`, and `final_workers=1`.
- Analysis:
  The run closes the integration-evidence gap between a static launch spec and
  actual official-vLLM execution. It should be cited as trusted static-prefix
  plugin evidence only, not as a dataset-quality result or production serving
  benchmark.
- ResearchKB:
  Harvest did not parse this result directory automatically; structured run
  recorded `run_8e4128a714704850`.

## 2026-06-18 - A800 Official vLLM Launch-Smoke for Qwen Patch6/Reuse22

- Experiment id:
  `a800_qwen17_scd_vllm_launchsmoke_20260618_cache`.
- Purpose:
  Extend official-vLLM launch-spec consumption evidence from Mistral to the
  second model family already present in the release evidence package:
  Qwen3-1.7B -> Qwen3-1.7B-gray with the `paper_scd_reuse_patch_k6` route.
- Environment:
  Host `A800`; activation
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`;
  code snapshot
  `/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_launchsmoke_20260618_081014`;
  vLLM `0.18.0`. The runtime snapshot and output directory were placed under
  `/home/student_t`, not NAS.
- GPU selection:
  A800 GPUs `3/4/6/7` were sampled for 60 seconds and stayed at utilization
  `0`; the final run used `CUDA_VISIBLE_DEVICES=7`.
- Inputs:
  Consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`.
  The SCD bundle/storage came from
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/`.
- Method:
  Generated a launch spec with `scd-vllm launch-spec` using
  `SCDPreloadedConnector`, `max_model_len=512`, and
  `gpu_memory_utilization=0.35`. Then ran `scd-vllm launch-smoke` with
  `batch_size=128`, `warmup_batches=1`, and `max_new_tokens=16`.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_launchsmoke_20260618_cache/`
  containing `launch_spec.json`, `launch_smoke.json`, stdout/stderr logs, and
  `preloaded_connector_stats/scd_preloaded_connector_568038.json`.
- Results:
  `launch_smoke_pass=True`, `generation_ok=True`,
  `bundle_token_match=True`, `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `disable_hybrid_kv_cache_manager=True`, `request_count=128`,
  `output_tokens=2048`, `prompt_token_count=27`,
  `tokens_per_second=6291.929930`, and `requests_per_second=393.245621`.
  Final connector stats: `cache_hits=7140`, `cache_misses=28`,
  `cache_events=7168`, `cache_hit_rate=0.996094`, `cached_files=28`,
  `cached_bytes=1835008`, and `final_workers=1`.
- Analysis:
  The run confirms that the plug-in launch contract is not Mistral-specific.
  It should still be reported as trusted static-prefix integration evidence,
  not as production serving or dataset-quality evidence.
- ResearchKB:
  Structured run recorded `run_9f2f4f360c20424c`.

## 2026-06-18 - A800mql Qwen Preloaded Connector Stats Batch Benchmark

- Experiment id:
  `a800_qwen17_scd_vllm_preloadedstats_20260618_cache`.
- Purpose:
  Replace the older Qwen batched SCD-vLLM benchmark evidence with an artifact
  that includes final worker-side `SCDPreloadedConnector` cache stats.
- Environment:
  Host `A800mql`; activation
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`;
  code snapshot
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_stats_20260618_0717`;
  vLLM `0.18.0`. The run used existing model and SCD artifact directories and
  did not create a virtual environment on NAS.
- GPU selection:
  GPUs `3/4/6/7` were busy with other workloads. GPUs `0/1/2` were sampled
  for 60 seconds and stayed at utilization `0`; the final run used
  `CUDA_VISIBLE_DEVICES=2`.
- Inputs:
  Consumer model
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`.
  Bundle:
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/reuse_patch_qwen17_gray_cmrc_route`.
  Storage:
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/example_connector_storage`.
- Method:
  Ran `scd-vllm batch-benchmark` with `batch_size=128`,
  `num_batches=3`, `warmup_batches=1`, `max_new_tokens=16`,
  `SCDPreloadedConnector`, `scd_vllm.preloaded_connector`,
  `connector_cache_device=cuda`, and native vLLM prefix-cache baseline
  enabled.
- Outputs:
  `Experiment/results/a800_qwen17_scd_vllm_preloadedstats_20260618_cache/`
  containing `metrics.json`, `baseline.json`, `prefix-cache.json`,
  `load.json`, and
  `preloaded_connector_stats/scd_preloaded_connector_587169.json`.
- Results:
  `benchmark_pass=True`, `load_generation_ok=True`,
  `bundle_token_match=True`, `storage_ready=True`,
  `prefix_cache_generation_ok=True`, `route_id=paper_scd_reuse_patch_k6`,
  `batch_size=128`, `expected_requests=384`, and `output_tokens=6144` for
  each stage. Throughput: plain vLLM `11069.329059` tokens/s, native
  prefix-cache `12135.789531` tokens/s, and SCDPreloaded load
  `5930.711074` tokens/s. Ratios: prefix-cache/plain `1.096344`,
  SCDPreloaded/plain `0.535779`, and SCDPreloaded/prefix-cache `0.488696`.
  Final connector stats: `cache_hits=14308`, `cache_misses=28`,
  `cache_events=14336`, `cache_hit_rate=0.998047`, `cached_files=28`,
  `cached_bytes=1835008`, and `final_workers=1`.
- Analysis:
  The run confirms that the Qwen public connector path is active inside the
  vLLM worker and that almost all repeated requests hit the preloaded cache.
  It should be cited as additional-model integration evidence and an overhead
  boundary. It should not be cited as a serving speedup over plain vLLM or
  vLLM prefix caching.
- ResearchKB:
  Harvest recorded `harvest_b5b072e585e94bb8`; structured run recorded
  `run_d6919cf38ef74054`.

## 2026-06-18 - Release Readiness Multidataset Quality Gate

- Experiment id:
  `release_readiness_multidataset_quality_gate_20260618`.
- Purpose:
  Convert the current multi-model, multi-dataset evidence boundary into an
  executable release check so future cleanup cannot silently remove or corrupt
  the branch-local quality artifacts.
- Method:
  Added `check_multidataset_quality_evidence` to
  `Experiment/core_code/runtime_repo/scripts/check_release_readiness.py`.
  The gate reads Mistral CMRC2018, HotpotQA, and WikiText-2 result JSON files,
  plus the Qwen3-1.7B gray three-dataset aggregate.
- Parameters and inputs:
  Mistral artifacts:
  `Experiment/results/scd_vllm_multidataset_full_strict_20260617/` and
  `Experiment/results/wikitext2_patch20_patch26_full_current_20260618_034812/`.
  Qwen artifact:
  `Experiment/results/qwen17_gray_full_strict_microsharded_20260617/`.
- Acceptance checks:
  CMRC2018 dev must have `3219` samples, HotpotQA dev must have `7405`
  samples, and WikiText-2 test must have `763` samples. Mistral QA rows must
  contain positive F1/EM and preserve or improve F1/EM versus Full Consumer.
  WikiText-2 rows are checked with NLL and TTFT, not F1. Qwen rows are checked
  as three-dataset portability/risk evidence, not speedup evidence.
- Result:
  `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py`
  passed after adding the new gate.
- Analysis:
  This does not add a new model run, but it materially improves release
  reliability: the publishable branch now has machine-checked evidence for the
  core multi-dataset quality boundary and the second-model risk boundary.

## 2026-06-18 - Programmatic Official-vLLM Launch Helper Validation

- Experiment id:
  `scd_vllm_official_launch_example_dryrun_20260618`.
- Purpose:
  Add and validate a release-facing Python example for consuming
  `scd-vllm launch-spec` output without requiring users to copy internal CLI
  code.
- Method:
  Implemented `examples/scd_vllm_official_launch.py` with a no-GPU `--dry-run`
  mode and an execution mode that delegates to `run_launch_smoke`.
- Inputs:
  Unit test constructs a minimal `launch_spec.json` containing official
  `vllm.LLM` kwargs, `SCDPreloadedConnector`, module path
  `scd_vllm.preloaded_connector`, storage metadata, route/token hashes, and
  trusted-bundle safety flags.
- Results:
  `PYTHONPATH=.;src pytest tests/test_scd_vllm_official_launch_example.py tests/test_release_tree_script.py -q`
  passed with `2 passed`. `python scripts\check_release_readiness.py` also
  passed after adding the helper to required files, py-compile coverage, source
  hygiene, docs snippets, and release-tree packaging.
- Analysis:
  This improves the open-source plug-and-play interface without broadening the
  claim. The helper validates a trusted static launch contract and remains
  explicitly outside production multi-tenant serving or dynamic request routing.

## 2026-06-18 - Release Readiness Train/Eval Split Evidence Gate

- Experiment id:
  `release_readiness_train_eval_split_gate_20260618`.
- Purpose:
  Convert the no-data-leakage release boundary into executable checks so the
  open-source branch cannot silently pass after replacing full dev/test
  evaluation artifacts with train-set or partial-evaluation evidence.
- Method:
  Added `check_train_eval_split_evidence` to
  `Experiment/core_code/runtime_repo/scripts/check_release_readiness.py`.
  The gate reads Mistral route metadata, CMRC strict summary metadata,
  WikiText-2 run configs, and Qwen strict README/config/route artifacts.
- Parameters and inputs:
  Mistral release routes:
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`
  and
  `configs/routes/scd_icml_mistrallite_to_mistral7_patch26_reuse6_cmrc200.json`.
  Mistral evidence:
  `Experiment/results/scd_vllm_multidataset_full_strict_20260617/` and
  `Experiment/results/wikitext2_patch20_patch26_full_current_20260618_034812/`.
  Qwen evidence:
  `Experiment/results/qwen17_gray_full_strict_microsharded_20260617/`.
- Acceptance checks:
  Routes must declare a CMRC2018 train-200 calibration bank. CMRC strict
  summaries must use strict mode, full dev evaluation, an empty qid allowlist,
  Full Consumer plus reuse_patch methods, and zero fallback/recompute layers.
  WikiText-2 configs must use `datasets=wikitext2`, `eval_limit=0`,
  `max_length=0`, and the CMRC200 trained bank. Qwen strict evidence must
  document CMRC2018 train calibration, CMRC2018 dev/HotpotQA dev/WikiText-2
  test evaluation, `max_eval_samples=0`, strict mode, and no
  fallback/recompute.
- Results:
  `PYTHONPATH=.;src pytest tests -q` passed with `106 passed in 25.40s`.
  `python scripts\check_release_readiness.py` passed.
  `git diff --check` passed.
  `python scripts\build_release_tree.py --out ..\..\..\release_preview --force`
  built `release_preview` and the preview's own readiness check passed.
  The preview contains zero `__pycache__` directories.
- Analysis:
  This does not add new model quality results. It makes the current
  Mistral/Qwen evidence boundary machine-checkable, especially the separation
  between CMRC2018 train calibration and dev/test evaluation. The release
  remains a trusted static-prefix SCD-vLLM integration plus strict-runtime
  quality evidence, not a production multi-tenant serving claim.
- ResearchKB:
  Structured run recorded `run_7eda4bb299554033`. Local harvest scanned the
  runtime repo and skipped it because no new parseable result artifact was
  found.

## 2026-06-18 - A800mql High-Batch Mistral SCDPreloadedConnector Benchmark

- Experiment id:
  `a800_mistral_d1ce508_b256_shm_20260618_0938`.
- Purpose:
  Strengthen the release-facing vLLM plug-in evidence with a larger
  repeated-prefix Mistral batch benchmark for both current Mistral routes.
- Environment:
  Host `A800mql`, `conda activate cache`, Python 3.12.12, vLLM 0.18.0,
  Torch 2.10.0+cu128. The code snapshot synced to the host was
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_d1ce508_20260618_0915`.
- GPU selection:
  The first attempt selected GPUs `3/4` after 60 seconds of utilization `0`,
  but vLLM baseline initialization failed because free memory was below
  `gpu_memory_utilization=0.80` after resident allocations. The final attempt
  selected GPUs `0/1` after 60 seconds of utilization `0` and ran with
  `gpu_memory_utilization=0.75`.
- Method:
  Ran `scd-vllm batch-benchmark` with `SCDPreloadedConnector`,
  `scd_vllm.preloaded_connector`, `--include-prefix-cache-baseline`,
  `--enforce-eager`, `--disable-hybrid-kv-cache-manager`,
  `batch_size=256`, `num_batches=5`, `warmup_batches=1`,
  `max_new_tokens=16`, and `max_model_len=512`. To avoid repeated NAS reads
  across the plain, prefix-cache, and SCDPreloaded stages, the Mistral-7B model
  was copied to `/dev/shm/scd_models/Mistral-7B-v0.1`.
- Inputs:
  Routes `patch20_reuse12` and `patch26_reuse6`; trusted reuse/patch bundles
  from the current MistralLite -> Mistral-7B release evidence.
- Outputs:
  Local artifact:
  `Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/`.
- Results:
  `patch20_reuse12` passed with plain vLLM `995.460475` tok/s, vLLM
  prefix-cache `5236.793824` tok/s, SCDPreloaded `3286.296158` tok/s,
  SCD/plain `3.301282`, SCD/prefix `0.627540`, `cache_hits=49120`,
  `cache_misses=32`, and `cache_hit_rate=0.999349`.
  `patch26_reuse6` passed with plain vLLM `1016.189398` tok/s, vLLM
  prefix-cache `5522.382560` tok/s, SCDPreloaded `3168.681346` tok/s,
  SCD/plain `3.118200`, SCD/prefix `0.573789`, `cache_hits=49120`,
  `cache_misses=32`, and `cache_hit_rate=0.999349`.
- Analysis:
  The high-batch evidence is stronger than the earlier b128 run: it confirms
  worker-side cache reuse at `1280` requests per route and shows that
  SCDPreloaded is faster than plain vLLM on this repeated-prefix shape.
  Native vLLM prefix caching remains faster, so the open-release claim remains
  bounded to plug-in integration, reuse evidence, and overhead measurement.
- Failure note:
  The failed GPU3/4 attempt was caused by vLLM EngineCore memory reservation,
  not by route correctness or connector function. The mitigation is to inspect
  free GPU memory, use currently idle cards with enough free memory, lower
  `gpu_memory_utilization` when resident memory exists, and keep the model copy
  off NAS for repeated benchmark stages.
- ResearchKB:
  Structured run recorded `run_fc6db6315e914a97`. The memory-reservation
  failure case was recorded as `problem_1453a6b452fd415c`. Local harvest
  recorded `harvest_b1fbe0511e455caa` for `patch20` and
  `harvest_ea4e76f6ab70227f` for `patch26`.

## 2026-06-18 10:09 +08:00 - A800 conda cache environment validation

- Experiment name or id:
  `a800_conda_cache_env_validation_20260618_1009`.
- Purpose:
  Confirm the runtime environment requested for remaining A800 SCD-vLLM work
  and explicitly avoid placing virtual environments on NAS.
- Method:
  Created a temporary Aline channel on `A800mql`, changed to
  `/home/c101-800-mql`, activated
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  and queried environment paths plus package versions.
- Environment:
  Host `A800mql`; activation `conda activate cache`; Python path
  `/home/c101-800-mql/.conda/envs/cache/bin/python`; conda prefix
  `/home/c101-800-mql/.conda/envs/cache`.
- Results:
  Verified Python `3.12.12`, PyTorch `2.10.0+cu128`, Transformers `5.3.0`,
  Datasets `4.8.5`, and vLLM `0.18.0`. The conda environment is under the
  A800 home directory, not NAS.
- Analysis:
  This environment is suitable for the next SCD-vLLM A800 runs. Future A800
  commands should reuse this conda environment through a persistent remote
  channel instead of creating a NAS-backed venv.
- Next actions:
  Use `conda activate cache` for A800 experiments, inspect GPU utilization
  before launch, and only use idle cards with enough free memory.

## 2026-06-18 10:23 +08:00 - Release-script path and train/eval split hardening

- Experiment name or id:
  `release_script_hygiene_20260618_1023`.
- Purpose:
  Reduce open-release reproducibility risk before more multi-model and
  multi-dataset SCD-vLLM experiments by removing stale remote defaults and
  train/eval leakage patterns from runnable helper scripts.
- Method:
  Searched the runtime and experiment scripts for stale `ser` and A800 account
  paths, checked ResearchKB context for KV-cache reuse reproducibility and
  safety risks, verified the current A800mql data path by read-only Aline
  probing, then patched scripts and release-readiness checks.
- Inputs:
  Current branch `codex/scd-icml-open-release-20260611`; runtime repo scripts
  under `Experiment/core_code/runtime_repo/scripts`; A800mql path probe for
  `/home/c101-800-mql/maqianli/no4/data/cmrc2018`.
- Environment:
  Local Windows PowerShell for code changes and tests. A800mql was used only
  for read-only path probing; no remote experiment was launched.
- Execution steps:
  Updated A800 helper defaults from `student_t` to `c101-800-mql`, updated ser
  defaults from `/disk_n/c101-3-1` to `/disk_n/c101-3-mql`, made runtime repo
  ser scripts derive `ROOT` from the local checkout where practical, removed
  train-to-dev copying from calibration helpers, and added release-readiness
  guards against stale paths and train/dev split spoofing.
- Outputs:
  Patched shell/Python helper scripts plus `.gitkeep` placeholders for missing
  canonical project directories.
- Results:
  `check_release_readiness.py` passed. Full runtime repo test suite passed with
  `107 passed`. `git diff --check` passed. `instance.json` mapped-path
  validation passed. Search found no remaining executable-script pattern that
  copies `cmrc2018_train.json` into `cmrc2018_dev.json`. Because local Windows
  did not have Bash on `PATH`, the changed shell scripts were also streamed
  read-only to `A800mql` and checked with `bash -n -`; all changed shell
  scripts passed.
- Analysis:
  This does not add new quality or throughput numbers, but it materially
  improves the release branch by making subsequent SCD-vLLM runs less likely to
  use stale host paths or contaminated train/eval splits.
- Conclusions:
  The next experimental step can resume from a cleaner release branch using
  A800 `conda activate cache` or ser `/disk_n/c101-3-mql/venvs/A800` defaults.
- Next actions:
  Launch the next full-dataset multi-model SCD-vLLM run only after checking
  current GPU utilization and confirming the selected model/data paths.
- ResearchKB:
  Structured run recorded `run_560eeb1fb94a4882`. Local harvest scanned the
  runtime repo and recorded no additional parseable result artifact because the
  updated evidence is represented in this log entry and the release-readiness
  command output.

## 2026-06-18 10:35 +08:00 - Release preview hygiene check after A800 cache-env correction

- Experiment name or id:
  `release_preview_cache_env_hygiene_20260618_1035`.
- Purpose:
  Verify that the open-release runtime package no longer exposes stale
  `student_t`/old-ser paths, NAS defaults, train/eval leakage patterns, or
  unsupported vLLM safety claims after switching A800 work to
  `conda activate cache`.
- Method:
  Probed A800mql with `source /opt/miniconda3/etc/profile.d/conda.sh && conda
  activate cache`, checked the selected Python executable and GPU visibility,
  patched stale path references, built a fresh release preview, scanned the
  preview excluding the readiness guard's own forbidden-string constants, and
  ran the preview test suite.
- Inputs:
  Branch `codex/scd-icml-open-release-20260611`; runtime repo under
  `Experiment/core_code/runtime_repo`; A800mql cache env; release preview output
  path `Experiment/results/release_preview_scd_icml_runtime_20260618_103502`
  before cleanup.
- Environment:
  Local Windows PowerShell for patching and pytest. A800mql `cache` conda env
  for read-only environment and `bash -n` verification; `which python` resolved
  to `/home/c101-800-mql/.conda/envs/cache/bin/python`; CUDA was available with
  8 visible A800 GPUs.
- Execution steps:
  Updated README/report/config paths, removed two internal historical artifact
  migration scripts, removed the Qwen32 helper's implicit NAS run-output
  fallback, streamed the modified shell script to A800mql for `bash -n`, built a
  fresh release preview, scanned the preview for forbidden release strings, and
  ran `PYTHONPATH=.;src pytest tests -q` inside the preview.
- Outputs:
  Patched runtime repo files plus verification logs in this entry. The generated
  preview directories were deleted after validation because they are
  generated source copies and are not ignored by git.
- Results:
  A800 environment probe confirmed `cache` uses
  `/home/c101-800-mql/.conda/envs/cache/bin/python` with torch
  `2.10.0+cu128`. Source-tree scan found no `/home/student_t`,
  `/disk_n/c101-3-1`, or `/nas/mql` outside the readiness guard. The fresh
  release preview readiness check passed. Preview user-visible scan returned
  `NO_HITS` for stale paths, train-to-dev copy pattern, and forbidden vLLM
  safety claims. Preview tests passed with `107 passed in 30.94s`.
- Analysis:
  This is a release-hygiene validation, not a new model-quality or throughput
  result. It reduces the risk that users reproduce with stale private paths or
  contaminated split handling.
- Conclusions:
  The release branch is cleaner for local packaging and later push. Further
  model/dataset experiments should use A800 `conda activate cache` or ser
  `/disk_n/c101-3-mql/venvs/A800` after fresh GPU checks.
- Next actions:
  Commit the hygiene changes and, when requested, push the branch. Do not add
  generated release preview directories to git.

## 2026-06-18 10:49 +08:00 - Current bc03cd4 Mistral SCD-vLLM high-batch benchmark

- Experiment name or id:
  `a800_mistral_bc03cd4_b256_gpu7_20260618_104914`.
- Purpose:
  Re-run the release-facing MistralLite -> Mistral-7B SCD-vLLM high-batch
  benchmark on the current `bc03cd4` open-release snapshot, so the publishable
  vLLM evidence is not tied only to an older `d1ce508` code snapshot.
- Method:
  Pushed the current runtime repo to
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_bc03cd4_20260618_1045`,
  activated A800 `cache`, sampled GPU state for about 70 seconds, selected GPU
  `7`, and ran `python -m scd_vllm.batch_benchmark` for both
  `patch20_reuse12` and `patch26_reuse6` with plain vLLM, vLLM prefix caching,
  and `SCDPreloadedConnector`.
- Parameters:
  `batch_size=256`, `num_batches=5`, `warmup_batches=1`,
  `max_new_tokens=16`, `max_model_len=4096`, `dtype=bfloat16`,
  `gpu_memory_utilization=0.90`, `CUDA_VISIBLE_DEVICES=7`,
  `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `connector_cache_device=cuda`.
- Inputs:
  Model `/dev/shm/scd_models/Mistral-7B-v0.1`; existing raw-bank bundles and
  connector storage from the full patch20/patch26 raw-bank validation runs.
- Environment:
  A800mql, `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate
  cache`; Python `/home/c101-800-mql/.conda/envs/cache/bin/python`; vLLM
  `0.18.0`; torch `2.10.0+cu128`.
- Outputs:
  Local artifact directory
  `Experiment/results/a800_mistral_bc03cd4_b256_gpu7_20260618_104914/`;
  remote source run
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_mistral_bc03cd4_b256_gpu7_20260618_104914`.
- Results:
  Both routes passed. `patch20_reuse12`: plain `803.700215` tok/s,
  prefix-cache `5516.152993` tok/s, SCDPreloaded `3185.841785` tok/s,
  SCD/plain `3.963968`, SCD/prefix `0.577548`, cache hits/misses
  `49120/32`, hit rate `0.999349`. `patch26_reuse6`: plain
  `1005.013664` tok/s, prefix-cache `5471.840579` tok/s, SCDPreloaded
  `2864.393751` tok/s, SCD/plain `2.850104`, SCD/prefix `0.523479`,
  cache hits/misses `49120/32`, hit rate `0.999349`.
- Analysis:
  This is a repeated-prefix trusted microbenchmark. It supports the claim that
  the current release snapshot can load SCD reconstructed KV through vLLM's
  external connector mechanism and outperform plain vLLM prefix recomputation
  on this controlled workload. It still does not support a claim that SCD beats
  vLLM native prefix caching, nor any multi-tenant or privacy-safety claim.
- Conclusions:
  The current open-release code snapshot has fresh high-batch vLLM evidence for
  both release routes. `patch20_reuse12` remains the stronger speed/reuse demo;
  `patch26_reuse6` remains the quality-first CMRC/WikiText route with a lower
  SCD/plain throughput ratio in this benchmark.
- Next actions:
  Record the run in ResearchKB, keep the artifacts local under
  `Experiment/results`, and use these numbers as the primary release-facing
  vLLM throughput evidence.

## 2026-06-18 11:07 +08:00 - Current c925185 Qwen3 gray SCD-vLLM high-batch boundary benchmark

- Experiment name or id:
  `a800_qwen17_gray_c925185_b256_gpu6_20260618_110715`.
- Purpose:
  Test the current release branch on a second model family with the public
  SCD-vLLM preloaded connector path, and determine whether Qwen3-1.7B ->
  Qwen3-1.7B-gray supports a speedup claim or only portability/risk evidence.
- Method:
  Pushed the current runtime repo to
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_c925185_20260618_1105`,
  activated A800 `cache`, sampled GPU state for about 70 seconds, selected GPU
  `6`, and ran `python -m scd_vllm.batch_benchmark` with plain vLLM, vLLM
  prefix caching, and `SCDPreloadedConnector`.
- Parameters:
  `batch_size=256`, `num_batches=5`, `warmup_batches=1`,
  `max_new_tokens=16`, `max_model_len=4096`, `dtype=bfloat16`,
  `gpu_memory_utilization=0.45`, `CUDA_VISIBLE_DEVICES=6`,
  `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `connector_cache_device=cuda`.
- Inputs:
  Model and tokenizer
  `/home/c101-800-mql/maqianli/SCD-ICML/assets/qwen/Qwen3-1.7B-gray`;
  bundle
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/reuse_patch_qwen17_gray_cmrc_route`;
  connector storage
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen17_scd_vllm_plugin_smoke_20260618_cache/example_connector_storage`.
- Environment:
  A800mql, `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate
  cache`; vLLM `0.18.0`; torch from the A800 cache environment.
- Outputs:
  Local artifact directory
  `Experiment/results/a800_qwen17_gray_c925185_b256_gpu6_20260618_110715/`;
  remote source run
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/a800_qwen17_gray_c925185_b256_gpu6_20260618_110715`.
- Results:
  The benchmark passed with expected requests `1280`. Plain vLLM throughput was
  `13218.147026` tok/s, vLLM prefix-cache throughput was `16298.884031`
  tok/s, and SCDPreloaded load throughput was `5944.215648` tok/s.
  SCDPreloaded/plain ratio was `0.449701`; SCDPreloaded/prefix-cache ratio was
  `0.364701`. Connector stats reported `42980` hits, `28` misses, hit rate
  `0.999349`, `28` cached files, and `1835008` cached bytes.
- Analysis:
  This validates that the current branch can run the Qwen gray SCDPreloaded
  path through vLLM at high batch size, but the result is a negative throughput
  boundary. It supports portability and overhead-risk documentation, not a Qwen
  speedup claim.
- Conclusions:
  Mistral remains the release-facing positive throughput evidence. Qwen3 gray
  should remain framed as second-model-family functional portability plus a
  caution that the current connector path can be slower than both plain vLLM
  and native prefix caching.
- Next actions:
  Record and harvest this run in ResearchKB, keep the artifact in
  `Experiment/results`, and avoid presenting it as acceleration evidence.

## 2026-06-18 12:08 +08:00 - Qwen2.5-32B storage materialization and strict SCD queue

- Experiment name or id:
  `a800_qwen32_preflight_queue_20260618`.
- Purpose:
  Switch the current SCD-vLLM release-risk test from Qwen3-1.7B gray evidence
  to the paper-relevant Qwen2.5-32B base -> Qwen2.5-32B-Instruct setting, while
  avoiding development-disk model placement and preventing train/eval leakage.
- Method:
  Stopped the temporary Qwen2.5-32B download targeting `/dev/shm`, removed the
  remaining `/dev/shm/scd_models/Qwen2.5-32B` partial copy, downloaded the base
  checkpoint to `/home/c101-800-mql/nas/models/Qwen2.5-32B`, generated an
  explicit base-to-instruct model config, ran the guarded Qwen32 preflight, and
  launched a strict GPU wait queue for the SCD run.
- Parameters:
  `required_gpus=2`, `required_free_mb=70000`, `monitor_seconds=60`,
  `train_limit=200`, `eval_limit=200`, `train_cmrc_split=train`,
  `eval_cmrc_split=dev`, `datasets=cmrc2018`, `max_new_tokens=32`,
  `max_length=0`, `cpu_fallback=false`, `recompute=false`.
- Inputs:
  Base model `/home/c101-800-mql/nas/models/Qwen2.5-32B`; instruct model
  `/home/cuihanshuai/models/Qwen2.5-32B-Instruct`; data root
  `/home/c101-800-mql/maqianli/no4/data`; runtime repo
  `/home/c101-800-mql/maqianli/SCD-ICML/runtime_repo_codex_38dcdca_tp_20260618_1128`.
- Environment:
  A800mql, `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate
  cache`; heavy run output directed to `/home/c101-800-mql/nas/SCD-ICML/runs`.
- Outputs:
  Local evidence directory
  `Experiment/results/a800_qwen32_preflight_queue_20260618/`; remote queue
  directory
  `/home/c101-800-mql/maqianli/SCD-ICML/runs/qwen32_strict_scd_queue_20260618_120318`.
- Results:
  The base checkpoint is materialized on NAS with `30` files, `17`
  safetensors, `model_type=qwen2`, architecture `Qwen2ForCausalLM`, `64`
  layers, and hidden size `5120`. The preflight validated both Qwen2.5-32B
  paths and the CMRC2018 train/dev files, but returned `ok=false` because no
  two GPUs satisfied the strict sustained-idle requirement at the time of
  sampling. The strict queue remains active and has not yet produced quality
  metrics.
- Analysis:
  This is preparation and guarded scheduling evidence, not a completed Qwen32B
  quality result. It confirms the model is no longer stored on the development
  disk or `/dev/shm`; Qwen32B evaluation is blocked only by GPU availability
  under the current strict policy.
- Conclusions:
  Do not use the earlier Qwen3-1.7B gray result as the current main Qwen
  release test. The next valid paper-relevant Qwen evidence must come from the
  queued Qwen2.5-32B base -> Qwen2.5-32B-Instruct run after two suitable A800
  GPUs are idle.
- Next actions:
  Monitor the queue, pull the NAS run directory after execution starts or
  completes, record F1/EM and throughput artifacts, and then update the release
  risk summary.

## 2026-06-18 12:16 +08:00 - ser full HotpotQA and WikiText-2 strict eval launch

- Experiment name or id:
  `extra_datasets_full_cmrcbank_20260618_121640`.
- Purpose:
  Replace prior smoke or small-limit extra-dataset checks with full-dataset
  strict evaluation for the current open-source SCD runtime, while keeping the
  reuse/patch bank trained on CMRC2018 train data and evaluating only on held
  out or external datasets.
- Method:
  Used ser `runtime_repo_codex_current_20260618`, which passed release
  readiness and includes the current vLLM/SCD release code. Launched WikiText-2
  full strict eval on GPU2/GPU3 with `limit=0`, and initially launched HotpotQA
  full on GPU0/GPU1. After observing the HotpotQA single stream would take
  hours, stopped only the unsharded HotpotQA process and relaunched HotpotQA as
  two qid-filtered shards: `shard_00` on GPU0/GPU1 immediately and `shard_01`
  queued to start on GPU2/GPU3 after WikiText-2 completes.
- Parameters:
  Route `reuse19/patch13`, `fallback=0`, `recompute=0`, `max_length=0`,
  `max_new_tokens=32`, `bandwidth_gbps=200`, `quant_group_size=128`,
  `dtype=bfloat16`, `seed=42`, `runtime_eval.max_eval_samples=0`.
- Inputs:
  Bank root
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222`;
  data root
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854/data`;
  route
  `route/route_cmrc2018_fixed_patchbudget13.json`; reuse checkpoint
  `ckpt_reuse/layerwise_seed42`; patch checkpoint
  `ckpt_patch/native_alllayers_seed42`.
- Environment:
  ser, `/disk_n/c101-3-mql/venvs/A800`, four 4090D GPUs initially idle.
- Outputs:
  Remote run directory
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_datasets_full_cmrcbank_20260618_121640`.
- Results:
  Launch evidence only so far. The loaders reported full sample counts:
  HotpotQA `7405` and WikiText-2 `763`. HotpotQA qid shards contain `3703`
  and `3702` qids, totaling `7405`. `shard_00` and WikiText-2 are running; the
  `shard_01` waiter is active. A postprocess waiter
  `postprocess_and_cmrc_after_hotpot.sh` is also active; it will verify the
  merged HotpotQA artifact has `7405` qids and `7405*3` per-method rows, then
  launch CMRC2018 dev full evaluation. No final F1/EM/PPL summary exists yet.
- Analysis:
  This run keeps the training source separate from evaluation: the reuse/patch
  bank comes from CMRC2018 train, while the current evaluations are full
  HotpotQA and full WikiText-2. Sharding changes scheduling only; it does not
  reduce the HotpotQA evaluation set.
- Conclusions:
  The full extra-dataset evidence is in progress. Do not report quality numbers
  from this run until the shard summaries and aggregate summary are produced.
- Next actions:
  Monitor both active processes, launch or verify `shard_01` after WikiText-2
  releases GPU2/GPU3, aggregate HotpotQA shards by concatenating
  `per_sample_metrics.jsonl`, verify the queued CMRC2018 dev full run starts
  after HotpotQA aggregation, pull result artifacts, record F1/EM/PPL, and
  update the release-risk summary.

### 2026-06-18 12:32 +08:00 scheduler update

- Change:
  Replaced the first postprocess waiter with a more parallel scheduler. CMRC2018
  dev is now split into two qid shards, `1610` and `1609` samples. CMRC shard
  `00` waits only for HotpotQA shard `00` and then starts on GPU0/GPU1; CMRC
  shard `01` waits only for HotpotQA shard `01` and then starts on GPU2/GPU3.
  A final aggregate waiter waits for WikiText-2, both HotpotQA shards, and both
  CMRC shards, then re-aggregates HotpotQA and CMRC from their
  `per_sample_metrics.jsonl` files.
- Rationale:
  The previous schedule would leave GPU0/GPU1 idle after HotpotQA shard `00`
  completed while waiting for shard `01`. The new schedule keeps the full
  datasets unchanged but reduces expected wall-clock time by launching CMRC
  as soon as each GPU pair is released.
- Verification:
  Confirmed the old postprocess waiter was stopped and the new waiters are
  active. Confirmed CMRC qid files contain `1610 + 1609 = 3219` samples. The
  aggregate script will assert per-method sample counts for HotpotQA `7405`,
  CMRC2018 dev `3219`, and WikiText-2 `763` before writing
  `aggregate_summary.json`.

### 2026-06-18 13:12 +08:00 WikiText-2 full partial result

- Result:
  WikiText-2 full evaluation completed with `763` samples. The primary metric
  for this dataset is PPL/NLL rather than F1. Full Consumer achieved
  `ppl=7.2709`, `avg_nll=1.9839`, `ttft_ms=60.3179`. Reuse-only achieved
  `ppl=927.3553`, `avg_nll=6.8323`, `ttft_ms=24.6725`. SCD achieved
  `ppl=23.7634`, `avg_nll=3.1681`, `ttft_ms=30.4605`.
- Analysis:
  The current CMRC-trained `reuse19/patch13` route strongly improves over
  reuse-only on WikiText-2 but is still far from the full-consumer PPL/NLL.
  This is a release-risk signal for cross-dataset transfer quality, not a
  completed full-result conclusion because HotpotQA and CMRC2018 dev are still
  running.
- Local artifacts:
  `Experiment/results/ser_extra_datasets_full_cmrcbank_20260618/wikitext2_n0/`
  and
  `Experiment/results/ser_extra_datasets_full_cmrcbank_20260618/partial_results_summary.md`.
- ResearchKB:
  Harvested the partial result directory as `harvest_373a99c91efffa70`
  (`summary.json`) and `harvest_14b5b0eef6bb2a6b`
  (`per_mode_summary.json`).

### 2026-06-18 12:40 +08:00 A800 Qwen32 storage correction

- Change:
  Corrected the A800 Qwen32 execution defaults so heavy model artifacts,
  Hugging Face caches, ModelScope caches, Torch caches, and run outputs target
  `/home/c101-800-mql/nas`, not the nearly full `/home` development disk or
  `/dev/shm`.
- Evidence:
  Live disk check showed `/home` had about `23G` free and was `99%` used,
  while `/home/c101-800-mql/nas` had about `64T` free. The base
  Qwen2.5-32B checkpoint already existed at
  `/home/c101-800-mql/nas/models/Qwen2.5-32B` with `30` files and `17`
  safetensors. The instruct checkpoint did not yet exist on NAS, so a copy
  from `/home/cuihanshuai/models/Qwen2.5-32B-Instruct` to
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct` was started as PID
  `1999513`.
- Verification:
  The updated Qwen32 model JSON configs parse successfully. The updated
  `run_a800_scd_icml_qwen32_repro.sh` and saved queue script pass `bash -n`
  on A800.
- Status:
  This is an environment and artifact-placement fix only. No Qwen32 quality
  metric should be reported from this entry. The older c101-owned Qwen32
  waiting queue was stopped before it could launch with the previous instruct
  path; a new queue should be launched from the current A800 account after the
  NAS instruct copy is complete.

### 2026-06-18 12:55 +08:00 A800 Qwen32 student_t storage queue

- Change:
  Started a new Qwen32 queue from the current A800 account `student_t`, using
  local Conda environment `cache` and a local lightweight runtime repo at
  `/home/student_t/maqianli/SCD-ICML/runtime_repo_codex_storage_20260618`.
  Heavy outputs and caches remain under `/home/c101-800-mql/nas/SCD-ICML`.
- Inputs:
  Base model `/home/c101-800-mql/nas/models/Qwen2.5-32B`; instruct model
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`; CMRC2018 data root
  `/home/c101-800-mql/maqianli/no4/data`.
- Outputs:
  Queue root
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_student_t_storage_queue_20260618_1252`;
  wrapper PID `2020827`.
- Status:
  The queue is waiting for two sustained-idle A800 GPUs with at least `70000`
  MB free each. Preflight attempt 1 failed only because no such GPU pair was
  available; it confirmed the NAS model paths and data paths were reachable.

### 2026-06-18 13:35 +08:00 A800 Qwen32 vLLM TP4 CMRC2018 full baseline

- Purpose:
  Replace Qwen1.7B-style evidence with a Qwen2.5-32B-Instruct run path and
  verify a full-dataset quality metric through official vLLM tensor parallelism.
- Method:
  Added `scd_vllm.dataset_eval`, which loads CMRC2018 samples, runs batched
  official vLLM generation, scores each prediction with the existing CMRC
  F1/EM scorer, and writes `per_sample_metrics.jsonl` plus `summary.json`.
  The run is a Full Consumer vLLM baseline, not an SCD reuse/patch quality
  result.
- Inputs:
  Model `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`; data root
  `/home/c101-800-mql/maqianli/no4/data`; split `cmrc2018 dev`; full split
  (`--limit 0`).
- Parameters:
  `tensor_parallel_size=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `dtype=bfloat16`,
  `max_model_len=2048`, `max_new_tokens=32`, `batch_size=16`,
  `gpu_memory_utilization=0.50`, `enforce_eager=true`.
- Preflight:
  GPUs `0,1,2,3` had `0%` utilization for six samples over 60 seconds before
  launch. Model and cache paths were on
  `/home/c101-800-mql/nas`, not the `/home` development disk.
- Outputs:
  Run root
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_cmrc_full_tp4_20260618_133422`;
  PID `2095269`; logs `run.log`, `run.err`; final metrics are expected at
  `summary.json` and `per_sample_metrics.jsonl`.
- Current status:
  vLLM `0.18.0` accepted the TP4 configuration and initialized ranks
  `0..3` with `world_size=4`. At the latest check, model loading was in
  progress from the 17 NAS safetensors shards; no F1/EM result is available
  yet and no quality claim should be made from this entry until
  `summary.json` exists.

### 2026-06-18 13:54 +08:00 A800 Qwen32 vLLM TP4 CMRC2018 full result

- Result:
  Qwen2.5-32B-Instruct Full Consumer baseline completed on the full CMRC2018
  dev split with `3219` samples. Final F1 was `0.4156507535`; EM was
  `0.0043491768`.
- Runtime:
  vLLM `0.18.0`; tensor parallel size `4`; GPUs `0,1,2,3`; `bfloat16`;
  `max_model_len=2048`; `max_new_tokens=32`; `batch_size=16`;
  `gpu_memory_utilization=0.50`; `enforce_eager=true`.
- Timing:
  Model load from NAS took `716.7035` seconds. Dataset evaluation took
  `312.7507` seconds, with `98,758` output tokens and
  `316.6041` output tokens per second over the generation sections.
- Artifacts:
  Remote run root:
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_cmrc_full_tp4_20260618_133422`.
  Local copy:
  `Experiment/results/a800_qwen32_vllm_cmrc_full_tp4_20260618/`, containing
  `summary.json`, `per_sample_metrics.jsonl`, command, preflight, stdout, and
  stderr logs.
- Analysis:
  This proves the current branch can load and evaluate Qwen32B through official
  vLLM TP4 from NAS-backed model storage. It is a Full Consumer baseline only,
  not an SCD reuse/patch result and not evidence for Qwen32B SCD speedup.
  The very low exact match appears to be mostly answer-format mismatch: the
  first prediction included a full sentence while the reference answer was the
  span `光荣和ω-force`. A follow-up should add stricter answer extraction or a
  CMRC-specific generation prompt before using Qwen32B baseline quality as a
  final comparison target.
- ResearchKB:
  Harvested local artifacts as `harvest_65368399d230de84` (`run.log`),
  `harvest_aeecc6873def53c2` (`per_sample_metrics.jsonl`), and
  `harvest_c25ea00488abbdce` (`summary.json`).

### 2026-06-18 14:17 +08:00 A800 Qwen32 vLLM TP4 CMRC2018 chat-template full result

- Purpose:
  Re-run the Qwen2.5-32B-Instruct Full Consumer baseline with the correct
  instruction-model prompt format after the plain completion prompt produced a
  misleadingly low baseline (`F1=0.4156507535`, `EM=0.0043491768`).
- Method:
  Added and used the `scd_vllm.dataset_eval` options
  `--apply-chat-template` and `--extra-instruction`. The prompt was wrapped
  with the tokenizer chat template and appended with `请只输出答案短语，不要解释。`.
- Inputs:
  Model `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`; data root
  `/home/c101-800-mql/maqianli/no4/data`; split `cmrc2018 dev`; full split
  (`3219` samples).
- Parameters:
  `tensor_parallel_size=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `dtype=bfloat16`,
  `max_model_len=2048`, `max_new_tokens=32`, `batch_size=16`,
  `gpu_memory_utilization=0.50`, `enforce_eager=true`.
- Result:
  Final F1 was `0.9098067394`; EM was `0.7086051569`. The first prediction
  changed from a full sentence in the plain-prompt run to the exact short
  answer `光荣和ω-force`, matching the reference.
- Timing:
  Model load from NAS took `639.7229` seconds. Dataset evaluation took
  `282.7484` seconds, with `30,371` output tokens and `107.7083` output
  tokens per second over generation sections.
- Artifacts:
  Remote run root:
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_cmrc_full_tp4_chat_20260618_140019`.
  Local copy:
  `Experiment/results/a800_qwen32_vllm_cmrc_full_tp4_chat_20260618/`.
- Analysis:
  Qwen2.5-32B-Instruct CMRC baselines must use chat-template prompting and a
  short-answer instruction. The earlier plain-prompt run is retained as a
  prompt-format failure diagnostic, not as the quality baseline. This result
  is still Full Consumer only; it does not establish Qwen32B SCD reuse/patch
  quality or speed.
- ResearchKB:
  Harvested local artifacts as `harvest_cbe4ec6c938ea710` (`run.log`),
  `harvest_6fc5be3f24df2b21` (`per_sample_metrics.jsonl`), and
  `harvest_c108d4acfe7fa2cb` (`summary.json`).

### 2026-06-18 14:20 +08:00 Post-Qwen32 chat-template release gate

- Purpose:
  Verify that the new vLLM dataset evaluation path and Qwen32B result records
  did not break the open-release readiness checks.
- Commands:
  `python scripts/check_release_readiness.py` from
  `Experiment/core_code/runtime_repo`; and
  `PYTHONPATH=src python -m pytest tests/test_scd_vllm_dataset_eval.py
  tests/test_scd_vllm_plugin.py tests/test_scd_vllm_official_launch_example.py`.
- Result:
  Release readiness check passed. The targeted vLLM tests passed with
  `11 passed`.
- Analysis:
  The chat-template option and tensor-parallel launch-spec support are covered
  by local tests and remain compatible with the release scanner. This does not
  close the remaining Qwen32B SCD reuse/patch evidence gap.

### 2026-06-18 14:25 +08:00 A800 Qwen32 cache placement audit

- Purpose:
  Check whether Qwen32 model and framework caches could spill onto the A800
  development disk before the queued strict SCD run starts.
- Method:
  Inspected A800 disk usage, active Qwen32/vLLM processes, model configs, and
  run logs. Compared the `/home` filesystem with the NAS storage mount.
- Result:
  A800 `/home` was `99%` used with about `21G` free, while
  `/home/c101-800-mql/nas` had about `64T` free. The active Qwen32 model config
  points to `/home/c101-800-mql/nas/models/Qwen2.5-32B` and
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`. The completed vLLM TP4
  CMRC baseline logs also loaded the model and tokenizer from the NAS model
  path.
- Change:
  Hardened `scripts/run_a800_scd_icml_qwen32_repro.sh` so default cache roots
  include `HF_ASSETS_CACHE`, `XDG_CACHE_HOME`, and `VLLM_CACHE_ROOT` under
  `/home/c101-800-mql/nas/SCD-ICML/model_cache`, in addition to `HF_HOME`,
  `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `MODELSCOPE_CACHE`, and `TORCH_HOME`.
- Next action:
  The updated wrapper was synced to the A800 runtime repo and validated with
  remote `bash -n`. Let the existing idle-GPU queue pick up the updated script
  before launch. The NAS cache subdirectories were created immediately under
  `/home/c101-800-mql/nas/SCD-ICML/model_cache` to avoid startup-time fallback
  to default `~/.cache` locations.

### 2026-06-18 14:44 +08:00 Qwen32 official-vLLM HotpotQA/WikiText full-dataset queue

- Purpose:
  Move the Qwen2.5-32B-Instruct official-vLLM validation beyond CMRC2018 by
  running full HotpotQA and WikiText-2 baselines through the same release
  evaluator. This is Full Consumer evidence, not SCD reuse/patch evidence.
- Code change:
  Extended `scd_vllm.dataset_eval` to load `hotpotqa`,
  `hotpotqa_compressed`, `wikitext2`, and `wikitext-2` in addition to
  `cmrc2018`. QA datasets retain F1/EM as the primary metric; WikiText-2 now
  computes prompt-token logprob NLL/PPL through vLLM `prompt_logprobs` and marks
  PPL as the primary metric. Added unit tests for dataset selection and NLL
  prompt/logprob helpers.
- Data:
  The A800mql `/home/c101-800-mql/maqianli/no4/data` HotpotQA file was only
  `133` bytes, so it was not used. Copied complete data from ser to A800 NAS:
  `/home/c101-800-mql/nas/SCD-ICML/data/HotpotQA/raw/hotpot_dev_distractor_v1.json`
  (`46,320,117` bytes, `7405` samples) and
  `/home/c101-800-mql/nas/SCD-ICML/data/wikitext-2/wiki.test.txt`
  (`1,290,590` bytes, `763` samples).
- Verification:
  Local targeted tests passed with `14 passed`; release readiness passed. On
  A800mql `conda activate cache`, `dataset_eval_import_check=ok`, vLLM version
  was `0.18.0`, and torch was `2.10.0+cu128`.
- Run:
  Started
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618_143845`.
  The queue waited for four GPUs with `util=0` and at least `48GB` free for
  `60` seconds, then selected `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Parameters:
  Model `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`; data root
  `/home/c101-800-mql/nas/SCD-ICML/data`; `tensor_parallel_size=4`;
  `gpu_memory_utilization=0.34`; HotpotQA `max_model_len=8192`,
  `batch_size=8`, chat template enabled with short-answer instruction; WikiText
  `max_model_len=2048`, `batch_size=16`, prompt NLL/PPL enabled.
- Current status:
  At `2026-06-18T14:43:45+08:00`, the HotpotQA stage was loading safetensors
  from NAS with no OOM or traceback observed. No final summary had been written
  yet.

### 2026-06-18 15:08 +08:00 Storage-backed vLLM cache guard

- Purpose:
  Prevent Qwen32B or other large model downloads from falling back to the A800
  `/home` development disk during official-vLLM dataset evaluation.
- Current disk state:
  A800 `/home` is still unsuitable for model downloads: `1.8T` total,
  `1.7T` used, about `18G` free. The NAS storage mount
  `/home/c101-800-mql/nas` has about `64T` free. The current Qwen32B-Instruct
  checkpoint is stored at `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`
  and occupies about `62G`.
- Change:
  Added `scd_vllm.cache_env.configure_model_cache` and wired `--cache-root`
  into `scd_vllm.dataset_eval` and `scd_vllm.vllm_baseline`. The helper creates
  and exports `HF_HOME`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`,
  `TRANSFORMERS_CACHE`, `MODELSCOPE_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`,
  and `VLLM_CACHE_ROOT` under the specified storage directory. When supplied,
  the helper also passes the HF hub cache as vLLM `download_dir`.
- Verification:
  Local targeted tests passed with `19 passed` across dataset eval, vLLM
  baseline, plugin, and official launch-spec tests. `scripts/check_release_readiness.py`
  passed and `git diff --check` reported no whitespace errors. The updated
  helper and vLLM entry points were synced to the A800 runtime repo, and remote
  import validation passed using
  `/home/c101-800-mql/.conda/envs/cache/bin/python`.
- Current A800 run status:
  The existing TP4 Qwen32B HotpotQA Full Consumer run continues from NAS paths.
  At `2026-06-18T15:07:08+08:00`, HotpotQA progress was `2800/7405` with
  running averages `F1=0.7529678148` and `EM=0.6007142857`. No `No space`,
  OOM, traceback, or runtime error was observed.
- Current ser run status:
  The full-context Mistral reuse19/patch13 run is still active. At
  `2026-06-18T15:07:10+08:00`, HotpotQA shard 01 was `2955/3702` and CMRC2018
  shard 00 was `254/1610`; shard 00 HotpotQA and WikiText-2 summaries already
  exist, with aggregation waiting for the remaining shards.

### 2026-06-18 15:24 +08:00 Qwen32 official-vLLM HotpotQA full baseline completed

- Purpose:
  Record the completed Qwen2.5-32B-Instruct Full Consumer baseline on complete
  HotpotQA dev data using the official-vLLM dataset evaluator. This is a
  baseline quality artifact, not a SCD reuse/patch result.
- Run:
  Remote run root:
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618_143845`.
  Local artifact copy:
  `Experiment/results/a800_qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618/`.
- Parameters:
  Model and tokenizer
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`;
  dataset `hotpotqa`; split `dev`; sample count `7405`; tensor parallel size
  `4`; `max_model_len=8192`; `max_new_tokens=32`; `batch_size=8`;
  `gpu_memory_utilization=0.34`; chat template enabled with the instruction
  `Answer with only the final short answer. Do not explain.`
- Result:
  Full Consumer HotpotQA `F1=0.7459153089`, `EM=0.5856853477`.
  Model load time was `1041.1066s`; eval time was `1253.9666s`; generation
  time was `1251.5142s`; output throughput was `29.5019` tokens/s. No OOM,
  `No space`, traceback, or runtime error was observed.
- Artifact handling:
  Copied `summary.json`, `per_sample_metrics.jsonl`, stdout/stderr logs, and
  `run_config.json` into the local result directory. Harvested the narrow local
  result directory with ResearchKB as project `KV Cache Reuse`; it recorded the
  summary and per-sample metrics.
- Next action:
  The same A800 run has started WikiText-2 evaluation and is currently loading
  the same NAS Qwen32 checkpoint through TP4. Wait for its PPL/NLL summary
  before closing the Qwen32 full-baseline coverage for the three-dataset set.

### 2026-06-18 15:36 +08:00 Qwen32 official-vLLM WikiText-2 full baseline completed

- Purpose:
  Complete the third Qwen2.5-32B-Instruct Full Consumer baseline dataset using
  official vLLM. This provides the language-modeling reference metric for
  future Qwen32 SCD comparisons; it is not a reuse/patch result.
- Run:
  Remote run root:
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618_143845`.
  Local artifact copy:
  `Experiment/results/a800_qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618/`.
- Parameters:
  Model and tokenizer
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct`;
  dataset `wikitext2`; split `test`; sample count `763`; tensor parallel size
  `4`; `max_model_len=2048`; `max_new_tokens=32`; `batch_size=16`;
  `gpu_memory_utilization=0.34`; prompt-NLL scoring enabled without chat
  template or extra instruction.
- Result:
  WikiText-2 Full Consumer `avg_nll=1.9711257546`, `PPL=7.1787534555`.
  The auxiliary generation-style score was `F1=0.2716968228`, `EM=0.0`, but
  the primary metric for this dataset is PPL/NLL. The NLL scorer used `29898`
  teacher tokens and had `prompt_logprob_missing_count=0`.
- Runtime:
  Model load time was `765.1147s`; eval time was `57.0315s`; generation time
  was `56.8122s`; NLL scoring time was `17.7976s`; output throughput was
  `429.7665` tokens/s. No OOM, `No space`, traceback, or runtime error was
  observed.
- Artifact handling:
  Copied `summary.json`, `per_sample_metrics.jsonl`, stdout/stderr logs into
  the local result directory. Harvested the narrow result directory with
  ResearchKB; it recorded the WikiText-2 summary and per-sample metrics.
- Consolidated baseline table:
  Wrote
  `Experiment/results/a800_qwen32_vllm_three_dataset_full_baselines_20260618/summary.json`
  and `summary.md`. Current Full Consumer baselines are:
  CMRC2018 `n=3219`, F1 `0.9098067394`, EM `0.7086051569`;
  HotpotQA `n=7405`, F1 `0.7459153089`, EM `0.5856853477`;
  WikiText-2 `n=763`, avg NLL `1.9711257546`, PPL `7.1787534555`.
- Next action:
  Treat these as Qwen32 Full Consumer reference rows only. Qwen32 SCD
  reuse/patch quality and speedup are still not established; the strict HF
  Qwen32 SCD queue remains constrained by available high-free-memory GPUs.

### 2026-06-18 15:44 +08:00 ser full-context HotpotQA diagnostic for CMRC-bank route

- Purpose:
  Check whether the current CMRC2018-trained MistralLite-to-Mistral-7B
  reuse19/patch13 bank transfers to full-context HotpotQA and WikiText-2. This
  is a diagnostic transfer check, not the compressed HotpotQA claim protocol.
- Remote run:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/extra_datasets_full_cmrcbank_20260618_121640`.
  Bank:
  `/disk_n/c101-3-mql/projects/SCD-ICML/runs/open_build_native_fixedroute_cmrc200_20260616_133222`.
  Route: reuse19/patch13, no fallback, no recompute.
- Local artifacts:
  Summary-level artifacts copied to
  `Experiment/results/ser_extra_datasets_full_cmrcbank_20260618/`.
  Large local per-sample copies were removed after ResearchKB harvest; the
  remote run directory keeps the original per-sample metrics.
- HotpotQA full-context result:
  Full Consumer `n=7405`, F1 `0.3539792271`, EM `0.2198514517`, NLL
  `0.9909363885`, PPL `2.6937556937`, TTFT `215.3274ms`, total
  `870.9411ms`.
  Reuse-only F1 `0.0084988530`, EM `0.0`, NLL `9.0839903929`, PPL
  `8813.0635398493`, TTFT `54.4804ms`, total `714.1467ms`.
  SCD F1 `0.1037241536`, EM `0.0779203241`, NLL `1.5248959037`, PPL
  `4.5946652566`, TTFT `60.8648ms`, total `132.1568ms`.
- WikiText-2 result:
  Full Consumer `n=763`, NLL `1.9838737528`, PPL `7.2708539936`;
  reuse-only NLL `6.8323367817`, PPL `927.3553094234`;
  SCD NLL `3.1681475597`, PPL `23.7634232360`.
- Interpretation:
  The selected CMRC-bank route remains much better than reuse-only on
  HotpotQA/WikiText-2, and it is substantially faster on HotpotQA total
  latency, but its quality is far below Full Consumer on both non-CMRC
  diagnostics. It should not be presented as a claim-facing cross-dataset
  generalization result. For HotpotQA claim support, use dataset-specific or
  compressed-protocol routes already separated in the release notes.
- Current status:
  CMRC2018 full-dev shards from the same run are still running; wait for them
  before writing the final aggregate summary for this full-dataset diagnostic.
