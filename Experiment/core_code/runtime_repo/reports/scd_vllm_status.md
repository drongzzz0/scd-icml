# SCD vLLM Integration Status

## Environment

- python: `3.10.20`
- platform: `Linux-6.11.0-21-generic-x86_64-with-glibc2.39`
- torch: `2.10.0+cu128`
- transformers: `4.57.6`
- vllm: `0.18.0`
- cuda_available: `True`

## Phase Results

- Phase 0 Repo Audit: PASS
- Phase 1 Env Probe: PASS
- Phase 2 HF Export: PASS - runs\scd_vllm\ser_hf_long_full_consumer_20260615\full_consumer tokens=229 layers=32 payload_bytes=30015488
- Phase 3 vLLM Baseline: PASS - runs\scd_vllm\ser_vllm_plain_baseline_20260615 prompt_tokens=14 max_new_tokens=16 mean_generation_seconds=0.2930715063897272 mean_ttft_seconds=None mean_ttft_proxy_seconds=0.02749731034661333
- Phase 4 TrueKV Oracle: PASS - runs\scd_vllm\ser_truekv_oracle_20260615 mode=truekv_oracle oracle_pass=True vllm=None save_files=32 load_files=32 greedy_1tok=True greedy_16tok=True blocker=None
- Phase 5 SCD Injection: PASS - runs\scd_vllm\ser_scd_injection_reuse_patch_cmrc_route_paperckpt_20260615 bundle_type=reuse_patch adapter_smoke_pass=True scd_injection_pass=True aligned_tokens=224 layers=32 payload_bytes=29362944 greedy_1tok=True greedy_ntok=True

## Blockers

- None recorded for completed scaffold phases.

## Reproduction Commands

```bash
bash scripts/scd_vllm/run_00_repo_audit.sh
bash scripts/scd_vllm/run_01_env_probe.sh
bash scripts/scd_vllm/run_02_hf_export.sh
bash scripts/scd_vllm/run_03_vllm_baseline.sh
bash scripts/scd_vllm/run_04_truekv_oracle.sh
bash scripts/scd_vllm/run_05_reconstructed_export.sh
bash scripts/scd_vllm/run_05_scd_injection.sh
bash scripts/scd_vllm/run_06_report.sh
```

## Next Steps

- Measure latency and quality for the passing reconstructed-SCD vLLM injection path.
