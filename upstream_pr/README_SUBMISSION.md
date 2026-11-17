# Submitting `SCDConnectorV1` to `vllm-project/vllm`

This directory contains everything needed to open a clean upstream Pull Request against [`vllm-project/vllm`](https://github.com/vllm-project/vllm) while also working out-of-the-box on **unmodified** vLLM releases via `vllm.general_plugins`.

---

## 1. Dual Integration Architecture

1. **Immediate Zero-Fork Plugin Mode (Works Today on Unmodified vLLM >= 0.18)**:
   Installing `scd-icml` (`pip install -e .`) registers `[project.entry-points."vllm.general_plugins"]` (`scd_icml = "scd_vllm.plugin:register"`). Every vLLM process (`LLM`, `AsyncLLM`, `vllm serve`, `EngineCore`, and GPU `Worker`) automatically calls `load_general_plugins()` at startup and registers:
   - `SCDConnectorV1`
   - `SCDGPUConnectorV1`
   - `SCDPreloadedConnector`

2. **Upstream Native Registration (`upstream_pr/`)**:
   Follows the official thin-wrapper pattern used by `FlexKVConnectorV1` and `LMCacheConnectorV1`:
   - `vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py` (~160 lines, Apache-2.0 header, delegates to `scd_vllm.vllm_v1_adapter.SCDConnectorV1Impl`)
   - `vllm/distributed/kv_transfer/kv_connector/factory.py` (4-line lazy `KVConnectorFactory.register_connector("SCDConnectorV1", ...)`)
   - `docs/features/disagg_prefill.md` (concise usage entry)
   - `examples/disaggregated/scd_connector/` (minimal runnable script + README)
   - `tests/v1/kv_connector/unit/test_scd_connector_v1.py` (self-contained unit test requiring no external dependencies in vLLM CI)

---

## 2. Step-by-Step PR Submission Commands (Human Review & DCO Required)

Per vLLM's contribution policy (`CONTRIBUTING.md` / `AGENTS.md`), every commit must be **human-reviewed** and carry your **Developer Certificate of Origin (`Signed-off-by`)** via `git commit -s`.

```bash
# 1. Clone or fetch latest upstream vllm main
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout -b feat/scd-connector-v1 origin/main

# 2. Apply the upstream PR bundle (or git apply scd_connector_v1.patch)
python /path/to/scd-icml-vllm-open/upstream_pr/apply_to_vllm_checkout.py --vllm-root .

# 3. Run upstream unit test & ruff formatting checks
pytest tests/v1/kv_connector/unit/test_scd_connector_v1.py -q
ruff check vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py \
           vllm/distributed/kv_transfer/kv_connector/factory.py \
           tests/v1/kv_connector/unit/test_scd_connector_v1.py \
           examples/disaggregated/scd_connector/cross_model_kv_handoff.py
ruff format --check vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py \
                    tests/v1/kv_connector/unit/test_scd_connector_v1.py

# 4. Review git diff personally, commit with DCO sign-off (-s), and push to your fork
git status
git diff
git add vllm/distributed/kv_transfer/kv_connector/v1/scd_connector.py \
        vllm/distributed/kv_transfer/kv_connector/factory.py \
        docs/features/disagg_prefill.md \
        examples/disaggregated/scd_connector \
        tests/v1/kv_connector/unit/test_scd_connector_v1.py
git commit -s -m "[KVConnector] Add SCDConnectorV1 for cross-model Semantic Cache Distillation"
git push -u my-fork feat/scd-connector-v1
```

Then open the Pull Request on GitHub using [`PR_BODY.md`](./PR_BODY.md) as the description.
