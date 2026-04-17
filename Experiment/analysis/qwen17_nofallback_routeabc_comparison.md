# Qwen3-1.7B -> Qwen3-1.7B-bf16 Route A/B/C No-Fallback 结果

## 设置

- 数据: CMRC, 200 samples
- 路由: `22` 层 reuse, `6` 层 patch, `0` 层 fallback
- 评测口径: `align_full_consumer_first_decode = true`
- 说明: 各 run 的 `full_consumer` TTFT 会因设备组合不同而有差异，因此路线判断以同 run 内的 `TTFT / full_consumer` 归一化比值为准。

## 主结果

| Route | Transfer mode | F1 | EM | TTFT (ms) | Same-run full F1 | Same-run full TTFT (ms) | TTFT / full | F1 retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A state->state->native KV | `reuse_patch` | 0.0000 | 0.0000 | 33.761 | 0.6904 | 73.135 | 0.462x | 0.000x |
| B state->direct KV | `reuse_patch` | 0.0000 | 0.0000 | 35.691 | 0.6904 | 86.989 | 0.410x | 0.000x |
| C state->state + short recompute | `reuse_patch_recompute` | 0.0000 | 0.0000 | 46.048 | 0.6904 | 66.756 | 0.690x | 0.000x |

## 补充观察

- Route A no-fallback 完全崩溃。虽然 `reuse_patch` 的 TTFT 只有 `33.761 ms`，但 F1 为 `0.0000`。
- Route B no-fallback 同样无法救回质量。它比 Route A 还快一些，但 F1 仍为 `0.0000`。
- Route C no-fallback 也没有复现旧的“质量修复”现象。`reuse_patch_recompute` 的平均 logp 比 Route A/B 好一些，但最终生成 F1 仍为 `0.0000`。
- 因此，在用户要求的“纯 `reuse + patch (+ recompute)`、绝不 fallback”定义下，当前 Qwen3-1.7B 近邻对上的 Route A/B/C 都不能达到论文宣称效果，甚至都不能维持可用任务质量。

## 本地同步产物

- Route A: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routea_state_native_nofallback_main200/`
- Route B: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_nofallback_main200/`
- Route C: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_nofallback_main200/`
