# Qwen3-1.7B 近邻对：同口径 TTFT 总表

## 目的

这份表把三个关键运行都改成同一个 first-token 计时口径后再比较：

- clean original semantic patch
- Route B direct-KV
- Route C patch + recompute

统一口径指：

- `full_consumer` 不再把 first token 视为 `logits_last -> argmax`
- 改为和 transfer 路径一致：`prefix cache + last prompt token + one real cache-based decode step`

## 结论先行

1. `full_consumer.decode_first_ms` 在旧口径下明显偏小，确实是**计时边界问题**。
2. 对齐后，`full_consumer.decode_first_ms` 变为 `24.4 ~ 27.6 ms`，和 transfer 路径的 `23.2 ~ 23.8 ms` 同一量级。
3. 这说明此前 `full_consumer` 看起来“首 token 极快”并不是真实的 decode 优势。
4. 但对齐后，实用结论没有翻转：
   - `reuse_fallback` 仍然是最强实用路径
   - Route C 仍被 `reuse_fallback` 支配
   - clean original patch 与 Route B patch 虽然 TTFT 低于同口径 `full_consumer`，但质量损失仍然过大

## 同口径总表

| experiment | method | F1 old | F1 aligned | TTFT old (ms) | TTFT aligned (ms) | decode_first old (ms) | decode_first aligned (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean original | `full_consumer` | 0.6945 | 0.6904 | 58.81 | 71.64 | 0.63 | 25.03 |
| clean original | `reuse_fallback` | 0.6605 | 0.6605 | 70.98 | 61.51 | 23.78 | 23.29 |
| clean original | `reuse_patch` | 0.5010 | 0.5010 | 75.84 | 65.52 | 23.80 | 23.27 |
| Route B | `full_consumer` | 0.6945 | 0.6904 | 45.42 | 70.83 | 0.31 | 27.62 |
| Route B | `reuse_fallback` | 0.6605 | 0.6605 | 62.04 | 60.67 | 23.54 | 23.69 |
| Route B | `reuse_patch` | 0.4951 | 0.4951 | 54.22 | 55.61 | 23.55 | 23.75 |
| Route C | `full_consumer` | 0.6945 | 0.6904 | 37.09 | 69.33 | 0.09 | 24.42 |
| Route C | `reuse_fallback` | 0.6605 | 0.6605 | 54.10 | 60.99 | 23.26 | 23.21 |
| Route C | `reuse_patch` | 0.0000 | 0.0000 | 51.40 | 57.26 | 23.10 | 23.27 |
| Route C | `reuse_patch_recompute` | 0.6195 | 0.6195 | 64.71 | 69.83 | 23.07 | 23.32 |

## 对齐后的主要比较

### clean original

- `full_consumer`: `71.64 ms`, `F1 = 0.6904`
- `reuse_fallback`: `61.51 ms`, `F1 = 0.6605`
- `reuse_patch`: `65.52 ms`, `F1 = 0.5010`

解释：

- 对齐后 clean original patch 不再是“又慢又差”
- 它现在是**更快但质量明显更差**
- 仍不能达到论文希望的速度-质量 operating point

### Route B

- `full_consumer`: `70.83 ms`, `F1 = 0.6904`
- `reuse_fallback`: `60.67 ms`, `F1 = 0.6605`
- `reuse_patch`: `55.61 ms`, `F1 = 0.4951`

解释：

- Route B patch 的 TTFT 优势是真实存在的
- 但代价仍然是过大的质量损失
- Route B 依然不如 `reuse_fallback` 实用

### Route C

- `full_consumer`: `69.33 ms`, `F1 = 0.6904`
- `reuse_fallback`: `60.99 ms`, `F1 = 0.6605`
- `reuse_patch_recompute`: `69.83 ms`, `F1 = 0.6195`

解释：

- 对齐后，Route C 不再比 `full_consumer` 慢很多
- 它和 `full_consumer` 基本打平：`69.83 vs 69.33 ms`
- 但它仍然低于 `reuse_fallback`：更慢、也更差

## 关键解读

真正被这次对齐修正的，是下面这个结论：

- **旧结论（不严谨）**：`full_consumer` 的 first decode 特别快，所以其它路线在 first decode 上吃了很大亏
- **新结论（同口径）**：first decode 基本一样，差异主要来自 `patch_infer`、`recompute` 和整体 prefill / rebuild / encode 开销

因此，当前瓶颈判断应改成：

1. `decode_first` 不是主要矛盾
2. Route C 的主要成本仍然是 `patch_infer + recompute`
3. clean original / Route B 的主要问题仍然是**质量不足**
4. 当前最强实用基线仍然是 `reuse_fallback`

## 当前最佳实用路径

在这组 `Qwen3-1.7B -> Qwen3-1.7B-bf16` 近邻对上，按同口径比较：

- 最优实用路径：`reuse_fallback`
- 原因：它保持了最高的实用 F1 / TTFT 折中
- Route C：不再被 `full_consumer` 明显拉开，但仍然不优于 `reuse_fallback`
- clean original 与 Route B patch：速度可以更快，但质量损失仍不足以支撑论文宣称

## 产物

- clean aligned: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_orig_aligned/summary.json`
- Route B aligned: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routeb_directkv_aligned_main200/summary.json`
- Route C aligned: `Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_aligned_main200/summary.json`
- Route C 单独分析: `Experiment/analysis/qwen17_routec_aligned_ttft_breakdown.md`
