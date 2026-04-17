# Qwen3-1.7B Route C 对齐首 Token 计时结果

## 目的

这次复跑只回答一个问题：为什么先前 `full_consumer.decode_first_ms` 明显小于其它路径，以及把 `full_consumer` 改成和 transfer 路径相同的首 token 计时口径后，TTFT 分解会变成什么样。

对齐方式是让 `full_consumer` 也走：

- `prefix cache`
- `last prompt token`
- 一次真实的 cache-based first decode

而不再直接对 `prefill` 阶段已经得到的 `logits_last` 做一次 `argmax`。

## 关键结论

1. 先前的巨大差异主要来自**计时口径不一致**，不是 `full_consumer` 的 first decode 真比其它方法快一个数量级。
2. 对齐后，`full_consumer.decode_first_ms = 24.42 ms`，已经和其它方法的 `23.2 ~ 23.4 ms` 基本一致。
3. 对齐后，Route C 的 `reuse_patch_recompute` 与 `full_consumer` 的 TTFT 几乎打平：
   - `full_consumer = 69.33 ms`
   - `reuse_patch_recompute = 69.83 ms`
4. 但更强的实用基线仍然是 `reuse_fallback`：
   - `TTFT = 60.99 ms`
   - `F1 = 0.6605`
5. 因此，**“full_consumer 的 decode-first 特别小”这个现象是测量定义造成的**；但**Route C 仍然没有成为最优实用路径**。

## 旧口径 vs 对齐口径

| method | F1(old) | F1(aligned) | TTFT old (ms) | TTFT aligned (ms) | decode_first old (ms) | decode_first aligned (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_consumer` | 0.6945 | 0.6904 | 37.09 | 69.33 | 0.09 | 24.42 |
| `reuse_fallback` | 0.6605 | 0.6605 | 54.10 | 60.99 | 23.26 | 23.21 |
| `reuse_patch_recompute` | 0.6195 | 0.6195 | 64.71 | 69.83 | 23.07 | 23.32 |

## 同口径 TTFT 分解

### `full_consumer`

- `consumer_prefill_ms = 44.91`
- `decode_first_ms = 24.42`
- `ttft_ms = 69.33`

### `reuse_fallback`

- `encode_ms = 1.59`
- `transfer_ms = 0.43`
- `rebuild_ms = 2.29`
- `recompute_ms = 33.46`
- `decode_first_ms = 23.21`
- `ttft_ms = 60.99`

### `reuse_patch_recompute` (Route C)

- `encode_ms = 1.60`
- `transfer_ms = 0.80`
- `rebuild_ms = 2.32`
- `patch_infer_ms = 11.04`
- `recompute_ms = 30.75`
- `decode_first_ms = 23.32`
- `ttft_ms = 69.83`

## 解释

旧实现里：

- `full_consumer` 的 `decode_first_ms` 只计 `argmax(logits_last)`
- reuse 路径的 `decode_first_ms` 计的是一次真实前向

所以旧结果里的差值并不能解释成模型推理能力差异，只能解释成**计时边界放置不同**。

对齐之后可以更清楚地看到：

- 真正拖慢 Route C 的不是 first decode
- 而是 `patch_infer_ms + recompute_ms`
- 其中主要瓶颈仍是 `recompute_ms`

## 实用判断

在这组 `Qwen3-1.7B -> Qwen3-1.7B-bf16` 的近邻对上：

- 如果只问“decode-first 是否本应差不多”，答案是**是的**
- 如果问“Route C 对齐后是否已经优于 full_consumer”，答案是**没有**
- 如果问“当前最实用的路径是什么”，答案仍然是 **`reuse_fallback`**

## 产物

- 对齐复跑汇总：`Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_aligned_main200/summary.json`
- 对齐复跑比较：`Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_aligned_main200/final_comparison.json`
- 原始 Route C 汇总：`Experiment/analysis/remote_artifacts/qwen17_to_bf16_cmrc_clean_routec_state_recompute_main200/summary.json`
