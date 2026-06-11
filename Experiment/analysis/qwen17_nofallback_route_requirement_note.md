# Qwen3-1.7B Route A/B/C 无 Fallback 约束说明

## 结论

在用户要求的定义下，先前已经跑出的 Route A/B/C 结果不能再作为有效路线结论使用，因为那些 run 并不是纯 `reuse + patch (+ recompute)`，而是混入了 route 自带的 `fallback_layers`。

## 为什么旧结果不满足要求

当前严格 runtime 对 `reuse_patch` 和 `reuse_patch_recompute` 的处理不是“自动禁止 fallback”，而是：

1. 从 route JSON 读取 `fallback_layers`
2. 先执行 reuse / patch / recompute
3. 最后再把 `fallback_layers` overlay 回 cache

对应代码：

- route 归一化保留 `fallback_layers`：
  `Experiment/analysis/remote_patch_work/strict_runtime.py:705`
- Route C 也保留 `fallback_layers`：
  `Experiment/analysis/remote_patch_work/strict_runtime.py:710`
- 真执行结束后再 overlay fallback：
  `Experiment/analysis/remote_patch_work/strict_runtime.py:1543`

因此，旧的 Route A/B/C 结果只能解释为“带 fallback 底座的 patch/recompute 变体”，不能解释为用户要求的纯路线。

## 现在应使用的 no-fallback 产物

- no-fallback route：
  `Experiment/analysis/route_variants/qwen17_cmrc_seed42_no_fallback_route.json`
- Route A no-fallback eval config：
  `Experiment/core_code/configs/qwen17_route_a_state_native_nofallback_eval.json`
- Route B no-fallback eval config：
  `Experiment/core_code/configs/qwen17_route_b_direct_kv_nofallback_eval.json`
- Route C no-fallback eval config：
  `Experiment/core_code/configs/qwen17_route_c_state_recompute_nofallback_eval.json`

## 这套 no-fallback 配置的语义

这套 route 删除了 `fallback_layers`。在当前 runtime 里，这意味着：

- `patch_layers` 仍按 route 指定执行
- 其余非 patch 层都会按 `reuse` 处理
- Route C 只保留 `reuse + patch + recompute`
- 不再额外执行 `fallback` overlay

另外，三个 no-fallback eval config 都显式关闭了：

- `runtime_eval.run_reuse_fallback = false`

这样后续结果不会再把 `reuse_fallback` 混进路线结论里。
