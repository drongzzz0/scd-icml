# 当前进展与问题总结

> 2026-04-04 口径更新:
> `reuse` 和 `patch` 路线都已经锁定。
> 旧的 grouped/shared-z reuse、旧 TTFT 记账、任何带 fallback/recompute 的旧 transfer 路线、以及旧 patch 变体，全部只保留为排障历史，不再作为当前结论依据。
> 后续默认只使用:
> `baseline`
> `reuse-only = layerwise + group_size=1 + batched + actual-transfer + no-fallback + no-recompute`
> `patch-only = strict patch_only + no reuse scaffold + no-fallback + no-recompute + shared nativekv patch + K-only rotary + fused KV + fused K-only knorm`

日期: `2026-04-01`

## 1. 当前目标

本项目当前要回答的问题是:

1. 原始 SCD 方法用在更相似的模型对上，能否达到论文宣称的 TTFT-质量效果。
2. 如果原始方法做不到，改进路线 A/B/C 或新的 patch 设计能否达到论文宣称效果。

当前优先模型线是 `Qwen` 系列，数据使用 `SCD` 论文同源设置，当前主要可运行诊断对是:

- `Qwen3-1.7B -> Qwen3-1.7B-bf16`
- `Qwen3-1.7B -> Qwen3-1.7B` self-self

## 2. 已完成的关键工作

### 2.1 评价标准和实验脚手架

已经完成:

- claim target 定义
- baseline protocol 固化
- TTFT/质量/claim-gap 评价合同
- 本地实验脚手架与远端 `ser` 运行包装

这部分已经足够支持“先判定是否复现论文主张，再判定改进路线是否救回”的实验流程。

### 2.2 Qwen3.5 资源获取

已经在 `ser` 上下载并确认:

- `Qwen/Qwen3.5-4B-Base`
- 本地已有 `Qwen3.5-4B`

所以 Qwen3.5 的“缺模型”问题已经关闭。

### 2.3 Qwen3.5 faithful baseline 阻塞点定位

真实 smoke run 已经证明: 当前 repo 不能直接跑 faithful 的 `Qwen3.5-4B-Base -> Qwen3.5-4B` 原始 SCD baseline。

原因不是 checkpoint 缺失，而是 **状态语义不兼容**:

- `Qwen3.5` 是 hybrid cache 架构
- `32` 层中 `24` 层是 `linear_attention`
- 只有 `8` 层是 `full_attention`
- cache 不仅有 `key_cache/value_cache`，还有 `conv_states/recurrent_states`
- 当前 SCD repo 默认假设“每层都只有标准 KV cache”

因此:

- 现在不能把“Qwen3.5 没跑出来”解释成论文方法失败
- 这首先是 repo 与 Qwen3.5 hybrid state 不兼容

## 3. 已确认的实证结果

### 3.1 `Qwen3-1.7B -> Qwen3-1.7B-bf16` no-fallback 路线结果

在用户要求的严格定义下:

- 不允许 fallback
- 只允许 `reuse + patch (+ recompute)`

Route A/B/C 都失败。

| 路线 | 传输模式 | F1 | TTFT |
|---|---|---:|---:|
| Route A state->state->native KV | `reuse_patch` | 0.0000 | 33.761 ms |
| Route B state->direct KV | `reuse_patch` | 0.0000 | 35.691 ms |
| Route C state->state + recompute | `reuse_patch_recompute` | 0.0000 | 46.048 ms |

结论:

- 在当前可运行近邻对上，Route A/B/C 都不能复现论文宣称效果
- 速度即使更快，也没有可用任务质量

### 3.2 `Qwen3-1.7B -> Qwen3-1.7B` self-self reuse 诊断

这里主要回答“是不是 reuse 本身实现有问题”。

结果:

| reuse checkpoint | group_size | reuse_only F1 | full_consumer F1 |
|---|---:|---:|---:|
| grouped shared-z | 4 | 0.0019 | 0.7375 |
| layerwise | 1 | 0.7372 | 0.7375 |

结论:

- self-self 下，`layerwise reuse` 近乎无损
- 崩掉的是 `grouped shared-z reuse codec`
- 所以后续所有 self-self patch 诊断都应以 `layerwise reuse` 为默认 baseline

### 3.2b self-self layerwise reuse 并行化加速已经跑通

用户要求先不做压缩，只把 `encode/decode` 并行处理，把 reuse 链路跑通并加速。这个链路已经完成，并且在远端做了新的严格 self-self rerun。

这次改动的核心不是改 checkpoint，而是把原来 `group_size=1`、`28` 个单层 group 的串行 encode/decode 循环，改成 batched `bmm` 版本；payload 仍然保持原始大小，不做压缩。

新旧对比如下:

| layerwise reuse 版本 | reuse_only F1 | full_consumer F1 | reuse_only TTFT | full_consumer TTFT | encode | transfer | rebuild | decode_first | payload | compression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧串行版 | 0.7372 | 0.7375 | 65.48 ms | 43.86 ms | 26.86 ms | 1.73 ms | 11.89 ms | 25.00 ms | 41.32 MB | 1.0 |
| 新 batched 并行版 | 0.7372 | 0.7375 | 40.67 ms | 45.05 ms | 0.88 ms | 1.73 ms | 12.51 ms | 25.55 ms | 41.32 MB | 1.0 |

结论:

- `reuse_only` 之前慢，不是因为传输开销大。`transfer_ms` 一直只有约 `1.73 ms`。
- 主要瓶颈是旧实现里逐层串行的 encode 计算。
- batched 之后，`encode_ms` 从 `26.86 -> 0.88 ms`，`reuse_only TTFT` 从 `65.48 -> 40.67 ms`。
- 在这次同 run 对比里，`reuse_only` 已经略快于 `full_consumer` (`40.67 ms < 45.05 ms`)。
- 当前 remaining bottleneck 已经不再是 encode/transfer，而是 `rebuild_ms` 和 `decode_first_ms`。
- 这一步是**纯并行化加速**，不是压缩收益；`payload` 仍等于原始 cache 大小。

### 3.3 self-self 线性 patch 修复

对 self-self patch 路线，已经完成两次关键修复:

1. pair capture 不再被错误截断到 `256` token
2. Qwen 路线从错误的 direct-KV 目标改为 `direct_kv_knorm_v1 + consumer-native rotary`

修复后的严格结果:

| 路线 | F1 | EM | TTFT |
|---|---:|---:|---:|
| `patch 0..27` | 0.0119 | 0.00 | 75.35 ms |
| `patch 1..27` | 0.7425 | 0.44 | 88.10 ms |

结论:

- 这证明 patch 路线不是“整体不可能”
- 当前线性 patch 在 self-self 上 **只有排除 `layer 0` 时才健康**

### 3.4 `layer 0` 控制实验

为了确认 `layer 0` 到底是 runtime 问题还是学习器问题，已经做了专门控制。

确认了以下事实:

- self-self 下 `xnorm_a.0 == xnorm_b.0`
- 用 consumer-native `k_proj/v_proj + k_norm + native rotary` 可以几乎完美重建真实 layer-0 KV
- `oracle_kv_patchonly` on `layer 0` 能保持 `reuse_only` 质量
- 真正坏的是学习得到的 `layer 0` translator

严格 route-level 控制结果:

| 模式 | F1 | EM | TTFT |
|---|---:|---:|---:|
| `reuse_only` | 0.7372 | 0.38 | 71.19 ms |
| learned `layer 0` patch only | 0.0124 | 0.00 | 51.51 ms |
| `oracle_kv_patchonly` | 0.7372 | 0.38 | 49.13 ms |

结论:

- `layer 0` 在原理上是可 patch 的
- 当前坏的是学习到的 `direct_kv_knorm_v1` layer-0 解
- 这不是 runtime cache overwrite 问题

### 3.5 self-self 非线性 MLP patch 跟进

用户要求把 patch 换成非线性的 `MLP`，已经执行完成。

具体使用的是 repo 里现成的:

- `fit_mode = xnorm_semantic_v1`
- `PatchMLP`
- `state_target_space = xnorm`
- `patch_apply_mode = overwrite`

严格结果:

| 路线 | F1 | EM | TTFT |
|---|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 59.10 ms |
| `reuse_only` | 0.7372 | 0.38 | 103.07 ms |
| `MLP patch 0..27` | 0.0000 | 0.00 | 102.16 ms |
| `MLP patch 1..27` | 0.0000 | 0.00 | 126.85 ms |

训练侧也显示:

- 只有 `layer 0..2` hidden fit 较好
- 从 `layer 3` 开始迅速恶化
- 深层 hidden/K/V 辅助误差都很大

结论:

- 当前 repo 里的现成非线性 `MLP patch` 路线 **完全不能用**
- 它比修复后的线性 `direct_kv_knorm_v1 excl0` 更差
- 因此“把 patch 换成非线性”这个动作本身，当前不能作为有效改进

### 3.6 self-self 更强 nonlinear residual direct-KV patch 跟进

在现成 `PatchMLP` 失败后，又做了一条更强的非线性路线:

- `fit_mode = direct_kv_knorm_residual_mlp_v1`
- 所有层先保留修复后的线性 `direct_kv_knorm_v1`
- 只对唯一坏掉的 `layer 0` 叠加 residual MLP
- residual trunk 宽度 `4096`，`4` 个残差块，`12` 个 epoch

第一次远端运行暴露了一个实现 bug:

- residual 数据预加载时，`xnorm_b` 在 CPU
- 原生 `k_proj/v_proj` 权重却还在 `cuda:1`
- 导致 `F.linear` 设备不一致报错

这个 bug 已经修复并完成 `v2` 重跑。

严格结果:

| 路线 | F1 | EM | TTFT |
|---|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 56.41 ms |
| `reuse_only` | 0.7372 | 0.38 | 84.25 ms |
| residual patch `0..27` | 0.0087 | 0.00 | 193.11 ms |
| residual patch `1..27` | 0.7425 | 0.44 | 122.10 ms |

`layer 0` 专门控制也没有改善:

| 模式 | F1 | EM | TTFT |
|---|---:|---:|---:|
| learned residual `layer 0` patch only | 0.0113 | 0.00 | 177.57 ms |
| `oracle_kv_patchonly` | 0.7372 | 0.38 | 68.06 ms |

训练指标也说明这个更强的非线性几乎没有学到有效修正:

- `base_k_rel_mse`: `1.213e7 -> 1.199e7`
- `base_v_rel_mse`: `2.330e7 -> 2.184e7`

结论:

- 更强的 direct-KV residual MLP 仍然**没有**救回 `layer 0`
- `all28` 结果比修复后的线性路线更差而且更慢
- `excl0` 质量虽然仍然健康，但只是复现了原来的线性健康点，没有带来任何收益
- 因此当前 self-self 上唯一健康的 patch 结果，仍然只有线性 `direct_kv_knorm_v1 + patch 1..27`

### 3.7 self-self hidden-residual native-KV 路线成功

在上面的 direct-KV 非线性路线继续失败后，又改成了另一种参数化:

- `fit_mode = xnorm_nativekv_residual_mlp_v1`
- 只在坏掉的 `layer 0` 上学习 hidden residual:
  - `xb_hat = xa + delta_h`
- 然后再走 consumer-native:
  - `k_proj/v_proj`
  - `k_norm`
  - native rotary

这条路线和前面失败路线的关键区别是:

- 不再直接学 `K/V`
- 而是从 self-self 本来就正确的 identity hidden 出发

训练指标直接说明这条参数化抓住了问题本质。`layer 0` 的保存指标为:

- `linear_base_k_rel_mse = 1497854.0816`
- `linear_base_v_rel_mse = 531215.5905`
- `base_h_rel_mse = 0.0`
- `base_k_rel_mse = 1.43e-11`
- `base_v_rel_mse = 1.55e-11`

也就是说:

- direct-KV 旧参数化在 `layer 0` 上天然是坏的
- 但 hidden residual 的 identity 初始化本身就是正确解
- 最优 checkpoint 实际停在初始化点

严格结果:

| 路线 | F1 | EM | TTFT |
|---|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 51.03 ms |
| `reuse_only` | 0.7372 | 0.38 | 103.67 ms |
| hidden-residual patch `0..27` | 0.7700 | 0.46 | 200.01 ms |
| hidden-residual patch `1..27` | 0.7425 | 0.44 | 123.42 ms |

`layer 0` 专门控制也已经恢复正确:

| 模式 | F1 | EM | TTFT | Patch infer |
|---|---:|---:|---:|---:|
| hidden residual `layer 0` patch only | 0.7372 | 0.38 | 193.12 ms | 127.91 ms |
| `oracle_kv_patchonly` | 0.7372 | 0.38 | 63.45 ms | 0.00 ms |

结论:

- `layer 0` 现在已经被**质量上修好**
- 旧失败不是 runtime cache overwrite 问题，而是 **direct-KV 参数化问题**
- 当前新瓶颈已经从 correctness 转成 efficiency:
  - 选中的 checkpoint 实际上接近 identity
  - 但 runtime 仍在执行 residual MLP
  - 所以 `patch_infer_ms` 非常大

### 3.8 hidden-residual 路线 runtime identity skip 跟进

在确认 `layer 0` 的最优 checkpoint 实际停在 identity 初始化后，又做了一次只改 runtime 的优化:

- 检测 `delta_h_head.weight/bias` 是否全零
- 如果是，则跳过 residual MLP forward
- 直接对 hidden state 走 consumer-native `k_proj/v_proj + k_norm + native rotary`

这一步不改训练结果，只移除无意义的 residual MLP 推理。

严格结果:

| 路线 | F1 | EM | TTFT | Patch infer |
|---|---:|---:|---:|---:|
| hidden-residual patch `0..27` | 0.7700 | 0.46 | 167.74 ms | 97.44 ms |
| hidden-residual patch `1..27` | 0.7425 | 0.44 | 126.55 ms | 46.49 ms |
| hidden-residual `layer 0` patch only | 0.7372 | 0.38 | 140.55 ms | 74.26 ms |

与优化前相比:

- `all28`: `TTFT 200.01 -> 167.74 ms`, `patch_infer 132.23 -> 97.44 ms`
- `layer0-only`: `TTFT 193.12 -> 140.55 ms`, `patch_infer 127.91 -> 74.26 ms`

结论:

- identity skip 是有效的
- 但它只去掉了“无意义 residual MLP”这一段
- 即使不跑 MLP，`layer 0` 仍然要付出较大的 native `hidden -> KV` 投影代价
- 所以当前 bottleneck 已进一步收缩为 consumer-native `k_proj/v_proj + k_norm + rotary`

### 3.9 `layer 0` patch 细分 profiling

为了确认 `layer 0` 的 patch-infer 大头到底是不是 `k_proj/v_proj`，又对 `layer0-only` 路线做了一次带同步计时的 profiling rerun。

注意:

- 这次 rerun 是分析用
- 因为额外插入了同步点，所以绝对 `patch_infer_ms` 不用于 headline 对比
- 有效信息是内部构成比例

细分结果如下:

| 组成 | 平均耗时 |
|---|---:|
| identity check | 73.92 ms |
| residual forward | 0.00 ms |
| `k_proj/v_proj` | 0.20 ms |
| `k_norm` | 0.18 ms |
| native rotary | 0.68 ms |

结论:

- 这次 profiling 说明 native `hidden -> KV` 链路本身只占约 `1 ms`
- 但后续逐样本日志审计又进一步发现: 这里的 `73.92 ms` 平均值主要来自**首样本热启动**
- 也就是说，大均值不是 steady-state 每样本都在付，而是 runtime patch module 的首次 materialize 被算进了第一个样本
- 所以下一步最直接的优化不是继续压 `k_proj/v_proj`，而是把 runtime patch module 预热挪到计时外面

### 3.10 hidden-residual 路线 runtime prewarm 跟进

在把 identity 判断改成 load-time 缓存之后，又继续做了一个真正影响均值的 runtime 优化:

- 对选中的 patch layers 先做一次 runtime module prewarm
- 把 patch module 首次加载/materialize 从计时样本外移走
- 保留 identity skip，不改训练结果

先在最干净的 `layer0-only` 严格路线上验证。

严格结果:

| 路线 | F1 | EM | TTFT | Patch infer |
|---|---:|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 51.29 ms | 0.00 ms |
| `reuse_only` | 0.7372 | 0.38 | 90.78 ms | 0.00 ms |
| hidden-residual `layer 0` patch only + prewarm | 0.7372 | 0.38 | 73.67 ms | 1.42 ms |
| `oracle_kv_patchonly` | 0.7372 | 0.38 | 63.94 ms | 0.00 ms |

对比前几版:

- `identity skip`: `TTFT 140.55 ms`, `patch_infer 74.26 ms`
- `load-time identity flag only`: `TTFT 143.22 ms`, `patch_infer 74.33 ms`
- `prewarm`: `TTFT 73.67 ms`, `patch_infer 1.42 ms`

逐样本分布也已经说明问题本质:

- `loadid` 版:
  - `patch_infer` 中位数 `1.40 ms`
  - 最大值 `3646.50 ms`
- `prewarm` 版:
  - `patch_infer` 中位数 `1.39 ms`
  - 最大值 `1.95 ms`

结论:

- `layer 0` hidden-residual 路线的 steady-state patch 开销本来就只有约 `1.4 ms`
- 之前 `74 ms` 级均值几乎完全是首样本热启动假象
- 加上 prewarm 后，`layer0-only` 路线已经从“质量正确但极慢”变成“质量正确且速度接近 oracle”
- 当前真正还没解决的速度问题，不再是 isolated `layer 0`，而是 **all-28-layer hidden-residual 路线** 还没有在同样的 prewarm 条件下重跑

### 3.11 hidden-residual `all28` prewarm rerun

在确认 `layer0-only` 的均值主要被首样本 warm-start 污染后，又把同样的 prewarm 机制直接应用到 `all28` hidden-residual 路线。

严格结果:

| 路线 | F1 | EM | TTFT | Patch infer |
|---|---:|---:|---:|---:|
| `full_consumer` | 0.7375 | 0.38 | 50.30 ms | 0.00 ms |
| `reuse_only` | 0.7372 | 0.38 | 88.33 ms | 0.00 ms |
| hidden-residual patch `0..27` + prewarm | 0.7700 | 0.46 | 99.14 ms | 33.26 ms |

与上一版 `identity skip` 对比:

- `F1`: `0.7700 -> 0.7700`
- `EM`: `0.46 -> 0.46`
- `TTFT`: `167.74 -> 99.14 ms`
- `patch_infer`: `97.44 -> 33.26 ms`

逐样本分布:

- `skipid`:
  - `patch_infer` 中位数 `25.37 ms`
  - 最大值 `3596.80 ms`
  - `TTFT` 中位数 `90.83 ms`
  - 最大值 `3664.12 ms`
- `prewarm`:
  - `patch_infer` 中位数 `32.01 ms`
  - 最大值 `40.06 ms`
  - `TTFT` 中位数 `96.54 ms`
  - 最大值 `114.91 ms`

结论:

- `all28` 路线的 headline 均值也确实被首样本 warm-start 污染过
- prewarm 把这个尖峰移出计时后，均值 `TTFT` 大幅下降，且质量完全不变
- 但 unlike `layer0-only`，`all28` 的 steady-state patch 开销本身仍然不小，均值 `patch_infer` 还在 `33.26 ms`
- 所以现在的状态是:
  - correctness 已稳定
  - warm-start 假象已清掉
  - 真正剩下的优化对象已经变成 **all28 steady-state patch path** 本身

## 4. 当前已经可以明确写下来的结论

### 4.1 已经可以稳定确认的部分

以下内容已经足够稳定，可以作为当前阶段结论:

1. 在当前可运行的 `Qwen3-1.7B -> Qwen3-1.7B-bf16` 近邻对上，原始/改进路线在 strict no-fallback 定义下都未达到论文宣称效果。
2. Route A/B/C 在 strict no-fallback 下全部 `F1 = 0.0`。
3. `grouped shared-z reuse` 是坏 baseline，不应再用于 claim 判断。
4. 现成的非线性 `xnorm_semantic_v1 / PatchMLP` 路线在 self-self 上也失败，不能作为救援方案。
5. 更强的 `direct_kv_knorm_residual_mlp_v1` 也失败了: 它既没救回 `layer 0`，也没有提升健康的 `excl0` 路线。
6. 但新的 `xnorm_nativekv_residual_mlp_v1` 已经证明 `layer 0` 本身是可修复的，旧 direct-KV collapse 来自参数化而不是 runtime 覆盖逻辑。
7. 在此基础上加入 runtime identity skip 后，TTFT 确实下降，但仍未达到一个有竞争力的速度点。
8. self-self `layerwise reuse` 的高 TTFT 已被定位并修复: 问题不在传输，而在串行 encode/decode 实现；batched fast path 已把 `reuse_only` TTFT 压到 `40.67 ms`，并在同 run 中略快于 `full_consumer 45.05 ms`。

### 4.2 当前仍保留的“未被最终否定”的部分

以下内容还不能直接判死:

1. faithful 的 `Qwen3.5 -> Qwen3.5` 原始 SCD baseline 还没有真正跑通，因为 repo 先被 hybrid cache 结构卡住。
2. self-self 线性 patch 的 `patch 1..27` 结果说明，某些 patch 设计在高相似条件下仍可能工作。
3. self-self hidden-residual `patch 0..27` 进一步说明，连原本坏掉的 `layer 0` 也不是不可 patch，只是原来的 direct-KV 参数化不对。
4. runtime identity skip 已经说明，当前 hidden-residual 路线的低效中确实有一部分来自“无意义 residual MLP 推理”，但去掉这部分后仍然不够快。
5. 因此“论文主张绝对错误”现在还不能写死，更准确的说法是:
   - 在当前可运行设置下没有被复现
   - 当前 repo 对 Qwen3.5 faithful baseline 不兼容
   - 当前 A/B/C、现成 MLP 路线、以及更强的 residual direct-KV 路线都没有把它救回来
   - 但 self-self hidden-residual 路线已经把 correctness 修到位，只是 TTFT 还远不够好

## 5. 当前的核心问题

目前最核心的问题不是一个，而是四个:

### 问题 1: faithful Qwen3.5 baseline 跑不通

- 这是方法接口问题
- 不是 checkpoint 可得性问题
- 不解决 hybrid state 兼容，就无法干净回答“论文原始方法在 Qwen3.5 更相似对上是否成立”

### 问题 2: 当前 runnable 近邻对没有复现论文主张

- 即使是更相似的 Qwen-family 近邻对
- 在严格 no-fallback 定义下也没有可用结果

### 问题 3: self-self patch 仍缺少“又对又快”的路线

当前已经有两个质量健康点:

- `direct_kv_knorm_v1`
- `consumer-native rotary`
- `patch_layers = 1..27`
- `xnorm_nativekv_residual_mlp_v1`
- `patch_layers = 0..27`
- `runtime identity skip`

这说明:

- patch 不是完全不可能
- `layer 0` 也不是不可修
- 但现在还没有一条同时满足“全层健康 + TTFT 合理”的路线
- 还不足以支撑“方法已被可靠复现”

### 问题 4: 当前只有一种 nonlinear 参数化在 correctness 上成功，但效率很差

已经可以明确区分三类情况:

- `xnorm_semantic_v1 / PatchMLP`: 失败
- `direct_kv_knorm_residual_mlp_v1`: 失败
- `xnorm_nativekv_residual_mlp_v1`: 质量成功，但 `patch_infer_ms` 过大
- `xnorm_nativekv_residual_mlp_v1 + identity skip`: 更快，但仍然偏慢

这说明下一步不能再用“再换一个更深 MLP 看看”这种小改法继续试。

如果继续做 nonlinear patch，重点应该改:

- native `hidden -> KV` 投影路径
- `k_norm + rotary` 的执行成本
- 以及 patch 只在确实需要的层启用

### 2026-04-01 补充: 为什么默认不是“全层都用 C 风格”

已经做过一个最干净的 probe:

- producer = consumer = `Qwen3-1.7B`
- strict no-fallback
- 把 `xnorm_nativekv_residual_mlp_v1` 从原本只在 `layer 0` 生效，扩成 `0..27` 全层 residual
- 且所有 residual 都保持 identity 初始化，避免训练质量本身干扰判断

结果不是“质量没出来”，而是更基础的工程阻塞:

- 原 checkpoint `weights.pt` 约 `2.15 GB`
- 合成后的 all-layer residual checkpoint `weights.pt` 约 `47.45 GB`
- 严格评测在真正进入第一条样本前，就在 consumer 侧加载 patch checkpoint 时 OOM
- 报错显示 24GB 4090D 上单个进程已经占到约 `23.42 GiB`，随后申请额外 `128 MiB` 失败

这说明当前参数化下，“全层都用 C 风格”不是一个免费替换项，而是会把 patch route 直接推到新的显存量级。

因此当前默认仍应保持:

- `layer 0`: 用 `xnorm_nativekv_residual_mlp_v1`
- deeper layers: 继续用线性 `direct_kv_knorm_v1`

如果后面还想认真做全层 residual，优先级应该是:

- checkpoint 分片 / lazy load
- residual 模块缩小
- 或只在少数层上用 residual，而不是直接全 28 层铺开

## 6. 当前默认实验建议

为了避免后续结果再被无效 baseline 污染，当前建议的默认设置是:

### 对 self-self reuse

- 只用 `layerwise reuse`
- 默认开启 batched encode/decode fast path
- 不再用 `grouped shared-z`

### 对 self-self patch

- 默认效率参考路线: `direct_kv_knorm_v1 + consumer-native rotary + patch_layers=1..27`
- 默认 correctness 参考路线: `xnorm_nativekv_residual_mlp_v1 + patch_layers=0..27`
- 默认 `layer 0` 优化参考路线: `xnorm_nativekv_residual_mlp_v1 + identity skip + prewarm`
- 不把当前 `xnorm_semantic_v1 / PatchMLP` 作为有效救援路线
- 不把当前 `direct_kv_knorm_residual_mlp_v1` 作为有效救援路线

### 对论文主张判断

- 目前不能说“已复现”
- 当前最稳妥表达应为:
  - runnable Qwen 近邻对上未复现
  - strict no-fallback 下 A/B/C 仍失败
  - self-self 上已经证明 patch correctness 可修，而且 `layer0-only` 在 `identity skip + prewarm` 后已接近 oracle
  - 当前还偏慢的是 healthy all-layer 路线，而不是 isolated `layer 0`
  - faithful Qwen3.5 baseline 仍被 hybrid cache 兼容性阻塞

## 7. 建议的下一步

当前最合理的三条路是:

### 路线 A: 先打通 faithful Qwen3.5 baseline

目标:

- 让 repo 正确处理 `Qwen3.5DynamicCache`
- 明确 capture / transfer / rebuild:
  - `full_attention` KV
  - `linear_attention` conv states
  - `linear_attention` recurrent states

意义:

- 这是最干净回答原始研究问题的路径
- 但工程量最大

### 路线 B: 转向标准 KV 的更相似模型对

目标:

- 找到标准 KV cache、相似度高、又能 faithful 跑原始 SCD 的模型对

意义:

- 更快回答“相似性是否足以救回论文主张”
- 工程风险比 Qwen3.5 hybrid cache 小

### 路线 C: 优化新 hidden-residual 路线的 runtime

当前已完成:

- 识别 `layer 0` residual 实际为 identity 的 checkpoint
- 在 runtime 跳过无效 residual MLP
- `all28` TTFT 从 `200.01 ms` 降到 `167.74 ms`
- `layer0-only` 在加入 prewarm 后降到 `TTFT 73.67 ms`, `patch_infer 1.42 ms`
- self-self `layerwise reuse` 在加入 batched encode/decode 后降到 `TTFT 40.67 ms`

意义:

- correctness 问题已经定位并修掉
- `layer 0` steady-state 路线已经基本打通
- reuse 链路的并行化加速已经跑通，且当前不是传输瓶颈
- 下一步最值得做的是继续压 `all28` hidden-residual 路线的 steady-state patch path，或者再测 same-rope 条件下能否安全去掉 `de_rope/re_rope`

## 8. 当前最简短判断

如果只用一句话概括当前阶段:

> 在当前可运行的 Qwen 近邻实验上，论文宣称效果尚未复现；strict no-fallback 下 A/B/C 仍失败，faithful Qwen3.5 baseline 仍被 hybrid cache 阻塞，但 self-self reuse 链路的并行化已经跑通，`layerwise reuse_only` 在不做压缩时已到 `TTFT 40.67 ms`，而 self-self hidden-residual 路线也已经把 correctness 修好；当前真正还需要压的是 all-28-layer patch 的 steady-state 开销，而不是传输或 isolated `layer 0` 本身。

## 9. 相关支撑文件

- `Experiment/analysis/qwen35_hybrid_cache_blocker.md`
- `Experiment/analysis/qwen17_nofallback_routeabc_comparison.md`
- `Experiment/analysis/qwen17_self_allreuse_reuse_route_diagnosis.md`
- `Experiment/analysis/qwen17_self_patch_route_diagnosis.md`
