# scd-new

`scd-new` 语义缓存蒸馏（Semantic Cache Distillation, SCD）验证项目的最终整理工作区。

## 一眼看懂

- 主代码目录：`Experiment/core_code/runtime_repo/`
- 当前认可的主方法族：
  - `baseline`
  - 修复后的 `reuse-only`
  - 严格版 `patch-only`
  - 仅在单独完成严格验证后才计入主线的严格版 `hybrid`
- 当前最强、最稳定的主结论：严格口径下的 Mistral 系列结果
- 仍未闭环的问题：最初的 Qwen3.5 相似模型对主假设

这个仓库现在只保留最新定稿版本所需的主线代码与研究材料，之前显式归档的 `legacy/` 旧内容已经移除。

## 当前状态

已经定稿的部分：

- 严格版运行时语义
- 严格版 TTFT 计时口径
- Mistral 主结果线
- `Experiment/core_code/runtime_repo/` 下的维护中代码快照

还没有真正定稿的部分：

- Qwen3.5 相似模型对是否正向复现原始 headline claim

原因：

- 目标中的 Qwen3.5 close-pair 仍然受到 hybrid cache 架构与当前 SCD 运行时假设不兼容的阻塞

可参考：

- `Experiment/analysis/qwen35_hybrid_cache_blocker.md`
- `.pipeline/docs/research_brief.json`

## Canonical 结果线

项目级的 route lock 文档是：

- `Experiment/analysis/transfer_route_lock_20260405.md`

反向模型对的严格锁定文档是：

- `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`

凡是被这两份 lock 文档判定为 fallback-heavy、recompute-heavy、grouped/shared-z，或者已经明确弃用的旧路线，都应当视为历史材料，而不是当前默认主线。

## 当前接受的主要结果

### MistralLite -> Mistral-7B

`CMRC2018 dev = 3219` 上的 full-dev 严格无 fallback 对比：

| 方法 | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.6911 | 0.4427 | 133.39 |
| `reuse-only` | 0.0109 | 0.0000 | 38.49 |
| `patch-only` | 0.7898 | 0.5769 | 67.37 |

解读：

- `reuse-only` 很快，但质量基本不可用
- `patch-only` 是当前唯一健康的默认跨模型迁移路线
- 在这个方向上，`patch-only` 同时提升了质量和 TTFT

规范参考：

- `Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md`

### MistralLite -> Mistral-7B 严格版 Hybrid

当前接受的 strict-stop 多层混合路线点：

| 方法 | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `patch-only` | 0.6285 | 0.285 | 53.93 |
| `hybrid` | 0.6072 | 0.250 | 46.18 |

解读：

- 这是一个有效的“质量受约束下的延迟换取”结果点
- 它比 `patch-only` 更快
- 但它不是一个比 `patch-only` 全面更优的 Pareto 解

规范参考：

- `Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md`

### Mistral-7B -> MistralLite

当前接受的反向模型对严格参考结果：

| 方法 | F1 | EM | TTFT ms |
| --- | ---: | ---: | ---: |
| `baseline` | 0.7829 | 0.6000 | 122.50 |
| `reuse-only` | 0.0212 | 0.0000 | 33.60 |
| `patch-only` | 0.7528 | 0.5000 | 75.94 |

解读：

- `reuse-only` 依然不可用
- `patch-only` 保住了大部分 baseline 质量，同时明显降低了 TTFT
- 这个反向方向的结果应当只通过锁定后的 strict artifact 来引用

规范参考：

- `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`

## Qwen 这条线

Qwen 线不是这个仓库当前已经完成的正向成功故事。

当前状态：

- `Experiment/analysis/claim_gap_summary.csv` 中目前可见的是一个 smoke 级别的 `fail`
- 可运行的 Qwen3-1.7B near-pair 线路，在严格无 fallback 口径下没有恢复原始 claim
- 真正目标中的 Qwen3.5 close-pair 线路仍然被 cache 架构不匹配卡住

因此，Qwen 相关材料更适合被当成：

- 失败分析
- 协议和评测口径演化过程
- claim-gap 定义与度量框架
- 为什么最终收口到更严格 Mistral 结果线的动机

不要把 Qwen 线表述成一个已经完成的正向复现。

## 代码入口

如果你要从当前维护中的最终代码线开始看，优先看这里：

- `Experiment/core_code/runtime_repo/`
- `Experiment/core_code/runtime_repo/README.md`
- `Experiment/core_code/runtime_repo/src/scd_repro/`
- `Experiment/core_code/runtime_repo/configs/`
- `Experiment/core_code/runtime_repo/scripts/`

辅助脚手架在这里：

- `Experiment/core_code/README.md`
- `Experiment/core_code/scd_validation/`
- `Experiment/core_code/configs/`
- `docs/final-version-map.md`

## 推荐阅读顺序

如果你隔了一段时间后重新回到这个项目，建议按这个顺序看：

1. `docs/final-version-map.md`
2. `Experiment/analysis/transfer_route_lock_20260405.md`
3. `Experiment/analysis/mistrallite_to_mistral7_fulldev_locked_routes_20260405.md`
4. `Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md`
5. `Experiment/analysis/mistral7_to_mistrallite_final_route_lock_20260409.md`
6. `Experiment/core_code/runtime_repo/README.md`

## 说明

- 很多研究笔记里仍然保留了旧工作区的绝对路径；这些路径更适合被理解为历史来源信息，而不是当前有效路径说明。
- 在 2026-04-17 的核对中，A800 上没有发现比本地 `runtime_repo` 更新的 canonical 代码快照，并且 `/nas/mql` 当时不可用，因此这个仓库应当被视为当前主源。
