# SCD 三路线验证清单（可直接交给 Codex / Claude Code / Cursor Agent）

## 0. 先给结论

如果你准备把当前这条线再救一次，而且你会把模型对换成**更相似的两个微调模型**，我建议的优先顺序是：

1. **路线 A：State -> State -> native Wk/Wv -> KV**
   - 也就是：预测 `H_in,B / x_norm,B`，再让 B 自己用 `Wk/Wv` 生成 KV。
   - 这是最稳的主线，最符合“B 自己的几何 / 原生生成机制”。

2. **路线 B：State -> direct KV**
   - 也就是：直接把 A 侧 code 解码成 `K_B, V_B`。
   - 这是最便宜、最可能吃到“模型更相似”红利的一条线；如果 pair 非常接近，它可能反超路线 A。

3. **路线 C：State -> State -> suffix recompute**
   - 也就是：先估一个 B 的中间状态，然后从这里开始在 B 侧重算后缀。
   - 这条线目前只建议做**诊断**，不建议作为主赌注。

### 我的先验排序

- **质量稳健性**：A > B >> C
- **速度 / 工程便宜程度**：B > A >> C
- **最值得先验证的顺序**：A 和 B 并行，C 只做少量诊断

---

## 1. 三条路线的统一定义

为保证对比公平，三条路线都遵守以下统一规则：

- Producer = 模型 A
- Consumer = 模型 B
- 两边使用**完全相同的 tokenizer、attention mask、cache 语义**
- 所有路线都在**同一批 calibration / validation / test prefix** 上比较
- 所有路线都报告：
  - payload bytes
  - producer encode time
  - transfer time
  - consumer reconstruction time
  - recompute time
  - end-to-end TTFT
  - task metric / KL / PPL

### 路线 A：S2S2KV（主推）

- 输入：A 的 `H_in,A^l`（或 `x_norm,A^l`）
- 传输：`z_H^l = Enc_H^l(H_in,A^l)`
- 恢复：`H_hat_in,B^l = Dec_H^l(z_H^l)`
- 生成 KV：
  - `K_hat_B^l = RoPE(H_hat_in,B^l Wk_B^l)`
  - `V_hat_B^l = H_hat_in,B^l Wv_B^l`
- **不做 suffix recompute**

### 路线 B：S2KV（主推）

- 输入：A 的 `H_in,A^l`（建议先用这个，方便和 A 路线公平对比）
- 传输：`z_H^l = Enc_H^l(H_in,A^l)`
- 恢复：
  - `K_hat_B^l = Dec_K^l(z_H^l)`
  - `V_hat_B^l = Dec_V^l(z_H^l)`
- Key 需要保证与 B 侧 cache 语义一致；如果在 content space 预测，则解码后 reRoPE
- **不做 suffix recompute**

> 可选附加变体（只有在时间充足时再做）：
> - `KV_A -> KV_B`，即以 `K_A/V_A` 为输入做 direct KV-to-KV。这个更接近你论文里的 Reuse，但不属于本轮三路线的必选项。

### 路线 C：S2S + SR（仅诊断）

- 输入：A 的 `H_in,A^l`
- 传输：`z_H^l = Enc_H^l(H_in,A^l)`
- 恢复：`H_hat_in,B^l = Dec_H^l(z_H^l)`
- 从第 `l` 层开始，在 B 上对 prefix 做 suffix recompute，生成后续层 native KV
- 初版只做：
  - **single-anchor**
  - 不做 dual-anchor
  - 不做复杂 fallback

---

## 2. 要先做“模型相似度预筛选”，不要上来就全量训练

目标：回答“换更相似的两个微调模型，是否真的能明显改善这三条路线”。

### 2.1 候选 pair 选择规则

优先挑：

- 同一 base model 派生出来的两个 FT 模型
- 相同 architecture / layer count / hidden size / head config
- 相同 tokenizer
- 尽量相近的微调任务或数据域
- 最好先选 3 档：
  - **Close**：很相似
  - **Medium**：中等相似
  - **Far**：明显差异大（你当前 pair 可作为 far baseline）

### 2.2 预筛选只用小数据集

- calibration prompts: 256 ~ 512 条
- context lengths: 512 和 2048
- 不做任何 translator 训练前，先取 paired traces

### 2.3 每个 pair 必算的相似度指标

对每层 `l`，都计算：

1. `cos_H[l]`：A/B 的 `H_in` 平均 token cosine
2. `cos_K[l]`, `cos_V[l]`：deRoPE 后 K / V 的 cosine
3. `r2_H_lin[l]`：线性 probe `H_A -> H_B` 的 R^2
4. `r2_KV_lin[l]`：线性 probe `H_A -> K_B/V_B` 的 R^2
5. `r2_KV_fromKV[l]`：线性 probe `K_A/V_A -> K_B/V_B` 的 R^2（可选但强烈建议）
6. `kl_token`：A 与 B 在 next-token distribution 上的平均 KL
7. `oracle_gap_task`：A 与 B 在目标任务上的原始性能差

### 2.4 Pair 排名方式

定义一个简单 composite score：

```text
sim_score = 0.30 * median(r2_H_lin)
          + 0.30 * median(r2_KV_lin)
          + 0.20 * median(cos_H)
          + 0.20 * (1 - normalized_kl_token)
```

然后：

- 选 top-1 作为 **close pair**
- 选中位 pair 作为 **medium pair**
- 选当前差异大的 pair 作为 **far pair**

---

## 3. Patch 层 / 关键层不要拍脑袋选

### 3.1 候选层选择

先对每层做 cheap oracle / probe：

- 对路线 A：用 oracle `H_in,B^l` 直接替换，测对最终 KL / PPL / task 的改善
- 对路线 B：用 oracle `K_B^l, V_B^l` 直接替换，测改善
- 对路线 C：用 oracle `H_in,B^l` 作为断点做 single-anchor recompute，测改善

### 3.2 选层策略

每条路线各自选：

- top-2 层
- top-4 层
- top-8 层

并保存：

- `gain_per_layer.csv`
- `gain_per_byte.csv`
- `gain_per_ms.csv`

> 核心原则：
> **选层依据必须是“单位预算收益”，不是只看绝对提升。**

---

## 4. 训练 / 建模规范（Codex 必须严格对齐）

### 4.1 统一数据切分

- calibration train: 70%
- calibration val: 15%
- calibration test: 15%
- 与最终 benchmark test 严格分离

### 4.2 统一 rank / 容量网格

#### 路线 A / B

- rank `r` in `{16, 32, 64, 96, 128}`
- decoder hidden size（若为 MLP）in `{0, 1x, 2x}` rank
- regularization `{1e-5, 1e-4, 1e-3}`

#### 路线 C

- anchor layer `l` from candidate set
- suffix depth：默认 recompute 到末层
- 只做 single-anchor；dual-anchor 仅当 single-anchor 非常强时再开

### 4.3 输入输出规范

#### 路线 A
- input = `H_in,A^l`
- target = `H_in,B^l`
- post-process = native `Wk/Wv_B`

#### 路线 B
- input = `H_in,A^l`
- target = `K_B^l, V_B^l`
- 若 Key 在 content space 训练，则 decode 后统一 reRoPE

#### 路线 C
- input = `H_in,A^l`
- target = `H_in,B^l`
- runtime = 断点后 recompute suffix

### 4.4 Loss 设计

#### 路线 A
```text
L_A = mse(H_hat_in,B, H_in,B)
    + optional cosine loss
```

#### 路线 B
```text
L_B = mse(K_hat_B, K_B) + mse(V_hat_B, V_B)
    + alpha * cosine(K_hat_B, K_B)
    + beta  * cosine(V_hat_B, V_B)
```

#### 路线 C
```text
L_C = mse(H_hat_in,B, H_in,B)
```
但路线 C 的真正评价不看训练 loss，而看：
- suffix 后最终 KL
- suffix 后任务质量
- recompute 额外 TTFT

---

## 5. 必做的 4 组实验

## Exp-0: Pair Screening（最先做）

目标：确认“更相似 pair 是否更容易翻译”。

输出：
- `pair_similarity_summary.csv`
- `layerwise_similarity_close_medium_far.csv`
- 一张图：不同 pair 的 `median r2_H`, `median r2_KV`, `token KL`

### 通过标准

只要看到：
- close pair 的 layerwise R^2 / cosine 明显高于 far pair
- close pair 的 raw next-token KL 明显更低

就说明“换更相似模型”这步是值得做的。

---

## Exp-1: Route Canary（小规模试探）

对每个 pair，只做：
- top-2 candidate layers
- rank in `{32, 64}`
- context = 512, 2048

比较三路线：
- 路线 A
- 路线 B
- 路线 C

输出：
- `route_canary_results.csv`
- 每条路线在 close / medium / far pair 上的质量-速度散点图

### 这一轮想回答的问题

1. close pair 上，A/B 是否都比 far pair 明显更好？
2. 路线 C 在 close pair 上是否“起死回生”？
3. 路线 B 是否在 close pair 上已经接近路线 A？

### Go / No-Go 规则

- 如果 close pair 上，路线 C 仍明显落后 A/B，**路线 C 直接降级为负例分析**。
- 如果 close pair 上，路线 B 与 A 的质量差 < 2~3% Oracle，但 TTFT 更低，**路线 B 升为重点线**。
- 如果 close pair 上，路线 A 明显稳于 B，且 TTFT 只多一点，**路线 A 升为主线**。

---

## Exp-2: Full Budgeted Comparison（主实验）

只对最好的 1~2 个 pair 做。

### 维度

- route ∈ {A, B, C}
- patch budget k ∈ {0, 2, 4, 8}
- rank ∈ {16, 32, 64, 96, 128}
- context ∈ {512, 2048, 8192}
- bandwidth ∈ {50, 100, 200, 500} Gbps 或你真实系统的带宽档位

### 统一输出字段

每条记录至少包含：

```text
pair_name
pair_similarity_bucket
route
layer_budget_k
selected_layers
rank
context_len
bandwidth_gbps
payload_mb
enc_ms
transfer_ms
recon_ms
recompute_ms
decode1_ms
ttft_ms
oracle_task_score
route_task_score
task_gap_pct
oracle_kl
route_kl
oracle_ppl
route_ppl
```

### 必出图

1. TTFT vs quality
2. payload vs quality
3. bandwidth sweep
4. context length sweep
5. layerwise reconstruction R^2 / error curves
6. hard-layer heatmap

---

## Exp-3: Failure Attribution（一定要做）

不论哪条路线赢，都要回答：

- 是不是 close pair 真的让 patch / reuse 变容易了？
- 到底是“层变容易了”，还是“只是 rank 放大后有效”？
- 难层究竟难在：
  - state alignment
  - KV alignment
  - suffix continuation

### 具体拆解

对 best pair / best route：

1. 报 layerwise `R^2_H`, `R^2_K`, `R^2_V`
2. 报 oracle 单层替换收益
3. 报 hardest layers 的 attention head 级别误差
4. 报 close vs far pair 的 hardest-layer 重合度

输出：
- `failure_attribution_report.md`

---

## 6. 如何判断三条路线谁更值得继续

### 选择路线 A 为主线，当且仅当：

- close pair 上，它的质量稳定最好；
- 相比路线 B，质量明显更高；
- 相比路线 C，TTFT 明显更低；
- 在 2~8 个 patch 层时就能接近 Oracle。

### 选择路线 B 为主线，当且仅当：

- close pair 上，路线 B 的质量与 A 的差距很小；
- 但它更快、更便宜、更简单；
- 随相似度提升，B 的收益斜率明显大于 A。

### 明确放弃路线 C，当且仅当：

- 即便在 close pair 上，single-anchor 也无法显著追平 A/B；
- 或者它虽然质量高一点，但 recompute TTFT 太贵；
- 或者它只能靠非常早的 anchor 才有效，导致节省不了多少 prefill。

### 明确放弃整个 SCD rescue，当且仅当：

- close pair 上，A/B/C 都无法在质量和 TTFT 上同时明显优于 full consumer prefill；
- 或者只有极高 rank / 极多 patch layers 才能有效，导致故事不再成立。

---

## 7. 给 Codex 的直接任务书（可原样复制）

你现在要实现一个实验框架，比较三条异构 cache transfer 路线在“模型相似度不同”的条件下的表现。目标不是做最终论文代码，而是快速验证哪条路线在更相似的模型对上最值得继续。

### 任务目标

实现并比较以下三条路线：

1. Route A = `H_A -> z_H -> H_B_hat -> native Wk/Wv_B -> K_B_hat, V_B_hat`
2. Route B = `H_A -> z_H -> direct K_B_hat, V_B_hat`
3. Route C = `H_A -> z_H -> H_B_hat -> suffix recompute on B`

### 强制要求

- 使用相同 tokenizer / identical prefix / identical attention mask
- 使用相同的数据切分
- 所有路线输出统一的 metrics csv
- 所有实验必须支持不同 model pair
- 先做 pair screening，再做 route canary，再做 full comparison

### 需要你生成的代码模块

1. `collect_paired_traces.py`
   - 输入：model A, model B, prompt dataset
   - 输出：每层 `H_in`, `K`, `V` traces

2. `pair_screen.py`
   - 计算：cosine, linear R2, token KL
   - 输出：pair similarity ranking

3. `train_route_a.py`
   - 训练 `H_A -> H_B`
   - runtime 端通过 B 的 `Wk/Wv` 生成 KV

4. `train_route_b.py`
   - 训练 `H_A -> K_B/V_B`

5. `eval_route_c.py`
   - 用 `H_A -> H_B_hat` 做 anchor
   - 在 B 端做 suffix recompute

6. `select_candidate_layers.py`
   - 基于 oracle gain / gain-per-byte / gain-per-ms 选层

7. `run_canary_grid.py`
   - 小规模网格实验

8. `run_full_grid.py`
   - 大规模预算化实验

9. `analyze_results.py`
   - 生成 CSV 汇总、Pareto 图、TTFT breakdown、context sweep 图

### 所有实验必须输出的文件

- `pair_similarity_summary.csv`
- `route_canary_results.csv`
- `full_grid_results.csv`
- `ttft_breakdown.csv`
- `quality_vs_payload.csv`
- `quality_vs_ttft.csv`
- `layerwise_probe_metrics.csv`
- `failure_attribution_report.md`

### 默认实验顺序

#### Step 1: Pair screening
- 3 个 pair：close / medium / far
- 256~512 条 prefix
- context 512 / 2048

#### Step 2: Route canary
- 每条路线
- top-2 layers
- rank 32 / 64
- 先在 close / far 上跑

#### Step 3: Main grid
- 只保留最有希望的 1~2 个 pair
- rank / patch budget / context / bandwidth 全 sweep

### 默认结论规则

- 如果 Route C 在 close pair 上仍明显差，直接停止扩展 C
- 如果 Route B 在 close pair 上质量接近 Route A 且更快，优先 B
- 如果 Route A 明显更稳，优先 A
- 如果三条路线都不成立，输出“stop SCD rescue”结论

---

## 8. 最后一句提醒

这次验证的核心不是“把三条路线都调到最好”，而是：

> **判断“更相似的模型对”是否真的改变了问题难度分布，从而让 SCD 的某一条路线进入可行区。**

如果答案是“只对 A/B 有用，对 C 仍然没用”，那你就有非常清楚的决策依据了。
