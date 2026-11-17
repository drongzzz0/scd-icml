# 实验记录

## 2026-11-17 · `icml-cmrc-apc-rerun-20261117-01`

- 本轮检验修正开头换行停止规则和批量 TTFT 计时后的差异，并补测原生 vLLM 开启 APC 的对照。
- A800 复用 11 月 17 日已验证的 MistralLite / Mistral-7B-v0.1 和 Patch20/Reuse12 bank。质量使用 `dev26` 和固定 CMRC dev 3,219 题，贪心生成至多 32 token。评分固定官方 CMRC2018 `c0eb1b6ba219847457e6af3180da722bbeb656af` 的 Python 3 评分器。质量运行保存 `attempts.jsonl`、两个预测 JSON 和分片摘要。
- 性能使用发布代码 `259b0ad` 加本地计时补丁、同版 vLLM 和 CUDA Graphs。长度 2,048、批大小 1/4/8/16、输出 32，原生无 APC、原生冷/热 APC、SCD 冷/热均记录外层 TTFT 和 wall time。保留实际输出数与命中证据，输出 `benchmark_results.json`。

### 性能复跑结果（13:08，A800 GPU 7）

- `--cudagraph-and-batch --batch-reps 3` 成功退出。数据位于 A800 `icml-cmrc-apc-rerun-20261117-01/results/performance/benchmark_results.json`。vLLM 为 `0.1.dev1+g2b1f75c80.d20261025`，Torch 为 `2.14.0+cu126`。单请求和批次均为此次重新执行所得。
- 批大小 16 时，五组 TTFT / wall 中位数分别为无 APC 原生 `2116.99 / 2669.33 ms`、冷 APC 原生 `280.57 / 834.36 ms`、热 APC 原生 `154.20 / 712.35 ms`、冷 SCD `387.30 / 945.57 ms`、热 SCD `326.07 / 884.36 ms`。温度为 0，已复核 60 次试次的实际输出长度、各请求 TTFT 边界和缓存命中。原生冷批首请求未命中，其余 15 个请求均命中完整共享前缀，热批均命中。
- 热 SCD 相对无 APC 原生为 `6.49x` TTFT、`3.02x` 完成时间加速。匹配冷热状态时，启用 APC 的原生对照在四种批大小的 TTFT 上均更低。3,584 token 的单请求原生 / H / E TTFT 分别为 `253.79 / 49.45 / 347.40 ms`。

### 质量复跑结果（13:22，A800 GPU 2 和 5）

- 两组各 3,219 道题。分片用 `--num-shards 2` 从固定清单取 1,610 / 1,609 道题，运行分别用时 983 / 1,031 秒，退出码均为 0。合并核对每个官方 QID 恰好出现一次、每个分片两组题号相同、预测文件与逐题记录一致。A800 原始分片位于 `icml-cmrc-apc-rerun-20261117-01/results/quality-0` 和 `quality-1`。
- 作者发布的 Python 3 v6 评分器、NLTK 3.9.2。用于生成的原始 dev 数据与用于评分的官方 SQuAD 格式数据在 3,219 题的题号、题面和答案集合上一致。修复前后的四组评分 `SKIP=0`，结果记录在 `official_scores.json`。

| 状态 | 原生 F1 / EM（%） | SCD F1 / EM（%） | SCD 差值（百分点） |
|---|---:|---:|---:|
| 11 月 4 日原始预测，用官方脚本复算 | 65.1122 / 47.8409 | 79.4799 / 52.9978 | +14.3677 / +5.1569 |
| 修复开头换行停止后重新推理 | 72.6450 / 47.8409 | 79.4973 / 52.9978 | +6.8523 / +5.1569 |

- 原生原有 704 个首 token 换行的请求本轮全部继续生成，642 个得到非空答案，62 个仍为空，其中 43 个直到上限仍只生成换行 token。642 个新非空答案的官方 EM 仍为 0。剩余 F1 差异集中在这 704 道题，其余 2,515 题上原生 / SCD F1 为 `83.3384 / 82.8969`。逐组官方重算见 `quality_groups.json`。

## 2026-11-17 · `icml-cmrc-apc-latency-20261117-01`

- 同题对照采用 `benchmark_a800_cmrc_quality.py --native-apc`，原生按篇章顺序保留 APC，SCD E 计入源 prefill。两组复用已经修正的停止规则，并记录每题预测、TTFT 和停止决定时间。启动脚本和评测入口已部署到 A800 该目录。
- 19 项相关 CPU 测试通过。13:53 至 13:56 的三次资源检查均显示八张卡有占用。全量 GPU 实验等待空闲卡。当前可引用的 APC 延迟来自上一条合成共享前缀批测，CMRC 全量的原生质量来自 APC 关闭的那轮运行。

### 14:07 启动四段计时全量对照

- 用户确认需要同题、全量、对应时间段的结果。补上目标 `LLMEngine.add_request` 的共同入口标记，四段为入口到目标提交、目标提交到首 token、后续生成、停止决定到返回。各阶段从同一个客户端单调时钟取样。保存 SCD 内部耗时作为辅助字段。
- A800 GPU 4 的三题小测成功退出，原生同篇章第二题命中 464 个缓存 token。两组六条记录的四段时间均加总得到请求耗时。随后在 A800 GPU 0 和 4 启动 `full-shard-0`、`full-shard-1`，每个分片先热身三题，再测两组。原生按篇章持续保留 APC，SCD 每题进行包含源 prefill 的 E 模式调用。

### 14:29 全量对照完成

两个分片退出码均为 0，分别包含 1,608 和 1,611 道题。两组覆盖同一批 3,219 个官方 QID，各题 prompt token 长度一致。固定官方 Python 3 v6 评分器下，两组 `SKIP=0`。原生 APC 的 F1 / EM 为 `72.2702 / 47.6856`，SCD E 为 `79.4973 / 52.9978`，差异为 `+7.2271 / +5.3122` 个百分点。

| 每题平均阶段耗时（ms） | 原生 APC | SCD E |
|---|---:|---:|
| 入口到目标引擎提交 | 0.244 | 105.150 |
| 目标引擎提交到首 token | 32.006 | 29.429 |
| 首 token 到停止决定 | 234.278 | 235.162 |
| 停止决定到接口返回 | 13.970 | 6.000 |
| TTFT | 32.250 | 134.579 |
| 入口到返回 | 280.499 | 375.742 |

SCD E 的准备 RPC 平均为 98.122 ms，内部的源 prefill、转换和发布分别为 79.535、7.620、2.867 ms。准入 RPC 平均 2.607 ms。原生 APC 复用输入 token 的 71.42%。两组平均观察 token 数为 14.96 和 14.38。逐题请求时间累计为 902.93 s 和 1,209.51 s。均值按单题分段加总复算，P50 和 P95 及两组逐题原始记录保存在运行产物中。两张卡上的两个独立分片都呈现原生 APC 平均 TTFT 和累计请求耗时较低。

`summarize_a800_cmrc_apc.py` 使用 `full-shard-0` 和 `full-shard-1` 目录的逐题记录与预测文件生成 `official_scores.json`、`latency_summary.json`。25 项相关 CPU 测试通过。

## 2026-11-17 · `cmrc-mistrallite-raw-apc-20261117`

- MistralLite 在 vLLM 源端开启 APC，两组共享每题读出的原始 BF16 KV。比较原样直传和 SCD Patch20/Reuse12，目标均为 Mistral-7B-v0.1，使用同卡独立进程 CUDA IPC、D2D 拷贝及相同的目标页写入路径。数据为校验后的 CMRC2018 dev 3,219 题，沿用 dev26 prompt、停止规则和贪心 32 token 上限。逐题保存阶段耗时、缓存命中、导入回执与预测，合并后用官方 v6 脚本评分。
- 18 项 CPU 回归通过。GPU 2 的 `smoke-01` 未进入推理，ZMQ 的 Unix socket 路径超出 107 字节。启动器改用远端独立短 IPC 路径，日志与正式产物继续写在本轮实验目录。
- GPU 2 的 `smoke-02` 完成 4 对请求，退出码为 0。四题分别命中 0、464、16、16 个源 APC token。八条记录均通过 IPC 收发端的 BF16 SHA-256 校验，32 层写入的前缀长度与输入逐题一致。修正目标进程关闭时对未加载的 HF 源 provider 的访问。扩大到相关 resident/fullprefix 路径的 CPU 检查共 47 项通过。
- 启动前逐卡核对利用率、显存与活动进程，选择空闲的 A800 GPU 2 和 5。以同一套代码启动 `full-shard-0`、`full-shard-1`，分别 1,608、1,611 题。每个分片热身三题后清空源 APC，保持随后正式请求的源 APC 连续。每题交替先跑直传或 SCD，取消仅用于小测的 BF16 哈希，保持加载回执校验。vLLM 为 `0.1.dev1+g2b1f75c80.d20261025.precompiled`，目标和源为 BF16、FLASH_ATTN、串行 eager。

### 15:27 全量完成

两个分片退出码均为 0，分别用时 937.96 和 950.44 秒。两组各 3,219 个唯一官方 QID，官方 Python 3 v6 评分 `SKIP=0`。6,438 条记录均从目标页缓存加载完整的 32 层前缀，覆盖除末 token 外的每个输入 token。源 APC 命中 3,217 题，复用 71.42% 的输入 token。两组传输数据逐题同字节数，平均 82,834,939 字节。直传保留源 post-RoPE K，模型配置的源、目标 RoPE θ 分别为 1,000,000 和 10,000。

| 指标 | 原始 KV 直传 | SCD Patch20/Reuse12 |
|---|---:|---:|
| F1（%） | 7.5923 | 79.4532 |
| EM（%） | 0.0932 | 53.1221 |
| 源 prefill（ms/题） | 60.987 | 60.987 |
| 转换 | 0.000 | 9.814 |
| 进程间传输 | 2.522 | 2.540 |
| 目标登记与准入 | 7.617 | 7.678 |
| 写入目标 KV | 1.292 | 1.302 |
| 写入后到停止判定 | 246.713 | 202.451 |
| 收尾 | 2.285 | 2.248 |
| TTFT | 95.565 | 110.886 |
| 请求入口到返回 | 321.417 | 287.020 |

SCD 的 F1 和 EM 分别增加 71.8608、53.0289 个百分点。两组空预测为 1,257 和 2，准确匹配 3 和 1,710 题。SCD 平均 TTFT 增加 15.32 ms，平均端到端减少 34.40 ms（10.70%），两组平均生成 17.56 和 14.36 个 token。端到端 P95 为 537.29 和 538.56 ms。传输时间含两侧 IPC 调用，CUDA Event 单计的 D2D 拷贝平均 0.109 和 0.112 ms。

使用 `scripts/summarize_a800_cmrc_raw_apc.py` 合并两个分片，逐题复核阶段加总并调用官方评分器，生成 `report/metrics.json`、`official_scores.json` 和两组逐题预测。

## 2026-11-17 · `cmrc-mistral7-reverse-apc-20261117`

- 反向为 Mistral-7B-v0.1 prefill → MistralLite decode。源端 APC 开启，目标端 APC 关闭并写入完整的 32 层前缀 KV。两个 tokenizer 重编码冻结 CMRC dev 的 3,219 题，逐题 token 完全一致。目标停止集合为 MistralLite 的 special token，贪心最多 32 token。源的 4,096 token 滑窗覆盖 1,312 token 的运行长度上界。
- A800 GPU 5 `raw-smoke-01` 退出码为 0。四题分别命中 0、464、16、16 个源 APC token，均实写 32 层，加载前缀长度为 492、497、742、1,273 token，发送和接收的 BF16 字节一致。45 项相关 CPU 检查通过，正向已有 3,219 题两组结果用同一评分接口复算一致。
- 找到反向标定归档 `rev_cmrc_all32_train200_fix_20260425`，从 CMRC train 捕获 200 条 prompt，seed 42，max length 1,024，BF16。旧 loader 所用的训练目录 `dev.json` 与原 train 文件的 SHA-256 一致，官方评测 dev 单独保留。归档 Reuse 为每层 rank128、K de-RoPE/re-RoPE，Patch 为全 32 层 shared residual、hidden 2048、bottleneck 512、2 个 block、1 epoch。反向测试取与正向相同的 20/12 层分配，两类参数均用反向归档权重，不使用反向 dev 数据选择路由。
- 使用 `rev_cmrc_all32_train200_fix_20260425` 归档。训练 prompt 与 CMRC train 前 200 条逐条相同，与 dev 的 QID、篇章、问题和 prompt 精确交集均为 0。紧凑 Patch bank 保留全部 12 个运行时 shared-residual 张量（508,309,951 字节），逐张量校验 dtype、shape 和原始字节。导入时复核输入 SHA 与 MistralLite 目标投影，生成 847,004,632 字节的 safetensors，退出码 0。
- GPU 1 和 5 分别执行 `raw-shard-0` 和 `raw-shard-1`。两组分开实测的原因是权重传输与原样直传评测并行。单组模式经 46 项 CPU 检查。16:15 在空闲 GPU 6 启动 `scd-smoke-01`，通过后在直传所用 GPU 及分片顺序上运行 SCD 全集。

### 16:29 全量完成

直传和 SCD 的两个分片均以退出码 0 完成，GPU 1 各 1,608 题，GPU 5 各 1,611 题。官方 Python 3 v6 评分均为 `SKIP=0`。逐题验证两组在同一 GPU 上处理同一题，APC 命中数、目标加载 token 数、32 层回执与传输字节数一致。两组各命中 3,217 题，复用 71.42% 的输入 token。SCD 四题小测的转换与 IPC 收发校验通过。正向 bank 选中的 20 个 Patch 残差头均为零，反向 bank 的对应 20 个残差头均非零，实际执行 shared residual MLP。

| 指标 | 反向原样直传 | 反向 SCD Patch20/Reuse12 |
|---|---:|---:|
| F1（%） | 8.9015 | 71.7516 |
| EM（%） | 0.6834 | 48.7729 |
| 源 prefill（ms/题） | 41.087 | 31.336 |
| KV 转换（ms/题） | 0.000 | 94.410 |
| 传输（ms/题） | 2.951 | 2.864 |
| 目标登记与准入（ms/题） | 7.313 | 7.334 |
| 写入目标 KV（ms/题） | 1.432 | 1.286 |
| Decode 至停止判定（ms/题） | 251.324 | 182.642 |
| 收尾（ms/题） | 3.655 | 9.120 |
| TTFT（ms/题） | 84.420 | 170.173 |
| 请求入口到返回（ms/题） | 307.763 | 328.991 |

SCD 增加 F1 62.8501 和 EM 48.0895 个百分点。两组平均生成 15.18 和 11.41 个 token，端到端均值相差 +21.23 ms。194 道两组实际生成 token 数相同的题，端到端为 433.40 和 510.82 ms。逐题扣去各自实测的源 prefill 后，TTFT 为 43.33 和 138.84 ms，端到端为 266.68 和 297.66 ms。

使用 `scripts/summarize_a800_cmrc_raw_apc.py` 做官方评分和阶段核验，再由 `scripts/compare_reverse_cohorts.py` 按相同题号合并两组实测并生成 `report/metrics.json`。

## 2026-11-17 · `cmrc-qwen17-gray-apc-20261117`

- 模型对为 Qwen/Qwen3-1.7B → soob3123/GrayLine-Qwen3-1.7B-BETA，28 层、8 KV heads、head dim128、hidden 2048，两端全注意力，RoPE θ=1,000,000。A800 的原 Qwen 基座缓存已不存在，从公开 ModelScope 恢复两片权重，在私有 A800 临时区逐片校验 SHA-256 与 ser 原归档一致。Gray 及其训练权重原已在 A800，两份模型权重校验和不同。
- 复用 `extra_model_qwen17_gray_full_strict_20260617_223703` 的完整共享残差 Patch、layerwise rank128 post-RoPE Reuse 和固定 Patch6 路由。选中 Patch 层为 0、22、24、25、26、27，余下 22 层 Reuse。训练 prompt 逐条由原 CMRC train 200 个唯一 QID 重建，和 dev 的题号、正文、问题、完整 prompt 精确交集为 0。选层来自 train200 采集数据。
- 源捕获按实际模型配置读取 28 层。运行时按权重实际的 post-RoPE 空间执行 Reuse，要求两端 RoPE 相同，Gray Patch 使用目标 K RMSNorm。相关 CPU 验证 202 项通过，包括此前的 Mistral 路径和新 post-RoPE 数值回归。A800 离线资产导入退出码 0，原冻结题面的 3,219 个 QID 用两端 tokenizer 重编码后逐题一致，最长 888 token。按篇分片、同题共享开启 APC 的源 prefill，目标完整写入缓存；仍用官方 dev 评分和贪心最多 32 token。
- GPU 2 `smoke-01` 通过，4 对请求均完整写入 28 层和 (N-1) token，IPC 两端 BF16 逐字节校验一致。同篇第二题实际命中 304 个 APC token。选定的 6 层均有非零残差头和 Gray 的 K RMSNorm 权重。

### 17:16 全量完成

A800 GPU 2 和 5 上两个独立分片分别运行 1,608 和 1,611 个同题对，退出码均为 0。vLLM 为 `0.1.dev1+g2b1f75c80.d20261025.precompiled`，源与目标 BF16、FLASH_ATTN、串行 eager、同卡 CUDA IPC。6,438 条记录均验证实际 28 层页写入和精确分段加总。两组均用官方 CMRC v6 脚本评分，`SKIP=0`。源端 APC 命中 2,373 题，复用 70.45% 的输入 token。

| 指标 | Qwen3-1.7B 原始 KV 直传 | SCD Patch6/Reuse22 |
|---|---:|---:|
| F1（%） | 70.6115 | 74.7728 |
| EM（%） | 42.1870 | 50.2641 |
| 源 prefill（ms/题） | 38.367 | 38.367 |
| 转换（ms/题） | 0.000 | 24.544 |
| 同卡进程间传输（ms/题） | 2.371 | 2.486 |
| 目标登记与准入（ms/题） | 6.181 | 6.375 |
| 写入目标 KV（ms/题） | 1.119 | 1.133 |
| Decode 至停止（ms/题） | 204.607 | 184.624 |
| 收尾（ms/题） | 15.606 | 15.877 |
| TTFT（ms/题） | 79.433 | 104.644 |
| 端到端（ms/题） | 268.251 | 273.406 |

SCD 的 F1/EM 增加 4.1613/8.0770 个百分点，TTFT 增加 25.21 ms、自然停止端到端增加 5.15 ms。平均生成长度为 13.18 与 11.84 token。在 2,343 道两组实际生成 token 数相同的题上，端到端均值为 242.07 与 267.35 ms，首 token 到停止为 147.03 与 147.39 ms。分别从源状态就绪计时的 TTFT 为 41.07 与 66.28 ms，端到端为 229.88 与 235.04 ms。

模型和归档权重可通过 `scripts/prepare_qwen17_pair.py` 绑定。各分片保存逐题预测、注入回执和时间，合并分段统计见 `report/metrics.json`。

## 固定评测输入

`--cases-root` 指向已准备好的 dev26 输入目录。`prepared.json` 包含原始数据摘要与每题的 QID、相对路径和 SHA-256。Qwen 实验用 `--retokenize` 对原题面重编码并核对两端 token。 [原始 CMRC dev 文件](https://raw.githubusercontent.com/ymcui/CMRC2018/c0eb1b6ba219847457e6af3180da722bbeb656af/data/cmrc2018_dev.json)

`--legacy-cmrc-parser` 使用 SHA-256 为 `450a9a05618a0288de91397976c070f158d9475a89d84996e066a155d541e49c` 的兼容解析器。公布的 F1 和 EM 由独立的官方脚本和 SQuAD 格式 dev 文件计算。 [固定兼容解析器](https://raw.githubusercontent.com/drongzzz0/scd-icml/4aa0c6888e802eebecee010e3e4bc2f1c7d80b89/Experiment/core_code/runtime_repo/src/scd_repro/cmrc.py) · [官方评分脚本](https://github.com/ymcui/cmrc2018/blob/c0eb1b6ba219847457e6af3180da722bbeb656af/baseline/cmrc2018_evaluate.py)
