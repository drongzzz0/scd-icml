# Project Plan Log

## 2026-05-12 - ICML 2026 camera-ready readiness review

- Scope: compare current SCD-ICML paper files against official ICML 2026 camera-ready requirements.
- Goal: identify blocking issues before camera-ready submission, prioritizing structure, build state, style options, required statements, metadata, assets, bibliography, and PDF freshness.
- Assumptions: `example_paper.tex` is the active source and `ICML2026_MaQianli.pdf` is the current compiled PDF unless the file inspection proves otherwise.
- Constraints: do not edit the paper content during this review pass unless separately requested; verify current file state rather than relying on old notes.
- Approach: check official ICML 2026 author instructions, inventory project files, compare source and PDF timestamps, inspect LaTeX for required camera-ready switches and placeholders, then run the most relevant static/build checks available.

## 2026-05-14 - SCD-ICML news draft

- Scope: draft a Chinese news/post article for the SCD-ICML work using the saved BNU AIFN ICML 2026 news page as the reference style.
- Goal: produce a reusable official-news-style Markdown draft that follows the reference structure: title, lead paragraph, conference introduction, research summary, figure captions, paper information, and funding/team note.
- Assumptions: `example_paper.tex` is the technical source for the SCD title, abstract, contribution claims, and experimental numbers; the saved HTML is only a style and structure reference.
- Constraints: do not assert final acceptance, author order, funding, or team ownership where the local files only contain placeholders; mark those items for confirmation before publication.
- Approach: extract the reference article skeleton, convert SCD technical details into non-specialist but formal Chinese prose, keep measurable results from the paper source, and isolate all missing facts in a clear checklist.

## 2026-05-20 - 线程 019d995e 迁移到 SCD-ICML

- 范围：把 Codex 线程 `019d995e-e381-7c11-b21a-01a8836a7c82` 中可继续使用的上下文，从原工作区 `C:\Users\mql16\Documents\Playground` 迁移到当前工作区 `C:\Users\mql16\Desktop\SCD-ICML`。
- 目标：让当前会话可以作为旧 SCD/ICML 工作的继续点，不依赖 Codex Desktop 切换 UI 线程或工作区。
- 假设：复制后的 JSONL 是长期证据归档；由于原始日志很大且不会自动合并进当前 UI 线程，需要一份日常可读的压缩摘要。
- 约束：`C:\Users\mql16\.codex` 按敏感路径只读处理；不改写原始 session 文件；旧项目记录只追加，不覆盖。
- 设计选择：原始复制日志放在 `docs/thread-migration/`，搜索索引放在 `tmp/thread-migration/`，人工可读的接续文档放在 `docs/thread-migration/`。
- 实施方式：先用 SHA256 验证复制日志，再总结旧线程的仓库、实验、数据集、camera-ready 和图表状态，并把迁移记录写入项目计划和实验记录。

## 2026-05-20 - 最终项目版本整理审计

- 范围：盘点当前 `SCD-ICML` 工作区、GitHub 仓库 `drongzzz0/scd-new`、以及用户指定的最终 camera-ready TeX 文件，确定最终项目版本边界。
- 目标：区分最终投稿包、最终代码主线、历史草稿、临时文件和可归档材料，为后续整理当前工作区提供可执行依据。
- 假设：用户指定的 `ICML-ready\latex_camera_ready_unzipped\example_paper.tex` 是最终 camera-ready 源；GitHub 仓库 `scd-new` 是后续整理代码主线，但可能还未完全整理完。
- 约束：本轮只读盘点和记录，不移动、不删除旧文件；本地 `scd-new` 仓库存在未提交修改，不能在未确认前覆盖或丢弃。
- 设计选择：将最终版本裁决写入 `docs/final-project-version-audit.md`，后续若整理文件，应先创建 `final/` 与 `archive/` 分层并只移动不删除。
- 实施方式：检查当前工作区文件层级、最终 TeX/PDF 编译状态、GitHub 仓库 HEAD 和 README/final-version-map、以及本地 `scd-new` 与 GitHub 的差异。

## 2026-05-20 - 执行最终项目分层整理

- 范围：按已确认的最终版本审计结果，对当前工作区进行实际整理。
- 目标：保留用户指定的最终投稿源路径，同时创建干净 `final/` 副本；把旧文件移入 `archive/`；把代码口径修正纳入本地 `scd-new`。
- 假设：移动旧文件必须可逆；不删除文件；`ICML-ready\latex_camera_ready_unzipped` 继续保留为最终投稿源位置。
- 约束：GitHub 推送依赖外部网络；如果本机无法连接 GitHub，保留本地 commit 并记录远端未同步状态。
- 设计选择：`final/paper/` 保存最终 PDF 与 source zip；`archive/` 按旧草稿、旧脚本、旧图表、旧 xKV、新闻材料、旧 camera-ready 候选包分层。
- 实施方式：先备份最终 TeX/PDF，再机械删除 inactive `\iffalse` 旧稿块，编译验证，生成 source zip，移动旧文件，提交本地代码 patch，并记录 push 状态。

## 2026-05-20 - 撤回 scd-new GitHub 推送

- 范围：撤回误推到 GitHub `drongzzz0/scd-new` 的 `a4d611c` commit。
- 目标：不改写历史、不强推，通过 revert commit 抵消已推送代码改动。
- 约束：后续不得在未明确授权时向 GitHub 执行写操作。
- 实施方式：执行 `git revert --no-edit a4d611c791f493ef8d814c81475eb2d616477f70`，生成并推送 `696583c599adbdeb3e1b5a463f4d824b2ecde120`。
- 当前状态：本地与远端 `main` 同步，`run_paper_eval.py` 相对 `f7d7854` 无 diff。

## 2026-05-22 - Cross-scale KV reuse v1.3 修订

- 范围：根据 `deep-research-report.md` 审阅意见，在保留 v1.2 原件的前提下生成 `cross_scale_kv_reuse_patch_plan_v1_3.md`。
- 目标：把 v1.2 中新增但未完全传导的 GQA-aware 成本、KV-native Patch 维度、materialization/network 成本、online risk proxy、Phase A loss、强系统 baseline 和 P0/P1/P2 执行顺序统一到正文。
- 假设：该方案是 SCD/ICML 后续整理与新研究方向，不混入已经定稿的 ICML camera-ready 投稿包。
- 约束：本轮只改本地 Markdown 文档；不改 GitHub；不运行实验；不声称任何新实验结果。
- 设计选择：新建 v1.3 文件而不是覆盖 v1.2；删除旧的粗略 small/large 占比和超小 Patch 预算叙事；默认主线为 KV-native token-wise/local-mixer Patch，sequence-mixing Patch 作为备选/消融。
- 实施方式：先复制 v1.2，再定点修订版本定位、§1.3 成本模型、§4.3 Patch 参数化、§4.4 online risk proxy、§4.3b 训练协议、§8/§9 实验与 baseline、§13 执行计划和附录 B。

## 2026-05-24 - main-new 模型族名调整

- 范围：按用户要求，将 `main-new.md` 中的 `Qwen2.5` 统一替换为 `Qwen3.5`。
- 约束：本轮只做模型族名的机械替换，不重新核算表格数值、配置字段或实验资源估算；未改 GitHub。

## 2026-05-24 - SCD-NEW 工作区导入后结构盘点

- 范围：扫描当前 `C:\Users\mql16\Desktop\SCD-ICML` 工作区，识别研究管线状态、论文/代码/证据/归档分层、运行时代码入口、依赖和明显风险。
- 目标：在不依赖用户重新说明上下文的前提下，给出当前项目做什么、主模块在哪里、哪些文件最重要、哪些设置缺失，以及下一步最值得做什么。
- 假设：当前导入的工作区名为 `SCD-NEW`，但物理路径仍是 `SCD-ICML`；`instance.json` 是 Dr. Claw 管线目录映射的来源；`final/` 与 `ICML-ready/` 是当前投稿材料主线；`Experiment/core_code/runtime_repo/` 是当前工作区内保留的运行时代码快照。
- 约束：只在当前项目目录内读写；不访问外部 canonical 本地仓库 `C:\Users\mql16\Desktop\MAC share\scd-new`；不推送 GitHub；不重新运行远程实验；不重新编译 LaTeX 以避免改动投稿产物。
- 设计选择：把项目按四条线分析：Dr. Claw 管线元数据、ICML camera-ready 投稿包、SCD runtime Python 包、历史证据与归档；把外部代码仓库清单视为状态说明而非当前目录内可验证事实。
- 实施方式：验证 `instance.json` 所有绝对路径均位于项目内；读取 `.pipeline`、`final/`、`ICML-ready/`、`docs/` 和 `Experiment/core_code/runtime_repo/`；提取 `pyproject.toml`、README、TeX、编译日志、证据记录和主要 Python 模块；用 `python -m py_compile` 对关键运行时代码做静态语法检查。

## 2026-05-24 - SCD-NEW 导入后工作区结构扫描

- 范围：扫描当前 `SCD-ICML` 工作区的项目状态文件、阶段目录、最终投稿包、证据文件、代码清单、顶层方案文档和归档/临时目录。
- 目标：判断导入后的项目结构、技术栈、入口点、依赖线索、当前实现状态、明显风险和最有价值的下一步。
- 假设：本轮是导入后盘点，不等同于新实验运行；`final/` 和 `ICML-ready/latex_camera_ready_unzipped/` 代表已整理的投稿材料，`main-new.md` 代表后续新研究方案主线。
- 约束：遵守项目沙盒，只在当前工作区内读取/记录；不访问外部 `scd-new` 仓库；不更新 `.pipeline/tasks/tasks.json` 或 `.pipeline/docs/research_brief.json`，因为用户要求先扫描而非重建 pipeline。
- 设计选择：优先用 `instance.json` 验证路径安全，再以 `final/README.md`、`final/code/scd-new-manifest.md`、`final/evidence/verified-results.md`、最终 LaTeX 源和项目审计文档作为事实来源。
- 实施方式：枚举文件类型和阶段目录内容，抽取论文标题/摘要/章节/图表引用，检查依赖清单是否存在，并汇总缺失 setup、pipeline 状态与可继续工作的入口。

## 2026-05-24 - Dr. Claw project startup

- 范围：为当前 `SCD-ICML` 工作区补齐 Dr. Claw 研究 pipeline 的启动状态。
- 目标：创建缺失的 `.pipeline/docs/research_brief.json`，填充 `.pipeline/tasks/tasks.json`，标记 intake 完成，并给后续 survey / ideation / experiment 阶段提供可执行任务入口。
- 假设：当前启动对象是 `main-new.md` 代表的 SCD 后续研究方向，而不是已经整理完成的 ICML camera-ready 投稿包；由于 follow-up 仍未完成 source-verified collision scan 和 P0 sanity check，默认从 `survey` 阶段开始。
- 约束：项目本地 `skills-index.md` 列出 `inno-pipeline-planner`、`academic-researcher`、`inno-idea-generation`，但对应 `SKILL.md` 文件在 `.agents/skills/`、`.claude/skills/`、`.gemini/skills/` 下均不存在；因此本轮按项目 workflow 与现有文档执行，不声称已运行这些 skill 程序。
- 设计选择：active stages 设为 `survey` / `ideation` / `experiment`；`publication` / `promotion` 延后到 P0 证据闭环后；首个下一步任务设为刷新相关工作与撞车扫描。
- 实施方式：生成 research brief、任务清单、启动报告 `Survey/reports/project-startup-2026-05-24.md`，更新 `.pipeline/config.json` 的 `intakeCompleted` 与 `effectiveStartingStage` 字段。
## 2026-05-24 - ICML completed-material archive for extension work

- Scope: move completed SCD-ICML camera-ready artifacts, superseded drafts, old
  migration records, temporary build outputs, and previous archive subfolders
  into `archive/icml-completed-2026-05-24/`.
- Goal: keep the project root focused on the next extension/journal phase while
  preserving all ICML materials for recovery and audit.
- Assumptions: `main-new.md` is the active extension plan; `docs/scd-next-paper-roadmap-2026-05-20.md`
  and `docs/scd-vulnerability-and-revision-2026-05-21.md` remain active context.
- Constraints: local-only cleanup; no deletion; no GitHub write; no movement
  outside `C:\Users\mql16\Desktop\SCD-ICML`.
- Implementation: moved `ICML-ready/`, `final/`, root superseded reports/drafts,
  completed docs, `tmp/`, and earlier archive subfolders into the consolidated
  archive folder.

## 2026-05-24 - Runtime baseline import for extension work

- Scope: copy the inspected `scd-new` runtime subdirectory into
  `Experiment/core_code/runtime_repo/`.
- Goal: make the active workspace self-contained for the next extension
  implementation phase without keeping the old ICML completed artifacts in the
  project root.
- Assumptions: commit `696583c599adbdeb3e1b5a463f4d824b2ecde120` is the
  reverted GitHub baseline inspected in the previous step; the runtime is a
  starting scaffold, not a finished implementation of `main-new.md`.
- Constraints: copied only local files; no GitHub write; did not import the full
  repository history or old paper/archive material.
- Implementation: copied
  `C:\Users\mql16\Desktop\MAC share\scd-new\Experiment\core_code\runtime_repo`
  to `Experiment/core_code/runtime_repo/` and added
  `Experiment/core_code/runtime_repo_import_manifest.md`.

## 2026-05-24 - Deep survey retry after skill restoration

- Scope: run the requested deep-research pass for `main-new.md` using the restored global skills `inno-deep-research`, `gemini-deep-research`, `academic-researcher`, `biorxiv-database`, `dataset-discovery`, and `inno-code-survey`.
- Goal: refresh the related-work collision scan, verify model-family assumptions, and map datasets/code references before committing to experiments.
- Assumptions: the active research line is same-family cross-scale KV/state expansion; no new experiment result exists yet; `Qwen3.5` naming in `main-new.md` must be source-verified before use.
- Constraints: `GEMINI_API_KEY` was not set, so the Gemini helper script was not run; `biorxiv-database` returned zero relevant results; `dataset-discovery` was limited by missing `gh` and Semantic Scholar HTTP 429; no repositories were cloned and no runtime code was changed.
- Design choices: treat C2C, PrefillShare, ICaRus, RelayCaching, and SmallKV as direct or near-direct collision threats; use Qwen2.5 `1.5B -> 7B` as the first dense-Transformer falsification pair; defer Qwen3.5 experiments until hybrid state support is designed.
- Implementation: added `Survey/reports/related-work-collision-scan.md`, `Survey/references/related-work-sources.md`, `Survey/reports/model-config-verification.md`, and `Survey/reports/dataset-and-code-survey-2026-05-24.md`; updated pipeline state so the next task is `ideation-001`.

## 2026-05-24 - Ideation route selection and evaluation

- Scope: execute the requested idea-generation/evaluation pass using `inno-idea-generation`, `inno-idea-eval`, `brainstorming-research-ideas`, and `creative-thinking-for-research`.
- Goal: turn the refreshed survey into a selected, falsifiable research route and evaluate it before starting experiments.
- Assumptions: the user's task description after the colon was empty, so this was routed to the current pipeline task `ideation-001`; no experimental evidence exists yet.
- Constraints: the `search_ai_papers.py` dependency referenced by `inno-idea-eval` was not installed, so active novelty verification used the existing survey plus direct source lookup; no repositories were cloned and no runtime code was modified.
- Design choices: select `Risk-Calibrated Cross-Scale KV Prediction` as a narrowed version of `main-new.md`; treat Apple/CoreNet KV Prediction as a direct high-overlap prior; keep Qwen2.5 `1.5B -> 7B` as the first falsification pair; defer Qwen3.5 hybrid-state work.
- Evaluation: 3-persona meta-review decision `accept`, overall score `6.40/10`; novelty score `5/10` due to KV Prediction/C2C/PrefillShare/ICaRus overlap.
- Implementation: added ideation artifacts under `Ideation/ideas/`, including `idea_candidates.md`, `selected_idea.txt`, `selected-research-route.md`, `novelty_grounding_report.txt`, persona reviews, JSON logs, and `eval_report.txt`; updated pipeline state so the next task is `experiment-001`.

## 2026-05-25 - Compression-guided route draft

- Scope: write a standalone technical-flow draft for the user's refined idea: first learn a large-to-small compression path, then use it as the scaffold for small-to-large KV expansion.
- Goal: clarify how the method differs from KV Prediction while preserving the useful small-model-to-large-KV prediction framing.
- Assumptions: the draft is for review only and does not yet replace `Ideation/ideas/selected-research-route.md` or pipeline task state.
- Design choices: rename the route to `Compression-Guided Risk-Calibrated Cross-Scale KV Expansion`; keep Qwen2.5 `1.5B -> 7B` as P0; treat Qwen2.5 `7B -> 72B` as the later setting where learned compression-path grouping matters most.
- Implementation: added `Ideation/ideas/compression-guided-cross-scale-kv-expansion.md`.

## 2026-05-25 - Clarify offline Patch training in compression-guided route

- Scope: refine `Ideation/ideas/compression-guided-cross-scale-kv-expansion.md` after user clarified that offline work must include both compression-path discovery and Small-to-Large Patch training.
- Goal: avoid the misleading interpretation that the learned compression path can simply be transposed and used online.
- Design choice: explicitly state that the compression path is a scaffold / structural prior / initialization / routing basis, while the Patch network is the actual generator of large-native KV.
- Implementation: updated the one-sentence summary, offline-stage flow, online-stage wording, Phase B definition, and final route summary.
- Follow-up cleanup: replaced remaining shorthand like "then reverse-expand" with wording that says the learned compression path guides Patch inputs, target-layer conditioning, initialization, and training.

## 2026-05-25 - Align active route and pipeline files to compression-guided Patch

- Scope: update the active route/process files after the user requested correction of the technical route.
- Goal: make `main-new.md`, ideation route files, and pipeline state consistently describe the method as offline compression-path discovery plus offline Small-to-Large Patch training, followed by online Patch generation and risk routing.
- Design choice: keep historical evaluation/review logs unchanged, because they record the earlier `Risk-Calibrated Cross-Scale KV Prediction` evaluation state; update only current route source files and pipeline pointers.
- Implementation: updated `main-new.md`, `Ideation/ideas/selected-research-route.md`, `Ideation/ideas/selected_idea.txt`, `Ideation/ideas/compression-guided-cross-scale-kv-expansion.md`, `.pipeline/docs/research_brief.json`, and `.pipeline/tasks/tasks.json`.

## 2026-05-25 - Replan pipeline for compression-guided Patch route

- Scope: run `inno-pipeline-planner` against the corrected current route.
- Goal: make the active pipeline start at experiment stage and break the next work into falsification gates for cost, trace capture, compression-path sanity, Patch training, risk routing, and end-to-end evaluation.
- Assumptions: survey and ideation are complete enough to preserve as historical records; no new experiment result exists yet; Qwen2.5-1.5B -> 7B remains the first falsification pair.
- Design choices: do not regenerate from scratch; retain completed startup/survey/ideation tasks; update only current route state, active experiment tasks, brief quality gates, and config stage metadata.
- Implementation: updated `.pipeline/docs/research_brief.json`, `.pipeline/tasks/tasks.json`, and `.pipeline/config.json`.

## 2026-05-25 - Complete experiment-001 cost-model audit

- Scope: execute the current pipeline task using the requested experiment-development, experiment-analysis, resource-preparation, and initial-analysis workflows.
- Goal: determine whether the compression-guided Patch route has enough arithmetic room to justify paired trace capture before heavy Patch training.
- Inputs: verified Qwen2.5 configs, related-work/baseline reports, selected route documents, runtime repo manifest, and `main-new.md`.
- Decision: conditional go to `experiment-002`; P0 should proceed to paired trace capture, but only after storage, tensor shapes, GPU placement, and materialization path are specified.
- Key risk: Patch FLOPs are not the immediate blocker; target-native KV materialization, paged write, cross-device/cross-node transfer, and selective recompute fraction are the main wall-clock risks.
- Implementation: added `Experiment/analysis/cost-model-break-even.md`, `Ideation/references/logs/prepare_resources_experiment_001.json`; appended `docs/experiment-log.md`; updated `.pipeline/docs/research_brief.json`, `.pipeline/tasks/tasks.json`, and `.pipeline/config.json`.

## 2026-06-02 - Sync Task T follow-ups and register AW plan

- Scope: synchronize the local project state after the 16k speed-recovery follow-up chain and repair the pipeline task registry.
- Goal: make the current experimental blocker explicit before any new GPU work: 16k quality is supported, but positive 16k wall-clock speed is not supported under the current prefix-depth runtime model.
- Evidence synchronized: AF layer27 decode-aware adapter, AG prefix-depth constrained route benchmark, AH upper-layer joint adapters, AI cross-layer xnorm native Patch, AJ sparse Patch runtime audit, AM boundary joint adapter, and W patch optimization summary.
- Evidence boundary: AF/AG/AH summaries were already local canonical files. W/AI/AJ/AM summaries were copied from `tmp_remote_docs` into `Experiment/analysis/long_context`; their source CSVs are not local. AR/AS/AT/AU/AV scripts exist in the root, but their result CSV/Markdown files are not in the local canonical analysis directory and are treated as pending remote-result verification.
- Design choice: mark Task T / `experiment-006b` as completed with a negative speed-recovery verdict, not as a successful speed claim. Register Task AW / `experiment-006c` as the next gated experiment plan.
- Implementation: added `Experiment/analysis/long_context/current_status_2026-06-02.md` and `Experiment/analysis/long_context/AW_upper_boundary_patch_ablation_plan.md`; repaired `.pipeline/tasks/tasks.json` into valid JSON; updated `.pipeline/docs/research_brief.json` and `.pipeline/config.json`.
- Next decision: pull/verify missing AR/AS/AT/AU/AV artifacts if they exist remotely, then either run AW as a gated upper-boundary Patch smoke or pivot to a runtime design for non-contiguous exact anchors.

## 2026-06-03 - Project route revision after layer22 Patch failures

- Scope: revise the active experimental interpretation after AX/AY/AZ layer22 repair attempts all failed the n=40 F1-retention gate.
- Goal: prevent further GPU spending on small/medium Patch variants that do not address the observed bottleneck.
- Evidence: AX layer22 native xnorm/KV retention 0.7346; AY layer22 sequence hidden-bridge retention 0.0717; AZ layer22 cross-layer Transformer residual retention 0.5087. The continuation gate was 0.90.
- Decision: keep layer22 exact for the current Qwen2.5-1.5B -> 7B route. Stop AY/AZ 200-sample expansion. Treat the original 40% wall-clock speed gate as unsupported by the current lightweight Patch family.
- Implementation: added `SPEED_GATE_REVISION_2026-06-03.md` and `PATCH_NEGATIVE_RESULT_WRITEUP_2026-06-03.md`; appended the experiment log.
- Next decision: either write the current result as a method-boundary/negative result, or launch a separately scoped large teacher-student Patch training phase with more QA-complete data and behavioral/attention-output supervision.

## 2026-06-03 - Publication claim boundary register

- Scope: complete `publication-001` by creating a claim-evidence register for the current experiment chain.
- Goal: prevent overclaiming and make the paper boundary explicit before drafting.
- Design choice: separate supported claims, refuted/prohibited claims, and future-work-only claims; use the 200-sample QA-complete baseline and AX/AY/AZ n=40 gates as the controlling evidence.
- Implementation: added `Publication/paper/claim-evidence-register.md`; marked `publication-001` complete in `.pipeline/tasks/tasks.json`; updated `.pipeline/docs/research_brief.json` to point to the claim register.
- Next decision: either draft the result/limitation section from the register, or start a new separately scoped teacher-student Patch training phase.

## 2026-06-03 - Results and limitations draft

- Scope: create paper-facing Results and Limitations text from the completed claim register.
- Goal: make the current negative speed result directly usable in a manuscript while avoiding unsupported acceleration claims.
- Design choice: emphasize three tables: QA-complete 16k quality/runtime, runtime decomposition, and layer22 repair attempts; explicitly state that the current route is a method-boundary result.
- Implementation: added `Publication/paper/results-and-limitations-draft.md`.
- Next decision: draft Introduction/Discussion around the method-boundary framing, or start a new teacher-student Patch training phase if the project continues experimentally.

## 2026-06-03 - Introduction and discussion draft

- Scope: continue publication preparation by drafting the framing sections for the negative speed result.
- Goal: make the manuscript argument coherent: the project started as a cross-scale Patch acceleration route, but the validated result is a method-boundary diagnosis centered on layer22.
- Design choice: avoid acceleration claims; emphasize QA-complete evaluation repair, quality/speed separation, exact-anchor runtime dominance, and layer22 as the hard anchor.
- Implementation: added `Publication/paper/introduction-and-discussion-draft.md`.
- Next decision: assemble a full paper skeleton, or pause writing and launch a new large teacher-student Patch training phase.

## 2026-06-04 - Task Y joint behavior-loss Patch gate closure

- Scope: enter the requested upper-block / route-level joint Patch phase after the lightweight NativeKV and hidden-bridge branches failed to close the 16k speed gate.
- Goal: test whether behavior-loss joint training can repair the late exact anchors [20,23,27] and solve the single-layer-success / multi-layer-collapse problem.
- Assumptions: QA-complete 16k HotpotQA/QA construction is the valid end-task dataset; stale missing-question traces are invalid; no prefix-k redesign, prompt-level router, dense concat cross-layer predictor, or full-layer decode-aware MLP is in scope.
- Design choice: use quick-screen gates before any 200-sample promotion. Upper-block evidence comes from AH/YB gates; route-level joint training uses teacher behavior losses (KL/CE/top-k) plus sampled KV reconstruction, not KV-only MSE.
- Result: Task Y failed. Upper-block candidates did not pass promotion; route-level plus27 retention was 0.8812, plus23_27 retention was 0.7122, plus20_23_27 retention was 0.7236, and all route-level candidates were slower than full large.
- Plan change: do not promote Task Y to 200-sample expansion or full-layer stress testing. The lightweight Patch family is now exhausted for the 16k speed claim. The next plan must either write the speed-gate revision / negative-result section or explicitly start a new larger teacher-student native-state Patch phase with a separate budget.

## 2026-06-04 - Task Z full-layer Patch-only phase closure

- Scope: start the explicitly requested full-layer joint Patch-only phase after Task Y failed to repair late exact anchors. This phase tests the 0-exact final form rather than another local route or single-layer repair.
- Goal: determine whether all 28 large-model KV/native states can be generated from the small model with enough signal to justify a larger teacher-student Patch phase.
- Assumptions: QA-complete 16k prompts are mandatory; stale missing-question traces remain invalid. Stage 0 is smoke-only; formal Stage 1 requires a de-duplicated 200/40/40 split and therefore at least 280 QA-complete records.
- Design choice: implement native-state Patch rather than arbitrary KV generation: small xnorm features feed a compact shared trunk plus layer-specific heads, predicted large xnorm is projected through the frozen Qwen2.5-7B native k_proj/v_proj and RoPE path, and route-level suffix KL/CE/top-k behavior losses are included with sampled xnorm/KV reconstruction.
- Compute constraint: full 16k behavior-loss backprop through a 28-layer patched cache OOMs on a 24GB GPU. The executable Stage 0 used a 2048-token tail cap for training while preserving the Question/Answer tail; evaluation still used full-length inference.
- Result: Stage 0 failed as a signal test. Loss decreased, but n=10 held-out full-layer Patch-only F1/EM were both 0.0000, runtime was 2956.47 ms versus 2055.76 ms full-large, and debug samples show repetitive templated outputs rather than answers.
- Plan change: do not expand full-layer Patch-only to Stage 2. Stage 1 is not currently runnable because only 200 QA-complete records are available. Any continuation requires both more QA-complete data and a memory-feasible training design, not another lightweight route tweak.


## 2026-06-03 - Task X NativeKV Patch repair standardization

- Scope: standardize the targeted NativeKV / compact cross-layer Patch repair evidence into the required Task X output format.
- Goal: determine whether unsafe layers can be repaired enough to reduce exact/recompute layers and recover positive wall-clock speed on 200 QA-complete 16k samples.
- Evidence: reused existing W/Y/AA/AB/AX/AY/AZ artifacts rather than retraining. Compact pooled residual failed on layers 5/20/22/23; native xnorm-to-KV repaired layers 5 and 22 locally; layer23 passed only in isolation; layer27 and remaining high anchors stayed unsafe.
- Decision: PASS-B only. Best route `Y_24_5_22` reaches F1=0.4439, EM=0.3200, F1 retention=93.68%, total=2195.72 ms, saving=-10.35% versus full large. This is local repair, not a positive-speed deployment.
- Implementation: added `Experiment/analysis/long_context/X_nativekv_patch_summary.md`, `X_nativekv_patch_quick_screen.csv`, `X_nativekv_patch_200eval.csv`, `X_nativekv_patch_per_sample.csv`, `X_nativekv_patch_best_route.json`, `X_nativekv_patch_debug_samples.md`, and `X_nativekv_patch_trace_audit.md`; appended `docs/experiment-log.md`.
- Next decision: either pursue a separately scoped stronger teacher-student / decode-aware NativeKV phase for layer27/23, or keep the publication framing as method-boundary / speed-negative.
## 2026-06-04 - YB Direction-B Hidden Bridge Smoke Closure

Scope: follow the user's selected `B` branch after the sparse-exact / layer27 repair path failed. This branch changes the Patch target from KV-only cache prediction to an upper-block native hidden-state bridge: predict the large layer-24 entry hidden state from small-side layer-22 features, then let Qwen2.5-7B layers 24-27 run normally.

Assumptions and constraints: no prefix-k redesign, no prompt-level router, no full decode-aware MLP retraining, no dense concat cross-layer predictor. The test is a gated smoke only: train on 20 QA-complete 16k samples, evaluate on 10 samples, and promote only if F1 retention reaches at least 0.80.

Implementation approach: add `run_task_yb_sequence_hidden_bridge.py`, a zero-init Conv1D sequence residual adapter on top of the YA per-block bridge. Two feature variants were tested: small layer-22 hidden and small layer-22 xnorm. Both use bridge layer 24, hidden=512, kernel=9, epochs=3.

Result: YB hidden retention = 0.4601; YB xnorm retention = 0.4835. These are effectively unchanged from YA linear bridge retention (0.4601 / 0.4835) and far below the 0.80 smoke promotion gate.

Plan change: do not promote Direction B to n=40 or n=200 under the current shallow bridge formulation. The active project plan returns to speed-gate revision / negative-result framing unless a separately scoped, larger teacher-student native-state Patch phase is explicitly started.


## 2026-06-04 - Plan change: cut-point route blocked at training feasibility

- Scope change: after Task Z failed, the project tested a new cut-point formulation instead of continuing lightweight layer-route search.
- Evidence: G0 c=20 oracle hidden passed with F1 retention 1.0, confirming the upper-block mechanism and hook/decode path are valid.
- Blocking condition: the required strong CutHiddenPredictor with behavior-first 16k training OOMed at 0 steps on 24GB GPU, before G1 could be evaluated.
- Next plan decision: either allocate larger-memory compute or approve a memory-feasible redesign that keeps the scientific contract; otherwise return to speed-gate revision / negative-result framing.

## 2026-06-07 - SCD-TPAMI workspace migration preparation

- Scope: inspect the current `SCD-ICML` project state and prepare a separate desktop workspace for a new `SCD-TPAMI` project.
- Goal: keep the current ICML follow-up workspace intact while creating a clean target directory for TPAMI-oriented migration.
- Assumptions: the target workspace should not blindly inherit stale absolute paths, venue labels, or experiment status from `SCD-ICML`; any copied pipeline metadata must be rewritten to point under `C:\Users\mql16\Desktop\SCD-ICML1`.
- Constraint: avoid destructive moves or overwrites. The first step is directory creation only; copying, pruning large artifacts, and path rewriting should be done as a separate explicit migration step.
- Current observation: `instance.json`, `.pipeline/docs/research_brief.json`, and `.pipeline/tasks/tasks.json` are present and still identify the project as `SCD-ICML`; the active task is the cut-point route review after predictor-training OOM.

## 2026-06-07 - SCD-TPAMI workspace migration completed

- Scope: renamed the newly created desktop target folder from `SCD-TPAMI` to `SCD-ICML1` and copied the full `SCD-ICML` workspace into it.
- Migration mode: copy-only; the original `C:\Users\mql16\Desktop\SCD-ICML` workspace remains intact.
- Workspace identity: the physical folder is `C:\Users\mql16\Desktop\SCD-ICML1`; the logical project label is `SCD-TPAMI`.
- Metadata update: rewrote canonical paths in `instance.json` and `.pipeline/docs/research_brief.json` from the source root to the new target root, and updated `.pipeline/config.json` to mark the migration state.
- Active continuation point: keep the cut-point route review as the current decision state unless the TPAMI plan is re-scoped.

## 2026-06-07 - Migration intent clarified

- Clarification: this folder, `C:\Users\mql16\Desktop\SCD-ICML1`, is the copied workspace for completed / old `SCD-ICML` project files and code organization.
- Correction to the previous migration-completed entry: this folder should not be treated as the logical `SCD-TPAMI` project. It is an ICML old-project archive / working copy.
- Metadata correction: `instance.json`, `.pipeline/docs/research_brief.json`, and `.pipeline/config.json` were reset to ICML archived-copy semantics while keeping canonical paths under `C:\Users\mql16\Desktop\SCD-ICML1`.

## 2026-06-07 - Workspace canonical path correction back to SCD-ICML

- Scope: correct the active `C:\Users\mql16\Desktop\SCD-ICML` workspace metadata after startup inspection found stale canonical paths pointing to `C:\Users\mql16\Desktop\SCD-ICML1`.
- Reason: the current thread is operating inside `SCD-ICML`, and the project rule requires `instance.json` canonical paths to remain descendants of the current project root before pipeline work continues.
- Metadata correction: rewrote absolute paths in `instance.json` and `.pipeline/docs/research_brief.json` from `C:\Users\mql16\Desktop\SCD-ICML1` to `C:\Users\mql16\Desktop\SCD-ICML`.
- Constraint: no experiment results, task statuses, or claim boundaries were changed.

## 2026-06-07 - Pipeline config cleanup after path correction

- Scope: remove stale migration/task metadata from `.pipeline/config.json` in the active `C:\Users\mql16\Desktop\SCD-ICML` workspace.
- Reason: `.pipeline/config.json` still described the prior copy into `SCD-ICML1` and kept `experiment-006c` / AW as the active task, while `.pipeline/tasks/tasks.json` and the research brief identify the cut-point predictor OOM review as the current decision point.
- Metadata correction: set `currentTaskId` to `approve-memory-feasible-cut-training-redesign-or-larger-gpu`, set the active task title to the cut-point predictor OOM review decision, and updated `lastStateSync` to the corrected `SCD-ICML` workspace state.
- Constraint: no experiment results, task statuses, research claims, or pipeline stage definitions were changed.

## 2026-06-07 - SCD-TPAMI content quarantine for manual deletion

- Scope: isolate files and directories that are clearly TPAMI-specific or remote TPAMI workspace copies into `SCD-TPAMI_to_delete/`.
- Reason: the active workspace should remain a single `SCD-ICML` project, while TPAMI-labeled materials should be grouped for later manual deletion.
- Moved items: `PATCH_TECH_PATH_STATUS_2026-05-31.md`, `CURRENT_PROGRESS_AND_RESULTS.md`, `docs/CURRENT_PROGRESS_SUMMARY_2026-05-27.md`, `docs/CURRENT_PROGRESS_SUMMARY_2026-05-29.md`, `remote_work/CURRENT_PROGRESS_AND_RESULTS.md`, `tmp_remote_docs/experiment-log.md`, and `remote-ser/scd-tpami/`.
- Constraint: kept `docs/project-plan-log.md` and `docs/experiment-log.md` in place as project-level history even though they mention TPAMI.

## 2026-06-07 - Delete isolated SCD-TPAMI folder

- Scope: delete the previously quarantined `SCD-TPAMI_to_delete/` directory from the active `SCD-ICML` workspace.
- Reason: the user confirmed manual cleanup should proceed after verification showed the directory was not referenced by `.pipeline/tasks/tasks.json`, `.pipeline/docs/research_brief.json`, or the GitHub `drongzzz0/new` tree.
- Deleted content: 33 files, 449637 bytes, all under `C:\Users\mql16\Desktop\SCD-ICML\SCD-TPAMI_to_delete`.
- Constraint: no other project files or pipeline state were deleted.

## 2026-06-07 - Restore GitHub new auxiliary core code into active workspace

- Scope: restore the GitHub `drongzzz0/new` auxiliary core-code scaffold into the active `C:\Users\mql16\Desktop\SCD-ICML\Experiment\core_code` tree.
- Reason: `Experiment/core_code/runtime_repo/` was complete, but the current root was missing the GitHub `new` helper layer listed in `docs/final-version-map.md`: `scd_validation/`, `configs/`, pair-screening/original-baseline entry scripts, and remote run scripts.
- Source: copied missing files from the local GitHub unpacked baseline `new-main/new-main/Experiment/core_code/`.
- Implementation: copied 64 missing files only; existing current files were not overwritten, and `runtime_repo/` source files were not changed.
- Metadata correction: fixed `Experiment/core_code/runtime_repo_import_manifest.md` to identify the GitHub baseline as `drongzzz0/new`.

## 2026-06-07 - SCD-ICML Mistral reproduction implementation plan

- Scope: build a `ser`-runnable reproduction path for the SCD-ICML paper results using the current codebase.
- Target: reproduce or approximately reproduce the Mistral 7B pair results; skip the Qwen-32B pair because the current `ser` machine cannot host it.
- Paper-facing model scope: `MistralLite -> Mistral-7B` plus the reverse `Mistral-7B -> MistralLite` when running the full Mistral matrix.
- Paper-facing dataset scope: CMRC2018, HotpotQA, and WikiText-2. CMRC/Hotpot report QA F1/EM; WikiText-2 primarily reports teacher-forced NLL/PPL.
- Implementation choice: do not use the synthetic/surrogate `infer + eval_cmd` path as claim evidence. Add a strict text-evaluation path that reuses the real transformers runtime, cache assembly, reuse/patch checkpoints, and route logic.
- Remote execution default: use `/disk_n/c101-3-mql/` paths on `ser`, `/disk_n/c101-3-mql/venvs/A800/bin/activate`, and currently idle GPUs selected by `nvidia-smi`.
- New code surface: `scd_repro.text_benchmarks`, `scd_repro.cli.eval_text_strict`, Mistral mql model configs, `configs/eval_scd_icml_main_methods.json`, and `scripts/run_ser_scd_icml_mistral_repro.sh`.
- Verification plan: first run a one-sample real-model smoke on `ser`; only then scale to the requested Mistral reproduction matrix.

## 2026-06-07 - Plan change: default public reproduction follows paper Mistral direction only

- Scope change: narrow the default open-source reproduction wrapper from the two-direction Mistral matrix to the paper-facing `MistralLite -> Mistral-7B` direction.
- Reason: the camera-ready paper explicitly anchors the canonical Mistral reproduction on `MistralLite -> Mistral-7B`; the reverse `Mistral-7B -> MistralLite` direction is useful as a diagnostic but should not be the default claim-facing run.
- Implementation choice: `scripts/run_ser_scd_icml_mistral_repro.sh` now defaults to `DIRECTIONS=lite_to_7b`. The reverse direction remains available only when explicitly requested with `DIRECTIONS=lite_to_7b,7b_to_lite`.
- Evidence: the verified `ser` run completed the canonical 200-sample strict evaluations for CMRC2018, HotpotQA, and WikiText-2 under `lite_to_7b`; no Qwen-32B run was attempted.
- Release framing: document the output as an approximate strict-runtime reproduction of the SCD-ICML trend, not an exact reprint of the camera-ready paper table.

## 2026-06-07 - Plan change: package the reproduction path for open-source use

- Scope change: move the Mistral reproduction path from an ad hoc remote run into release-facing runtime documentation and reusable summary tooling.
- Reason: the open-source code should let users regenerate the strict-runtime result table from completed dataset summaries, especially after long runs are interrupted or resumed.
- Implementation choice: add `scd_repro.cli.summarize_mistral_repro`, register `summarize-mistral-repro` in `pyproject.toml`, and make `scripts/run_ser_scd_icml_mistral_repro.sh` call the CLI instead of embedding one-off Python at the end of the shell script.
- Documentation choice: add `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md` as the release-facing reproduction entry point and keep the broader project evidence note in `docs/scd-icml-mistral-reproduction-2026-06-07.md`.
- Verification: locally compiled the new CLI and parsed `pyproject.toml`; on `ser`, compiled the CLI, ran `bash -n` on the wrapper, and regenerated the aggregate summary for the verified 200-sample run with 18 expected rows.

## 2026-06-07 - Plan change: add portable release setup and readiness gate

- Scope change: add a portable setup layer around the verified `ser` reproduction so the eventual GitHub release is not tied only to `/disk_n/c101-3-mql/` paths.
- Reason: users should be able to copy a model-config template, point `DATA_ROOT` to their local benchmark files, and run the same strict Mistral reproduction path without editing the shell wrapper.
- Implementation choice: add `configs/models_mistrallite_to_mistral7_template.json`, `DATA_PREPARATION.md`, and environment-variable overrides `MODELS_CONFIG_LITE_TO_7B` / `MODELS_CONFIG_7B_TO_LITE` in `scripts/run_ser_scd_icml_mistral_repro.sh`.
- Release hygiene choice: add `OPEN_SOURCE_RELEASE_MANIFEST.md` and `scripts/check_release_readiness.py` to distinguish the claim-facing required files from older historical scripts/configs.
- Verification: local release readiness check passed; remote release readiness check passed on the synced runtime repo and verified the existing 200-sample run root contains the expected 18 aggregate rows for `lite_to_7b`.

## 2026-06-07 - Plan change: create non-destructive release preview package

- Scope change: build a release-preview tree rather than deleting historical files from the active runtime repo.
- Reason: the user needs a credible open-source package, but old scripts/configs may still be useful as traceability evidence until the final publication tree is approved.
- Implementation choice: add `scripts/build_release_tree.py`, include it in `OPEN_SOURCE_RELEASE_MANIFEST.md`, and make `scripts/check_release_readiness.py` compile to a temporary directory so release checks do not leave `__pycache__` artifacts in the preview package.
- Preview output: local preview at `Experiment/core_code/runtime_repo/release_preview/scd_icml_runtime_release`; remote preview at `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release`.
- Verification: local and remote preview readiness checks passed; both previews contain 60 files, no `__pycache__` directories, and the remote check validated the verified 200-sample `lite_to_7b` aggregate result table.
- Cleanup boundary: the preview report lists 6 root-level historical files, 71 old configs, and 34 old scripts as not included in the default release surface, but no source files were deleted.

## 2026-06-07 - Plan change: make release wrapper safer on shared GPUs

- Scope change: update the release wrapper defaults so public reproduction runs wait for sufficient free GPU memory before capture, train, and eval stages.
- Reason: direct users should not get an avoidable CUDA OOM because the wrapper selected a low-memory-used but still busy card. This is especially important for `ser`, where other jobs may occupy GPUs during reproduction.
- Implementation choice: set default gates to `WAIT_CAPTURE_FREE_MB=18000`, `WAIT_TRAIN_FREE_MB=6000`, and `WAIT_EVAL_FREE_MB=18000`; keep environment variables overrideable for users who have inspected their GPU state.
- Verification: local and remote readiness checks passed after the change; the release preview was rebuilt on both machines and still validates the existing 200-sample `lite_to_7b` aggregate table.
- Runtime decision: a release-preview real-model smoke was not launched at 2026-06-07 22:25 CST because `ser` had only one sufficiently idle GPU. The next smoke should wait until at least two GPUs have about 18GB free.

## 2026-06-07 - Plan change: make automatic GPU selection resilient to shared-server races

- Scope change: strengthen the wrapper's automatic GPU selection so a public release run does not get stuck on an initially selected GPU if another job occupies it before model loading starts.
- Reason: the first release-preview smoke inherited or selected `CUDA_VISIBLE_DEVICES=3,1`, then waited on GPU1 after GPU1 became busy. This is safe but too brittle for public users on shared servers.
- Implementation choice: add `pick_free_gpus`, `AUTO_SELECTED_CUDA_VISIBLE_DEVICES`, and re-selection inside two-GPU capture/eval waits. When automatic mode is active, the wait loop can re-scan all physical GPUs and update `CUDA_VISIBLE_DEVICES` if a different pair satisfies the free-memory threshold.
- Verification: local source readiness passed; local release preview rebuilt and passed; remote source and release preview readiness passed; remote preview script contains `AUTO_SELECTED_CUDA_VISIBLE_DEVICES` and `reselected CUDA_VISIBLE_DEVICES`.
- Runtime status: stopped the pre-reselection waiting smoke and launched release-preview smoke `scd_icml_release_preview_smoke_20260607_223858_autogpu_v2` with `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `DIRECTIONS=lite_to_7b`, and `DATASETS=cmrc2018`. It installed the preview package, wrote 2 CMRC prompts, and is safely waiting for a second GPU with about 18GB free.

## 2026-06-07 - Plan change: add release smoke status checker

- Scope change: add a lightweight status tool for queued/completed release smoke runs.
- Reason: release-preview smoke can legitimately wait for shared GPU resources; a one-command status checker prevents confusing a safe wait with a failed or hung run.
- Implementation choice: add `scripts/check_release_smoke_status.py`, include it in the release preview builder and readiness checks, and document it in `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Verification: local compile/readiness and preview rebuild passed; remote preview rebuild passed; running the status checker against `scd_icml_release_preview_smoke_20260607_223858_autogpu_v2` returned `waiting_gpu`, PID running, zero failed stages, no summary yet, and log-tail evidence of the 18GB GPU gate wait.

## 2026-06-07 - Plan change: support single-GPU release smoke without weakening full reproduction

- Scope change: add a release-preview smoke path that can run on one idle 24GB GPU while preserving the default two-GPU full Mistral reproduction path.
- Reason: `ser` had only GPU3 fully idle; the original smoke safely waited for a second 18GB-free GPU, but this made package-level validation dependent on shared-server availability.
- Implementation choice: keep the full reproduction default at two logical CUDA devices, but make `scripts/run_ser_scd_icml_mistral_repro.sh` infer the required GPU count from the configured CUDA devices. Added `configs/models_ser_mistrallite_to_mistral7_mql_single_gpu.json` and documented the single-GPU smoke command in `REPRODUCING_SCD_ICML.md`.
- Runtime robustness choice: updated `capture_pair.py` so same-device capture runs producer capture first, moves captured tensors to CPU, releases the producer model, then loads the consumer and writes paired chunks. Updated `eval_text_strict.py` so same-device strict eval stages producer prefills on CPU and then loads the consumer. Updated `TransformersRuntimeSession.close()` and `TransformersCaptureSession.close()` to release layer/tokenizer references as well as model weights.
- Release hygiene: included `configs/data_ser.yaml`, the single-GPU model config, and the touched runtime modules in release readiness/build checks. `check_release_readiness.py` now compiles `capture_pair.py`, `eval_text_strict.py`, `strict_runtime.py`, and `adapters.py`.
- Verification: local source readiness and release preview readiness passed. Remote source readiness and release preview readiness passed. Remote preview readiness with the verified 200-sample run root also passed, validating the 18-row `lite_to_7b` aggregate table.
- Smoke result: remote release-preview single-GPU smoke completed under `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_release_preview_smoke_single_gpu_20260607_230105` with 6 successful stages and aggregate summary files. The smoke uses `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `DATASETS=cmrc2018`, `MAX_LENGTH=256`, and `MAX_NEW_TOKENS=8`; it is a wiring check only, not claim evidence.

## 2026-06-08 - Plan correction: remove fallback/recompute from claim-facing SCD

- Scope change: the release-facing SCD implementation must not use fallback layers or recomputation inside `scd_full`.
- Reason: the previous `scd_full` route generated from patch metadata put non-patch layers into `fallback_layers`, so its strong 200-sample result was a fallback-hybrid diagnostic rather than paper-faithful SCD evidence.
- Implementation choice: `scripts/make_patch_meta_route.py` now generates routes with `fallback_layers=[]`, `selected_recompute_layers=[]`, `reuse_layers=<all non-patch layers>`, and `patch_layers=<selected patch layers>`.
- Runtime guard: `strict_runtime.py` now rejects `scd_full` routes containing fallback layers; all-layer patch-only is allowed when `patch_layers` covers every layer and `reuse_layers=[]`.
- Training correction: the all-layer shared residual patch config now sets `force_residual_layers_as_selected=true`, and the shared patch trainer honors that flag so the `alllayers` config actually trains patch layers 0..31 rather than only profile-recommended layers.
- Release gate: `check_release_readiness.py` now requires the no-fallback/no-recompute eval config, checks route semantics with a temporary patch-meta route, compiles the shared patch trainer module, and expects 15 claim-facing rows instead of the older 18-row table with selective recompute.
- Verification: local and remote source readiness checks passed. A real-model single-GPU smoke at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_allpatch_smoke_nofallback_20260608_0010` completed with route `patch_layers=0..31`, `reuse_layers=[]`, `fallback_layers=[]`, and 5 aggregate rows.
- Full-run status: the true no-fallback/all-patch 200-sample Mistral run is queued at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_allpatch_repro_20260608_0020` under PID `1136660`, waiting safely for a second GPU with at least 18GB free.

## 2026-06-08 - Plan refinement: keep release preview strictly no-fallback

- Scope refinement: the default open-source release preview should not include the older `eval_scd_icml_main_methods.json` diagnostic config, because it enables selective recompute and can be mistaken for the claim-facing path.
- Implementation choice: removed `configs/eval_scd_icml_main_methods.json` from `scripts/build_release_tree.py` and `scripts/check_release_readiness.py` required release files. The file remains in the working tree for traceability but is now listed only under `Not Included` in `RELEASE_BUILD_REPORT.md`.
- Runtime hygiene: set `runtime.cpu_fallback=false` in the Mistral release-facing eval, reuse-train, and all-layer patch-train configs, and extended readiness checks to reject CPU fallback in those configs. This avoids silently running a CPU fallback path when GPU requirements are not met.
- Verification: local source readiness and release-preview readiness passed. Remote source readiness and remote release-preview readiness also passed after syncing the changed configs/scripts/docs to `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.

## 2026-06-08 - Plan refinement: make patch-only explicit in the claim table

- Scope refinement: the release-facing Mistral summary should include an explicit `patch_only` row instead of requiring readers to infer patch-only quality from an all-layer `scd_full` route.
- Reason: the GitHub reference line validated strict `reuse-only` and `patch-only`; the open-source package should expose the same anchor directly while still supporting route-composed no-fallback `scd_full`.
- Implementation choice: set `run_patch_only=true` in `configs/eval_scd_icml_no_fallback_no_recompute.json`; update `check_release_readiness.py` to expect 18 rows, require `patch_only`, and reject run roots whose route is not all-layer Patch when the patch-only row is part of the release table.
- Operational note: the already queued 200-sample run at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_true_scd_allpatch_repro_20260608_0020` had already generated its temporary eval config, so that run-root config was updated in place to `run_patch_only=true` before model stages started.
- Verification: local source readiness and release-preview readiness passed; remote source readiness and release-preview rebuild/readiness passed after syncing the changed files.

## 2026-06-08 - Plan refinement: support true mixed no-fallback routes

- Scope refinement: keep the all-layer `patch_only` anchor as the default quality check, but expose a real mixed SCD route for the paper semantics: Patch selected layers and decode every other layer by Reuse.
- Implementation choice: `scripts/make_patch_meta_route.py` now accepts `--patch-layers` with comma-separated ids and ranges such as `10-13,15-18`. Without this option it keeps the all-layer default. The wrapper exposes the same control as `ROUTE_PATCH_LAYERS` and disables the all-layer `patch_only` eval row when a subset mixed route is requested.
- Validation utility: added `scripts/validate_scd_route.py` so any generated route can be checked directly for empty `fallback_layers`, empty recompute fields, disjoint patch/reuse layers, and full layer coverage.
- Documentation: `README.md`, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` now distinguish all-layer patch-only anchor from true mixed no-fallback route generation.
- Verification: local readiness and local release-preview rebuild passed. A static route smoke with a 4-layer fake model and `--patch-layers 1,3` produced `reuse_layers=[0,2]`, `patch_layers=[1,3]`, `fallback_layers=[]`, and no recompute map. `validate_scd_route.py` accepted the good mixed route and rejected a bad route containing `fallback_layers=[0]`. Remote `bash -n` syntax check on the local wrapper copy passed.
- Remote sync: stopped the old waiting-only PID `1136660` before any model stage started, synced the new wrapper and route tools to the remote source tree, and relaunched the same run root under PID `1181263`. Remote source readiness and remote release-preview readiness now pass with the mixed-route capable wrapper.
- Current run status: the relaunched 200-sample run is still waiting at the GPU gate. Its `run_config.json` records `route_patch_layers=""`, so this particular run remains the all-layer patch-only anchor; its eval config enables `run_patch_only=true` and `run_scd_full=true` with fallback/recompute disabled.

## 2026-06-08 - Plan correction: expose GitHub strict-stop hybrid search

- Scope correction: the canonical mixed-route path should be the GitHub strict-stop search, not a manually specified `ROUTE_PATCH_LAYERS` layer set.
- Reason: the GitHub reference treats `hybrid` as valid only after separate strict validation: patch-only anchor plus greedy reuse search under a replay F1 gate, with `fallback_layers=[]` and no recomputation.
- Implementation choice: add `scripts/run_ser_scd_icml_mistral_strict_hybrid_search.sh`. It requires `BASE_RUN_ROOT`, reuses the completed CMRC2018 reuse and patch checkpoints, runs `search-blockwise-hybrid`, and validates `selected_hybrid_route.json` with `scripts/validate_scd_route.py`.
- Documentation choice: keep `ROUTE_PATCH_LAYERS` as a manual diagnostic/ablation switch only. Any release-facing mixed-route claim must come from the strict-stop search wrapper and its `search_report.json`.
- Validation gate: `scripts/check_release_readiness.py` now requires the strict hybrid wrapper, the `search-blockwise-hybrid` console entry point, and `scripts/validate_strict_hybrid_search.py`. The new validator checks `search_report.json`, replay F1 gate settings, selected metrics, accepted blocks, and the no-fallback/no-recompute route contract; `--require-mixed` is required before making a mixed-route claim.

## 2026-06-08 - Plan refinement: add explicit open-source license

- Scope refinement: make the release preview a complete open-source package rather than a code snapshot with only a `pyproject.toml` license field.
- Reason: `pyproject.toml` declares MIT, but the runtime release preview did not include a `LICENSE` file.
- Implementation choice: add a standard MIT `LICENSE` file with copyright holder `SCD reproduction team`, include it in `scripts/build_release_tree.py`, and require it in `scripts/check_release_readiness.py`.
- Verification: local source readiness and local release-preview readiness passed; remote source readiness and remote release-preview readiness passed; both previews now include `LICENSE`.

## 2026-06-08 - Plan refinement: import-smoke release entry points

- Scope refinement: ensure the release preview works as a Python package, not only as a copied file tree.
- Reason: a public user should be able to import every console-script entry point declared in `pyproject.toml`; missing modules or undeclared dependencies should be caught before upload.
- Implementation choice: extend `scripts/check_release_readiness.py` with an import smoke that prepends `<root>/src` and imports each module referenced by `[project.scripts]`.
- Verification: local source readiness, local release-preview readiness, remote source readiness, and remote release-preview readiness all passed after adding the import smoke.

## 2026-06-08 - Plan correction: restore paper-facing mixed Reuse + Patch path

- Scope correction: the open-source release target is the SCD-ICML paper method, not the GitHub seed code path. GitHub-style `reuse-only`, `patch-only`, and strict-stop hybrid helpers are treated as historical or diagnostic material unless separately requested.
- Paper basis: the ICML source describes SCD as sparse `Patch` at selected transition layers and `Reuse` on all remaining layers, with `T_fill=0` because Reuse and Patch jointly cover all layers. The default Patch budget is `k=6`.
- Implementation choice: make `scripts/run_ser_scd_icml_mistral_repro.sh` default to `ROUTE_MODE=paper_profile` and `ROUTE_PATCH_BUDGET=6`, use the paper `k=6` Patch config, generate routes with `scripts/make_paper_scd_route.py`, and keep `fallback_layers=[]` plus empty recompute fields.
- Release scope choice: remove `scripts/make_patch_meta_route.py` and GitHub strict-stop search helpers from the default release preview. They remain in the working tree only for traceability/diagnostics, not as the canonical paper reproduction path.
- Entry-point choice: remove the `search-blockwise-hybrid` console script from the default `pyproject.toml` so the public entry points do not advertise the GitHub-style hybrid search as the paper method.
- Eval choice: keep the default claim table to 15 rows: 3 datasets times 5 methods (`full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, `scd_full`). `patch_only` is disabled in the default eval table and remains diagnostic.
- Verification plan: local and remote readiness plus release-preview rebuild must pass after the scope correction; the final quality gate remains a completed 200-sample Mistral run with a mixed `k=6` paper route.
- Current verification: local source readiness and local preview rebuild pass; remote source readiness and remote preview rebuild pass after resync. Remote preview contains `make_paper_scd_route.py` and excludes patch-meta, strict-hybrid, and search-blockwise helpers from the default package surface.
- Remaining runtime gate: remote PID `1248354` is still waiting at the GPU memory gate for `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_paper_mixed_repro_20260608_013851`; no summary exists yet, so quality claims remain unverified.

## 2026-06-08 - Plan refinement: make release status checker paper-route aware

- Scope refinement: extend `scripts/check_release_smoke_status.py` from smoke-only wording to release smoke or reproduction runs, and map `route_mode=paper_profile` to the actual wrapper stage `route_paper_scd`.
- Reason: the 200-sample paper-mixed run is still queued at the GPU gate; a correct status checker prevents confusing a legitimate wait with a missing route-template stage after the run starts.
- Verification: local source readiness passed, local release preview rebuilt and passed, remote source readiness passed, remote release preview rebuilt and passed. Running the status checker on PID `1248354` returned `state=waiting_gpu`, `pid_running=true`, expected stage `lite_to_7b_route_paper_scd`, and no summary files yet.

## 2026-06-08 - Plan record: add paper release audit

- Scope: add `docs/scd-icml-paper-release-audit-2026-06-08.md` as a concise requirement-by-requirement audit for the current paper-facing open-source package.
- Reason: code/package readiness is now separate from the still-pending 200-sample metric gate; the audit makes this distinction explicit for later cleanup and code organization.
- Content: records objective boundary, source/preview evidence, release surface, eval contract, route contract, remote runtime evidence, and the remaining `check_release_readiness.py --run-root` gate.

## 2026-06-08 - Plan correction: k=6 mixed is insufficient without fallback

- Scope correction: the strict paper-style route must remain no-fallback and no-recompute, but the default `k=6` sparse Patch budget is not enough for the current Mistral-Lite -> Mistral-7B artifacts when all non-patch layers are served by Reuse.
- Evidence: the 200-sample fast run `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_fast_mixed_oldpatch_20260608_131847` produced valid mixed routes with `fallback_n=0` and `recompute_n=0`, but `scd_full` quality was close to reuse-only on CMRC2018, HotpotQA, and WikiText-2.
- Root cause: the previous strong 200-sample `scd_full` result relied on fallback layers. The available patch checkpoint also only contained 8 trained patch layers, because `train_patch.py` stopped at `recommended_patch_layers` instead of filling to the configured patch budget.
- Implementation correction: update `train_patch.py` so high patch budgets fill remaining layers by profile `expected_gain` after exhausting recommended layers.
- New reproduction direction: train a high-coverage patch checkpoint and evaluate a mixed route with layer 0 served by Reuse and layers 1..31 served by Patch. This remains no-fallback/no-recompute and is the closest valid mixed analogue to the known patch-strong behavior.
- Current remote run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_mixed_patch31_reuse0_train_20260608_143847`, using GPU acceleration on `cuda:0`.

## 2026-06-08 - Plan refinement: prioritize mergeable parallel patch shards

- Scope refinement: keep the shared residual Patch run on GPU0 as a backup, but prioritize a non-shared residual MLP Patch path because its layer-local checkpoint keys can be trained in shards and merged safely.
- Reason: the shared residual checkpoint contains global trunk parameters and cannot be safely combined from independent layer subsets. The non-shared `xnorm_nativekv_residual_mlp_v1` checkpoint uses per-layer keys, so independent shards can be merged before route generation.
- Implementation choice: add a parallel Patch train config and `scripts/merge_patch_ckpts.py`, include both in release/readiness checks, and launch two shards using the freed cards.
- Current remote run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_2gpu_20260608_152928`.
- Shard plan: GPU1 trains layers `1..16`; GPU2 trains layers `17..31`; final evaluation route remains `reuse_layers=[0]`, `patch_layers=1..31`, `fallback_layers=[]`, and no recomputation.
- Next gate: wait for both shard checkpoints, merge them, generate a validated paper-SCD route, then run the 200-sample evaluation table.

## 2026-06-08 - Plan refinement: switch to 4-GPU shard training

- Scope refinement: use all four available GPUs for the mergeable non-shared Patch path instead of waiting for two large shards and a non-mergeable shared-trunk backup.
- Reason: the 2-GPU shard run was correct but still hours-scale because each shard contained 15-16 layers and the current implementation performs per-layer 200-chunk scans before residual training.
- Implementation choice: add `scripts/run_ser_scd_icml_parallel_patch_shards.sh` as the public helper for the four-shard launch/merge workflow, include it in release readiness and preview build, and stop only the old processes tied to the superseded 2-GPU and shared-backup run roots.
- Current remote run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_parallel_nonshared_patch31_4gpu_20260608_161331`.
- Shard plan: GPU0 trains `1..8`, GPU1 trains `9..16`, GPU2 trains `17..24`, GPU3 trains `25..31`.
- Final route remains unchanged: `reuse_layers=[0]`, `patch_layers=1..31`, `fallback_layers=[]`, and no recomputation.
- Next gate: wait for all four shard checkpoints, run `scripts/merge_patch_ckpts.py`, generate/validate the final route, then evaluate `full_consumer`, `raw_kv`, `quantized_4bit`, `reuse_only`, and `scd_full` on the 200-sample table.

## 2026-06-08 - Plan correction: quality-gated sparse SCD, not patch31 final

- Scope correction: the final reproduction should not claim `reuse_layers=[0], patch_layers=1..31` as the target SCD route. That route is useful only for training a high-coverage Patch candidate pool and measuring an upper bound.
- Paper alignment: SCD uses Patch on selected transition layers and Reuse on all remaining layers; the selection objective is to preserve quality within a tolerance while minimizing transfer/runtime cost.
- Implementation approach: keep the running 4-GPU shard training because it supplies per-layer Patch checkpoints, then generate candidate routes at multiple Patch budgets and evaluate them under the same no-fallback/no-recompute strict runtime.
- Verification gate: final claim requires a completed candidate run whose route validates as mixed, has no fallback/recompute fields, and is the smallest passing Patch set under the selected quality threshold.

## 2026-06-08 - Plan refinement: early sparse candidates and linear-only fallback

- Scope refinement: do not wait for the full `1..31` Patch pool when smaller paper-style sparse candidates may pass the quality gate earlier.
- Early route plan: as soon as `shard_09_16` and `shard_17_24` finish, merge them into a middle-layer Patch pool and evaluate `k=8` and `k=11` on CMRC2018 50-sample probes.
- Selection plan: use `scripts/select_quality_gated_scd_route.py` to choose the smallest Patch set satisfying the configured F1-retention threshold.
- Backup implementation: add a linear-only xnorm-native-KV Patch config that saves direct Patch weights without residual MLP refinement. This is a speed-oriented diagnostic fallback if the residual shard path is too slow or fails, not the default final result.
- Operational constraint: do not launch the linear-only fallback while the current four residual shard processes are CPU/I/O bound and alive, because the host load is already high and another training run would likely slow the active run.

## 2026-06-08 - Plan update: launch corrected 2-GPU linear top11 backup

- Scope update: after two GPUs became available, launch a low-priority linear-only Patch backup for the profile top11 layers while leaving the main residual candidate-pool run alive.
- Code correction: moved `patch.xnorm_nativekv_linear_only` handling out of `train_direct_kv_knorm_residual_patch` and into `train_xnorm_nativekv_residual_patch`, so the linear-only config now matches its intended `fit_mode=xnorm_nativekv_residual_mlp_v1` path.
- Verification: local `py_compile`, local release readiness, remote `py_compile`, remote function-block assertions, remote config assertion, and remote release readiness all passed.
- New backup run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_linear_top11_2gpu_20260608_175651`.
- Shard plan: GPU3 trains layers `10..15`; GPU0 trains layers `16..20`; both use `configs/mistral_self_patch_xnorm_nativekv_linear_parallel_20260404.json` and `nice -n 10` with low thread counts.
- Postprocess plan: background PID `750655` waits for both shard `meta.json` files, merges the linear Patch pool, generates k=`5,6,8,10,11` paper-SCD candidate routes, validates them, and runs CMRC2018 50-sample quality-gate probes.

## 2026-06-08 - Plan correction: abandon linear-only and focus residual top11

- Evidence: the linear-only top11 checkpoint completed, but CMRC2018 50-sample probes showed it did not recover quality. `k=5` retained about 2.1% of full-consumer F1 and `k=11` retained about 1.9%, both essentially reuse-only behavior.
- Decision: stop remaining linear-only candidate evals because smaller linear budgets cannot pass if the largest trained top11 budget fails.
- Decision: stop the broad patch31 residual run because it had no shard `meta.json` after more than two hours and was consuming very high CPU while training layers outside the likely sparse route.
- New plan: launch a focused four-GPU residual-MMLP top11 run covering only layers `10..20`, then evaluate paper-SCD sparse candidates k=`5,6,8,10,11`.
- New run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_4gpu_20260608_183110`.
- Shard plan: GPU0 `10..12`, GPU1 `13..15`, GPU2 `16..18`, GPU3 `19..20`.

## 2026-06-08 - Plan refinement: skip diagnostic linear-base error scans

- Scope refinement: avoid the expensive per-layer direct-linear error-statistics pass before residual training.
- Reason: the pass only writes `linear_base_*` diagnostic metrics; residual training computes its own `base_* -> trained` metrics and does not depend on that pre-scan.
- Implementation: add `patch.skip_linear_base_error_stats=true` support to `train_xnorm_nativekv_residual_patch` and enable it in the residual parallel config.
- Operational update: stopped the first focused residual top11 run before it wrote any checkpoint and relaunched with the faster config.
- New run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_skiplin_4gpu_20260608_185457`.

## 2026-06-08 - Plan correction: residual checkpoint best-state criterion

- Evidence: the skip-linear residual top11 run completed and all paper-SCD routes were valid mixed Reuse+Patch routes with no fallback or recompute, but CMRC2018 50-sample probes retained only about 2.2% to 4.1% of full-consumer F1.
- Root cause: residual training initialized `XNormNativeKVResidualPatchLayer` as identity and selected `best_state` by total loss including `lambda_delta * delta_reg`. The one-epoch trained states slightly improved h/k/v reconstruction, but the delta penalty made total loss worse than the identity initialization, so checkpoints kept the identity state.
- Implementation correction: select residual `best_state` by reconstruction quality loss (`h/k/v`, or `k/v` for direct residual) and keep delta regularization only in the optimization objective. Record `quality_loss` in metrics.
- Config correction: increase residual epochs from `1` to `3` and reduce `lambda_delta` from `1e-4` to `1e-5`.
- Operational update: launch a corrected four-GPU top11 run after the failed control evals completed.
- New run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_4gpu_20260608_201408`.
- Next gate: wait for shard checkpoints, merge, evaluate k=`5,6,8,10,11`, then choose the smallest route passing a quality-retention threshold if any.

## 2026-06-08 - Plan correction: teacher-aligned conditional route selection

- Scope correction: the main Patch/Reuse route selector is no longer fixed `k=6`, fixed top-k, or restore-one-from-all-Reuse. Those routes remain baselines and diagnostic candidates only.
- Reason: all-Reuse is an off-manifold starting point for the current MistralLite -> Mistral-7B transfer. Single-layer restore gains measured from that state do not reliably capture downstream distribution shift or pairwise layer interactions.
- New method: formulate route selection as teacher-aligned conditional search against the full consumer. Use route-level KV drift, hidden drift, logit KL, PPL delta, and cost; start from a high-quality/all-Patch route and prune Patch layers into Reuse only when the conditional marginal loss stays within configured budgets.
- Implementation choice: add `src/scd_repro/route_selection/` with route schema, source-target layer mapping, teacher metrics, conditional pruning, pairwise interaction utilities, and a proxy beam selector. Add script entry points `estimate_layer_mapping.py`, `profile_routes.py`, `select_patch_reuse_route.py`, and `evaluate_selected_route.py`.
- Engineering boundary: the new selector never fabricates missing teacher metrics. If a trial route has not been evaluated, it writes pending route JSON artifacts for the existing strict eval wrapper.
- Documentation update: README, `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md` now describe fixed top-k as baseline/diagnostic and the teacher-aligned conditional selector as the main route-selection path.
- Verification: local py-compile passed for the new modules/scripts; local `python -m unittest discover -s Experiment/core_code/runtime_repo/tests -v` passed 10 tests; local release readiness passed. The new files were synced to remote `ser`, where py-compile, the same 10 tests, release readiness, and `estimate_layer_mapping.py` smoke all passed.

## 2026-06-08 - Plan update: corrected residual top11 run with GPU preload and thread caps

- Superseded run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_4gpu_20260608_201408` was stopped because residual dataset projection still ran on CPU and each process consumed excessive CPU.
- Code correction: residual preload now projects native KV targets on the training device rather than forcing CPU projection.
- Operational correction: `scripts/run_ser_scd_icml_parallel_patch_shards.sh` now caps `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` at `2` by default.
- Active run: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_residual_top11_qualitybest_e3_d1e5_gpupreload_threads2_4gpu_20260608_211359`.
- Evidence: all four shards reached `xnorm native-kv stats progress: chunk 200/200`, skipped linear-base diagnostics, and completed their first residual layer with real reconstruction improvement. Examples: layer 10 `h/k/v` rel-MSE `0.194485/0.068328/0.220374 -> 0.184139/0.056349/0.177670`; layer 13 `0.249286/0.091023/0.321349 -> 0.237137/0.076298/0.260241`.
- Next gate: wait for remaining layers, shard `meta.json`, postprocess merge, and CMRC2018 candidate evals. Use the results as diagnostic route metrics and, where available, as inputs to teacher-aligned conditional selection rather than as a fixed-k final claim.

## 2026-06-09 - Plan update: all-patch upper bound and streaming eval fix

- Evidence: corrected residual top11 Reuse+Patch candidates k=`5,6,8,10,11` all remained near reuse-only quality on CMRC2018 50-sample probes; best F1 retention was only about `4.69%`.
- Diagnostic decision: train and evaluate an all-layer Patch-only upper bound before continuing sparse mixed route search, to separate Patch capacity from route interaction failure.
- Runtime correction: strict eval now supports CPU-loaded patch checkpoints and runtime patch window overrides through `runtime_eval.patch_ckpt_device`, `runtime_eval.runtime_patch_window_size`, `runtime_eval.runtime_patch_persist_shards`, and `runtime_eval.runtime_patch_persist_modules`.
- Successful diagnostic config: `patch_ckpt_device=cpu`, `runtime_patch_window_size=1`, `runtime_patch_persist_shards=false`, and `runtime_patch_persist_modules=false`; this prevents the earlier all-layer patch prewarm OOM on 24GB GPUs.
- Result: all-layer Patch-only on CMRC2018 50 samples reached F1 `0.788977574686301`, EM `0.52`, avg NLL `0.6808870162091324`, exceeding the full-consumer probe F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`.
- Boundary: this is a quality upper-bound diagnostic, not a serving result, because streaming 32 residual patch modules from CPU makes patch inference about `27.64` seconds per sample.
- Next plan: use the all-patch result as evidence that Patch capacity exists, then resume teacher-aligned conditional Patch/Reuse search. The route search should prune layers from all-Patch toward Reuse only when route-level teacher metrics and quality gates stay within budget.

## 2026-06-09 - Plan update: teacher-aligned route selector plus prefix-screen gate

- Scope update: use the pasted route-selection direction as the project mainline. Fixed `k=6`, fixed top-k, restore-one-from-all-Reuse, and all-Reuse sensitivity remain baselines/diagnostics only.
- Engineering update: the runtime release now includes a teacher-aligned route-selection package and CLIs: route schema with per-layer `action/source_layer`, source-target mapping fallbacks, KV/hidden/KL/PPL objective helpers, conditional Reuse pruning, pairwise interaction utilities, and strict-eval proxy gates.
- Metric boundary: do not fabricate unavailable KV/hidden teacher drift. Until those profilers are fully connected, strict eval outputs may be used as real proxy gates through `f1_retention`, `avg_nll_delta`, and TTFT.
- Verification: local and remote unittests now cover 12 route-selection cases; local and remote release readiness pass.
- Experiment update: CMRC2018 10-sample prefix screen over Reuse counts `4,8,12,16,20,24,28,31` selected Reuse 8 / Patch 24 under gate F1 retention `>=0.95` and avg NLL delta `<=0.05`.
- Operational next step: run CMRC2018 50-sample verification for the selected Reuse 8 / Patch 24 route and the conservative Reuse 4 / Patch 28 route in parallel on four GPUs. If Reuse 8 remains within gate, promote it as the current SCD-ICML mixed-route candidate; if it fails but Reuse 4 passes, keep Reuse 4 as the conservative route and continue pairwise/conditional refinement around layers added between 4 and 8.

## 2026-06-09 - Plan correction: CMRC50 rejects reuse04/reuse08 under NLL gate

- Evidence: CMRC2018 50-sample validation showed Reuse 8 / Patch 24 retained F1 above full-consumer (`1.039x`) but worsened avg NLL by `0.738`; Reuse 4 / Patch 28 retained F1 above full-consumer (`1.162x`) but worsened avg NLL by `0.584`.
- Decision: do not promote Reuse 8 / Patch 24 or Reuse 4 / Patch 28 as final teacher-aligned routes under the current objective. F1-only selection is too weak for this project because the likelihood drift is large.
- New route-search boundary: evaluate lower Reuse counts `1,2,3` on CMRC2018 50 samples. If these also fail the NLL gate, the next correction should be pairwise/single-layer route analysis around the safest individual Reuse layers, not larger prefix routes.
- Operational update: launched `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_conditional_lowreuse_cmrc50_20260609_0418` with prefix budgets `1,2,3` and the same no-fallback/no-recompute strict runtime.

## 2026-06-09 - Plan update: current quality-safe route is Reuse 2 / Patch 30

- Evidence: low-Reuse CMRC2018 50-sample screen selected Reuse 2 / Patch 30 under F1 retention `>=0.95` and avg NLL delta `<=0.05`.
- Selected route: Reuse layers `[8,25]`; all other 30 layers Patch.
- Metrics: SCD F1 `0.8130814875493471` vs full F1 `0.618514853953628`, F1 retention `1.3145706725586679`; SCD avg NLL `0.658217633134098` vs full avg NLL `0.7055898962204568`, avg NLL delta `-0.04737226308635878`.
- Rejection boundary: Reuse 3 / Patch 29, adding layer `1`, fails the NLL gate with avg NLL delta about `0.546`. Reuse 4 and Reuse 8 also failed CMRC50 NLL gates despite high F1.
- Current conclusion: the route selector should not use fixed prefix length or F1-only retention. Under the current strict eval proxy objective, only layers `8` and `25` are safe Reuse decisions.
- Next implementation plan: add pairwise/conditional interaction diagnostics around the first few candidate layers and investigate runtime optimization for high-Patch routes. The current selected route is quality-safe but not latency-ready because CPU-window Patch streaming still costs about `25.6` seconds TTFT.

## 2026-06-09 - Plan update: pairwise evidence supports `[8,25]`

- Evidence: explicit CMRC50 routes for `[1]`, `[25]`, `[1,8]`, and `[1,25]` completed, combined with existing `[8]`, `[8,25]`, and `[1,8,25]` results.
- Pairwise report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_pairwise_1_8_25_cmrc50_20260609_0525/lite_to_7b/conditional_report/pairwise_interactions_avg_nll.json`.
- Result: pair `(8,25)` has near-zero avg-NLL interaction (`+0.000906`) and remains quality-safe. Pair `(1,8)` has positive interaction (`+0.205160`), and layer `1` alone already fails the avg-NLL gate (`+0.331580`).
- Route-selection conclusion: the current quality-safe route `[8,25]` should remain the selected Reuse set under the current CMRC50 proxy objective. Layer `1` should be marked conflict/unsafe unless a later patch/reuse implementation fixes its drift.
- Next plan: because the selected route is Patch-heavy and still slow, run runtime window-size sweeps for Reuse `[8,25]` / Patch 30 to find the fastest no-fallback/no-recompute configuration that fits the 24GB GPUs.

## 2026-06-09 - Plan update: default Patch runtime window set to 8

- Evidence: CMRC2018 5-sample runtime sweep for the selected Reuse `[8,25]` / Patch 30 route found window 8 as the best successful TTFT setting: window 2 `25500` ms, window 4 `22628` ms, window 8 `21856` ms, window 16 `21911` ms. Window 32 OOMed on GPU 1.
- Implementation decision: use `PATCH_CKPT_DEVICE=cpu`, `RUNTIME_PATCH_WINDOW_SIZE=8`, and non-persistent Patch shards/modules as the default in the strict eval wrappers. This is faster than window 1/2 and avoids the OOM mode seen at window 32.
- Verification: local py-compile, local 12-test unittest, local release readiness, remote bash syntax check, remote py-compile, remote release readiness, and a 1-sample default-window smoke all passed. The smoke confirmed generated runtime config has window size 8.
- Remaining bottleneck: the current quality-safe route is still Patch-heavy, so even the best safe window remains around `21.9` seconds TTFT on the 5-sample smoke. Further speedup needs either safer additional Reuse decisions, a faster Patch execution path, or a more compact Patch module.

## 2026-06-09 - Plan update: selected route promoted to Reuse 8 / Patch 24 without layer 1

- Evidence: conditional expansion from the safe Reuse `[8,25]` route found additional safe Reuse layers when layer `1` was excluded. CMRC20 single-add and combination screens passed for layers `9`, `11`, `13`, `21`, `22`, and `24`.
- CMRC50 validation root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830`.
- New selected route: Reuse layers `[8,9,11,13,21,22,24,25]`; Patch layers `[0,1,2,3,4,5,6,7,10,12,14,15,16,17,18,19,20,23,26,27,28,29,30,31]`.
- Quality gate: F1 retention `>=0.95`, avg NLL delta `<=0.05`, and SCD TTFT `<=18000` ms.
- Selected route metrics on CMRC2018 50 samples:
  - SCD F1 `0.7565720072590736` vs full F1 `0.618514853953628`; retention `1.2232074984504675`.
  - SCD avg NLL `0.6004365290340221` vs full avg NLL `0.7055898962204568`; avg NLL delta `-0.10515336718643464`.
  - SCD TTFT `17121.27815722488` ms; payload `127.93375` MB.
- Selected route artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse_pareto_cmrc50_20260609_0830/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_ttft18s.json`.
- Decision: promote this route as the current SCD-ICML mixed-route candidate. It supersedes the previous Reuse `[8,25]` / Patch 30 candidate for speed and remains within the CMRC50 proxy quality gate.
- Boundary: layer `1` remains unsafe/conflict-marked under current evidence. The earlier failed Reuse 8 route included layer `1`; the new Reuse 8 route avoids it and therefore should not be described as fixed-k recovery.
- Next plan: run pairwise/interaction reporting for the final candidate set if a heatmap/table is needed for the paper, then update release notes and manuscript wording to present fixed-k and restore-one-from-all-Reuse as baselines only.

## 2026-06-09 - Plan update: route screen scheduler changed to work queue

- Reason: the conditional route screen previously launched one route per GPU pair and waited for the whole batch before starting the next batch. When one GPU pair finished earlier, it could sit idle until the slower pair completed.
- Implementation: `scripts/run_ser_scd_icml_conditional_route_screen.sh` now writes candidate route paths to `logs/route_queue.txt` and starts one worker per `GPU_PAIRS` entry. Each worker uses `flock` on `logs/route_queue.lock` to dequeue the next route immediately after finishing its current route.
- Failure handling: route worker failures are recorded in `logs/route_queue.failed`; the main script waits for all workers and exits nonzero if any worker failed.
- Verification: the updated script was synced to remote `ser`; remote `bash -n scripts/run_ser_scd_icml_conditional_route_screen.sh` passed, and `/usr/bin/flock` is available.
- Operational note: already-running launchers keep their old shell code. New route screen launches use the work-queue scheduler automatically.

## 2026-06-09 - Plan update: maximize Reuse count under quality gate

- Objective update: selection now treats runtime speed as proportional to the number of Reuse layers. TTFT is recorded for diagnostics but is not used as a quality gate because the current timing calculation is considered unreliable.
- Gate retained for quality: F1 retention `>=0.95` and avg NLL delta `<=0.05` on CMRC2018 strict eval.
- Work-queue validation: run `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510` showed worker 0 finishing a Reuse17 candidate at `2026-06-09T15:20:00+08:00` and immediately starting the queued Reuse16 candidate on GPUs `0,1`, while worker 1 finished at `15:20:09` and idled because the queue was empty.
- CMRC50 result: both tested Reuse17 / Patch15 routes failed the avg NLL gate (`0.057953` and `0.078611`). The highest passing route is Reuse16 / Patch16 with Reuse layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`.
- Current selected route artifact: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_no_ttft.route.json`.
- Current selected report: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_expand_reuse17_cmrc50_workqueue_20260609_1510/lite_to_7b/conditional_report/selected_route_f1ret095_nll005_no_ttft.json`.

## 2026-06-09 - Plan correction: explicit CMRC train/dev split

- Issue: the previous CMRC loader and training-prompt generator could select `cmrc2018_dev.json` when building files named as train prompts. This creates a data-leakage risk because train-time prompt calibration and eval may use the same examples.
- Code correction: CMRC loading now requires an explicit split (`train` or `dev`) and no longer falls back across train/dev. The train prompt path defaults to `TRAIN_CMRC_SPLIT=train`; strict eval defaults to `EVAL_CMRC_SPLIT=dev` / `CMRC_SPLIT=dev`.
- Metadata correction: generated CMRC train prompt files now write sibling `*.meta.json` files with dataset, split, limit, sample count, and qids, so future runs can verify the split used by a prompt artifact.
- Validation: local and remote unit tests passed, and remote loader smoke on `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854` confirmed separate sample ids: train begins with `TRAIN_186_QUERY_0`, dev begins with `DEV_0_QUERY_0`.
- Result-status change: all route-search and CMRC results produced before this split correction, including Reuse16/Reuse17 CMRC50 and Reuse17 CMRC200, must be treated as diagnostic only and not used as final paper evidence. Clean results require retraining/calibrating Patch prompts from the train split and evaluating only on the dev split.
- Next plan: rerun the selected candidate routes after clean train/dev separation. Until those reruns finish, the current route ordering is useful for debugging but not claim-safe.

## 2026-06-09 - ICML poster draft generated

- Scope: create a first A0 landscape poster draft for the archived SCD-ICML camera-ready work.
- Goal: produce a usable poster artifact quickly from verified local source material, with print-ready PNG/PDF and a PowerPoint wrapper for viewing/sharing.
- Assumptions: the poster should use the archived camera-ready ICML package as the source of paper-facing claims, not the follow-up route-search results that were marked diagnostic after the CMRC train/dev split correction.
- Constraints: no new experimental claims; all poster work stays under `Promotion/slides/poster/`; the PPTX is image-backed for this first draft rather than fully editable text/shapes.
- Design choices: ICML maroon/red theme, A0 landscape, four-column layout: problem/contributions/evidence boundary; SCD framework and Reuse/Patch mechanics; main quality-efficiency and latency/bytes results; Patch policy, error truncation, and TTFT breakdown.
- Outputs: `Promotion/slides/poster/scd_icml_poster_draft.png`, `Promotion/slides/poster/scd_icml_poster_draft.pdf`, `Promotion/slides/poster/scd_icml_poster_draft.pptx`, `Promotion/slides/poster/POSTER_CONTENT_PLAN.md`, `Promotion/slides/poster/POSTER_SPEECH.md`, and `Promotion/slides/poster/build_poster.py`.
- Verification: officecli validation passed with no issues; the PPTX reports one A0 custom-size slide with one embedded poster image and correct alt text; screenshot render displayed the complete poster without clipping.

## 2026-06-09 - Plan update: shared Patch training uses chunk-first streaming

- Reason: Patch16 clean retraining exposed a repeated disk-read bottleneck in `xnorm_nativekv_shared_residual_mlp_v1`: after solving direct `wk/wv`, the old path scanned all pair chunks once per selected layer for linear error stats, then scanned chunks again for shared residual training.
- Implementation decision: update `Experiment/core_code/runtime_repo/src/scd_repro/direct_kv_residual_patch.py` so shared residual Patch training streams chunks first, processes GPU work in `layer_block_size` blocks, and reuses the same loaded per-chunk arrays for direct linear error stats and residual training.
- I/O decision: add optional mmap loading to `Experiment/core_code/runtime_repo/src/scd_repro/io_utils.py`; the shared Patch path defaults to CPU+mmap pair-artifact loading with automatic fallback when mmap is unsupported.
- Scope boundary: direct `wk/wv` stats still require one full chunk pass, but the former 16x per-layer error pass is removed for Patch16 and higher patch counts.
- Verification: local py-compile passed, local synthetic streaming smoke passed, remote py-compile passed, and remote synthetic streaming smoke passed.

## 2026-06-09 - Plan update: clean selected route promoted to Reuse17(add3) / Patch15

- Scope update: after the CMRC train/dev split fix, continue route selection from the clean Reuse16/Patch16 run and maximize Reuse layer count while treating TTFT as diagnostic only.
- Quality gate retained: F1 retention `>=0.95` and avg NLL delta `<=0.05` on CMRC2018 dev strict eval.
- Baseline clean route: Reuse16/Patch16 with Reuse layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]` passed with SCD F1 `0.6687529784411572` and avg NLL delta `0.028654080715454944`.
- Reuse17 scan result: adding layer `3` is the only confirmed passing single-add candidate from the clean Reuse16 baseline.
- Selected route:
  - Reuse layers `[3,5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`.
  - Patch layers `[0,1,2,4,6,7,10,12,15,16,17,18,19,26,31]`.
  - No fallback layers and no selected recompute layers.
- Selected route metrics:
  - Full Consumer F1 `0.618514853953628`, EM `0.36`, avg NLL `0.7055898962204568`.
  - SCD F1 `0.7178651651494944`, EM `0.44`, avg NLL `0.7506309336299498`, payload `107.37296875` MB.
  - F1 retention `1.1606272033093565`, avg NLL delta `0.04504103740949306`.
- Reuse18 boundary: adding any second extra layer on top of Reuse17(add3) failed the current avg NLL gate. The best F1 candidate `[3,6]` had avg NLL delta `0.23395337932182547`; candidate `[3,31]` still had avg NLL delta `0.08388605450741937`.
- Decision: current maximum confirmed clean Reuse count is `17`. Do not continue blindly increasing Reuse layers under the same gate; next useful work is either larger-sample validation for Reuse17(add3), patch/reuse quality improvements, or a deliberate gate/objective change.
- Artifacts:
  - Selected route: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/route/route_cmrc2018_seed42_reuse17_add3_patch15_clean.json`.
  - Selected eval summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/cmrc2018_reuse17_add3_patch15_clean_dev50/summary.json`.
  - Reuse17 scan summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/reuse17_single_add_scdonly_clean_dev50/summary_reuse17_single_add_scdonly.json`.
  - Reuse18 scan summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/strict_eval/reuse18_from_add3_scdonly_clean_dev50/summary_reuse18_from_add3_scdonly.json`.

## 2026-06-09 - Plan update: CMRC200 invalidates Reuse17 and narrowly rejects Reuse16 under strict NLL gate

- Scope update: CMRC50-selected routes were revalidated on CMRC2018 dev `n=200` after the train/dev split fix.
- Gate retained for this validation: F1 retention `>=0.95` and avg NLL delta `<=0.05`; TTFT remains diagnostic only.
- Reuse17(add3)/Patch15 CMRC200:
  - Full Consumer F1 `0.657307250029563`, avg NLL `0.5692842822281626`.
  - SCD F1 `0.6715361993886206`, avg NLL `0.6668955393563621`.
  - F1 retention `1.0216473336608072`, avg NLL delta `0.09761125712819951`.
  - Decision: reject under the strict NLL gate despite acceptable F1.
- Reuse16/Patch16 CMRC200:
  - Full Consumer F1 `0.657307250029563`, avg NLL `0.5692842822281626`.
  - SCD F1 `0.6447946591903111`, avg NLL `0.6225295365171866`.
  - F1 retention `0.980963863035606`, avg NLL delta `0.053245254289023936`.
  - Decision: narrowly reject under the strict NLL gate; it misses by about `0.00325` avg NLL.
- Current route-selection conclusion:
  - CMRC50 alone is not sufficient for final route selection.
  - Under CMRC200 and the strict `0.05` avg-NLL gate, neither the clean Reuse17(add3) route nor the clean Reuse16 route is confirmed as final.
  - The previous CMRC50 selected route should remain a candidate, not a final claim-facing route.
- Implementation decision:
  - The clean Patch16 checkpoint cannot evaluate Reuse15 or adjusted Reuse16 candidates that require Patch on layers currently assigned to Reuse, because it lacks Patch weights for those layers.
  - Launch extra Patch training for current Reuse16 layers `[5,8,9,11,13,14,20,21,22,23,24,25,27,28,29,30]`, using the same `xnorm_nativekv_shared_residual_mlp_v1` fit mode.
  - Merge the extra Patch shards with the existing clean Patch16 checkpoint to create an expanded Patch pool.
  - Then scan Reuse15 and adjusted Reuse16 candidates on CMRC200 using scd-only eval plus the known Full Consumer baseline.
- Active run:
  - Extra Patch run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706`.
  - Shards: GPU 0 `extra_05_14=[5,8,9,11,13,14]`; GPU 2 `extra_20_24=[20,21,22,23,24]`; GPU 3 `extra_25_30=[25,27,28,29,30]`; GPU 1 avoided due to an external process.
- Verification already completed in this step:
  - Local py-compile passed for the touched runtime/eval modules.
  - Local unit tests passed: 15 tests.
  - Local release readiness check passed for `Experiment/core_code/runtime_repo`.

## 2026-06-09 - Plan correction: primary route-selection metric is F1/EM, NLL is auxiliary

- Reason: the project objective is to maximize Reuse layers as the speed proxy while keeping answer quality loss small. For CMRC-style QA, answer quality should be judged primarily by F1 and EM, not by avg NLL.
- Metric decision: select the highest Reuse count whose F1/EM remain close to or better than Full Consumer on the same dev subset. Use avg NLL as an auxiliary risk signal for calibration/instability, not as a hard veto over acceptable F1/EM.
- Consequence for current evidence: Reuse17(add3)/Patch15 on CMRC200 is no longer rejected solely because avg NLL delta is `0.09761125712819951`; its F1 `0.6715361993886206` and EM `0.38` exceed the Full Consumer baseline F1 `0.657307250029563` and EM `0.365`, so it remains a valid candidate under the F1/EM-primary objective.
- Next action: continue scanning Reuse18 and higher candidates on CMRC2018 dev `n=200`, prioritizing candidates using prior CMRC50 F1/EM ordering. Report NLL alongside F1/EM as a risk note.

## 2026-06-10 - Plan update: current F1/EM-primary boundary is Reuse19/Patch13

- Scope update: after switching to F1/EM-primary selection, continue increasing Reuse layers on CMRC2018 dev `n=200` without fallback or recompute.
- Current selected route: Reuse19(add3,add6,add17)/Patch13.
  - Reuse layers `[3,5,6,8,9,11,13,14,17,20,21,22,23,24,25,27,28,29,30]`.
  - Patch layers `[0,1,2,4,7,10,12,15,16,18,19,26,31]`.
- Full+SCD confirmation:
  - Full Consumer: F1 `0.657307250029563`, EM `0.365`, avg NLL `0.5692842822281626`, TTFT `100.85939839249477` ms.
  - SCD: F1 `0.6427511605994372`, EM `0.35`, avg NLL `0.9345715692188692`, TTFT `40.80477777868509` ms, payload `103.4208984375` MB.
  - Retention/delta: F1 retention `0.9778549689974191`, EM retention `0.958904109589041`, EM delta `-0.015000000000000013`, avg NLL delta `0.36528728699070656`, TTFT delta `-60.054620613809675` ms.
- Reuse20 boundary: all 13 one-layer Reuse20 extensions from the selected Reuse19/Patch13 route have now been tested on CMRC200 and did not preserve F1; the best tested candidate had F1 retention `0.9300`, below the working F1/EM boundary.
- Checkpoint update: a full all-32 shared Patch checkpoint was successfully trained and verified at `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_patch_extra_reuse16_layers_20260609_222706/lite_to_7b/ckpt_patch/patch_cmrc2018_seed42_all32_shared_clean`, covering selected/residual layers `[0..31]`.
- Decision: treat Reuse19/Patch13 as the current maximum verified route for CMRC200 under the F1/EM-primary objective. Keep the NLL increase as an explicit risk note for later calibration or paper-facing discussion.

## 2026-06-10 - Plan update: selected Reuse19 route packaged for release

- Scope update: move the verified Reuse19/Patch13 CMRC200 boundary from remote run directories into the release-facing runtime repository so the open-source path has a fixed selected-route artifact.
- Added route artifact:
  - `Experiment/core_code/runtime_repo/configs/routes/scd_icml_mistrallite_to_mistral7_reuse19_patch13_cmrc200.json`.
  - The route is sanitized for release and does not rely on remote absolute-path metadata.
  - It validates as a mixed no-fallback/no-recompute route with Reuse19/Patch13.
- Added eval config:
  - `Experiment/core_code/runtime_repo/configs/eval_scd_icml_full_scd_no_fallback_no_recompute.json`.
  - This config runs only `full_consumer` and `scd_full`, with all fallback, recompute, raw-KV, 4-bit, reuse-only, and patch-only side methods disabled.
- Documentation updates:
  - `README.md` now names the current Reuse19/Patch13 boundary and the CMRC200 F1/EM retention evidence.
  - `REPRODUCING_SCD_ICML.md` now includes the selected route, the Full+SCD-only rerun command, and the caution that avg NLL is an auxiliary calibration-risk signal.
  - `OPEN_SOURCE_RELEASE_MANIFEST.md` now includes the selected route and the new Full+SCD eval config.
- Release tooling update:
  - `scripts/check_release_readiness.py` now requires and validates the selected route and Full+SCD-only eval config.
  - `scripts/build_release_tree.py` now includes both new files in the release preview.
- Verification:
  - JSON parsing passed for the new route/config.
  - `scripts/validate_scd_route.py --require-mixed` passed on the selected route.
  - Unit tests passed: 15 tests.
  - `scripts/check_release_readiness.py --root Experiment/core_code/runtime_repo` passed.
  - `scripts/build_release_tree.py --force` rebuilt `release_preview/scd_icml_runtime_release`, and the preview's own readiness check passed.
- Remaining risk:
  - The selected route is CMRC200-verified; HotpotQA/WikiText-2 claim-facing aggregate evidence still needs a fresh no-fallback/no-recompute run before broad table claims.
  - Avg NLL is substantially worse than Full Consumer for the selected route and should remain an explicit risk note.

## 2026-06-10 - Plan correction: NLL is advisory for QA route selection

- User clarification: the current objective is to find the largest Reuse layer count while keeping F1/EM loss small. Time is currently approximated by Reuse layer count.
- Selection rule: for QA datasets such as CMRC2018 and HotpotQA, F1/EM are the primary quality metrics. Avg NLL must be reported as an auxiliary calibration-risk signal, but it must not override acceptable F1/EM unless an experiment is explicitly labeled as an NLL-gated diagnostic.
- Code-policy update: `scripts/select_quality_gated_scd_route.py` should default to maximum-Reuse selection under the primary answer-quality retention gate; `--max-avg-nll-delta` is advisory unless paired with `--enforce-avg-nll-delta`.
- Consequence: the current CMRC200 Reuse19/Patch13 boundary remains the selected route because F1 retention is `0.9778549689974191` and EM retention is `0.958904109589041`. The NLL increase remains a risk note, not a rejection reason.
- Experiment priority: do not launch further WikiText-only route expansion as the main search. Any active or completed WikiText scan should be recorded as auxiliary NLL/PPL evidence only.

## 2026-06-10 - Plan update: selected-route aggregate evidence completed

- Scope update: the release-facing selected Reuse19/Patch13 route now has a fresh no-fallback/no-recompute 5-method x 3-dataset aggregate run.
- Aggregate run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426`.
- Verification: remote `python scripts/check_release_readiness.py --root . --run-root /disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_5method_aggregate_20260610_034426` passed.
- Interpretation:
  - CMRC2018 supports the selected route under the F1/EM-primary objective.
  - HotpotQA is only a sanity check because absolute F1/EM are very low for both Full Consumer and SCD.
  - WikiText-2 shows the selected CMRC route is not a universal language-modeling route; keep it as NLL/PPL risk evidence, not as a QA route-selection blocker.

## 2026-06-10 - Plan update: Reuse20-to-Reuse32 degradation curve completed

- Scope update: completed the requested quality degradation curve after the selected Reuse19/Patch13 boundary.
- Run root: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604`.
- Output summary: `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_reuse20_to_32_quality_curve_3datasets_20260610_100604/lite_to_7b/conditional_report/reuse20_to_32_quality_curve_with_reuse19_baseline.md`.
- Route sequence: start from Reuse19/Patch13, add layers in CMRC Reuse20 F1-ranked order `[26,10,31,4,7,2,19,12,1,16,0,15,18]`, evaluate Reuse20 through Reuse31 as mixed routes, and use verified `reuse_only` rows as the Reuse32 endpoint.
- Decision impact: Reuse20 already drops to CMRC F1 retention `0.929953`, below the `0.95` target. Reuse21 and above degrade rapidly, so Reuse19/Patch13 remains the current maximum verified route under the F1/EM-primary objective.

## 2026-06-10 - ICML poster v2 changed to A0 landscape visual-first draft

- Scope update: respond to the poster design correction that the poster should be horizontal and less dominated by red.
- Design choice:
  - Keep A0 landscape size at `118.9cm x 84.1cm`.
  - Replace the heavy ICML red treatment with an off-white background, charcoal thesis band, blue/teal technical accents, and muted maroon only for secondary emphasis.
  - Use a visual-first layout: top title, one central thesis, left rail for motivation/contribution/boundary, center method diagram, right rail for evidence, and bottom stat strip.
- Claim boundary:
  - The v2 poster still uses archived camera-ready SCD-ICML paper claims and source figures.
  - Later route-search and release diagnostics are not promoted into the poster unless explicitly requested.
- Outputs:
  - `Promotion/slides/poster/build_poster_v2_landscape.py`
  - `Promotion/slides/poster/scd_icml_poster_v2_landscape.png`
  - `Promotion/slides/poster/scd_icml_poster_v2_landscape.pdf`
  - `Promotion/slides/poster/scd_icml_poster_v2_landscape.pptx`
  - `Promotion/slides/poster/POSTER_CONTENT_PLAN_v2.md`
- Verification:
  - Rendered the PNG/PDF with `python Promotion\slides\poster\build_poster_v2_landscape.py`.
  - Created a one-slide A0 landscape PPTX wrapper with the poster image placed at `0cm,0cm` and sized to `118.9cm x 84.1cm`.
  - `officecli validate` passed with no errors.
  - `officecli view --mode stats` reported 1 slide, 1 picture, and no missing picture alt text.
  - `officecli view --mode screenshot` rendered the full slide without visible cropping.

## 2026-06-14 - ICML poster v3 understated horizontal template

- Scope update: respond to the requested direction of fewer results, less attention-grabbing visual treatment, and a low-key template.
- Design choice:
  - Keep A0 landscape size at `118.9cm x 84.1cm`.
  - Use a quiet academic style: warm white background, thin gray rules, one muted blue accent, and muted sage/clay secondary accents.
  - Remove the v2 dark thesis band, large metric cards, bottom stat strip, and multi-chart result rail.
  - Center the poster on method explanation: question/design/boundary on the left, method overview and mechanism detail in the center, one source figure and one small evidence table on the right.
- Claim boundary:
  - The v3 poster intentionally minimizes result emphasis.
  - It still uses archived camera-ready SCD-ICML paper claims and source figures only.
  - Follow-up route-search and release diagnostics are not promoted into this poster.
- Outputs:
  - `Promotion/slides/poster/build_poster_v3_understated_landscape.py`
  - `Promotion/slides/poster/scd_icml_poster_v3_understated_landscape.png`
  - `Promotion/slides/poster/scd_icml_poster_v3_understated_landscape.pdf`
  - `Promotion/slides/poster/scd_icml_poster_v3_understated_landscape.pptx`
  - `Promotion/slides/poster/POSTER_CONTENT_PLAN_v3.md`
- Verification:
  - Rendered the PNG/PDF with `python Promotion\slides\poster\build_poster_v3_understated_landscape.py`.
  - Created a one-slide A0 landscape PPTX wrapper with the poster image placed at `0cm,0cm` and sized to `118.9cm x 84.1cm`.
  - `officecli validate` passed with no errors.
  - `officecli view --mode stats` reported 1 slide, 1 picture, and no missing picture alt text.
  - `officecli view --mode screenshot` rendered the full slide without visible cropping.

## 2026-06-10 - Plan update: HotpotQA baseline prompt fixed

- Scope update: repair the HotpotQA baseline path after the previous Full Consumer result showed abnormally low F1/EM and evidence-continuation outputs.
- Root cause: the HotpotQA loader concatenated the full distractor context and placed `Question/Answer` at the end; with `max_length=1024`, tokenizer right truncation could drop the actual question and answer marker, causing the base model to continue evidence text.
- Design choice:
  - Keep CMRC2018 and WikiText-2 loaders unchanged.
  - For HotpotQA, build a shorter answer-oriented prompt.
  - Prefer annotated `supporting_facts` titles when present in the dev data, then fall back to question-token paragraph ranking.
  - Cap the selected evidence so `Question ... Answer:` remains in the prompt tail.
- Implementation:
  - Updated `Experiment/core_code/runtime_repo/src/scd_repro/text_benchmarks.py`.
  - Added `Experiment/core_code/runtime_repo/tests/test_text_benchmarks.py`.
  - Rebuilt the release preview so the packaged runtime contains the fixed loader.
- Interpretation update:
  - HotpotQA results after this change are gold-evidence-trimmed QA results, not full distractor retrieval results.
  - The fixed Full baseline is now suitable for model/route quality comparison, but it changes the meaning of prior HotpotQA rows and they should be regenerated before use.

## 2026-06-10 - Plan correction: HotpotQA must use full distractor context

- User correction: HotpotQA should use the complete HotpotQA distractor context, not gold-evidence-trimmed or paragraph-filtered context.
- Scope correction:
  - Reverted HotpotQA loader behavior to preserve the full context from each dataset row.
  - Kept the safer prompt structure with the question repeated before and after evidence.
  - Changed the main remote reproduction scripts' default `MAX_LENGTH` from `1024` to `0`, so HotpotQA is not silently truncated unless a caller explicitly asks for truncation.
- Claim boundary:
  - The gold-evidence-trimmed HotpotQA result from the immediately prior fix is invalid for project claims.
  - The valid repaired HotpotQA baseline is the full-context, no-truncation run recorded in `docs/experiment-log.md`.

## 2026-06-10 - Plan update: HotpotQA retrain on disjoint split and fixed-route profile fast path

- Scope update: retrained Reuse and Patch on HotpotQA using the CMRC-selected Reuse19/Patch13 layer split, then evaluated on a disjoint HotpotQA heldout split.
- Data-leakage policy:
  - Training prompts used the first 200 full-context HotpotQA samples.
  - Heldout evaluation used the next 200 full-context HotpotQA samples.
  - Launcher evidence reported `train=200 eval=200 overlap=0`.
- Result interpretation:
  - Full Consumer heldout HotpotQA: F1 `0.38375940307679435`, EM `0.235`.
  - SCD with HotpotQA-trained Reuse/Patch on the fixed CMRC route: F1 `0.09271782717663724`, EM `0.065`.
  - This confirms that retraining Reuse/Patch on HotpotQA did not rescue the CMRC-selected Reuse19/Patch13 route on full-context HotpotQA.
- Engineering correction:
  - `remote_scripts/run_hotpotqa_train_selected_route.sh` now defaults `SKIP_FIXED_ROUTE_PROFILE=1`.
  - For fixed-route runs, it generates a minimal profile JSON and lets `train_patch --route` select the CMRC patch layers directly.
  - A full `profile_layers` pass is still available by setting `SKIP_FIXED_ROUTE_PROFILE=0`.
  - The current completed run used the older full-profile path because it had already passed that stage before the fast-path change was made.

## 2026-06-10 - Plan update: HotpotQA full-bank route composition

- Scope update: change the HotpotQA route experiment from route-specific patch training to bank-based composition.
- Design:
  - Reuse bank: use the existing HotpotQA full-layer reuse checkpoint from `/disk_n/c101-3-mql/projects/SCD-ICML/runs/hotpotqa_train_selected_cmrc_layers_20260610_122944/lite_to_7b/ckpt_reuse/hotpotqa_seed42`.
  - Patch bank: train a full patch-only bank over all 32 layers using the same HotpotQA train split and pair artifact.
  - Mixed route: evaluate the CMRC-selected Reuse19/Patch13 route by drawing reuse layers from the full reuse bank and patch layers from the full patch bank.
- Data policy:
  - Preserve the existing disjoint split: first 200 HotpotQA full-context samples for bank training, next 200 samples for heldout evaluation.
  - Keep `MAX_LENGTH=0` so HotpotQA uses complete distractor context.
- Implementation:
  - Added `remote_scripts/run_hotpotqa_full_banks_from_existing.sh`.
  - The script writes a `full_patch_bank_route` with patch layers `0..31` for training, a separate `mixed_route` copied from the CMRC-selected route for evaluation, and a minimal full-layer profile so `train_patch --route` trains the bank directly.

## 2026-06-10 - Plan update: main-result reproducibility audit

- Scope update: focus the release/paper audit on whether the Mistral SCD-ICML
  main result is runnable and non-fabricated under the open-source path; Qwen32B
  is explicitly out of scope for this pass.
- Claim boundary:
  - CMRC2018 dev `n=200` Reuse19/Patch13 remains the current positive
    F1/EM-primary result.
  - HotpotQA full-context is a negative/limitation result for the CMRC-selected
    route, not a preserved-quality result.
  - WikiText-2 is an auxiliary NLL/PPL risk signal.
  - The old GitHub HotpotQA patch-only result is a separate compressed-prompt
    benchmark and must not be merged with current full-context HotpotQA.
- Documentation change:
  - Updated release-facing README, reproduction guide, and release manifest to
    distinguish structural readiness from quality retention.
  - Added `docs/scd-main-result-reproducibility-audit-2026-06-10.md` as the
    current audit snapshot for code-cleanup and release decisions.

## 2026-06-10 - Plan update: old HotpotQA patch-only snapshot boundary

- Scope update: reviewed the old GitHub snapshot HotpotQA patch-only run under
  `/disk_n/c101-3-mql/projects/SCD-ICML/kv/snapshots/scd-github-f7d7854`.
- Decision:
  - Treat the old Patch-only F1 `0.3516585390945674`, EM `0.2521269412559082`
    as a real compressed-prompt HotpotQA reproduction target.
  - Do not merge it into the current full-context HotpotQA result table, because
    the old loader performs question-aware evidence compression to about 900
    characters and uses `--max-length 512`.
  - The old full7405 eval includes the 100 questions used to train the patch
    checkpoint, but a question-text overlap check shows that removing those 100
    samples leaves Patch-only at F1 `0.35144349416338316`, EM
    `0.2516084873374401`; the strong old Patch-only result is therefore not
    primarily explained by that small overlap.
  - The old training capture prompt and full7405 eval prompt also differ
    Chinese vs. English, so this path should be treated as a historical
    compressed-prompt target rather than merged into current full-context
    HotpotQA.
  - Keep current full-context HotpotQA conclusions separate unless an explicit
    compressed-HotpotQA ablation is added.
- Evidence added:
  - `docs/experiment-log.md` now records the 7405 paired diagnosis and the
    heldout100 compressed-prompt comparison, with question-text overlap checked
    against the full7405 eval.

## 2026-06-10 - Plan update: paper claims separated from current release evidence

- Scope update: audit the archived ICML camera-ready TeX against the current
  verified release evidence.
- Finding:
  - The archived ICML TeX still contains historical broad claims such as
    Mistral F1 `0.807 -> 0.769`, Mistral SCD F1 `0.7850`, WikiText-2 PPL
    `4.65`, and Qwen-32B table rows.
  - These are not the current open-source release evidence. The current verified
    Mistral release boundary is CMRC2018 dev `n=200`, Reuse19/Patch13, Full
    Consumer F1 `0.657307` / EM `0.365` versus SCD F1 `0.642751` / EM `0.350`.
  - HotpotQA full-context and WikiText-2 are limitations/diagnostics, not broad
    preserved-quality evidence.
- Documentation decision:
  - Added `docs/scd-paper-claim-to-evidence-audit-2026-06-10.md`.
  - Marked the older 2026-06-08 `k=6` release-status documents as superseded.
  - Updated `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`
    so the required config list and run-root example match the selected
    Reuse19/Patch13 default path.
- Verification:
  - Local source readiness passed.
  - Local release preview was rebuilt and passed readiness.
  - Remote runtime manifest/docs were synced to `ser`; remote release preview
    rebuild and readiness commands exited with code 0.

## 2026-06-10 - Plan update: compressed HotpotQA boundary propagated into release docs

- Scope update: propagate the refined old HotpotQA Patch-only diagnosis into
  release-facing documentation and the current paper-results fragment.
- Decision:
  - Keep the old GitHub snapshot HotpotQA Patch-only result as a real
    compressed-prompt benchmark, not as current full-context HotpotQA evidence.
  - Record the leakage audit nuance explicitly: the old full7405 eval includes
    the 100 questions used to train the old Patch checkpoint, but removing those
    samples leaves Patch-only at F1 `0.351443`, EM `0.251608`.
  - This reduces the risk of mislabeling the old result as either fabricated or
    purely leakage-driven, while still keeping it outside the main release
    benchmark.
- Documentation updated:
  - `Experiment/core_code/runtime_repo/README.md`.
  - `Experiment/core_code/runtime_repo/REPRODUCING_SCD_ICML.md`.
  - `Experiment/core_code/runtime_repo/OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - `Publication/paper/scd-icml-current-results-revision-2026-06-10.tex`.
  - `docs/scd-paper-claim-to-evidence-audit-2026-06-10.md`.
- Verification:
  - Local release source readiness passed.
  - Local release preview was rebuilt and passed readiness.
  - Release preview README, reproduction guide, and manifest all contain the
    question-text overlap note and the `0.3514` / `0.2516` retained result.

## 2026-06-10 - Plan update: current-claim index and historical artifact warnings

- Scope update: reduce the risk that historical paper/package artifacts are
  mistaken for the current open-source claim boundary.
- Finding:
  - `docs/scd-icml-mistral-reproduction-2026-06-07.md` still records the older
    reproduction snapshot, including stale HotpotQA/WikiText interpretation and
    old paper-anchor numbers.
  - The archived camera-ready package remains useful for provenance but still
    contains broad historical Mistral/Qwen/WikiText claims in the TeX/PDF.
- Decision:
  - Keep these artifacts for traceability rather than deleting or rewriting the
    historical record.
  - Add explicit superseded/current-claim notices so future code cleanup or
    paper revision starts from the current verified evidence rather than the old
    package.
- Changes:
  - Added `docs/scd-current-claim-index-2026-06-10.md` as the current
    source-of-truth index.
  - Added a superseded notice to
    `docs/scd-icml-mistral-reproduction-2026-06-07.md`.
  - Added a superseded notice to
    `archive/icml-completed-2026-05-24/camera-ready-and-final/final/README.md`.
  - Added
    `archive/icml-completed-2026-05-24/camera-ready-and-final/final/evidence/CURRENT_CLAIM_NOTICE.md`.
  - Added non-rendering TeX comments to both historical `example_paper.tex`
    copies under `archive/icml-completed-2026-05-24/camera-ready-and-final/`.
- Verification:
  - `rg` confirms the current-claim notices are present in the top-level old
    report, archive README/evidence notice, and both historical TeX sources.
  - Local release source readiness still passes.

## 2026-06-10 - Plan update: HotpotQA compressed path restored as separate protocol

- Scope update: convert the old GitHub snapshot HotpotQA Patch-only behavior
  from an external historical note into an explicit, named runtime dataset
  option.
- Decision:
  - Keep CMRC2018 Reuse19/Patch13 as the current positive main-quality claim.
  - Keep HotpotQA full-context/no-truncation as a limitation row for the current
    selected route.
  - Keep the old HotpotQA Patch-only result as a separate compressed-prompt
    benchmark, now exposed as `--dataset hotpotqa_compressed`; do not use it as
    evidence for full-context HotpotQA quality preservation.
- Changes:
  - Added question-aware HotpotQA compression to the current runtime loader.
  - Added `hotpotqa_compressed` to `eval_text_strict.py`.
  - Added unit tests covering the compressed protocol while preserving
    full-context HotpotQA behavior.
  - Updated release-facing docs and rebuilt `release_preview`.
  - Rebuilt the current safe paper PDF and copied it to
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`.
- Verification:
  - Runtime tests: 19 passed.
  - Source and release-preview readiness checks passed.
  - Rebuilt PDF text scan found no stale broad-claim strings and retained the
    current key result numbers.

## 2026-06-10 - Plan update: release package must include model-free tests

- Scope update: strengthen the open-source package from static readiness only
  to static readiness plus a model-free test path that users can run before
  launching GPU jobs.
- Finding:
  - The runtime README Quick Start still referenced missing placeholder config
    files (`configs/models.yaml`, `configs/data.yaml`, `configs/train.yaml`).
    This was a real reproduction-risk footgun even though the claim-facing
    Mistral wrapper and readiness check were valid.
- Decision:
  - The first README command path must use only files actually included in the
    release preview.
  - The release preview should include `tests/`, and readiness should fail if
    those lightweight tests are omitted.
  - The current claim index should name actual summary files, not just run-root
    directories, for HotpotQA and historical compressed HotpotQA evidence.
- Changes:
  - Replaced the generic README Quick Start with install, readiness, unittest,
    and Mistral smoke-test commands.
  - Added `tests/` to `scripts/build_release_tree.py`.
  - Added the lightweight test files and missing quick-start config guard to
    `scripts/check_release_readiness.py`.
  - Updated `OPEN_SOURCE_RELEASE_MANIFEST.md` and
    `docs/scd-current-claim-index-2026-06-10.md`.
  - Rebuilt the safe paper PDF after adding the `hotpotqa_compressed`
    protocol name to the historical HotpotQA paragraph.
- Verification:
  - Source unittest, source pytest, source readiness, release-preview unittest,
    and release-preview readiness all passed.
  - Remote result files were inspected directly on `ser` for the CMRC main
    result, full-context HotpotQA limitation, HotpotQA full-bank heldout
    limitation, and historical compressed HotpotQA Patch-only result.

## 2026-06-10 - Plan update: current release preview smoke-tested on ser

- Scope update: add an actual remote smoke test of the current release preview,
  not just the older remote runtime tree.
- Decision:
  - Treat the smoke run as a wiring/portability check, not as claim evidence.
    The claim evidence remains the CMRC200 selected-route run root.
  - Keep Qwen-32B out of scope.
- Execution:
  - Copied local `release_preview/scd_icml_runtime_release` to
    `/disk_n/c101-3-mql/projects/SCD-ICML/release_preview_from_codex_20260610_2328/runtime_repo`.
  - Ran model-free install, unittest, and readiness checks on the copied release.
  - Ran the README-style minimal CMRC smoke on GPUs 0 and 1 with
    `TRAIN_LIMIT=2`, `EVAL_LIMIT=1`, `MAX_LENGTH=256`, and
    `MAX_NEW_TOKENS=8`.
- Result:
  - Smoke run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/release_preview_smoke_20260610_2328`.
  - `check_release_smoke_status.py` reported `state=complete`.
  - All expected stages completed: capture pair, train reuse, profile layers,
    train patch, route template, and CMRC2018 eval.
  - Aggregate summary JSON/MD files were generated.
  - GPUs were idle again after the run.
## 2026-06-10 - Full non-Qwen paper reproduction scope reset

- Scope: reset the active objective from the narrower CMRC selected-route release check to the full SCD-ICML non-Qwen reproduction target.
- Goal: reproduce every non-Qwen32B paper table/figure trend with open-source code, allowing approximate values only when they preserve the paper conclusion and do not materially diverge from the paper claim.
- Finding: the current strict Reuse19/Patch13 route supports only a narrow CMRC trend; it does not yet reproduce the full paper result set. Full-context HotpotQA currently contradicts the broad preserved-quality claim, and WikiText-2 PPL is not aligned with the paper table for the selected route.
- Important distinction: the older GitHub/new-main full-dev Patch-only result is the strongest current paper-trend anchor, but it is not the same as the current no-fallback mixed Reuse+Patch route.
- Design choice: recover the old Patch-only/full-dev path as a public reproducible anchor, then rebuild mixed Reuse+Patch routes from full Reuse-only and full Patch-only banks, optimizing for maximum Reuse layer count subject to F1/EM retention.
- Artifact: `docs/scd-icml-full-paper-reproduction-matrix-2026-06-10.md`.
- Next action: integrate the historical Patch-only path into the current runtime release surface, then run dataset-specific Patch/Reuse bank training and route search for CMRC2018, HotpotQA, and WikiText-2 without train/eval leakage.

## 2026-06-11 - Plan update: Patch-only anchor recovered, mixed route remains the main gap

- Scope update: the old GitHub/new-main Patch-only path is no longer only
  historical evidence; it has a fresh current-runtime 200-sample rerun on the
  three non-Qwen benchmark protocols.
- New evidence:
  - CMRC2018 Patch-only anchor: Full F1/EM `0.657307/0.365000`,
    Reuse-only `0.012848/0.000000`, Patch-only `0.774237/0.515000`, TTFT
    `100.20 -> 52.32 ms`.
  - HotpotQA-compressed Patch-only anchor: Full F1/EM
    `0.330552/0.250000`, Reuse-only `0.007903/0.000000`, Patch-only
    `0.372672/0.305000`, TTFT `78.94 -> 44.77 ms`.
  - WikiText-2 Patch-only anchor: Full PPL `7.584029`, Reuse-only PPL
    `13.631409`, Patch-only PPL `7.913751`, TTFT `62.21 -> 40.66 ms`.
- Design choice:
  - Treat Patch-only as the recovered quality anchor and upper-bound target.
  - Continue treating the final paper-facing method as mixed Reuse+Patch, not
    Patch-only.
  - Use F1/EM as the primary gate on QA datasets and PPL as the primary gate on
    WikiText-2; NLL remains auxiliary except where it is the metric behind PPL.
- Code/package update:
  - `scripts/run_ser_scd_icml_patchonly_anchor.sh` is now part of the release
    surface and supports optional `RESUME=1` for interrupted runs.
  - `scripts/run_ser_scd_icml_conditional_route_screen.sh` is the preferred
    work-queue entry point for mixed route screening because workers dequeue the
    next route as soon as their GPU pair finishes.
- Next action:
  - Launch or resume conditional mixed-route screening against the recovered
    anchor.
  - Select the maximum-Reuse route that retains QA F1/EM and WikiText-2 PPL.
  - Regenerate paper tables/figures from the accepted route and saved JSON/CSV
    evidence.

## 2026-06-11 - Plan update: CMRC boundary found, cross-dataset validation running

- Scope update:
  - CMRC2018 now has a snapshot-bank mixed Reuse+Patch prefix curve through
    `reuse24_patch08`.
  - The conservative CMRC boundary is `reuse19_patch13`; `reuse20_patch12` is
    a borderline candidate because F1 retention is slightly below `0.95` while
    EM remains close to Full Consumer.
- Design choice:
  - Do not select the final paper-facing route from CMRC alone.
  - Validate `reuse16/19/20` on full-context HotpotQA and WikiText-2 using
    dataset-specific snapshot Reuse/Patch banks.
  - Keep QA F1/EM as the primary gate; use WikiText-2 PPL as the primary
    language-modeling gate.
- Code/package update:
  - `scripts/collect_route_eval_metrics.py` and
    `src/scd_repro/route_selection/profiler.py` now include EM in route-level
    metrics so reports are not F1-only for QA tasks.
- Current remote runs:
  - HotpotQA full-context 200-sample boundary:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_hotpotqa_fullctx200_20260611_001`.
  - WikiText-2 200-sample boundary:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_boundary_wikitext2_200_20260611_001`.
- Next action:
  - Collect boundary metrics as soon as those runs finish.
  - If `reuse19` or `reuse20` survives on full-context HotpotQA and WikiText-2,
    run the selected HotpotQA route with `EVAL_LIMIT=0` for the full dev set.

## 2026-06-11 - Plan update: main-result reproduction status is partial, not complete

- Scope update:
  - The current evidence supports the recovered Patch-only anchor across all
    three non-Qwen benchmark protocols.
  - The mixed Reuse+Patch result is not yet globally complete because
    HotpotQA-compressed constrains the reusable-layer budget much more strongly
    than CMRC2018 or WikiText-2.
- Current boundary:
  - CMRC2018: conservative maximum is `reuse19_patch13`; `reuse20_patch12` is
    borderline by F1 retention but close by EM.
  - WikiText-2: `reuse16_patch16`, `reuse19_patch13`, and `reuse20_patch12`
    all remain within roughly 10% relative PPL increase on the 200-sample
    boundary run.
  - HotpotQA-compressed: `reuse04_patch28` is strong, `reuse08_patch24` has a
    small absolute F1 drop, and `reuse09_patch23`/`reuse12_patch20` fail by F1
    retention. `reuse10/11` were still running/pending at this update.
- Design choice:
  - Do not claim that all main results are reproduced until the selected
    HotpotQA-compressed route is rerun on the full 7405-sample protocol.
  - Treat WikiText-2's PPL gate separately from QA F1/EM; do not let NLL/PPL
    override QA F1/EM when selecting QA routes.
- Next action:
  - Finish the remaining HotpotQA-compressed curve points for completeness.
  - Decide between dataset-specific routes and a common conservative route.
  - Regenerate the paper-facing tables/figures only after the full HotpotQA
    route has been validated.

## 2026-06-11 - Plan update: full HotpotQA validation is now the gating item

- Scope update:
  - The next decisive evidence is the full 7405-sample HotpotQA-compressed
    mixed-route validation, not additional 200-sample screening alone.
- Current action:
  - `reuse08_patch24` full7405 is running on GPU pair `2,3`.
  - `reuse04_patch28` full7405 is queued to start on GPU pair `0,1` after the
    remaining 200-sample mid-curve job completes.
- Decision rule:
  - If `reuse08_patch24` preserves Full Consumer F1/EM within the acceptable
    small-loss range on full7405, it becomes the HotpotQA maximum-Reuse
    candidate.
  - If `reuse08_patch24` fails but `reuse04_patch28` passes, use
    `reuse04_patch28` as the HotpotQA-safe route and keep CMRC/WikiText routes
    dataset-specific unless a common conservative route is required.
- Remaining gap:
  - Main results still cannot be marked fully reproduced until full7405
    HotpotQA metrics are collected and the paper-facing result tables/figures
    are regenerated from current artifacts.

## 2026-06-11 - Plan update: HotpotQA 200-sample boundary closed

- Scope update:
  - The HotpotQA-compressed 200-sample mid-curve is complete through
    `reuse11_patch21`.
- Boundary decision:
  - `reuse09_patch23`, `reuse10_patch22`, and `reuse11_patch21` all fail by
    F1 retention, with retention around `0.73-0.76`.
  - `reuse08_patch24` remains the maximum-Reuse candidate worth validating on
    full7405.
  - `reuse04_patch28` is the safer fallback candidate already running on
    full7405.
- Current GPU use:
  - GPU pair `0,1`: HotpotQA-compressed full7405 `reuse04_patch28`.
  - GPU pair `2,3`: HotpotQA-compressed full7405 `reuse08_patch24`.
- Next action:
  - Wait for full7405 summaries and then decide the HotpotQA paper-facing route
    from actual full-dev F1/EM, not from the 200-sample screen alone.

## 2026-06-11 - Plan update: result-table postprocessing added to release code

- Scope update:
  - The runtime release now includes a paper-table postprocessing helper:
    `scripts/build_scd_icml_result_tables.py`.
- Rationale:
  - The remaining paper-facing work requires regenerating tables from current
    JSON artifacts, not manually copying values from logs.
  - This helper provides a stable bridge from completed Patch-only anchors and
    mixed-route `route_metrics.json` files to Markdown/CSV review tables.
- Verification:
  - Local py_compile, empty-input smoke, unit tests, release readiness, and
    release preview build all passed after adding the script.
  - The script was synced to the active `ser` runtime for use when the full7405
    jobs complete.

## 2026-06-11 - Plan update: dataset-specific route selectors are now reproducible

- Scope update:
  - Route selection is no longer only a manual interpretation of tables.
  - The release selector now supports:
    - QA F1/EM retention gates.
    - QA absolute F1/EM drop gates for low-baseline tasks.
    - WikiText-2 PPL relative-increase gates.
- Current machine-selected routes:
  - CMRC2018: `reuse19_patch13` under F1 retention `>=0.95`.
  - HotpotQA-compressed: `reuse08_patch24` under absolute F1 drop `<=0.03`.
  - WikiText-2: `reuse20_patch12` under PPL relative increase `<=0.10`.
- Design implication:
  - A single common high-Reuse route is not supported by the current evidence;
    HotpotQA is the limiting dataset.
  - The paper-facing result should either use dataset-specific routes or use a
    conservative common route after full7405 confirms the HotpotQA boundary.

## 2026-06-11 - Plan update: partial figure bundle can now be regenerated

- Scope update:
  - The postprocessing path now produces not only Markdown/CSV tables but also
    per-dataset route-curve SVGs from completed route metrics.
- Current artifact:
  - `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_002`.
- Generated figure inputs:
  - `route_curve_cmrc2018.csv/md/svg`.
  - `route_curve_hotpotqa_compressed.csv/md/svg`.
  - `route_curve_wikitext2.csv/md/svg`.
- Remaining gap:
  - These are partial boundary-curve figures from current route screens. The
    final paper-facing figures still need to be regenerated after full7405
    HotpotQA results are available and the final dataset-specific/common route
    decision is made.

## 2026-06-11 - Plan update: full7405 postprocessing automated

- Scope update:
  - The two HotpotQA-compressed full7405 runs are still long-running, but result
    collection no longer requires manual intervention.
- Automation:
  - Installed a remote waiter at
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_and_postprocess.sh`.
  - It waits for both full7405 summaries, then produces:
    `full7405_route_metrics.json`,
    `hotpotqa_full7405_f1_drop003_selection.json`, and
    `result_tables_with_full7405/`.
- Decision point after waiter completes:
  - If full7405 `reuse08_patch24` passes the absolute F1-drop gate, use it as
    the HotpotQA maximum-Reuse route.
  - If `reuse08_patch24` fails and `reuse04_patch28` passes, use `reuse04` as
    the HotpotQA-safe route.
  - If both fail, the HotpotQA route needs another lower-Reuse candidate or a
    training/protocol fix before paper-facing claims can be closed.

## 2026-06-11 - Plan update: local result evidence index created

- Scope update:
  - Added `Experiment/results/README.md` as the local index for result bundles
    and remote evidence paths.
- Current state:
  - The local partial result bundle remains:
    `Experiment/results/scd_icml_result_tables_partial_20260611_002/`.
  - HotpotQA-compressed full7405 jobs are still running:
    `reuse04_patch28` around `868/7405`, `reuse08_patch24` around `1071/7405`
    at the latest check.
- Next action:
  - Let full7405 validation finish, then pull or regenerate the final
    `result_tables_with_full7405/` bundle locally.
  - Only after those rows are incorporated should the main-result reproduction
    be described as complete.

## 2026-06-11 - Plan update: artifact coverage audit added

- Scope update:
  - Added `docs/scd-icml-reproduction-coverage-audit-2026-06-11.md`.
- New requirement before completion:
  - Update the paper/release docs from old full-context HotpotQA diagnostics to
    the `hotpotqa_compressed` full7405 protocol once final results exist.
  - Replace stale CMRC/WikiText table values with generated table outputs.
  - Either add reproducible generators/evidence for non-conceptual static
    figures or replace them with generated route-curve figures from current
    artifacts.

## 2026-06-11 - Plan update: release docs now warn about HotpotQA protocol shift

- Scope update:
  - Updated runtime release docs to state that full-context HotpotQA rows are
    diagnostics and that the current paper-facing HotpotQA validation is
    `hotpotqa_compressed` full7405.
- Verification:
  - Local release readiness and release preview build passed.
  - The updated docs were synced to the active `ser` runtime and remote
    readiness returned exit code `0`.
- Remaining gate:
  - Wait for full7405 results, then regenerate tables and update the paper
    source itself.

## 2026-06-11 - Plan update: profile-based figure reproduction added

- Scope update:
  - Added a release-facing profile figure generator:
    `scripts/build_scd_icml_analysis_figures.py`.
  - Added local generated evidence under
    `Experiment/results/profile_analysis_figures_20260611_001/`.
- Coverage improvement:
  - `patch_error1` and `layer_selection` now have reproducible profile-backed
    SVG replacements from profile JSON.
  - `stack_viz` / t-SNE remains unresolved because representation samples are
    not available in profile JSON.
- Verification:
  - Local compile, 22 tests, readiness, and release preview build passed.
  - Remote 22 tests passed after syncing the script and compressed HotpotQA
    tests.

## 2026-06-11 - Plan update: t-SNE alignment evidence indexed

- Scope update:
  - Added `Experiment/results/tsne_alignment_evidence_20260611_001/` with
    archived real Figure 8 inputs and regenerated `stack_viz` outputs.
- Coverage improvement:
  - The previous `stack_viz` gap is no longer "static PDF only"; it now has
    CSV/meta inputs and a plotting script in the local evidence bundle.
- Remaining decision:
  - Decide whether archived Figure 8 provenance is acceptable for the final
    release, or whether current-runtime KV extraction must be rerun.

## 2026-06-11 - Plan update: offline cost evidence generator and bundle added

- Scope update:
  - Added `scripts/build_offline_cost_table.py` and
    `tests/test_offline_cost_table_script.py` to the release-facing runtime.
  - Added local evidence bundle
    `Experiment/results/offline_cost_evidence_20260611_001/`.
- Coverage improvement:
  - The offline calibration/training cost row is now linked to raw
    `/usr/bin/time -v` logs and a reproducible table generator.
  - Available measured rows cover capture-pair, profile-layers, Reuse training,
    and Patch training from the clean reuse16 Patch16 run.
- Constraint:
  - The paper-mixed `train_patch` time file exists but is empty, so it remains
    a documented missing measurement rather than a numeric claim.
- Remaining gate:
  - HotpotQA-compressed full7405 still needs to finish before the final
    paper-facing result tables and route decision can be closed.

## 2026-06-11 - Plan update: stack/t-SNE plotter promoted to runtime

- Scope update:
  - Added `scripts/build_stack_viz.py` and `tests/test_stack_viz_script.py` to
    the release-facing runtime.
  - Regenerated `Experiment/results/tsne_alignment_evidence_20260611_001/stack_viz.*`
    through the runtime script instead of only the local provenance script.
- Coverage improvement:
  - `stack_viz` now has a documented release-facing plotting command for saved
    KV-distance and t-SNE CSV/meta artifacts.
- Constraint:
  - The current inputs are still archived real representation-analysis
    artifacts, not a fresh extraction from the current runtime.
- Remaining gate:
  - Decide whether archived representation-analysis provenance is acceptable
    for final release; otherwise run fresh KV extraction after the main
    full7405 results close.

## 2026-06-11 - Plan update: stack/t-SNE current extraction completed

- Scope update:
  - Extended `scripts/extract_stack_viz_inputs.py` so it can extract sampled
    source/target/translated KV vectors from pairs plus Reuse checkpoints and
    can regenerate CSV/meta from a sampled vector NPZ.
  - Added `analysis` optional dependency group for `scikit-learn`.
- Evidence:
  - Fresh extraction from
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_clean_reuse16_cmrc50_train200_dev50_20260609_1723/lite_to_7b/`.
  - Local bundle:
    `Experiment/results/stack_viz_current_extraction_20260611_001/`.
- Coverage improvement:
  - `stack_viz` is no longer archive-only. It now has current-runtime sampled
    vectors, true t-SNE CSV/meta, and regenerated PDF/PNG/SVG.
- Remaining gate:
  - The main completion blocker remains HotpotQA-compressed full7405 and the
    final table/paper synchronization after that run finishes.

## 2026-06-11 - Plan update: release result-table output hardened

- Scope update:
  - Extended `scripts/build_scd_icml_result_tables.py` to emit LaTeX tables
    (`patch_anchor_table.tex` and `mixed_route_boundary_table.tex`) alongside
    CSV/Markdown outputs.
  - Added `tests/test_result_tables_script.py` and included it in
    `scripts/check_release_readiness.py`.
  - Updated release documentation so the paper-facing HotpotQA protocol is
    consistently `hotpotqa_compressed` full7405; full-context rows are
    diagnostics only.
- Verification:
  - Regenerated the current partial result-table bundle as
    `Experiment/results/scd_icml_result_tables_partial_20260611_003/` from the
    same completed remote JSON artifacts; this supersedes `002` for table
    extraction because it includes LaTeX outputs.
  - Local `python -m py_compile scripts\build_scd_icml_result_tables.py
    tests\test_result_tables_script.py` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 32 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced the changed files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    with 32 tests.
  - Remote `scripts/build_release_tree.py --out
    release_preview/scd_icml_runtime_release --force` passed and produced
    `RELEASE_BUILD_REPORT.md`.
- Remaining gate:
  - HotpotQA-compressed full7405 remains in progress; do not close the main
    result table until its summaries are available.

## 2026-06-11 - Plan update: current-safe paper synchronized to current artifacts

- Scope update:
  - Updated
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` so its
    CMRC selected-route table, maximum-Reuse boundary table, and non-CMRC
    boundary table match
    `Experiment/results/scd_icml_result_tables_partial_20260611_003/`.
  - Reworded HotpotQA text so `hotpotqa_compressed` full7405 is the pending
    paper-facing row and full-context HotpotQA is diagnostic only.
  - Resized the tight current-boundary tables to avoid overfull boxes.
- Verification:
  - `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=current_safe_build
    scd-icml-current-safe-paper-2026-06-10.tex` passed.
  - `bibtex scd-icml-current-safe-paper-2026-06-10` passed from
    `Publication/paper/current_safe_build`.
  - Final `pdflatex` pass produced
    `Publication/paper/current_safe_build/scd-icml-current-safe-paper-2026-06-10.pdf`
    with no overfull boxes or undefined citation/reference warnings in the
    final log check.
- Remaining gate:
  - Final paper tables still cannot be closed until HotpotQA-compressed
    full7405 finishes and replaces the 200-sample screen row.

## 2026-06-11 - Plan update: paper artifact bundle added to release path

- Scope update:
  - Added `scripts/build_paper_artifact_bundle.py` and
    `tests/test_paper_artifact_bundle_script.py` to the release-facing runtime.
  - Wired the script into `scripts/check_release_readiness.py` and
    `scripts/build_release_tree.py`.
  - Documented bundle assembly in `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Evidence:
  - Generated
    `Experiment/results/paper_artifact_bundle_current_20260611_001/`.
  - Manifest reports `status=ok`, `copied=29`, `missing=0`.
- Verification:
  - Local `python -m py_compile scripts\build_paper_artifact_bundle.py
    tests\test_paper_artifact_bundle_script.py` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest
    tests.test_paper_artifact_bundle_script -v` passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 34 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced runtime changes to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    with 34 tests; latest log is `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview passed and produced
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release/RELEASE_BUILD_REPORT.md`.
- Remaining gate:
  - The bundle is a current partial paper artifact package; it must be
    regenerated after HotpotQA-compressed full7405 summaries are available.

## 2026-06-11 - Plan update: TTFT and latency-quality artifacts added

- Scope update:
  - Extended `scripts/build_scd_icml_result_tables.py` to emit
    `ttft_breakdown_table.csv/md/tex` and per-dataset
    `latency_quality_pareto_*.svg` artifacts from completed patch-anchor and
    mixed-route summaries.
  - Extended `scripts/build_paper_artifact_bundle.py` so these derived tables
    and figures are part of the paper artifact bundle.
- Evidence:
  - Remote result table bundle:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_004`.
  - Local result table bundle:
    `Experiment/results/scd_icml_result_tables_partial_20260611_004/`.
  - Current paper artifact bundle:
    `Experiment/results/paper_artifact_bundle_current_20260611_002/`, with
    manifest `status=ok`, `copied=37`, `missing=0`.
- Verification:
  - Local targeted tests for result tables and artifact bundle passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 34 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced updated runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted tests for result tables and artifact bundle passed.
  - Remote `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    with 34 tests; latest log is `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview regenerated at `2026-06-11 06:11:38 +0800` and
    includes `scripts/build_scd_icml_result_tables.py` plus
    `scripts/build_paper_artifact_bundle.py`.
- Remaining gate:
  - These artifacts are still partial until HotpotQA-compressed full7405 rows
    replace the 200-sample HotpotQA screen rows.

## 2026-06-11 - Plan update: current-safe paper uses generated figures

- Scope update:
  - Extended `scripts/build_scd_icml_analysis_figures.py` so profile-backed
    layer/error and patch-budget figures are emitted as PDF/PNG/SVG.
  - Regenerated `Experiment/results/profile_analysis_figures_20260611_002/`.
  - Regenerated
    `Experiment/results/paper_artifact_bundle_current_20260611_003/` with
    manifest `status=ok`, `copied=43`, `missing=0`.
  - Copied generated PDF figures into
    `Publication/paper/generated_figures/`.
  - Updated
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` to reference
    generated profile and stack/t-SNE figures instead of old static PDFs, and
    replaced the offline-cost paragraph/table with measured rows from the
    generated cost artifact.
- Verification:
  - Local targeted tests for analysis figures and artifact bundle passed.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 34 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced updated runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted tests for analysis figures and artifact bundle passed.
  - Remote `PYTHONPATH=src python -m unittest discover -s tests -v` passed
    with 34 tests; latest log is `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview regenerated at `2026-06-11 06:22:23 +0800` and
    includes `scripts/build_scd_icml_analysis_figures.py` plus
    `scripts/build_paper_artifact_bundle.py`.
  - `pdflatex -interaction=nonstopmode -halt-on-error
    -output-directory=current_safe_build
    scd-icml-current-safe-paper-2026-06-10.tex` passed twice.
  - The final log check found no missing figures, LaTeX errors, overfull boxes,
    or undefined citation/reference warnings.
- Remaining gate:
  - The paper is still a current-safe partial paper; it must be regenerated
    after HotpotQA-compressed full7405 replaces the screen-only HotpotQA row.

## 2026-06-11 - Plan update: paper artifact guard added

- Scope update:
  - Added `scripts/check_paper_artifacts.py` to validate that a paper source
    references existing figures, does not reference stale figure paths, and
    points to a complete paper artifact bundle manifest.
  - Added `tests/test_paper_artifact_check_script.py`.
  - Added the check script to release readiness, release tree construction,
    `README.md`, `REPRODUCING_SCD_ICML.md`, and
    `OPEN_SOURCE_RELEASE_MANIFEST.md`.
  - Copied `fig1.pdf` and `fig2.pdf` into `Publication/paper/` so the current
    safe paper source is self-contained for top-level conceptual figures.
- Verification:
  - `python scripts\check_paper_artifacts.py ...` passed against
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` and
    `Experiment/results/paper_artifact_bundle_current_20260611_003/`.
  - The check report is
    `Experiment/results/paper_artifact_bundle_current_20260611_003/paper_artifact_check_report.json`.
  - Local `$env:PYTHONPATH='src'; python -m unittest discover -s tests -v`
    passed with 36 tests.
  - Local `python scripts\check_release_readiness.py --root .` passed.
  - Local `python scripts\build_release_tree.py --out
    release_preview\scd_icml_runtime_release --force` passed.
  - Synced updated runtime files to
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo`.
  - Remote targeted test `tests.test_paper_artifact_check_script` passed.
  - Remote full unittest passed with 36 tests; latest log is
    `/tmp/scd_icml_full_unittest_latest.log`.
  - Remote release preview report exists at
    `/disk_n/c101-3-mql/projects/SCD-ICML/repro_current_from_codex_20260607-154030/runtime_repo/release_preview/scd_icml_runtime_release/RELEASE_BUILD_REPORT.md`
    and includes `scripts/check_paper_artifacts.py`.
- Remaining gate:
  - This guards the current paper artifact package, but it does not close the
    main reproduction gate. HotpotQA-compressed full7405 F1/EM is still pending.

## 2026-06-11 - Plan update: neighboring-baseline claim boundary is checked

- Scope update:
  - Extended `scripts/check_paper_artifacts.py` with literal
    `--require-text` and `--forbid-text` checks.
  - Updated the current paper artifact check to require the neighboring
    compression baseline exclusion text:
    `not part of the current verified release evidence` and
    `do not use them as a claim-facing comparison`.
  - Updated runtime release docs so the exclusion is reproducible by command,
    not just by manual reading.
- Verification:
  - Local paper artifact check passed and wrote
    `Experiment/results/paper_artifact_bundle_current_20260611_003/paper_artifact_check_report.json`
    with `errors=[]`.
  - Local full unittest passed with 37 tests.
  - Local release readiness and release preview passed.
  - Remote targeted test `tests.test_paper_artifact_check_script` passed with
    3 tests.
  - Remote full unittest passed with 37 tests.
  - Remote release preview report exists and includes
    `scripts/check_paper_artifacts.py`.
- Remaining gate:
  - This closes the unsupported neighboring-baseline item by checked exclusion
    from the current release-facing paper. It does not close HotpotQA-compressed
    full7405 or final table regeneration.

## 2026-06-11 - Plan update: patch-budget replacement table enters bundle

- Scope update:
  - Regenerated the current paper artifact bundle as
    `Experiment/results/paper_artifact_bundle_current_20260611_004/`.
  - Included `tables/patch_gain_curve.md` alongside the existing
    `patch_gain_curve.csv` and `figures/patch_budget_curve.*` artifacts.
  - Updated the current-safe paper evidence root to point to bundle `004`.
- Verification:
  - `paper_artifact_manifest.json` reports `status=ok`, `copied=44`,
    `missing=0`.
  - `scripts/check_paper_artifacts.py` passed against the current-safe paper and
    bundle `004`, including the required patch-budget bundle file and
    neighboring-baseline exclusion text.
  - Targeted paper artifact bundle tests passed.
  - Local full unittest passed with 37 tests; local release readiness and
    release preview passed.
  - Remote targeted bundle tests passed; remote full unittest passed with 37
    tests; remote release preview passed.
  - The current-safe paper rebuilt successfully after the evidence-root update
    to bundle `004`, producing a 15-page PDF with no missing figures, LaTeX
    errors, overfull boxes, or undefined citation/reference warnings.
- Remaining gate:
  - This improves coverage for patch-budget/patch-ablation replacement
    artifacts, but final completion still waits for HotpotQA-compressed
    full7405 and final table regeneration.

## 2026-06-11 - Plan update: final full7405 bundle step is scripted

- Scope update:
  - Added `scripts/finalize_scd_icml_full7405_bundle.py` to wrap final
    post-full7405 paper artifact assembly and paper artifact checking.
  - Added a unit test for the wrapper.
  - Added the wrapper to release readiness, release tree, README,
    `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Verification:
  - Local targeted finalizer test passed.
  - Local full unittest passed with 38 tests; local release readiness and
    release preview passed.
  - Remote targeted finalizer test passed.
  - Remote full unittest passed with 38 tests; remote release preview passed and
    includes `scripts/finalize_scd_icml_full7405_bundle.py`.
- Remaining gate:
  - The finalizer is ready, but cannot be run on final evidence until the
    HotpotQA-compressed full7405 summaries and waiter-generated
    `result_tables_with_full7405/` exist.

## 2026-06-11 - Plan update: premature final packaging is guarded

- Scope update:
  - Strengthened `scripts/finalize_scd_icml_full7405_bundle.py` so it validates
    `mixed_route_boundary_table.csv` before bundling.
  - The default finalizer now requires at least one table `source` field to
    contain `full7405`.
  - Added a regression test that rejects pre-full7405 tables.
- Verification:
  - Local full unittest passed with 39 tests; release readiness and release
    preview passed.
  - Remote targeted finalizer tests passed with 2 tests; remote full unittest
    passed with 39 tests; remote release preview passed.
  - Remote negative smoke against
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_result_tables_partial_20260611_004`
    returned nonzero and reported the missing `full7405` source substring.
- Remaining gate:
  - This prevents premature final packaging but does not complete the objective.
    Finalization still waits on HotpotQA-compressed full7405 summaries.

## 2026-06-11 - Plan update: full7405 finalization waits automatically

- Scope update:
  - Added `scripts/wait_finalize_scd_icml_full7405_bundle.sh` so the final
    paper-bundle step can wait for the full7405 result table and run the guarded
    finalizer immediately after the postprocess table appears.
  - Added the wait wrapper to the release tree and release readiness checks.
  - Updated README and reproduction docs to call it through `bash
    scripts/wait_finalize_scd_icml_full7405_bundle.sh`, avoiding executable-bit
    assumptions after `scp` or archive extraction.
- Verification:
  - Local py_compile, 39-test unittest, release readiness, and release preview
    passed after the wrapper and docs update.
  - Remote syntax checks and 39-test unittest passed after syncing to `ser`.
  - Remote release preview includes the wait wrapper, and the synced docs include
    the `bash scripts/...` invocation.
  - A remote watcher is now running under
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_postprocess_20260611_001/wait_finalize_bundle.log`.
- Remaining gate:
  - HotpotQA-compressed full7405 is still running. Latest observed progress was
    `reuse04_patch28=4805/7405` and `reuse08_patch24=4830/7405`; no full7405
    summary, final route-selection report, or final paper artifact bundle exists
    yet.

## 2026-06-11 - Plan update: full7405 postprocess is release-scripted

- Scope update:
  - Added `scripts/wait_postprocess_scd_icml_full7405_tables.sh` to the runtime
    release so the full7405 eval summaries can be converted into metrics,
    selector reports, and `result_tables_with_full7405/` without relying on an
    ad hoc remote script.
  - The wrapper uses colon-separated env lists to keep remote `nohup env ...`
    invocations robust.
  - Added the wrapper to release readiness, release preview, README,
    `REPRODUCING_SCD_ICML.md`, and `OPEN_SOURCE_RELEASE_MANIFEST.md`.
- Verification:
  - Local full unittest and release checks passed after the wrapper addition.
  - Remote shell syntax check passed.
  - Remote release preview contains the wrapper and the updated colon-separated
    README command.
- Remote state:
  - The old ad hoc postprocess waiter was stopped.
  - The release-script postprocess watcher is active and waiting for
    `summaries_ready=0/2`.
  - Latest full7405 progress was `reuse04_patch28=5091/7405` and
    `reuse08_patch24=5102/7405`.
- Remaining gate:
  - Wait for the full7405 summaries, inspect the generated HotpotQA F1/EM
    metrics, and let the finalizer produce the post-full7405 paper artifact
    bundle.

## 2026-06-11 - Plan update: release checks now cover shell wrappers

- Scope update:
  - Strengthened `scripts/check_release_readiness.py` so shell launchers and
    full7405 wait wrappers are checked for LF line endings, bash shebangs,
    strict mode, and `bash -n` syntax when bash is available.
  - This closes a release-quality gap exposed by the postprocess/finalizer
    wrappers.
- Verification:
  - Local release readiness, 39-test unittest, and release preview passed.
  - Remote readiness and release preview returned success after syncing the
    stronger readiness script.
- Remote state:
  - Full7405 is still running. Latest observed progress was
    `reuse04_patch28=5243/7405` and `reuse08_patch24=5248/7405`.
  - No full7405 aggregate summaries or final result tables existed at the
    latest check.

## 2026-06-11 - Plan update: bandwidth-sensitivity artifact added to final table contract

- Scope update:
  - Strengthened the paper-facing result generator so it now emits
    `bandwidth_sensitivity.csv`, `bandwidth_sensitivity.md`, and
    `bandwidth_sensitivity.svg`.
  - The artifact is payload-based and intentionally limited to rows with real
    `payload_mb`; it does not invent bandwidth curves for mixed-route rows whose
    current route metrics have `payload_bytes=None`.
  - Added these outputs to the paper artifact bundle required-file list and to
    the full7405 finalizer tests, so the eventual post-full7405 bundle will
    contain the bandwidth-sensitivity evidence.
- Verification:
  - Local targeted tests for result tables, paper bundle, and full7405 finalizer
    passed.
  - Local release readiness, local release preview, and local full unittest
    passed.
  - Updated scripts/tests were synced to `ser`; remote targeted tests passed.
- Remote state:
  - HotpotQA-compressed full7405 remains the gating item.
  - Latest observed progress was `reuse04_patch28=5556/7405` and
    `reuse08_patch24=5549/7405`.
  - The active postprocess and finalizer watchers are still waiting for
    summaries/result tables.
- Remaining gate:
  - Let full7405 finish, inspect the generated HotpotQA F1/EM selection report,
    then allow the finalizer to build a bundle whose result tables include both
    full7405 rows and the new bandwidth-sensitivity outputs.

## 2026-06-11 - Plan update: release docs require bandwidth-sensitivity output

- Scope update:
  - Added `bandwidth_sensitivity` documentation to the runtime README,
    reproduction guide, and open-source release manifest.
  - Strengthened `scripts/check_release_readiness.py` so the release docs must
    mention the new bandwidth-sensitivity artifact contract.
- Verification:
  - Local readiness, targeted tests, and release preview passed.
  - The updated files were synced to `ser`.
  - Remote readiness and release preview returned success.
  - Remote release-preview grep confirmed the new snippet in README,
    `REPRODUCING_SCD_ICML.md`, `OPEN_SOURCE_RELEASE_MANIFEST.md`, and the
    readiness checker.
- Remote state:
  - HotpotQA-compressed full7405 is still running.
  - Latest observed progress was `reuse04_patch28=5713/7405` and
    `reuse08_patch24=5700/7405`.
  - No full7405 summary or final result table exists yet.
- Remaining gate:
  - Continue waiting for both full7405 summaries; then inspect F1/EM, run or
    verify postprocess output, and finalize the paper artifact bundle.

## 2026-06-11 - Plan update: current partial bundle refreshed to 005

- Scope update:
  - Rebuilt the pre-full7405 result table bundle as
    `Experiment/results/scd_icml_result_tables_partial_20260611_005`.
  - Built `Experiment/results/paper_artifact_bundle_current_20260611_005` from
    the refreshed result tables, profile figures, stack/t-SNE artifacts, and
    offline-cost artifacts.
  - This does not replace the final full7405 bundle; it only keeps the current
    local evidence bundle aligned with the latest artifact contract.
- Verification:
  - `paper_artifact_bundle_current_20260611_005/paper_artifact_manifest.json`
    reports `status=ok` and `missing=[]`.
  - `paper_artifact_check_report.json` passed against the current-safe paper and
    required `tables/bandwidth_sensitivity.md`.
  - Local release readiness and targeted tests passed.
- Remote state:
  - HotpotQA-compressed full7405 remains in progress.
  - Latest observed progress was `reuse04_patch28=5774/7405` and
    `reuse08_patch24=5758/7405`.
  - No full7405 summary or final result table exists yet.
- Remaining gate:
  - Wait for full7405 summaries, inspect HotpotQA F1/EM selection, then build
    the final full7405 paper bundle.

## 2026-06-11 - Plan update: full7405 still running at 08:15

- Remote state:
  - Latest observed progress was `reuse04_patch28=5900/7405` and
    `reuse08_patch24=5879/7405`.
  - Summary files are still absent.
  - Postprocess and finalizer watchers are active and waiting.
- Remaining gate:
  - Continue once both summaries exist; no duplicate launch is needed.

## 2026-06-11 - Plan update: local full tests refreshed while full7405 runs

- Verification:
  - Local full unittest passed with 39 tests.
  - Local release preview passed after the latest release-doc and artifact
    contract updates.
- Remote state:
  - Latest observed progress was `reuse04_patch28=5990/7405` and
    `reuse08_patch24=5968/7405`.
  - Summary files are still absent.
- Remaining gate:
  - Continue monitoring; once both summaries exist, inspect the generated
    full7405 F1/EM and finalize the paper artifact bundle.

## 2026-06-11 - Plan update: full7405 still healthy at 08:23

- Remote state:
  - Latest observed progress was `reuse04_patch28=6081/7405` and
    `reuse08_patch24=6050/7405`.
  - Summary files are still absent.
  - Existing postprocess and finalizer watchers remain active.
- Remaining gate:
  - Wait for both full7405 summaries; then inspect F1/EM and finalize the
    full7405 paper bundle.

## 2026-06-11 - Plan update: full7405 still healthy at 08:24

- Remote state:
  - Latest observed progress was `reuse04_patch28=6116/7405` and
    `reuse08_patch24=6084/7405`.
  - Summary files are still absent.
  - Postprocess and finalizer watchers remain active and waiting.
- Remaining gate:
  - Inspect full7405 F1/EM after summaries exist; no duplicate launch is needed.

## 2026-06-11 - Plan update: full7405 still healthy at 08:33

- Local release state:
  - Full unittest passed with 39 tests.
  - Runtime release readiness passed.
  - Release preview build passed and the preview tree also passed readiness.
- Remote state:
  - Latest observed progress was `reuse04_patch28=6294/7405` and
    `reuse08_patch24=6265/7405`.
  - Summary files are still absent.
  - Postprocess and finalizer watchers remain active and waiting:
    `summaries_ready=0/2`, `table_ready=no`.
- Remaining gate:
  - Continue monitoring until both full7405 summaries exist, then inspect
    HotpotQA F1/EM, pull `result_tables_with_full7405/`, and finalize the
    paper artifact bundle.

## 2026-06-11 - Plan update: full7405 gate closed

- Result:
  - HotpotQA-compressed full7405 completed for `reuse04_patch28` and
    `reuse08_patch24`.
  - The full7405 selector chose `reuse04_patch28`: Full F1/EM
    `0.291855/0.206077`, SCD F1/EM `0.328284/0.242944`.
  - `reuse08_patch24` failed the full7405 F1-drop gate with SCD F1/EM
    `0.242740/0.169885`.
- Code fix:
  - Added `PYTHONPATH="${RUNTIME}/src${PYTHONPATH:+:${PYTHONPATH}}"` to both
    full7405 shell watcher scripts so remote helper imports work from a plain
    shell invocation.
- Final artifacts:
  - `Experiment/results/scd_icml_result_tables_full7405_20260611_001`.
  - `Experiment/results/paper_artifact_bundle_with_full7405_20260611_001`.
  - `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex` now reflects
    full7405 completion rather than a pending row.
  - `Publication/paper/current_safe_build/scd-icml-current-safe-paper-2026-06-10.pdf`
    was regenerated by `pdflatex`.
- Verification:
  - Local paper artifact gate passed against the full7405 bundle.
  - Full unittest passed.
  - Release readiness passed.
  - Release preview build passed and the preview tree passed readiness.
- Remaining scope:
  - Qwen32B remains excluded by user instruction.
  - Exact historical neighboring-baseline reruns and exact old patch-budget
    sweeps remain optional; they are not part of the current verified
    release-facing evidence.

## 2026-06-11 - Plan update: HotpotQA maximum reuse boundary

- Change:
  - Added full7405 boundary evals for HotpotQA-compressed
    `reuse05_patch27`, `reuse06_patch26`, and `reuse07_patch25`.
- Why:
  - `reuse04_patch28` passes the full7405 F1/EM gate and `reuse08_patch24`
    fails it, but the maximum acceptable reuse depth cannot be claimed without
    checking the intervening reuse budgets.
- Execution:
  - Remote run root:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_hotpotqa_compressed_full7405_reuse05_07_20260611_001`.
  - Launcher PID: `3380160`.
  - Scheduler uses GPU pairs `0,1` and `2,3` with a simple work queue.
  - `reuse07_patch25` is now assigned to a wait-then-run helper using the
    Full+SCD-only eval config after `reuse05` or `reuse06` frees one GPU pair;
    this avoids rerunning Raw/4-bit/Reuse-only baselines that are not needed for
    the maximum-reuse F1/EM gate.
- Remaining gate:
  - Wait for the three full7405 summaries, combine them with the existing
    `reuse04` and `reuse08` results, then update route selection, result
    tables, bundle, and paper text if the selected HotpotQA route changes.

## 2026-06-11 - Plan update: boundary eval fast path

- Change:
  - Replaced the running all-method full7405 evals for HotpotQA
    `reuse05_patch27` and `reuse06_patch26` with Full+SCD-only evals.
- Why:
  - The boundary decision uses the primary F1/EM gate, and the selector only
    requires `full_consumer` and `scd_full`. Raw KV, 4-bit KV, and Reuse-only
    are already covered elsewhere and are not needed for this maximum-reuse
    boundary.
- Current execution:
  - `reuse05_patch27` fast-path PID: `3397409`.
  - `reuse06_patch26` fast-path PID: `3397412`.
  - `reuse07_patch25` remains assigned to the wait-then-run Full+SCD-only
    helper.
- Remaining gate:
  - Wait for the Full+SCD-only summaries, then let the boundary postprocess
    watcher combine `reuse04` through `reuse08` and update the final paper
    artifacts if the selected HotpotQA route changes.

## 2026-06-11 - Plan update: final paper artifact stale-state guard

- Change:
  - Strengthened the full7405 finalizer's default paper check to require an
    explicit HotpotQA full7405 selected-route statement and to reject stale
    pending-state text.
  - Added required table-label checks for the current safe paper's
    claim-facing tables so the bundle cannot pass if those tables are removed
    from the paper source.
- Why:
  - The full objective is not only to produce metric files, but to ensure the
    open-source paper artifact bundle cannot pass while the paper still says
    full7405 is pending or otherwise stale.
- Verification:
  - Local targeted unittest for the finalizer and paper artifact checker passed.
  - Local full unittest passed with 40 tests, and release readiness passed.
  - The local release preview tree was rebuilt and passed readiness with the
    stricter finalizer included.
  - The same targeted tests passed on the `ser` runtime used by the active
    boundary watcher.
- Remaining gate:
  - When `reuse05`/`reuse06`/`reuse07` complete, the finalizer watcher should
    now validate both the generated bundle and the current paper text before
    the final full7405 boundary bundle is accepted.

## 2026-06-11 - Plan update: release documentation state refresh

- Change:
  - Updated the runtime release docs so HotpotQA full7405 is no longer
    described as initially pending.
- Why:
  - The first full7405 bracket has evidence now (`reuse04_patch28` passes,
    `reuse08_patch24` fails). Leaving stale docs would make the open-source
    package look internally inconsistent even if the code and tables are right.
- Verification:
  - Release readiness passed after the doc updates.
  - The release preview was rebuilt and passed readiness.
  - The updated docs were synced to the active `ser` runtime directory.
- Remaining gate:
  - Once the boundary refinement rows finish, update the same docs again only
    if the selected HotpotQA route changes from `reuse04_patch28`.

## 2026-06-11 - Plan update: release readiness stale-doc guard

- Change:
  - Added explicit stale HotpotQA full7405 documentation checks to
    `scripts/check_release_readiness.py`.
- Why:
  - The release package should fail readiness if docs regress to saying the
    compressed-protocol full7405 table is pending after it has already been
    closed for the first passing/failing bracket.
- Verification:
  - Local readiness passed.
  - Local full unittest passed with 40 tests.
  - The release preview was rebuilt and passed readiness.
  - The updated readiness script was synced to the active `ser` runtime.
  - Remote readiness in the active `ser` runtime passed with return code `0`.
- Remaining gate:
  - Re-run readiness after the boundary refinement result is pulled and docs
    are updated with the final selected HotpotQA route.

## 2026-06-11 - Plan update: checked remote finalizer

- Change:
  - Added a second remote finalizer watcher with `SKIP_PAPER_CHECK=0` and a
    synced copy of the current safe paper source.
- Why:
  - The original remote finalizer watcher intentionally skips paper-source
    checking by default. The final release gate needs both bundle/table checks
    and paper-text checks.
- Current execution:
  - Checked finalizer PID: `3429341`.
  - Output directory:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_full7405_boundary_postprocess_20260611_001/paper_artifact_bundle_with_full7405_boundary_checked`.
- Remaining gate:
  - After `reuse05`/`reuse06`/`reuse07` summaries exist and postprocess writes
    the result table, verify that this checked finalizer produces
    `finalize_report.json` and `paper_artifact_check_report.json` with no
    errors.

## 2026-06-11 - Plan update: selected-route paper-text guard

- Change:
  - Added selector-aware paper-text validation to the full7405 finalizer.
- Why:
  - If boundary refinement selects `reuse05`, `reuse06`, or `reuse07`, the
    final release must not silently keep paper text that says
    `Reuse04/Patch28`.
- Verification:
  - Local targeted tests passed with 7 tests.
  - Local full unittest passed with 41 tests.
  - Release readiness and rebuilt release-preview readiness passed.
  - Remote targeted finalizer/checker tests passed in the active `ser`
    runtime.
- Remaining gate:
  - If the checked finalizer fails after boundary postprocess, inspect
    `selected_route_paper_requirement` in `finalize_report.json`, update the
    paper route text to match the selector, and rerun finalization.

## 2026-06-11 - Plan update: selector-route guard documentation

- Change:
  - Updated release docs to explain that the full7405 finalizer enforces the
    selector's chosen `ReuseXX/PatchYY` token in the paper source.
- Verification:
  - Local readiness passed.
  - Release preview was rebuilt and passed readiness.
  - Remote readiness passed after syncing the docs to `ser`.
- Remaining gate:
  - After boundary refinement, update docs and paper only if the selected
    HotpotQA route changes; the finalizer will enforce that consistency.

## 2026-06-11 - Plan update: local verification during boundary wait

- Change:
  - Re-ran the local runtime tests, release readiness, release-preview build,
    release-preview readiness, and current paper PDF compilation while the
    remote full7405 HotpotQA boundary jobs continued.
- Why:
  - The final objective requires both experiment evidence and open-source code
    completeness. Verifying the static/runtime package now reduces the final
    closeout to result-table synchronization, paper route-text consistency, and
    final artifact checks once the boundary jobs finish.
- Verification:
  - Local unittest passed with 41 tests.
  - Local release readiness passed.
  - Release preview rebuilt and passed readiness.
  - Current paper source compiled successfully with
    `pdflatex -include-directory=current_safe_build` after two passes.
  - Remote GPU inspection confirmed `reuse05` is using CUDA `0,1` and
    `reuse06` is using CUDA `2,3`; `reuse07` remains queued until one pair
    finishes.
- Remaining gate:
  - Wait for `reuse05`/`reuse06`/`reuse07` summaries, then regenerate/pull the
    boundary tables and rerun the checked paper-artifact finalizer.

## 2026-06-11 - Plan update: HotpotQA leakage-filtered closeout

- Change:
  - Added `scripts/filter_strict_eval_by_qids.py` and release/readiness coverage
    for qid-filtered strict-eval reaggregation.
- Why:
  - The HotpotQA-trained checkpoint source has 100 training questions that
    overlap with the full7405 dev file. Final HotpotQA claims should not rely
    only on unfiltered full7405 metrics.
- Verification:
  - The script's local and remote targeted tests passed.
  - Local and remote release readiness passed after adding the script and docs.
  - On the completed 04/08 routes, filtering out the 100 overlapping qids keeps
    the conclusion unchanged: `reuse04_patch28` passes and `reuse08_patch24`
    fails the F1-drop gate.
- Remaining gate:
  - After `reuse05`/`reuse06`/`reuse07` finish, generate both unfiltered
    full7405 and no-train-overlap summaries/tables. Use the filtered result as
    the leakage-sensitive HotpotQA evidence in the final audit.

## 2026-06-11 - Plan update: HotpotQA reuse07 qid-sharded acceleration

- Change:
  - Replaced the remaining `reuse07_patch25` single-process full7405 strict
    evaluation with two disjoint qid shards and an automatic merge watcher.
- Why:
  - `reuse05_patch27` and `reuse06_patch26` were complete, leaving only
    `reuse07_patch25`. The single-process run was using only one GPU pair and
    was projected to finish significantly later, while GPU pair `2,3` was idle.
    Qid sharding preserves the same sample set and route while improving
    hardware utilization.
- Implementation approach:
  - Split the `hotpotqa_compressed` full7405 qids into `3703` and `3702`
    examples.
  - Run shard0 on CUDA `0,1` and shard1 on CUDA `2,3`.
  - Merge `per_sample_metrics.jsonl` from both shards and reaggregate with
    `scripts/filter_strict_eval_by_qids.py`.
  - Write the merged route-level summary back to
    `conditional_reuse_reuse07_patch25/scd_icml_mistral_summary.json`, which is
    the same file watched by the existing full7405 postprocess script.
- Verification gate:
  - Do not treat `reuse07_patch25` as complete until both shard outputs exist,
    the merged summary reports `7405` qids, and the full7405 postprocess watcher
    consumes it successfully.

## 2026-06-11 - Plan update: HotpotQA full7405 boundary closed

- Change:
  - Closed the HotpotQA-compressed full7405 boundary after adding
    `reuse05_patch27`, `reuse06_patch26`, and `reuse07_patch25` to the
    completed 04/08 pair.
- Outcome:
  - Unfiltered selector chooses `reuse06_patch26`: Full F1/EM
    `0.291855/0.206077`, SCD F1/EM `0.304212/0.221742`.
  - `reuse07_patch25` fails with F1 drop `0.034595`; `reuse08_patch24` fails
    with F1 drop `0.049115`.
  - No-train-overlap selector after excluding 100 overlapping qids also chooses
    `reuse06_patch26`: Full F1/EM `0.292006/0.205886`, SCD F1/EM
    `0.304488/0.221492`.
- Paper/artifact update:
  - Updated the current paper source to report HotpotQA-compressed full7405 as
    `Reuse06/Patch26`.
  - Re-ran the checked finalizer on `ser`; it passed with `status=ok`,
    `bundle_status=ok`, selected-route paper requirement `Reuse06/Patch26`, and
    paper artifact check `errors=[]`.
  - Pulled the final boundary tables, checked paper bundle, and selector reports
    into `Experiment/results/`.
- Remaining gate:
  - Run the final local verification pass: unit tests, release readiness,
    release-preview readiness, paper compilation, and artifact report sanity
    checks.

## 2026-06-11 - Plan update: neighboring-baseline appendix rerun path

- Change:
  - Added current-runtime public rerun paths for the original appendix
    neighboring compression baselines:
    `run_svdllm_style_kv_cmrc.py`, `run_slicegpt_style_kv_cmrc.py`, and
    `run_tensorllm_style_kv_cmrc.py`.
  - Added `build_neighboring_baseline_table.py` to consolidate completed
    baseline summaries plus an SCD summary into CSV/Markdown/LaTeX appendix
    table artifacts.
  - Included these scripts and the new unit test in release readiness and
    release preview.
- Why:
  - The current safe paper correctly excludes unsupported historical neighboring
    baseline rows from claim-facing evidence, but the active full objective asks
    for all non-Qwen paper tables/figures to be reproducible. The appendix
    table therefore needs an executable public rerun path rather than only a
    checked exclusion.
- Verification:
  - Local full test suite passed with `43` tests using `PYTHONPATH=src`.
  - Local and remote release readiness passed after the additions.
  - Release preview rebuilt successfully and passed readiness.
  - Remote `ser` smoke ran all three adapted neighboring baselines with
    `limit=2`, and the table consolidator generated
    `neighboring_baselines.{csv,md,tex}`.
- Current boundary:
  - The `n=2` smoke proves execution and table generation only. It is not a
    paper-quality numeric rerun and must not be used to restore the appendix
    table values.
- Next plan:
  - Run full-sample CMRC neighboring-baseline reruns for SVD-LLM-style,
    SliceGPT-style, and TensorLLM-style adapted KV transfer.
  - Rebuild `neighboring_baselines.{csv,md,tex}` from full summaries.
  - Decide whether to restore the appendix table in the paper source or keep it
    as a non-claim supplementary artifact.

## 2026-06-11 - Plan update: neighboring-baseline full rerun in progress

- Change:
  - Launched the CMRC200 full neighboring-baseline wrapper on `ser`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/neighboring_baselines_cmrc200_20260611_001`.
- Execution plan:
  - Run SVD-LLM-style and TensorLLM-style in parallel on two GPU pairs.
  - After both finish, run SliceGPT-style with the CMRC200 pair artifact from
    the GitHub snapshot.
  - Build the consolidated appendix table with
    `build_neighboring_baseline_table.py`.
- Current status:
  - Wrapper PID: `3565103`.
  - SVD-LLM-style PID: `3565119`.
  - TensorLLM-style PID: `3565120`.
  - Latest observed state: both child jobs were still in model-loading/disk I/O,
    with no traceback in their logs.
- Gate:
  - Do not mark the appendix neighboring-baseline table reproduced until
    `summary.json` exists for all three methods and
    `table/neighboring_baselines.md` is generated from full `LIMIT=200`
    summaries.

## 2026-06-11 - Plan update: neighboring-baseline protocol alignment gate

- Scope update:
  - The full CMRC200 adapted neighboring-baseline rerun has completed for
    SVD-LLM-style, SliceGPT-style, and TensorLLM-style adapted KV transfer.
  - The first generated appendix table is not final because it combines
    `max_length=256` adapted-baseline rows with the existing SCD selected-route
    row from `max_length=1024`.
- Implementation update:
  - `scripts/build_neighboring_baseline_table.py` now records protocol metadata
    in the manifest and emits warnings when protocol fields or Full-consumer F1
    values differ across sources.
  - Local release readiness, full unit tests, and release-preview rebuild
    passed after the change.
- Active run:
  - Launched the same-protocol SCD Reuse19/Patch13 CMRC200 evaluation at
    `max_length=256`:
    `/disk_n/c101-3-mql/projects/SCD-ICML/runs/scd_icml_selected_reuse19_cmrc200_maxlen256_for_neighboring_20260611_001`.
  - GPU policy: use GPU0/1 and avoid GPU2, which was already occupied by
    another user process.
- Next gate:
  - After SSH access to `ser` recovers, confirm the SCD256 summary, rebuild
    the neighboring-baseline table against same-protocol summaries, copy the
    aligned table into `Experiment/results`, and update the reproduction matrix
    from pending to covered.

## 2026-06-11 - Plan update: prevent mixed-protocol neighboring table defaults

- Scope update:
  - The neighboring-baseline table path now has two gates:
    completed adapted-baseline summaries and an explicitly supplied
    same-protocol SCD summary.
- Implementation update:
  - `run_ser_neighboring_baselines_cmrc.sh` no longer defaults `SCD_SUMMARY` to
    an older selected-route run. Without `SCD_SUMMARY`, it leaves the table as
    pending instead of generating a mixed-protocol artifact.
  - `check_release_readiness.py` now enforces this protocol-safety behavior.
  - Reproduction docs and the release manifest now state that `SCD_SUMMARY`
    must match the adapted-baseline protocol before table consolidation.
- Verification:
  - Local targeted neighboring-table tests passed.
  - Local full tests passed with `44` tests.
  - Local release readiness passed.
  - Release preview rebuilt and passed readiness.
- Remaining gate:
  - Sync the latest wrapper/docs to `ser` after SSH recovers, then finish the
    SCD256 summary check and same-protocol table regeneration.

## 2026-06-11 - Plan update: hard fail on neighboring protocol mismatch

- Scope update:
  - The neighboring-baseline appendix table must now pass an explicit
    same-protocol check before it can be generated as a paper-facing artifact.
- Implementation update:
  - `build_neighboring_baseline_table.py` gained
    `--require-compatible-protocol`.
  - `run_ser_neighboring_baselines_cmrc.sh` passes that flag whenever
    `SCD_SUMMARY` is provided.
  - Release readiness checks for both the pending-table behavior and the hard
    compatibility flag.
- Verification:
  - Local neighboring-table tests passed, including a test that mismatched
    `max_length` fails under `--require-compatible-protocol`.
  - Local full tests passed with `45` tests.
  - Local release readiness and release-preview readiness passed.
- Remaining gate:
  - Same as before: sync to `ser` when SSH recovers, then use the SCD256 summary
    to generate the final same-protocol table.

## 2026-06-11 - Plan update: neighboring appendix restored with max_length1024

- Scope update:
  - The neighboring-baseline appendix is no longer pending. It has a
    same-protocol current-runtime CMRC200 rerun at `max_length=1024`.
- Reason for rerun:
  - The same-protocol `max_length=256` table passed protocol checks but had a
    weak Full Consumer baseline and only `81.0%` SCD retention. That would not
    preserve the paper's qualitative claim.
  - The `max_length=1024` rerun aligns with the selected-route SCD summary and
    restores the paper-style trend.
- Result:
  - Final local artifact:
    `Experiment/results/neighboring_baselines_cmrc200_maxlen1024_20260611_001/table/neighboring_baselines.md`.
  - Manifest warnings are empty under `--require-compatible-protocol`.
  - SCD retention/speedup: `97.8%` / `2.47x`.
  - Adapted baselines remain far behind: SVD-LLM-style `2.4%` retention,
    SliceGPT-style `2.8%`, TensorLLM-style `0.3%`.
- Remaining gate:
  - Run the final local verification pass and then audit whether any remaining
    non-Qwen paper table/figure still lacks a current replacement or accepted
    reproduced artifact.
## 2026-06-11 - Plan update: neighboring appendix wired into final paper bundle

- Scope update:
  - The same-protocol neighboring-baseline rerun is now wired into the current
    paper source and final artifact bundle.
  - The old safe-paper claim boundary that excluded unsupported neighboring
    rows has been replaced by a checked appendix table.
  - Patch-only/Reuse-only/Full anchor rows are now also represented in the paper
    source through Table `patchonly_anchors`.
- Implementation approach:
  - Added Table `neighboring_baselines` to
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.tex`.
  - Added Table `patchonly_anchors` to the same paper source using the final
    bundle's patch anchor values.
  - Extended `build_paper_artifact_bundle.py`,
    `finalize_scd_icml_full7405_bundle.py`, and
    `wait_finalize_scd_icml_full7405_bundle.sh` so final paper bundles include
    `neighboring_baselines.{csv,md,tex}` and
    `neighboring_baselines_manifest.json`.
  - Updated paper artifact checks to require the restored neighboring table,
    the patch-only anchor table, and forbid the superseded
    unsupported-neighboring exclusion text.
- Verification:
  - The final local bundle is
    `Experiment/results/paper_artifact_bundle_final_neighboring_checked_20260611_001`.
  - The recompiled paper PDF is
    `Publication/paper/scd-icml-current-safe-paper-2026-06-10.pdf`.
  - Local full unit suite, release readiness, release-preview build, paper
    artifact check, and LaTeX compilation passed.
  - The updated runtime finalizer/bundle scripts and tests were synced to `ser`
    and passed remote release readiness plus targeted paper-bundle tests.
- Next review focus:
  - The final non-Qwen completion audit now treats literal historical
    patch-budget/rank reruns as non-blocking because current generated
    replacements satisfy the trend-compatible reproduction criterion.
## 2026-06-11 - GitHub branch staging for current SCD-ICML release

- Scope: prepare a clean GitHub branch update for the current non-Qwen SCD-ICML open-source release state without merging into `main`.
- Assumptions: the active workspace root is not itself a Git repository; the unpacked `new-main/new-main` tree is the closest local GitHub `drongzzz0/new` baseline; the release-facing code lives under `Experiment/core_code/runtime_repo`.
- Constraints: do not mutate `main`; do not push unrelated archives, caches, large run outputs, or temporary directories; keep Qwen32B explicitly out of scope; preserve the current evidence boundary and dataset-specific route claims.
- Implementation approach: build a local staging repository from `new-main/new-main`, overlay current curated code/docs/results metadata from the active workspace, commit on a dedicated branch, then attempt to push that branch to GitHub. If GitHub authentication or HTTPS connectivity blocks pushing, leave the local branch ready for manual push.
- Outcome: created local staging branch `codex/scd-icml-open-release-20260611` under `tmp_github_branch_update/scd-new-branch-staging` with commit `5ae3ea8`. Local runtime verification passed with 45 unit tests and release readiness. Push to `git@github.com:drongzzz0/new.git` failed with `Permission denied (publickey)`, so `main` remains untouched and no remote branch was created from this machine.

## 2026-06-11 - Plan update: payload-bytes bandwidth simulation included

- Scope update:
  - Bandwidth-sensitivity evidence now covers mixed reuse+patch SCD routes, not
    only Patch-only and Reuse-only anchors.
  - The current artifact generation contract includes explicit `payload_bytes`
    and `payload_source` fields for latency-quality and bandwidth-sensitivity
    tables.
- Implementation approach:
  - Infer mixed-route payload bytes from same-dataset Patch-only/Reuse-only
    anchor payload per layer and each route's `patch_n/reuse_n`.
  - Preserve direct payload measurements as `payload_source=direct_anchor` and
    mark inferred mixed-route values as
    `estimated_from_patch_reuse_anchor_per_layer`.
  - Rebuild the checked paper artifact bundle under
    `Experiment/results/paper_artifact_bundle_final_payload_bytes_checked_20260611_001`.
- Verification:
  - Result-table unit test passed.
  - Paper artifact check passed for the payload-bytes bundle.
  - Release readiness passed.
  - Full runtime unit suite passed with `PYTHONPATH=src;.` from
    `Experiment/core_code/runtime_repo`.
- Next review focus:
  - If this bandwidth simulation is used in paper text, state that mixed-route
    payload bytes are inferred from anchor payload per layer unless direct
    per-route transfer-byte instrumentation is added later.

## 2026-06-14 - Plan update: guarded A800 Qwen32B reproduction path

- Scope: add a paper-claim-aligned Qwen-32B reproduction entry point without
  allowing missing checkpoints, VL checkpoints, CPU fallback, or partially idle
  GPUs to produce misleading evidence.
- Assumptions:
  - The final camera-ready paper should be treated as fixed unless another
    camera-ready window is confirmed later.
  - Qwen-32B must use text CausalLM base/instruct checkpoints, not the
    available Qwen2.5-VL-32B-Instruct checkpoint.
  - A800 jobs should use idle GPUs by default; if no clearly idle GPUs are
    available, the runner should monitor for sustained zero utilization before
    starting.
- Constraints:
  - Do not report Qwen-32B SCD quality or latency results unless the actual
    32B text model pair is present and the full guarded pipeline completes.
  - Do not silently use Qwen1.7B diagnostic scripts as Qwen-32B paper evidence.
  - Do not use CPU fallback for Qwen-32B training/evaluation.
- Implementation approach:
  - Add A800 model, reuse-bank, patch-bank, and strict-eval config templates
    under `Experiment/core_code/runtime_repo/configs/`.
  - Add `scripts/preflight_qwen32_a800.py` to validate model paths, reject
    visual-language checkpoints, check required dataset files, and select only
    GPUs with sustained zero utilization and enough free memory.
  - Add `scripts/run_a800_scd_icml_qwen32_repro.sh` as the single guarded
    capture/train/profile/route/eval wrapper. The wrapper activates the `cache`
    conda environment when available and stops at preflight failure.
  - Fix native residual patch training so the consumer model is loaded with the
    requested dtype instead of always forcing float32, which is required for
    Qwen-32B memory feasibility.
- Current outcome:
  - Static validation passed locally and on A800.
  - The A800 preflight correctly blocked execution because the configured
    Qwen2.5-32B text base/instruct model paths were not present and the current
    machine did not provide two sustained-idle GPUs with at least 70 GB free
    memory.
- Next action:
  - Materialize the correct text Qwen2.5-32B base/instruct checkpoints at the
    configured paths or update the config to their real paths, then rerun the
    guarded A800 wrapper when enough idle GPU memory is available.

## 2026-06-18 - SCD-vLLM Plugin Doctor Preflight

- Scope: add a static preflight step for the plug-and-play SCD-vLLM package so
  users can validate route, checkpoint, model, and vLLM environment assumptions
  before launching a serving or injection run.
- Goal:
  Reduce release risk by making strict no-fallback/no-recompute validation
  explicit and machine-readable through a `scd-vllm doctor` command.
- Constraints:
  Keep the command non-destructive and lightweight. It should not require GPU
  execution or model loading by default. A stricter local-model check is exposed
  as an opt-in flag for deployment environments.
- Design choices:
  Treat fallback and recompute layers as hard failures, require patch+reuse
  layers to cover all transformer layers without overlap, and warn rather than
  fail when vLLM is unavailable unless the user is preparing a real serving
  environment.
- Implementation approach:
  Add `src/scd_vllm_plugin/doctor.py`, expose it through the `scd-vllm` CLI,
  document the preflight flow, and cover the route checks with focused tests.
- Verification plan:
  Run release readiness, focused `scd_vllm_plugin` tests, and the full runtime
  unittest suite before committing. Do not push the branch until the user
  explicitly authorizes it.

### 2026-06-18 update - bundle artifact preflight

- Change:
  Extend `scd-vllm doctor` from raw reuse/patch bank preflight to also accept
  `--bundle-dir` for already reconstructed SCD cache bundles.
- Rationale:
  The A800 open-release artifacts already contain reconstructed reuse+patch
  bundles. Validating these artifacts directly is closer to the plug-and-play
  deployment path than forcing users to provide raw training banks.
- Verification:
  Local focused tests now cover bundle route-hash validation, and A800
  `cache` verified real `reuse19_patch13` and `patch20_reuse12` bundles plus a
  `patch20_reuse12` vLLM ExampleConnector smoke run.

### 2026-06-18 update - raw-bank end-to-end plugin evidence

- Change:
  Verified the full raw-bank path for the current SCD-vLLM plugin: raw
  CMRC-trained reuse and patch checkpoints, strict route preflight,
  reconstructed bundle export, ExampleConnector storage conversion, and vLLM
  cache-load smoke.
- Rationale:
  Bundle-only smoke tests prove deployment artifact loading, but the public
  release also needs evidence that users can start from trained reuse/patch
  banks and produce a usable plugin artifact without a forked vLLM codebase.
- Verification:
  A800mql with `conda activate cache` completed
  `a800_scd_vllm_raw_bank_run_20260618_025602`: doctor pass, strict
  `reuse19_patch13` route, `bundle_type=reuse_patch`, `aligned_tokens=224`,
  `payload_bytes=29,362,944`, and plugin smoke `pass=True`.
- Scope boundary:
  Greedy continuation equality is still false for this aggressive route, so the
  result is integration evidence rather than a standalone quality claim.

### 2026-06-18 update - patch20/reuse12 raw-bank vLLM demo evidence

- Change:
  Added `configs/routes/scd_icml_mistrallite_to_mistral7_patch20_reuse12_cmrc200.json`
  to the release tree and validated it through the full raw-bank SCD-vLLM
  export-convert-inject path.
- Rationale:
  The earlier raw-bank run proved the pipeline on the more aggressive
  `reuse19_patch13` route, but that route failed the greedy continuation check.
  The open-source plugin needs a demo route that is strict reuse+patch-only,
  has dataset-level QA evidence, and passes reconstructed vLLM smoke checks.
- Verification:
  A800mql with `conda activate cache` completed
  `a800_scd_vllm_raw_bank_patch20_run_20260618_032217`: doctor pass, strict
  `patch20_reuse12` route, `bundle_type=reuse_patch`, `aligned_tokens=224`,
  `payload_bytes=29,362,944`, plugin smoke `pass=True`,
  `greedy_1tok_match=True`, and `greedy_ntok_match=True`.
- Scope boundary:
  This remains integration evidence for the vLLM plugin path. Dataset quality
  claims should continue to cite the full CMRC2018 and HotpotQA F1/EM runs.

### 2026-06-18 update - full WikiText-2 evidence for Mistral release routes

- Change:
  Added full WikiText-2 strict evaluation evidence for `patch20_reuse12` and
  `patch26_reuse6`, and added the `patch26_reuse6` route config to the release
  tree as the quality-first reference route.
- Rationale:
  The release status already had full CMRC2018 and HotpotQA evidence, but the
  Mistral three-dataset matrix lacked current full WikiText-2 evidence for the
  same strict no-fallback/no-recompute routes.
- Verification:
  On `ser`, GPUs `0,1,2,3` were idle for 60 seconds before launch.
  `wikitext2_patch20_patch26_full_current_20260618_034812` completed with
  `763` samples for both routes. `patch20_reuse12` produced SCD NLL/PPL
  `2.381655 / 10.822796` with TTFT speedup `1.741x`; `patch26_reuse6`
  produced SCD NLL/PPL `2.206730 / 9.085953` with TTFT speedup `1.642x`.
- Scope boundary:
  WikiText-2 quality should be discussed via NLL/PPL, not F1/EM. Both strict
  routes improve TTFT but worsen PPL relative to Full Consumer.

### 2026-06-18 update - release evidence boundary audit

- Change:
  Added `docs/scd-open-release-evidence-audit-2026-06-18.md` and aligned
  `OPEN_SOURCE_RELEASE_MANIFEST.md` with the current route roles:
  `patch20_reuse12` as the vLLM demo route and `patch26_reuse6` as the
  quality-first reference route.
- Rationale:
  The manifest still contained older `reuse19_patch13` and full7405
  consolidation language that could be confused with the current open-release
  claim boundary.
- Verification:
  `check_release_readiness.py` now requires the manifest to mention
  `patch20_reuse12`, `current vLLM plugin demo route`, `patch26_reuse6`,
  `quality-first reference route`, and `NLL/PPL`.
- Scope boundary:
  The release audit is a claim-boundary artifact, not a new experiment. It
  keeps Qwen3-1.7B gray as portability/risk evidence rather than a speedup
  claim.

### 2026-06-18 update - patch26/reuse6 vLLM smoke closure

- Change:
  Ran the full raw-bank-to-vLLM smoke path for
  `patch26_reuse6` on A800mql under `conda activate cache`, copied the local
  result artifacts into
  `Experiment/results/a800_scd_vllm_raw_bank_patch26_run_20260618_044521/`,
  and updated the release evidence docs.
- Rationale:
  The previous claim boundary treated `patch26_reuse6` as quality-first but
  not separately vLLM-smoke-validated. That left an unnecessary release gap for
  the route that is best on CMRC2018 and WikiText-2 quality metrics.
- Verification:
  The patch26 run passed doctor, plugin pipeline, SCD injection,
  token equality, and greedy one-token / multi-token equality. GPU `0,1`
  utilization stayed at `0` for the 60-second pre-launch monitor.
- Design choice:
  Keep `patch20_reuse12` as the higher-Reuse/speed demo route, but allow
  `patch26_reuse6` to be presented as the quality-first vLLM-validated route.

### 2026-06-18 update - release preview vLLM packaging gate

- Change:
  Fixed `scripts/build_release_tree.py` so the release preview includes
  `docs/scd_vllm_plugin_mvp.md`, `src/scd_vllm/`, and
  `src/scd_vllm_plugin/`; added `tests/test_release_tree_script.py` to build
  the preview and assert the vLLM plugin surface is present.
- Rationale:
  The source tree readiness check passed, but the generated preview omitted
  the vLLM plugin files required by the `scd-vllm` console script. This made
  the release artifact weaker than the checked source tree.
- Verification:
  `python scripts/build_release_tree.py --out release_preview/scd_icml_runtime_release --force`
  now passes preview readiness and leaves `0` `__pycache__` directories.
  Full local tests now report `94 passed`.
- Scope boundary:
  This is a packaging/release gate fix; it does not change route metrics or
  dataset quality claims.

### 2026-06-18 update - A800 cache env probe and vLLM status report gate

- Change:
  Recorded the A800mql `conda activate cache` vLLM environment probe, refreshed
  `reports/scd_vllm_status.md`, and made the release preview/readiness gate
  include both vLLM reports.
- Rationale:
  The vLLM implementation spec requires a concrete environment probe and a
  status report with phase results. Keeping these files inside the release
  preview makes the open-source package auditable without relying on chat
  history or remote-only logs.
- Verification target:
  Rebuild `release_preview/scd_icml_runtime_release`, run
  `scripts/check_release_readiness.py`, and rerun the local test suite.
- Scope boundary:
  This changes packaging and evidence traceability only. It does not change
  model weights, route metrics, or dataset quality claims.

### 2026-06-18 update - vLLM phase-script release surface

- Change:
  Added the `scripts/scd_vllm/run_00_*.sh` through `run_06_*.sh` phase
  commands and `configs/scd_vllm_mvp.yaml` to the release preview/readiness
  surface, and refreshed `reports/scd_vllm_repo_audit.md`.
- Rationale:
  `reports/scd_vllm_status.md` tells users to run the vLLM phase scripts, so a
  release preview that omitted those scripts was not self-contained. The repo
  audit also still named a stale `src/scd_vllm/scd_injection.py` file and
  described TrueKV as an unresolved blocker, which conflicted with the current
  A800 smoke evidence.
- Verification:
  Source readiness and release-preview readiness both pass. The generated
  release preview contains all eight `scripts/scd_vllm` phase wrappers,
  includes `configs/scd_vllm_mvp.yaml`, and has `0` `__pycache__` directories.
  Full local tests remain `94 passed`.
- Scope boundary:
  This is release-surface hardening only. It does not alter route selection,
  model artifacts, or dataset metric values.

### 2026-06-18 update - Qwen3-1.7B plain-vLLM compatibility evidence

- Change:
  Added A800mql Qwen3-1.7B self -> self vLLM env-probe and plain-generation
  baseline artifacts under
  `Experiment/results/a800_qwen17_vllm_baseline_20260618_0530/`, and updated
  the vLLM status/evidence docs.
- Rationale:
  The main SCD route evidence is still MistralLite -> Mistral-7B. The release
  goal also asks for multiple models, so the next useful low-risk step was to
  prove that the current A800 `conda activate cache` vLLM environment can load
  and generate with a second model family.
- Verification:
  GPUs `0,1,2,5` stayed at utilization `0` for 60 seconds before launch. The
  Qwen env probe reported KV compatibility `True`, and the plain vLLM baseline
  generated 16-token outputs at mean `77.092046` tokens/s with 1-token proxy
  time `0.013509s`.
- Scope boundary:
  This is compatibility evidence only. It is not a Qwen SCD Reuse/Patch
  route-quality result and not a Qwen speedup claim.

### 2026-06-18 update - Qwen3-1.7B gray SCD-vLLM plugin evidence

- Change:
  Copied the validated Qwen3-1.7B-gray consumer model and the strict Qwen
  reuse/patch banks from ser to the A800 local project disk, then ran
  `scd-vllm doctor` and the raw-bank-to-vLLM plugin smoke on A800 with
  `conda activate cache`.
- Rationale:
  The previous Qwen evidence was only a plain vLLM self-model baseline. The
  release goal needs evidence that the plug-and-play SCD-vLLM adapter can also
  carry a second model family's reconstructed `reuse_patch` bundle.
- Verification:
  The Qwen route passed strict doctor checks with patch/reuse `6/22` and
  fallback/recompute `0/0`. The plugin pipeline passed with
  `adapter_smoke_pass=True`, `scd_injection_pass=True`,
  `bundle_token_match=True`, `same_token_ids=True`, and
  `loaded_generation_ok=True`.
- Scope boundary:
  This is adapter portability evidence. It still does not support a Qwen
  speedup claim, because the strict Qwen full-dataset runtime evidence remains
  cautionary on latency.

### 2026-06-18 update - Qwen SCD-vLLM batch benchmark gate

- Change:
  Added a public `scd-vllm batch-benchmark` command, tests, release-readiness
  checks, and A800 evidence for the Qwen3-1.7B -> Qwen3-1.7B-gray
  `paper_scd_reuse_patch_k6` bundle.
- Rationale:
  The plug-and-play vLLM story had doctor and single-prefix smoke evidence but
  no batched command evidence. The new gate checks the repeated-prefix
  `ExampleConnector` load path without turning it into a production serving or
  multi-tenant safety claim.
- Verification:
  On A800mql with `conda activate cache`, GPUs `3,4,6,7` were busy during the
  idle check; GPUs `0,1,2,5` stayed idle, so the benchmark used GPU `0`.
  `batch_size=128`, `num_batches=3`, expected requests `384`;
  `benchmark_pass=True`, `load_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
- Scope boundary:
  The measured SCD load path reached `2000.185735` tokens/s versus plain vLLM
  baseline `6514.906148` tokens/s, so this is integration/overhead evidence
  only. It must not be advertised as a Qwen throughput speedup.

### 2026-06-18 update - add vLLM prefix-cache control to Qwen batch gate

- Change:
  Extended `scd-vllm batch-benchmark` with
  `--include-prefix-cache-baseline`, then reran the Qwen3-1.7B ->
  Qwen3-1.7B-gray repeated-prefix benchmark on A800.
- Rationale:
  Plain vLLM alone is an incomplete baseline for a repeated-prefix workload.
  vLLM's native prefix caching is the correct built-in control for this serving
  shape, so the release artifact should report plain vLLM, prefix-cache vLLM,
  and SCD load together.
- Verification:
  On A800mql with `conda activate cache`, GPUs `3,4,6,7` stayed at utilization
  `0` for 60 seconds; GPU `7` was used. `batch_size=128`, `num_batches=3`,
  expected requests `384`; `benchmark_pass=True`,
  `prefix_cache_generation_ok=True`, `load_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
- Scope boundary:
  Prefix caching reached `7016.957157` tokens/s versus plain vLLM
  `6189.640815` tokens/s. SCD load reached only `1968.496077` tokens/s, or
  `0.280534`x of prefix-cache throughput. This reinforces the release boundary:
  current SCD-vLLM is a plug-in integration prototype and overhead-risk
  artifact, not a throughput-improving serving connector.

### 2026-06-18 update - add external preloaded SCD connector prototype

- Change:
  Added `scd_vllm.preloaded_connector.SCDPreloadedConnector`, exposed the
  connector through vLLM's `kv_connector_module_path` mechanism, and extended
  `scd-vllm batch-benchmark` with `--kv-connector`,
  `--kv-connector-module-path`, and `--connector-cache-device`.
- Rationale:
  The previous Qwen batch benchmark showed the disk-backed `ExampleConnector`
  path was much slower than vLLM baselines. The next useful implementation step
  was to remove repeated safetensors reads for identical prefixes inside a
  batch without modifying the installed vLLM package.
- Verification:
  On A800mql with `conda activate cache`, GPUs `3,4,6,7` stayed at utilization
  `0` for 60 seconds; GPU `7` was used. The preloaded connector run passed with
  `benchmark_pass=True`, `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `load_generation_ok=True`, `prefix_cache_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
- Result boundary:
  SCDPreloaded reached `4294.997427` tokens/s versus the previous disk-backed
  SCD load `1968.496077` tokens/s, a material connector improvement. It remains
  below plain vLLM `6194.909625` tokens/s and vLLM prefix-cache
  `6786.831734` tokens/s, so the release can claim a functional external
  connector prototype and reduced overhead, not serving speedup.

### 2026-06-18 update - gate Mistral SCDPreloadedConnector evidence

- Change:
  Ran the external `SCDPreloadedConnector` batch benchmark on the two main
  MistralLite -> Mistral-7B release routes and added those artifacts to release
  readiness.
- Rationale:
  The prior preloaded-connector gate was Qwen-only, which was useful for
  portability but not enough for the main release path. The release gate should
  verify the Mistral routes that carry the paper-facing code path.
- Verification:
  On A800mql with
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  GPUs `0` and `1` stayed at utilization `0` across repeated checks.
  `patch20_reuse12` ran on GPU `0`; `patch26_reuse6` ran on GPU `1`.
  Both produced `benchmark_pass=True`, `kv_connector=SCDPreloadedConnector`,
  `kv_connector_module_path=scd_vllm.preloaded_connector`,
  `load_generation_ok=True`, `prefix_cache_generation_ok=True`,
  `bundle_token_match=True`, and `storage_ready=True`.
- Result boundary:
  `patch20_reuse12` reached SCDPreloaded `2870.665243` tok/s versus plain
  vLLM `953.216251` tok/s and vLLM prefix-cache `4083.576373` tok/s.
  `patch26_reuse6` reached SCDPreloaded `2823.938822` tok/s versus plain
  vLLM `963.863263` tok/s and vLLM prefix-cache `4327.769903` tok/s. This
  supports plug-in integration and a repeated-prefix speedup over plain vLLM,
  but native vLLM prefix caching remains the stronger static-prefix baseline.

### 2026-06-18 update - add SCDPreloadedConnector cache stats gate

- Change:
  Added worker-side cache hit/miss JSON stats to
  `scd_vllm.preloaded_connector.SCDPreloadedConnector`, aggregated those stats
  into `scd-vllm batch-benchmark` metrics, and made the batch benchmark
  automatically disable vLLM's hybrid KV cache manager when the preloaded
  connector is used.
- Rationale:
  The previous benchmark evidence was timing-only. For release review, it is
  stronger to prove actual connector-side tensor reuse and to avoid a fragile
  user-facing requirement to remember `--disable-hybrid-kv-cache-manager` for
  vLLM `0.18.0`.
- Verification:
  On A800mql with
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  GPUs `0` and `1` stayed at utilization `0` for 60 seconds before launch.
  The first load attempt failed with vLLM's HMA error; after the automatic HMA
  fix, the load stages passed and produced stats artifacts.
- Result boundary:
  `patch20_reuse12` now has `benchmark_pass=True`, SCDPreloaded
  `2773.109399` tok/s, plain vLLM `969.093202` tok/s, prefix-cache
  `4103.775535` tok/s, `cache_hits=16352`, `cache_misses=32`, and
  `cache_hit_rate=0.998047`. `patch26_reuse6` now has `benchmark_pass=True`,
  SCDPreloaded `2657.446443` tok/s, plain vLLM `988.035039` tok/s,
  prefix-cache `4233.686187` tok/s, `cache_hits=16352`,
  `cache_misses=32`, and `cache_hit_rate=0.998047`. This strengthens the
  plug-in/cache-reuse evidence but still does not support a general speedup
  claim over vLLM native prefix caching.

### 2026-06-18 update - add official-vLLM launch spec contract

- Change:
  Added `scd-vllm launch-spec` and the corresponding Python API
  `build_preloaded_launch_spec`. The command emits JSON containing official
  `vllm.LLM(**llm_kwargs)` inputs, `SCDPreloadedConnector` KV transfer config,
  optional `SCD_PRELOADED_STATS_DIR`, bundle metadata, storage summary, and
  explicit safety boundaries.
- Rationale:
  The previous plugin interface proved export/smoke/batch behavior but still
  required users to manually assemble vLLM kwargs and remember that vLLM
  `0.18.0` rejects the connector if HMA is enabled. A launch-spec contract
  makes the adapter more plug-and-play while preserving the no-vLLM-fork
  design.
- Verification:
  Added unit tests for the launch-spec API and CLI JSON output. Targeted tests
  `tests/test_scd_vllm_plugin.py` and `tests/test_scd_vllm_batch_benchmark.py`
  passed, and release readiness passed.
- Scope boundary:
  The launch spec is a single-tenant trusted-bundle contract. It does not claim
  multi-tenant safety, arbitrary request routing, or production lifecycle
  management.

### 2026-06-18 update - add official-vLLM launch-smoke execution gate

- Change:
  Added `scd-vllm launch-smoke` and the Python API `run_launch_smoke`. The CLI
  now consumes `launch_spec.json`, launches official `vllm.LLM`, runs a
  trusted-prefix generation batch, and records token-hash matching plus
  `SCDPreloadedConnector` stats. The CLI parent process recollects connector
  stats after the vLLM child process exits so JSON artifacts include final
  cache-hit evidence.
- Rationale:
  `launch-spec` proved the configuration contract, but release users still
  needed a direct command that demonstrates the JSON is actually consumable by
  official vLLM. This closes that evidence gap without claiming production
  serving, multi-tenant safety, or dynamic request routing.
- Verification:
  Local tests passed with `PYTHONPATH=.;src pytest tests -q` (`105 passed`) and
  `python scripts\check_release_readiness.py`. On A800, using
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`, GPU
  `2` stayed at utilization `0` for 60 seconds before launch. The run used
  `CUDA_VISIBLE_DEVICES=2` and produced `launch_smoke_pass=True`,
  `generation_ok=True`, `bundle_token_match=True`, `request_count=8`,
  `output_tokens=128`, and `tokens_per_second=596.609039`.
- Result boundary:
  The evidence is for a single trusted static prefix from the Mistral
  `patch26_reuse6` SCD bundle. It confirms official-vLLM consumption and
  connector-side cache reuse (`cache_hits=480`, `cache_misses=32`,
  `cache_hit_rate=0.9375`), but it remains a smoke/integration artifact rather
  than a dataset-quality or production-serving benchmark.

### 2026-06-18 update - extend official-vLLM launch-smoke to Qwen

- Change:
  Ran `scd-vllm launch-smoke` on the Qwen3-1.7B -> Qwen3-1.7B-gray
  `paper_scd_reuse_patch_k6` bundle and added a release readiness gate for
  both Mistral and Qwen launch-smoke artifacts.
- Rationale:
  Mistral had the strongest official-vLLM launch-spec consumption evidence,
  but the open-source release goal explicitly requires multiple models. Qwen
  already had plugin smoke and batch-benchmark evidence; adding launch-smoke
  closes the same official-vLLM-consumption gap for the second model family.
- Verification:
  On A800 with
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  GPUs `3/4/6/7` were sampled for 60 seconds and stayed at utilization `0`.
  The final run used `CUDA_VISIBLE_DEVICES=7` and produced
  `launch_smoke_pass=True`, `generation_ok=True`, `bundle_token_match=True`,
  `request_count=128`, `output_tokens=2048`, and
  `tokens_per_second=6291.929930`.
- Result boundary:
  This confirms official-vLLM consumption and connector-side cache reuse on a
  second model (`cache_hits=7140`, `cache_misses=28`,
  `cache_hit_rate=0.996094`). It remains a trusted static-prefix integration
  artifact, not a production-serving or dataset-quality benchmark.

### 2026-06-18 update - require Qwen worker-side preloaded-cache stats

- Change:
  Re-ran the Qwen3-1.7B -> Qwen3-1.7B-gray repeated-prefix
  `scd-vllm batch-benchmark` using `SCDPreloadedConnector`, collected final
  worker-side connector stats, and switched the release readiness gate from
  the older `preloadedbench` artifact to
  `a800_qwen17_scd_vllm_preloadedstats_20260618_cache`.
- Rationale:
  The previous Qwen batch artifact showed benchmark correctness and throughput
  but did not gate on the actual worker cache-hit file. For a publishable
  plug-and-play vLLM adapter, release evidence should prove that the public
  connector path is active inside the vLLM worker, not merely that generation
  completed.
- Verification:
  On A800mql with
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate cache`,
  GPUs `3/4/6/7` were busy, and GPUs `0/1/2` stayed at utilization `0` for
  60 seconds. The final run used `CUDA_VISIBLE_DEVICES=2` and produced
  `benchmark_pass=True`, `load_generation_ok=True`,
  `bundle_token_match=True`, `storage_ready=True`, and
  `prefix_cache_generation_ok=True`.
- Result boundary:
  This strengthens Qwen portability evidence and records worker-side
  `SCDPreloadedConnector` cache reuse (`cache_hits=14308`,
  `cache_misses=28`, `cache_hit_rate=0.998047`). It remains a trusted
  repeated-prefix integration benchmark, not a Qwen serving-speed claim,
  because SCDPreloaded is still slower than plain vLLM and native prefix-cache
  on this microbenchmark.

### 2026-06-18 update - add release gate for multidataset quality evidence

- Change:
  Extended `scripts/check_release_readiness.py` so release validation reads
  branch-local full-dataset quality JSON artifacts instead of relying only on
  documentation snippets. The gate now checks Mistral CMRC2018 dev, HotpotQA
  dev, WikiText-2 test, and the Qwen3-1.7B gray three-dataset risk aggregate.
- Rationale:
  The open-source branch needs to remain publishable after future cleanup and
  refactoring. If a result directory is deleted, sample counts drift, or the
  Qwen risk evidence is accidentally framed as a speedup result, readiness
  should fail before release.
- Verification:
  `python Experiment\core_code\runtime_repo\scripts\check_release_readiness.py`
  passed after the new gate was added. The checked quality artifacts have
  sample counts `3219` for CMRC2018 dev, `7405` for HotpotQA dev, and `763`
  for WikiText-2 test.
- Result boundary:
  Mistral QA evidence remains the positive F1/EM path. WikiText-2 is checked
  with NLL/TTFT and remains a risk-boundary dataset. Qwen3-1.7B gray remains
  portability/risk evidence, not a release-facing speedup claim.

### 2026-06-18 update - add programmatic official-vLLM launch helper

- Change:
  Added `examples/scd_vllm_official_launch.py` to the release surface. The
  helper reads a `scd-vllm launch-spec` JSON, performs a no-GPU `--dry-run`
  validation, or executes the trusted-prefix official-vLLM smoke path from
  Python.
- Rationale:
  The CLI already had `launch-spec` and `launch-smoke`, but a publishable
  plug-and-play adapter should also show users the programmatic integration
  shape. A dry-run mode lets users validate connector kwargs, storage metadata,
  token/route identity fields, and safety flags before occupying a GPU.
- Verification:
  Added `tests/test_scd_vllm_official_launch_example.py` and extended
  `tests/test_release_tree_script.py` so the helper is tested and included in
  the release preview. Targeted tests and release readiness passed.
- Result boundary:
  This is still a single-tenant trusted-bundle launch helper. It does not add
  multi-tenant serving, dynamic routing, request authentication, or privacy
  isolation.

### 2026-06-18 update - gate train/eval split evidence

- Change:
  Extended the release-readiness plan so `check_release_readiness.py` checks
  train/eval split evidence in addition to route strictness, multidataset
  quality metrics, vLLM connector evidence, and release-tree packaging.
- Rationale:
  The release branch should be open-source safe even after result cleanup. The
  gate now fails if the branch-local evidence no longer proves that CMRC2018
  train-200 is used as the calibration/training bank while CMRC2018 dev,
  HotpotQA dev, and WikiText-2 test are used for evaluation.
- Implementation boundary:
  The new checks read route metadata, strict CMRC summary metadata, WikiText-2
  run configs, and Qwen strict README/config/route artifacts. It does not
  broaden claims beyond the existing trusted static-prefix SCD-vLLM boundary.
- Verification:
  `PYTHONPATH=.;src pytest tests -q` passed with `106 passed`;
  `python scripts\check_release_readiness.py` passed; `git diff --check`
  passed; and `python scripts\build_release_tree.py --out ..\..\..\release_preview --force`
  built a preview whose own readiness check passed.

### 2026-06-18 update - promote high-batch Mistral preloaded evidence

- Change:
  Replaced the release-facing Mistral repeated-prefix microbenchmark evidence
  with the new A800mql high-batch run
  `Experiment/results/a800_mistral_d1ce508_b256_shm_20260618_0938/`.
  `check_release_readiness.py` now requires the new `patch20_reuse12` and
  `patch26_reuse6` artifacts, `batch_size>=256`,
  `expected_requests>=1280`, worker-side `SCDPreloadedConnector` stats, cache
  hit rate at least `0.99`, and SCDPreloaded throughput faster than the plain
  vLLM baseline.
- Rationale:
  Earlier b128 evidence proved connector function and worker cache hits, but
  the release branch should use the strongest available plug-in evidence before
  code cleanup and branch publication. The new run also avoids repeated NAS
  model reads by loading Mistral from `/dev/shm`, while keeping the execution
  environment as A800mql `conda activate cache`.
- Result boundary:
  This remains a trusted repeated-prefix static-bundle microbenchmark. It
  supports plug-in integration, worker-side cache reuse, and measured overhead
  boundaries. It does not claim a general serving speedup over native vLLM
  prefix caching, dynamic routing, multi-tenant isolation, or privacy safety.

### 2026-06-18 update - repair Dr. Claw workspace path metadata

- Change:
  Updated `instance.json`, `.pipeline/docs/research_brief.json`, and
  `.pipeline/tasks/tasks.json` so their project-local paths no longer point to
  the stale `/Users/weimingming/dr-claw/scd-new` workspace. The current branch
  staging workspace is now the canonical local path in these metadata files.
- Rationale:
  The SCD-ICML open-release branch should not carry stale `scd-new` absolute
  paths in the Dr. Claw control files. Keeping these paths current prevents
  future pipeline reads, task artifacts, and stage transitions from resolving
  to the wrong project.
- Verification:
  Parsed all three JSON files and validated every mapped `instance.json` path
  as a descendant of the current staging workspace.

### 2026-06-18 update - enforce launch-spec safety boundary

- Change:
  Added an execution-time safety validator for SCD-vLLM launch specs and
  extended release readiness to verify the stored Mistral and Qwen
  `launch_spec.json` artifacts. A launch spec is accepted only if it declares
  `multi_tenant_safe=false`, `dynamic_request_routing=false`, and
  `trusted_bundle_required=true`.
- Rationale:
  KV-cache sharing and cache reuse are not safe to present as general
  multi-tenant serving without explicit isolation engineering. The open-source
  branch should fail closed when a launch spec omits that boundary, instead of
  relying only on prose warnings.
- Verification:
  Targeted SCD-vLLM plugin tests and release readiness passed after adding the
  validator and artifact checks.

### 2026-06-18 update - pin A800 runtime environment to conda cache

- Change:
  Set the active A800 execution policy for remaining SCD-vLLM and SCD-ICML
  reproduction work to use `source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate cache`. Do not create or place Python virtual environments on
  NAS for A800 jobs.
- Rationale:
  The `cache` conda environment is already installed under the A800 user home
  area and contains the required GPU runtime stack. Keeping the environment off
  NAS avoids slow or fragile interpreter/package access during model loading and
  vLLM execution.
- Verification:
  Aline validation on `A800mql` confirmed `CONDA_DEFAULT_ENV=cache`,
  `CONDA_PREFIX=/home/c101-800-mql/.conda/envs/cache`,
  Python `3.12.12`, PyTorch `2.10.0+cu128`, Transformers `5.3.0`, Datasets
  `4.8.5`, and vLLM `0.18.0`.

### 2026-06-18 update - harden release scripts against stale paths and split leakage

- Change:
  Updated ser/A800 helper scripts so their defaults no longer point to the old
  `/disk_n/c101-3-1` or `/home/student_t` accounts. A800 scripts now default
  to `/home/c101-800-mql/maqianli/...` paths and use the already activated
  Python environment by default. Runtime repo ser scripts now default to
  `/disk_n/c101-3-mql/venvs/A800` and, where applicable, derive `ROOT` from
  the checked-out runtime repo.
- Change:
  Removed the risky pattern where helper scripts copied
  `cmrc2018_train.json` into a `cmrc2018_dev.json` slot. Train/calibration
  prompt generation now calls `load_cmrc2018_samples(..., split="train")`,
  while strict eval stages use a separate `EVAL_ROOT` populated with the real
  `cmrc2018_dev.json`.
- Change:
  Extended `check_release_readiness.py` so release shell scripts fail if they
  contain stale host paths or copy CMRC train data into the dev split.
- Rationale:
  The open-source branch should not make it easy to run future SCD-vLLM or
  SCD-ICML experiments on stale accounts or with train/eval leakage. This is a
  release-readiness fix, not a new result claim.
- Verification:
  `python Experiment\\core_code\\runtime_repo\\scripts\\check_release_readiness.py`
  passed. `PYTHONPATH=.;src pytest tests -q` passed with `107 passed`.
  `git diff --check` passed. A script search found no remaining
  `cmrc2018_train.json` to `cmrc2018_dev.json` copy pattern.

### 2026-06-18 update - restore canonical pipeline directories

- Change:
  Added `.gitkeep` placeholders for the empty canonical directories referenced
  by `instance.json`: `Survey/references`, `Survey/reports`,
  `Ideation/ideas`, `Ideation/references`, `Experiment/datasets`,
  `Promotion/homepage`, `Promotion/audio`, and `Promotion/video`.
- Rationale:
  The Dr. Claw project metadata should resolve to existing in-workspace
  directories before future research or release tasks write artifacts.
- Verification:
  Validated every mapped `instance.json` path as both a descendant of the
  current staging workspace and an existing filesystem path.

### 2026-06-18 update - tighten A800 cache-env release hygiene

- Change:
  Confirmed A800 commands should use `conda activate cache`, with Python
  resolving to `/home/c101-800-mql/.conda/envs/cache/bin/python`. Removed
  remaining old-account paths from user-facing README/report text and model
  config files. Removed two internal historical artifact migration scripts
  that referenced stale `/home/student_t` and `/nas/mql` locations. Removed the
  A800 Qwen32 helper's automatic NAS run-output fallback so the default stays
  under `/home/c101-800-mql/maqianli`.
- Rationale:
  The open-source branch should not expose stale private machine paths, should
  not encourage NAS-backed environments or hidden migration scripts, and should
  be verifiable from a clean release preview.
- Verification:
  A fresh release preview was built under `Experiment/results` and its
  readiness check passed. A user-visible preview scan found no stale
  `/home/student_t`, `/disk_n/c101-3-1`, `/nas/mql`, train-to-dev copy pattern,
  or forbidden vLLM safety claim. The preview test suite passed with
  `107 passed`. The temporary preview directories were removed after
  verification to avoid committing generated source copies.

### 2026-06-18 update - Qwen32B validation uses A800 storage for heavy artifacts

- Change:
  For the live A800 Qwen2.5-32B validation run, keep the Conda environment on
  local disk via `conda activate cache`, but place the downloaded Qwen2.5-32B
  base checkpoint under `/home/c101-800-mql/nas/models/Qwen2.5-32B` and direct
  heavy run outputs to `/home/c101-800-mql/nas/SCD-ICML/runs`. Removed the
  temporary `/dev/shm/scd_models/Qwen2.5-32B` partial copy after confirming the
  NAS model was complete enough for preflight.
- Rationale:
  `/home` on A800 had only about 23G free, while the storage mount had about
  64T free. Qwen32B checkpoint materialization and full outputs should not
  consume the development disk or `/dev/shm`. This is a run-time placement
  decision for the current validation job, not a change to the public release
  defaults.
- Verification:
  Verified the NAS Qwen2.5-32B directory contains `30` files, including `17`
  safetensors and a `config.json` with `model_type=qwen2` and architecture
  `Qwen2ForCausalLM`. Verified `/dev/shm/scd_models/Qwen2.5-32B` was removed.

### 2026-06-18 update - make A800 Qwen32 storage placement the release default

- Change:
  Updated the A800 Qwen32 release script so the default run parent is a
  writable storage path, not `/home` or `/dev/shm`. The script now exports
  `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `MODELSCOPE_CACHE`, and
  `TORCH_HOME` under `/home/c101-800-mql/nas/SCD-ICML/model_cache` unless the
  caller overrides `MODEL_CACHE_ROOT`. Updated the Qwen2.5-32B base/instruct
  model config template to point to `/home/c101-800-mql/nas/models`.
- Rationale:
  `/home` on A800 had only about 23G free, which is not enough for reliable
  32B model materialization or repeated run artifacts. Open-source helper
  scripts should default to the storage mount so future runs do not silently
  fill the development disk.
- Verification:
  Parsed the updated JSON model configs successfully and validated the updated
  shell scripts with `bash -n` on A800. Started copying the existing
  Qwen2.5-32B-Instruct checkpoint from `/home/cuihanshuai/models` to
  `/home/c101-800-mql/nas/models/Qwen2.5-32B-Instruct` before launching a new
  Qwen32 run from the current account.

### 2026-06-18 update - add Qwen32-capable vLLM tensor-parallel launch path

- Change:
  Extended the official-vLLM launch-spec path with `tensor_parallel_size` so
  `scd-vllm launch-spec` and `build_preloaded_launch_spec` can emit
  `vllm.LLM(**llm_kwargs)` configurations for multi-GPU consumer models. Added
  a vLLM dataset-evaluation entry point for CMRC2018 so Qwen32B can be checked
  on a real full dataset instead of a single-prompt smoke.
- Rationale:
  Qwen2.5-32B does not fit reliably in the current strict single-device HF
  runtime while the A800 cards have resident memory from other processes. The
  releasable vLLM integration must be able to express TP4/TP2 model loading,
  and the result workflow needs full-dataset F1/EM artifacts rather than only
  connector smoke metrics.
- Verification:
  Local targeted tests passed with `9 passed` for
  `tests/test_scd_vllm_plugin.py` and
  `tests/test_scd_vllm_official_launch_example.py`. On A800 `conda activate
  cache`, a minimal launch-spec check produced
  `launch_spec_pass=true`, `tensor_parallel_size=4`, and a NAS Qwen32 model
  path in `llm_kwargs`. A Qwen2.5-32B-Instruct CMRC2018 dev full vLLM baseline
  was launched on A800 GPUs `0,1,2,3` after a 60-second zero-utilization
  preflight.

### 2026-06-18 update - correct Qwen32B Instruct baseline prompt format

- Change:
  Added optional chat-template prompting and extra task instruction support to
  the vLLM dataset evaluation path. Re-ran the Qwen2.5-32B-Instruct CMRC2018
  dev Full Consumer baseline with tokenizer chat template and a short-answer
  instruction.
- Rationale:
  The first Qwen32B Full Consumer baseline used a plain completion prompt and
  produced full-sentence answers, giving a misleadingly low EM. Instruct-model
  baselines must use their expected chat format before being used as a quality
  comparison target for SCD.
- Verification:
  Local prompt-formatting and vLLM plugin tests passed with `11 passed`.
  The A800 chat-template full run completed on all `3219` CMRC2018 dev samples
  with `F1=0.9098067394` and `EM=0.7086051569`, compared with
  `F1=0.4156507535` and `EM=0.0043491768` for the plain completion diagnostic
  run.

### 2026-06-18 update - harden Qwen32 storage-backed cache placement

- Change:
  Extended the A800 Qwen32 wrapper's storage-backed cache policy beyond model
  weights and Hugging Face hub cache. The wrapper now also creates and exports
  `HF_ASSETS_CACHE`, `XDG_CACHE_HOME`, and `VLLM_CACHE_ROOT` under
  `/home/c101-800-mql/nas/SCD-ICML/model_cache` by default, and records these
  paths in `run_config.json`.
- Rationale:
  A800 `/home` is effectively full (`1.8T` total, `21G` free at the latest
  check), while the NAS mount has about `64T` free. Qwen32B runs must not let
  model downloads, vLLM caches, or framework asset caches fall back to the
  development disk.
- Verification:
  Confirmed the active Qwen32 model config and vLLM baseline logs load
  Qwen2.5-32B checkpoints from `/home/c101-800-mql/nas/models`. Synced the
  updated wrapper to the A800 runtime repo and validated it with remote
  `bash -n`; the remote model config resolves both Qwen32B directories under
  `/home/c101-800-mql/nas/models`.

### 2026-06-18 update - extend official-vLLM full-dataset baselines

- Change:
  Extended `scd_vllm.dataset_eval` from a CMRC2018-only evaluator to a shared
  full-dataset official-vLLM baseline entry point for `cmrc2018`, `hotpotqa`,
  `hotpotqa_compressed`, `wikitext2`, and `wikitext-2`. The summary now records
  the dataset's primary metric: F1/EM for QA datasets and PPL/NLL for
  WikiText-2. README now includes a full-dataset official-vLLM baseline
  command.
- Rationale:
  The release branch needs a Qwen32B-capable vLLM path that can check more than
  CMRC2018 and more than a smoke prompt. WikiText-2 must not be presented as a
  pure F1 task; its quality gate is PPL/NLL.
- Verification:
  Local targeted tests passed with `14 passed` across
  `tests/test_scd_vllm_dataset_eval.py`, `tests/test_scd_vllm_plugin.py`, and
  `tests/test_scd_vllm_official_launch_example.py`. Release readiness passed.
  On A800mql `conda activate cache`, the evaluator import checks passed and
  vLLM reported version `0.18.0`. Complete HotpotQA and WikiText-2 data were
  copied to `/home/c101-800-mql/nas/SCD-ICML/data`; sample counts are `7405`
  for HotpotQA and `763` for WikiText-2.
- Current run:
  Started the A800mql TP4 full-dataset Qwen2.5-32B-Instruct queue at
  `/home/c101-800-mql/nas/SCD-ICML/runs/qwen32_vllm_hotpot_wiki_a800mql_nopip_20260618_143845`.
  It selected `CUDA_VISIBLE_DEVICES=0,1,2,3` after a 60-second idle preflight
  and began loading the model from NAS.

### 2026-06-18 update - require storage-backed caches for large-model vLLM runs

- Change:
  Added an explicit `--cache-root` policy to the official-vLLM baseline and
  full-dataset evaluation path. Large-model jobs should point this at NAS or
  another storage mount, not at `/home` or a development disk.
- Rationale:
  A800 `/home` has only about `18G` free while Qwen32B model weights occupy
  about `62G`. Even when the model path itself is local-to-NAS, tokenizer,
  Hugging Face, ModelScope, Torch, XDG, and vLLM caches can otherwise fall back
  to default home-cache locations.
- Implementation approach:
  `scd_vllm.cache_env.configure_model_cache` now creates the standard cache
  subdirectories under a user-supplied storage root and sets the corresponding
  environment variables before vLLM/tokenizer imports and model loading.
  The vLLM `download_dir` is also set to the storage-backed HF hub cache.
- Verification:
  Targeted vLLM tests pass locally and the updated code imports successfully on
  A800 using the `cache` Conda environment's Python interpreter.

### 2026-06-18 update - Qwen32 official-vLLM Full Consumer baseline coverage

- Change:
  Completed and consolidated Qwen2.5-32B-Instruct Full Consumer baselines for
  CMRC2018, HotpotQA, and WikiText-2 through official vLLM TP4.
- Current reference rows:
  CMRC2018 dev `n=3219`, F1 `0.9098067394`, EM `0.7086051569`;
  HotpotQA dev `n=7405`, F1 `0.7459153089`, EM `0.5856853477`;
  WikiText-2 test `n=763`, avg NLL `1.9711257546`, PPL `7.1787534555`.
- Artifact:
  The consolidated table is in
  `Experiment/results/a800_qwen32_vllm_three_dataset_full_baselines_20260618/`.
- Boundary:
  These are Full Consumer reference baselines only. They prove the release
  branch can load and evaluate Qwen32 through official vLLM TP4 on three
  datasets, but they do not prove Qwen32 SCD reuse/patch quality or speedup.
