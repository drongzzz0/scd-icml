---
marp: true
theme: default
paginate: true
size: 16:9
title: Revisiting SCD on Similar Mistral Pairs
footer: Strict Reproduction of SCD on Mistral Pairs
style: |
  section {
    font-family: "Aptos", "Helvetica Neue", Helvetica, Arial, sans-serif;
    background:
      radial-gradient(circle at top right, #dbeafe 0%, rgba(219, 234, 254, 0) 22%),
      linear-gradient(180deg, #f8fbff 0%, #ffffff 30%, #ffffff 100%);
    color: #0f172a;
    padding: 54px 64px 72px 64px;
    font-size: 26px;
    line-height: 1.28;
  }
  h1, h2, h3 {
    color: #0f172a;
    margin: 0 0 0.35em 0;
    font-weight: 750;
    letter-spacing: -0.02em;
  }
  h1 { font-size: 1.65em; }
  h2 { font-size: 1.08em; color: #2563eb; }
  p, li, table { font-size: 0.92em; }
  strong { color: #0f172a; }
  code {
    background: #eff6ff;
    color: #1d4ed8;
    padding: 0.08em 0.28em;
    border-radius: 6px;
    font-size: 0.92em;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.4em;
  }
  th {
    background: #dbeafe;
    color: #0f172a;
    font-weight: 700;
  }
  th, td {
    border: 1px solid #bfdbfe;
    padding: 10px 12px;
    text-align: left;
  }
  .lead {
    justify-content: center;
    text-align: left;
  }
  .lead h1 {
    font-size: 2.05em;
    max-width: 11em;
  }
  .subtitle {
    font-size: 1.08em;
    color: #334155;
    margin-top: 0.15em;
  }
  .support {
    color: #475569;
    font-size: 0.8em;
    margin-top: 0.9em;
  }
  .badges {
    display: flex;
    gap: 12px;
    margin-top: 22px;
    flex-wrap: wrap;
  }
  .badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.02em;
  }
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
    align-items: start;
  }
  .card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #dbeafe;
    border-radius: 18px;
    padding: 18px 20px;
  }
  .card h3 {
    margin: 0 0 0.35em 0;
    font-size: 0.95em;
    color: #1d4ed8;
  }
  .claimbox {
    margin-top: 18px;
    padding: 14px 18px;
    border-radius: 16px;
    background: #eff6ff;
    border-left: 8px solid #2563eb;
    color: #1e293b;
  }
  .warn {
    color: #991b1b;
    background: #fef2f2;
    border-left: 8px solid #b91c1c;
  }
  .ok {
    color: #0f172a;
    background: #eff6ff;
    border-left: 8px solid #2563eb;
  }
  .metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 18px;
  }
  .metric {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 14px 16px;
  }
  .metric .k {
    color: #64748b;
    font-size: 0.66em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .metric .v {
    display: block;
    margin-top: 0.2em;
    font-size: 1.16em;
    font-weight: 800;
    color: #0f172a;
  }
  .small {
    font-size: 0.72em;
    color: #475569;
  }
  .tight li { margin: 0.22em 0; }
  .route {
    font-size: 0.7em;
    line-height: 1.38;
  }
  .appendix h1 { color: #1e293b; }
---

<!-- _class: lead -->

# Revisiting SCD on Similar Mistral Pairs

<div class="subtitle">What survives after a strict no-fallback reimplementation?</div>

<div class="badges">
  <span class="badge">baseline</span>
  <span class="badge">reuse-only</span>
  <span class="badge">patch-only</span>
  <span class="badge">hybrid</span>
</div>

<div class="claimbox ok">
This deck uses only the currently validated strict-stop forward artifact.
Reverse-direction results are intentionally excluded until the strict rerun is revalidated under the same contract.
</div>

<div class="support">
Question: after removing fallback leakage, aligning TTFT accounting, and enforcing strict stop rules, does the SCD trade-off still survive on similar Mistral-family pairs?
</div>

---

# Problem Setup

<div class="grid2">
  <div class="card">
    <h3>Offline chain</h3>
    <ul class="tight">
      <li><code>capture_pair</code> on 200 calibration prefixes</li>
      <li><code>train_reuse</code> for layerwise reuse translator</li>
      <li><code>train_patch</code> for all-layer patch-only anchor</li>
      <li><code>search</code> for no-fallback hybrid route</li>
    </ul>
  </div>
  <div class="card">
    <h3>Online chain</h3>
    <ul class="tight">
      <li><code>encode</code> producer prefix</li>
      <li><code>transfer</code> hidden or KV payload</li>
      <li><code>rebuild / patch</code> consumer cache</li>
      <li><code>decode_first</code> aligned first token boundary</li>
    </ul>
  </div>
</div>

<div class="metric-row">
  <div class="metric">
    <span class="k">Task Quality</span>
    <span class="v">F1 / EM</span>
  </div>
  <div class="metric">
    <span class="k">Latency</span>
    <span class="v">TTFT / total ms</span>
  </div>
  <div class="metric">
    <span class="k">Systems Cost</span>
    <span class="v">payload / breakdown</span>
  </div>
</div>

<div class="small">
The target is a TTFT-quality trade-off, not TTFT alone. Every mode is compared against the same consumer-native full-prefill baseline.
</div>

---

# What SCD Claims

The paper-level target is <strong>simultaneous speedup and near-native quality</strong>, not compression in isolation.

| Paper anchor | TTFT ratio target | Quality retention target |
|---|---:|---:|
| Qwen-family anchor | 0.484 | 0.967 |
| MistralLite -> Mistral-7B at 200 Gbps | 0.377 | 0.969 |

<div class="claimbox ok">
Operational meaning: roughly <strong>2.0x to 2.65x</strong> lower TTFT while keeping quality very close to the native consumer.
</div>

---

# Why Strict Reproduction Matters

<div class="grid2">
  <div class="card">
    <h3>Loose evaluation can overstate gains</h3>
    <ul class="tight">
      <li>search can continue after replay quality already failed</li>
      <li>fallback or recompute can leak consumer-native information</li>
      <li>TTFT boundaries can differ across methods</li>
    </ul>
  </div>
  <div class="card">
    <h3>Strict evaluation fixes the contract</h3>
    <ul class="tight">
      <li>stop immediately when replay has no quality-passing candidate</li>
      <li><code>fallback = []</code></li>
      <li><code>recompute = []</code></li>
      <li>measure all methods at the same first-decode boundary</li>
    </ul>
  </div>
</div>

<div class="claimbox warn">
If these issues are not fixed, an apparent win can be methodological leakage rather than a real state-transfer gain.
</div>

---

# Fixed Evaluation Contract

| Method | Definition |
|---|---|
| <code>baseline</code> | consumer full prefill with aligned first decode |
| <code>reuse-only</code> | layerwise reuse, actual transfer timing, no fallback |
| <code>patch-only</code> | all-layer MLP patch, no reuse scaffold, no fallback, no recompute |
| <code>hybrid</code> | patch-only anchor plus greedy reuse search under strict replay F1 gate |

<div class="grid2">
  <div class="card">
    <h3>Locked settings</h3>
    <ul class="tight">
      <li><code>xkv_group</code>, <code>group_size = 1</code></li>
      <li><code>rank_k = rank_v = 128</code></li>
      <li><code>xnorm_nativekv_shared_residual_mlp_v1</code></li>
      <li>patch anchor = all 32 layers</li>
    </ul>
  </div>
  <div class="card">
    <h3>Strict hybrid rule</h3>
    <ul class="tight">
      <li>relative <code>F1 drop &lt;= 5%</code> versus patch-only</li>
      <li>block size <code>1</code>, replay top-k <code>5</code></li>
      <li>selector = <code>F1 first, TTFT second</code></li>
      <li>stop at first replay step with zero quality-pass</li>
    </ul>
  </div>
</div>

---

# Forward Strict Result

## MistralLite -> Mistral-7B

| Method | F1 | EM | TTFT ms |
|---|---:|---:|---:|
| baseline | 0.5969 | 0.250 | 103.62 |
| reuse-only | 0.0141 | 0.000 | 33.52 |
| patch-only | 0.6285 | 0.285 | 53.93 |
| hybrid | 0.6072 | 0.250 | 46.18 |

<div class="metric-row">
  <div class="metric">
    <span class="k">Hybrid vs baseline</span>
    <span class="v">2.24x lower TTFT</span>
  </div>
  <div class="metric">
    <span class="k">Hybrid vs patch-only</span>
    <span class="v">-7.75 ms TTFT</span>
  </div>
  <div class="metric">
    <span class="k">Quality budget</span>
    <span class="v">within 5% F1 gate</span>
  </div>
</div>

<div class="claimbox ok">
Interpretation: <code>reuse-only</code> is fast but unusable; <code>patch-only</code> is the quality anchor; the strict hybrid is a real no-fallback trade-off point, not a fallback artifact.
</div>

---

# Hybrid Route Anatomy

<div class="grid2">
  <div class="card">
    <h3>Selected route</h3>
    <ul class="tight">
      <li>reuse layers: <strong>14</strong></li>
      <li>patch layers: <strong>18</strong></li>
      <li>fallback layers: <strong>0</strong></li>
      <li>recompute layers: <strong>0</strong></li>
    </ul>
  </div>
  <div class="card">
    <h3>Stop criterion outcome</h3>
    <ul class="tight">
      <li>step 14 replay quality-pass count: <strong>2</strong></li>
      <li>step 15 replay quality-pass count: <strong>0</strong></li>
      <li>strict search must stop at the start of step 15</li>
    </ul>
  </div>
</div>

<div class="claimbox ok">
Final route is the last route before replay quality fully collapses. That is why the valid hybrid keeps 14 reuse layers instead of the older over-extended route.
</div>

<div class="route">
Reuse layers: 2, 3, 4, 8, 9, 10, 20, 22, 23, 24, 25, 26, 27, 30<br/>
Patch layers: 0, 1, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 28, 29, 31
</div>

---

# Why Hybrid Beats Patch-Only on TTFT

| Component | patch-only ms | hybrid ms |
|---|---:|---:|
| encode | 0.81 | 0.76 |
| transfer | 12.70 | 12.22 |
| rebuild | 0.00 | 1.47 |
| patch_infer | 18.48 | 10.19 |
| decode_first | 21.95 | 21.54 |
| total TTFT | 53.93 | 46.18 |

<div class="grid2">
  <div class="card">
    <h3>What changes</h3>
    <ul class="tight">
      <li>hybrid pays a small <code>rebuild</code> cost</li>
      <li>but materially cuts <code>patch_infer</code></li>
    </ul>
  </div>
  <div class="card">
    <h3>What stays similar</h3>
    <ul class="tight">
      <li><code>encode</code> and <code>transfer</code> barely change</li>
      <li><code>decode_first</code> stays aligned by construction</li>
    </ul>
  </div>
</div>

---

# What We Exclude From the Final Claim

## Mistral-7B -> MistralLite

<div class="claimbox warn">
The old reverse-direction result was produced by an earlier fallback-based route and is <strong>not valid</strong> for this deck.
</div>

<div class="grid2">
  <div class="card">
    <h3>Discarded path</h3>
    <ul class="tight">
      <li>older <code>Q2 / fallback</code> chain</li>
      <li>not comparable to the strict contract above</li>
      <li>cannot be mixed into rebuttal headline claims</li>
    </ul>
  </div>
  <div class="card">
    <h3>Current stance</h3>
    <ul class="tight">
      <li>reverse direction remains unresolved in this deck</li>
      <li>only same-chain strict results should enter rebuttal text</li>
      <li>until revalidated, treat reverse claims as excluded</li>
    </ul>
  </div>
</div>

---

# Takeaways

<div class="metric-row">
  <div class="metric">
    <span class="k">Validated</span>
    <span class="v">forward strict positive result</span>
  </div>
  <div class="metric">
    <span class="k">Fragile</span>
    <span class="v">reuse-only quality</span>
  </div>
  <div class="metric">
    <span class="k">Required</span>
    <span class="v">strict accounting</span>
  </div>
</div>

<div class="claimbox ok">
Current rebuttal-safe message: SCD-style gains survive only in a narrower, direction-sensitive, and more tightly controlled form than the original broad claim suggests.
</div>

<div class="small">
Use only these points in the spoken summary:
1. the strict contract changes conclusions,
2. the forward Mistral direction still admits a valid hybrid point,
3. reverse-direction claims remain excluded until a same-contract rerun is validated.
</div>

---

<!-- _class: appendix -->

# Appendix: Exact Strict-Stop Numbers

| quantity | value |
|---|---:|
| patch-only F1 | 0.6285 |
| allowed absolute F1 drop | 0.03143 |
| minimum replay-passing F1 | 0.59711 |
| hybrid F1 | 0.6072 |
| hybrid TTFT ms | 46.18 |
| TTFT delta vs patch-only | -7.75 |
| F1 delta vs patch-only | -0.02132 |

<div class="small">
Source artifact: <code>Experiment/analysis/mistrallite_to_mistral7_multilayer_strictstop_step15_20260407.md</code>
</div>
