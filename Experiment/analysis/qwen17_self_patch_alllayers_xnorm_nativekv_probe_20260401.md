# Qwen3-1.7B Self-Self All-Layer XNorm-NativeKV Residual Probe

Date: 2026-04-01

## Goal

Test whether the "C-style" patch formulation can be extended from `layer 0 only` to `all 28 patch layers` in the cleanest possible setting:

- producer = consumer = `Qwen3-1.7B`
- strict no-fallback runtime
- patch route only
- residual modules initialized to identity so the probe isolates runtime and parameterization cost, not training quality

## Probe Design

Starting point:

- existing healthy checkpoint: `ckpt_patch_xnorm_nativekv_residual_mlp_all28_len1024_seed42`
- actual residual coverage in that checkpoint: `residual_layers = [0]`

New synthetic checkpoint:

- path on `ser`: `runs/qwen17_self_cmrc/ckpt_patch_xnorm_nativekv_residual_mlp_alllayers_identity_len1024_seed42`
- action: expand `xnorm_nativekv_residual_patch.<layer>.*` to all selected layers `0..27`
- initialization: `delta_h_head.weight = 0`, `delta_h_head.bias = 0` on every layer, so each residual module is identity at initialization

Strict eval attempt:

- route: `route_patchonly_all28_seed42.json`
- eval config: `eval_patchonly_diag50.json`
- models config: temporary no-fallback config with consumer on `cuda:3`, producer on `cuda:2`
- remote run dir: `runs/qwen17_self_cmrc/strict_diag_self_patchonly_xnorm_nativekv_residual_mlp_alllayers_all28_len1024_50_20260401_xnorm_nativekv_allresid_v3`

## What Happened

The all-layer probe did not reach the first sample.

Observed facts:

- source checkpoint `weights.pt`: `2,147,759,307` bytes
- synthetic all-layer checkpoint `weights.pt`: `47,447,056,167` bytes
- the eval process spent several minutes reading that single patch checkpoint before touching evaluation output
- `/proc/<pid>/io` showed roughly `1.57 GB / 10 s` sustained read during checkpoint load

The run then failed during patch checkpoint loading with CUDA OOM on the consumer GPU:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB.
GPU 3 has a total capacity of 23.53 GiB ... this process has 23.42 GiB memory in use.
Of the allocated memory 22.97 GiB is allocated by PyTorch
```

This failure happened before any prompt-level evaluation result was emitted.

## Interpretation

This probe answers the immediate engineering question behind "why not make all layers use the C-style method":

- under the current checkpoint format and runtime load path, turning `xnorm_nativekv_residual_mlp_v1` into a true all-layer method changes the memory regime completely
- even identity-initialized all-layer residual modules are too large to load on a stable `24 GB` consumer GPU
- so the blocker is not just training difficulty or downstream quality; it is already a deployment/runtime feasibility issue

In other words, the current default split is not arbitrary:

- `layer 0`: `xnorm_nativekv_residual_mlp_v1` is justified because it fixes the only broken layer and is loadable
- deeper layers: linear `direct_kv_knorm_v1` remains the practical default because the current all-layer residual parameterization does not fit on 4090D

## Consequence For Current Route Choice

For the current Qwen3-1.7B self-self line, "all layers use the C-style residual MLP" should not be treated as the default next run.

The realistic next options are:

1. make the residual route load lazily or in sharded/chunked form
2. shrink the residual module substantially
3. use residual MLP only on a small subset of problematic layers
4. if exact all-layer residual testing is still required, move the run to an `80 GB` A800 burst
