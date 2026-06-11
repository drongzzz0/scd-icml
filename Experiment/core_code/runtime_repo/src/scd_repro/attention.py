from __future__ import annotations

import numpy as np


def broadcast_kv_to_q_heads(kv: np.ndarray, num_attention_heads: int) -> np.ndarray:
    """将 KV 头广播到 Q 头，以支持 GQA 口径的注意力计算。

    输入:  [B, KVH, T, HD]
    输出: [B, QH, T, HD]
    """
    if kv.ndim != 4:
        raise ValueError(f"期望形状为 [B,KVH,T,HD]，实际为 {kv.shape}")
    _b, kvh, _t, _hd = kv.shape
    if num_attention_heads % kvh != 0:
        raise ValueError("num_attention_heads 必须是 num_key_value_heads 的整数倍")
    repeat = num_attention_heads // kvh
    return np.repeat(kv, repeats=repeat, axis=1)


def build_causal_swa_mask(seq_len: int, window: int) -> np.ndarray:
    """布尔掩码 [T,T]；值为真表示该位置可被注意到。"""
    q = np.arange(seq_len)[:, None]
    k = np.arange(seq_len)[None, :]
    causal = k <= q
    left = np.maximum(0, q - window + 1)
    within_window = k >= left
    return causal & within_window


def masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"形状不匹配: {a.shape} vs {b.shape}")
    if mask.ndim != 2:
        raise ValueError("掩码必须为 [T,T]")
    valid = mask.astype(bool)
    # 支持形状为 [..., T, T] 的张量
    while valid.ndim < a.ndim:
        valid = valid[np.newaxis, ...]
    diff = (a - b) ** 2
    denom = float(np.sum(valid))
    if denom == 0:
        return 0.0
    return float(np.sum(diff * valid) / denom)

