from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_json_yaml(path: str | Path) -> dict[str, Any]:
    """在无额外依赖下加载 JSON 兼容的 YAML。

    我们有意将配置文件保持为严格 JSON 语法，使其既是合法 YAML，
    又可直接由标准库 json 解析。
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"未找到配置文件: {p}")
    raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ConfigError(f"配置文件为空: {p}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"配置必须为 JSON 兼容的 YAML，解析失败 {p}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"配置根节点必须是对象: {p}")
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8-sig")
