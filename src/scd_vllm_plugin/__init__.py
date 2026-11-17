"""``scd-vllm`` 命令行 facade。

外部用户只需要通过这里暴露的 engine/config 调用 vLLM 原型流程；
底层 bundle、转换、doctor 和 launch smoke 仍分别保留在各自模块中。
"""

from __future__ import annotations

from .engine import SCDVLLMConfig, SCDVLLMEngine, build_preloaded_launch_spec, run_launch_smoke

__all__ = ["SCDVLLMConfig", "SCDVLLMEngine", "build_preloaded_launch_spec", "run_launch_smoke"]
