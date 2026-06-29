"""面向 vLLM 的 SCD 集成辅助包。

这个包故意和 ``scd_repro`` 分开：``scd_repro`` 负责研究复现和离线
评测，``scd_vllm`` 负责把已经验证过的 SCD KV bundle 转成 vLLM 可读
形态，并提供环境检查、TrueKV oracle、baseline、injection smoke 和报告。
当前边界是可信静态 prefix 原型，不是通用多租户在线缓存系统。
"""

__all__ = [
    "bundle",
    "env_probe",
    "kv_spec",
    "metrics",
    "preloaded_connector",
    "repo_audit",
    "report",
]
