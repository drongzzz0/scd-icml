from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..io_utils import save_json
from ..oracle_diag import build_eval_config, parse_anchor_pairs, parse_csv_ints, plan_oracle_diag_runs
from .common import load_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行 RS-SCD m=1 oracle 诊断矩阵")
    p.add_argument("--mode", choices=["single_anchor", "full_suffix", "dual_anchor"], default="single_anchor")
    p.add_argument("--models-config", required=True)
    p.add_argument("--eval-config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--reuse-ckpt", required=True)
    p.add_argument("--patch-ckpt", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--layers", default="", help="逗号分隔的 patch layer 列表，例如 8,12,16")
    p.add_argument("--recompute-lengths", default="", help="逗号分隔的 k 列表，例如 0,4,8,12")
    p.add_argument("--anchor-pairs", default="", help="dual-anchor 模式使用，例如 8+14,10+16")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-length", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bandwidth-gbps", type=float, default=200.0)
    p.add_argument("--quant-group-size", type=int, default=128)
    p.add_argument("--producer-device", default="")
    p.add_argument("--consumer-device", default="")
    p.add_argument("--dtype", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _materialize_run(
    *,
    out_root: Path,
    run_name: str,
    route: dict,
    eval_cfg: dict,
) -> tuple[Path, Path, Path]:
    route_path = out_root / "routes" / f"{run_name}.json"
    eval_cfg_path = out_root / "eval_cfgs" / f"{run_name}.json"
    run_out = out_root / "runs" / run_name
    save_json(route_path, route)
    save_json(eval_cfg_path, eval_cfg)
    run_out.mkdir(parents=True, exist_ok=True)
    return route_path, eval_cfg_path, run_out


def _build_eval_cmd(
    *,
    python_exe: str,
    models_config: str,
    eval_cfg_path: Path,
    data_root: str,
    route_path: Path,
    reuse_ckpt: str,
    patch_ckpt: str,
    run_out: Path,
    limit: int,
    max_length: int,
    max_new_tokens: int,
    bandwidth_gbps: float,
    quant_group_size: int,
    producer_device: str,
    consumer_device: str,
    dtype: str,
    seed: int,
) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "scd_repro.cli.eval_cmrc_strict",
        "--models-config",
        models_config,
        "--eval-config",
        str(eval_cfg_path),
        "--data-root",
        data_root,
        "--route",
        str(route_path),
        "--reuse-ckpt",
        reuse_ckpt,
        "--patch-ckpt",
        patch_ckpt,
        "--out",
        str(run_out),
        "--limit",
        str(int(limit)),
        "--max-length",
        str(int(max_length)),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--bandwidth-gbps",
        str(float(bandwidth_gbps)),
        "--quant-group-size",
        str(int(quant_group_size)),
        "--seed",
        str(int(seed)),
    ]
    if producer_device:
        cmd.extend(["--producer-device", producer_device])
    if consumer_device:
        cmd.extend(["--consumer-device", consumer_device])
    if dtype:
        cmd.extend(["--dtype", dtype])
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    models_cfg = load_cfg(args.models_config)
    base_eval_cfg = load_cfg(args.eval_config)
    architecture = models_cfg.get("architecture", {}) if isinstance(models_cfg, dict) else {}
    compute = models_cfg.get("compute", {}) if isinstance(models_cfg, dict) else {}
    runtime = models_cfg.get("runtime", {}) if isinstance(models_cfg, dict) else {}
    num_layers = int(architecture.get("num_layers") or 0)
    if num_layers <= 0:
        raise ValueError("models-config 缺少 architecture.num_layers")

    layers = parse_csv_ints(args.layers)
    recompute_lengths = parse_csv_ints(args.recompute_lengths)
    anchor_pairs = parse_anchor_pairs(args.anchor_pairs) if str(args.anchor_pairs or "").strip() else []
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    producer_device = str(args.producer_device or compute.get("model_a_device") or runtime.get("device") or "")
    consumer_device = str(args.consumer_device or compute.get("model_b_device") or runtime.get("device") or "")
    dtype = str(args.dtype or runtime.get("dtype") or "")

    if args.mode in {"single_anchor", "full_suffix"} and not layers:
        raise ValueError(f"mode={args.mode} 需要提供 --layers")
    if args.mode == "single_anchor" and not recompute_lengths:
        raise ValueError("mode=single_anchor 需要提供 --recompute-lengths")
    if args.mode == "dual_anchor" and not anchor_pairs:
        raise ValueError("mode=dual_anchor 需要提供 --anchor-pairs")

    runs = plan_oracle_diag_runs(
        num_layers=num_layers,
        layers=layers,
        recompute_lengths=recompute_lengths or [0],
        limit=int(args.limit),
        plan_kind=str(args.mode),
        anchor_pairs=anchor_pairs or None,
    )

    manifest_rows: list[dict[str, object]] = []
    for run in runs:
        eval_cfg = build_eval_config(base_eval_cfg, runtime_eval_overrides=run.runtime_eval)
        route_path, eval_cfg_path, run_out = _materialize_run(
            out_root=out_root,
            run_name=run.name,
            route=run.route,
            eval_cfg=eval_cfg,
        )
        cmd = _build_eval_cmd(
            python_exe=str(args.python),
            models_config=str(args.models_config),
            eval_cfg_path=eval_cfg_path,
            data_root=str(args.data_root),
            route_path=route_path,
            reuse_ckpt=str(args.reuse_ckpt),
            patch_ckpt=str(args.patch_ckpt),
            run_out=run_out,
            limit=run.limit,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            bandwidth_gbps=float(args.bandwidth_gbps),
            quant_group_size=int(args.quant_group_size),
            producer_device=producer_device,
            consumer_device=consumer_device,
            dtype=dtype,
            seed=int(args.seed),
        )
        row = run.to_manifest_row()
        row.update(
            {
                "route_path": str(route_path),
                "eval_cfg_path": str(eval_cfg_path),
                "out_dir": str(run_out),
                "cmd": cmd,
            }
        )
        manifest_rows.append(row)
        if args.skip_existing and (run_out / "summary.json").exists():
            continue
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    manifest = {
        "models_config": str(args.models_config),
        "eval_config": str(args.eval_config),
        "data_root": str(args.data_root),
        "reuse_ckpt": str(args.reuse_ckpt),
        "patch_ckpt": str(args.patch_ckpt),
        "num_layers": num_layers,
        "mode": str(args.mode),
        "layers": layers,
        "recompute_lengths": recompute_lengths,
        "anchor_pairs": anchor_pairs,
        "runs": manifest_rows,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
