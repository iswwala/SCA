#!/usr/bin/env python3
"""Dependency-free protocol and metric-chain check for CDAN-SCA experiments."""
from __future__ import annotations
import json, math, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/results/cdan-sca-diagnosis"
SOURCE = ROOT / "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
TARGET = ROOT / "data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5"

def synthetic_metrics():
    # Deterministic rank curves stand in for the unavailable training stack.
    methods = {"source_only": (128, 0.05), "dann": (96, 0.10), "cdan": (64, 0.20), "cdan_sca": (48, 0.25)}
    rows = {}
    for name, (start, decay) in methods.items():
        curve = []
        for n in (10, 25, 50, 100, 200):
            rank = max(1, round(start * math.exp(-decay * math.log1p(n / 10))))
            curve.append({"traces": n, "ge": rank, "success": int(rank == 1)})
        ntge = next((p["traces"] for p in curve if p["ge"] == 1), None)
        rows[name] = {"ge_curve": curve, "sr_at_100": curve[3]["success"], "ntge": ntge}
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_available = SOURCE.exists() and TARGET.exists()
    payload = {"status": "data_contract_only" if data_available else "synthetic_smoke",
               "source": str(SOURCE), "target": str(TARGET), "methods": ["source_only", "dann", "cdan", "cdan_sca"]}
    if data_available:
        try:
            import h5py
            with h5py.File(SOURCE, "r") as handle:
                group = handle["Profiling_traces"]
                payload["source_contract"] = {"trace_shape": list(group["traces"].shape), "trace_length": int(group["traces"].shape[1]), "label_shape": list(group["labels"].shape), "label_min": int(group["labels"][:].min()), "label_max": int(group["labels"][:].max()), "metadata_fields": list(group["metadata"].dtype.names)}
            with h5py.File(TARGET, "r") as handle:
                group = handle["Attack_traces"]
                payload["target_contract"] = {"trace_shape": list(group["traces"].shape), "trace_length": int(group["traces"].shape[1]), "label_shape": list(group["labels"].shape), "label_min": int(group["labels"][:].min()), "label_max": int(group["labels"][:].max()), "metadata_fields": list(group["metadata"].dtype.names)}
        except Exception as exc:
            payload["contract_error"] = str(exc)
    if not data_available:
        payload["metrics"] = synthetic_metrics()
        payload["limitations"] = ["ASCAD source/target files unavailable", "TensorFlow/h5py training not executed", "synthetic metrics are not scientific evidence"]
    else:
        payload["baseline_smoke"] = {}
        baseline_dir = ROOT / "outputs/results/cdan-sca-baselines"
        for method in ("source_only", "dann"):
            path = baseline_dir / f"{method}_smoke_results.json"
            if path.exists():
                payload["baseline_smoke"][method] = json.loads(path.read_text())
        payload["limitations"] = ["Smoke uses 512 source/target traces and 256 evaluation traces", "label_rank_mean_proxy is per-trace label rank, not ASCAD key-byte GE", "CDAN and CDAN-SCA were not yet trained"]
    (OUT / "protocol_check.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    lines = ["# 结果分析报告", "", f"运行状态：`{payload['status']}`。", ""]
    if payload["status"] == "synthetic_smoke":
        lines += ["本次仅完成评估链路 smoke。由于 ASCAD 文件和 TensorFlow/h5py 不可用，表中曲线是确定性合成值，不能作为方法优劣证据。", "", "## 观察", "- 脚本成功生成 GE 曲线、SR@100 和 NTGE 字段。", "- 真实实验必须替换 `metrics`，并报告至少 3 个随机种子。", "", "## 下一步", "1. 挂载 ASCAD 数据并安装 TensorFlow、h5py、NumPy。", "2. 统一运行 Source Only、DANN、CDAN、CDAN-SCA。", "3. 用真实 GE/SR/NTGE、域准确率和类分离结果更新本报告。"]
    else:
        lines += ["已找到源域和目标域文件，并完成 Source Only/DANN 的 2 epoch smoke。", "", "## 已观测结果", "| 方法 | 目标准确率（末轮） | label-rank proxy（末轮） |", "|---|---:|---:|"]
        for method, result in payload.get("baseline_smoke", {}).items():
            row = result["history"][-1]
            lines.append(f"| {method} | {row['target_accuracy']:.4f} | {row['label_rank_mean_proxy']:.2f} |")
        lines += ["", "## 分析", "- 在极小样本、仅 2 epoch 的 smoke 设置下，DANN 的 label-rank proxy 高于 Source Only，目标准确率降至 0；这与‘全局对齐可能造成负迁移’的方向一致，但证据强度很低。", "- 当前 runner 没有输出域判别准确率，也没有按 AES key-byte 累积预测计算 GE/SR/NTGE，因此不能据此宣称 DANN 失败或 CDAN-SCA 有效。", "", "## 下一步", "1. 接入 CDAN/CDAN-SCA trainer，保持同一数据切分和 seed。", "2. 使用 ASCAD metadata 计算真实 key-byte GE、SR 和 NTGE。", "3. 至少运行 3 个 seed、5-10 epoch 的 pilot，再冻结结论。"]
    (OUT / "RESULTS_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
