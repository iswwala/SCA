# CDAN-SCA Proposed Method and Ablations / CDAN-SCA 提出方法与消融实验

Date / 日期: 2026-08-13

## Purpose / 目的

**English:** This experiment group is for the proposed CDAN-SCA method and its internal ablations. These variants should not be reported as external baselines.

**中文：** 本实验组用于 CDAN-SCA 提出方法及其内部消融。这些变体不应作为外部 baseline 汇报。

## Variants / 变体

| Variant / 变体 | Conditional input / 条件输入 | Entropy weight / 熵权重 | Role / 角色 |
|---|---|---|---|
| CDAN-SCA full / 完整 CDAN-SCA | `T(f,g) = f outer g` with random projection / 随机投影条件表示 | `w = 1 - H(g)/log(256)` | Proposed method / 提出方法 |
| w/o entropy / 去掉熵权重 | Enabled / 启用 | Disabled / 关闭 | Ablation / 消融 |
| w/o conditional input / 去掉条件输入 | Disabled, use `f` / 关闭，使用 `f` | Optional off / 通常关闭 | Ablation against DANN-like marginal alignment inside our framework / 与框架内 DANN 式边缘对齐对比 |
| projection dimension sweep / 投影维度扫描 | Enabled / 启用 | Enabled / 启用 | Sensitivity / 敏感性分析 |

## Reporting Rule / 汇报规则

**English:** In the paper tables, existing methods should appear under "Baselines"; CDAN-SCA and its variants should appear under "Ours/Ablations".

**中文：** 论文表格中，现有方法放在 "Baselines"；CDAN-SCA 及其变体放在 "Ours/Ablations"。

## Minimum Comparison Layout / 最小对比布局

Baselines / 基线:
- Source Only
- DANN
- CDPA/MMD if reproduced
- AdaBN if implemented

Ours and ablations / 我们的方法与消融:
- CDAN-SCA full
- CDAN-SCA w/o entropy
- CDAN-SCA w/o conditional input
- Projection dimension variants
