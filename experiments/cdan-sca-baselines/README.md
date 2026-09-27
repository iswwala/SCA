# Existing-Method Baseline Experiments / 现有方法基线实验

Date / 日期: 2026-08-13

## Baseline Sufficiency Judgment / 基线充分性判断

**English:** The current baseline set is not yet sufficient for a CDAN-SCA paper claim. The repository contains useful baseline references and a legacy DANN/CDAN-like script, but it does not yet provide a unified, fair, reproducible comparison across existing methods under the same data splits and GE evaluation.

**中文：** 当前 baseline 还不足以支撑 CDAN-SCA 的论文主张。仓库中已有有用的基线参考代码和一个 legacy DANN/CDAN 风格脚本，但还没有在相同数据划分、相同训练条件和相同 GE 评估口径下统一比较现有方法。

**Important correction / 重要修正：**

**English:** CDAN-SCA is the proposed method and must not be reported as a baseline. Variants such as "without entropy weighting" or "without conditional alignment" should be reported as ablations of the proposed method, not as baseline methods.

**中文：** CDAN-SCA 是我们提出的方法，不能作为 baseline 汇报。去掉熵权重、去掉条件对齐等变体应作为 proposed method 的消融实验，而不是 baseline 方法。

## Required Baselines / 必要基线

| Method / 方法 | Purpose / 目的 | Included? / 当前是否充分 |
|---|---|---|
| Source Only / 仅源域训练 | Measures raw cross-device transfer gap / 衡量原始跨设备迁移差距 | Partially / 部分具备 |
| DANN | Tests whether marginal adversarial alignment helps or hurts / 检验边缘对抗对齐是否有效或造成负迁移 | Legacy only / 仅 legacy 原型 |
| CDPA/MMD-style method | SCA-specific UDA baseline from related work / 侧信道领域已有 UDA 基线 | Reference code only / 仅参考代码 |
| AdaBN/statistical alignment | Recent simple adaptation baseline / 近期简单统计适应基线 | Missing / 缺失 |
| Official ASCAD CNN / ASCAD 官方 CNN | Standard supervised profiling backbone / 标准监督建模攻击主干 | Reference code only / 仅参考代码 |

## Proposed-Method Variants Are Ablations / 提出方法的变体属于消融

These are **not baselines**:

以下不是 baseline：

| Variant / 变体 | Role / 角色 |
|---|---|
| CDAN-SCA full / 完整 CDAN-SCA | Proposed method / 本文提出方法 |
| CDAN-SCA without entropy / 去掉熵权重 | Ablation / 消融 |
| CDAN-SCA without conditional module / 去掉条件模块 | Ablation; equivalent to testing marginal adversarial alignment in the same framework / 消融；等价于在同一框架中测试边缘对抗对齐 |
| Projection dimension variants / 投影维度变体 | Ablation/sensitivity / 消融或敏感性分析 |

## Immediate Experimental Goal / 立即实验目标

**English:** First build a fair smoke-test pipeline for existing baselines such as Source Only and DANN on one ASCAD source-target pair. CDPA/MMD and AdaBN should be added next. CDAN-SCA should be evaluated later in a separate proposed-method/ablation experiment.

**中文：** 首先在一个 ASCAD 源-目标配对上建立 Source Only 和 DANN 等现有基线的小规模 smoke-test 流程。随后补充 CDPA/MMD 和 AdaBN。CDAN-SCA 应在单独的 proposed-method/ablation 实验中评估。

## Canonical First Pair / 第一组标准配对

Source / 源域:

```text
data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5
Profiling_traces
```

Target / 目标域:

```text
data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5
Attack_traces
```

Fallback target / 备选目标域:

```text
data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5
Attack_traces
```

## Fairness Rules / 公平性规则

- Same backbone and classifier / 相同主干网络与分类器
- Same source trace count / 相同源域轨迹数
- Same target trace count for adaptation / 相同目标域适应轨迹数
- Same optimizer, batch size, and training epochs / 相同优化器、batch size 和训练轮数
- Same preprocessing / 相同预处理
- Same evaluation traces and target byte / 相同评估轨迹与目标字节
- Same random seed / 相同随机种子

## Method Switches / 方法开关

| Method / 方法 | Domain input / 域判别器输入 | Entropy weight / 熵权重 |
|---|---|---|
| Source Only | Disabled / 关闭 | Disabled / 关闭 |
| DANN | `f` | Disabled / 关闭 |
| CDPA/MMD | Moment/discrepancy alignment, implementation pending / 矩或分布差异对齐，待实现 | Disabled / 关闭 |
| AdaBN | Batch-normalization statistics adaptation, implementation pending / BN 统计量适应，待实现 | Disabled / 关闭 |

## Run Commands / 运行命令

Smoke run / 小规模冒烟实验:

```bash
python experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py --method source_only --epochs 1 --limit-source 512 --limit-target 512 --limit-eval 256
python experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py --method dann --epochs 1 --limit-source 512 --limit-target 512 --limit-eval 256
```

Full pilot / 完整试点:

```bash
python experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py --method source_only --epochs 10 --limit-source 5000 --limit-target 5000 --limit-eval 2000
python experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py --method dann --epochs 10 --limit-source 5000 --limit-target 5000 --limit-eval 2000
```

## Current Blocker / 当前阻塞

**English:** The current Python environment does not have `tensorflow` or `h5py`, so experiments cannot be executed in this session yet.

**中文：** 当前 Python 环境缺少 `tensorflow` 和 `h5py`，因此本轮暂时无法真正执行训练实验。

Install dependencies / 安装依赖:

```bash
uv pip install tensorflow h5py scikit-learn matplotlib numpy
```

or / 或：

```bash
pip install tensorflow h5py scikit-learn matplotlib numpy
```

After installation, rerun the smoke commands above.

安装完成后，重新运行上面的 smoke 命令。
