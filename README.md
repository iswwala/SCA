# SCA-UDA

面向跨设备迁移场景的神经网络侧信道分析研究工作区。

本项目主要研究 AES 侧信道分析，重点关注建模攻击模型以及无监督域适应方法在不同设备、不同实现、不同探针位置和不同轨迹分布之间的迁移能力。

## 研究目标

- 复现并比较建模侧信道分析和跨设备/域适应方向的代表性基线方法。
- 分析不同方法在 AES 泄露场景下的优势、适用重点和局限性。
- 研究适用于跨设备侧信道分析的改进模型结构和训练方法。
- 分离管理数据集、外部基线、实验代码和生成结果，以便追踪实验过程并复现实验结果。

## 目录结构

```text
SCA_UDA/
  baselines/              # 外部基线或复现的基线方法
    ascad/                # ASCAD 官方代码及相关资源
    cdpa/                 # 跨设备建模攻击基线
    official_scripts/     # 早期官方/基线预处理脚本

  src/                    # 当前维护的研究框架
    framework/            # 当前 CDAN/监督训练框架

  experiments/            # 实验入口、配置和历史试验
    legacy/               # 为保留实验追踪信息而保存的历史脚本和配置

  data/                   # 本地大规模数据集；不要随意提交或移动

  outputs/                # 实验生成的结果文件
    results/              # 图表、GE 曲线、日志和比较汇总
    models/               # 实验导出的训练权重和检查点

  references/             # 论文和阅读材料
    papers/               # PDF 论文及方法参考资料

  docs/                   # 研究笔记、方法总结和实验记录
```

## 新工作放置位置

开展新实验前，请先阅读 [`experiments/README.md`](experiments/README.md) 中的当前实验登记信息，以及 [`docs/experiments/README.md`](docs/experiments/README.md) 中的实验协议和结果链接。

- 新的模型组件：`src/framework/models/`
- 新的训练器或训练流程：`src/framework/trainers/`
- 新的数据集加载和预处理工具：`src/framework/utils/`
- 新的实验配置：`src/framework/configs/` 或 `experiments/<experiment_name>/configs/`
- 一次性探索脚本：`experiments/<experiment_name>/`
- 最终图表、GE 曲线和结果表：`outputs/results/<experiment_name>/`
- 保存的模型权重和检查点：`outputs/models/<experiment_name>/`
- 论文笔记和方法比较：`docs/literature/`

## 已完成实验记录

- `experiments/cdan-sca-diagnosis/` 包含跨设备诊断实验协议和不依赖深度学习环境的契约检查脚本；生成的结果分析位于 `outputs/results/cdan-sca-diagnosis/`。
- 当前记录的 smoke 实验使用 `~/venvs/tf` 环境，源域为 `ASCAD.h5/Profiling_traces`，目标域为 `ASCAD_desync50.h5/Attack_traces`，源域和目标域各使用 512 条轨迹，使用 256 条轨迹进行评估，batch size 为 64，训练 2 个 epoch，随机种子为 42。该实验只用于检查流程，不能作为最终性能结论。

## 建议的研究记录方式

每个主要实验单独建立一个目录：

```text
experiments/
  2026-xx-cross-device-new-arch/
    README.md             # 研究假设、源/目标设备、泄露模型和评价指标
    configs/
    scripts/
```

为每个方法或论文保留一份简要笔记：

```text
docs/literature/
  dann.md
  cdan.md
  cdpa.md
  ascad.md
```

每份笔记建议包含：

- 核心思想
- 适用场景
- 方法优势
- 方法局限
- 在当前 AES 跨设备场景下可以改进的方向

## 数据管理规范

由于数据集可能较大，`data`可保留在自己主机上，不要将原始轨迹、生成的数据集、模型检查点或大规模结果压缩包提交到版本库。
