# 实验目录入口

项目实验按“一个研究问题 = 一个目录”组织。每个目录固定包含：

```text
<experiment>/
  README.md      # 假设、数据、指标、交付物和结论边界
  configs/       # 可复现参数
  scripts/       # 可执行入口
```

## 当前实验

| 目录                    | 用途                             | 状态                     |
| ----------------------- | -------------------------------- | ------------------------ |
| `cdan-sca-baselines/` | Source Only、DANN 等现有方法基线 | 已有 smoke/pilot         |
| `cdan-sca-diagnosis/` | 数据契约、评估链路和负迁移诊断   | 已完成第一轮             |
| `cdan-sca-ablation/`  | CDAN-SCA 内部消融                | 设计完成，待统一 trainer |
| `legacy/`             | 历史脚本，仅用于追溯             | 不作为新实验入口         |

实验输出统一写入 `outputs/results/<experiment>/`，模型写入 `outputs/models/<experiment>/`。原始数据只通过 `data/` 链接读取，不复制进仓库。
