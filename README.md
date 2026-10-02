# CDAN-SCA

面向跨设备和跨分布场景的深度学习侧信道分析研究项目。

项目主要研究建模侧信道攻击中的无监督域适应问题：源域拥有带标签 profiling 轨迹，目标域只有无标签轨迹，最终使用 GE、key rank、成功率和攻击轨迹成本评价方法是否真正改善了密钥恢复。

## 当前研究主线

当前基础主线方法是 **CDAN-SCA**：

- 使用一维波形网络提取特征；
- 将特征和 256 类分类预测组成条件表示；
- 通过 GRL 和域判别器进行条件域对抗；
- 使用目标域熵或置信度控制不可靠样本的适应贡献。

当前优先探索的 SCA-specific 扩展是**校准感知 CDAN-SCA**，但它仍属于待实验确认的候选创新。

## 开始阅读

建议按以下顺序阅读：

1. [`docs/research/新人规划.md`](docs/research/%E6%96%B0%E4%BA%BA%E8%A7%84%E5%88%92.md)
2. [`docs/research/主方法.md`](docs/research/%E4%B8%BB%E6%96%B9%E6%B3%95.md)
3. [`docs/research/datasets_and_links.md`](docs/research/datasets_and_links.md)  数据集
4. [`docs/research/reference_experiment_analysis.md`](docs/research/reference_experiment_analysis.md)  此领域参考论文实验；参考文献在 docs/literature 下
5. [`experiments/README.md`](experiments/README.md)

## 目录结构

```text
SCA/
  baselines/                    # 外部基线、官方资源和历史复现材料
  src/                          # 当前维护的模型、训练器和工具
    framework/
      models/
      trainers/
      utils/
      configs/

  experiments/                  # 当前阶段实验提交入口
    00_common_e0/               # 共同最小端到端实验
    01_member_a_baselines/      # 基线实验
    02_member_b_method/         # 结构和主方法实验
    03_member_c_robustness/     # 评估、敏感性实验
    04_integrated_results/      # 后期汇总目录
    cdan-sca-baselines/         # 既有历史基线目录
    cdan-sca-diagnosis/         # 既有诊断和评估协议目录
    cdan-sca-ablation/          # 既有主方法消融说明目录
    legacy/                     # 历史代码，用于追溯

  outputs/                      # 后期正式结果归档入口
    results/                    # 经过复核的结果、表格和图表
    models/                     # 经过复核的模型和检查点

  data/                         # 本地数据，不提交原始轨迹
  references/                   # 论文和参考资料
  docs/                         # 研究规划、方法、综述和实验规范
```

## 实验提交约定

当前阶段实验先提交到 `experiments/`。每个实验至少包含：

```text
<ex_xx>/
  README.md       # 研究问题、数据、方法、指标、结论边界
  configs/        # 完整可复现配置
  scripts/        # 运行脚本和结果汇总脚本
  reports/        # 实验报告
  results/        # 小型结果表、JSON、日志摘要
  figures/        # GE 曲线和分析图
  manifests/      # 命令、seed、版本、数据哈希和输出清单
```

实验报告必须明确：

- `source_train`、`source_val`、`target_adapt`、`target_attack`；
- 方法、backbone、训练预算和随机种子；
- GE、key rank、SR、NTGE 和总轨迹成本；
- 失败、异常和结论边界；
- 运行命令和实际生成文件。
  等等
  
## 后期结果迁移

当某组实验完成复核、确定用于论文或项目结论后，再进行归档迁移：

```text
experiments/<experiment>/results/
    -> outputs/results/<experiment>/

本地模型检查点
    -> outputs/models/<experiment>/
```

迁移后仍保留原实验目录中的报告、配置和 manifest，保证结果可以追溯。`outputs/` 和 `models/` 用于稳定结果归档，不替代当前阶段的实验提交目录。

## 数据和大文件

以下内容不要直接提交到 Git：

- 原始 `.h5`、`.npy`、`.npz` 数据集；
- 大型 prediction 文件；
- 模型 checkpoint；
- 大型训练日志和压缩包。

实验目录只提交配置、代码、报告、小型汇总结果、图表和 manifest。大型文件保存在本地，并在 manifest 中记录路径、文件大小和哈希。

## 评价原则

分类准确率和 domain accuracy 只能作为辅助指标。正式结论必须优先依据：

- GE 曲线；
- key rank；
- 固定轨迹预算下的成功率；
- 达到目标 GE 或 rank 所需的轨迹数；
- 是否计入目标域适应轨迹后的总攻击成本。

完整实验安排和验收规则以 [`docs/research/新人规划.md`](docs/research/%E6%96%B0%E4%BA%BA%E8%A7%84%E5%88%92.md) 为准。
