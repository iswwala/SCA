# 实验管理规范

本目录是当前阶段实验提交的统一入口。实验结构、提交要求和报告格式以 [`docs/research/新人规划.md`](../docs/research/%E6%96%B0%E4%BA%BA%E8%A7%84%E5%88%92.md) 为准。

## 当前阶段规则

当前阶段的原则是：

```text
实验结果先提交到 experiments/
        -> 复核配置和关键结果
        -> 形成统一汇总
        -> 后期将正式结果迁移到 outputs/ 和 models/
```

实验不能只提交代码，也不能只提交截图。每个实验必须有配置、运行入口、报告、结果摘要和运行清单。

## 实验分区

```text
experiments/
  00_common_e0/
    README.md
    configs/
    scripts/
    reports/
    results/
    figures/
    manifests/

  01_member_a_baselines/
    README.md
    a_e1_data_audit/
    a_e2_uda_baselines/
    a_e3_cdpa_or_adabn/
    a_e4_real_device/

  02_member_b_method/
    README.md
    b_e1_architecture_same_domain/
    b_e2_architecture_shift/
    b_e3_cdan_sca/
    b_e4_ablation/
    b_e5_method_exports/

  03_member_c_robustness/
    README.md
    c_e1_official_cnn_baseline/
    c_e2_shift_noise_robustness/
    c_e3_attack_budget/
    c_e4_target_sample_budget/
    c_e5_hyperparameter_sensitivity/
    c_e6_metric_consistency/
    c_e7_final_audit/

  04_integrated_results/
    README.md
    reports/
    results/
    figures/
    manifests/
```

已有的 `cdan-sca-baselines/`、`cdan-sca-diagnosis/`、`cdan-sca-ablation/` 和 `legacy/` 目录保留用于历史追踪和复用。新的正式实验优先放入上述结构目录，避免历史脚本和新结果混在一起。

## 单个实验目录要求

每个实验目录固定使用以下结构：

```text
<experiment_id>/
  README.md
  configs/
  scripts/
  reports/
  results/
  figures/
  manifests/
```

### `README.md`

说明：

- 实验维护记录和复核状态；
- 研究问题和假设；
- 源域、目标域和偏移类型；
- 数据划分和目标字节；
- 方法、backbone 和对比方法；
- 指标、运行命令和结论边界。

### `configs/`

保存完整参数，至少包括：

- 数据集和源/目标域；
- `source_train`、`source_val`、`target_adapt`、`target_attack`；
- 轨迹长度、标签、目标字节和预处理；
- 方法、backbone、optimizer、learning rate、batch size、epoch；
- domain loss、适应权重、投影维度和熵权重；
- seed、样本量和评估步长。

### `scripts/`

保存可重复运行的训练、评估、绘图和结果汇总脚本。脚本不能依赖未记录的命令行参数或个人临时路径。

### `reports/`

报告必须包括：

1. 研究问题和假设；
2. 数据与四类数据划分；
3. 方法和固定变量；
4. 运行配置与 seed；
5. GE、key rank、SR、NTGE 和攻击成本；
6. 结果解释；
7. 失败、异常和限制；
8. 结论边界和下一步实验。

### `results/`

优先提交小型、可追踪的结果：

```text
metrics.csv
summary.json
diagnosis.json
table_*.csv
training_history.json
```

大型 prediction 和 checkpoint 不提交到 Git，只在 manifest 中记录其本地位置和哈希。

### `figures/`

文件名必须包含实验编号和含义，例如：

```text
c_e2_ge_shift_noise.png
b_e4_ablation_ntge.csv
a_e2_method_comparison.png
```

不要使用 `final.png`、`new.png` 等无法追踪的文件名。

### `manifests/`

每次正式运行至少记录：

- 运行命令；
- 运行日期；
- seed；
- 配置文件；
- 代码版本或 commit；
- 数据版本或数据哈希；
- 输出文件清单；
- smoke、pilot 或 formal 状态。

## 当前提交与后期迁移

### 当前提交

当前将完整实验目录提交到 `experiments/` 下对应位置，例如：

```text
experiments/03_member_c_robustness/c_e2_shift_noise_robustness/
```

其中报告、配置、脚本、小型结果表和图表都放在该实验目录内。

### 后期迁移

实验经过复核并确定进入论文后，再迁移正式产物：

```text
experiments/<experiment>/results/
    -> outputs/results/<experiment>/

本地 checkpoint
    -> outputs/models/<experiment>/
```

迁移后不删除原实验目录。原目录保留实验过程、配置、报告和 manifest，`outputs/` 与 `models/` 只作为稳定结果和模型归档位置。

## 禁止提交内容

- 原始数据集；
- `.h5`、`.npy`、`.npz` 等大型数据文件；
- 大型 prediction 文件；
- 模型 checkpoint；
- 未整理的临时日志和压缩包；
- 只存在个人机器上的未记录参数。

## 提交前检查

```text
[ ] 目录中有 README.md
[ ] 配置中写明 source/target 和四类数据划分
[ ] 运行命令可以复现关键步骤
[ ] 报告包含 GE/key rank/SR/NTGE
[ ] 说明 target_adapt 是否计入总攻击成本
[ ] 结果表和图能够追溯到 seed 与配置
[ ] manifest 已记录代码、数据和输出信息
[ ] 没有提交原始数据和大型 checkpoint
```
