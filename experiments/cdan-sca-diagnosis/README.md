# CDAN-SCA 诊断实验

本实验验证项目当前最核心的机制假设：跨设备迁移中，降低域判别准确率并不必然改善猜测熵（GE）；类别条件对齐应比全局边缘对齐更能保持类别结构。

## 实验问题与固定协议

| 项目 | 固定值 |
|---|---|
| 源域 | `ASCAD.h5/Profiling_traces` |
| 目标域 | `ASCAD_desync50.h5/Attack_traces` |
| 标签 | ASCAD 256 类标签，目标字节 2 |
| 方法 | Source Only、DANN、CDAN、CDAN-SCA |
| 主指标 | GE 曲线、SR@100、NTGE（首次 GE=1 的轨迹数） |
| 辅助指标 | target accuracy、domain accuracy、目标预测熵、类中心分离 |
| 公平性 | 相同 backbone、预处理、batch、epoch、学习率、seed 和评估轨迹 |

真实训练命令沿用 `experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py`。完成 TensorFlow/h5py 安装并准备数据后，应对四种方法使用同一组参数运行；CDAN 和 CDAN-SCA 的训练器接入后，再将其结果写入同一 JSON schema。

## 本轮可执行诊断

```bash
python experiments/cdan-sca-diagnosis/scripts/run_protocol_check.py
```

脚本不依赖第三方包，输出：

- `outputs/results/cdan-sca-diagnosis/protocol_check.json`
- `outputs/results/cdan-sca-diagnosis/RESULTS_ANALYSIS.md`

若检测到 ASCAD 文件，脚本只做数据契约检查（组名、维度、标签范围、样本数），不会替代训练。若数据不可用，则运行确定性的 synthetic smoke，仅验证 GE/SR/NTGE 和报告生成链路。

## 真实 SCA 评估

训练模型后先导出目标攻击集概率到 `N x 256` 的 NumPy 文件，再运行：

```bash
MPLCONFIGDIR=/tmp/mpl ~/venvs/tf/bin/python \
  experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py \
  --predictions outputs/predictions/cdan_sca_seed42.npy \
  --dataset data/raw/ASCAD_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5 \
  --target-byte 2 --step 100
```

输出目录包含 `summary.json`、每个方法的 GE `CSV` 和 `ge_curves.png`。`SR@N` 是在固定轨迹数达到 GE=1 的成功指示；多 seed 的成功率应在汇总脚本中对多个 prediction 文件聚合。

## 交付内容列表

1. 本 README：假设、数据、指标、运行协议和复现入口。
2. `scripts/run_protocol_check.py`：数据契约检查与无依赖 smoke 实验。
3. `protocol_check.json`：机器可读的运行状态、输入、指标和限制。
4. `RESULTS_ANALYSIS.md`：自动生成的结果分析、证据边界和下一步行动。
5. 真实训练完成后，应追加四方法的逐 seed JSON、GE/SR 图、汇总 CSV 和 checkpoint 路径。

## 解释规则

synthetic smoke 只能证明评估代码可运行，不能支持“CDAN-SCA 优于 DANN”的科学结论。真实数据报告必须同时给出 GE/SR/NTGE 和域准确率；仅报告 accuracy 或 domain accuracy 不足以证明 SCA 改善。
