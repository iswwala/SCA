# Baseline Pilot 结果分析

配置：ASCAD profiling -> `ASCAD_desync50` attack，512 源样本、512 目标适应样本、256 评估样本、batch 64、5 epochs、seed 42/43/44。

| 方法 | 末轮 target accuracy（均值） | 末轮 label-rank proxy（均值） |
|---|---:|---:|
| Source Only | 0.0039 | 130.51 |
| DANN | 0.0052 | 130.71 |

三次运行中 DANN 相比 Source Only 的 proxy 平均高约 0.20，差异很小且跨 seed 波动明显。该结果只能说明当前极小规模 pilot 尚未观察到稳定收益；不能替代真实 key-byte GE、SR、NTGE，也不能证明 CDAN-SCA 的有效性。下一步应在同一契约下接入 CDAN/CDAN-SCA，并扩展样本量和训练轮数。
