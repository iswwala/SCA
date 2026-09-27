# 结果分析报告

运行状态：`synthetic_smoke`。

本次仅完成评估链路 smoke。由于 ASCAD 文件和 TensorFlow/h5py 不可用，表中曲线是确定性合成值，不能作为方法优劣证据。

## 观察
- 脚本成功生成 GE 曲线、SR@100 和 NTGE 字段。
- 真实实验必须替换 `metrics`，并报告至少 3 个随机种子。

## 下一步
1. 挂载 ASCAD 数据并安装 TensorFlow、h5py、NumPy。
2. 统一运行 Source Only、DANN、CDAN、CDAN-SCA。
3. 用真实 GE/SR/NTGE、域准确率和类分离结果更新本报告。
