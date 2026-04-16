import numpy as np
import h5py
import matplotlib.pyplot as plt

"""
目标域窗口分析脚本：分析 ASCAD variable 数据集的信号活动，确定最佳裁剪区域"""

# 加载少量数据
with h5py.File('data\\raw\\ASCAD_variable\\ascad-variable.h5', 'r') as f:
    traces = f['Profiling_traces']['traces'][:1000]

# 计算方差分布
variances = np.var(traces, axis=0)

# 可视化
plt.figure(figsize=(12, 4))
plt.plot(variances)
plt.axvline(350, color='r', linestyle='--', label='Center start (350)')
plt.axvline(1050, color='g', linestyle='--', label='Center end (1050)')
plt.xlabel('Sample point')
plt.ylabel('Variance')
plt.title('Signal Activity by Position')
plt.legend()
plt.show()

# 输出最佳区域
window_size = 700
best_start = np.argmax(np.convolve(variances, np.ones(window_size), 'valid'))
print(f"Best cropping region: {best_start} - {best_start + window_size}")