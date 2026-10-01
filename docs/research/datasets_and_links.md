# 侧信道分析数据集与相关链接

本文档整理当前研究方向涉及的主要数据集、官方页面、下载地址和推荐用途。

## 1. 数据集分类

当前数据集可以分为四类：

1. **基础建模侧信道数据集：** 用于训练模型和验证 GE、SR、NTGE 等指标；
2. **受控域偏移数据集：** 通过去同步、噪声和时钟扰动模拟目标域变化；
3. **真实跨设备数据集：** 用于验证模型在不同芯片或板卡之间的迁移能力；
4. **复杂掩码和采集条件数据集：** 用于检验模型在防护、探针位置和实现变化下的鲁棒性。

## 2. ASCAD v1 fixed-key

### 数据集简介

ASCAD v1 fixed-key 是当前项目最适合用于第一阶段实验的数据集，主要特点包括：

- ATMEGA8515 平台；
- 一阶布尔掩码 AES；
- 固定密钥；
- 约 50,000 条 profiling 轨迹和 10,000 条 attack 轨迹；
- 处理后轨迹通常保留 700 个感兴趣采样点；
- 提供原始版本、`desync50` 和 `desync100` 数据库。

### 主要用途

- 复现 ASCAD 官方 CNN；
- 训练 Source Only、DANN、CDAN 等模型；
- 进行受控时序偏移实验；
- 计算猜测熵、密钥排名、成功率和恢复密钥所需轨迹数。

### 相关链接

- 官方代码仓库：[https://github.com/ANSSI-FR/ASCAD](https://github.com/ANSSI-FR/ASCAD)
- 固定密钥数据包：[https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e](https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e)
- ASCAD 论文：[https://doi.org/10.1007/s13389-019-00220-8](https://doi.org/10.1007/s13389-019-00220-8)
- ASCAD 早期技术报告：[https://eprint.iacr.org/2018/053.pdf](https://eprint.iacr.org/2018/053.pdf)

### 数据库名称

```text
ASCAD.h5
ASCAD_desync50.h5
ASCAD_desync100.h5
```

## 3. ASCAD v1 variable-key

### 数据集简介

ASCAD variable-key 同样基于 ATMEGA8515，但使用变量密钥和更大的数据规模：

- 约 200,000 条 profiling 轨迹；
- 约 100,000 条 attack 轨迹；
- 提供原始版本、`desync50` 和 `desync100`；
- 适合研究变量密钥、较大数据规模和时序偏移。

### 相关链接

- 官方说明：[https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md](https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md)
- 原始轨迹：[https://www.data.gouv.fr/api/1/datasets/r/3217dcc0-184f-402b-8914-e31cc120c51c](https://www.data.gouv.fr/api/1/datasets/r/3217dcc0-184f-402b-8914-e31cc120c51c)
- 变量密钥数据库：[https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00](https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00)
- `desync50` 数据库：[https://www.data.gouv.fr/api/1/datasets/r/4ad6d44a-f6de-483f-807f-d0ccab76d2a9](https://www.data.gouv.fr/api/1/datasets/r/4ad6d44a-f6de-483f-807f-d0ccab76d2a9)
- `desync100` 数据库：[https://www.data.gouv.fr/api/1/datasets/r/f1936388-71be-408f-b8ec-472bb3398e39](https://www.data.gouv.fr/api/1/datasets/r/f1936388-71be-408f-b8ec-472bb3398e39)

## 4. ASCADv2：STM32 掩码 AES

### 数据集简介

ASCADv2 面向 STM32 Cortex-M4 上更复杂的 AES 实现，包含：

- 仿射掩码；
- 洗牌操作；
- 随机密钥；
- 多任务标签；
- 长功耗轨迹；
- ChipWhisperer 采集条件。

### 主要用途

- 复杂掩码侧信道分析；
- 多任务模型；
- STM32 设备上的迁移学习；
- 更强防护条件下的模型鲁棒性研究。

### 相关链接

- 官方数据集页面：[https://www.data.gouv.fr/en/datasets/ascadv2/](https://www.data.gouv.fr/en/datasets/ascadv2/)
- 提取后的数据库：[https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5](https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5)
- 原始数据文件哈希：[https://files.data.gouv.fr/anssi/ascadv2/sha1.txt](https://files.data.gouv.fr/anssi/ascadv2/sha1.txt)
- STM32 AES 实现：[https://github.com/ANSSI-FR/SecAESSTM32](https://github.com/ANSSI-FR/SecAESSTM32)
- ASCAD 官方代码：[https://github.com/ANSSI-FR/ASCAD](https://github.com/ANSSI-FR/ASCAD)
- 相关 ePrint：[https://eprint.iacr.org/2021/592](https://eprint.iacr.org/2021/592)

完整原始数据规模较大，初期实验建议优先使用提取后的数据库。

## 5. CDPA 跨设备侧信道数据集

### 数据集简介

CDPA 将 profiling device 和 target device 视为不同 domain，主要用于研究无标签目标域条件下的跨设备建模侧信道分析。

### 相关链接

- 官方代码和数据仓库：[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack)
- 论文：[https://doi.org/10.46586/tches.v2021.i4.27-56](https://doi.org/10.46586/tches.v2021.i4.27-56)

### 5.1 XMEGA 多芯片数据

数据特点：

- 未防护软件 AES-128；
- 8 个 XMEGA 芯片；
- 不同芯片对应不同设备域；
- 适合真实跨芯片迁移实验。

链接：

[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/XMEGA](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/XMEGA)

### 5.2 SAKURA AES 多板卡数据

数据特点：

- 未防护硬件 AES-128；
- 3 块 SAKURA-G 板卡；
- 适合真实跨板卡迁移实验。

链接：

[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/SAKURA_AES](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/SAKURA_AES)

### 5.3 CHES CTF 2018 数据

数据特点：

- 掩码软件 AES-128；
- 包含不同设备条件；
- 适合研究掩码、防护和实现差异带来的域偏移。

链接：

- CDPA 项目页面：[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/CHES_CTF_2018](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Devices/CHES_CTF_2018)
- 原始数据来源：[https://github.com/AISyLab/EnsembleSCA](https://github.com/AISyLab/EnsembleSCA)

## 6. CDPA 不同实现数据

### 数据集简介

该部分基于 ASCAD 构造受控实现差异，用于模拟目标设备启用不同噪声或侧信道防护：

- Gaussian Noise；
- Desynchronization；
- Clock Jitter。

这些数据属于受控域偏移，适合用于控制变量实验，但不能直接等同于真实跨设备数据。

### 相关链接

- 官方页面：[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Implementations](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Implementations)
- Gaussian Noise 示例：[https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addGaussianNoise_Demo.ipynb](https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addGaussianNoise_Demo.ipynb)
- Desynchronization 示例：[https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addDesync100_Demo.ipynb](https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addDesync100_Demo.ipynb)
- Clock Jitter 示例：[https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addClockJitters_Demo.ipynb](https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Implementations/ASCAD_addClockJitters_Demo.ipynb)

## 7. XMEGA-EM 不同探针位置数据

### 数据集简介

该数据集使用近场电磁探针采集未防护 AES-128，主要变化来自探针位置和人工测量误差。

### 主要用途

- 研究 EM 特征对探针位置变化的稳定性；
- 研究不同采集位置造成的域偏移；
- 验证模型在不同采集条件下的迁移能力。

### 相关链接

- 官方页面：[https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Probe_Positions](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack/tree/main/Different_Probe_Positions)
- 示例 notebook：[https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Probe_Positions/XMEGA_EM_CDPA_Demo.ipynb](https://nbviewer.jupyter.org/github/CDPA-SCA/Cross-Device-Profiled-Attack/blob/main/Different_Probe_Positions/XMEGA_EM_CDPA_Demo.ipynb)

## 8. 推荐使用顺序

### 第一阶段：基础模型和受控域偏移

```text
ASCAD fixed-key
ASCAD_desync50
ASCAD_desync100
```

目标：

- 跑通官方 CNN；
- 建立 Source Only、DANN、CDAN 的统一训练接口；
- 统一 GE、SR 和 NTGE 评价；
- 分析时序偏移下的负迁移现象。

### 第二阶段：真实跨设备验证

```text
CDPA-XMEGA
CDPA-SAKURA AES
```

目标：

- 单源设备到单目标设备迁移；
- 留一设备测试；
- 比较 Source Only、DANN、CDAN、CDPA/MMD 和改进方法。

### 第三阶段：扩展泛化验证

```text
XMEGA-EM
CHES CTF 2018
ASCADv2
```

目标：

- 探针位置变化；
- 掩码和实现差异；
- 复杂多任务标签；
- 方法的适用边界和失败条件。

## 9. 数据使用注意事项

1. ASCAD 的 `desync50`、`desync100`、Gaussian Noise 和 Clock Jitter 属于受控域偏移，不是真实跨设备实验。
2. CDPA-XMEGA、SAKURA AES、CHES CTF 2018 和 XMEGA-EM 更适合支撑真实设备或真实采集条件下的迁移结论。
3. 不同数据集的设备、实现、密钥、目标字节、标签定义和轨迹格式可能不同，使用前必须核对。
4. 论文中应区分 profiling 数据、目标域适应数据和最终 attack evaluation 数据。
5. 最终结果应优先报告 key rank、GE、SR 和 NTGE，分类准确率只能作为辅助指标。
6. 大型原始轨迹和压缩数据不建议提交到 GitHub，项目仓库中保留下载链接、配置和处理脚本即可。
