# CDAN-SCA 文献缺口分析与论文切入点

## 1. 核心定位

本研究适合从现有跨设备侧信道攻击论文中提炼一个明确问题：

> 现有工作已经证明 cross-device SCA 很重要，但大多依赖 PCA、DTW、FFT、LDA、多设备训练、手工预处理或有标签微调；它们没有从 256 类密钥分类的条件分布对齐角度系统解决负迁移问题。

因此，CDAN-SCA 的论文叙事可以定位为：

> 面向 256 类密钥假设分类任务，提出一种适用于 profiling SCA 的条件域对抗迁移框架，从边缘分布对齐转向联合/条件分布对齐，以缓解跨设备攻击中的负迁移问题。

需要注意：不要把 `f ⊗ g` 多线性条件表示和 entropy conditioning 写成完全原创，因为原始 CDAN 论文已经包含这些思想。更稳妥的写法是：

- 将条件域对抗方法系统迁移到 256 类 profiling SCA 场景；
- 分析 DANN 式边缘分布对齐在 SCA 中失效的机制；
- 提出 SCA-aware 的训练、评价和消融协议；
- 在跨设备/跨域 SCA 数据上证明条件分布对齐比全局边缘对齐更适合密钥恢复任务。

参考：

- CDAN: [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)
- DANN: [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/v17/15-239.html)

## 2. 可引用的关键论文与研究缺口

| 论文 | 已有问题 / gap | CDAN-SCA 的切入方式 |
|---|---|---|
| **Practical Approaches Towards Deep-Learning Based Cross-Device Power Side Channel Attack** | 明确指出跨设备功耗 SCA 会受到 device-to-device variation 影响。该类方法通常依赖 PCA、DTW、多设备训练等预处理或工程技巧，端到端自适应能力有限。 | 可写成：现有方法主要靠手工特征变换和多设备训练缓解域偏移，缺少无标签目标域条件对齐机制。链接：[arXiv 1907.02674](https://arxiv.org/abs/1907.02674) |
| **EM-X-DL: Efficient Cross-Device Deep Learning Side-Channel Attack with Noisy EM Signatures** | 解决 noisy EM 和 cross-device 问题，但仍依赖多个训练设备、PCA/LDA/FFT、设备选择和超参数调节。 | 可定位为：CDAN-SCA 尝试减少对手工特征变换和多训练设备选择的依赖，通过目标域无标签 trace 做自适应。链接：[arXiv 2011.06139](https://arxiv.org/abs/2011.06139) |
| **Crossed-IoT Device Portability of EM-SCA: Challenges and Dataset** | 强调 device variability、环境差异和采集设置差异会显著影响 EM-SCA，可用解决方案仍然有限。 | 适合用于 Introduction，证明 cross-device portability 是现实且公开的问题。该文偏 IoT/EM 场景，不是标准 AES key recovery，可作为问题背景引用。链接：[arXiv 2310.03119](https://arxiv.org/abs/2310.03119) |
| **Ensuring Cross-Device Portability of EM-SCA** | 实验发现预训练模型跨相同或相似设备迁移效果很差，很多情况下准确率低于 20%；transfer learning 有效，但通常需要目标域样本或适配步骤。 | 可写成：现有 transfer learning 说明适配是必要的，但没有针对“目标域无标签 + 256 类密钥分类”的条件分布对齐问题。链接：[arXiv 2312.11301](https://arxiv.org/abs/2312.11301) |
| **Playing with Blocks: Toward Re-usable Deep Learning Models for SCA Profiled Attacks** | 目标是构造可复用 DL-SCA 模块，减少每次安全评估重新设计网络的成本，证明模型可迁移性是重要方向。 | 可将 CDAN-SCA 写成对“可复用攻击模型”的进一步推进：不仅复用神经网络模块，还通过条件域对抗让源域模型适配目标设备。链接：[arXiv 2203.08448](https://arxiv.org/abs/2203.08448) |
| **Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning** | 关注跨算法、跨实现、跨 countermeasure 的泛化攻击，强调手工调参和 trace preprocessing 成本高。 | 可引用它说明“自动化、低手工预处理、可泛化 SCA”是领域趋势。CDAN-SCA 更聚焦 cross-device 256 类 key recovery。链接：[arXiv 2306.07249](https://arxiv.org/abs/2306.07249) |
| **A Review and Comparison of AI Enhanced Side Channel Analysis** | 综述 AI/DL-SCA 现状，说明大量工作集中在 ASCAD、CNN/ResNet、生成式增强等方向，cross-device/domain shift 仍未被充分解决。 | 适合放在 Related Work 开头，支撑“现有 AI-SCA 多关注建模能力，较少系统处理跨设备域偏移”的论断。链接：[arXiv 2402.02299](https://arxiv.org/abs/2402.02299) |

## 3. 最强论文切入点

建议不要将论文写成“我们首次提出 CDAN 或熵权重”，因为原始 CDAN 已经包含 multilinear conditioning 和 entropy conditioning。

更推荐的贡献表述如下：

### 3.1 DANN 在 256 类 profiling SCA 中的负迁移分析

传统 DANN 对齐的是源域与目标域的边缘特征分布 `P(f)`。但 profiling SCA 本质上是 256 类密钥假设分类任务，不同类别的泄漏特征可能呈现稀疏、高维、多模态结构。

因此，全局边缘分布对齐可能会把不同密钥类别的特征错误拉近，破坏分类边界，导致负迁移。

可验证内容：

- t-SNE / UMAP 可视化 Source Only、DANN、CDAN-SCA 的源域和目标域特征；
- 类内距离、类间距离、Fisher ratio；
- class-wise MMD 或按伪标签分组的 centroid distance；
- DANN 域判别器准确率下降但 key rank 没有改善的反例。

### 3.2 从边缘分布对齐转向联合/条件分布对齐

CDAN-SCA 的核心不是简单让两个设备的整体特征相似，而是让：

```text
P_s(f, y) ≈ P_t(f, y)
```

或者近似地，让相同密钥类别内部的源域和目标域特征对齐。

实现方式：

```text
feature extractor: F(x) = f
classifier: G(f) = g
conditional representation: h = f ⊗ g
domain discriminator: D(h)
```

这样，域判别器看到的不只是特征 `f`，还包含分类预测 `g`，因此它可以感知当前样本属于哪个密钥类别，从而减少不同类别被强行全局拉近的风险。

### 3.3 SCA-aware 训练和评价协议

普通机器学习论文常用 accuracy，但 SCA 论文必须使用攻击指标：

- Guessing Entropy, GE；
- Key Rank 曲线；
- 达到 `GE = 1` 或 `Rank = 0` 所需 trace 数；
- Success Rate under N traces；
- 多随机种子、多 attack subset 的均值和置信区间。

这部分可以作为方法贡献之一：不是只看分类正确率，而是将 domain adaptation 的训练目标与 SCA 的真实攻击评价严格对应起来。

## 4. 推荐论文故事线

Introduction 可以按下面逻辑组织：

1. Profiling SCA 是密码硬件安全评估中的强攻击模型，但依赖 profiling device 与 target device 的一致性。
2. 跨设备场景下，device-to-device variation、采集噪声、时钟偏移、环境差异会导致模型迁移失败。
3. 现有 cross-device SCA 工作已经提出 PCA、DTW、FFT、LDA、多设备训练、transfer learning 等方法，但仍依赖手工预处理、多训练设备或目标域标签。
4. DANN 等通用域自适应方法看似适合该问题，但在 256 类密钥分类中可能因为全局边缘对齐导致类混叠和负迁移。
5. 本文提出 CDAN-SCA，将条件域对抗适配到 profiling SCA，使用 `f ⊗ g` 条件表示和熵加权策略，在无标签目标域上实现更稳健的跨设备特征对齐。
6. 实验在 ASCAD controlled domain shift 和真实 cross-device 数据上验证，使用 GE、key rank、MTD 等 SCA 指标评估攻击效果。

## 5. 可写的贡献点

论文贡献可以写成四点：

1. **Problem diagnosis**  
   系统分析 DANN 式边缘分布对齐在 256 类 profiling SCA 中产生负迁移的原因，并通过可视化和类间/类内距离指标验证。

2. **Method adaptation**  
   提出 CDAN-SCA，将条件域对抗学习引入跨设备 SCA，使模型对齐源域和目标域的条件/联合分布，而不是仅对齐边缘特征分布。

3. **Entropy-aware target alignment**  
   在目标域无标签条件下，基于预测熵动态调节域对抗损失，降低低置信度伪标签对条件对齐过程的污染。

4. **Attack-centric evaluation**  
   在多个跨域/跨设备 SCA 场景中使用 GE、key rank、MTD 和 success rate 进行评估，并与 Source Only、DANN、vanilla CDAN、传统预处理迁移方法进行对比。

## 6. 建议实验设计

### 6.1 Baselines

至少包括：

- Source Only；
- Target Oracle；
- DANN；
- vanilla CDAN；
- CDAN-SCA；
- PCA/DTW/FFT/LDA 类传统迁移方案；
- 多设备训练或 transfer learning 方法，如果数据条件允许。

### 6.2 数据集设置

建议区分两类实验：

#### Controlled domain shift

可使用 ASCAD：

- fixed key → variable key；
- no desync → desync 50；
- no desync → desync 100；
- 不同 profiling trace 数量；
- 不同噪声或随机平移强度。

这类实验应写成“受控域偏移实验”，不要直接称为真实 cross-device。

#### Real cross-device

最好至少包含 3 个物理设备，理想情况下 5 个以上。

推荐协议：

```text
N-1 source devices → 1 held-out target device
```

目标域只允许使用无标签 traces。每个设备轮流作为 target device，报告平均结果和方差。

### 6.3 消融实验

建议消融项：

- 无域自适应；
- 仅 DANN 边缘对齐；
- 仅条件对抗，无熵权重；
- 条件对抗 + 熵权重；
- MLP discriminator vs CNN discriminator；
- 不同对抗权重 `λ`；
- 不同目标域无标签 traces 数；
- 不同 feature dimension；
- 不同 random multilinear map dimension。

## 7. 需要避免的表述

建议避免：

- “首次提出 CDAN”；
- “首次提出熵加权”；
- “根治负迁移”；
- “真正解决跨设备攻击”；
- “在所有场景下都有效”。

更稳妥的表述：

- “首次系统研究条件域对抗在 256 类 profiling SCA 中的适用性”；
- “显著缓解 DANN 式边缘对齐导致的负迁移”；
- “在多种跨域/跨设备设置下稳定降低 key rank 和 MTD”；
- “为无标签目标域下的跨设备 SCA 提供一种 attack-aware domain adaptation 框架”。

## 8. 推荐投稿方向

更推荐：

- **IACR TCHES / CHES**：最贴近密码硬件、嵌入式安全和侧信道攻击；
- **IEEE TIFS**：适合安全 + 机器学习 + 信息安全应用叙事。

不太推荐作为第一目标：

- **IACR TOSC**：更偏对称密码算法设计与分析，不是侧信道实现攻击主场；
- **Journal of Cryptology**：难度较高，通常需要更强理论贡献。单纯实验型 DL-SCA 不一定占优势。

## 9. 一句话总结

CDAN-SCA 最有说服力的故事不是“把 CDAN 用到 SCA”，而是：

> 现有跨设备 SCA 方法已经确认域偏移问题存在，但没有解释和解决 256 类密钥分类下边缘对齐导致的负迁移。CDAN-SCA 从条件/联合分布对齐出发，用 SCA-aware 的训练和攻击评价协议，在无标签目标域跨设备攻击中显著降低 key rank 和所需 traces 数。

