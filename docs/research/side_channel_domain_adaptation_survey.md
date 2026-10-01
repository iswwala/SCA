# 建模侧信道分析中的域适应与跨设备迁移综述调查

**调查日期：** 2026-09-19
**调查范围：** 深度学习建模 SCA、ASCAD 基准、跨设备 SCA、无监督域适应、攻击中心评价和真实采集验证。
**调查目的：** 为 SCA-UDA/CDAN-SCA 项目定位研究空白，避免把已有通用 UDA 方法直接包装成新的 SCA 方法。

## 1. 问题背景

建模侧信道分析通常在一个 profiling device 上收集有标签轨迹，训练分类模型预测与密钥相关的中间值，再在目标设备的无标签攻击轨迹上输出概率，最后通过概率累积计算候选密钥的排名、猜测熵和成功率。

理想化实验往往让 profiling 和 attack 轨迹来自相同设备、相同实现和相同采集条件。真实场景中，源设备与目标设备可能在芯片个体、实现方式、探针位置、噪声、时钟、掩码和采集环境上存在差异。此时训练分布和攻击分布不一致，源域分类能力不等于目标设备密钥恢复能力。

因此，该领域同时涉及三个层面：

1. **表示学习：** 从功耗或电磁波形中提取密钥相关特征；
2. **域迁移：** 减少源域和目标域的分布差异；
3. **攻击评价：** 判断迁移是否真的减少恢复密钥所需的轨迹数。

其中第三层是侧信道分析区别于普通分类任务的关键。

## 2. 文献谱系

### 2.1 深度学习建模 SCA 与 ASCAD

ASCAD 数据库及其配套代码将建模 SCA 转化为可复现的深度学习分类任务，推动了固定密钥、掩码、轨迹窗口和 GE 评价的标准化。典型流程不是以单条轨迹分类准确率作为最终目标，而是将连续攻击轨迹的类别概率映射回 256 个密钥候选并累积评分。

对本项目而言，ASCAD 的价值在于：

- 提供稳定的 256 类任务和官方 CNN 结构；
- 允许控制去同步、噪声和采样变化；
- 适合先做机制诊断、消融和敏感性实验。

但 ASCAD 的受控变体不能自动等同于真实跨设备数据。若论文只使用 fixed-key、desync 和人工噪声，结论应写成“受控域偏移下的迁移”，而不是“解决真实跨设备攻击”。

### 2.2 通用无监督域适应

#### DANN

DANN 使用特征提取器、任务分类器和域判别器，通过梯度反转学习任务判别而域不可辨识的表示。它主要对齐边缘特征分布，适合作为检验“全局域对齐是否有用”的基线。

在 SCA 中，DANN 的潜在问题是：256 个泄露类别并不是一个单峰分布。全局对齐可能把不同类别的模式混合，得到更难区分的特征，同时域判别器的准确率下降。这为“domain confusion 与 key recovery 不等价”提供了可检验假设。

#### CDAN

CDAN 将分类器预测与特征共同输入域判别器，常见做法是多线性条件表示 `f ⊗ g`，并可结合熵条件化降低不确定目标样本的影响。其动机是处理多模态分类分布，理论上比只对齐 `P(f)` 更适合保留类别结构。

但直接把 CDAN 用于 256 类 SCA 仍有几个未解决问题：

- 目标域预测早期可能是错误的；
- 低熵可能代表 confident wrong；
- `f ⊗ g` 的维度和稳定性需要工程化处理；
- CDAN 的改善必须最终体现在 GE/SR/NTGE，而不是只体现在 domain accuracy。

#### MMD、CORAL、AdaBN 和统计对齐

统计对齐方法通常比对抗训练更容易复现、训练更稳定，适合做低成本基线。CDPA 代表了侧信道领域中使用 MMD 解决跨设备分布差异的重要路线；AdaBN 等方法则从批归一化统计量或多阶矩角度缓解设备差异。

这类方法的优点是开销较低、解释直观；局限是可能只对齐总体统计量，无法显式维护泄露类别结构。

### 2.3 侧信道领域的跨设备迁移

#### CDPA：MMD-based cross-device profiled SCA

CDPA 将源设备和目标设备视为不同 domain，在经典深度学习建模 SCA 后加入无标签目标轨迹的 MMD 约束，并在 Atmel XMEGA、SAKURA AES 和公开变体上评估。它是本项目必须纳入的 SCA-specific baseline。

CDPA 的重要启示是：

- 即使算法、密钥和标签语义相同，设备差异也可能让源模型失效；
- UDA 在 SCA 中不能只用通用图像分类指标描述；
- 真实多设备数据比单一 ASCAD 变体更能支撑 cross-device claim。

其不足也为本项目提供空间：MMD 主要从分布差异角度约束特征，尚未直接回答高基数类别结构、目标置信度可靠性和 GE 机制之间的关系。

#### 其他模型迁移与注意力方向

后续工作探索了跨设备 power/EM SCA、注意力增强的域对抗网络、自动编码器和特征统计匹配等路线。这些工作说明研究重点正在从“同设备高准确率”转向“模型能否在不同设备和采集条件下保持攻击能力”。

但不同论文常使用不同数据划分、不同预处理和不同攻击字节，结果不容易直接横向比较。因此，本项目若要有贡献，必须把统一评估协议本身做扎实。

## 3. 方法比较表

| 方法族            | 主要对齐对象      |     是否需要目标标签 | 优点                   | 在 SCA 中的风险            | 本项目角色         |
| ----------------- | ----------------- | -------------------: | ---------------------- | -------------------------- | ------------------ |
| Source Only       | 不做对齐          |                   否 | 简单、提供迁移下界     | 对设备偏移敏感             | 必要基线           |
| DANN              | 边缘特征`P(f)`  |                   否 | 通用、经典             | 类别模态碰撞               | 机制诊断基线       |
| CDAN              | 条件联合结构近似  |                   否 | 保留预测条件           | 依赖伪标签质量，计算开销高 | 通用条件基线       |
| MMD/CDPA          | 核或统计分布差异  |                   否 | 稳定、SCA 领域已有工作 | 类别结构约束弱             | 领域强基线         |
| AdaBN/统计适应    | BN 统计或矩       |                   否 | 低成本、易复现         | 表示能力有限               | 实用基线           |
| 校准感知适应      | 条件结构 + 可靠性 |                   否 | 针对 confident wrong   | 需要可靠性估计             | 候选主方法         |
| 原型保持适应      | 类别几何结构      |                   否 | 显式维护类间隔         | 类别覆盖不足时不稳定       | 候选主方法         |
| GE-aware training | 攻击目标          | 通常需要目标评估信息 | 与 SCA 目标一致        | 可能泄漏目标标签           | 只能做严格受限研究 |

## 4. 评价指标的层次

### 4.1 分类指标

- source accuracy；
- target accuracy，仅可作为辅助；
- cross-entropy；
- confusion matrix。

这些指标适合验证模型是否学会分类，但不能单独证明密钥恢复效果。

### 4.2 域适应指标

- domain discriminator accuracy；
- domain AUC；
- MMD/CORAL 等距离；
- 目标预测熵；
- ECE、Brier score 或熵-正确率曲线。

域指标只能说明域信息是否被隐藏，不能说明泄露信息是否被保留。

### 4.3 侧信道原生指标

- guessing entropy / key rank curve；
- success rate at fixed trace counts；
- traces to disclosure / NTGE；
- 多 seed 的曲线均值、标准差和最差结果。

主论文必须把这一层作为主要证据。一个方法即便 target accuracy 提升，如果 GE 不降或 SR 不升，也不能称作更好的攻击方法。

## 5. 当前领域的主要研究空白

### Gap 1：域混淆与密钥恢复之间缺少系统诊断

很多工作把 domain discrepancy 下降当作适应成功的代理指标，但侧信道攻击最终需要恢复密钥。需要直接研究：

```text
domain confusion -> class structure -> probability quality -> key rank
```

这条链条中任何一环都可能断裂。

### Gap 2：高基数泄露类别下的条件适应证据不足

图像域适应中的多模态类别通常数量较少，而 AES identity leakage 常是 256 类问题，且类别概率最终要参与 key-hypothesis scoring。类别条件对齐是否真的维护了泄露模态，需要专门的类结构和 GE 分析。

### Gap 3：熵权重的可靠性假设没有被充分检验

熵权重默认低熵表示可信预测，但目标域强偏移时可能出现大量 confident wrong。应报告置信度和正确率的关系，而不是直接把低熵样本视为好样本。

### Gap 4：受控域偏移与真实跨设备常被混用

去同步、加噪和时钟扰动适合控制变量，但不能替代不同芯片、不同实现或不同采集设置。论文应将“受控偏移”和“真实跨设备”分层报告。

### Gap 5：实验协议不统一，结果难以横向比较

不同工作可能在以下方面不同：

- profiling/attack group；
- 目标字节；
- 预处理；
- attack trace 数量；
- checkpoint 选择；
- 是否使用目标标签进行调参；
- GE 的累积方式。

统一协议和数据契约本身是本项目的重要工程贡献，但必须配合新的机制或方法结果，不能单独包装为算法创新。

### Gap 6：物理采集验证仍然不足

公开基准可以验证方法，但真实采集能检验数据整理、同步、噪声和设备间差异。低成本 ChipWhisperer、STM32、AVR 或类似设备可以作为后期外部验证，而不必一开始就设计昂贵复杂的硬件系统。

## 6. 对本项目的定位建议

本项目最强的定位：

> “面向高基数 profiling SCA，研究边缘/条件域适应如何影响类别结构和密钥排名，并设计带有目标可靠性或类别几何约束的攻击中心域适应框架。”

这个定位有三个好处：

1. 即使原始 CDAN 没有稳定提升，也能产出机制诊断；
2. 方法创新可以根据实验结果选择校准或原型，不锁死实现路线；
3. 真实设备验证和公开 ASCAD 受控实验可以自然组成证据层次。

## 7. 综述结论

该领域已经从“证明深度学习可以做 SCA”进入“模型能否跨设备、跨实现、跨采集条件工作”的阶段。通用 UDA 方法提供了工具，但并没有自动解决侧信道特有的三个问题：

- 类别数高且概率结构参与密钥评分；
- 目标域通常没有标签，置信度可能不可靠；
- 最终评价是密钥排名和恢复轨迹数，而非普通分类准确率。

因此，当前项目若想形成高水平论文，应围绕以下闭环展开：

```text
跨设备偏移
  -> 边缘对齐/条件对齐的机制差异
  -> 类别结构与置信度可靠性
  -> GE/SR/NTGE 攻击结果
  -> 适用范围和失败边界
```

## 8. 参考文献与资料

以下优先列出论文原文、官方代码或 DOI；仓库中也保存了部分 PDF 副本于 `references/papers/ref/`。

1. Ganin, Y. et al. **Domain-Adversarial Training of Neural Networks.** JMLR, 2016. [JMLR](https://www.jmlr.org/papers/v17/15-239.html)
2. Long, M. et al. **Conditional Adversarial Domain Adaptation.** NeurIPS, 2018. [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html)
3. Benadjila, R. et al. **Deep Learning for Side-Channel Analysis and Introduction to ASCAD Database.** Journal of Cryptographic Engineering, 2020. [DOI](https://doi.org/10.1007/s13389-019-00220-8)
4. ANSSI-FR. **ASCAD public database and scripts.** [GitHub](https://github.com/ANSSI-FR/ASCAD)
5. Cao, P. et al. **Cross-Device Profiled Side-Channel Attack with Unsupervised Domain Adaptation.** IACR TCHES, 2021, 27-56. [DOI](https://doi.org/10.46586/tches.v2021.i4.27-56). [Official repository](https://github.com/CDPA-SCA/Cross-Device-Profiled-Attack). 本地副本：`references/papers/ref/TCHES2021_4_02 (1).pdf`。
6. Zaid, G. et al. **Practical Approaches Towards Deep-Learning Based Cross-Device Power Side Channel Attack.** 2019. [arXiv](https://arxiv.org/abs/1907.02674)
7. Ramezanpour, K. et al. **EM-X-DL: Efficient Cross-Device Deep Learning Side-Channel Attack with Noisy EM Signatures.** 2020. [arXiv](https://arxiv.org/abs/2011.06139)
8. Meng, F. et al. **Adversarial Profiled Side-Channel Attack with Unsupervised Domain Adaptation.** ICCC, 2023. [IEEE DOI](https://doi.org/10.1109/ICCC59590.2023.10507514). 本地副本：`references/papers/ref/Adversarial_Profiled_Side-Channel_Attack_with_Unsupervised_Domain_Adaptation.pdf`。
9. Krček, M. and Perin, G. **Autoencoder-enabled Model Portability for Reducing Hyperparameter Tuning Efforts in Side-channel Analysis.** IACR ePrint 2023/019. [ePrint](https://eprint.iacr.org/2023/019)
10. Picek, S. et al. **No (good) loss no gain: systematic evaluation of loss functions in deep learning-based side-channel analysis.** Journal of Cryptographic Engineering, 2023. [DOI](https://doi.org/10.1007/s13389-023-00320-6)
11. Karayalcin, S. et al. **It’s a Kind of Magic: A Novel Conditional GAN Framework for Efficient Profiling Side-channel Analysis.** Extended version in local repository: `references/papers/ref/2023-1108.pdf`. The exact publication venue/version should be verified before citing in a final paper.

## 9. 与本项目相关的材料

- `docs/research/CDAN-SCA_research_assessment.md`：已有创新性风险和缺口判断；
- `experiments/cdan-sca-baselines/README.md`：当前基线覆盖和公平性规则；
- `experiments/cdan-sca-diagnosis/README.md`：GE/SR/NTGE 评价目标；
