# CDAN-SCA Research Assessment / CDAN-SCA 研究评估

Date / 日期: 2026-08-13

Skills used / 使用的 skills:
- `autoresearch`: project-level research orchestration / 项目级研究编排
- `brainstorming-research-ideas`: structured gap and hypothesis generation / 结构化空白分析与假设生成
- `creative-thinking-for-research`: problem reformulation and extension discovery / 问题重构与扩展方向发掘

## 1. Positioning / 研究定位

**English:** CDAN-SCA targets negative transfer in cross-device profiled side-channel analysis. Its central move is to replace global marginal feature alignment with prediction-conditioned, class-aware adversarial alignment.

**中文：** CDAN-SCA 面向跨设备建模侧信道攻击中的负迁移问题。它的核心转向是：用预测条件化、类别感知的对抗对齐，替代传统的全局边缘特征分布对齐。

**Two-sentence pitch / 两句话版本：**

**English:** Cross-device profiled SCA often fails because a model trained on a source device cannot preserve its leakage-class decision boundaries on a target device. CDAN-SCA aligns the joint structure of features and class predictions, so target traces are adapted in a class-aware way instead of being globally mixed.

**中文：** 跨设备建模 SCA 经常失败，是因为源设备训练出的模型无法在目标设备上保持泄露类别的判别边界。CDAN-SCA 对齐特征与类别预测的联合结构，使目标域轨迹以类别感知方式迁移，而不是被全局混合。

## 2. Why the Idea Is Strong / 为什么这个想法有潜力

### 2.1 Real Problem / 真问题

**English:** The practical profiled SCA setting rarely gives the attacker a perfectly identical clone of the target device. Even small shifts caused by device variation, acquisition differences, probe placement, environmental factors, or countermeasures can make a strong profiling model fail.

**中文：** 真实建模 SCA 场景中，攻击者通常拿不到与目标设备完全一致的克隆设备。设备差异、采集差异、探针位置、环境变化或防护措施引入的轻微偏移，都可能让原本强的模型失效。

### 2.2 Plausible Failure Diagnosis / 失效机理解释合理

**English:** DANN learns domain-invariant features by confusing a domain discriminator, which tends to align the marginal feature distribution `P(f)`. In AES SCA with 256 identity classes, the class structure is high-cardinality and multimodal. Global alignment can reduce domain discrepancy while simultaneously collapsing or mixing class modes.

**中文：** DANN 通过混淆域判别器学习域不变特征，本质上倾向于对齐边缘特征分布 `P(f)`。但 AES SCA 的 identity leakage 通常是 256 类高基数、多模态分类问题。全局对齐可能降低域差异，却同时压缩或混淆类别模态。

### 2.3 Method Matches the Diagnosis / 方法与诊断对应

**English:** CDAN conditions the domain discriminator on both feature representation `f` and classifier prediction `g`, often through multilinear conditioning such as `f outer g`. This is better aligned with the need to preserve class boundaries while adapting across devices.

**中文：** CDAN 将特征表示 `f` 与分类器预测 `g` 一起输入域判别器，常见形式是多线性条件化，例如 `f outer g`。这更符合跨设备迁移中既要对齐域、又要保持类别边界的需求。

### 2.4 SCA-Specific Adaptation / SCA 场景适配点

**English:** Entropy weighting is important because target labels are unavailable. High-entropy target predictions should contribute less to the adversarial loss, otherwise wrong pseudo-label structure may contaminate alignment.

**中文：** 熵加权很重要，因为目标域没有标签。高熵、低置信度的目标预测不应过强参与对抗对齐，否则错误伪标签结构会污染迁移过程。

## 3. Proposed Method: Domain Adaptation / 提出方法：域自适应

### 3.1 Why Domain Adaptation Is Needed / 为什么需要域自适应

**English:** Device differences induce feature-distribution shift. Even for the same key-dependent leakage class, traces from the source and target devices may occupy different regions in feature space. A conventional deep profiling model trained only on the source device can therefore have very low target-device accuracy and poor key-rank performance.

**中文：** 设备差异会造成特征分布偏移。即使属于同一个密钥相关泄露类别，源设备和目标设备的波形也可能在特征空间中处于不同位置。因此，只在源域训练的传统深度学习建模攻击模型，在目标设备上的准确率和密钥排序表现可能很差。

### 3.2 DANN: Domain-Adversarial Neural Network / DANN：域对抗神经网络

**English:** DANN learns domain-invariant features. It uses a feature extractor `F`, a label classifier `C`, and a domain discriminator `D`. A gradient reversal layer makes `F` minimize the source classification loss while maximizing the domain discrimination loss. The alignment target is the marginal feature distribution `P(f)`.

**中文：** DANN 的目标是学习域不变特征。它包含特征提取器 `F`、标签分类器 `C` 和域判别器 `D`。梯度反转层使 `F` 一方面最小化源域分类损失，另一方面最大化域判别损失。其对齐目标是边缘特征分布 `P(f)`。

**Limitation / 问题：**

**English:** DANN performs global alignment. In 256-class SCA, different key hypotheses naturally form different feature modes. Global alignment can pull features from different classes closer together, damage the classifier boundary, and cause negative transfer.

**中文：** DANN 进行全局对齐。在 256 类 SCA 中，不同密钥假设天然对应不同特征模态。全局对齐可能强行拉近不同类别的特征，破坏分类边界，从而造成负迁移。

### 3.3 CDAN: Conditional Domain-Adversarial Network / CDAN：条件域对抗网络

**English:** CDAN is the foundation of our framework. Its key idea is to inject class-prediction information into the domain discriminator, so adaptation becomes class-aware. Instead of feeding only the feature `f` into the domain discriminator, CDAN feeds the multilinear conditioning representation `f outer g`, where `g` is the 256-dimensional classifier probability vector.

**中文：** CDAN 是我们框架的基础。它的核心思想是在域判别器中引入类别预测信息，使域适应变成类别感知的。域判别器的输入不再只是特征 `f`，而是多线性条件表示 `f outer g`，其中 `g` 是分类器输出的 256 维概率向量。

**Alignment target / 对齐目标：**

**English:** CDAN moves the alignment target from the marginal distribution `P(f)` to the joint structure `P(f, y)`, approximated through `f outer g`. In SCA terms, it aims to align feature distributions inside each key-byte hypothesis rather than mixing all classes globally.

**中文：** CDAN 将对齐目标从边缘分布 `P(f)` 转向联合结构 `P(f, y)`，并通过 `f outer g` 近似实现。在 SCA 中，这意味着对齐每个密钥字节假设内部的特征分布，而不是把所有类别全局混合。

**CDAN+E / 熵加权 CDAN：**

**English:** Following the entropy conditioning idea in CDAN-style adaptation, high-confidence target samples should receive larger adversarial weights, while high-entropy target samples should be down-weighted to reduce pseudo-label contamination.

**中文：** 借鉴 CDAN 风格适应中的熵条件化思想，高置信度目标样本应获得更大的对抗权重，而高熵目标样本应被降权，以减少错误伪标签污染。

### 3.4 CDAN-SCA Framework / CDAN-SCA 框架

**English:** CDAN-SCA combines a 1D residual CNN backbone, a conditional adversarial module, and entropy-weighted pseudo-label reliability control for side-channel traces.

**中文：** CDAN-SCA 将一维残差 CNN 主干、条件对抗模块和熵加权伪标签可靠性控制结合起来，专门适配侧信道波形数据。

| Module / 模块 | Function / 功能 | Technical detail / 技术细节 |
|---|---|---|
| Backbone / 主干网络 | Extract feature `f` from raw traces and output 256-class prediction `g` / 从原始波形提取特征 `f`，输出 256 类预测 `g` | 1D residual CNN / 一维残差 CNN |
| Conditional adversarial module / 条件对抗模块 | Feed `f outer g` to the domain discriminator / 将 `f outer g` 输入域判别器 | Random projection is used when `d_f x 256` is too large / 当 `d_f x 256` 过大时使用随机投影降维 |
| Entropy-weighted pseudo-labeling / 熵加权伪标签 | Weight target samples according to prediction confidence / 按预测置信度为目标域样本加权 | `w = 1 - H(g) / log 256` / `w = 1 - H(g) / log 256` |

**Entropy definition / 熵定义：**

```text
H(g) = - sum_{k=1}^{256} g_k log(g_k)
w = 1 - H(g) / log(256)
```

**English:** When the classifier is confident, `H(g)` is low and `w` is close to 1. When the classifier is uncertain, `H(g)` approaches `log(256)` and `w` approaches 0.

**中文：** 当分类器预测置信度高时，`H(g)` 较低，`w` 接近 1；当分类器不确定时，`H(g)` 接近 `log(256)`，`w` 接近 0。

## 4. Key Risks / 关键风险

### Risk 1: Novelty may look like direct CDAN transfer / 创新性可能被认为只是套用 CDAN

**English:** Reviewers may ask: "Is this just CDAN applied to SCA?" The answer must be built from SCA-specific evidence: DANN causes class-mode collision, domain confusion does not imply lower GE, and conditional alignment repairs this conflict.

**中文：** 审稿人可能会问：“这是不是只是把 CDAN 用到 SCA？”回应必须依赖 SCA 特有证据：DANN 会造成类别模态碰撞，域混淆改善不等于 GE 降低，而条件对齐能缓解这一冲突。

### Risk 2: Entropy is not always reliable / 熵不一定可靠

**English:** Under severe target shift, models may be confidently wrong. Entropy weighting helps only if confidence correlates with pseudo-label correctness or with improved key ranking.

**中文：** 在强目标域偏移下，模型可能“自信地错误”。熵加权只有在置信度与伪标签正确性或密钥排序改善相关时才有效。

### Risk 3: Conditional representation cost / 条件表示成本

**English:** Direct `f outer g` is expensive. With `feature_dim=4096` and `class_dim=256`, the exact outer product exceeds one million dimensions per sample. Random multilinear projection is practical but needs ablation.

**中文：** 直接计算 `f outer g` 成本很高。若 `feature_dim=4096`、`class_dim=256`，每个样本的外积维度超过一百万。随机多线性投影更实用，但必须做消融。

### Risk 4: SCA metrics differ from classification metrics / SCA 指标不同于普通分类指标

**English:** Accuracy is secondary. The main claims must be backed by guessing entropy, success rate, and traces to disclosure.

**中文：** 准确率是辅助指标。核心结论必须由猜测熵、成功率和达到密钥恢复所需轨迹数支撑。

### Risk 5: Cross-device validity / 跨设备有效性

**English:** Desynchronization or simulated noise experiments are useful, but the strongest claim requires real source-target device splits or carefully justified acquisition-condition shifts.

**中文：** 去同步或模拟噪声实验有价值，但最强的跨设备主张需要真实源-目标设备划分，或经过严格说明的采集条件偏移。

## 5. Current Repository State / 当前项目状态

**English:** The repository has a reasonable research skeleton and early CDAN-SCA components, but it is not yet a complete UDA experimental framework.

**中文：** 当前仓库已有合理的研究骨架和早期 CDAN-SCA 组件，但还不是完整的 UDA 实验框架。

Existing components / 已有内容:
- Project structure docs: `README.md`, `docs/PROJECT_STRUCTURE.md`
- Baseline areas: `baselines/ascad`, `baselines/cdpa`
- Model components: `conditional_feature.py`, `domain_discriminator.py`, `grl.py`, `cdan_model.py`
- Supervised training: `train_supervised.py`, `trainer_cadn_supervised.py`
- Rank/GE utilities: `utils/metrics.py`, `test_supervised.py`

Missing pieces / 缺口:
- Unified source-target UDA trainer / 统一源-目标 UDA 训练器
- Entropy-weighted adversarial loss / 熵加权对抗损失
- Fair baselines for Source Only, DANN, CDAN, CDAN-SCA / 公平基线对比
- Standard source-target dataset configuration / 标准源-目标数据配置
- Unified GE/SR/NTGE evaluation pipeline / 统一 GE、SR、NTGE 评估流程
- Experiment protocols and findings logs / 实验协议与发现日志

## 6. Related Work Signals / 相关工作信号

### DANN

**English:** DANN learns features that are discriminative for the source task and indiscriminate with respect to source-target domain identity through a gradient reversal layer. This is the correct baseline for global adversarial alignment.

**中文：** DANN 通过梯度反转层学习对源任务有判别性、同时对源/目标域身份不可区分的特征。它是全局对抗对齐的核心基线。

Source / 来源: https://jmlr.org/beta/papers/v17/15-239.html

### CDAN

**English:** CDAN explicitly argues that adversarial adaptation may struggle with multimodal classification distributions, and proposes conditioning the domain discriminator on classifier predictions through multilinear conditioning and entropy conditioning.

**中文：** CDAN 明确指出，对抗式域适应在多模态分类分布上可能遇到困难，并提出用分类器预测对域判别器进行条件化，包括多线性条件化和熵条件化。

Source / 来源: https://proceedings.neurips.cc/paper_files/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html

### Cross-device profiled SCA with UDA / 跨设备建模 SCA 与 UDA

**English:** CDPA frames cross-device profiled SCA as a domain discrepancy problem and uses MMD-based fine-tuning with unlabeled target traces. It reports evaluation on multiple devices and datasets, making it a key SCA-specific baseline.

**中文：** CDPA 将跨设备建模 SCA 表述为域差异问题，并使用基于 MMD 的无标签目标轨迹微调。它在多设备和多数据集上评估，是该方向的重要 SCA 基线。

Source / 来源: https://www.researchgate.net/publication/364384328_Cross-device_profiled_side-channel_attack_with_unsupervised_domain_adaptation

### Recent feature/statistics alignment directions / 近期特征与统计对齐方向

**English:** Recent work also explores feature/statistical alignment such as AdaBN and multi-order moment alignment with MMD/CORAL/conditional entropy. These works suggest that feature-level mismatch remains active, but they also leave room for class-conditional and GE-aware alignment.

**中文：** 近期工作也在探索 AdaBN、MMD/CORAL/条件熵等特征或统计对齐方法。这说明特征级失配仍是活跃问题，同时也给类别条件对齐和 GE 感知对齐留下空间。

Sources / 来源:
- AdaBN cross-device SCA: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003304298
- RFA-SCA: https://www.techscience.com/cmc/v88n3/68078/html

## 7. Core Hypotheses / 核心研究假设

### H1: DANN creates class-mode collision / DANN 会造成类别模态碰撞

**English:** DANN may reduce domain discrepancy while damaging inter-class separation. If domain discriminator accuracy drops but GE worsens or class separation decreases, this supports the negative-transfer diagnosis.

**中文：** DANN 可能降低域差异，却破坏类别间分离。如果域判别器准确率下降，但 GE 变差或类间分离下降，就支持负迁移诊断。

### H2: Conditional alignment preserves SCA decision boundaries / 条件对齐更能保持 SCA 判别边界

**English:** CDAN-SCA should outperform DANN because the adversarial signal is conditioned on leakage-class predictions.

**中文：** CDAN-SCA 应优于 DANN，因为其对抗信号由泄露类别预测条件化。

### H3: Entropy weighting helps only when confidence is meaningful / 熵加权依赖置信度可靠性

**English:** Entropy weighting should improve moderate-shift adaptation, but may fail under severe shift if the classifier is confidently wrong.

**中文：** 熵加权应能改善中等偏移场景，但在严重偏移下，如果分类器自信地错误，可能失效。

### H4: Random multilinear projection has a dimension-stability trade-off / 随机多线性投影存在维度与稳定性权衡

**English:** Projection reduces computation, but projection dimension and trainability may affect adaptation stability.

**中文：** 投影能降低计算量，但投影维度和是否可训练会影响适应稳定性。

### H5: CDAN-SCA is best when leakage semantics are preserved / CDAN-SCA 最适合同语义泄露迁移

**English:** Conditional alignment should help when source and target share the same leakage semantics. If masking or leakage model mismatch changes the label semantics, the method may need additional constraints.

**中文：** 当源域和目标域共享相同泄露语义时，条件对齐最有效。若掩码或泄露模型不匹配改变了标签语义，则需要额外约束。

## 8. Claim Calibration / 论文表述校准

Current draft language should be softened until experiments are complete.

当前中文构想中的部分表述需要在实验完成前适当收敛。

Recommended changes / 建议修改:
- "从根本上杜绝了负迁移" -> "显著缓解由全局边缘对齐引发的负迁移"
- "所有跨设备场景下均显著优于" -> "在所评估的跨设备场景中取得稳定改进"
- "大量、完整且严格" -> replace with concrete datasets, baselines, metrics, and repetitions / 用具体数据集、基线、指标和重复次数替代

Suggested paper claim / 建议论文主张:

**English:** We show that marginal adversarial alignment can improve domain confusion while degrading key-rank performance in high-cardinality SCA. CDAN-SCA mitigates this conflict by conditioning adaptation on leakage-class predictions and reducing the influence of uncertain target samples.

**中文：** 我们表明，在高基数 SCA 分类任务中，边缘对抗对齐可能改善域混淆却恶化密钥排序表现。CDAN-SCA 通过将适应过程条件化于泄露类别预测，并降低不确定目标样本的影响，缓解了这一冲突。
