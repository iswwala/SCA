# CDAN-SCA Research Plan / CDAN-SCA 研究规划

Date / 日期: 2026-08-13

Skills used / 使用的 skills:
- `autoresearch`: two-loop project planning / 双循环项目规划
- `brainstorming-research-ideas`: gap-driven direction generation / 基于研究空白的方向生成
- `creative-thinking-for-research`: extension discovery through reformulation, analogy, and constraint manipulation / 通过重构、类比和约束操纵发掘扩展方向

## 1. Project Goal / 项目目标

**English:** Build and validate CDAN-SCA, a conditionally adversarial unsupervised domain adaptation framework for cross-device profiled side-channel attacks, with a special focus on why marginal alignment fails in 256-class leakage classification.

**中文：** 构建并验证 CDAN-SCA：一种面向跨设备建模侧信道攻击的条件对抗无监督域适应框架，重点解释边缘分布对齐为何会在 256 类泄露分类中失效。

## 2. Main Research Question / 核心研究问题

**English:** Can class-conditional adversarial alignment reduce cross-device negative transfer in profiled SCA better than global marginal alignment, and under what shift conditions does it help or fail?

**中文：** 与全局边缘对齐相比，类别条件对抗对齐是否能更有效降低建模 SCA 中的跨设备负迁移？它在哪些偏移条件下有效，又在哪些条件下失败？

## 3. Expected Contributions / 预期贡献

1. **Failure diagnosis / 失效诊断**
   - English: Show that DANN-style marginal alignment can damage 256-class SCA decision boundaries.
   - 中文：证明 DANN 式边缘对齐可能破坏 256 类 SCA 的判别边界。

2. **Method / 方法**
   - English: Propose CDAN-SCA with conditional multilinear domain discrimination and entropy-weighted target adaptation.
   - 中文：提出包含条件多线性域判别和目标域熵加权的 CDAN-SCA。

3. **Evaluation / 评估**
   - English: Evaluate with SCA-native metrics: guessing entropy, success rate, and traces to disclosure.
   - 中文：使用 SCA 原生指标评估：猜测熵、成功率和恢复密钥所需轨迹数。

4. **Boundary map / 边界图谱**
   - English: Identify shift conditions where CDAN-SCA helps, degrades, or needs additional constraints.
   - 中文：识别 CDAN-SCA 在哪些偏移条件下有效、退化或需要额外约束。

## 4. Method Definition / 方法定义

### 4.1 Why Domain Adaptation / 为什么需要域自适应

**English:** Cross-device SCA suffers from feature-distribution shift. Source and target traces associated with the same key-dependent leakage class may appear in different feature-space locations. A source-only model therefore often fails on the target device.

**中文：** 跨设备 SCA 面临特征分布偏移。源域和目标域中属于同一密钥相关泄露类别的波形，可能在特征空间中处于不同位置。因此，仅源域训练的模型往往在目标设备上失效。

### 4.2 DANN Baseline / DANN 基线

**English:** DANN consists of a feature extractor `F`, a label classifier `C`, and a domain discriminator `D`. A gradient reversal layer makes `F` minimize classification loss while maximizing domain discrimination loss. Its alignment target is the marginal feature distribution `P(f)`.

**中文：** DANN 包含特征提取器 `F`、标签分类器 `C` 和域判别器 `D`。梯度反转层使 `F` 同时最小化分类损失并最大化域判别损失。它的对齐目标是边缘特征分布 `P(f)`。

**Key weakness / 核心弱点：**

**English:** In 256-class SCA, global marginal alignment can pull different key-class modes together and damage the classifier boundary, producing negative transfer.

**中文：** 在 256 类 SCA 中，全局边缘对齐可能把不同密钥类别模态拉近，破坏分类边界，产生负迁移。

### 4.3 CDAN Foundation / CDAN 框架基础

**English:** CDAN conditions the domain discriminator on both feature `f` and classifier output `g`. The discriminator input is the multilinear representation `f outer g`, where `g` is a 256-dimensional probability vector.

**中文：** CDAN 将域判别器条件化于特征 `f` 和分类器输出 `g`。域判别器输入为多线性表示 `f outer g`，其中 `g` 是 256 维概率向量。

**Alignment target / 对齐目标：**

**English:** The alignment target becomes the joint structure `P(f, y)`, so adaptation is performed inside class-conditioned modes rather than globally.

**中文：** 对齐目标变成联合结构 `P(f, y)`，因此适应发生在类别条件模态内部，而不是全局混合。

### 4.4 CDAN-SCA Modules / CDAN-SCA 模块

| Module / 模块 | Function / 功能 | Technical detail / 技术细节 |
|---|---|---|
| Backbone / 主干网络 | Extract feature `f` and predict 256-class probability `g` / 提取特征 `f` 并输出 256 类概率 `g` | 1D residual CNN for traces / 面向波形的一维残差 CNN |
| Conditional adversarial module / 条件对抗模块 | Feed `f outer g` into domain discriminator / 将 `f outer g` 输入域判别器 | Random projection when `d_f x 256` is too large / 当 `d_f x 256` 过大时使用随机投影降维 |
| Entropy-weighted pseudo-labeling / 熵加权伪标签 | Down-weight uncertain target predictions / 降低低置信目标预测的权重 | `w = 1 - H(g) / log 256` |

Entropy weight / 熵权重:

```text
H(g) = - sum_{k=1}^{256} g_k log(g_k)
w = 1 - H(g) / log(256)
```

**English:** The weight is close to 1 for confident target predictions and close to 0 for uncertain target predictions.

**中文：** 当目标域预测置信度高时，该权重接近 1；当预测不确定时，该权重接近 0。

## 5. Autoresearch Structure / 自主研究结构

### Inner Loop / 内层循环

**Goal / 目标:** Run tightly scoped experiments with one hypothesis, one controlled change, and measurable GE/SR outcomes.

每轮只验证一个假设、一个受控变化，并用 GE/SR 等指标衡量结果。

Protocol for each experiment / 每个实验协议:
- Hypothesis / 假设
- Source-target pair / 源-目标域配对
- Method variant / 方法变体
- Fixed hyperparameters / 固定超参数
- Expected outcome / 预期结果
- Metrics / 指标
- Failure condition / 失败判据

### Outer Loop / 外层循环

**Goal / 目标:** Every 3-5 experiments, synthesize what worked, what failed, and whether the paper story changed.

每 3-5 个实验后综合反思：哪些有效、哪些失败、论文故事是否需要调整。

Outer-loop questions / 外层问题:
- Did lower domain discrepancy actually reduce GE? / 域差异降低是否真的降低 GE？
- Did conditional alignment preserve class separation? / 条件对齐是否保持了类间分离？
- Did entropy weighting suppress bad target samples or amplify confident errors? / 熵加权是抑制了坏样本，还是放大了自信错误？
- Does the result support deepening, broadening, pivoting, or concluding? / 结果支持深入、拓展、转向还是收束？

## 6. Work Packages / 工作包

### WP1: Experimental Contract / 实验契约

**Deliverable / 产出:** `experiments/cdan-sca-diagnosis/README.md`

Tasks / 任务:
- Define source-target dataset pairs / 定义源-目标数据配对
- Define trace window and target byte / 定义轨迹窗口和目标字节
- Define label type: ID or HW / 定义标签类型：ID 或 HW
- Define canonical metrics / 定义标准指标
- Define output layout / 定义输出目录结构

Minimum metrics / 最小指标集:
- Target accuracy / 目标域准确率
- Guessing entropy curve / 猜测熵曲线
- Success rate at fixed trace counts / 固定轨迹数下成功率
- Number of traces to GE=1 / 达到 GE=1 所需轨迹数
- Domain discriminator accuracy / 域判别器准确率
- Class separation or collision metric / 类间分离或碰撞指标

### WP2: Unified Baselines / 统一基线

**Deliverable / 产出:** one trainer/config interface for four methods / 一个训练器与配置接口支持四种方法

Methods / 方法:
1. Source Only / 仅源域训练
2. DANN / 全局边缘对抗对齐
3. CDAN without entropy / 无熵权重条件对齐
4. CDAN-SCA / 条件对齐 + 熵权重

Fairness rules / 公平性规则:
- Same backbone / 相同主干网络
- Same classifier / 相同分类器
- Same optimizer and learning schedule / 相同优化器和学习率计划
- Same source/target batch size / 相同源/目标 batch
- Same preprocessing / 相同预处理
- Same evaluation script / 相同评估脚本

### WP3: CDAN-SCA Trainer / CDAN-SCA 训练器

**Deliverable / 产出:** `src/framework/trainers/trainer_uda.py`

Training step / 训练步骤:
1. Load source labeled batch `(x_s, y_s)` / 加载源域有标签 batch
2. Load target unlabeled batch `x_t` / 加载目标域无标签 batch
3. Compute features `f_s, f_t` / 计算特征
4. Compute predictions `g_s, g_t` / 计算类别预测
5. Apply source classification loss / 源域分类损失
6. Build domain input:
   - DANN: `f`
   - CDAN/CDAN-SCA: `T(f, g) = f outer g`, with random projection if needed / `T(f, g) = f outer g`，必要时使用随机投影
7. Apply gradient reversal / 应用梯度反转
8. Compute weighted domain loss / 计算加权域损失
9. Optimize total loss / 优化总损失

Loss / 损失:

```text
L = L_cls_source + lambda_adv * L_domain
```

Entropy-weighted target domain loss / 熵加权目标域损失:

```text
H(g_t) = -sum_k g_tk log(g_tk)
w_t = 1 - H(g_t) / log(256)
```

**English:** For Source Only and DANN, `T(f, g)` and entropy weighting are disabled. For CDAN, `T(f, g)` is enabled but entropy weighting is disabled. For CDAN-SCA, both conditional adversarial input and entropy weighting are enabled.

**中文：** 对 Source Only 和 DANN，关闭 `T(f, g)` 与熵加权；对 CDAN，启用 `T(f, g)` 但关闭熵加权；对 CDAN-SCA，同时启用条件对抗输入和熵加权。

### WP4: Diagnosis Experiments / 诊断实验

**Goal / 目标:** Prove or refute the central mechanism.

Primary diagnosis / 主诊断:
- If DANN lowers domain discrepancy but worsens GE or class separation, then global alignment is harmful.
- 如果 DANN 降低域差异但恶化 GE 或类间分离，则说明全局对齐有害。

Required plots / 必要图:
- GE curves for all methods / 所有方法的 GE 曲线
- Domain accuracy vs GE / 域准确率与 GE 的关系
- Class centroid separation before/after adaptation / 适应前后的类别中心分离
- Feature visualization for selected classes / 选定类别的特征可视化
- Entropy distribution of target samples / 目标样本熵分布

### WP5: Ablation and Sensitivity / 消融与敏感性

Required ablations / 必做消融:
- No adaptation / 无适应
- DANN vs CDAN / DANN 与 CDAN
- CDAN without entropy vs CDAN-SCA / 无熵 CDAN 与 CDAN-SCA
- Projection dimension: 64, 128, 256 / 投影维度
- `lambda_adv`: weak, medium, strong / 对抗权重
- Entropy strategy: none, raw entropy, thresholded entropy, temperature-scaled entropy / 熵策略

### WP6: Paper and Presentation / 论文与展示

Deliverables / 产出:
- `docs/research/findings.md`
- `outputs/results/cdan-sca-diagnosis/`
- Architecture figure / 方法结构图
- GE comparison figure / GE 对比图
- Ablation table / 消融表
- Related work table / 相关工作对比表
- Paper outline / 论文大纲

## 7. Two-Week Pilot / 两周试点计划

### Week 1 / 第一周

Day 1-2:
- Finalize one source-target pair / 确定一个源-目标配对
- Create experiment README / 创建实验 README
- Verify Source Only training and GE evaluation / 验证 Source Only 训练和 GE 评估

Day 3-4:
- Implement DANN path in UDA trainer / 实现 DANN 训练路径
- Run DANN smoke test / 跑 DANN 小规模冒烟实验
- Save domain accuracy and GE / 保存域准确率与 GE

Day 5-7:
- Implement CDAN conditional path / 实现 CDAN 条件路径
- Add projection-dimension config / 加入投影维度配置
- Compare Source Only, DANN, CDAN on one pair / 在一个配对上比较三种方法

### Week 2 / 第二周

Day 8-9:
- Add entropy weighting / 加入熵权重
- Compare CDAN vs CDAN-SCA / 比较 CDAN 与 CDAN-SCA

Day 10-11:
- Add class separation diagnostics / 加入类间分离诊断
- Add entropy-confidence analysis / 加入熵-置信度分析

Day 12-13:
- Run first ablations / 跑第一批消融
- Generate GE and feature figures / 生成 GE 与特征图

Day 14:
- Update findings / 更新 findings
- Decide: deepen, broaden, pivot, or conclude pilot / 决定深入、拓展、转向或结束试点

## 8. Extension Directions / 可扩展研究方向

### Direction 1: Calibration-Aware CDAN-SCA / 校准感知 CDAN-SCA

**Problem / 问题:** Entropy weighting assumes confidence is reliable, but target-domain predictions may be confidently wrong.

**Idea / 思路:** Add calibration before entropy weighting: temperature scaling, source validation calibration, target consistency, or confidence thresholding.

**Hypothesis / 假设:** Calibration-aware entropy weighting improves robustness under severe device shift.

**Experiments / 实验:**
- Entropy vs correctness plot using hidden target labels only for evaluation / 用隐藏目标标签评估熵与正确性的关系
- Expected calibration error on source validation and target evaluation / 源验证与目标评估 ECE
- GE comparison with and without calibration / 校准前后 GE 对比

**Priority / 优先级:** High. This directly strengthens the weak point of CDAN-SCA.

### Direction 2: GE-Aware Adaptation / GE 感知域适应

**Problem / 问题:** Classification accuracy does not fully represent attack success.

**Idea / 思路:** Use GE or traces-to-disclosure as a validation/checkpoint selection signal, and later explore differentiable key-rank surrogates.

**Hypothesis / 假设:** GE-aware checkpoint selection improves attack efficiency even when accuracy changes are small.

**Experiments / 实验:**
- Select checkpoints by source validation loss, target pseudo entropy, and target GE evaluation / 按源验证损失、目标伪熵、目标 GE 选择 checkpoint
- Compare traces to GE=1 / 比较达到 GE=1 的轨迹数

**Priority / 优先级:** High for paper credibility; medium for method novelty.

### Direction 3: Prototype-Conditioned Alignment / 原型条件对齐

**Problem / 问题:** Classifier predictions may be noisy early in training.

**Idea / 思路:** Maintain source class prototypes in feature space, then align high-confidence target samples to class prototypes before or alongside adversarial alignment.

**Hypothesis / 假设:** Prototype constraints reduce pseudo-label drift and improve class separation.

**Experiments / 实验:**
- Add source prototype loss / 加入源类原型损失
- Add high-confidence target prototype attraction / 加入高置信目标样本原型吸引
- Compare class centroid separation and GE / 比较类中心分离与 GE

**Priority / 优先级:** High as a second-paper or strengthened main-method variant.

### Direction 4: Shift Boundary Benchmark / 偏移边界基准

**Problem / 问题:** Existing work often reports a few datasets but does not map when UDA helps or hurts.

**Idea / 思路:** Create controlled shifts: time desynchronization, amplitude scaling, additive noise, sampling drift, probe shift, masking mismatch.

**Hypothesis / 假设:** CDAN-SCA helps most under shifts that preserve leakage semantics, but fails under shifts that alter label semantics.

**Experiments / 实验:**
- Sweep shift severity / 扫描偏移强度
- Plot method performance regions / 绘制方法有效区域
- Identify failure boundaries / 找出失效边界

**Priority / 优先级:** High for a strong evaluation section.

### Direction 5: Multi-Source CDAN-SCA / 多源 CDAN-SCA

**Problem / 问题:** Real attackers may have several profiling devices or acquisition settings.

**Idea / 思路:** Use multiple labeled source devices and one unlabeled target, combining conditional adversarial alignment with domain generalization.

**Hypothesis / 假设:** Multiple sources reduce dependence on one source distribution and improve target robustness.

**Experiments / 实验:**
- Leave-one-device-out setup / 留一设备测试
- Single-source vs multi-source comparison / 单源与多源对比
- Source diversity vs target GE / 源域多样性与目标 GE

**Priority / 优先级:** Medium-high, depending on device data availability.

### Direction 6: Class-Aware Curriculum Adaptation / 类别感知课程式适应

**Problem / 问题:** Adapting all target samples from the beginning may inject noise.

**Idea / 思路:** Start with high-confidence, high-separation classes or samples, then gradually expand to harder target samples.

**Hypothesis / 假设:** Curriculum adaptation reduces early pseudo-label contamination.

**Experiments / 实验:**
- Entropy threshold schedule / 熵阈值调度
- Per-class confidence quota / 每类置信度配额
- Compare early-training stability / 比较早期训练稳定性

**Priority / 优先级:** Medium.

### Direction 7: Causal or Leakage-Mechanism Alignment / 因果或泄露机制对齐

**Problem / 问题:** Domain adaptation may align device artifacts rather than true leakage mechanisms.

**Idea / 思路:** Separate invariant leakage-related features from device-specific nuisance features using reconstruction, contrastive objectives, or intervention-style augmentations.

**Hypothesis / 假设:** Explicitly suppressing nuisance variation improves cross-device generalization beyond adversarial alignment.

**Experiments / 实验:**
- Trace augmentations for amplitude/time/device nuisance / 幅值、时间、设备扰动增强
- Contrastive invariance test / 对比不变性测试
- Leakage-sensitivity visualization / 泄露敏感性可视化

**Priority / 优先级:** Medium, but potentially high novelty.

### Direction 8: Open-Set or Partial-Class Target Adaptation / 开集或部分类目标适应

**Problem / 问题:** In practical SCA, target traces may not cover all classes evenly, especially under limited target collection.

**Idea / 思路:** Study whether conditional alignment fails when target pseudo-label coverage is sparse or biased.

**Hypothesis / 假设:** CDAN-SCA needs class-balance correction when target pseudo-labels are concentrated in a small subset of classes.

**Experiments / 实验:**
- Subsample target traces to create class imbalance / 子采样目标轨迹制造类别不平衡
- Add class-balanced entropy weighting / 加入类别平衡熵权重
- Measure per-class collision and GE / 测量逐类碰撞与 GE

**Priority / 优先级:** Medium.

## 9. Ranked Research Directions / 方向优先级

| Rank / 排名 | Direction / 方向 | Why / 理由 |
|---|---|---|
| 1 | Core CDAN-SCA diagnosis / 核心 CDAN-SCA 诊断 | Needed to prove main paper claim / 支撑主论文核心主张 |
| 2 | Calibration-aware entropy / 校准感知熵权重 | Directly addresses pseudo-label risk / 直接解决伪标签风险 |
| 3 | Shift boundary benchmark / 偏移边界基准 | Makes evaluation rigorous and memorable / 让评估更严谨且有辨识度 |
| 4 | Prototype-conditioned alignment / 原型条件对齐 | Strong method extension if CDAN-SCA is unstable / 若 CDAN-SCA 不稳，是强扩展 |
| 5 | GE-aware adaptation / GE 感知适应 | Aligns training/evaluation with SCA goals / 让训练评估更贴合 SCA |
| 6 | Multi-source CDAN-SCA / 多源 CDAN-SCA | Valuable if enough device data exists / 数据足够时很有价值 |
| 7 | Causal leakage alignment / 因果泄露对齐 | High novelty but higher uncertainty / 新颖但不确定性更高 |
| 8 | Open-set or partial-class target adaptation / 开集或部分类目标适应 | Useful robustness angle / 有用的鲁棒性补充 |

## 10. Paper Outline / 论文大纲

1. Introduction / 引言
   - Cross-device SCA is practical and difficult / 跨设备 SCA 现实且困难
   - Global UDA can fail in high-cardinality leakage classification / 全局 UDA 在高基数泄露分类中可能失败
   - CDAN-SCA aligns class-conditioned joint structure / CDAN-SCA 对齐类别条件联合结构

2. Background / 背景
   - Profiled SCA and GE / 建模 SCA 与 GE
   - DANN and marginal alignment / DANN 与边缘对齐
   - CDAN, CDAN+E, and conditional alignment / CDAN、CDAN+E 与条件对齐

3. Failure Analysis / 失效分析
   - Why `P(f)` alignment conflicts with 256-class SCA / 为什么 `P(f)` 对齐与 256 类 SCA 冲突
   - Empirical diagnosis with DANN / DANN 实证诊断

4. Method / 方法
   - Feature extractor and classifier / 特征提取器与分类器
   - Conditional adversarial module using `f outer g` / 基于 `f outer g` 的条件对抗模块
   - Random projection for high-dimensional conditioning / 高维条件表示的随机投影
   - Entropy-weighted pseudo-label mechanism with `w = 1 - H(g) / log 256` / 使用 `w = 1 - H(g) / log 256` 的熵加权伪标签机制
   - Training objective / 训练目标

5. Experiments / 实验
   - Datasets and source-target pairs / 数据集与源-目标配对
   - Baselines / 基线
   - Metrics / 指标
   - Main results / 主结果
   - Ablations / 消融
   - Boundary analysis / 边界分析

6. Discussion / 讨论
   - When CDAN-SCA works / 何时有效
   - When it fails / 何时失效
   - Practical security implications / 实际安全评估意义

## 11. Immediate Next Tasks / 立即行动

1. Create `experiments/cdan-sca-diagnosis/README.md`.
2. Implement unified `trainer_uda.py`.
3. Add configs for `source_only`, `dann`, `cdan`, and `cdan_sca`.
4. Standardize GE evaluation output.
5. Run one small source-target experiment to validate the pipeline.
6. Generate first GE curves and class-separation diagnostics.
7. Update this plan after the first outer-loop reflection.

中文：
1. 创建 `experiments/cdan-sca-diagnosis/README.md`。
2. 实现统一的 `trainer_uda.py`。
3. 添加 `source_only`、`dann`、`cdan`、`cdan_sca` 配置。
4. 标准化 GE 评估输出。
5. 跑一个小规模源-目标实验验证流程。
6. 生成第一批 GE 曲线和类间分离诊断图。
7. 第一次外层反思后更新本文档。
