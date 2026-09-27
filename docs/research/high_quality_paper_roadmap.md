# 面向高水平论文的研究规划

**规划日期：** 2026-09-19  
**适用项目：** SCA-UDA / CDAN-SCA 研究仓库  
**目标：** 形成一篇以跨设备建模侧信道分析为对象、以攻击指标和机制诊断为核心、具有可验证方法创新的论文。

## 1. 推荐的论文故事

### 1.1 一句话版本

> 研究高基数 profiling SCA 中域适应的真实目标：域不可辨识度下降是否真的带来更好的密钥恢复，以及如何通过类别结构、置信度可靠性和攻击指标约束减少负迁移。

### 1.2 论文主线

论文不应写成“实现了一个 CDAN 网络”，而应写成以下逻辑链：

1. 跨设备偏移使 Source Only 的目标域攻击失效；
2. DANN 等边缘对齐方法可能改善 domain confusion，但破坏 256 类泄露决策结构；
3. 普通分类准确率无法充分反映 key rank 和 GE；
4. 条件对齐是一个合理方向，但原始 CDAN 仍依赖不稳定的目标预测；
5. 因此需要一个具有 SCA-specific 可靠性或类别结构约束的适应框架；
6. 通过 GE/SR/NTGE、类别分离、校准、敏感性和跨设备验证判断它何时有效、何时失败。

### 1.3 推荐的贡献结构

最终论文争取形成 3 个主贡献：

1. **机制贡献：** 建立“域混淆改善不等于攻击改善”的实验诊断，并量化全局对齐对类别模态和密钥排序的影响。
2. **方法贡献：** 在 CDAN 基础上加入一个真正面向 SCA 的机制。首选方向是“校准感知的类别条件适应”或“类别原型保持的条件适应”，二者先做预实验再择一作为主方法。
3. **评价贡献：** 建立统一的攻击中心评估协议，覆盖受控域偏移、真实多设备数据、GE/SR/NTGE、校准和失败边界。

不要同时把校准、原型、课程学习、GE loss、多源适应和因果解耦都写成主创新。过多模块会使审稿人难以判断每个模块的作用，也会造成实验量失控。

## 2. 方法方向的选择原则

### 2.1 首选方向：校准感知的条件适应

原始熵加权隐含假设：低熵等于高正确率。但强域偏移下可能出现 confident wrong。建议先验证：

- 源验证集上的温度校准能否改善目标置信度排序；
- 预测熵和真实目标标签正确性是否相关；
- 校准后的权重是否减少错误目标样本对 domain loss 的污染；
- 最终改善是否体现在 GE/SR，而不仅是 accuracy。

如果成立，方法可以概括为：

```text
source classification
+ conditional domain alignment
+ calibrated target reliability weighting
```

这里的创新点不是“使用 entropy”，而是把目标置信度可靠性作为 SCA 域适应的显式问题，并用攻击结果验证它。

### 2.2 备选方向：类别原型保持

如果校准方向效果不稳定，使用类原型或类中心约束：

- 用源域有标签特征维护类别原型；
- 只对高置信目标样本建立目标类原型；
- 约束同类跨域距离，同时监控异类间隔；
- 将原型约束作为 CDAN 的补充，而不是完全替代域对抗。

该方向的论文价值来自“保持泄露类别几何结构”，而不是简单加入一个 contrastive loss。

### 2.3 仅作为评估规则：GE-aware checkpoint

GE-aware checkpoint 很适合提高论文严谨性，但目标 GE 不能用于无监督训练中的模型选择，否则会引入目标标签泄漏。建议：

- 主实验用源验证 loss、无标签目标熵或预注册的固定 epoch 选择；
- 目标 GE 只用于最终评估；
- 额外报告 oracle GE checkpoint 作为上界或诊断，不纳入主要性能结论。

## 3. 研究工作包

### WP0：实验契约和数据审计

**目标：** 让不同方法的结果可比较。

必须确定并写入配置：

- 源域、目标域、profiling/attack group；
- 目标字节和泄露标签定义；
- 轨迹窗口、采样率、裁剪、归一化；
- 源域有标签、目标域无标签的使用边界；
- 训练/适应/最终攻击评估的样本分离；
- 固定随机种子和数据顺序；
- GE 曲线的累积方式、步长、真实密钥读取方式。

**验收：** 用一份小数据能够在同一命令下生成完整 prediction、GE、SR、NTGE、配置和日志。

### WP1：官方模型与 Source Only 基线

**目标：** 证明数据、预处理、模型和 GE 评估链路正确。

建议首先复现 ASCAD 官方 CNN 结构，再以同一 backbone 运行 Source Only。官方基线不能只作为代码参考，必须在当前环境和当前评估脚本中产生可追踪结果。

**验收：** 源域监督性能合理，目标域攻击结果可由正式 GE/SR/NTGE 表达，且至少 3 个 seed 的结果可复现。

### WP2：统一 UDA trainer 和公平基线

统一 trainer 至少支持：

| 方法 | 适应信号 | 论文角色 |
|---|---|---|
| Source Only | 无目标域适应 | 原始迁移下界 |
| DANN | 特征 `f` 的边缘对齐 | 全局对抗基线 |
| CDAN | 条件表示 `T(f, g)` | 通用条件对抗基线 |
| CDPA/MMD | 统计/核分布差异 | SCA 领域基线 |
| AdaBN/统计对齐 | 低成本统计适应 | 简单实用基线 |
| Proposed | 条件结构 + SCA-specific 约束 | 主方法 |

所有方法必须共享：

- backbone；
- classifier；
- trace preprocessing；
- source/target batch 规模；
- optimizer 和训练预算；
- attack evaluation；
- seeds 和数据预算。

### WP3：机制诊断

用一个固定源-目标对先回答以下问题：

1. DANN 是否降低 domain accuracy？
2. domain accuracy 下降时，GE 是否同步下降？
3. 目标预测熵是否降低，还是只是变得更自信？
4. 类别中心距离、类内距离和类间距离如何变化？
5. 负迁移是否集中发生在某些 key classes、trace budgets 或 shift severity？

必须报告“支持假设”和“否定假设”的结果，不应只挑选有利曲线。

### WP4：一个 SCA-specific 方法创新

在 WP3 的结果基础上二选一：

**路线 A：校准感知适应**

- 源验证集温度校准或其他可靠性估计；
- 目标域条件对抗权重使用校准后的可靠性；
- 报告 ECE/Brier、熵-正确率关系和 GE。

**路线 B：原型保持适应**

- 维护源域类别原型；
- 对高置信目标样本建立动态目标原型；
- 同类跨域收缩、异类保持间隔；
- 报告类结构和密钥排名变化。

选择标准不是“哪个模块更复杂”，而是哪个模块能解释 WP3 中观察到的失败机制，并在多个域偏移上稳定复现。

### WP5：跨域和跨设备验证

按难度分层：

1. ASCAD fixed-key -> desync50；
2. ASCAD fixed-key -> desync100；
3. ASCAD -> Gaussian noise 或 clock jitter 变体；
4. CDPA XMEGA 多设备；
5. CDPA SAKURA AES 或其他真实采集条件；
6. 若条件允许，再加入自采集低成本设备。

前 3 项是受控域偏移，不应直接表述为真实跨设备泛化。后 2 项才能支撑更强的 cross-device claim。

### WP6：鲁棒性、敏感性和失败边界

至少做：

- source/target 样本量变化；
- target unlabeled budget 变化；
- `lambda_adv` 变化；
- conditional projection dimension 变化；
- 训练 epoch 和 checkpoint 规则；
- target class imbalance；
- shift severity；
- 3 到 5 个 seed；
- 标签打乱负对照；
- 无适应、错误标签和随机预测的 sanity check。

## 4. 预注册假设与判据

| 编号 | 假设 | 支持证据 | 否定或失败判据 |
|---|---|---|---|
| H1 | 边缘对齐会出现 accuracy/GE 不一致 | domain accuracy 降低但 GE 不改善或变差 | DANN 在多配对上稳定改善 GE |
| H2 | 条件对齐比边缘对齐更能保持类结构 | 类间距离和 GE 同时改善 | CDAN 只改善 accuracy，不改善 GE |
| H3 | 熵权重依赖置信度可靠性 | 校准后熵与正确率关系更稳定 | 强偏移下出现大量 confident wrong |
| H4 | 一个 SCA-specific 约束能减少负迁移 | 主方法在多个 seed/配对中改善 GE | 改善只存在于单一数据或单一 seed |
| H5 | 方法优势取决于泄露语义保持 | 同语义偏移有效，语义变化时退化 | 所有偏移都同样有效，或完全不具迁移性 |

如果 H1/H2 不成立，论文应转为“跨设备 SCA 域适应方法的系统比较与失效边界”，不要强行保留负迁移故事。

## 5. 推荐实验矩阵

### 主结果

| 数据层次 | 源域 | 目标域 | 主要作用 |
|---|---|---|---|
| 受控 | ASCAD fixed-key | desync50 | 主开发配对 |
| 受控 | ASCAD fixed-key | desync100 | 强时序偏移 |
| 受控 | ASCAD fixed-key | noise/jitter | 偏移类型变化 |
| 真实 | CDPA device A | device B/C | 跨设备验证 |
| 多设备 | CDPA 多源 | 留一设备目标 | 泛化边界，可选 |

### 方法和指标

每个数据配对至少比较：

- Source Only；
- DANN；
- CDAN；
- CDPA/MMD 或 AdaBN 中至少一个；
- Proposed；
- Proposed 去掉关键模块的消融。

每个方法至少报告：

- target accuracy，仅作辅助；
- domain accuracy 或 domain AUC；
- target entropy 和 ECE/Brier；
- GE 曲线；
- 固定轨迹数 SR；
- NTGE/达到 rank 0 的轨迹数；
- 3 个以上 seed 的均值、标准差和每次曲线。

## 6. 论文图表规划

### 主文建议图

1. **问题和方法概览图：** profiling device、target device、源标签、目标无标签和攻击评分流程。
2. **机制图：** Source Only、DANN、CDAN 在特征/类别结构上的差异。
3. **主 GE 曲线：** 受控偏移和真实设备分别展示，避免只给 accuracy bar chart。
4. **domain confusion 与 GE 的关系图：** 展示二者不等价的证据。
5. **校准/原型机制图：** 解释为什么 proposed 约束减少错误适应。
6. **失败边界图：** shift severity、目标数据量或类别不平衡下的方法区域。

### 主文建议表

1. 数据集、设备、标签、偏移类型和样本预算；
2. 方法和训练设置；
3. 主结果：GE/SR/NTGE；
4. 消融：条件模块、可靠性模块、投影、对抗权重；
5. 计算量、参数量和推理开销；
6. 与已有 SCA-specific UDA 方法的比较。

## 7. 结果决策树

### 情形 A：Proposed 稳定优于 DANN 和 Source Only

继续做真实跨设备验证、敏感性和论文写作。主张聚焦于“攻击中心的类别条件适应”。

### 情形 B：Proposed 只在受控偏移有效

论文主张限定为“受控域偏移下的机制和方法”，不能声称解决真实跨设备迁移；优先补充真实设备失败分析。

### 情形 C：Proposed 不优于 CDAN，但机制诊断成立

转为“边缘域适应在高基数 SCA 中的失效边界与攻击指标评估”论文。负结果和诊断本身仍有价值，但不能把 proposed 写成有效方法。

### 情形 D：数据、CDPA 或真实训练无法恢复

缩小为公开 ASCAD 受控偏移研究，并明确限制结论；不要使用“真实跨设备”标题或摘要表述。

## 8. 时间规划

按有可用 GPU、真实数据和 1 个主方法创新估算，建议 12 至 16 周：

| 阶段 | 周期 | 产出 |
|---|---:|---|
| 数据与环境恢复 | 1-2 周 | 数据契约、官方 CNN、GE sanity check |
| 统一 baseline | 2-3 周 | Source Only/DANN/CDAN/CDPA 结果 |
| 机制诊断 | 2 周 | GE、domain、类结构、熵分析 |
| 方法创新预实验 | 2-3 周 | 校准或原型方向的选择 |
| 主实验与消融 | 2-3 周 | 多 seed、多配对、失败边界 |
| 真实跨设备验证 | 1-2 周 | CDPA 或真实采集结果 |
| 写作与复核 | 2 周 | 论文、附录、代码和审计记录 |

若第 6 周仍没有真实 GE 主结果，应暂停扩展模型，优先解决数据和评估链路。

## 9. 论文结构

1. Introduction：现实问题、研究缺口、贡献；
2. Background：profiling SCA、GE、UDA、DANN/CDAN；
3. Problem Formulation：源域有标签、目标域无标签、域偏移和攻击目标；
4. Failure Diagnosis：边缘对齐、类别模态、accuracy/GE 不一致；
5. Proposed Method：条件适应和一个 SCA-specific 约束；
6. Experimental Protocol：数据、划分、基线、指标、seed 和数据预算；
7. Results：主结果、机制、消融、鲁棒性、真实设备；
8. Limitations：语义变化、目标数据量、计算开销、失败情况；
9. Conclusion：适用范围而非泛化承诺。

## 10. 投稿策略

### 第一目标

优先考虑 **TCHES/CHES 风格**投稿，因为该方向重视密码硬件安全、侧信道威胁模型、实验严谨性和公开数据复现。当前项目问题和已有 CDPA/ASCAD 材料与该范围匹配。

### 更高层级目标的附加要求

若要考虑 IEEE TIFS 或更广泛的高水平 venue，需要额外具备：

- 多个真实设备或采集条件；
- 更强的 SCA-specific 方法创新；
- 与 CDPA、DANN、CDAN、统计对齐和近期可迁移方法的完整对比；
- 多数据集、多 seed 和显著性/稳定性分析；
- 清晰的计算开销与安全意义讨论。

## 11. 立即执行清单

- [ ] 恢复 ASCAD 或准备可验证的 CDPA 数据路径；
- [ ] 修复 `cdan_model.py` 的输入和特征维度接口；
- [ ] 实现统一 source-target trainer；
- [ ] 让所有方法输出正式 GE/SR/NTGE；
- [ ] 将当前 pilot 标为 smoke/pilot，不再作为方法证据；
- [ ] 完成 Source Only/DANN/CDAN/CDPA 公平对比；
- [ ] 根据 H1-H3 结果选择校准或原型主线；
- [ ] 完成真实设备或明确限定论文范围；
- [ ] 建立 `findings.md`，记录每轮假设、结果和论文决策；
- [ ] 最后再冻结论文标题、方法名称和贡献表述。

