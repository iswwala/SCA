# SCA-UDA 项目投稿可行性评估

**评估日期：** 2026-09-19  
**评估对象：** 当前仓库 `/home/iswwala/projects/SCA` 及其现有实验记录  
**评估目标：** 判断项目体量是否足以发展为论文，明确当前证据边界、主要缺口和投稿前必须完成的工作。

## 1. 结论摘要

当前项目**具备发展为论文的研究体量，但尚未达到可投稿状态**。

更准确地说，项目已经有：

- 明确且有现实意义的问题：跨设备建模侧信道分析中的域偏移与负迁移；
- 可以形成论文故事的理论切入点：全局边缘对齐不一定等价于密钥恢复能力提升，256 类泄露分类中的类别结构可能被破坏；
- 公开数据、官方 ASCAD 代码、CDPA 多设备基线材料和初步实验目录；
- CDAN、梯度反转、条件特征、熵加权等方法组件和研究规划。

但当前还不能声称：

- CDAN-SCA 已经正确实现并完成真实训练；
- CDAN-SCA 已经优于 Source Only、DANN 或 CDPA/MMD；
- 当前 pilot 已经证明了负迁移机制；
- 项目已经完成真实跨设备验证；
- 现有 GE 曲线代表真实数据上的科学结果。

目前最合适的论文定位是：

> **以高基数建模 SCA 中“域混淆指标与密钥恢复指标不一致”为核心问题，系统诊断边缘域适应、条件域适应和 SCA 原生攻击指标之间的关系，并提出一个经过消融与跨设备验证的、具有 SCA-specific 约束的域适应方法。**

若按照本文规划补齐真实实验，首要投稿目标可以放在 **TCHES/CHES 方向**；若要冲击更广泛的高水平机器学习或安全期刊，需要增加真实设备、多源跨设备、强基线和更明确的方法创新，不能仅靠“将 CDAN 应用于 SCA”作为贡献。

## 2. 使用的研究 skills

本评估结合了以下 skills 的工作方式：

- `autoresearch`：把项目拆成内层实验循环、外层研究判断和可追踪的 findings；
- `ml-paper-writing`：从论文贡献、证据链、基线公平性和审稿风险反推实验设计；
- `robustness-checker`：优先检查数据划分、指标、随机种子、敏感性和失败边界；
- `result-report-generator`：区分实验事实、诊断结果、假设和不能支持的结论；
- `related-paper-analyzer`：按问题、方法、数据、指标和局限组织领域综述。

## 3. 项目当前资产盘点

| 方面 | 当前已有内容 | 证据等级 | 对论文的意义 |
|---|---|---:|---|
| 研究问题 | 跨设备 SCA、无监督域适应、类别结构与负迁移 | 较强 | 可以形成清晰问题定义 |
| 理论切入点 | DANN 对齐 `P(f)`，CDAN 对齐条件联合结构 | 中等 | 可作为机制假设，但需要实验证明 |
| 公开数据材料 | ASCAD 官方代码、CDPA 数据包和说明 | 中等 | 具备复现实验入口 |
| 模型组件 | CNN 特征提取器、分类器、GRL、条件特征、域判别器 | 部分完成 | 说明有工程基础，不代表方法可运行 |
| 训练流程 | 有监督训练脚本和 baseline smoke；缺统一 UDA trainer | 不足 | 是当前最大工程缺口 |
| 评价指标 | 有 GE/SR/NTGE 评估脚本雏形和历史输出 | 部分完成 | 需要统一输入、密钥、字节和协议 |
| 实验结果 | 合成 protocol smoke、小规模 Source Only/DANN pilot | 很弱 | 只能证明脚本链路，不能支撑方法结论 |
| 真实设备验证 | 仓库内有 CDPA 公开材料，但当前数据路径不可用 | 未完成 | 真实跨设备主张尚未成立 |
| 复现性 | 有 README、结果目录和部分日志 | 部分完成 | 需要配置、版本、seed、数据哈希和完整命令 |

## 4. 当前实验结果应如何解释

### 4.1 Protocol smoke

`outputs/results/cdan-sca-diagnosis/protocol_check.json` 明确标记为 `synthetic_smoke`，并写明：

- ASCAD 源域和目标域文件不可用；
- TensorFlow/h5py 训练没有执行；
- GE、SR 和 NTGE 是确定性合成值；
- 合成指标不是科学证据。

因此，这部分结果只能证明评估字段和文件输出链路存在，不能用于论文中的性能表。

### 4.2 Baseline pilot

`outputs/results/cdan-sca-baselines/PILOT_ANALYSIS.md` 记录了 ASCAD profiling 到 `ASCAD_desync50` 的小规模 Source Only/DANN 对比：

- 源域和目标域各约 512 条轨迹；
- 评估约 256 条轨迹；
- 5 个 epoch，3 个 seed；
- target accuracy 与 `label-rank proxy` 差异很小。

这一结果最多支持：在当前极小规模设置和当前实现下，尚未观察到稳定的 DANN 收益。它不支持以下更强结论：

- DANN 一定无效；
- DANN 一定导致负迁移；
- CDAN-SCA 一定优于 DANN；
- `label-rank proxy` 等价于正式 key-byte GE。

此外，当前 pilot 脚本 `experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py`：

- 使用单独定义的三层 CNN，而不是 `src/framework/models/feature_extractor.py` 中的维护主干；
- 默认按每条轨迹单独标准化，需要和官方 ASCAD 预处理口径核对；
- 只输出 target accuracy 和逐样本 label-rank proxy；
- 没有把正式 key-byte GE、SR、NTGE 和 domain accuracy 统一写入结果；
- batch 生成会丢弃不足一个 batch 的尾部样本。

所以这部分应被归档为 **pipeline pilot**，而不是论文结果。

## 5. 必须优先修复的硬问题

### P0：统一训练与模型接口

当前维护代码存在明显接口风险：

- `build_feature_extractor` 的参数是 `input_shape`，但 `cdan_model.py` 以 `input_dim` 位置参数调用；
- 特征提取器输出 `4096` 维，而 `ConditionalFeature` 默认 `feature_dim=1024`；
- `ConditionalFeature` 和 `domain_discriminator` 的输入维度通过默认值耦合，没有从 backbone 自动推导；
- `cdan_model.py` 只构建单输入 Keras Model，没有实现源域有标签、目标域无标签的完整训练步；
- 当前 `trainer_cadn_supervised.py` 是源域监督训练器，不是 UDA trainer。

投稿前必须先完成一个统一接口，至少支持：

```text
Source Only -> DANN -> CDAN -> proposed variant
```

所有方法必须使用相同的数据迭代器、主干、分类器、优化器策略、评估代码和结果 schema。

### P0：恢复真实数据与数据契约

当前 `data` 是指向 `/mnt/d/SCA_UDA/data` 的符号链接，但该路径在当前环境不存在。仓库中虽然有 CDPA 的压缩数据包，但不能把“文件存在”直接当作“已完成可运行数据集”。

必须记录：

- 数据集、版本和下载来源；
- 源域/目标域划分；
- profiling/attack group；
- 轨迹长度、裁剪窗口和归一化方式；
- 标签定义和目标字节；
- 明文、密钥和 metadata 的读取方式；
- 数据包解压后的文件哈希或版本标识。

### P0：统一正式 SCA 指标

主结果不能只使用分类准确率。至少需要：

- key-byte guessing entropy curve；
- 固定轨迹数的 success rate；
- traces to disclosure 或 NTGE；
- 至少 3 个随机种子的均值和离散程度；
- 每个源-目标配对的结果，而不是只报告平均值。

目标标签可以用于离线评估，但不能参与无监督适应、模型选择或 checkpoint 选择。若使用目标域 GE 选择 checkpoint，必须单独标注为 oracle 分析。

### P1：补齐公平基线

最低基线集合：

1. Official ASCAD CNN / Source Only；
2. DANN；
3. CDAN without entropy；
4. CDPA/MMD-style SCA baseline；
5. AdaBN 或简单统计对齐；
6. proposed method。

其中 CDAN 的去熵、去条件模块、投影维度变化属于 proposed method 的消融，不应冒充独立 baseline。

### P1：解决创新性风险

“把 CDAN 用在 SCA 上”本身很容易被审稿人认为是直接迁移已有方法。论文必须至少形成以下三层贡献链：

1. **机制层：** 证明域判别器准确率下降不等于密钥恢复变好，并分析类别模态碰撞；
2. **方法层：** 引入真正面向 SCA 的约束，例如校准感知的目标样本加权、类别原型保持或 GE-aware checkpoint 规则；
3. **验证层：** 在受控偏移和真实多设备数据上都报告 GE/SR/NTGE，并公开失败边界。

## 6. 投稿体量判断

| 维度 | 当前判断（5 分制） | 说明 |
|---|---:|---|
| 问题重要性 | 4 | 跨设备 profiling SCA 具有明确现实动机 |
| 研究问题清晰度 | 4 | 已有 DANN/CDAN/负迁移主线 |
| 当前方法创新性 | 2 | 目前主要是已有 UDA 思路的 SCA 化，需加入 SCA-specific 机制 |
| 工程完成度 | 2 | 模型组件存在，但统一 UDA 训练链路未完成 |
| 实证证据 | 1 | 当前主要是 synthetic smoke 和小规模 pilot |
| 数据覆盖 | 2 | 有公开材料，但当前数据挂载和转换尚未打通 |
| 评价规范性 | 2 | 有 GE 代码雏形，但 pilot 使用 proxy，口径未统一 |
| 可复现性 | 2 | 有 README 和日志，缺环境、配置和完整真实运行记录 |
| 投稿潜力 | 3 | 完成关键实验后可达到专题安全期刊/会议的研究规模 |

### 结论

- **现在投稿：不建议。** 容易被指出没有真实主结果、没有完整基线、方法接口不闭合。
- **按当前主线补齐：可以形成一篇中等规模、较完整的 TCHES/CHES 风格论文。**
- **若目标是更高水平：需要把贡献从“方法移植”提升到“攻击指标驱动的机制诊断与方法设计”，并增加真实多设备或物理采集验证。**

## 7. 推荐论文主张边界

在真实实验完成前，不应使用：

- “证明 CDAN-SCA 在所有跨设备场景下有效”；
- “彻底解决负迁移”；
- “显著优于所有现有方法”；
- “真实设备泛化已经得到验证”。

建议采用以下可证伪主张：

> 在高基数 profiling SCA 中，边缘对抗对齐可能降低域可辨识度，却不一定改善密钥排序；当域偏移保持泄露语义时，加入类别条件结构和可靠性控制有望缓解这一冲突。本文通过统一的 GE/SR/NTGE 协议、受控偏移实验和真实跨设备数据检验该假设，并报告其适用边界。

这一定义允许方法失败时仍然形成有价值的诊断结果，不会把论文押在某一条未经验证的模型路径上。

## 8. 最终判断

项目体量是够的，当前缺的是**证据链闭合**而不是继续堆叠模型模块。下一阶段不应先扩展更多网络，而应按以下顺序推进：

1. 恢复并审计真实数据；
2. 统一 Source Only/DANN/CDAN/CDPA 的训练和 GE 评估；
3. 复现或否定负迁移诊断；
4. 只选择一个 SCA-specific 方法创新深入；
5. 做受控偏移、真实跨设备、消融和失败边界；
6. 最后再决定投稿到 TCHES/CHES 还是更高层级 venue。

