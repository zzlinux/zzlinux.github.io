---
title: "长 Context、长 Horizon 与 Context Compaction：Agent Benchmark 全景与选型建议"
date: "2026-08-11"
summary: "系统梳理 Agent 长上下文、长程执行与上下文压缩三类能力的区别，比较 LOCA-bench、Context-Bench、Odysseys、DeepWeb-Bench、DeepSWE 等代表性评测，并给出 Context Compaction 的三层实验方案。"
category: "LLM"
subcategory: "Agent Evaluation"
tags: ["Agent", "Benchmark", "Long Context", "Long Horizon", "Context Compaction", "Context Engineering"]
type: "article"
---

随着 Agent 从一次问答走向持续搜索、调用工具、修改环境和交付完整结果，“支持多少 token”已经不足以描述系统能力。一个 Agent 可能拥有 1M Context Window，却在几十步后忘记目标；也可能只使用 32K 工作窗口，通过检索、压缩和外部记忆完成数小时任务。

要正确评测这类系统，必须区分三个相关但不同的维度：**长 Context、长 Horizon 和 Context Compaction**。

## 三个概念的边界

| 维度 | 核心问题 | 常见失败 |
| --- | --- | --- |
| 长 Context | 一次推理能看到并有效利用多少信息？ | 忘记早期信息、漏掉远距离依赖、被噪声干扰 |
| 长 Horizon | 能否连续正确执行很多步、持续很长时间？ | 目标漂移、重复操作、状态混乱、无法恢复 |
| Context Compaction | 历史超过预算时，能否减少 Context 而不破坏后续行为？ | 丢失约束、错误摘要、压缩后重复探索 |

长 Horizon 通常会自然制造长 Context：更多行动产生更多 Observation、工具结果和中间决策，Context 逐渐膨胀并发生 Context Rot，最终需要压缩、检索或外部记忆。但两者并不等价。

- 给模型一次性输入 50 万 token 的代码仓库，只回答一个架构问题，是长 Context、短 Horizon；
- 在多个 SaaS 系统中连续执行 50 次调用，每次输入都很短，是短 Context、长 Horizon；
- 在大型仓库中完成跨模块功能开发，则同时需要长 Context 和长 Horizon。

## DeepSeek-V4 的 Agent 评测覆盖了什么

[DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348)报告了九项公开 Agent 指标。代码和搜索 Agent 任务最多允许 500 个交互步骤，最大上下文为 512K，因此结果衡量的是模型、工具、提示词、Context 管理和执行环境的联合能力，并不能直接验证完整 1M Context Agent。

| Benchmark | 主要能力 | 长 Context 相关性 | 长 Horizon 相关性 |
| --- | --- | ---: | ---: |
| SWE-bench Verified | Python 仓库 Bug 修复 | 中低 | 中等 |
| SWE-bench Pro | 多文件、复杂软件工程 | 中高 | 高 |
| SWE-bench Multilingual | 多语言仓库修改 | 中等 | 中等 |
| Terminal-Bench 2.0 | 终端探索与连续执行 | 中等 | 高 |
| BrowseComp | 多轮搜索与证据发现 | 中等 | 高 |
| HLE with tools | 工具辅助的专家推理 | 中低 | 中低 |
| MCP-Atlas | MCP 工具发现与编排 | 中等 | 中等 |
| Toolathlon | 跨 SaaS 工作流与状态维护 | 中等 | 高 |
| GDPval-AA | 专业数字工作交付物 | 中等 | 中高 |

其中，SWE-bench Pro 最接近“仓库长 Context + 长程执行”，Terminal-Bench 主要测长 Horizon，Toolathlon 主要测跨工具状态保持，BrowseComp 主要测长程搜索。这组指标覆盖面广，但没有一项专门隔离 Context 压缩能力。

## 代码领域：长仓库 Context 与长程研发

### LongCodeBench / LongSWE-Bench

[LongCodeBench](https://arxiv.org/abs/2505.07897)专门面向最高 1M token 的真实代码场景，包含大型仓库问答 LongCodeQA 和真实 Issue 修复 LongSWE-Bench。它适合比较不同 Context 长度、Full-context、RAG 与 Agent 自主探索策略，是评测代码长 Context 的首选。

### ReCUBE

[ReCUBE](https://arxiv.org/abs/2603.25770)要求模型根据仓库其余源码、文档和依赖配置重建被遮蔽的文件，并用跨文件测试验证。它能较好地隔离相关文件发现、跨文件依赖利用、Full-context 与检索式 Context 的差异。

### RepoReasoner

[RepoReasoner](https://arxiv.org/abs/2607.25996)通过跨文件执行结果预测和调用链恢复，检查模型能否在噪声仓库中理解架构依赖。它偏代码推理，不是完整的“搜索—编辑—测试—恢复”闭环。

### DeepSWE、SWE-bench Pro 与 SWE-Marathon

[DeepSWE](https://arxiv.org/abs/2607.07946)、[SWE-bench Pro](https://arxiv.org/abs/2509.16941)和 [SWE-Marathon](https://www.swe-marathon.org/)更偏长 Horizon 软件工程：任务需要多文件修改、长时间探索和反复测试，会产生很长的交互历史，但最终成功率同时受到模型推理、工具、规划和恢复能力影响，不能单独归因为 Context 能力。

## 通用领域：长 Context × 长 Horizon

| Benchmark | 主要场景 | Context 压力 | Horizon 压力 | 适合评测 |
| --- | --- | ---: | ---: | --- |
| LOCA-bench | 可控环境状态增长 | 很高 | 高 | Context Rot、压缩和 Scaffold |
| Letta Context-Bench | 多文件、多跳关系检索 | 高 | 中高 | Agentic Context Engineering |
| Odysseys | 多站点真实 Web 工作流 | 高 | 很高 | 通用浏览器 Agent |
| DeepWeb-Bench | 大规模跨来源研究 | 很高 | 高 | Deep Research、Evidence Memory |
| PlanBench-XL | 1,665 个工具中的规划 | 高 | 很高 | 大工具空间、失败恢复 |
| TheAgentCompany | 模拟公司数字工作 | 中高 | 很高 | 企业数字员工 |
| Gaia2 | 动态、异步环境 | 中高 | 很高 | 状态更新、时效性和协作 |
| Toolathlon | 跨 SaaS 工作流 | 中等 | 高 | 工具编排和状态变更 |

### LOCA-bench：最关键的通用测试床

[LOCA-bench](https://arxiv.org/abs/2602.07962)保持任务语义不变，通过自动控制环境状态让 Context 持续增长，从而观察 Agent 在不同 Context 压力下的性能曲线。它可以比较 Full-history、滑动窗口、工具结果清理、摘要、Skeleton Compression、外部 Archive 与按需恢复等策略，是目前最直接的 Agent Context Compaction benchmark。

### Letta Context-Bench：会不会主动选择 Context

[Letta Context-Bench](https://www.letta.com/blog/context-bench/)把虚构实体和关系分散在文件系统中，要求 Agent 使用搜索和打开文件工具完成多跳查询。它重点测“应该检索什么、加载什么、何时停止”，可控、抗污染，但偏合成和只读检索。

需要注意，Letta Context-Bench 与代码领域的 [ContextBench](https://arxiv.org/abs/2602.05892)不是同一项目；后者使用带人工 Gold Context 的真实 Issue，测量代码 Agent 的 Context Recall、Precision 和 Efficiency。

### Odysseys：非代码领域的 DeepSWE

[Odysseys](https://arxiv.org/abs/2604.24964)包含 200 个真实、多站点、长程 Web 任务，要求 Agent 在持续浏览中保留前面发现的信息并完成跨站比较、旅行规划和研究。它很像“非代码版 DeepSWE”，但真实互联网变化会降低可复现性。

### DeepWeb-Bench：大量证据与长链推导

[DeepWeb-Bench](https://arxiv.org/abs/2605.21482)强调大规模跨来源证据收集、冲突处理、长链推导和结论校准，适合 Deep Research、搜索轨迹压缩和 Evidence Memory。它比短答案型搜索 benchmark 更能暴露“证据找到了但没有完整利用”的问题。

### PlanBench-XL、TheAgentCompany 与 Gaia2

[PlanBench-XL](https://arxiv.org/abs/2606.22388)在 1,665 个工具中评测检索、规划和受阻后的路径恢复；[TheAgentCompany](https://arxiv.org/abs/2412.14161)构造完整的小型公司环境，包含网站、文档、代码和同事通信；[Gaia2](https://arxiv.org/abs/2602.11964)则让环境独立变化，要求 Agent 重新确认状态、处理过期信息和时间约束。三者分别补充大工具空间、企业工作流和动态世界状态。

## Context Compaction 的专项评测

### LOCA-bench：端到端主指标

LOCA-bench 可以在同一任务语义下控制 Context 增长，因此能绘制“Context 长度—成功率—成本”曲线，比较不同压缩策略的有效工作区间。

### TRACE：定位某次压缩是否破坏行为

[TRACE](https://arxiv.org/abs/2608.06503)从同一个环境 checkpoint 构造两条后续轨迹：一条保留原始 Context，另一条使用压缩 Context。通过比较任务结果、重复探索、受阻动作和多次运行稳定性，可以更接近因果地判断某次 Compaction 是否损害了后续执行。

### ACON 的真实任务组合

[ACON](https://arxiv.org/abs/2510.00615)不是独立 benchmark，而是一套同时压缩历史和 Observation 的方法。它使用 AppWorld、OfficeBench 和 Multi-objective QA 检查压缩是否减少峰值 token，同时保持真实工具和办公任务成功率。

### ContextBudget：预算收紧时如何退化

[ContextBudget](https://arxiv.org/abs/2604.01664)把 Context 管理建模为带预算的序列决策，适合比较 16K、32K、64K、128K 等预算下的成功率、压缩次数和信息损失。好的压缩策略应当随预算收紧平滑退化，而不是在某个阈值突然崩溃。

### 单次 Prompt Compression

如果评测对象只是把一个长 Prompt 压缩后交给模型，可以使用 LongBench、ZeroSCROLLS、NaturalQuestions、MuSiQue 或 LongCodeBench。但必须同时测下游任务、输入 Grounding、关键信息保留和压缩率；只测 token reduction 会奖励“删得多但无法使用”的方法。

## Benchmark 选型建议

| 目标 | 推荐主 benchmark | 建议补充 |
| --- | --- | --- |
| 基础模型长 Context | LongCodeBench、LongBench v2 | RepoReasoner、自建证据位置分桶 |
| 通用长 Horizon Agent | Odysseys、TheAgentCompany | PlanBench-XL、Gaia2、Toolathlon |
| Context Engineering | Letta Context-Bench、LOCA-bench | ReCUBE、代码 ContextBench |
| Deep Research | DeepWeb-Bench | Odysseys、BrowseComp-Plus、带引用报告 |
| Context Compaction | LOCA-bench | TRACE、AppWorld、Odysseys |
| 动态状态与恢复 | Gaia2、PlanBench-XL | Toolathlon、企业私有沙箱 |

如果只能为通用 Context Compaction 选一项，优先选择 **LOCA-bench**；如果要建立一套可信评测，则建议使用 **LOCA-bench + TRACE + AppWorld/Odysseys**。

## 统一实验协议

不要只报告“压缩了多少 token”。实验至少需要冻结模型、系统提示词、工具 Schema、任务集、步数与随机预算，并控制：

| 控制变量 | 建议设置 |
| --- | --- |
| Context budget | 16K、32K、64K、128K、256K |
| 最大步骤 | 50、100、200、500 |
| 压缩策略 | 无压缩、滑动窗口、摘要、检索、Archive |
| 重复运行 | 每题至少 3～5 次 |
| 信息规模 | 达到工作 Context budget 的 5～10 倍 |

评测证据分为三层：

1. **L1 端到端结果**：成功率、连续得分、累计输入 token、峰值 Context、工具调用、成本、时延和运行方差；
2. **L2 压缩后轨迹**：从各自第一次压缩边界开始，分析重复探索、状态遗忘、目标漂移、失败恢复和工具错误；
3. **L3 局部 Compaction**：把每份压缩结果与它自己的源证据比较，检查目标、约束、事实、已完成事项、失败原因、未决问题和下一步计划是否保留。

不同策略可能压缩不同的历史范围，因此不应直接比较两份摘要文字，也不能把只发生在“双方都触发压缩”的样本当作无偏总体。最终结论必须由端到端结果、压缩后轨迹和局部信息保留共同支持。

## 结论

长 Context 解决“能看到和利用多少信息”，长 Horizon 解决“能持续正确地做多久”，Context Compaction 则解决“历史超过预算后，怎样保留未来决策真正需要的信息”。

对于通用 Agent，不建议用单个排行榜数字代表三种能力。更可靠的做法是：用长 Context benchmark 检查信息利用，用真实长程任务检查规划和执行，再用 LOCA-bench 与 TRACE 专门检查压缩边界及其后果。**压缩率只是成本指标，压缩后的可继续执行性才是质量指标。**
