---
title: "DeepSeek-V4 的 Agentic Benchmarks：特点、能力维度与评测选型"
date: "2026-08-11"
summary: "梳理 DeepSeek-V4 使用的 9 项公开 Agent 评测，解释它们分别测量代码修改、终端执行、深度搜索、MCP 工具编排和专业数字工作的哪些能力，并给出面向真实业务的 benchmark 选型建议。"
category: "LLM"
subcategory: "Agent Evaluation"
tags: ["DeepSeek-V4", "Agent", "Benchmark", "Coding Agent", "Tool Use", "Deep Research"]
type: "article"
---

DeepSeek-V4 的 Agent 评测并不是一组同质化的“智能体考试”。它覆盖了四类明显不同的工作：在真实代码仓库里修改代码、在终端环境中连续执行操作、借助搜索和 Python 完成研究，以及在大量工具或办公应用之间编排工作流。

这篇文章不重复解读模型架构，而是回答一个更实用的问题：**这些 benchmark 各自在测什么，适合评估什么场景，又有哪些数字之外的限制？**

## TL;DR

DeepSeek-V4 报告了 9 项公开 Agent 指标，可以按能力分为四组：

1. **代码仓库 Agent**：SWE-bench Verified、SWE-bench Pro、SWE-bench Multilingual；
2. **终端执行 Agent**：Terminal-Bench 2.0；
3. **搜索研究 Agent**：BrowseComp、Humanity's Last Exam with tools；
4. **工具与专业工作 Agent**：MCP-Atlas、Toolathlon、GDPval-AA。

没有一个单项分数能代表完整的 Agent 能力。选型时最重要的不是“哪个榜最难”，而是它的**环境、动作空间、成功判据和失败模式**是否接近你的产品。

## DeepSeek-V4 的统一评测配置

根据 [DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348)，代码 Agent 任务使用内部开发的评测框架，只提供最小化的 Bash 和文件编辑工具；单任务最多 500 个交互步骤，最大上下文为 512K。搜索 Agent 任务则提供 Web Search 和 Python，同样限制为最多 500 步、512K 上下文。

这带来三个解读前提：

- 表中的结果是**模型、提示词、工具接口、上下文管理和运行环境的联合结果**，不是脱离 harness 的裸模型能力；
- 虽然 DeepSeek-V4 的模型主题是百万 token 上下文，但这些 Agent 评测最多使用 512K，因此不能把结果直接理解为“百万上下文 Agent 已得到验证”；
- Terminal-Bench 2.0 的原始数据存在部分环境问题。报告为了与其他模型保持可比性仍列出原始集成绩，同时提到在 Verified 子集上 DeepSeek-V4-Pro 约为 72.0。

## 九项公开评测的能力地图

下表中的成绩来自 DeepSeek-V4 报告的 V4-Pro-Max 配置。不同 benchmark 的指标、评测器和任务分布并不相同，**分数不能横向比较**。

| Benchmark | 核心特点 | 主要评测能力 | 适用场景 | V4-Pro-Max |
| --- | --- | --- | --- | ---: |
| Terminal-Bench 2.0 | 隔离终端中的真实、困难任务 | 命令行探索、长链执行、环境恢复 | DevOps、数据处理、系统维护 | 67.9 Acc |
| SWE-bench Verified | 500 个经人工验证的 Python 仓库问题 | Issue 定位、补丁生成、回归验证 | Python 缺陷修复 Agent | 80.6 Resolved |
| SWE-bench Pro | 1,865 个更长、更复杂的专业软件任务 | 多文件修改、功能开发、长程代码理解 | 企业级 Coding Agent | 55.4 Resolved |
| SWE-bench Multilingual | 9 种编程语言、42 个仓库 | 跨语言和构建生态泛化 | 多语言研发平台 | 76.2 Resolved |
| BrowseComp | 1,266 个难检索、多跳事实问题 | 搜索规划、实体消歧、证据聚合 | Deep Research、情报检索 | 83.4 Acc |
| HLE with tools | 约 2,500 个专家级跨学科问题 | 搜索/Python 辅助的高难推理 | 专业问答、研究助理 | 48.2 Acc |
| GDPval-AA | 真实职业任务和可交付物，由盲评产生 Elo | 文档、表格、演示等专业数字工作 | 通用办公 Agent、数字员工 | 1554 Elo |
| MCP-Atlas Public | 36 个真实 MCP Server、220 个工具 | 工具发现、参数选择、跨 Server 编排 | MCP/API Agent | 73.6 Pass@1 |
| Toolathlon | 32 个应用、604 个工具的跨应用任务 | 状态跟踪、长链调用、错误恢复 | SaaS 自动化、企业工作流 | 51.8 Pass@1 |

## 1. 代码仓库 Agent：从修 Bug 到专业研发

### SWE-bench Verified：标准化的 Python Issue 修复

[SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) 从公开 Python 仓库中选取 500 个经过人工核验的问题。Agent 获得 Issue 描述和修复前的仓库，需要生成补丁；隐藏测试既检查原问题是否解决，也检查原有行为是否被破坏。

它适合测量：

- 从自然语言 Issue 定位代码区域；
- 阅读中等规模 Python 仓库；
- 修改实现并利用测试迭代；
- 在修复与回归风险之间取得平衡。

它尤其适合 Python Bug 修复 Agent 的快速对比，但不应承担“通用软件工程能力”的全部结论。公开题目长期暴露，容易受到训练数据污染；同时，OpenAI 在 2026 年的复核中指出，前沿模型可能复现已知补丁或利用题目细节，因此[不再把它视为足以区分前沿 Coding Agent 的指标](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)。

### SWE-bench Pro：更接近企业研发的长程任务

[SWE-bench Pro](https://arxiv.org/abs/2509.16941) 扩展到 1,865 个任务和 41 个活跃仓库，包含公开、隐藏和商业代码分区。相比 Verified 的典型缺陷修复，它更强调多文件修改、复杂功能和小时到天级的专业研发工作。

它更适合评估：

- 大仓库中的跨模块理解；
- 需求实现、重构与复杂缺陷修复；
- 长链规划和多轮测试；
- 企业私有代码环境中的迁移能力。

但“更难”不自动等于“更可靠”。OpenAI 对 SWE-bench Pro 的[任务质量审计](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)发现相当比例的问题可能存在测试或规格缺陷。因此使用时应同时报告有效任务子集、失败归因以及对评测器的人工抽检，而不是只保留一个 resolved rate。

### SWE-bench Multilingual：跨语言、跨工具链泛化

[SWE-bench Multilingual](https://www.swebench.com/multilingual.html) 包含 300 个精选任务，覆盖 C、C++、Go、Java、JavaScript、TypeScript、PHP、Ruby 和 Rust，共涉及 42 个仓库，仍沿用 Issue、仓库快照和隐藏测试的基本范式。

它测试的不只是“会不会写另一种语法”，还包括能否正确使用各语言的构建系统、依赖管理、测试框架和工程习惯。它适合多语言 IDE Agent、代码托管平台和企业研发助手；如果业务大量依赖 CUDA、移动端或内部构建系统，仍需要补充自己的仓库任务。

## 2. Terminal-Bench 2.0：在环境中把事情做完

[Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) 包含 89 个困难、真实的终端任务。每题有独立环境、人工编写的参考方案和测试，Agent 必须通过 Shell 探索环境并完成目标。

与代码补丁 benchmark 相比，它更关注“执行过程”：

- 能否根据反馈持续修正命令；
- 能否管理文件、进程、依赖和系统状态；
- 能否在不熟悉的环境中定位故障；
- 能否从局部失败中恢复，而不是一次生成最终答案。

因此它很适合 DevOps、数据工程、系统配置和命令行 Copilot。它的限制是题量较小，并且容器镜像、网络和依赖状态会影响结果；部署前应把稳定环境和重试策略纳入评测协议。

## 3. 搜索研究 Agent：找得到与答得对

### BrowseComp：难检索信息的持续搜索

[BrowseComp](https://openai.com/index/browsecomp/) 有 1,266 个问题，答案通常不是搜索一次即可得到，而要经过多轮查询改写、网页导航、多跳关联和实体消歧。它能有效区分普通搜索增强问答与真正持续探索的 Research Agent。

适合的能力包括：

- 把模糊问题拆成可检索的子问题；
- 在候选实体之间消歧；
- 从多个网页拼合证据；
- 判断当前证据不足并继续搜索。

不过它最终主要按短答案判分，不能充分评估引用是否完整、报告是否可读、来源是否权威，也不覆盖网页中的写操作。因此评测 Deep Research 产品时，最好再加入“带引用的长报告”和来源质量审查。

### Humanity's Last Exam with tools：工具辅助的专家知识推理

[Humanity's Last Exam](https://labs.scale.com/leaderboard/humanitys_last_exam) 汇集约 2,500 个来自 100 多个学科的专家级问题。加入搜索和 Python 后，它关注模型能否为高难问题找到外部知识、进行计算并给出精确答案。

它适合测专业研究助理、科学问答和复杂分析中的“检索 + 推理”上限，但多数任务仍是封闭式精确答案。它不能替代对长期项目管理、写作质量、用户沟通或真实系统操作的评测。

## 4. 工具与专业工作 Agent：会调用工具，也要完成工作

### MCP-Atlas：从大量工具中发现正确路径

[MCP-Atlas](https://arxiv.org/abs/2602.00933) 包含 1,000 个由专家创建的任务、36 个真实 MCP Server 和 220 个工具，其中公开集与私有集各 500 题。题目不会直接告诉 Agent 应该使用哪个 Server、工具或参数，并混入大量干扰工具。

它重点测量：

- 从工具描述中发现候选能力；
- 选择正确 Server、函数和参数；
- 在多个 Server 之间组合工作流；
- 根据中间返回值继续调用。

它非常适合 MCP 客户端、企业 API 助手和工具路由器。需要注意的是，其基于 claim 的 rubric 更偏最终答案是否包含目标事实，不一定逐项验证外部世界状态。因此对“创建日程、修改订单、发消息”一类有副作用的产品，应额外检查实际状态变化、权限和幂等性。

### Toolathlon：跨应用的长链事务执行

[Toolathlon（Tool Decathlon）](https://toolathlon.xyz/docs/dataset) 包含 108 个任务、32 个应用和 604 个工具，平均每题约需 20 次调用。场景覆盖 Gmail、Calendar、Notion、Canvas、WooCommerce、Kubernetes 和 BigQuery 等系统，并在预置的真实状态上执行跨应用工作流。

它很接近“企业自动化 Agent”的核心困难：从多个数据源读取状态，完成有依赖顺序的操作，并在每次调用后维护正确的世界模型。确定性评测脚本也让“真的改对了系统状态”比只检查语言答案更可信。

它适合 SaaS 自动化、个人助理和业务流程 Agent，但任务只有 108 个，运行成本较高，应用组合也不可能覆盖每家企业的权限模型和内部系统。生产评测仍应复刻自己的应用栈和高风险动作。

### GDPval-AA：能否产出专业人士认可的交付物

[GDPval](https://openai.com/index/gdpval/) 面向金融、医疗、法律等职业中的真实知识工作。GDPval-AA 在 Shell 与 Web Agent 环境里生成文档、幻灯片、图表和电子表格等交付物，再通过盲法两两比较形成 Elo 分数。

它与前面几项最大的区别是：评测目标不只是工具调用正确，而是**最终产物是否像一份可用的专业工作成果**。因此适合通用办公 Agent、数字员工和 artifact 生成系统。

Elo 也带来解释上的限制：分数依赖参评模型池、裁判偏好和 harness，跨时间或跨榜单不能简单对齐。上线前仍需要用目标用户、公司模板和真实验收标准做补充评审。

## 报告中的两项补充 Agent 评测

除了九项公开指标，DeepSeek-V4 还报告了两类更专门的测试：

- **Lean 形式化数学 Agent**：使用 Lean 编译器和语义 tactic 搜索，最多 500 次工具调用，由严格验证器判断证明是否成立。它能测量“生成—编译反馈—修正”的形式化证明闭环，但不代表通用工具 Agent；
- **内部研发 Coding Agent**：从 50 多名内部工程师提供的约 200 个 PyTorch、CUDA、Rust、C++ 任务中筛选 30 个，覆盖功能开发、修 Bug、重构和诊断，并在原始仓库和环境中用人工 rubric 验收。它更接近真实研发，却因规模小、数据私有而难以复现。

后者提醒我们：公开榜单适合建立共同坐标，但最能预测生产效果的，通常还是来自自身工作流的私有任务。

## 如何为自己的 Agent 选择评测组合

| 产品场景 | 建议主 benchmark | 必要补充 |
| --- | --- | --- |
| Python Bug 修复 | SWE-bench Verified | 新鲜私有仓库 Issue、人工代码审查 |
| 专业软件工程 | SWE-bench Pro + Terminal-Bench | 内部长程需求、成本和回归率 |
| 多语言研发平台 | SWE-bench Multilingual + Terminal-Bench | 目标语言的真实构建系统与私有代码 |
| Deep Research | BrowseComp + HLE with tools | 带引用长报告、来源质量和时效性 |
| MCP/API 助手 | MCP-Atlas + Toolathlon | 权限、状态变更、幂等性和回滚 |
| 企业办公 Agent | GDPval-AA + Toolathlon | 公司模板、用户盲评、合规检查 |
| DevOps / 数据 Agent | Terminal-Bench + Toolathlon 中的 K8s/BigQuery | 线上沙箱、故障恢复和安全边界 |
| 形式化数学 Agent | Lean/PutnamBench 类验证任务 + HLE | 证明可读性、搜索成本和超时率 |

一个实用原则是同时保留三层评测：

1. **公共 benchmark**：用于和外部模型建立共同坐标；
2. **领域 benchmark**：复现目标技术栈、工具集合和任务长度；
3. **生产回放**：来自真实用户请求，覆盖权限、异常、歧义和环境变化。

## 看分数时还应追问什么

Agent 的 Pass@1 或 resolved rate 只是终点指标。为了判断系统是否真的可用，还应同步记录：

- 任务成功率的多次运行方差，而不是只跑一次；
- token、搜索、工具调用和沙箱的总成本；
- P50/P95 完成时间与最大步骤数；
- 失败发生在规划、工具选择、参数、执行、验证还是恢复；
- 对提示注入、越权操作和不可逆副作用的抵抗能力；
- 环境发生并发变化后能否重新读取状态并恢复；
- 长任务中断后能否保存进度、继续执行和接受用户纠偏。

DeepSeek-V4 的这组 benchmark 对代码、搜索、终端和工具编排覆盖较广，但仍不足以回答真实产品里的用户交互、多方权限、动态环境、多日持久任务和安全副作用。若目标是上线 Agent，这些维度不应留到 benchmark 之后再看，而应直接进入验收标准。

## 结论

DeepSeek-V4 的 Agentic 评测组合价值在于覆盖面，而不是某一个特别高的数字：SWE 系列观察仓库级研发，Terminal-Bench 观察环境中的连续执行，BrowseComp 与 HLE 观察工具辅助研究，MCP-Atlas 与 Toolathlon 观察跨工具编排，GDPval-AA 则把终点推进到专业交付物质量。

真正有效的评测策略，是先明确产品要完成的工作，再选择最接近其环境、动作和成功判据的公开 benchmark，最后用新鲜的私有任务补齐权限、安全、成本、时延和可恢复性。**Benchmark 是能力坐标系，不是生产验收本身。**
