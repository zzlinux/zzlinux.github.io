---
title: "通俗易懂的 PPO 算法理解"
date: "2026-06-03"
tags: ["强化学习", "PPO", "RLHF"]
summary: "沿着一条自然的逻辑链，从最原始的强化学习目标一步步推到 PPO，让每个公式都不是硬背出来的，而是“为什么必须这么做”的结果。"
category: "强化学习"
subcategory: "PPO"
type: "article"
pinned: true
---
# 通俗易懂的 PPO 算法理解：从策略梯度一步步推到 PPO-Clip

PPO，全称 **Proximal Policy Optimization**，中文通常叫 **近端策略优化**。

它是强化学习里非常经典、非常常用的算法，也是 LLM 后训练中 RLHF 的核心算法之一。很多人第一次学习 PPO 时会觉得公式很多：策略梯度、Advantage、TD Error、GAE、Importance Sampling、Ratio、Clip、KL、Value Loss……每个概念单独看都能懂，但串起来就容易断。

这篇文章的目标是：**沿着一条自然的逻辑链，从最原始的强化学习目标一步步推到 PPO，让每个公式都不是硬背出来的，而是“为什么必须这么做”的结果。**

---

## 1. PPO 到底想解决什么问题？

强化学习的目标是学习一个策略：

$$
\pi_\theta(a|s)
$$

它表示：在状态 $s$ 下，选择动作 $a$ 的概率。

我们希望这个策略在环境中拿到尽可能高的累计奖励：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

其中一条轨迹是：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots)
$$

整条轨迹的回报是：

$$
R(\tau)=\sum_{t=0}^{T}\gamma^t r_t
$$

所以强化学习最核心的问题就是：

> 怎么调整策略参数 $\theta$，让期望回报 $J(\theta)$ 变大？

PPO 的核心思想可以先用一句话概括：

> 如果一个动作比平均好，就提高它的概率；如果它比平均差，就降低它的概率；但是每次更新不能太猛。

这里的“不能太猛”，就是 PPO 名字里 **Proximal，近端** 的含义。

---

## 2. 从策略梯度开始

我们想最大化：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

写成积分形式：

$$
J(\theta)=\int p_\theta(\tau)R(\tau)d\tau
$$

对参数求梯度：

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

这里不好直接处理 $\nabla_\theta p_\theta(\tau)$，于是使用 **log-derivative trick**：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
$$

代入得到：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
[
\nabla_\theta\log p_\theta(\tau)R(\tau)
]
$$

轨迹概率可以分解为：

$$
p_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t|s_t)
p(s_{t+1}|s_t,a_t)
$$

环境转移概率 $p(s_{t+1}|s_t,a_t)$ 不依赖策略参数 $\theta$，所以：

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

于是得到最基础的策略梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
R(\tau)
\right]
$$

这句话的直觉是：

> 如果一条轨迹回报高，就增加这条轨迹中动作的概率；如果回报低，就降低这些动作的概率。

但这很快会遇到一个问题：**整条轨迹回报 $R(\tau)$ 的方差太大。**

---

## 3. 为什么不能直接用整条轨迹回报？

假设一条轨迹中：

```text
第 3 步做了一个好动作；
第 20 步因为环境随机性失败；
最终整条轨迹回报很低。
```

如果直接用整条轨迹的回报 $R(\tau)$ 去更新所有动作，那么第 3 步那个好动作也会被惩罚。

这就是强化学习里的 **credit assignment，贡献归因** 问题：

> 最终奖励或惩罚到底应该算到哪些动作头上？每个动作应该承担多少贡献？

直接用整条轨迹回报有几个问题：

1. 它包含整条轨迹的随机性，不只反映当前动作好坏；
2. 一条轨迹里的所有动作共享同一个总回报，无法精确判断每一步的贡献；
3. 轨迹越长，累积的随机项越多，回报方差越大；
4. 同一个动作可能因为后续不同随机事件，得到完全不同的最终回报。

所以我们不想用 $R(\tau)$ 评价每一个动作。我们更想知道：

> 当前状态下，这个动作比平均动作好多少？

这就引出了 $Q$、$V$、$A$。

---

## 4. 从 Return 到 $Q$、$V$、Advantage

定义动作价值函数：

$$
Q^\pi(s_t,a_t)
=
\mathbb{E}_\pi[R_t\mid s_t,a_t]
$$

它表示：

> 在状态 $s_t$ 下采取动作 $a_t$，之后继续按策略 $\pi$ 行动，预期能拿到多少回报。

定义状态价值函数：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi[R_t\mid s_t]
$$

它表示：

> 到了状态 $s_t$，如果继续按当前策略行动，平均能拿多少回报。

于是定义优势函数：

$$
A^\pi(s_t,a_t)
=
Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

它表示：

> 动作 $a_t$ 比当前状态下的平均动作好多少。

如果 $A_t>0$，说明这个动作比平均好，应该提高它的概率；如果 $A_t<0$，说明这个动作比平均差，应该降低它的概率。

于是策略梯度可以写成：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
[
\nabla_\theta\log\pi_\theta(a_t|s_t)A_t
]
$$

这一步非常关键。它把“整条轨迹好不好”变成了：

> 当前动作相对于当前状态平均水平好不好。

这显著降低了梯度估计的方差。

---

## 5. Value Function 从哪里来？Bellman 方程

状态价值函数定义为：

$$
V^\pi(s_t)=\mathbb{E}_\pi[R_t\mid s_t]
$$

而 return 可以递推：

$$
R_t=r_t+\gamma R_{t+1}
$$

所以：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi[
r_t+\gamma R_{t+1}
\mid s_t
]
$$

关键问题是：为什么可以把 $R_{t+1}$ 和 $V(s_{t+1})$ 联系起来？

因为：

$$
V^\pi(s_{t+1})
=
\mathbb{E}_\pi[R_{t+1}\mid s_{t+1}]
$$

也就是说：

> 如果已经知道自己到了 $s_{t+1}$，那么从这个状态开始的未来平均回报就是 $V^\pi(s_{t+1})$。

严格来说，这里用了全期望公式：

$$
\mathbb{E}[R_{t+1}\mid s_t]
=
\mathbb{E}[
\mathbb{E}[R_{t+1}\mid s_{t+1}]
\mid s_t
]
$$

而：

$$
\mathbb{E}[R_{t+1}\mid s_{t+1}]
=
V^\pi(s_{t+1})
$$

所以：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t
]
$$

这就是 Bellman 方程。

用一句话理解：

> 当前状态价值 = 当前一步奖励 + 折扣后的下一状态价值，然后对所有可能性取平均。

---

## 6. 从 Bellman 方程到 TD Error

Bellman 方程是期望形式：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t
]
$$

但实际训练中，我们只有采样到的一条 transition：

$$
(s_t,a_t,r_t,s_{t+1})
$$

于是构造单步 TD target：

$$
y_t^{TD}=r_t+\gamma V(s_{t+1})
$$

当前 value 预测是 $V(s_t)$，两者之差就是 TD error：

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

它表示：

> 走了一步之后，发现“真实一步奖励 + 下一状态价值”与原来对当前状态的预测差了多少。

如果 $\delta_t>0$，说明原来低估了这个状态；如果 $\delta_t<0$，说明原来高估了这个状态。

同时，TD error 也可以看成单步 advantage 的估计：

$$
A_t\approx \delta_t
$$

因为：

$$
A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
$$

而单步近似下：

$$
Q(s_t,a_t)\approx r_t+\gamma V(s_{t+1})
$$

所以：

$$
A_t
\approx
r_t+\gamma V(s_{t+1})-V(s_t)
=
\delta_t
$$

---

## 7. GAE：在 TD 和 Monte Carlo 之间折中

单步 TD error $\delta_t$ 方差低，但偏差可能较高，因为它依赖 value function 的准确性。

Monte Carlo advantage：

$$
R_t-V(s_t)
$$

偏差低，但方差高，因为它依赖完整未来回报。

GAE，也就是 **Generalized Advantage Estimation**，做的是折中：

$$
A_t^{GAE}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$

展开：

$$
A_t^{GAE}
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
$$

这里 $\lambda$ 控制看多远。

当 $\lambda=0$，后面的项全部消失：

$$
A_t^{GAE}=\delta_t
$$

就是单步 TD。

当 $\lambda=1$，GAE 会接近：

$$
R_t-V(s_t)
$$

也就是 Monte Carlo advantage。

所以 $\lambda$ 控制的是：

$$
\text{低方差高偏差的 TD}
\leftrightarrow
\text{低偏差高方差的 Monte Carlo}
$$

之间的折中。

经典 PPO 常用：

$$
\gamma=0.99,\quad \lambda=0.95
$$

在 LLM PPO 里，经常使用：

$$
\gamma=1.0,\quad \lambda=0.95
$$

---

## 8. PPO 为什么需要旧策略和 importance ratio？

理论上的策略梯度希望数据来自当前策略：

$$
a_t\sim \pi_\theta(a_t|s_t)
$$

但 PPO 为了提高样本效率，不是采样一步更新一步，而是：

```text
用当前策略采样一批数据；
固定这批数据；
用这批数据更新多轮。
```

采样这批数据时的策略记为：

$$
\pi_{\theta_{\text{old}}}
$$

在采样的一瞬间，它就是当前策略，所以 PPO 仍然是 on-policy。

但在同一批数据上更新几步后，当前策略已经变成 $\pi_\theta$，而数据还是来自 $\pi_{\theta_{\text{old}}}$，因此需要重要性采样修正。

定义新旧策略概率比：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

于是构造 surrogate objective：

$$
L^{PG}(\theta)
=
\mathbb{E}_t[
r_t(\theta)A_t
]
$$

这里 $r_t>1$ 表示新策略比旧策略更喜欢这个动作；$r_t<1$ 表示新策略比旧策略更不喜欢这个动作。

这一步很重要，因为它把目标变成了：

> 用旧策略采样的数据，估计新策略该如何调整动作概率。

---

## 9. 为什么不能直接最大化 $r_tA_t$？

如果直接最大化：

$$
r_t(\theta)A_t
$$

会出现策略更新过猛的问题。

如果 $A_t>0$，说明动作好，优化器会不断增大 $r_t$，也就是大幅提高这个动作概率。

如果 $A_t<0$，说明动作差，优化器会不断减小 $r_t$，也就是大幅降低这个动作概率。

这方向没错，但幅度可能太大。一旦新策略离旧策略太远，这批旧数据就不再可靠，训练也容易崩。

所以我们希望：

$$
r_t(\theta)\approx 1
$$

也就是新旧策略不要差太多。

这就是 PPO 名字里 **Proximal，近端** 的体现。

PPO 的近端不是指参数距离：

$$
\|\theta-\theta_{\text{old}}\|
$$

很近，而是指策略分布接近：

$$
\pi_\theta(\cdot|s)
\approx
\pi_{\theta_{\text{old}}}(\cdot|s)
$$

用 ratio 表示就是：

$$
r_t(\theta)
\in
[1-\epsilon,1+\epsilon]
$$

---

## 10. 为什么 PPO-Clip 不是简单 clip ratio？

一个自然想法是：

$$
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
$$

但这个形式有一个问题：

> 它会无差别截断所有越界情况，包括那些本来应该被惩罚的坏更新。

举两个例子。

如果：

$$
A_t>0,\quad r_t<1-\epsilon
$$

说明动作本来是好动作，但新策略把它概率降低太多。这是坏更新，应该保留惩罚信号，让优化器把它纠正回来。

如果直接 clip，就会把 $r_t$ 截到 $1-\epsilon$，反而掩盖了错误。

再比如：

$$
A_t<0,\quad r_t>1+\epsilon
$$

说明动作本来是坏动作，但新策略把它概率提高太多。这也是坏更新，也应该继续惩罚。

如果直接 clip，也会掩盖这个问题。

所以 PPO 不只是想限制 ratio 本身，而是想限制：

> 有利方向上的过度收益。

这就是为什么 PPO 使用 $\min$ 构造保守目标。

---

## 11. PPO-Clip 目标函数

PPO 的核心目标是：

$$
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}
(
r_t(\theta),
1-\epsilon,
1+\epsilon
)
A_t
\right)
\right]
$$

因为 PPO 是最大化这个目标，取 $\min$ 意味着：

> 在未裁剪目标和裁剪目标之间，永远选择更保守、更小的收益估计。

它实现了两件事：

1. 限制过度有利的更新；
2. 保留错误方向的惩罚信号。

分情况看：

当 $A_t>0$，说明动作好，应该提高概率，但当 $r_t>1+\epsilon$ 之后不再继续奖励。

当 $A_t<0$，说明动作差，应该降低概率，但当 $r_t<1-\epsilon$ 之后不再继续奖励。

同时，如果 $A_t>0, r_t<1-\epsilon$，或者 $A_t<0, r_t>1+\epsilon$，这种错误方向的更新，PPO 会保留惩罚。

这就是 PPO-Clip 稳定性的关键。

---

## 12. Value Function Loss 是怎么来的？

PPO 不只有 actor，还通常有 critic。

Actor 是策略：

$$
\pi_\theta(a|s)
$$

Critic 是价值函数：

$$
V_\phi(s)
$$

Critic 的定义是：

$$
V^\pi(s_t)=\mathbb{E}_\pi[R_t\mid s_t]
$$

所以训练 critic 本质上是监督学习：

> 输入状态 $s_t$，预测未来回报。

因此自然使用平方误差：

$$
L^{VF}(\phi)
=
\mathbb{E}_t[
(V_\phi(s_t)-R_t^{target})^2
]
$$

那么 $R_t^{target}$ 从哪里来？

GAE 算出来的是 advantage $A_t$，而 advantage 的定义是：

$$
A_t
=
Q(s_t,a_t)-V(s_t)
$$

所以：

$$
Q(s_t,a_t)
=
A_t+V(s_t)
$$

在采样轨迹中，$Q(s_t,a_t)$ 可以看作这个状态-动作对应的目标回报，因此：

$$
R_t^{target}
=
A_t+V_{\phi_{\text{old}}}(s_t)
$$

注意：$A_t$ 不是采取动作后的总收益，它是相对优势。真正的动作后预期收益是 $Q(s_t,a_t)$，而 $A_t+V(s_t)$ 是在把“相对优势”还原成“绝对回报目标”。

所以 value loss 就是：

$$
L^{VF}(\phi)
=
\mathbb{E}_t[
(V_\phi(s_t)-R_t^{target})^2
]
$$

---

## 13. PPO 总损失

实际训练时，PPO 的总 loss 通常是：

$$
\mathcal{L}_{PPO}
=
-
L^{CLIP}
+
c_1L^{VF}
-
c_2H
$$

其中：

- $-L^{CLIP}$ 是 actor loss，因为代码里通常做最小化；
- $L^{VF}$ 是 critic loss；
- $H$ 是 entropy，用来鼓励探索，防止策略太早变得过于确定。

所以 PPO 同时训练：

1. actor：让好动作概率变大，坏动作概率变小；
2. critic：预测每个状态的未来回报；
3. entropy：保留一定随机性。

---

## 14. PPO 的完整训练流程

一轮 PPO 可以概括成：

```text
1. 用当前策略 π_old 采样一批轨迹或 response
2. 记录 old_log_probs 和 old_values
3. 计算 reward
4. 用 TD error / GAE 计算 advantages
5. 构造 returns = advantages + old_values
6. 对这批数据做多轮 mini-batch 更新
7. 每次更新重新计算 new_log_probs 和 new_values
8. 计算 ratio = exp(new_log_probs - old_log_probs)
9. 用 PPO-Clip loss 更新 actor
10. 用 value loss 更新 critic
11. 丢掉这批数据，进入下一轮重新采样
```

其中外层循环负责 on-policy 采样，内层循环负责对同一批数据做多轮更新。

这也是为什么 PPO 里经常有：

- rollout batch size；
- mini-batch size；
- PPO epochs。

rollout batch size 是一次采样多少数据；mini-batch size 是每次梯度更新用多少数据；PPO epochs 是同一批数据重复更新几轮。

---

## 15. LLM PPO 中的对应关系

在大语言模型中，PPO 的概念对应如下：

| 强化学习概念 | LLM 中的对应 |
|---|---|
| state $s_t$ | prompt + 已生成 token |
| action $a_t$ | 下一个 token |
| policy $\pi_\theta$ | 当前语言模型 |
| old policy | rollout 时的语言模型 |
| reference policy | 冻结的 SFT 模型 |
| reward | reward model / verifier 分数 |
| value $V(s_t)$ | value head 输出 |
| trajectory | 一条完整 response |

LLM PPO 中，token-level ratio 是：

$$
r_t(\theta)
=
\frac{
\pi_\theta(y_t|x,y_{<t})
}{
\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})
}
$$

代码里通常写成：

```python
ratio = torch.exp(new_log_probs - old_log_probs)
```

reward 通常包括两部分：

$$
r_t
=
-
\beta[
\log\pi_{\text{old}}(y_t|s_t)
-
\log\pi_{\text{ref}}(y_t|s_t)
]
+
\mathbf{1}_{t=T}R_{RM}(x,y)
$$

也就是说：

- 每个 token 都有 KL penalty；
- reward model 对完整回答的分数通常放在最后一个 token 上；
- 前面的 token 会通过 return / GAE 接收到最终 reward 的影响。

---

## 16. 最终用一句话理解 PPO

PPO 的推导可以压缩成这一条链：

$$
\text{最大化期望回报}
\Rightarrow
\text{策略梯度}
\Rightarrow
\text{用 Advantage 降低方差}
\Rightarrow
\text{用 GAE 估计 Advantage}
\Rightarrow
\text{用 old policy 数据提高样本效率}
\Rightarrow
\text{用 importance ratio 修正分布差异}
\Rightarrow
\text{用 clip + min 限制策略更新幅度}
\Rightarrow
\text{得到 PPO-Clip}
$$

PPO 最核心的三个量是：

### 1. Advantage：告诉你方向

$$
A_t>0
\Rightarrow
提高动作概率
$$

$$
A_t<0
\Rightarrow
降低动作概率
$$

### 2. Ratio：告诉你改了多少

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
$$

### 3. Clip + Min：控制步子别太大

$$
L^{CLIP}
=
\mathbb{E}
[
\min(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
)
]
$$

一句话总结：

> PPO 是一种保守的策略梯度算法：用 advantage 判断动作该增强还是削弱，用 ratio 衡量新旧策略变化，用 clip 和 min 控制策略每次只在旧策略附近安全更新。

---

# 常见 QA：理解 PPO 时最容易卡住的问题

## Q1：为什么直接用整条轨迹回报 $R(\tau)$ 方差很大？

因为整条轨迹回报包含太多随机因素，不能准确说明每一步动作的贡献。

一条轨迹最后得高分，不代表每个动作都好；最后得低分，也不代表每个动作都差。

所以 PPO 使用：

$$
A_t=Q(s_t,a_t)-V(s_t)
$$

衡量当前动作相对于当前状态平均水平的贡献。

---

## Q2：PPO 为什么叫近端策略优化？

因为它限制新策略不要离旧策略太远。

这个“近”不是参数空间近，而是策略分布近：

$$
\pi_\theta(\cdot|s)
\approx
\pi_{\theta_{\text{old}}}(\cdot|s)
$$

用 ratio 表示就是：

$$
r_t(\theta)
\approx 1
$$

---

## Q3：TD Error 是怎么来的？

从 Bellman 方程来：

$$
V(s_t)
=
\mathbb{E}
[
r_t+\gamma V(s_{t+1})
\mid s_t
]
$$

单样本下：

$$
y_t^{TD}=r_t+\gamma V(s_{t+1})
$$

所以：

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

它表示当前 value 预测和一步 Bellman target 的差。

---

## Q4：为什么 $\lambda=0$ 时 GAE 是单步 TD？

因为：

$$
A_t^{GAE}
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
$$

当 $\lambda=0$，后面所有项都为 0，只剩：

$$
A_t^{GAE}=\delta_t
$$

---

## Q5：PPO 是 on-policy，为什么还要 old policy？

PPO 的 old policy 是“采样这一批数据时的当前策略”。

采样时它是当前策略，所以 PPO 是 on-policy。

但采样后 PPO 会用同一批数据更新多轮，此时新策略已经变了，所以要用：

$$
\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}
$$

修正差异。

---

## Q6：为什么 PPO-Clip 不直接 clip $r_t$？

因为直接 clip 会掩盖坏方向的错误更新。

PPO 的目标是：

$$
\min(
r_tA_t,
\operatorname{clip}(r_t)A_t
)
$$

它只限制过度有利的更新，同时保留错误方向的惩罚信号。

---

## Q7：Value Function Loss 是策略梯度推出来的吗？

不是。

Value loss 本质是监督回归：

$$
V_\phi(s_t)
\approx
R_t^{target}
$$

所以自然用：

$$
(V_\phi(s_t)-R_t^{target})^2
$$

它训练的是 critic，而不是 actor。

---

## Q8：为什么 $R_t^{target}=A_t+V(s_t)$？

因为：

$$
A_t=Q(s_t,a_t)-V(s_t)
$$

所以：

$$
Q(s_t,a_t)=A_t+V(s_t)
$$

也就是说，$A_t$ 是相对优势，加回 $V(s_t)$ 才是绝对回报目标。

---

## Q9：策略梯度最开始是 $\nabla\log\pi$，为什么 PPO 里变成了 ratio？

没有消失。

PPO 的 ratio 是：

$$
r_t
=
\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}
=
\exp(\log\pi_\theta-\log\pi_{\theta_{\text{old}}})
$$

并且：

$$
\nabla r_t
=
r_t\nabla\log\pi_\theta
$$

所以 PPO 的梯度里仍然包含：

$$
\nabla\log\pi_\theta
$$

只是通过 ratio 体现出来了。

---

## Q10：LLM PPO 里为什么最终 reward 放在最后一个 token？

因为 reward model 通常评价完整回答：

$$
R_{RM}(x,y)
$$

它是 sequence-level reward，所以自然作为 terminal reward 放在最后。

前面的 token 会通过 return / GAE 接收到这个最终奖励。

---

# 结尾

PPO 看起来公式很多，但它的逻辑其实很统一：

> 我们想让策略变好，所以用策略梯度；  
> 为了减少噪声，用 advantage；  
> 为了估计 advantage，用 value、TD error 和 GAE；  
> 为了提高样本效率，用 old policy 数据和 importance ratio；  
> 为了防止策略更新过猛，用 clip 和 min；  
> 最终得到稳定、保守、工程上好用的 PPO。

这就是 PPO 的核心。
