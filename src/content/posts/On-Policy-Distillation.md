---
title: "On-Policy Distillation"
date: "2026-06-09"
tags: ["强化学习", "OPD"]
summary: "OPD 的理论解析"
category: "强化学习"
subcategory: "OPD"
type: "article"
pinned: true
---

# On-Policy Distillation（在策略蒸馏）的理论解析
 
> 记号约定：$\mathrm{sg}(\theta)$ 表示停止梯度（stop-gradient）；$s_t=(x,y_{<t})$ 表示当前状态；$a_t=y_t$ 表示当前生成的 Token。

# 一、优化目标

本文介绍 **On-Policy Distillation（OPD，在策略蒸馏）** 的基础理论。我们从 OPD 的优化目标出发。设训练提示词 $x$ 从某个分布 $\mathcal{D}$ 中采样，并且对每个提示词 $x$，我们都能访问与之对应的**教师模型** $\pi_{T(x)}$。OPD 的目标是最小化当前参数化模型 $\pi_\theta$（也称为**学生模型**）与教师模型之间的反向 KL 散度，即：

$$
\begin{align}\mathcal{L} \left(\textcolor{red}{ \theta} \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}(\cdot|x)}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) \right]. \end{align}
$$

将反向 KL 项展开，可以得到 $\mathcal{L}\left(\textcolor{red}{\theta}\right)$ 的以下等价形式。

---

**引理 1（OPD 目标函数的等价形式）.** *$\mathcal{L}\left(\textcolor{red}{\theta}\right)$ 可以等价地改写为：*

$$
\begin{align*}
\mathcal{L} \left( \textcolor{red}{ \theta} \right) &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta} }\left( \cdot |x \right)}\left[ \log \frac{\pi _{\textcolor{red}{ \theta}}\left( y|x \right)}{\pi _{T\left( x \right)}\left( y|x \right)} \right] 
\\
\,\,     &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{ \theta}}\left( y_t|x,y_{<t} \right)}{\pi _{T\left( x \right)}\left( y_t|x,y_{<t} \right)}} \right] 
\\
\,\,     &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\textcolor{red}{ \theta}}\left( \cdot |x,y_{<t} \right) ,\pi _{T\left( x \right)}\left( \cdot|x,y_{<t} \right) \right)} \right] 
\end{align*}
$$

---

为了进一步简化 $\mathcal{L}\left(\textcolor{red}{\theta}\right)$ 的表达，我们引入**状态占用度（state occupancy）**的概念。对于任意策略 $\pi$，定义其状态占用度 $d_\pi$ 为：

$$
\begin{align}d_{\pi}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right].\end{align}
$$

进一步定义 $\mathcal{S}$ 为**状态空间**，即 $(x,y_{<t})$ 所有可能取值构成的集合。下面的引理表明，可以在 $\mathbb{E}_{x\sim\mathcal D}\mathbb{E}_{y\sim\pi(\cdot|x)}\left[\sum_t \cdot\right]$ 与 $\mathbb{E}_{s\sim d_\pi}[\cdot]$ 两种期望写法之间进行转换。

---

**引理 2（状态占用度下的期望转换）.** *对于任意策略 $\pi$ 和实值函数 $f:\mathcal{S}\rightarrow\mathbb{R}$，有：*

$$
\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left( x,y_{<t} \right)} \right] =\mathbb{E} _{s\sim d_{\pi}}\left[ f\left( s \right) \right]
$$

*证明：注意到指示函数满足：*

$$
\sum_{s\in \mathcal{S}}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}}=1
$$

*因此，直接计算可得：*

$$
\begin{align*}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left( x,y_{<t} \right)} \right] &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\underset{=1}{\underbrace{\left( \sum_{s\in \mathcal{S}}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right) }}f\left( x,y_{<t} \right)} \right] \\\,\,                                     &=\sum_{s\in \mathcal{S}}{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\} f\left( x,y_{<t} \right)} \right]}\\\,\,                                     &=\sum_{s\in \mathcal{S}}{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\} f\left( s \right)} \right]}\\\,\,                                     &=\sum_{s\in \mathcal{S}}{\underset{=d_{\pi}\left( s \right)}{\underbrace{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right] }}\cdot f\left( s \right)}\\\,\,                                     &=\mathbb{E} _{s\sim d_{\pi}}\left[ f\left( s \right) \right] \end{align*}
$$

---

结合引理 1 和引理 2，可以直接得到：

$$
\begin{align} 
\mathcal{L} \left( \textcolor{red}{ \theta} \right) =\mathbb{E} _{s\sim d_{\textcolor{red}{ \theta} }}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{ \theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right] =\mathbb{E} _{s\sim d_{\textcolor{red}{ \theta} }}\mathbb{E} _{a\sim \pi _{\textcolor{red}{ \theta} }\left( \cdot |s \right)}\left[ \log \frac{\pi _{\textcolor{red}{ \theta} }\left( a|s \right)}{\pi _{T\left( s \right)}\left( a|s \right)} \right], 
\end{align}
$$

其中，$d_\theta$ 是 $d_{\pi_\theta}$ 的简写，$T(s)$ 表示状态 $s$ 所对应的教师模型。

# 二、OPD 的策略梯度

下面推导 OPD 的策略梯度。

---

**定理 1（OPD 的策略梯度）.** *对于任意参数 $\theta$，OPD 的策略梯度为：*

$$
\begin{align}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta}  \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\hat{A}_{\textcolor{red}{ \theta}}\left( s_t,a_t \right)  \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{ \theta}}\left( a_t|s_t \right)} \right],\end{align}
$$

*其中 $s_t:=(x,y_{<t})$、$a_t:=y_t$，而 $\hat A_\theta(s_t,a_t)$ 可以采用以下任意一种形式：*

*(i) $\sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}=\frac{\log \pi _{\theta}\left( y|x \right)}{\log \pi _{T\left( x \right)}\left( y|x \right)}$ ：整条轨迹的对数概率比*

*(ii) $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}$：从当前 Token $a_t$ 开始的后续子轨迹对数概率比之和*

*（iii） $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} -b(s_{\textcolor{red}{t}})$：形式（ii）加入基线后的版本*

*（iv） $\log \frac{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{T\left( x \right)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}+\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)}$ ：当前 Token 的对数概率比，加上其后子轨迹中累积的反向 KL 散度*

*（v） $\log \frac{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{T\left( x \right)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}+\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)} -b(s_{\textcolor{red}{t}})$：形式（iv）加入基线后的版本* 

*证明：根据引理 1，直接计算可得：*

$$
\small{\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right) &=\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \log \frac{\pi _{\textcolor{red}{\theta}}\left( y|x \right)}{\pi _{T\left( x \right)}\left( y|x \right)} \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \log \frac{\pi _{\textcolor{red}{\theta}}\left( y|x \right)}{\pi _{T\left( x \right)}\left( y|x \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y|x \right) +\nabla _{\theta}\left( \log \frac{\pi _{\textcolor{red}{\theta}}\left( y|x \right)}{\pi _{T\left( x \right)}\left( y|x \right)} \right) \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \log \frac{\pi _{\textcolor{red}{\theta}}\left( y|x \right)}{\pi _{T\left( x \right)}\left( y|x \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y|x \right) \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( y_{t^{\prime}}|x,y_{<t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( y_{t^{\prime}}|x,y_{<t^{\prime}} \right)}} \right) \sum_{t=0}^{\left| y \right|-1}{\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y_t|x,y_{<t} \right)} \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right],\end{align*}}
$$

*这就证明了可以取 $\hat A_\theta(s_t,a_t)=\sum_{t'=0}^{|y|-1}\log\frac{\pi_\theta(a_{t'}|s_{t'})}{\pi_{T(x)}(a_{t'}|s_{t'})}$。下面证明形式（ii）同样成立。考虑两个固定时间索引 $0\le t'<t\le |y|-1$，可以证明：*

$$
\small{\begin{align*}\mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_{t^{\prime}},a_{t^{\prime}} \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_{t^{\prime}},a_{t^{\prime}} \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot 0 \right] \\\,\,                                               &=0,\end{align*}}
$$

*其中期望 $\mathbb E$ 是对 $x\sim\mathcal D$、$y\sim\pi_\theta$ 取的。这个因果性结论意味着：对于每个已访问 Token $a_t$，其 OPD 策略梯度 $\nabla_\theta\log\pi_\theta(a_t|s_t)$ **不需要**考虑它对**此前** Token 的 KL 散度造成的影响，只需要考虑它对**当前及后续**可能访问 Token 的 KL 散度所产生的影响。因此：*

$$
\small{\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right)   &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right]\\\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right].   \end{align*}}
$$

*下面说明形式（iv）为何成立。对于任意两个固定时间索引 $0\le t<t'\le |y|-1$，直接计算可得：*

$$
\small{\begin{align*}
\mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_t,a_t \right) \right] \right] 
\\
\,\,                                               &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}|\left( s_t,a_t \right) \right] \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] 
\\
\,\,                                               &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}|s_{t^{\prime}} \right] \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] 
\\
\,\,                                               &=\mathbb{E} \left[ \mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right],
\end{align*}}
$$

*其中期望 $\mathbb E$ 是对 $x\sim\mathcal D$、$y\sim\pi_\theta$ 取的。结合上式与形式（ii）的证明，可得：* 

$$
\small{\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right)   &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right]\\\ &=
\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \log \frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{T\left( x \right)}\left( a_t|s_t \right)}+\sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)} \right) \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] 
.   \end{align*}}
$$

*对于任意仅依赖当前状态 $s_t$ 的函数 $b(s_t)$，容易证明：*

$$
\begin{align*}\mathbb{E} \left[ b\left( s_t \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ b\left( s_t \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot 0 \right] \\\,\,                               &=0,\end{align*}
$$

*其中期望 $\mathbb E$ 是对 $x\sim\mathcal D$、$y\sim\pi_\theta$ 取的。这意味着，在期望意义下，加上或减去只依赖状态的基线项 $b(s_t)$，不会改变每个状态—动作对 $(s_t,a_t)$ 上的 OPD 策略梯度。因此，可以使用带基线的形式（iii）和（v）作为 $\hat A_\theta(s_t,a_t)$。*

---

# 三、单 Token 反向 KL 损失究竟在优化什么

一种非常常见的 OPD 实现，是使用当前 Token 的对数概率比 $\log\frac{\pi_{\mathrm{sg}(\theta)}(a_t|s_t)}{\pi_{T(x)}(a_t|s_t)}$ 作为策略梯度的权重，即：

$$
\begin{align}g _{\mathrm{tok}}\left( \textcolor{red}{\theta}\right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{T\left( x \right)}\left( a_t|s_t \right)}\cdot\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right].\end{align}
$$

*其中 $s_t:=(x,y_{<t})$、$a_t:=y_t$。* 可以看出，式（5）是式（4）的一个特殊变体，对应 $\hat A_\theta(s_t,a_t)=\log\frac{\pi_\theta(a_t|s_t)}{\pi_{T(x)}(a_t|s_t)}$。由定理 1 可知：

$$
g_{\mathrm{tok}}\left( \textcolor{red}{\theta} \right) \ne \nabla _{\theta}\mathcal{L} \left(  \textcolor{red}{\theta} \right).
$$

下面分析这种方法实际上优化的是什么，以及梯度 $g_{\mathrm{tok}}$ 对应的目标函数是什么。回顾式（3），OPD 的目标可以写成：

$$
\mathcal{L} \left( \textcolor{red}{\theta} \right) =\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right].
$$

若将状态占用度 $d_\theta$ 中的参数停止梯度，并定义：

$$
\begin{align}\mathcal{L} _{\mathrm{tok}}\left( \textcolor{red}{\theta} \right) =\mathbb{E} _{s\sim d_{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right],\end{align}
$$

根据链式法则，有：

$$
\small{\begin{align}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right) &=\mathbb{E} _{s\sim d_{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}}\left[ \nabla _{\theta}\mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right)\right]+\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta} \right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right] 
\\
\,\,         &=\nabla _{\theta}\mathcal{L} _{\mathrm{tok}}\left( \textcolor{red}{\theta}\right) +\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right] . \end{align}}
$$

进一步展开 $\nabla_\theta\mathcal L_{\mathrm{tok}}(\theta)$，可以得到：

$$
\begin{align*}\nabla _{\theta}\mathcal{L} _{\mathrm{tok}}\left(\textcolor{red}{\theta}\right) &=\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \nabla _{\theta}\mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right] 
\\
\,\,            &\overset{\left( a \right)}{=}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\nabla _{\theta}\mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s_t \right) ,\pi _{T\left( x \right)}\left( \cdot |s_t \right) \right)} \right] 
\\
\,\,            &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\nabla _{\theta}\mathbb{E} _{a_t\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |s_t \right)}\left[ \log \frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{T\left( x \right)}\left( a_t|s_t \right)} \right]} \right] 
\\
\,\,            &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{E} _{a_t\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |s_t \right)}\left[ \log \frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{T\left( x \right)}\left( a_t|s_t \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right) \right]} \right] 
\\
\,\,            &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{T\left( x \right)}\left( a_t|s_t \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] 
\\
\,\,            &=g_{\mathrm{tok}}\left( \theta \right) \end{align*}
$$

其中（a）由引理 2 得到。**这个结果表明：在式（4）中使用 $\hat A_\theta(s_t,a_t)=\log\frac{\pi_\theta(a_t|s_t)}{\pi_{T(x)}(a_t|s_t)}$ 的 OPD 实现，在当前参数点实际优化的是 $\mathcal L_{\mathrm{tok}}(\theta)$。然而，$\nabla_\theta\mathcal L_{\mathrm{tok}}(\theta)$ 只是真实 OPD 梯度 $\nabla_\theta\mathcal L(\theta)$ 的一部分；它缺少关于状态占用度 $d_\theta$ 的梯度，即 $\nabla_\theta\mathbb E_{s\sim d_\theta}[\mathrm{KL}(\pi_{\mathrm{sg}(\theta)}(\cdot|s),\pi_{T(s)}(\cdot|s))]$。**

# 四、状态占用度的梯度

如式（8）所示，OPD 目标函数的梯度满足：

$$
\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right) =\nabla _{\theta}\mathcal{L} _{\mathrm{tok}}\left( \textcolor{red}{\theta}\right) +\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right], \end{align*}
$$

其中 $\mathcal L_{\mathrm{tok}}(\theta)$ 定义见式（6）。对 $\mathcal L_{\mathrm{tok}}(\theta)$ 做梯度下降，会在学生模型实际访问到的状态 $s$ 上最小化反向 KL 散度 $\mathrm{KL}(\pi_\theta(\cdot|s),\pi_{T(s)}(\cdot|s))$。但这种优化只覆盖学生已经访问到的状态，**并没有考虑如何优化整条轨迹所诱导的状态分布**。下面推导状态占用度的梯度，并说明：当我们对原始 OPD 目标 $\mathcal L(\theta)$ 做梯度下降时，在每个状态 $s$ 上优化下一个 Token 的生成，必须考虑该 Token 会如何改变其后子轨迹中学生模型与教师模型之间的 KL 散度。首先给出一个关于状态占用度梯度的定理。

---

**定理 2（参数化状态占用度下期望的梯度）.** *对于任意参数 $\theta$ 和实值函数 $f:\mathcal S\rightarrow\mathbb R$：*

$$
\begin{align}
\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ f\left( s \right) \right] =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] 
\end{align}
$$

*其中 $s_t:=(x,y_{<t})$、$a_t:=y_t$。*

*证明：回顾状态占用度的定义：*

$$
\begin{align*}d_{\textcolor{red}{\theta}}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right].\end{align*}
$$

*因此直接计算可得：*

$$
\begin{align*}\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ f\left( s \right) \right] &=\sum_{s\in \mathcal{S}}{\nabla _{\theta}d_{\textcolor{red}{\theta}}\left( s \right) f\left( s \right)}\\\,\,                  &=\sum_{s\in \mathcal{S}}{\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right] \cdot f\left( s \right)}\\\,\,                  &=\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left( s_t \right)} \right] \\\,\,                  &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \left( \sum_{t=0}^{\left| y \right|-1}{f\left( s_t \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y|x \right) \right] \\\,\,                  &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] \\\,\,                  &\overset{\left( a \right)}{=}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] \end{align*}
$$

*其中（a）来自以下事实：对于任意两个固定时间索引 $0\le t'\le t\le |y|-1$，有：*

$$

\begin{aligned}
	\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right]\\
	\,\,&=\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right]\\
	\,\,&=\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \right]\\
	\,\,&=\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot 0 \right]\\
	\,\,&=0\\
\end{aligned}

$$

*其中期望 $\mathbb E$ 是对 $x\sim\mathcal D$、$y\sim\pi_\theta$ 取的。*

---

直接应用定理 2，可得：

$$
\begin{align*}&\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right]\\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\textcolor{red}{\theta}}\left( \cdot |s \right) ,\pi _{T\left( x \right)}\left( \cdot |s \right) \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] \\\,\,                                              &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] \\\,\,                                              &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\underset{\mathrm{future\ log\ ratio}}{\underbrace{\log \frac{\pi _{\textcolor{red}{\theta}}\left( y_{\textcolor{red}{>t}}|s_{t+1} \right)}{\pi _{T\left( x \right)}\left( y_{\textcolor{red}{>t}}|s_{t+1} \right)}}}\cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)} \right] \end{align*}
$$

# 五、使用 Top-K Token 近似完整 KL 散度

> **待补充：** 原文此处尚未完成。

# 六、总结

### 6.1 优化目标

OPD 的目标，是最小化学生模型 $\pi_\theta$ 与教师模型 $\pi_{T(x)}$ 在文本生成分布上的反向 KL 散度：

$$
\begin{align*}\mathcal{L} \left(\textcolor{red}{ \theta} \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}(\cdot|x)}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) \right], \end{align*}
$$

它可以改写为：

$$
\begin{align*} 
\mathcal{L} \left( \textcolor{red}{ \theta} \right) =\mathbb{E} _{s\sim d_{\textcolor{red}{ \theta} }}\left[ \mathrm{KL}\left( \pi _{\textcolor{red}{ \theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right] =\mathbb{E} _{s\sim d_{\textcolor{red}{ \theta} }}\mathbb{E} _{a\sim \pi _{\textcolor{red}{ \theta} }\left( \cdot |s \right)}\left[ \log \frac{\pi _{\textcolor{red}{ \theta} }\left( a|s \right)}{\pi _{T\left( s \right)}\left( a|s \right)} \right]. 
\end{align*}
$$

其中，$d_\pi$ 为状态占用度，定义为：

$$
\begin{align*}d_{\pi}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right].\end{align*}
$$

### 6.2 梯度分解

OPD 目标 $\mathcal L(\theta)$ 的完整梯度由两部分构成：

$$
\small{\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta} \right)=\nabla _{\theta}\mathcal{L} _{\mathrm{tok}}\left( \textcolor{red}{\theta}\right) +\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right], \end{align*}}
$$

其中，$\mathcal L_{\mathrm{tok}}(\theta)=\mathbb E_{s\sim d_{\mathrm{sg}(\theta)}}[\mathrm{KL}(\pi_\theta(\cdot|s),\pi_{T(s)}(\cdot|s))]$。

- **第 1 部分：在给定状态 $(x,y_{<t})$ 时，优化当前 Token $y_t$ 的生成**

$$
\begin{align*}\nabla _{\theta}\mathcal{L} _{\mathrm{tok}}\left(\textcolor{red}{\theta}\right)&=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\nabla _{\theta}\text{KL}\left( \pi_{\textcolor{red}{\theta}}(\cdot|x,y_{<t}),\pi_{T(x)}(\cdot|x,y_{<t})\right)} \right] \\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( y_t|x,y_{<t} \right)}{\pi _{T\left( x \right)}\left( y_t|x,y_{<t} \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y_t|x,y_{<t}\right)} \right] 
 \end{align*}
$$

- **第 2 部分：在给定状态 $(x,y_{\le t})$ 时，优化未来 Token 序列 $y_{>t}$ 的生成**

$$
\begin{align*}&\nabla _{\theta}\mathbb{E} _{s\sim d_{\textcolor{red}{\theta}}}\left[ \mathrm{KL}\left( \pi _{\mathrm{sg}\left( \textcolor{red}{\theta}\right)}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) \right]\\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\underset{\mathrm{future\ log\ ratio}}{\underbrace{\log \frac{\pi _{\textcolor{red}{\theta}}\left( y_{\textcolor{red}{>t}}|x,y_{\le t} \right)}{\pi _{T\left( x \right)}\left( y_{\textcolor{red}{>t}}|x,y_{\le t} \right)}}}\cdot \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y_t|x,y_{<t} \right)} \right] \end{align*}
$$

将两部分合并，即得到 OPD 目标函数的完整梯度：

$$
\begin{align*}\nabla _{\theta}\mathcal{L} \left( \textcolor{red}{\theta}  \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{ \theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\hat{A}_{\textcolor{red}{ \theta}}\left( s_t,a_t \right)  \cdot \nabla _{\theta}\log \pi _{\textcolor{red}{ \theta}}\left( a_t|s_t \right)} \right],\end{align*}
$$

其中 $s_t:=(x,y_{<t})$、$a_t:=y_t$，$\hat A_\theta(s_t,a_t)$ 可以采用以下形式：

(i) $\sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}=\frac{\log \pi _{\theta}\left( y|x \right)}{\log \pi _{T\left( x \right)}\left( y|x \right)}$ ：整条轨迹的对数概率比

（ii） $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}=\log \frac{\pi _{\theta}\left( y_{\textcolor{red}{\ge t}}|x,y_{<t} \right)}{\pi _{T\left( x \right)}\left( y_{\textcolor{red}{\ge t}}|x,y_{<t} \right)}$：从 Token $y_t$ 开始的后续子轨迹对数概率比之和

（iii） $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} -b(s_{\textcolor{red}{t}})$：形式（ii）加入基线后的版本

（iv） $\log \frac{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{T\left( x \right)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}+\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)}$ ：当前 Token 的对数概率比，加上其后子轨迹中累积的反向 KL 散度

（v） $\log \frac{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{T\left( x \right)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}+\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)} -b(s_{\textcolor{red}{t}})$：形式（iv）加入基线后的版本