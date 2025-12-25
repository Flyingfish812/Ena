# 数学推导

## 导言：评估目标澄清（Evaluation Objective Clarification）

本项目中的“多尺度评估”并非旨在回答“模型在某一尺度上的误差是否较小”，而是回答以下更具物理意义的问题：

> **在逐步引入更小空间尺度（更高波数）的过程中，模型的预测结果在何种尺度以下不再携带可区分于噪声的物理信息？**

换言之，我们关心的不是误差大小本身，而是：

* 模型预测在不同空间尺度上的**结构一致性**；
* 模型是否在某一尺度以下仅输出统计意义上的噪声或平滑残余；
* 是否存在一个“有效恢复尺度上界”，可用于指导模型设计与后续分析中的**自适应尺度裁切**。

因此，**误差幅值类指标（如 NRMSE）仅作为诊断量存在，而不作为定义“可恢复尺度”的主判据。**

## 线性基线：POD 子空间中的最小二乘反演（数学推导）

### 0. 记号与数据空间

对每个样本（时间帧）$t=1,\dots,T$，原始场为
$$
X_t\in\mathbb{R}^{H\times W\times C}.
$$
展平算子 $\operatorname{vec}(\cdot)$ 将其映射到
$$
x_t=\operatorname{vec}(X_t)\in\mathbb{R}^{D},\qquad D=HWC.
$$

定义样本矩阵（把每个样本作为一行）
$$
\mathbf{X}=
\begin{bmatrix}
x_1^\top\
\vdots\
x_T^\top
\end{bmatrix}
\in\mathbb{R}^{T\times D}.
$$

---

### 1. POD（截断的正交基表示）

令 $\mu\in\mathbb{R}^{D}$ 为样本均值（是否取决于是否做中心化/去均值），定义中心化样本
$$
\tilde x_t = x_t-\mu.
$$
将中心化样本矩阵写为 $\tilde{\mathbf{X}}\in\mathbb{R}^{T\times D}$。

对 $\tilde{\mathbf{X}}$ 做 POD（可视为对协方差 $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ 的特征分解，或对 $\tilde{\mathbf{X}}$ 的 SVD），得到一组按能量排序的正交空间模态（列向量）
$$
U=\begin{bmatrix}u_1&u_2&\cdots\end{bmatrix}\in\mathbb{R}^{D\times D},\qquad U^\top U=I.
$$
取前 $r$ 个模态组成截断基
$$
U_r=[u_1,\dots,u_r]\in\mathbb{R}^{D\times r},\qquad U_r^\top U_r=I_r.
$$

POD 假设（截断模型）是：每个场可由低维系数 $a_t\in\mathbb{R}^r$ 线性表示：
$$
x_t \approx \mu + U_r a_t.
$$
在该正交基下的“真系数”（最小二乘意义下的投影系数）为
$$
a_t^\star=\arg\min_{a\in\mathbb{R}^r}|x_t-(\mu+U_r a)|_2^2
=U_r^\top(x_t-\mu).
$$

---

### 2. 稀疏观测模型（采样算子 + 噪声）

在空间网格上随机选取观测点集合 $\Omega\subset{1,\dots,D}$，其中每个观测点对应展平向量的一个坐标（代码里是按 $H\times W$ 选点并复制到通道，数学上等价于给定一个索引集合 $\Omega$）。

用选择矩阵（采样算子）
$$
S\in{0,1}^{M\times D},\qquad M=|\Omega|
$$
表示“从 $D$ 维向量中抽取 $\Omega$ 上的分量”，于是无噪声观测为
$$
y_t=Sx_t\in\mathbb{R}^{M}.
$$
加入独立同分布高斯噪声
$$
\varepsilon_t\sim\mathcal{N}(0,\sigma^2I_M),
$$
得到带噪观测
$$
\tilde y_t = y_t+\varepsilon_t = Sx_t+\varepsilon_t.
$$

将 POD 截断模型代入观测方程：
$$
\tilde y_t
\approx S(\mu+U_r a_t)+\varepsilon_t
= S\mu + (SU_r)a_t+\varepsilon_t.
$$
记
$$
A := SU_r\in\mathbb{R}^{M\times r},
\qquad b_t := \tilde y_t - S\mu\in\mathbb{R}^{M},
$$
则观测方程写为近似线性回归
$$
b_t \approx A a_t + \varepsilon_t.
$$

---

### 3. 线性基线：POD 系数的最小二乘估计

在噪声为高斯、协方差为 $\sigma^2I$ 的假设下，最大似然估计等价于最小二乘：
$$
a_t^{\mathrm{LS}}
= \arg\min_{a\in\mathbb{R}^r}\ |Aa-b_t|*2^2
= \arg\min*{a}\ | (SU_r)a-(\tilde y_t-S\mu)|_2^2.
$$

* 若 $A$ 满列秩（$\mathrm{rank}(A)=r$，通常需要 $M\ge r$ 且观测位置“足够好”），则解唯一：
  $$
  a_t^{\mathrm{LS}}=(A^\top A)^{-1}A^\top b_t.
  $$

* 若 $A$ 退化或欠定（例如 $M<r$ 或 $\mathrm{rank}(A)<r$），则最小二乘解不唯一；常用的选择是**最小范数解**：
  $$
  a_t^{\mathrm{LS}} = A^+ b_t,
  $$
  其中 $A^+$ 是 Moore–Penrose 伪逆。这也是数值库常见默认行为。

---

### 4. 回到物理空间的重建

将估计系数带回 POD 截断模型：
$$
\hat x_t^{\mathrm{lin}} = \mu + U_r a_t^{\mathrm{LS}}.
$$
必要时再 reshape 为 $\hat X_t^{\mathrm{lin}}\in\mathbb{R}^{H\times W\times C}$。

---

### 5. 误差度量（用于实验汇总）

对每个 $t$ 可定义任意场级误差 $E(\hat x_t, x_t)$（如 NMSE/NMAE/PSNR 等），再对 $t=1,\dots,T$ 求均值、方差得到曲线点。

此外，你的流程还会在 POD 系数空间对误差做“按模态/按频带分组”的统计：给定一组模态索引分组 ${\mathcal{B}*k}$（例如 L/M/H 三段），可定义系数误差的分组量
$$
\mathrm{RMSE}(\mathcal{B}*k)
=\sqrt{\frac{1}{T|\mathcal{B}*k|}\sum*{t=1}^T\sum*{j\in\mathcal{B}*k}(a*{t,j}^{\mathrm{LS}}-a*{t,j}^\star)^2},
$$
以及相对误差版本（NRMSE）等；也可定义“仅用部分模态重建”的场级 NMSE，用来刻画尺度累积误差随模态段增长的变化。

（到这里为止，推导链条是完全闭合且自洽的：POD 给出低维线性子空间；采样算子给出欠定/适定线性观测；最小二乘给出系数估计；再回到物理空间。）

---

## MLP基线

### 一、问题设定与符号

仍沿用同一组 POD 基底（已在前文给出）：对任意样本 $t$，展平场 $x_t\in\mathbb{R}^D$，均值 $\mu\in\mathbb{R}^D$，截断基底 $U_r\in\mathbb{R}^{D\times r}$，满足近似表示
$$
x_t \approx \mu + U_r a_t,\qquad a_t\in\mathbb{R}^r.
$$
并定义“真实 POD 系数”
$$
a_t^\star = U_r^\top (x_t-\mu).
$$

采样由固定的空间 mask 决定，等价于选择矩阵（采样算子）
$$
S\in{0,1}^{M\times D},\quad M\ll D,
$$
它抽取观测坐标集合 $\Omega$ 上的分量，观测向量为
$$
y_t = Sx_t \in\mathbb{R}^M.
$$
噪声模型（训练/评估可选）为
$$
\tilde y_t = y_t + \varepsilon_t,\qquad \varepsilon_t\sim \mathcal N(0,\sigma^2 I_M).
$$

---

### 二、MLP 基线的学习目标

#### 2.1 监督学习任务：从观测到系数

MLP 的目标是学习一个参数化映射
$$
f_\theta:\mathbb{R}^M\to\mathbb{R}^r,
$$
使得在带噪观测 $\tilde y_t$ 上输出系数预测
$$
\hat a_t = f_\theta(\tilde y_t)
$$
尽量逼近真实系数 $a_t^\star$。

训练样本由 ${(\tilde y_t, a_t^\star)}_{t=1}^N$ 组成，其中 $N$ 为用于训练/验证的快照数量。

#### 2.2 损失函数：系数空间的均方误差

采用标准 MSE 作为监督损失：
$$
\mathcal{L}(\theta)
===================

\frac{1}{N}
\sum_{t=1}^N
\left|
f_\theta(\tilde y_t)-a_t^\star
\right|_2^2.
$$
在 mini-batch SGD（更准确说 Adam）下，使用 batch $B$ 的无偏估计：
$$
\mathcal{L}_B(\theta)
=====================

\frac{1}{|B|}
\sum_{t\in B}
\left|
f_\theta(\tilde y_t)-a_t^\star
\right|_2^2.
$$

---

### 三、输入与标签的构造（Observation Dataset 的数学形式）

给定全量快照展平矩阵
$$
\mathbf{X}=
\begin{bmatrix}
x_1^\top\
\vdots\
x_N^\top
\end{bmatrix}\in\mathbb{R}^{N\times D},
$$
数据集构造分两步：

1. **标签（真实系数）预计算**
   对每个样本 $t$，计算
   $$
   a_t^\star = U_r^\top (x_t-\mu).
   $$
   记
   $$
   \mathbf{A}^\star=
   \begin{bmatrix}
   (a_1^\star)^\top\
   \vdots\
   (a_N^\star)^\top
   \end{bmatrix}\in\mathbb{R}^{N\times r}.
   $$

2. **输入（观测向量）在线生成**
   对每个样本 $t$，抽取稀疏观测并加噪：
   $$
   y_t = Sx_t\in\mathbb{R}^M,\qquad
   \tilde y_t = y_t + \varepsilon_t.
   $$

因此数据集提供的训练对为
$$
(\tilde y_t,; a_t^\star),\quad t=1,\dots,N.
$$

---

### 四、MLP 模型与训练过程（优化问题）

#### 4.1 网络结构（抽象表达）

令 $f_\theta$ 为多层感知机（MLP），可抽象成 $L$ 层仿射变换与非线性激活的复合：
$$
h^{(0)} = \tilde y\in\mathbb{R}^{M},
$$
$$
h^{(\ell)} = \phi!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right),\quad \ell=1,\dots,L-1,
$$
$$
f_\theta(\tilde y) = W^{(L)} h^{(L-1)} + b^{(L)}\in\mathbb{R}^{r},
$$
其中 $\phi(\cdot)$ 为逐元素非线性（如 ReLU/SiLU 等），参数集合 $\theta={W^{(\ell)},b^{(\ell)}}$。

（具体隐藏层维度属于实现细节，数学上只需该映射是可微的参数化函数。）

#### 4.2 训练目标与优化

训练求解的就是经验风险最小化：
$$
\theta^\star
=

\arg\min_{\theta}
\frac{1}{N}
\sum_{t=1}^N
\left|
f_\theta(\tilde y_t)-a_t^\star
\right|_2^2
;+;
\lambda|\theta|_2^2,
$$
其中 $\lambda\ge 0$ 对应权重衰减（weight decay）的 $L_2$ 正则项（若启用）。

采用 Adam 进行迭代更新（以一阶/二阶动量的自适应学习率形式近似求解），直到达到指定 epoch 数；同时使用验证集损失选择最佳模型参数（早停式的 best-checkpoint 选择）。

---

### 五、推理与场重建（从 $\hat a$ 回到 $\hat x$）

模型训练完成后，对新样本 $x$ 的稀疏观测 $\tilde y$：

1. 预测 POD 系数：
   $$
   \hat a = f_{\theta^\star}(\tilde y).
   $$

2. 用同一组 POD 基底重建场：
   $$
   \hat x^{\mathrm{mlp}} = \mu + U_r \hat a.
   $$

这给出“MLP 基线”的最终场级重建结果。

---

### 六、误差指标与多尺度（按系数/按场）

#### 6.1 场空间误差（基本指标）

对任一重建 $\hat x$ 与真值 $x$，定义误差向量 $e=\hat x-x$。基础指标包括：

* MSE：
  $$
  \mathrm{MSE} = \frac{1}{D}|e|_2^2.
  $$
* RMSE：
  $$
  \mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
  $$
* NMSE：
  $$
  \mathrm{NMSE}=\frac{|e|_2^2}{|x|_2^2}.
  $$
* NMAE：
  $$
  \mathrm{NMAE}=\frac{\frac1D|e|_1}{\frac1D|x|_1}.
  $$
* PSNR（给定数据范围 $R$）：
  $$
  \mathrm{PSNR}=10\log_{10}\frac{R^2}{\mathrm{MSE}}.
  $$

若对 batch ${(\hat x_t,x_t)}$ 统计，可对上述标量在 $t$ 上取均值（并可忽略分母为零导致的 NaN）。

#### 6.2 系数空间误差：逐模态 RMSE / NRMSE

在系数空间中，给定预测 $\hat a_t$ 与真值 $a_t^\star$，逐模态 RMSE 谱定义为（对每个模态 $k$）：
$$
\mathrm{RMSE}_k
=

\sqrt{
\frac{1}{N}
\sum_{t=1}^N
(\hat a_{t,k}-a_{t,k}^\star)^2
}.
$$

逐模态 NRMSE 有两种归一化方式：

* 若给定 POD 特征值 $\lambda_k$（或等价能量尺度），则
  $$
  \mathrm{NRMSE}_k=\frac{\mathrm{RMSE}_k}{\sqrt{\lambda_k}}.
  $$
* 若不提供 $\lambda_k$，则用真值系数在时间维度的标准差 $\sigma_k=\mathrm{std}*t(a*{t,k}^\star)$ 归一化：
  $$
  \mathrm{NRMSE}_k=\frac{\mathrm{RMSE}_k}{\max(\sigma_k,\varepsilon)}.
  $$

#### 6.3 系数空间误差：按 band 分组（per-band）

给定一组模态区间（band）划分
$$
\mathcal{B}={(s_b,e_b)},
$$
每个 band 对应模态索引集合 ${k: s_b\le k < e_b}$。定义 band 内 NRMSE 为
$$
\mathrm{NRMSE}_{b}
=

\sqrt{
\frac{
\sum_{t=1}^N \sum_{k\in b} (\hat a_{t,k}-a_{t,k}^\star)^2
}{
\sum_{t=1}^N \sum_{k\in b} (a_{t,k}^\star)^2
}
}.
$$
（$RMSE_b$ 则是对 band 内平方误差取平均再开方。）

#### 6.4 基于模态分组的“部分重建 NMSE”（single-group / cumulative）

给定一组模态分组 ${G_g}$（如按 16 个模态一组），构造“仅使用某组模态”的系数
$$
a^{(g)}*{t,k}=
\begin{cases}
a*{t,k}, & k\in G_g,\
0, & \text{otherwise},
\end{cases}
$$
以及“从低阶累积到第 $g$ 组”的系数
$$
a^{(\le g)}*{t,k}=
\begin{cases}
a*{t,k}, & k\in \cup_{j\le g}G_j,\
0, & \text{otherwise}.
\end{cases}
$$
相应部分重建场为
$$
x_t^{(g)}=\mu+U_r a_t^{(g)},\qquad
x_t^{(\le g)}=\mu+U_r a_t^{(\le g)}.
$$
对预测与真值分别做上述“单组/累积”重建后，可在场空间计算 NMSE：
$$
\mathrm{NMSE}^{(g)}=\frac{|x^{(g)}*{\text{hat}}-x^{(g)}*{\text{true}}|*2^2}{|x^{(g)}*{\text{true}}|*2^2},
\qquad
\mathrm{NMSE}^{(\le g)}=\frac{|x^{(\le g)}*{\text{hat}}-x^{(\le g)}_{\text{true}}|*2^2}{|x^{(\le g)}*{\text{true}}|_2^2},
$$
并对样本求均值或保留逐样本向量，从而得到“有效截止模态/多尺度误差累积曲线”一类分析对象。

## 空间尺度评估

### 一、离散场与物理尺度标定

考虑二维空间网格上的标量场（多通道情形见附注）。对每个时间帧 $t\in{1,\dots,T}$，给定真值与预测场：
$$
x_t(i,j),\ \hat x_t(i,j),\qquad i=0,\dots,N_x-1,\ j=0,\dots,N_y-1.
$$
其中 $(N_x,N_y)=(640,80)$。

空间域为
$$
(x,y)\in[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]
=[-0.5,7.5]\times[-0.5,0.5],
$$
故
$$
L_x=x_{\max}-x_{\min}=8,\qquad L_y=y_{\max}-y_{\min}=1,
$$
并采用等距网格
$$
\Delta x=\frac{L_x}{N_x},\qquad \Delta y=\frac{L_y}{N_y}.
$$

对 Cylinder Flow 数据，圆柱半径 $R=0.0625$，直径
$$
D=2R=0.125.
$$
本节的“物理尺度”统一以 $D$ 作为特征长度，以 $\lambda/D$ 表示波长尺度（若数据集无量纲化，这仍是最自然的物理尺度基准）。

---

### 二、二维离散傅里叶变换与波数网格

对每个时间帧 $t$，定义二维离散傅里叶变换（DFT）：
$$
\hat x_t(p,q)=
\sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1}
x_t(i,j),
e^{-2\pi i\left(\frac{pi}{N_x}+\frac{qj}{N_y}\right)},
\qquad
p=0,\dots,N_x-1,\ q=0,\dots,N_y-1.
$$
预测场 $\hat x_t(i,j)$ 的傅里叶谱记为 $\widehat{\hat x}_t(p,q)$，误差场
$$
e_t(i,j)=\hat x_t(i,j)-x_t(i,j)
$$
的傅里叶谱为 $\hat e_t(p,q)=\widehat{\hat x}_t(p,q)-\hat x_t(p,q)$。

为建立“尺度”（长度）与频域索引的对应，引入离散波数（空间频率）网格。令整数频率索引
$$
\tilde p=
\begin{cases}
p, & 0\le p\le \frac{N_x}{2},\
p-N_x, & \frac{N_x}{2}<p<N_x,
\end{cases}
\qquad
\tilde q=
\begin{cases}
q, & 0\le q\le \frac{N_y}{2},\
q-N_y, & \frac{N_y}{2}<q<N_y,
\end{cases}
$$
对应的物理波数为
$$
k_x(\tilde p)=\frac{2\pi}{L_x}\tilde p,\qquad
k_y(\tilde q)=\frac{2\pi}{L_y}\tilde q.
$$
于是每个频域网格点 $(p,q)$ 对应波数向量
$$
\mathbf{k}(p,q)=\big(k_x(\tilde p),,k_y(\tilde q)\big),
\qquad
k(p,q)=|\mathbf{k}(p,q)|_2.
$$

---

### 三、从波数到波长尺度

定义与波数模长对应的“等效波长”（仅在 $k>0$ 时有意义）：
$$
\lambda(p,q)=\frac{2\pi}{k(p,q)}.
$$
为了得到无量纲尺度变量，采用以圆柱直径 $D$ 归一化的尺度：
$$
s(p,q)=\frac{\lambda(p,q)}{D}=\frac{2\pi}{D,k(p,q)}.
$$

由于离散采样的 Nyquist 限制，最小可分辨波长（在各向同性近似下）满足
$$
\lambda_{\min}\approx 2\max(\Delta x,\Delta y),
\qquad
s_{\min}=\frac{\lambda_{\min}}{D}.
$$
因此尺度分解在 $s<s_{\min}$ 的区间不具备可辨识性；论文中所有尺度分段应满足下限不小于 $s_{\min}$。

---

### 四、尺度分段（band）与尺度滤波算子

设定一组尺度区间（band）：
$$
\mathcal{S}_1=[s_1^{\min},s_1^{\max}),\
\mathcal{S}_2=[s_2^{\min},s_2^{\max}),\ \dots,
\mathcal{S}_B=[s_B^{\min},s_B^{\max}),
$$
例如按对数间隔或按导师提出的“数量级”式划分（在 Cylinder Flow 中可用 $s=\lambda/D$ 做基准）。

对每个 band $b$，在频域定义尺度掩码（带通滤波器）：
$$
H_b(p,q)=
\begin{cases}
1, & s(p,q)\in \mathcal{S}_b,\
0, & \text{otherwise}.
\end{cases}
$$
也可用平滑过渡的 $H_b$ 实现“弱分割”（例如在 band 边界附近用连续窗函数使 $H_b\in[0,1]$）。

于是 band 的频域分量为
$$
\hat x_t^{(b)}(p,q)=H_b(p,q),\hat x_t(p,q),
\qquad
\widehat{\hat x}_t^{(b)}(p,q)=H_b(p,q),\widehat{\hat x}_t(p,q),
\qquad
\hat e_t^{(b)}(p,q)=H_b(p,q),\hat e_t(p,q).
$$

---

### 五、尺度分辨误差评估

#### 5.1 频域能量与误差能量（不需要 iFFT）

定义真值场在 band $b$ 上的频域能量（对单帧）：
$$
E_{x,t}^{(b)}=
\sum_{p,q}\left|\hat x_t^{(b)}(p,q)\right|^2,
$$
误差能量：
$$
E_{e,t}^{(b)}=
\sum_{p,q}\left|\hat e_t^{(b)}(p,q)\right|^2.
$$
对时间/样本取平均得到统计量：
$$
E_{x}^{(b)}=\frac{1}{T}\sum_{t=1}^T E_{x,t}^{(b)},
\qquad
E_{e}^{(b)}=\frac{1}{T}\sum_{t=1}^T E_{e,t}^{(b)}.
$$

由此定义尺度分辨的相对误差（频域 NRMSE 型）：
$$
\mathrm{NRMSE}_{\mathrm{F}}^{(b)}
=

\sqrt{\frac{E_e^{(b)}}{E_x^{(b)}+\epsilon}},
$$
其中 $\epsilon$ 为避免分母为零的极小正数。

该指标回答：“在波长区间 $\mathcal{S}_b$（以 $D$ 归一化）内，重建误差相对于真值能量的比例是多少”。

#### 5.2 径向谱（连续尺度曲线）

若希望得到“误差随尺度连续变化”的曲线，可对波数模长 $k$ 做径向分箱。令 ${[k_m,k_{m+1})}*{m=1}^M$ 为波数 bins，并定义对应的频域环形集合
$$
\Omega_m={(p,q): k(p,q)\in[k_m,k*{m+1})}.
$$
则真值能谱与误差能谱定义为
$$
E_x(k_m)=\frac{1}{T}\sum_{t=1}^T\sum_{(p,q)\in\Omega_m}|\hat x_t(p,q)|^2,
\qquad
E_e(k_m)=\frac{1}{T}\sum_{t=1}^T\sum_{(p,q)\in\Omega_m}|\hat e_t(p,q)|^2,
$$
以及尺度分辨误差
$$
\mathrm{NRMSE}_{\mathrm{F}}(k_m)=\sqrt{\frac{E_e(k_m)}{E_x(k_m)+\epsilon}}.
$$
再由 $\lambda_m=2\pi/k_m$ 与 $s_m=\lambda_m/D$ 进行横轴变换，即得到以 $\lambda/D$ 为横轴的尺度误差曲线。

---

### 六、尺度分量的空间域可视化（需要 iFFT）

若需要像“POD 模态分组图”那样展示不同尺度在空间域的形态，对每个 band $b$，定义 band-pass 的空间分量（单帧）：
$$
x_t^{(b)}(i,j)=\mathcal{F}^{-1}{\hat x_t^{(b)}}(i,j),
\qquad
\hat x_t^{(b)}(i,j)=\mathcal{F}^{-1}{\widehat{\hat x}_t^{(b)}}(i,j),
\qquad
e_t^{(b)}(i,j)=\hat x_t^{(b)}(i,j)-x_t^{(b)}(i,j).
$$
其中 $\mathcal{F}^{-1}$ 为二维逆 DFT。

在空间域可对每个尺度段计算 NMSE（或 RMSE）：
$$
\mathrm{NMSE}*{\mathrm{S}}^{(b)}=
\frac{\sum*{i,j}\left(\hat x_t^{(b)}(i,j)-x_t^{(b)}(i,j)\right)^2}
{\sum_{i,j}\left(x_t^{(b)}(i,j)\right)^2+\epsilon},
$$
并在 $t$ 上取平均得到尺度段的空间域误差指标。该指标与 5.1 的频域指标在能量守恒（Parseval）意义下是一致的尺度评估，只是表述域不同。

尺度可视化图可采用三行结构：

* 真值尺度分量 $x_t^{(b)}$
* 预测尺度分量 $\hat x_t^{(b)}$
* 误差尺度分量 $e_t^{(b)}$

并以 $\mathcal{S}_b$（或 $\lambda$ 的范围）标注每个 panel。

---

### 七、多通道场的处理（简述）

若场具有通道 $c=1,\dots,C$，则对每个通道分别定义 FFT 与尺度分段，并在评估时采用以下任一规范：

1. **逐通道评估**：对每个通道独立计算 $\mathrm{NRMSE}^{(b)}$，最后报告各通道结果。
2. **能量合并评估**：定义总能量为各通道能量之和，例如
   $$
   E_x^{(b)}=\sum_{c=1}^C E_{x,c}^{(b)},\qquad E_e^{(b)}=\sum_{c=1}^C E_{e,c}^{(b)},
   $$
   再代入 NRMSE 定义。

### 八、自适应尺度截止波数 $k^*$ 的定义（Adaptive Spectral Cutoff）

本项目中，我们关心的问题并非单一波数处的误差大小，而是：

> **从物理尺度的角度，一个模型在逐步引入更小尺度（更高波数）信息时，能够有效恢复到何种尺度为止。**

为此，我们引入一种**基于累计频谱能量与重建误差的自适应尺度截止定义**，用于刻画模型在空间频域中的“可恢复尺度极限”。

---

#### 8.1 累计低通重建误差（Cumulative Low-pass NRMSE）

在前述径向谱分箱的基础上，设
$$
{k_m}*{m=1}^M
$$
为按从小到大排序的波数 bins 中心值（仅考虑 $k_m \ge k*{\min}$，其中 $k_{\min}$ 为物理上有意义的最小波数，通常取 $k_{\min}=1$）。

定义截至波数 $K$ 的**累计真值能量**与**累计误差能量**为：
$$
E_x^{(\le K)}
=

\sum_{k_m \le K} E_x(k_m),
\qquad
E_e^{(\le K)}
=

\sum_{k_m \le K} E_e(k_m),
$$
其中 $E_x(k_m)$、$E_e(k_m)$ 分别为真值场与误差场在径向波数 bin $k_m$ 上的平均频域能量（见 §5.2）。

据此定义累计低通重建误差（累计 NRMSE）：
$$
\mathrm{NRMSE}_{\le K}
=

\sqrt{
\frac{E_e^{(\le K)}}{E_x^{(\le K)}+\epsilon}
},
$$
其中 $\epsilon>0$ 为防止分母为零的极小正数。

该量刻画了：**若仅要求模型重建到波数不超过 $K$（即空间尺度不小于 $\ell=1/K$），其整体重建误差水平**。

---

#### 8.2 自适应尺度截止波数 $k^*$

随着 $K$ 的增大，累计误差 $\mathrm{NRMSE}_{\le K}$ 通常表现为：

* 在低波数（大尺度）区间快速下降；
* 在达到某一波数后趋于平缓；
* 继续引入更高波数分量，对整体重建误差的改善不再显著。

基于这一现象，定义**自适应尺度截止波数** $k^*$ 为：

$$
k^*=

\min\left\{
K \ge k_{\min}
\ \middle|\ 
\mathrm{NRMSE}_{\le K+\Delta K}
-
\mathrm{NRMSE}_{\le K}
< \varepsilon
\quad
\text{连续 } m \text{ 个 bins}
\right\},
$$

其中：

* $\Delta K$ 为相邻径向波数 bin 的步长；
* $\varepsilon>0$ 为“误差改善阈值”，用于判定累计误差是否已进入平台区；
* $m$ 为连续稳定判定的 bin 数，用于抑制谱抖动带来的偶然波动。

该定义等价于在数值上寻找 **累计误差曲线 $\mathrm{NRMSE}_{\le K}$ 的饱和点（plateau onset）**。

---

#### 8.3 物理含义与尺度解释

由定义可知：

* 当 $K < k^*$ 时，引入更高波数（更小尺度）分量，模型整体重建精度仍能显著提升；
* 当 $K \ge k^*$ 时，更高频的尺度分量即使被纳入评估，其参与与否对整体预测精度已无实质性影响。

因此，$k^*$ 在物理上对应于：

> **模型能够有效恢复的最小空间尺度的倒数。**

等价地，可定义对应的物理尺度：
$$
\ell^* = \frac{1}{k^*},
\qquad
s^* = \frac{\ell^*}{D},
$$
其中 $D$ 为特征长度（如圆柱直径）。尺度 $\ell^*$ 及 $s^*$ 即为模型在当前观测条件下的**有效分辨率极限**。

---

#### 8.4 与其他尺度指标的关系（说明性）

需要指出的是：

* 单一波数处的尺度误差 $\mathrm{NRMSE}_{\mathrm{F}}(k)$（§5.2）描述的是**局部尺度误差**；
* 本节定义的 $k^*$ 则描述的是**累计尺度贡献是否仍具信息增益**。

两者关注的问题不同，但在尺度分析中是互补的。
在实际分析与汇报中，常同时报告：

* $\mathrm{NRMSE}_{\mathrm{F}}(k)$：误差随尺度的分布；
* $k^*$（或 $\ell^*$）：模型可恢复尺度的整体截止位置。

## 九、基于尺度相关性的可恢复尺度判定（v2.0）

### 9.1 频域尺度相关系数（Spectral Correlation）

对每个时间帧 $t$，在径向波数 bin $\Omega_m$ 上，定义预测场与真实场的尺度相关系数：

$$
\rho_t(k_m)
=

\frac{
\sum_{(p,q)\in\Omega_m}
\Re\big(
\widehat{\hat x}*t(p,q),\overline{\hat x_t(p,q)}
\big)
}{
\sqrt{
\sum*{(p,q)\in\Omega_m}|\widehat{\hat x}*t(p,q)|^2
;
\sum*{(p,q)\in\Omega_m}|\hat x_t(p,q)|^2
}
}.
$$

对时间维取平均，得到统计意义上的尺度相关曲线：

$$
\rho(k_m)
=

\frac{1}{T}
\sum_{t=1}^T
\rho_t(k_m).
$$

该量满足：

* $\rho(k)\approx 1$：预测与真实场在该尺度上高度一致；
* $\rho(k)\approx 0$：预测与真实场在该尺度上不相关（噪声主导）；
* $\rho(k)<0$：预测与真实场出现系统性反相。

---

### 9.2 可恢复尺度截止的定义（Adaptive Spectral Cutoff, v2.0）

基于尺度相关系数，引入自适应尺度截止波数 $k^*$：

$$
k^*
=

\min\left{
k_m \ge k_{\min}
;\middle|;
|\rho(k_m)| < \delta
\quad \text{并在后续 } m \text{ 个 bins 内保持}
\right},
$$

其中：

* $k_{\min}$ 为物理上有意义的最小波数（通常取 $k_{\min}=1$）；
* $\delta$ 为相关性阈值，对应“与噪声不可区分”的统计界限；
* $m$ 为连续稳定判定的 bin 数，用于抑制谱抖动。

对应的最小可恢复物理尺度为：
$$
\ell^*=\frac{1}{k^*},
\qquad
s^*=\frac{\ell^*}{D}.
$$

该定义直接回答：

> **模型在何种空间尺度以下，其预测结果不再携带与真实物理场相关的结构信息。**

---

### 9.3 与误差型指标的关系（定位说明）

需要强调的是：

* 局部尺度误差 $\mathrm{NRMSE}_{\mathrm{F}}(k)$ 描述的是误差幅值；
* 尺度相关系数 $\rho(k)$ 描述的是结构一致性；
* 两者并不等价，且在高频区间可能给出完全不同的判断。

在本项目中：

* $\rho(k)$ 作为 **可恢复尺度判定的主指标**；
* $\mathrm{NRMSE}*{\mathrm{F}}(k)$ 与累计 $\mathrm{NRMSE}*{\le K}$ 仅作为**辅助诊断量**。
