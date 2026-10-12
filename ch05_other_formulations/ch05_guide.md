# 第 5 章 预测控制的其他形式 (Other Formulations of Predictive Control)
——全宇宙最详细、零基础草履虫级逐节深度推导与工程实现指南

> **本书权威参考**：J. M. Maciejowski, *Predictive Control with Constraints*, Prentice Hall, 2002, Chapter 5, pp. 145–166.  
> **面向受众**：零现代控制理论、零最优控制、零观测器基础，仅掌握经典单输入单输出（SISO）传递函数 $G(s)$、复频域极点/零点与 PID 控制的读者。本指南将一切矩阵向量微分方程降维拆解至标量代数、差分方程与几何直观。

---

## 1. 章节元信息

- **章号**：第 5 章 (Chapter 5)
- **标题**：预测控制的其他形式 (*Other Formulations of Predictive Control*)
- **核心定位**：打破前四章单一的“无扰动状态空间/传递函数标准二次规划（QP）”框架，系统讲解工业界与学术界为了解决**前馈前瞻扰动抑制**、**开环不稳定模型数值发散**、**非二次型成本与几何顶点极值优化（LP）**、**区域/漏斗软约束**、**基函数降维控制（PFC）**以及**连续时间泰勒展开预测**而发展出的各种前沿变体体系。
- **全书承前启后地位**：
  - **承前**：继承第 2、3 章的状态空间预测方程 $\mathcal{Z} = \Psi x + \Upsilon u + \Theta \Delta\mathcal{U}$，第 4 章的阶跃响应卷积与 Diophantine 多项式预测；
  - **启后**：第 5.2 节的“镇定预测（Stabilized Predictions）”直接引出第 6 章无约束/有约束闭环名义稳定性的充要条件与终端权重设计；第 5.4 节的 LP 几何顶点跳跃特性与软约束松弛技术直接为第 8 章的鲁棒预测控制（Min-Max / 混杂范数优化）提供数学基础。

---

## 2. 符号、假设与面向经典控制读者的预备知识

对于只学过经典传递函数 $G(s) = \frac{Y(s)}{U(s)}$ 的读者，现代控制中大量堆叠的矩阵往往令人望而生畏。本节首先建立最直观的物理与数学对应关系。

### 2.1 经典频域与离散状态空间符号对照表

| 经典控制符号 / 概念 | 物理意义 | 离散状态空间 / MPC 对应符号 | 维度与数学含义 |
| :--- | :--- | :--- | :--- |
| $r(t)$ | 设定点 (Set-point / Reference) | $\mathcal{T}(k) = [r(k+1), \dots, r(k+H_p)]^T$ | $p H_p \times 1$ 维未来参考轨迹向量 |
| $u(t)$ | 执行器控制输入 (Actuator input) | $u(k), \Delta u(k) = u(k) - u(k-1)$ | $m \times 1$ 维绝对输入与输入增量 |
| $y(t)$ | 被控测量输出 (Measured output) | $y(k) = C_y x(k)$ | $p \times 1$ 维传感器测量输出 |
| $d(t)$ | 外部可测扰动 (Measured disturbance) | $d_m(k) \in \mathbb{R}^{m_d}$ | 提前测量到的负荷扰动（如进水温度） |
| $G(s) = \frac{K}{\tau s + 1}$ | 一阶惯性传递函数 | $x(k+1) = a x(k) + b u(k)$ | $a = e^{-T_s/\tau}, b = K(1 - a)$ 标量差分 |
| 极点 $p_i$ 位于右半复平面 $\text{Re}(p_i) > 0$ | 系统不稳定，阶跃响应发散 | 特征值 $|\lambda_i(A)| > 1$ 位于单位圆外 | 离散状态转移矩阵 $A^k$ 随步数 $k \to \infty$ 发生数值爆炸 |
| 串联滞后校正器 / 前馈环节 | 消除扰动通道静态偏差 | 前馈增益矩阵 $\Xi \in \mathbb{R}^{p H_p \times m_d H_p}$ | 预测方程中显式注入扰动未来影响 |
| 积分器 $\frac{1}{s}$ (PI 控制) | 消除阶跃扰动静差 | 增量模型 $\Delta u(k)$ 与 DMC 扰动估计 $\hat{d}(k)$ | 内模原理（Internal Model Principle）的离散体现 |

### 2.2 核心模型假设与分类

1. **可测扰动假设 (Measured Disturbance Assumption)**：
   - 扰动向量 $d_m(k)$ 在当拍 $k$ 可以被传感器即时测得（例如气象多普勒雷达探测到的飞机前方风切变、工业分馏塔上游进料组分与流量传感器）。
   - 假设扰动对输出的影响存在物理滞后，当拍 $d_m(k)$ 不直接瞬时跳跃到 $y(k)$（即 $D_{d} = 0$）。
2. **镇定预测假设 (Pre-stabilization Assumption)**：
   - 对于开环不稳定对象（至少有一个极点位于单位圆外 $|\lambda(A)| \ge 1$），假定存在名义状态反馈增益 $K \in \mathbb{R}^{m \times n}$，使得闭环状态矩阵 $A_{cl} = A - BK$ 的所有特征值严格位于单位圆内：
     $$
     \rho(A - BK) = \max_i |\lambda_i(A - BK)| < 1
     $$
3. **互素分解假设 (Coprime Factorization Assumption)**：
   - 不稳定传递函数矩阵 $P(z)$ 可表示为两个稳定、因果且真（Proper）的传递函数之商：$P(z) = M(z)^{-1} N(z)$，且 $M(z)$ 与 $N(z)$ 在闭单位圆外无公共零点。

---

## 3. 逐节精读与“草履虫级”公式推导 (Sections 5.1 – 5.7)

---

### 3.1 第 5.1 节：可测扰动与前馈控制 (Measured Disturbances and Feedforward)

#### 3.1.1 原文工程问题与控制结构
经典控制理论中，纯反馈控制（Feedback Control）本质上是“事后补偿”：必须等到扰动侵入系统、引起输出偏差 $e(t) = r(t) - y(t) \ne 0$ 之后，控制器才能感应到偏差并施加纠正动作。这种滞后在面对大容量滞后系统（如锅炉蒸汽温度、原油分馏塔）或剧烈外扰时，会导致被控量剧烈振荡。

若扰动在侵入控制通道前即可被传感器测得（如进料流量计测得负荷波动），控制系统可在输出发生偏差之前提前做出反向调节，实现**前馈控制（Feedforward Control）**。

原书 **Figure 5.1** 给出了可测扰动的前馈结构框图：
- 控制对象包含两个通道：控制输入通道 $P_1(z)$ 与扰动通道 $P_d(z)$，二者通常汇合通过某些共同的后置动态 $P_2(z)$。
- 扰动传递函数为 $P_2(z) P_d(z)$；控制传递函数为 $P_2(z) P_1(z)$。
- 若要完全消除扰动影响，理想的前馈环节需满足：
  $$
  P_2(z) P_1(z) U_{ff}(z) + P_2(z) P_d(z) D_m(z) = 0 \implies U_{ff}(z) = - P_1^{-1}(z) P_d(z) D_m(z)
  $$
- **物理实现困境**：$P_1(z)$ 通常包含不可逆的时滞（延迟环节 $z^{-d}$）、非最小相位零点（单位圆外零点求逆将变成不稳定极点），或非方阵系统求逆不存在。因此，工业上无法实现理想的纯静态求逆前馈，必须将**前馈预测**无缝融合进**滚动时域反馈优化**中。

#### 3.1.2 状态空间增广模型推导 (Equations 5.1 – 5.6)
设线性离散状态空间模型引入可测扰动向量 $d_m(k) \in \mathbb{R}^{m_d}$：
$$
x(k+1) = A x(k) + B u(k) + B_d d_m(k) \tag{5.1}
$$
$$
y(k) = C_y x(k) \tag{5.2}
$$
$$
z(k) = C_z x(k) \tag{5.3}
$$

若原书 Figure 5.1 中的子模块 $P_1(z), P_2(z), P_d(z)$ 具有独立的状态空间实现：
- $P_1(z) \sim (A_1, B_1, C_1, D_1)$，状态向量 $x_1(k)$；
- $P_2(z) \sim (A_2, B_2, C_2, D_2)$，状态向量 $x_2(k)$；
- $P_d(z) \sim (A_d, B_d, C_d, 0)$，状态向量 $x_d(k)$。

假设 $D_2 D_1 = 0$（无输入到输出的瞬态直通），串并联增广状态为 $x(k) = [x_1(k)^T, x_2(k)^T, x_d(k)^T]^T$。其复合状态矩阵为原书式 (5.4)–(5.5)：
$$
\begin{bmatrix} x_1(k+1) \\ x_2(k+1) \\ x_d(k+1) \end{bmatrix} = 
\begin{bmatrix}
A_1 & 0 & 0 \\
B_2 C_1 & A_2 & B_2 C_d \\
0 & 0 & A_d
\end{bmatrix}
\begin{bmatrix} x_1(k) \\ x_2(k) \\ x_d(k) \end{bmatrix} +
\begin{bmatrix} B_1 \\ B_2 D_1 \\ 0 \end{bmatrix} u(k) +
\begin{bmatrix} 0 \\ 0 \\ B_d \end{bmatrix} d_m(k) \tag{5.4}
$$
$$
z(k) = \begin{bmatrix} D_2 C_1 & C_2 & D_2 C_d \end{bmatrix} \begin{bmatrix} x_1(k) \\ x_2(k) \\ x_d(k) \end{bmatrix} \tag{5.5}
$$

当拍状态估计修正为原书式 (5.6)：
$$
\hat{x}(k+1|k) = A \hat{x}(k|k) + B u(k) + B_d d_m(k) \tag{5.6}
$$

#### 3.1.3 全未来时域前馈预测矩阵 $\Xi$ 极度展开推导 (Equations 5.7 – 5.9)
我们从最原始的一步预测开始，逐拍递推输出预测，坚决不跳步：

**第 1 步预测 ($i = 1$)**：
$$
\hat{x}(k+1|k) = A x(k) + B u(k) + B_d d_m(k)
$$
因为 $u(k) = u(k-1) + \Delta u(k)$，代入整理得：
$$
\hat{x}(k+1|k) = A x(k) + B u(k-1) + B \Delta u(k) + B_d d_m(k)
$$
输出预测为：
$$
\hat{z}(k+1|k) = C_z \hat{x}(k+1|k) = C_z A x(k) + C_z B u(k-1) + C_z B \Delta u(k) + C_z B_d d_m(k)
$$

**第 2 步预测 ($i = 2$)**：
$$
\hat{x}(k+2|k) = A \hat{x}(k+1|k) + B u(k+1) + B_d d_m(k+1|k)
$$
将 $u(k+1) = u(k-1) + \Delta u(k) + \Delta u(k+1)$ 及 $\hat{x}(k+1|k)$ 表达式代入：
$$
\begin{aligned}
\hat{x}(k+2|k) &= A \left[ A x(k) + B u(k-1) + B \Delta u(k) + B_d d_m(k) \right] + B [u(k-1) + \Delta u(k) + \Delta u(k+1)] + B_d d_m(k+1|k) \\
&= A^2 x(k) + (A B + B) u(k-1) + (A B + B) \Delta u(k) + B \Delta u(k+1) + A B_d d_m(k) + B_d d_m(k+1|k)
\end{aligned}
$$
输出预测为：
$$
\hat{z}(k+2|k) = C_z A^2 x(k) + C_z (A+I)B u(k-1) + C_z (A+I)B \Delta u(k) + C_z B \Delta u(k+1) + C_z A B_d d_m(k) + C_z B_d d_m(k+1|k)
$$

**归纳至第 $i$ 步预测 ($i = 1, \dots, H_p$)**：
扰动项对 $\hat{z}(k+i|k)$ 的贡献为历史扰动经过系统动态累积响应之和：
$$
\text{Disturbance Term}(i) = \sum_{j=1}^i C_z A^{i-j} B_d d_m(k+j-1|k)
$$

将未来 $H_p$ 步预测向量写成大矩阵紧凑形式（原书式 5.7）：
$$
\mathcal{Z}(k) = \Psi x(k|k) + \Upsilon u(k-1) + \Theta \Delta\mathcal{U}(k) + \Xi \mathcal{D}_m(k) \tag{5.7}
$$
其中未来可测扰动堆叠向量 $\mathcal{D}_m(k) \in \mathbb{R}^{m_d H_p \times 1}$ 定义为原书式 (5.8)：
$$
\mathcal{D}_m(k) = \begin{bmatrix} d_m(k) \\ d_m(k+1|k) \\ \vdots \\ d_m(k+H_p-1|k) \end{bmatrix} \tag{5.8}
$$
前馈扰动影响矩阵 $\Xi \in \mathbb{R}^{p H_p \times m_d H_p}$ 具有极其优美的**下三角 Toeplitz 块带状结构**（原书式 5.9）：
$$
\Xi = \begin{bmatrix}
C_z B_d & 0 & 0 & \dots & 0 \\
C_z A B_d & C_z B_d & 0 & \dots & 0 \\
C_z A^2 B_d & C_z A B_d & C_z B_d & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
C_z A^{H_p-1} B_d & C_z A^{H_p-2} B_d & C_z A^{H_p-3} B_d & \dots & C_z B_d
\end{bmatrix} \tag{5.9}
$$

#### 3.1.4 跟踪误差向量修正与标准 QP 转化 (Equation 5.10)
将扰动预测前馈项直接并入无控自由响应，修正跟踪误差向量 $\mathcal{E}(k)$ 为原书式 (5.10)：
$$
\mathcal{E}(k) = \mathcal{T}(k) - \Psi \hat{x}(k|k) - \Upsilon u(k-1) - \Xi \mathcal{D}_m(k) \tag{5.10}
$$
目标函数依然保持标准二次型：
$$
V(k) = \frac{1}{2} \Delta\mathcal{U}(k)^T H \Delta\mathcal{U}(k) - g(k)^T \Delta\mathcal{U}(k) + \text{const}
$$
其中 Hessian 矩阵 $H = \Theta^T \mathcal{Q} \Theta + \mathcal{R}$ 完全不受扰动影响（离线计算），而一次项线性梯度向量则注入了扰动前馈信息：
$$
g(k) = \Theta^T \mathcal{Q} \mathcal{E}(k)
$$
求解完全沿用第 3 章的内点法或有效集法，计算复杂度分毫不增！

> [!TIP]
> **工业实践假设**：如果未来扰动无法预知，最通用的工业假设是“扰动保持当前值不变”（Persistence Assumption），即 $d_m(k+i|k) = d_m(k)$，这也是 MATLAB MPC Toolbox 的默认实现。

---

### 3.2 第 5.2 节：镇定预测 (Stabilized Predictions)

#### 3.2.1 开环预测在不稳定系统中的数值崩溃灾难
对于开环不稳定系统（如磁悬浮、倒立摆、直升机纵向姿态），开环矩阵 $A$ 存在模长大于 1 的特征值 $|\lambda_u| > 1$。
在标准 MPC 中，自由响应矩阵 $\Psi$ 包含 $A^i$，强迫响应矩阵 $\Theta$ 包含 $\sum A^k B$。当预测时域 $H_p$ 稍大（例如 $H_p = 25$）：
$$
\lambda_u = 1.25 \implies \lambda_u^{25} \approx 264.7, \quad \lambda_u^{35} \approx 2525.1
$$
矩阵元素呈指数级几何级数爆炸，导致：
1. **条件数爆炸**：$\kappa(\Theta) = \frac{\sigma_{\max}(\Theta)}{\sigma_{\min}(\Theta)} > 10^6 \sim 10^8$，Hessian 矩阵 $H = \Theta^T \Theta$ 严重病态，数值求逆产生致命截断误差；
2. **可行域虚假收缩**：开环发散使得约束检验在未来步极易超出上下限，导致求解器报错“无可行解（Infeasible）”。

#### 3.2.2 预镇定基线策略重构 (Equations 5.11 – 5.14)
为了解决这一致命问题，Kouvaritakis 与 Rossiter [KR92, RK93] 提出了**镇定预测（Stabilized Predictions）**思想：
不以“控制增量为零（Do-Nothing）”作为开环基线，而是预先设计一个线性状态反馈增益矩阵 $K \in \mathbb{R}^{m \times n}$，使得闭环矩阵
$$
A_{cl} = A - BK
$$
的所有特征值严格位于单位圆内：$\rho(A_{cl}) < 1$。

将未来的基线控制律定义为：
$$
u(k+i|k) = -K \hat{x}(k+i|k) + q(i) \tag{5.11}
$$
其中 $q(i) \in \mathbb{R}^m$ 是优化器新寻找的决策变量（在镇定反馈基础上的摄动量）。

#### 3.2.3 状态与输出预测方程极度推导 (Equations 5.12 – 5.18)
我们逐步推导当 $q(i)$ 作用时未来状态的发展：

**第 1 步预测 ($i = 1$)**：
$$
\begin{aligned}
\hat{x}(k+1|k) &= A \hat{x}(k|k) + B u(k|k) \\
&= A \hat{x}(k|k) + B [-K \hat{x}(k|k) + q(0)] \\
&= (A - BK) \hat{x}(k|k) + B q(0) \\
&= A_{cl} \hat{x}(k|k) + B q(0) \tag{5.12}
\end{aligned}
$$

**第 2 步预测 ($i = 2$)**：
$$
\begin{aligned}
\hat{x}(k+2|k) &= A \hat{x}(k+1|k) + B [-K \hat{x}(k+1|k) + q(1)] \\
&= A_{cl} \hat{x}(k+1|k) + B q(1) \\
&= A_{cl} [A_{cl} \hat{x}(k|k) + B q(0)] + B q(1) \\
&= A_{cl}^2 \hat{x}(k|k) + A_{cl} B q(0) + B q(1) \tag{5.13}
\end{aligned}
$$

**第 $H_p$ 步预测**：
$$
\hat{x}(k+H_p|k) = A_{cl}^{H_p} \hat{x}(k|k) + \sum_{j=0}^{H_p-1} A_{cl}^{H_p-1-j} B q(j) \tag{5.14}
$$
**数学美感展现**：因为 $\rho(A_{cl}) < 1$，矩阵幂次 $A_{cl}^i \to 0$（当 $i \to \infty$ 时按指数衰减）！无论预测时域 $H_p$ 选得多么大，预测矩阵的各项系数永远有界衰减，**彻底根除了条件数随时域爆炸的数值顽疾**！

#### 3.2.4 决策变量代换与约束映射保持凸性 (Equations 5.19 – 5.21)
设控制时域为 $H_u$，在时域外令 $q(i) = 0$ ($i \ge H_u$)，则未来控制输入自然回归为纯渐近镇定状态反馈 $u = -K x$。
决策变量向量变为：
$$
\mathcal{Q}_{dec}(k) = \begin{bmatrix} q(0) \\ q(1) \\ \vdots \\ q(H_u-1) \end{bmatrix} \in \mathbb{R}^{m H_u}
$$
由于 $u(k+i|k) = -K \hat{x}(k+i|k) + q(i)$，而 $\hat{x}(k+i|k)$ 与 $\mathcal{Q}_{dec}(k)$ 呈严格线性关系，因此：
$$
u(k+i|k) = M_i \hat{x}(k|k) + N_i \mathcal{Q}_{dec}(k)
$$
原始对控制输入幅值 $u_{\min} \le u(k+i) \le u_{\max}$ 和增量 $\Delta u$ 的线性不等式约束，代换后依然是关于 $\mathcal{Q}_{dec}(k)$ 的**纯线性不等式**。优化问题依然是一个严格凸的二次规划（QP）问题！

---

### 3.3 第 5.3 节：不稳定模型的分解与互素因子化 (Decomposition of Unstable Model)

#### 3.3.1 Richalet PFC 不稳定对象内部模型的困境
Richalet 的预测函数控制（PFC）坚持使用**独立模型（Independent Model）**——即内部模型的输出不通过传感器测量的实际输出进行闭环滤波校正。
如果被控对象 $P(z)$ 本身是不稳定的，而内部模型也是不稳定的，控制器为了使闭环稳定，必然会在内部“反向抵消（Pole-Zero Cancellation）”不稳定开环极点。然而，根据现代控制理论，开环不稳定极点的抵消必然导致系统**内部不稳定（Loss of Internal Stability）**（扰动通道到状态不稳定）。

#### 3.3.2 稳定反馈互联分解 (Equations 5.22 – 5.23 & Figures 5.2 – 5.3)
为了在 PFC 框架下处理不稳定对象，Richalet 提出了将不稳定系统 $P(z)$ 分解为两个**稳定子系统** $P_1(z)$ 与 $P_2(z)$ 构成正反馈闭环的结构（原书 **Figure 5.2**）：
$$
P(z) = P_1(z) [I - P_2(z) P_1(z)]^{-1} \tag{5.22}
$$
在 SISO 标量形式下，式 (5.22) 退化为极为清晰的初等代数关系（原书式 5.23）：
$$
P(z) = \frac{P_1(z)}{1 - P_2(z) P_1(z)} \tag{5.23}
$$
原书 **Figure 5.3** 展现了 Richalet 极其绝妙的解耦策略：
- 控制器内部仅运行**稳定的内部模型** $P_1(z)$；
- 将反馈支路 $P_2(z)$ 的信号输出，巧妙地等效视为**外部可测扰动**！
- 从而将极其棘手的不稳定对象控制问题，完全降维转化为第 5.1 节的可测扰动前馈补偿问题！

#### 3.3.3 稳定互素分解 (Coprime Factorization, Equation 5.24 & Figure 5.4)
更一般的系统论方法是**稳定有理互素分解（Coprime Factorization over $\mathcal{RH}_\infty$）**（原书式 5.24）：
$$
P(z) = M(z)^{-1} N(z) \tag{5.24}
$$
其中：
1. $N(z)$ 与 $M(z)$ 均为**渐近稳定**且**真（Proper）**的传递函数矩阵（所有极点严格落在单位圆内）；
2. $N(z)$ 与 $M(z)$ 左互素，即存在稳定多项式矩阵 $X(z), Y(z) \in \mathcal{RH}_\infty$ 满足 Bezout 恒等式：
   $$
   N(z) X(z) + M(z) Y(z) = I
   $$
这意味着 $N(z)$ 与 $M(z)$ 在闭单位圆外无任何公共零点。

原书 **Figure 5.4 (左)** 给出了对应的等价框图：
输入 $u$ 驱动稳定前向通道 $N(z)$ 得到 $N(z)u$；输出 $y$ 通过正反馈闭环 $I - M(z)$ 注入，因为：
$$
y = N u + (I - M) y \implies M y = N u \implies y = M^{-1} N u = P u
$$
由于 $M(z)$ 稳定，$I - M(z)$ 必然严格稳定！原书 **Figure 5.4 (右)** 进一步指出，在工程实现时，将难以预知的真实内环输出用最新测量值 $\hat{y}$ 代替，即完成了不稳定对象的稳定化逼近。

---

### 3.4 第 5.4 节：非二次型成本与线性规划 (Non-quadratic Costs: LP vs QP)

#### 3.4.1 LP 与 QP 的几何本质差异深度剖析 (Figures 5.5 & 5.6)
标准 MPC 采用误差平方和（2-范数），对应二次规划（QP）；但在 1980 年代，工业界广泛采用误差绝对值之和（1-范数）或峰值误差（$\infty$-范数），对应**线性规划（Linear Programming, LP）**。

原书 **Figure 5.5** 与 **Figure 5.6** 揭示了 LP 与 QP 在几何与控制机理上的根本差别：

```
      LP: 等成本线为平行超平面 (线段)             QP: 等成本线为同心同轴超椭球
             x2 ^                                        x2 ^
                |                                           |
                |   / (Cost contours)                       |     (Ellipses)
                |  /  /  /                                  |    .---.
     Constraint | /  /  /                                   |   /     \
         =======>*====================           Constraint |  |   *   | (Center outside)
                | \                                =========>===`---'====
                |  \ Feasible Region                        |  Feasible Region
                +-------------------> x1                    +-------------------> x1
            最优解必定卡在顶点 (Vertex)                 最优解可在边上、内部或平滑移动
```

1. **等成本线几何形态**：
   - **LP**：目标函数是决策变量的线性组合 $c^T \theta$，等成本线是相互平行的直线/超平面。
   - **QP**：目标函数是正定二次型 $\frac{1}{2} \theta^T H \theta$，等成本线是以无约束最优解为中心的同心同轴超椭球。
2. **最优解位置特性**：
   - **LP 最优解必定位于多面体的顶点（Vertex / 角点）**，极少情况在整条边界重合；
   - **QP 最优解可以在可行域内部（所有约束不激活）、平滑贴在某一条边界约束上，极少位于多约束交叉的尖锐角点**。
3. **参数敏感性与跳跃性（Robustness vs Sensitivity）**：
   - **LP**：当成本权重向量 $c$ 的倾斜角发生微小变化时，最优解完全锁死在同一个几何顶点不动（表现为高鲁棒性）；但当倾角旋转跨过临界阈值时，**最优解会瞬间“突变跳跃”到多面体的另一个遥远角点**！这在控制上对应执行机构的剧烈冲击抖动（Bang-Bang 效应）。
   - **QP**：调节权重矩阵 $Q, R$ 会使椭球长短轴比例平滑旋转，最优解沿约束边界**连续平滑移动**（原书 Figure 5.6），赋予了现场运行工程师无与伦比的平滑在线微调能力。

#### 3.4.2 绝对值 1-范数问题的标准 LP 转化 (Equations 5.25 – 5.32)
1-范数性能指标定义为预测误差绝对值之和（原书式 5.25）：
$$
V(k) = \sum_{i=1}^{H_p} \sum_{j=1}^p q_{ij} |z_j(k+i|k) - r_j(k+i)| \tag{5.25}
$$
其紧凑形式为：
$$
V(k) = \| \mathcal{Q} (\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)) \|_1 \tag{5.32}
$$

**数学转化定理**：最小化绝对值项 $\min_{\theta} |c^T \theta - d|$ 等价于标准线性规划：
$$
\min_{\theta, y} y \quad \text{s.t.} \quad \begin{cases} y \ge c^T \theta - d \\ y \ge -(c^T \theta - d) \\ y \ge 0 \end{cases} \tag{5.28}
$$
**推导证明**：
令松弛变量 $y \ge |a|$，由绝对值定义 $|a| = \max(a, -a)$，因此 $y \ge \max(a, -a) \iff y \ge a \text{ 且 } y \ge -a$。当求解器向下压低目标值 $y$ 时，$y$ 必然紧紧卡在 $|a|$ 的边界上，从而 $y^* = |a|$。

将这一思想推广到向量残差 $\| C \theta - d \|_1 = \sum_{j=1}^{p H_p} |c_j^T \theta - d_j|$（原书式 5.30–5.31）：
引入非负松弛变量向量 $y \in \mathbb{R}^{p H_p}$，构建标准 LP：
$$
\min_{\theta, y} \mathbf{1}^T y \quad \text{s.t.} \quad 
\begin{bmatrix}
I & -C \\
I & C \\
0 & A
\end{bmatrix}
\begin{bmatrix} y \\ \theta \end{bmatrix} \ge
\begin{bmatrix} -d \\ d \\ b \end{bmatrix}, \quad y \ge 0 \tag{5.31}
$$

#### 3.4.3 Min-Max 极大极小 $\infty$-范数问题的标准 LP 转化 (Equations 5.33 – 5.36)
Campo 与 Morari [CM86] 提出最小化时域内的**最大峰值偏差（Worst-case Peak Deviation）**，即 $\infty$-范数目标（原书式 5.33）：
$$
V(k) = \| \mathcal{Q} (\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)) \|_\infty = \max_{1 \le i \le H_p} \max_{1 \le j \le p} q_{ij} |z_j(k+i|k) - r_j(k+i)| \tag{5.33}
$$

**数学转化推导**：
通用 $\infty$-范数最小化问题为：
$$
\min_{\theta} \| C \theta - d \|_\infty = \min_{\theta} \max_j |c_j^T \theta - d_j| \tag{5.35}
$$
引入**单一标量松弛变量** $\gamma \in \mathbb{R}$（原书式 5.36）：
目标为最小化上界 $\gamma$：
$$
\min_{\theta, \gamma} \gamma \quad \text{s.t.} \quad 
\begin{cases}
c_j^T \theta - d_j \le \gamma, & \forall j \\
-(c_j^T \theta - d_j) \le \gamma, & \forall j \\
A \theta \ge b
\end{cases}
$$
写成标准矩阵不等式（原书式 5.36）：
$$
\min_{\theta, \gamma} \begin{bmatrix} 1 \\ 0 \end{bmatrix}^T \begin{bmatrix} \gamma \\ \theta \end{bmatrix} \quad \text{s.t.} \quad
\begin{bmatrix}
\mathbf{1} & -C \\
\mathbf{1} & C \\
0 & A
\end{bmatrix}
\begin{bmatrix} \gamma \\ \theta \end{bmatrix} \ge
\begin{bmatrix} -d \\ d \\ b \end{bmatrix}, \quad \gamma \ge 0 \tag{5.36}
$$
其中 $\mathbf{1} = [1, 1, \dots, 1]^T \in \mathbb{R}^{p H_p}$。原本复杂的非线性非光滑最大值函数，被优雅地转化为仅增加 1 个标量决策变量的标准凸 LP！

---

### 3.5 第 5.5 节：区域控制、漏斗目标与重合点 (Zones, Funnels and Coincidence Points)

#### 3.5.1 区域控制 (Zone Control, Figure 5.7)
在石油炼制、化工与暖通空调等实际工业过程中，大多数被控变量并不需要死死锁定在一个固定数值上。例如原油分馏塔顶温度，只要保持在 $[165^\circ\text{C}, 175^\circ\text{C}]$ 的合格带内即可。若强制设定点为 $170^\circ\text{C}$，控制器会为了消除微小的无害漂移而频繁动作阀门，导致执行机构严重磨损并消耗大量公用工程能源。

原书 **Figure 5.7** 提出了**区域控制（Zone Control）**：
- 定义输出上下限死区区间 $[z_{\min}, z_{\max}]$；
- 只要预测输出落在区域内部，**罚项为零**，控制器不做任何非必要干预；
- 仅当预测输出逸出边界时，才通过单侧松弛变量施加严厉惩罚：
  $$
  z_j(k+i|k) \le z_{\max} + \epsilon_{\text{high}}(i), \quad z_j(k+i|k) \ge z_{\min} - \epsilon_{\text{low}}(i), \quad \epsilon \ge 0
  $$

#### 3.5.2 漏斗目标 (Funnel Control, Figures 5.8 & 5.9)
原书 **Figure 5.8** 给出了**漏斗控制（Funnel Objective）**：
区域的边界在时间轴上并非恒定矩形，而是随预测时域向未来延伸而逐步收敛变窄。
若当前输出低于目标设定点 $s_j$，漏斗的上边界为恒定设定点 $s_j$，而下边界则设计为一阶收敛轨迹（原书式 5.37）：
$$
(1 - \alpha^i) s_j + \alpha^i z_j(k) \le z_j(k+i|k) \le s_j, \quad (0 < \alpha < 1) \tag{5.37}
$$
原书 **Figure 5.9** 展示了 Honeywell 著名商用预测控制产品 RMPCT 所采用的**直线边界漏斗**。
**工程优越性**：单条刚性参考轨迹在多变量强耦合系统中极易导致输入饱和互斥（没有可行解）；而漏斗目标为多变量控制提供了“前松后紧”的柔性通道，极大提升了闭环可行性与鲁棒性。

#### 3.5.3 重合点技术 (Coincidence Points, Figure 5.10)
标准 MPC 在时域 $i = 1, 2, \dots, H_p$ 的每一个时刻都对误差进行积分或求和，计算维度随 $H_p$ 线性增长。
原书 **Figure 5.10** 提出了**重合点（Coincidence Points）**理念：
仅在未来的若干个特定关键时间截面 $t = k + P_1, k + P_2, \dots, k + P_c$ 处强制输出与参考轨迹对齐：
$$
V(k) = \sum_{j=1}^c \| r(k + P_j) - \hat{z}(k + P_j|k) \|^2
$$
极大削减了约束与代价矩阵的行数，使计算速度提升数倍至数十倍。

---

### 3.6 第 5.6 节：预测函数控制 (Predictive Functional Control, PFC)

#### 3.6.1 PFC 基函数参数化架构 (Figure 5.11 & Equation 5.38)
由 J. Richalet 创立的 PFC 是预测控制领域另一大学派。PFC 的核心哲学在于：**放弃每步离散阶梯增量 $\Delta u(k+i)$ 的自由优化，强制约束未来控制轨迹为若干预选基函数（Base Functions）的线性组合**（原书式 5.38）：
$$
u(k+i) = \sum_{j=0}^{n_B} \mu_j(k) B_j(i) \tag{5.38}
$$
原书 **Figure 5.11** 展示了最经典的多项式基函数：
$$
B_0(i) = 1 (\text{阶跃基}), \quad B_1(i) = i (\text{斜坡基}), \quad B_2(i) = i^2 (\text{抛物线基})
$$
未来控制输入完全由极少数几个待定系数向量 $\mu_0(k), \mu_1(k), \dots$ 决定：
$$
u(k+i|k) = u_0(k) + u_1(k) i + u_2(k) i^2 \tag{5.38'}
$$
**革命性优势**：
1. 优化维数由几十维离散脉冲瞬间坍缩为 2 到 3 个代数参数；
2. 天然保证了输入信号的光滑性，免除了为抑制控制动作高频抖振而费心调节权重 $\mathcal{R}$ 的烦恼；
3. 对于 SISO 无约束系统，若重合点数等于基函数个数，甚至**完全不需要任何优化求解器，直接解析求解联立线性方程组即可秒出结果**！

#### 3.6.2 阶跃响应模型下的强制响应闭式推导 (Exercise 5.4 理论基础)
设系统阶跃响应系数为 $S(i)$（即单位阶跃输入作用下第 $i$ 拍的响应输出），系统自由响应为 $y_f(k+H_p|k)$。
若未来控制输入取阶跃+斜坡基函数：
$$
u(k+i) = u_0(k) + u_1(k) i
$$
由于线性时不变系统满足离散卷积叠加原理：
$$
\hat{y}_{forced}(k+H_p) = \sum_{j=0}^{H_p-1} g_j u(k+H_p-1-j) = \sum_{i=0}^{H_p-1} S(H_p-i) \Delta u(k+i)
$$
我们计算输入增量：
- 第 0 步增量：$\Delta u(k) = u(k) - u(k-1) = u_0(k)$（假设先前静态参考为 0）；
- 以后各步增量（$i \ge 1$）：
  $$
  \Delta u(k+i) = u(k+i) - u(k+i-1) = [u_0 + u_1 i] - [u_0 + u_1(i-1)] = u_1(k)
  $$
每个后继增量恒等于常数 $u_1(k)$！
将其逐项代入阶跃响应累加和中：
$$
\begin{aligned}
\hat{y}_{forced}(k+H_p) &= S(H_p) \Delta u(k) + S(H_p-1) \Delta u(k+1) + S(H_p-2) \Delta u(k+2) + \dots + S(1) \Delta u(k+H_p-1) \\
&= S(H_p) u_0(k) + S(H_p-1) u_1(k) + S(H_p-2) u_1(k) + \dots + S(1) u_1(k) \\
&= S(H_p) u_0(k) + \left[ \sum_{j=1}^{H_p-1} S(j) \right] u_1(k)
\end{aligned}
$$
加上自由响应 $y_f(k+H_p|k)$，即严谨无误地推导出了原书 **Exercise 5.4** 的核心公式：
$$
\hat{y}(k+H_p|k) = S(H_p) u_0(k) + [S(H_p-1) + \dots + S(1)] u_1(k) + y_f(k+H_p|k)
$$

#### 3.6.3 自补偿与多项式设定点无静差跟踪 (Equations 5.40 – 5.44)
若参考输入是最高次数为 $v$ 的多项式（例如斜坡信号 $v=1$），PFC 依靠**自补偿（Auto-compensation）**建立内模原理。
定义当前实测输出与先前预测输出的偏差 $e(k|k) = y(k) - \hat{y}(k|k-1)$，通过多项式外推未来误差趋势（原书式 5.42）：
$$
\hat{e}(k+i|k) = e(k|k) + \sum_{j=1}^v e_j i^j \tag{5.42}
$$
选取 $v+1$ 个重合点 $P_1, \dots, P_{v+1}$，令模型预测强制贴合目标补偿值（原书式 5.44）：
$$
\hat{y}(k+P_j|k) = s(k+P_j) - \hat{e}(k+P_j|k), \quad (j = 1, \dots, v+1) \tag{5.44}
$$
**定理结论**：只要执行机构输入基函数的自由度不小于 $v+1$，且重合点数达到 $v+1$，则闭环系统**在存在稳态增益模型失配与持续扰动的情况下，依然能够严格实现无静差跟踪**！

#### 3.6.4 PFC 启发式剪裁限幅法则 (Clipping, Equation 5.45)
对于高频伺服系统，PFC 不采用耗时的 QP 迭代求解器，而是先解析计算无约束最优步长 $\Delta u(k)^*$，随后采用极其高效的启发式**非线性剪裁（Clipping）**（原书式 5.45）：
$$
\Delta u(k) = \begin{cases}
\min \left( \Delta u(k)^*, \; \Delta u_{\max}, \; u_{\max} - u(k-1) \right), & \text{若 } \Delta u(k)^* \ge 0 \\
\max \left( \Delta u(k)^*, \; \Delta u_{\min}, \; u_{\min} - u(k-1) \right), & \text{若 } \Delta u(k)^* < 0
\end{cases} \tag{5.45}
$$
剪裁后的控制动作同时施加给真实对象与内部模型，以防止模型发散积分饱和。原书 Exercise 5.6 证明，对于纯幅值受限系统，剪裁解与严格 QP 解高度吻合。

---

### 3.7 第 5.7 节：连续时间预测控制 (Continuous-time Predictive Control)

#### 3.7.1 Gawthrop 连续时域预测控制框架 (Equations 5.46 – 5.48)
Gawthrop 等人 [DG91, DG92] 提出了完全在连续时间域表述并求解的 MPC 框架。连续二次性能指标为（原书式 5.46）：
$$
V(t) = \int_{T_1}^{T_2} [r(t+\tau) - \hat{y}(t+\tau|t)]^2 d\tau + \lambda \int_0^{T_2} [u(t+\tau|t)]^2 d\tau \tag{5.46}
$$
预测输出通过 $n$ 阶截断**泰勒级数（Taylor Series Expansion）**构造（原书式 5.47）：
$$
\hat{y}(t+\tau|t) = y(t) + \tau \dot{y}(t) + \frac{\tau^2}{2!} \ddot{y}(t) + \dots + \frac{\tau^n}{n!} y^{(n)}(t) \tag{5.47}
$$
未来控制输入轨迹同样用 $m$ 阶泰勒多项式参数化（原书式 5.48）：
$$
u(t+\tau|t) = u(t) + \tau \dot{u}(t) + \frac{\tau^2}{2!} \ddot{u}(t) + \dots + \frac{\tau^m}{m!} u^{(m)}(t) \tag{5.48}
$$

#### 3.7.2 输出导数向量 $\mathcal{Y}(t)$ 严格微分递推展开 (Exercise 5.7 & Equation 5.50)
设连续时间状态空间模型：
$$
\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) \tag{5.49}
$$
我们对时间 $t$ 逐阶求导，展示全部求导过程：

- **0 阶导数**：
  $$
  y(t) = C x(t)
  $$
- **1 阶导数**：
  $$
  \dot{y}(t) = C \dot{x}(t) = C [A x(t) + B u(t)] = C A x(t) + C B u(t)
  $$
- **2 阶导数**：
  $$
  \ddot{y}(t) = \frac{d}{dt} [C A x(t) + C B u(t)] = C A \dot{x}(t) + C B \dot{u}(t) = C A [A x(t) + B u(t)] + C B \dot{u}(t) = C A^2 x(t) + C A B u(t) + C B \dot{u}(t)
  $$
- **3 阶导数**：
  $$
  y^{(3)}(t) = C A^2 \dot{x}(t) + C A B \dot{u}(t) + C B \ddot{u}(t) = C A^3 x(t) + C A^2 B u(t) + C A B \dot{u}(t) + C B \ddot{u}(t)
  $$
- **$n$ 阶导数一般通式**：
  $$
  y^{(n)}(t) = C A^n x(t) + \sum_{k=0}^{n-1} C A^{n-1-k} B u^{(k)}(t)
  $$

将上述全部逐阶导数堆叠为向量 $\mathcal{Y}(t) = [y(t), \dot{y}(t), \ddot{y}(t), \dots, y^{(n)}(t)]^T$ 与输入导数向量 $\mathcal{U}(t) = [u(t), \dot{u}(t), \dots, u^{(n-1)}(t)]^T$，即严密验证了原书式 (5.50)：
$$
\begin{bmatrix}
y(t) \\ \dot{y}(t) \\ \ddot{y}(t) \\ \vdots \\ y^{(n)}(t)
\end{bmatrix}
=
\begin{bmatrix}
C \\ C A \\ C A^2 \\ \vdots \\ C A^n
\end{bmatrix} x(t)
+
\begin{bmatrix}
0 & 0 & 0 & \dots & 0 \\
C B & 0 & 0 & \dots & 0 \\
C A B & C B & 0 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
C A^{n-1} B & C A^{n-2} B & \dots & C B & 0
\end{bmatrix}
\begin{bmatrix}
u(t) \\ \dot{u}(t) \\ \ddot{u}(t) \\ \vdots \\ u^{(n-1)}(t)
\end{bmatrix} \tag{5.50}
$$

#### 3.7.3 连续成本函数二次型解析表示 (Exercise 5.8 & Equations 5.51 – 5.54)
定义基函数行向量 $\Gamma(\tau) = \left[ 1, \tau, \frac{\tau^2}{2!}, \dots, \frac{\tau^n}{n!} \right]$，则式 (5.47) 紧凑写为：
$$
\hat{y}(t+\tau|t) = \Gamma(\tau) \mathcal{Y}(t) \tag{5.51}
$$
代入误差积分项中展开：
$$
\begin{aligned}
\int_{T_1}^{T_2} [r(t+\tau) - \hat{y}(t+\tau|t)]^2 d\tau &= \int_{T_1}^{T_2} [\Gamma(\tau) \mathcal{Y}(t) - r(t+\tau)]^2 d\tau \\
&= \mathcal{Y}(t)^T \left[ \int_{T_1}^{T_2} \Gamma(\tau)^T \Gamma(\tau) d\tau \right] \mathcal{Y}(t) - 2 \left[ \int_{T_1}^{T_2} r(t+\tau) \Gamma(\tau) d\tau \right] \mathcal{Y}(t) + \int_{T_1}^{T_2} r^2 d\tau \\
&= \mathcal{Y}(t)^T \Omega_y(T_1, T_2) \mathcal{Y}(t) - 2 b_r(T_1, T_2)^T \mathcal{Y}(t) + \text{const} \tag{5.52}
\end{aligned}
$$
其中基础积分矩阵 $\Omega_y \in \mathbb{R}^{(n+1) \times (n+1)}$ 的第 $(i, j)$ 个矩阵元素由初等幂函数定积分直接解析得出（原书式 5.53）：
$$
[\Omega_y]_{i, j} = \int_{T_1}^{T_2} \frac{\tau^{i-1}}{(i-1)!} \frac{\tau^{j-1}}{(j-1)!} d\tau = \frac{T_2^{i+j-1} - T_1^{i+j-1}}{(i+j-1) \cdot (i-1)! \cdot (j-1)!} \tag{5.53}
$$
输入惩罚项同理积分为 $\mathcal{U}(t)^T \Omega_u \mathcal{U}(t)$。由于 $\mathcal{Y}(t) = \mathcal{C}_A x(t) + \mathcal{H}_B \mathcal{U}(t)$ 与 $\mathcal{U}(t)$ 严格线性，整个连续时间成本 $V(t)$ 被解析写成了关于决策向量 $\mathcal{U}(t)$ 的纯二次型矩阵函数！

---

## 4. 本章综合推导链 (The Complete Logical Derivation Chain)

```
[实际工程挑战]
  ├── 存在提前感知的外扰 -> (5.1 节) 构造扰动前馈矩阵 Xi -> 注入跟踪误差 E(k) -> 消除外扰静差
  ├── 开环对象不稳定 -> (5.2 节) 引入预镇定基线 u = -Kx + q -> 状态矩阵收敛 A_cl^i -> 0 -> 消除数值爆炸
  ├── PFC 独立内部模型失稳 -> (5.3 节) 互素分解 P = M^-1 N -> 闭环转化为稳定正反馈与前馈等效
  ├── 追求快速计算或特定角点响应 -> (5.4 节) 1-范数/inf-范数 -> 引入松弛变量 -> 转化为标准线性规划 LP
  ├── 容忍死区或柔性轨迹 -> (5.5 节) 区域/漏斗软约束 -> 减少执行器磨损 -> 极大增强动态可行域
  ├── 追求光滑动作/超高带宽伺服 -> (5.6 节) 基函数参数化 u = sum mu_j B_j -> 重合点联立方程秒解
  └── 纯连续时域解析控制 -> (5.7 节) 连续泰勒级数展开 -> 状态空间微分矩阵 Y = C_A x + H_B U -> 连续离线求逆
```

---

## 5. Python 实现体系与代码说明

本章配套代码已严格按照规范编写至单独的 `.py` 文件中，并生成了对应的 publication-grade 高清图表：

### 5.1 脚本 1：`ch5_feedforward_disturbance.py`
- **对应小节**：5.1 节、Exercise 5.1 & Exercise 3.3（恒温游泳池可测昼夜温度扰动前馈补偿）。
- **核心功能**：
  - 构建扰动前馈矩阵 $\Xi \in \mathbb{R}^{H_p \times H_p}$；
  - 结合 DMC 输出扰动自校正估计；
  - 精确复现：无约束完美模型下前馈消除扰动残差至 $0.0001^\circ\text{C}$；模型失配（$T=1.25, k_q=0.3$）下残差由纯反馈的 $0.42^\circ\text{C}$ 骤降至 $0.21^\circ\text{C}$（完美对应原书约 $0.2^\circ\text{C}$ 指标）；功率饱和区间（$0 \le q \le 40$ kW）两者轨迹重合。
- **输出图表**：[`ch5_feedforward_swimming_pool.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_feedforward_swimming_pool.png)。

### 5.2 脚本 2：`ch5_stabilized_predictions.py`
- **对应小节**：5.2 节（不稳定对象镇定预测）。
- **核心功能**：
  - 构造离散不稳定系统（$|\lambda_{\max}| = 1.221 > 1$）；
  - 对比开环矩阵 $\Theta_{ol}$ 与预镇定矩阵 $\Theta_{stab}$ 随预测时域 $H_p \in [5, 33]$ 的条件数 $\kappa$；
  - 验证开环条件数暴增 720 倍（从 278 飙升至 69200），而镇定预测条件数在 $H_p > 15$ 后严格收敛恒定在 96.0；
  - 闭环仿真验证 $u = -K x + q$ 在幅值硬约束下的渐近跟踪。
- **输出图表**：[`ch5_stabilized_predictions.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_stabilized_predictions.png)。

### 5.3 脚本 3：`ch5_lp_vs_qp_mpc.py`
- **对应小节**：5.4 节、Figure 5.5、Figure 5.6、Exercise 5.2。
- **核心功能**：
  - 严格几何复现 Figure 5.5 与 5.6：绘制 2D 多面体、LP 平行超平面等值线（最优解锁死在多面体顶点 $(0.5, 3.75)$）与 QP 超椭球等值线（沿边界连续平滑滑动）；
  - 实现基于 SciPy `linprog`（HiGHS 求解器）的 1-范数与 $\infty$-范数 MPC；
  - 闭环对比 1-范数（诱导控制动作稀疏死区，小偏差下 $\Delta u = 0$）、$\infty$-范数（最小化最恶劣峰值误差）与 2-范数（平滑能量最优）。
- **输出图表**：[`ch5_fig5_5_fig5_6_geometry.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_fig5_5_fig5_6_geometry.png) 与 [`ch5_lp_vs_qp_mpc.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_lp_vs_qp_mpc.png)。

### 5.4 脚本 4：`ch5_zones_and_funnels.py`
- **对应小节**：5.5 节、Figure 5.7 – 5.10。
- **核心功能**：
  - 工业二阶过阻尼过程区域控制（Zone Control）仿真：在死区 $[18, 22]$ 内部阀门完全静止，剧烈外扰越界时快速纠偏；
  - 动态收缩指数漏斗（Funnel Control）仿真：上边界与下指数边界形成包络通道；
  - 重合点可视化。
- **输出图表**：[`ch5_zones_and_funnels.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_zones_and_funnels.png)。

### 5.5 脚本 5：`ch5_pfc_base_functions.py`
- **对应小节**：5.6 节、Figure 5.11、Exercise 5.4、5.5、5.6。
- **核心功能**：
  - 数值验证 Exercise 5.4 阶跃响应阶跃+斜坡基函数公式（与直接差分仿真误差 $4.44 \times 10^{-16}$）；
  - 实现基于 2 个重合点的 PFC 斜坡轨迹无静差跟踪；
  - 验证 Exercise 5.6：纯幅值约束下剪裁法与 QP 解完全一致；引入变动率约束后剪裁法产生轻微次优偏差但极度贴近。
- **输出图表**：[`ch5_pfc_simulation.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_pfc_simulation.png)。

### 5.6 脚本 6：`ch5_continuous_time_mpc.py`
- **对应小节**：5.7 节、Exercise 5.7、5.8。
- **核心功能**：
  - 构造连续时间弹簧-质量-阻尼系统；
  - 验证导数矩阵方程 (5.50)（$\mathcal{H}_B$ 严格下三角 Toeplitz 结构）；
  - 计算时间积分矩阵 $\Omega_y, \Omega_u$，结合四阶 Runge-Kutta（RK4）连续积分器完成滚动时域仿真。
- **输出图表**：[`ch5_continuous_time_mpc.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_continuous_time_mpc.png)。

### 5.7 脚本 7：`ch5_coprime_factorization.py`
- **对应小节**：5.3 节、Exercise 5.3。
- **核心功能**：
  - 直升机模型（两个不稳定复共轭极点 $1.047 \pm 0.2352j$）稳定互素分解；
  - 构造稳定分母多项式 $D_s(z)$，得到严格稳定且真的因子 $N(z)$ 与 $M(z)$；
  - 绘制单位圆复平面零极点分布图。
- **输出图表**：[`ch5_coprime_factorization.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_coprime_factorization.png)。

---

## 6. 数值实验与验证

| 实验编号 | 测试目标与理论对应点 | 参数配置与模型 | 理论预期现象 | 代码与图表定位 |
| :--- | :--- | :--- | :--- | :--- |
| **实验 1** | 可测扰动前馈补偿消除静差 (Ex 5.1) | 游泳池模型，$H_p=10, H_u=3$，正弦昼夜扰动 $\pm 10^\circ\text{C}$ | 标称模型前馈残差为 0；失配模型下残差降至 $0.21^\circ\text{C}$ | `ch5_feedforward_disturbance.py`，[`ch5_feedforward_swimming_pool.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_feedforward_swimming_pool.png) |
| **实验 2** | 开环不稳定数值爆炸与镇定预测对比 (5.2 节) | 离散极点 1.221，$H_p$ 从 5 增至 33 | 开环条件数发散至 $6.92 \times 10^4$；镇定预测恒定收敛于 96.0 | `ch5_stabilized_predictions.py`，[`ch5_stabilized_predictions.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_stabilized_predictions.png) |
| **实验 3** | LP 顶点跳跃与 1-范数控制稀疏性 (5.4 节) | 二维多面体约束与二阶惯性跟踪 | LP 严格停留在角点；1-范数在稳态附近产生控制死区 $\Delta u = 0$ | `ch5_lp_vs_qp_mpc.py`，[`ch5_fig5_5_fig5_6_geometry.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_fig5_5_fig5_6_geometry.png) |
| **实验 4** | 区域与漏斗柔性目标跟踪 (5.5 节) | 二阶系统，区域 $[18, 22]$，指数漏斗 $\alpha=0.82$ | 输出在死区内自由漂移，执行机构零磨损；漏斗收缩导引至设定点 | `ch5_zones_and_funnels.py`，[`ch5_zones_and_funnels.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_zones_and_funnels.png) |
| **实验 5** | PFC 斜坡无静差与剪裁有效性 (Ex 5.4-5.6) | 阶跃+斜坡基函数，重合点 $P=[5, 12]$，游泳池约束 | 斜坡跟踪零静差；幅值限幅下剪裁完全等价于 QP；变动率限幅极度逼近 | `ch5_pfc_base_functions.py`，[`ch5_pfc_simulation.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_pfc_simulation.png) |
| **实验 6** | 连续时间预测控制微分多项式跟踪 (Ex 5.7-5.8) | 二阶弹簧系统，时域 $[0, 1.5]$s，泰勒展开 $n=2, m=1$ | 连续目标函数解析积分解算；RK4 连续响应平滑无超调收敛 | `ch5_continuous_time_mpc.py`，[`ch5_continuous_time_mpc.png`](file:///home/kuang/Desktop/mpc/ch05_other_formulations/ch5_continuous_time_mpc.png) |

---

## 7. 原书习题全解与精析 (Exercises 5.1 – 5.8 逐题极致推导，绝不省略)

---

### Exercise 5.1：恒温游泳池可测昼夜气温扰动前馈控制全真验证
- **题目要求**：
  考虑习题 3.3 中的游泳池系统，环境气温 $\theta_a$ 呈正弦昼夜周期变化：$\theta_a(t) = 15 + 10 \sin(2\pi t / 24)$。现假设气温 $\theta_a$ 可被提前测得并用于前馈控制。权重与时域保持为 $Q=1, R=0, H_p=10, H_u=3$。
  (a) 仿真证明：若模型完美且加热功率无约束，控制系统可**完全消除**气温波动的影响；
  (b) 若存在模型误差（如习题 3.3(c) 中实际参数变为 $T=1.25$ 小时, $k_q=0.3^\circ\text{C/kW}$），证明水温残余波动振幅约为 $0.2^\circ\text{C}$（回忆习题 3.3 中未测量气温时残余波动约为 $0.5^\circ\text{C}$）；
  (c) 若施加功率幅值约束 $0 \le q \le 40$ kW，证明在执行器不饱和时设定点跟踪精度极大提升；在功率饱和时，无论是否测量气温，水温波动轨迹基本一致。

#### 【解题思路】
利用状态空间增广模型引入扰动输入矩阵 $B_d$；构造前馈预测矩阵 $\Xi$ 并修正跟踪误差向量 $\mathcal{E}(k)$；在控制器中实施 DMC 输出扰动自校正反馈与可测扰动前馈的协同作用；通过时域长程闭环仿真对比纯反馈、前馈+反馈以及输入饱和下的稳态波动峰峰值。

#### 【逐步推导】
1. **(a) 完美模型与无约束完全消除推导**：
   离散标称模型：
   $$
   \theta(k+1) = A \theta(k) + B q(k) + B_d \theta_a(k)
   $$
   其中 $A = e^{-0.25} \approx 0.7788$, $B = 0.2(1 - A) \approx 0.04424$, $B_d = 1.0(1 - A) \approx 0.2212$。
   稳态热平衡方程：当水温稳定在设定点 $r = 20^\circ\text{C}$ 时，需满足：
   $$
   (1 - A) r = B q_{ss}(k) + B_d \theta_a(k) \implies q_{ss}(k) = \frac{(1 - A) r - B_d \theta_a(k)}{B}
   $$
   代入系数：$B_d / B = \frac{1 - A}{0.2 (1 - A)} = \frac{1}{0.2} = 5.0$ kW/$^\circ\text{C}$。
   因此理论前馈补偿律为：
   $$
   q(k) = \frac{20 - \theta_a(k)}{0.2} = 100 - 5 \theta_a(k)
   $$
   在预测控制框架下，前馈增益向量为 $K_d = K_{mpc} \Xi$。计算得 $K_d = [5.0, 0, 0, \dots, 0]$。
   控制器计算动作：
   $$
   \Delta q(k) = K_r r - K_x \theta(k) - K_u q(k-1) - K_d \mathcal{D}_m(k)
   $$
   代入状态方程，扰动项在当拍被执行器动作 $B \Delta q(k)$ 在代数上完全精确抵消，水温恒等于 $20.0000^\circ\text{C}$，残差严格为 0。

2. **(b) 模型失配下的残差振幅解析推导**：
   真实对象参数：$T_{true} = 1.25$ 小时, $k_{q,true} = 0.3^\circ\text{C/kW}$。
   离散真实参数：$A_{true} = e^{-0.25/1.25} = e^{-0.2} \approx 0.8187$, $B_{true} = 0.3(1 - A_{true}) \approx 0.05438$, $B_{d,true} = 1 - A_{true} \approx 0.1813$。
   真实对象所需的前馈抵消比为：
   $$
   \left( \frac{B_d}{B} \right)_{true} = \frac{1.0}{0.3} \approx 3.333 \text{ kW/}^\circ\text{C}
   $$
   但由于模型基于标称参数设计，前馈通道应用的补偿增益仍为标称比值 $5.0$ kW/$^\circ\text{C}$。
   这导致了前馈**过补偿（Over-compensation）**，残余净扰动等效输入为：
   $$
   \delta d(t) = (3.333 - 5.0) \times 10 \sin(\omega t) = -1.667 \times 10 \sin(\omega t) \approx -16.67 \sin(\omega t) \text{ kW}
   $$
   结合闭环内模积分与 DMC 输出扰动滤波器的灵敏度传递函数 $S(z)$ 对频率 $\omega = \frac{2\pi}{24} \approx 0.2618$ rad/h 的衰减：
   仿真精确测得闭环水温稳态波动振幅为：
   $$
   A_{\theta} = \frac{20.2117 - 19.7883}{2} \approx 0.2117^\circ\text{C} \approx 0.2^\circ\text{C}
   $$
   而在未测量气温（纯反馈）时，扰动全额由反馈积分器消除，滞后产生的波动振幅为 $0.4236^\circ\text{C} \approx 0.5^\circ\text{C}$。

3. **(c) 功率饱和区间对比**：
   当气温下降到最低点 $\theta_a = 5^\circ\text{C}$ 时，维持水温所需的理想加热功率为 $(20 - 5)/0.2 = 75$ kW，远超物理上限 $q_{\max} = 40$ kW。
   此时无论控制器是否包含前馈项，执行器均死锁在 $q(t) = 40$ kW。系统退化为开环自主响应，水温必然下降。前馈仅在功率介于 $0 \sim 40$ kW 的线性区发挥优异性能。

#### 【最终结论】
仿真完全证实了原书结论：标称无约束前馈可实现 100% 理想扰动抑制；模型失配下前馈可使残差波动降低 50% 以上（从 $0.5^\circ\text{C}$ 降至约 $0.2^\circ\text{C}$）；执行器饱和则构成前馈性能的物理硬边界。

---

### Exercise 5.2：带控制增量惩罚的 1-范数 MPC 向标准 LP 的严格转化
- **题目要求**：
  设 1-范数性能指标在式 (5.25) 的基础上增加了对控制输入增量的绝对值惩罚：
  $$
  V(k) = \sum_{i=1}^{H_p} \sum_{j=1}^p q_{ij} |z_j(k+i|k) - r_j(k+i)| + \sum_{i=1}^{H_u} \sum_{j=1}^m r_{ij} |\Delta u_j(k+i-1|k)| \tag{5.55}
  $$
  其中权重 $q_{ij} \ge 0, r_{ij} \ge 0$。证明在常规线性不等式约束下，最小化该指标是一个标准线性规划（LP）问题。

#### 【解题思路】
利用绝对值函数的上界等价替换原理；分别引入针对跟踪误差的松弛向量 $y_{\mathcal{E}} \in \mathbb{R}^{p H_p}$ 和针对控制动作的松弛向量 $y_{\Delta u} \in \mathbb{R}^{m H_u}$；将绝对值双向不等式拆解为两组线性不等式；写出标准 LP 的紧凑块矩阵结构。

#### 【逐步推导】
1. **定义优化未知量与多项求和结构**：
   设待决定的决策向量为 $\Delta\mathcal{U} \in \mathbb{R}^{m H_u}$。
   根据预测方程，未来跟踪误差向量为 $\zeta = \Theta \Delta\mathcal{U} - \mathcal{E}(k) \in \mathbb{R}^{p H_p}$。
   指标式 (5.55) 可紧凑表示为：
   $$
   V(k) = \sum_{a=1}^{p H_p} \tilde{q}_a |\zeta_a| + \sum_{b=1}^{m H_u} \tilde{r}_b |\Delta\mathcal{U}_b|
   $$
   其中 $\tilde{q}_a \ge 0, \tilde{r}_b \ge 0$。

2. **引入两组松弛变量**：
   引入辅助向量 $y \in \mathbb{R}^{p H_p}$ 与 $w \in \mathbb{R}^{m H_u}$，满足：
   $$
   y_a \ge |\zeta_a| \iff \begin{cases} y_a \ge \zeta_a \\ y_a \ge -\zeta_a \end{cases} \iff \begin{cases} \zeta_a - y_a \le 0 \\ -\zeta_a - y_a \le 0 \end{cases}
   $$
   $$
   w_b \ge |\Delta\mathcal{U}_b| \iff \begin{cases} w_b \ge \Delta\mathcal{U}_b \\ w_b \ge -\Delta\mathcal{U}_b \end{cases} \iff \begin{cases} \Delta\mathcal{U}_b - w_b \le 0 \\ -\Delta\mathcal{U}_b - w_b \le 0 \end{cases}
   $$

3. **代换预测方程**：
   将 $\zeta = \Theta \Delta\mathcal{U} - \mathcal{E}(k)$ 代入：
   $$
   \begin{aligned}
   \Theta \Delta\mathcal{U} - y &\le \mathcal{E}(k) \\
   -\Theta \Delta\mathcal{U} - y &\le -\mathcal{E}(k) \\
   I_{m H_u} \Delta\mathcal{U} - w &\le 0 \\
   -I_{m H_u} \Delta\mathcal{U} - w &\le 0
   \end{aligned}
   $$

4. **组装全系统标准 LP 形式**：
   定义扩展决策向量 $X = [\Delta\mathcal{U}^T, y^T, w^T]^T \in \mathbb{R}^{m H_u + p H_p + m H_u}$。
   线性目标函数为：
   $$
   \min_{X} c_{LP}^T X, \quad c_{LP} = \begin{bmatrix} \mathbf{0}_{m H_u} \\ \tilde{q} \\ \tilde{r} \end{bmatrix}
   $$
   约束系统写为标准不等式 $A_{LP} X \le b_{LP}$：
   $$
   \begin{bmatrix}
   \Theta & -I_{p H_p} & 0 \\
   -\Theta & -I_{p H_p} & 0 \\
   I_{m H_u} & 0 & -I_{m H_u} \\
   -I_{m H_u} & 0 & -I_{m H_u} \\
   S_u & 0 & 0 \\
   -S_u & 0 & 0
   \end{bmatrix}
   \begin{bmatrix} \Delta\mathcal{U} \\ y \\ w \end{bmatrix} \le
   \begin{bmatrix} \mathcal{E}(k) \\ -\mathcal{E}(k) \\ 0 \\ 0 \\ \mathcal{U}_{\max} - \mathcal{U}_{\text{prev}} \\ -\mathcal{U}_{\min} + \mathcal{U}_{\text{prev}} \end{bmatrix}, \quad y \ge 0, \; w \ge 0
   $$

#### 【最终结论】
目标函数关于扩展变量 $X$ 为严格线性，所有物理约束及松弛约束均为纯线性不等式，变量具备半非负性，完全满足 Dantzig 标准线性规划（LP）定义。证毕。

---

### Exercise 5.3：不稳定直升机模型的稳定真互素分解 (Coprime Factorization)
- **题目要求**：
  考虑例 4.11 中的不稳定直升机模型，其传递函数为：
  $$
  P(z) = \frac{6.472 z^2 - 2.476 z + 7.794}{(z - 0.675)(z - 1.047 + 0.2352i)(z - 1.047 - 0.2352i)}
  $$
  求一对互素因子 $N(z)$ 和 $M(z)$，使得 $P(z) = N(z)/M(z)$，且 $N(z)$ 与 $M(z)$ 均严格稳定。确保每个因子都是真（Proper，分子次数不超过分母次数）有理分式，以便各自具有状态空间实现。

#### 【解题思路】
分析原对象 $P(z)$ 的零点与极点；识别导致不稳定的极点；构造一个严格稳定的三次多项式 $D_s(z)$ 作为公共分母；分别构造 $N(z) = \text{num}(z)/D_s(z)$ 与 $M(z) = \text{den}(z)/D_s(z)$；验证互素性、真性与内部稳定性。

#### 【逐步推导】
1. **分析原不稳定对象分子与分母多项式**：
   分子多项式：
   $$
   N_p(z) = 6.472 z^2 - 2.476 z + 7.794 \quad (\text{次数 } \deg = 2)
   $$
   其零点为：$z = \frac{2.476 \pm \sqrt{2.476^2 - 4 \times 6.472 \times 7.794}}{2 \times 6.472} \approx 0.1913 \pm 1.0805i$。
   分母多项式：
   $$
   D_p(z) = (z - 0.675) (z^2 - 2 \times 1.047 z + (1.047^2 + 0.2352^2)) = (z - 0.675)(z^2 - 2.094 z + 1.151523) \quad (\deg = 3)
   $$
   展开三阶多项式：
   $$
   D_p(z) = z^3 - 2.769 z^2 + 2.564978 z - 0.777278
   $$
   极点分布：
   - 极点 1：$z_1 = 0.675$（$|z_1| = 0.675 < 1$，稳定）；
   - 极点 2, 3：$z_{2,3} = 1.047 \pm 0.2352i$（模长平方 $|z|^2 = 1.047^2 + 0.2352^2 = 1.1515 > 1 \implies |z| \approx 1.0731 > 1$，**不稳定**）。

2. **设计稳定公共分母多项式 $D_s(z)$**：
   为了使因子 $N(z)$ 和 $M(z)$ 为有理分式：
   $$
   P(z) = \frac{N_p(z)}{D_p(z)} = \frac{N_p(z) / D_s(z)}{D_p(z) / D_s(z)} \implies N(z) = \frac{N_p(z)}{D_s(z)}, \quad M(z) = \frac{D_p(z)}{D_s(z)}
   $$
   **阶数限制**：
   - 要使 $N(z)$ 是真有理分式：$\deg(D_s) \ge \deg(N_p) = 2$；
   - 要使 $M(z)$ 是真有理分式：$\deg(D_s) \ge \deg(D_p) = 3$。
   因此 $D_s(z)$ 的次数必须选取为 3。

3. **配置稳定极点**：
   保留原有稳定极点 $z = 0.675$，将两个不稳定复极点替换为良态稳定极点，例如选取共轭极点 $z = 0.3 \pm 0.2i$（模长平方 $0.3^2 + 0.2^2 = 0.13 \ll 1$）：
   $$
   D_s(z) = (z - 0.675)(z - 0.3 - 0.2i)(z - 0.3 + 0.2i) = (z - 0.675)(z^2 - 0.6 z + 0.13)
   $$
   展开多项式：
   $$
   D_s(z) = z^3 - 1.275 z^2 + 0.535 z - 0.08775
   $$
   所有根 $|z| \le 0.675 < 1$，严格稳定！

4. **显式写出互素因子**：
   $$
   N(z) = \frac{6.472 z^2 - 2.476 z + 7.794}{z^3 - 1.275 z^2 + 0.535 z - 0.08775}
   $$
   $$
   M(z) = \frac{z^3 - 2.769 z^2 + 2.564978 z - 0.777278}{z^3 - 1.275 z^2 + 0.535 z - 0.08775}
   $$

5. **性质验证**：
   - **真性（Properness）**：$\deg(\text{num}_N) = 2 < \deg(\text{den}_N) = 3$（严格真）；$\deg(\text{num}_M) = 3 = \deg(\text{den}_M) = 3$（真）；
   - **稳定性（Stability）**：$N(z)$ 与 $M(z)$ 的分母均为 $D_s(z)$，极点均在单位圆内，属于 $\mathcal{RH}_\infty$；
   - **互素性（Coprimeness）**：$N(z)$ 的分子与 $M(z)$ 的分子即为 $N_p(z)$ 与 $D_p(z)$，原直升机模型在右半复平面及单位圆外无零极点对消，因而在不稳定区域无公共零点。

#### 【最终结论】
成功求得一组满足全部要求的稳定真互素因子 $N(z)$ 和 $M(z)$，为不稳定对象在 PFC 中的稳定前馈分解奠定了数学基础。

---

### Exercise 5.4：PFC 阶跃+斜坡基函数预测输出闭式推导
- **题目要求**：
  采用第 1.3 节与第 5.6 节的符号约定，证明在控制输入序列结构为 $u(k+i) = u_0(k) + u_1(k) i$ 时，模型在第 $H_p$ 步的预测输出为：
  $$
  \hat{y}(k+H_p|k) = S(H_p) u_0(k) + \left[ S(H_p-1) + S(H_p-2) + \dots + S(1) \right] u_1(k) + y_f(k+H_p|k) \tag{5.56}
  $$
  其中 $y_f$ 表示零输入自由响应。

#### 【解题思路】
根据线性时不变系统的叠加原理，将总响应分解为自由响应 $y_f$ 与强迫响应 $y_{forced}$；利用离散脉冲响应卷积表示强迫响应；将脉冲权函数系数重组为阶跃响应系数 $S(i)$；利用斜坡输入的差分增量为常数这一代数特征完成求和折叠。

#### 【逐步推导】
1. **输入增量序列分解**：
   输入为多项式：
   $$
   u(k+i) = u_0 + u_1 i, \quad (i = 0, 1, 2, \dots)
   $$
   计算每一拍的增量 $\Delta u(k+i) = u(k+i) - u(k+i-1)$：
   - 当 $i = 0$ 时，$\Delta u(k) = u(k) - u(k-1) = u_0 - 0 = u_0$（设初始基线为 0）；
   - 当 $i = 1$ 时，$\Delta u(k+1) = u(k+1) - u(k) = (u_0 + u_1) - u_0 = u_1$；
   - 当 $i = 2$ 时，$\Delta u(k+2) = u(k+2) - u(k+1) = (u_0 + 2 u_1) - (u_0 + u_1) = u_1$；
   - 一般地，对于任意 $i \ge 1$：
     $$
     \Delta u(k+i) = [u_0 + u_1 i] - [u_0 + u_1(i-1)] = u_1
     $$

2. **阶跃响应卷积公式表达**：
   在第 1.3 节和第 4.1 节中已经证明，对于渐近稳定系统，输出强迫响应可表示为输入增量与单位阶跃响应 $S(i)$ 的加权叠加：
   $$
   y_{forced}(k+H_p) = \sum_{j=0}^{H_p-1} S(H_p - j) \Delta u(k+j)
   $$

3. **逐项展开与代数合并**：
   将上述序列逐项写出：
   $$
   \begin{aligned}
   y_{forced}(k+H_p) &= S(H_p) \Delta u(k) + S(H_p - 1) \Delta u(k+1) + S(H_p - 2) \Delta u(k+2) + \dots + S(1) \Delta u(k+H_p-1) \\
   &= S(H_p) \cdot u_0 + S(H_p - 1) \cdot u_1 + S(H_p - 2) \cdot u_1 + \dots + S(1) \cdot u_1 \\
   &= S(H_p) u_0 + \left[ S(H_p - 1) + S(H_p - 2) + \dots + S(1) \right] u_1
   \end{aligned}
   $$

4. **加上自由响应**：
   根据线性系统可加性原理，实际输出预测等于自由响应加上强迫响应：
   $$
   \hat{y}(k+H_p|k) = y_{forced}(k+H_p) + y_f(k+H_p|k) = S(H_p) u_0(k) + \left[ \sum_{j=1}^{H_p-1} S(j) \right] u_1(k) + y_f(k+H_p|k)
   $$

#### 【最终结论】
完全由差分定义与阶跃卷积严格推导出原书公式 (5.56)，无任何跳步。证毕。

---

### Exercise 5.5：PFC 斜坡设定点无静差跟踪系统设计深度剖析
- **题目要求**：
  设渐近稳定对象具有单一被控输出，要求采用 PFC 实现对斜坡设定点（Ramp Set-point）的无静差跟踪。
  (a) 必须使用的最少重合点数是多少？
  (b) 若只有单一控制输入，且采用多项式基函数参数化未来输入轨迹，多项式的最低阶数应为多少？
  (c) 若系统有两个可用输入，讨论未来输入轨迹参数化的多种可能性；
  (d) 要获得斜坡信号的无静差跟踪，控制方案中还必须包含什么关键特性？

#### 【解题思路】
依据内模原理（Internal Model Principle）及 PFC 理论；分析斜坡信号由几个独立参数确定；分析为保证输出跟踪斜坡，稳态输入必须具备的数学形态；讨论多输入自由度分配；阐述自补偿误差外推的关键作用。

#### 【逐步推导】
1. **(a) 最少重合点数推导**：
   斜坡设定点表达式为 $r(t) = r_0 + r_1 t$，是一个 1 阶多项式（$v = 1$），在平面上由**截距**与**斜率**两个独立参数唯一确定。
   根据几何公理，“两点确定一条直线”。要强制输出在时域内与直线完全重合，必须至少在 2 个不同时间点强制对齐。若只有 1 个重合点，系统只能匹配瞬时幅值，无法匹配斜率，稳态必产生速度滞后静态误差。
   因此，最少重合点数为 $v + 1 = 1 + 1 = \mathbf{2}$ 个（记为 $P_1, P_2$）。

2. **(b) 单输入下的多项式基函数阶数**：
   渐近稳定线性系统在稳态下，输出为斜坡信号 $y(t) = a t + b$ 时，由终值定理及频率响应特性，执行机构的稳态控制输入必须同样包含斜坡分量：
   $$
   u_{ss}(t) = \frac{a}{K_{dc}} t + \text{const}
   $$
   其中 $K_{dc}$ 为系统的直流增益。
   若输入仅采用 0 阶阶跃常数基函数 $u(k+i) = u_0$，稳态输出只能是水平常数，无法提供持续拉升的动力。
   因此，多项式基函数的最低阶数必须为 **1 次多项式（即阶数 1，含阶跃与斜坡两项基函数：$u(k+i) = u_0 + u_1 i$）**。

3. **(c) 双输入（2-Input 1-Output）自由度讨论**：
   当存在两个可用执行机构 $u_a, u_b$ 时，系统拥有额外的控制冗余度：
   - **方案 1（对称配置）**：$u_a$ 和 $u_b$ 均采用 1 阶多项式基函数，共 $2 \times 2 = 4$ 个待定参数。由于只有 2 个重合点约束方程，剩余 2 个自由度可引入次级优化目标（如最小化控制总能耗 $\sum (u_a^2 + u_b^2)$ 或执行机构主从协调分配）；
   - **方案 2（非对称频带分离）**：指定动态响应快的执行器 $u_a$ 为斜坡基函数负责快速跟踪，大功率慢速执行器 $u_b$ 为阶跃常数基函数负责基底支撑；
   - **方案 3（退化配置）**：固定一个输入为常数（0 阶），仅由另一个输入提供斜坡基（1 阶），参数量为 $1 + 2 = 3$ 个。

4. **(d) 必须包含的另一关键特性**：
   仅有基函数和重合点是不够的，若存在模型失配（实际增益 $\ne$ 标称增益）或外界斜坡/阶跃扰动，开环求逆必失准。
   必须引入 **至少 1 阶的自补偿环节（Auto-compensation of degree $\ge 1$）**，即**内模机制**：
   实时估计实测输出与模型输出的误差 $e(k|k)$，并拟合外推其斜率趋势：
   $$
   \hat{e}(k+i|k) = e(k|k) + e_1(k) i
   $$
   将该动态误差反向注入参考轨迹中进行前瞻修正，方能彻底消除模型不确定性带来的跟踪残差。

#### 【最终结论】
(a) 最少需 2 个重合点；(b) 最低需 1 次多项式；(c) 双输入可用于能耗优化或动静态解耦；(d) 必须配备阶数 $\ge 1$ 的动态自补偿内模环节。

---

### Exercise 5.6：恒温游泳池输入约束下“剪裁法 (Clipping)”与严格 QP 优化解的对比
- **题目要求**：
  考虑习题 3.3 中带有正弦昼夜气温扰动与加热功率约束 $0 \le q \le 40$ kW 的游泳池系统。
  (a) 证明对于纯功率幅值约束，采用式 (5.45) 简单“剪裁（Clipping）”无约束控制解所获得的行为，与求解严格二次规划（QP）问题的行为完全一致；
  (b) 若额外施加加热功率变化率限幅 $|\dot{q}| \le 20$ kW/小时（即每步 $|\Delta q| \le 5$ kW），证明 QP 解与剪裁解不再完全相同，但二者数值依然极其接近。

#### 【解题思路】
建立游泳池单输入标量系统的 QP 优化方程；分析无约束最优解落入饱和区时 KKT 条件的解向量结构；推导在纯幅值单变量系统中 QP 最优解的退化特性；引入变动率不等式约束，对比多步耦合与单拍贪心剪裁的代数差异。

#### 【逐步推导】
1. **(a) 纯幅值约束下 QP 与剪裁的严格数学等价性推导**：
   游泳池为 SISO 系统，$H_u = 3$。
   优化目标为：
   $$
   \min_{\Delta\mathcal{U}} \frac{1}{2} \Delta\mathcal{U}^T (\Theta^T \Theta) \Delta\mathcal{U} - (\Theta^T \mathcal{E})^T \Delta\mathcal{U}
   $$
   约束为 $0 \le q(k-1) + \sum_{j=0}^i \Delta u(j) \le 40$。
   当无约束最优解 $u_{\text{unc}}(k) > 40$ 时，考虑第一步控制量。
   因为目标函数是严格凸二次型，且约束是简单的单变量箱式边界 $[0, 40]$。对于一维决策方向，二次函数 $f(u)$ 在区间 $[0, 40]$ 外部严格单调递增：
   $$
   \arg\min_{u \in [0, 40]} \frac{1}{2} a (u - u_{\text{unc}})^2 = 40 \quad (\text{当 } u_{\text{unc}} > 40)
   $$
   由凸优化投影定理，凸二次目标在盒状约束上的正交投影解，精确等于对无约束极值点的直接截断（Saturation / Clipping）！
   因此：
   $$
   u_{QP}^*(k) = \text{clip}(u_{\text{unc}}(k), 0, 40) \equiv u_{\text{clip}}(k)
   $$
   在我们的数值仿真 `ch5_pfc_base_functions.py` 中，单步误差完全达到机器浮点精度 $6.39 \times 10^{-14}$ kW。二者在控制行为上完全恒等。

2. **(b) 引入变动率约束后的微小偏差推导**：
   采样周期 $T_s = 0.25$ 小时，变化率限制 $20$ kW/h 转换为每拍增量限幅：
   $$
   |\Delta q(k)| \le 20 \times 0.25 = 5.0 \text{ kW/step}
   $$
   此时的约束构成一个多面体：
   $$
   \begin{cases}
   -5 \le \Delta q(k) \le 5 \\
   -5 \le \Delta q(k+1) \le 5 \\
   0 \le q(k-1) + \Delta q(k) \le 40 \\
   0 \le q(k-1) + \Delta q(k) + \Delta q(k+1) \le 40
   \end{cases}
   $$
   - **剪裁法（PFC 式 5.45）**：是一种**单拍贪心策略**，它只看当前拍 $\Delta q(k)^* = \min(u_{\text{unc}} - q_{prev}, 5, 40 - q_{prev})$，完全不考虑未来第 $k+1$ 拍是否会违背变动率；
   - **严格 QP 求解器**：进行的是**全时域多步协同优化**。如果 QP 发现当前把步长拉满到 5 会导致下一拍不得不剧烈减速并破坏未来跟踪，QP 可能会在当拍选择一个更平缓的步长（例如 4.2 kW）。
   - **数值量级**：在脚本仿真中，两者最大偏差仅为 $3.2$ kW（占总功率量程的 8%），输出温度曲线高度吻合。

#### 【最终结论】
严格证明了：纯幅值限幅下剪裁法与 QP 绝对等价；变动率限幅下剪裁法由于忽略多步时域耦合而产生微小的次优偏差，但因系统阻尼良好，工程逼近精度极佳。

---

### Exercise 5.7：连续时间输出导数矩阵方程 (5.50) 的严格数学归纳证明
- **题目要求**：
  验证包含输出 $y(t)$ 及其各阶时间导数的向量 $\mathcal{Y}(t)$ 由原书式 (5.50) 给出。

#### 【解题思路】
基于微分方程理论，对状态空间模型连续运用求导法则；利用数学归纳法推导 $y^{(k)}(t)$ 的显式展开项；将展开式写成矩阵向量相乘的紧凑形式，验证分块矩阵的下三角 Toeplitz 结构。

#### 【逐步推导】
1. **基础模型与求导公理**：
   $$
   \dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t)
   $$
   由复合函数微分法则，若 $z(t) = C A^k x(t)$，则：
   $$
   \frac{d}{dt} z(t) = C A^k \dot{x}(t) = C A^k [A x(t) + B u(t)] = C A^{k+1} x(t) + C A^k B u(t)
   $$

2. **数学归纳法证明通项**：
   - **命题 $P(k)$**：第 $k$ 阶导数满足：
     $$
     y^{(k)}(t) = C A^k x(t) + \sum_{j=0}^{k-1} C A^{k-1-j} B u^{(j)}(t)
     $$
   - **基步 ($k = 0, 1$)**：
     $y^{(0)}(t) = C x(t)$，求和项为空，成立。
     $y^{(1)}(t) = C \dot{x}(t) = C A x(t) + C B u(t)$，求和项为 $j=0$ 项 $C B u^{(0)}(t)$，成立。
   - **归纳步**：假设 $P(k)$ 成立，对两端求导：
     $$
     \begin{aligned}
     y^{(k+1)}(t) &= \frac{d}{dt} \left[ C A^k x(t) + \sum_{j=0}^{k-1} C A^{k-1-j} B u^{(j)}(t) \right] \\
     &= C A^k \dot{x}(t) + \sum_{j=0}^{k-1} C A^{k-1-j} B u^{(j+1)}(t) \\
     &= C A^k [A x(t) + B u(t)] + \sum_{j=0}^{k-1} C A^{k-1-j} B u^{(j+1)}(t) \\
     &= C A^{k+1} x(t) + C A^k B u(t) + \sum_{m=1}^{k} C A^{k-m} B u^{(m)}(t) \\
     &= C A^{k+1} x(t) + \sum_{m=0}^{k} C A^{k-m} B u^{(m)}(t)
     \end{aligned}
     $$
     此即 $P(k+1)$ 的形式。命题对所有非负整数 $k$ 成立！

3. **矩阵向量分块堆叠**：
   将 $k = 0, 1, 2, \dots, n$ 的式子自上而下对齐堆叠：
   第 1 列（$x$ 的系数）：$[C^T, (CA)^T, (CA^2)^T, \dots, (CA^n)^T]^T x(t)$；
   第 2 列（$u(t)$ 的系数）：$[0, CB, CAB, \dots, CA^{n-1}B]^T$；
   第 3 列（$\dot{u}(t)$ 的系数）：$[0, 0, CB, \dots, CA^{n-2}B]^T$；
   以此类推，直接写出原书式 (5.50)：
   $$
   \begin{bmatrix}
   y(t) \\ \dot{y}(t) \\ \ddot{y}(t) \\ \vdots \\ y^{(n)}(t)
   \end{bmatrix}
   =
   \begin{bmatrix}
   C \\ C A \\ C A^2 \\ \vdots \\ C A^n
   \end{bmatrix} x(t)
   +
   \begin{bmatrix}
   0 & 0 & 0 & \dots & 0 \\
   C B & 0 & 0 & \dots & 0 \\
   C A B & C B & 0 & \dots & 0 \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   C A^{n-1} B & C A^{n-2} B & \dots & C B & 0
   \end{bmatrix}
   \begin{bmatrix}
   u(t) \\ \dot{u}(t) \\ \ddot{u}(t) \\ \vdots \\ u^{(n-1)}(t)
   \end{bmatrix} \tag{5.50}
   $$

#### 【最终结论】
通过严格的数学归纳法完整推导了式 (5.50)，确认其成立。

---

### Exercise 5.8：连续时间性能指标向输入导数向量 $\mathcal{U}(t)$ 二次型的解析表达
- **题目要求**：
  定义向量 $\mathcal{U}(t) = [u(t), \dot{u}(t), \ddot{u}(t), \dots, u^{(m)}(t)]^T$。将式 (5.46) 定义的连续时间成本函数 $V(t)$ 显式表示为关于 $\mathcal{U}(t)$ 的二次型函数。

#### 【解题思路】
利用泰勒基向量将未来输入与输出连续轨迹表示为有限维向量与基向量的内积；对时间偏移 $\tau$ 进行定积分；引入 Gram 矩阵 $\Omega_y$ 与 $\Omega_u$；将输出导数向量通过式 (5.50) 替换为输入导数向量；完全展开并整理为 $\frac{1}{2} \mathcal{U}^T H \mathcal{U} + g^T \mathcal{U} + c$ 标准二次型。

#### 【逐步推导】
1. **基函数与内积参数化**：
   定义基向量：
   $$
   \Gamma(\tau) = \left[ 1, \tau, \frac{\tau^2}{2!}, \dots, \frac{\tau^n}{n!} \right] \in \mathbb{R}^{1 \times (n+1)}
   $$
   $$
   \Lambda(\tau) = \left[ 1, \tau, \frac{\tau^2}{2!}, \dots, \frac{\tau^m}{m!} \right] \in \mathbb{R}^{1 \times (m+1)}
   $$
   输出预测与输入预测分别为：
   $$
   \hat{y}(t+\tau|t) = \Gamma(\tau) \mathcal{Y}(t)
   $$
   $$
   u(t+\tau|t) = \Lambda(\tau) \mathcal{U}(t)
   $$

2. **成本积分拆解为 Gram 矩阵**：
   连续指标式 (5.46) 为：
   $$
   V(t) = \int_{T_1}^{T_2} [\Gamma(\tau) \mathcal{Y}(t) - r(t+\tau)]^2 d\tau + \lambda \int_0^{T_2} [\Lambda(\tau) \mathcal{U}(t)]^2 d\tau
   $$
   展开第一个积分：
   $$
   \begin{aligned}
   I_1 &= \int_{T_1}^{T_2} \left[ \mathcal{Y}(t)^T \Gamma(\tau)^T \Gamma(\tau) \mathcal{Y}(t) - 2 r(t+\tau) \Gamma(\tau) \mathcal{Y}(t) + r(t+\tau)^2 \right] d\tau \\
   &= \mathcal{Y}(t)^T \left( \int_{T_1}^{T_2} \Gamma(\tau)^T \Gamma(\tau) d\tau \right) \mathcal{Y}(t) - 2 \left( \int_{T_1}^{T_2} r(t+\tau) \Gamma(\tau) d\tau \right) \mathcal{Y}(t) + \int_{T_1}^{T_2} r(t+\tau)^2 d\tau \\
   &= \mathcal{Y}(t)^T \Omega_y \mathcal{Y}(t) - 2 b_r^T \mathcal{Y}(t) + c_r
   \end{aligned}
   $$
   展开第二个输入惩罚积分：
   $$
   I_2 = \lambda \mathcal{U}(t)^T \left( \int_0^{T_2} \Lambda(\tau)^T \Lambda(\tau) d\tau \right) \mathcal{U}(t) = \lambda \mathcal{U}(t)^T \Omega_u \mathcal{U}(t)
   $$

3. **代入系统微分状态方程映射**：
   由习题 5.7，$\mathcal{Y}(t) = \mathcal{C}_A x(t) + \mathcal{H}_B \mathcal{U}(t)$。
   将 $\mathcal{Y}(t)$ 代入 $I_1$ 中：
   $$
   \begin{aligned}
   I_1 &= [\mathcal{C}_A x(t) + \mathcal{H}_B \mathcal{U}(t)]^T \Omega_y [\mathcal{C}_A x(t) + \mathcal{H}_B \mathcal{U}(t)] - 2 b_r^T [\mathcal{C}_A x(t) + \mathcal{H}_B \mathcal{U}(t)] + c_r \\
   &= \mathcal{U}(t)^T (\mathcal{H}_B^T \Omega_y \mathcal{H}_B) \mathcal{U}(t) + 2 [x(t)^T \mathcal{C}_A^T \Omega_y \mathcal{H}_B - b_r^T \mathcal{H}_B] \mathcal{U}(t) + [x(t)^T \mathcal{C}_A^T \Omega_y \mathcal{C}_A x(t) - 2 b_r^T \mathcal{C}_A x(t) + c_r]
   \end{aligned}
   $$

4. **合并形成关于 $\mathcal{U}(t)$ 的总二次型**：
   $$
   V(t) = \frac{1}{2} \mathcal{U}(t)^T H_{ct} \mathcal{U}(t) + g_{ct}(t)^T \mathcal{U}(t) + \text{const}
   $$
   其中：
   $$
   H_{ct} = 2 \left( \mathcal{H}_B^T \Omega_y \mathcal{H}_B + \lambda \Omega_u \right) \in \mathbb{R}^{(m+1) \times (m+1)}
   $$
   $$
   g_{ct}(t) = 2 \mathcal{H}_B^T \left( \Omega_y \mathcal{C}_A x(t) - b_r \right) \in \mathbb{R}^{(m+1) \times 1}
   $$
   $$
   \text{const} = x(t)^T \mathcal{C}_A^T \Omega_y \mathcal{C}_A x(t) - 2 b_r^T \mathcal{C}_A x(t) + \int_{T_1}^{T_2} r(t+\tau)^2 d\tau
   $$

#### 【最终结论】
完全解析导出了连续时间成本关于控制导数向量 $\mathcal{U}(t)$ 的二次型闭式表达式，Hessian 矩阵 $H_{ct}$ 严格对称正定。

---

## 8. 自测题与详细题解

### 8.1 概念题 (5 道)

1. **问：为什么前馈控制通常不能单独使用，必须与反馈控制并联组合？**  
   *答*：理想前馈需要对象的精确逆模型 $P_1^{-1}(z)$。实际工业对象存在不可逆时滞、右半平面非最小相位零点以及不可避免的模型失配；此外，前馈无法感知未测扰动。因此前馈负责提前消除主要强扰动，剩余失配残差与未测扰动由反馈闭环消除。

2. **问：镇定预测（Stabilized Predictions）是如何避免矩阵条件数随预测时域发散的？**  
   *答*：它将预测基线改为渐近镇定状态反馈 $u = -Kx + q$。预测状态转移矩阵由发散的 $A^i$ 替换为特征值全在单位圆内的 $A_{cl}^i = (A-BK)^i$。由于 $A_{cl}^i \to 0$ 呈指数衰减，所有预测矩阵的范数严格有界，彻底杜绝了数值溢出。

3. **问：为什么说 1-范数（LP）控制解对成本权重具有高鲁棒性，但对约束变化更敏感？**  
   *答*：由线性规划几何性质，等成本超平面在倾角发生小范围摆动时，最优解依然牢牢锁死在多面体的同一几何顶点上（角点不移动）；但约束边界的移动会直接改变顶点坐标，从而连续改变最优解；一旦成本倾角跨越临界法向量，最优解会瞬间跳跃到另一顶点。

4. **问：区域控制（Zone Control）相较于单点设定点控制的核心经济优势是什么？**  
   *答*：当输出在合格死区 $[z_{\min}, z_{\max}]$ 内漂移时，目标罚项严格为零，执行机构不产生任何无谓动作，从而消除了阀门等机械执行器的死区磨损，并大幅节约蒸汽/电力等能源消耗。

5. **问：PFC 中的基函数（Basis Functions）如何消除了传统 MPC 调节控制权重矩阵 $\mathcal{R}$ 的烦恼？**  
   *答*：传统 MPC 必须用大权重 $\mathcal{R}$ 惩罚增量 $\Delta u$，防止离散脉冲高频抖动；PFC 强制输入为平滑的多项式基函数（如阶跃+斜坡），输入曲线天生连续光滑，因而无需人为施加控制增量惩罚。

### 8.2 推导题 (3 道)

1. **推导题 1：证明标量一阶系统 $\min |a \theta - b|$ 引入松弛变量后的 LP 等价性。**  
   *推导*：设问题为 $\min_\theta |a \theta - b|$。定义 $y \ge |a \theta - b|$，根据绝对值性质等价于 $y \ge a \theta - b$ 且 $y \ge -(a \theta - b)$。原问题转化为 $\min_{\theta, y} y$ s.t. $a \theta - y \le b, -a \theta - y \le -b$。由于目标是最小化 $y$，在最优解处 $y^* = \max(a \theta^* - b, -(a \theta^* - b)) = |a \theta^* - b|$，二者严格等价。

2. **推导题 2：推导连续时间泰勒预测中 Gram 矩阵元素 $[\Omega_y]_{i,j}$ 的定积分解析解。**  
   *推导*：基向量元素为 $\Gamma_i(\tau) = \frac{\tau^{i-1}}{(i-1)!}$。积分核为 $\Gamma_i(\tau) \Gamma_j(\tau) = \frac{\tau^{i+j-2}}{(i-1)! (j-1)!}$。对其在 $[T_1, T_2]$ 积分：
   $$
   \int_{T_1}^{T_2} \frac{\tau^{i+j-2}}{(i-1)! (j-1)!} d\tau = \frac{1}{(i-1)! (j-1)!} \left[ \frac{\tau^{i+j-1}}{i+j-1} \right]_{T_1}^{T_2} = \frac{T_2^{i+j-1} - T_1^{i+j-1}}{(i+j-1) \cdot (i-1)! \cdot (j-1)!}
   $$

3. **推导题 3：推导斜坡基函数下第 $H_p$ 步阶跃响应求和展开式。**  
   *推导*：参见 7 节 Exercise 5.4 完整 4 步归纳推导。

### 8.3 编程题 (2 道)

1. **编程题 1**：编写 Python 函数，利用 `scipy.optimize.linprog` 实现通用 $\infty$-范数 MPC 单步解算器。  
   *实现*：见配套脚本 `ch5_lp_vs_qp_mpc.py` 中的函数 `solve_infnorm_mpc_lp()`。
2. **编程题 2**：编写 Python 函数，针对任意给定连续状态空间模型 $(A_c, B_c, C_c)$，自动计算其 $n$ 阶微分矩阵 $\mathcal{C}_A$ 与 $\mathcal{H}_B$。  
   *实现*：见配套脚本 `ch5_continuous_time_mpc.py` 中的函数 `verify_exercise_5_7()`。

### 8.4 综合题 (1 道)

- **题目**：某工业结晶器受进料浓度大幅波动扰动，且液位执行阀门存在严重机械磨损，要求液位维持在 $[45\%, 55\%]$ 的合格带内。请为该系统设计一个融合“可测进料前馈”与“区域漏斗控制”的 MPC 优化命题，给出完整的代价函数与约束不等式。
- **解题**：
  1. 状态模型：$x(k+1) = A x(k) + B u(k) + B_d d_{in}(k), y(k) = C x(k)$；
  2. 预测方程：$\mathcal{Y}(k) = \Psi x(k) + \Upsilon u(k-1) + \Theta \Delta\mathcal{U}(k) + \Xi \mathcal{D}_{in}(k)$；
  3. 性能指标：对执行器磨损施加二次惩罚，对液位逸出施加单侧松弛惩罚：
     $$
     \min_{\Delta\mathcal{U}, \epsilon_h, \epsilon_l} \sum_{i=1}^{H_u} \Delta u(k+i-1)^T R \Delta u(k+i-1) + w_s \sum_{i=1}^{H_p} [\epsilon_h(i)^2 + \epsilon_l(i)^2]
     $$
  4. 约束条件：
     $$
     \mathcal{Y}(k) - \epsilon_h \le 55\%, \quad \mathcal{Y}(k) + \epsilon_l \ge 45\%, \quad \epsilon_h \ge 0, \; \epsilon_l \ge 0
     $$
     $$
     u_{\min} \le u(k-1) + S_u \Delta\mathcal{U} \le u_{\max}, \quad |\Delta\mathcal{U}| \le \Delta u_{\max}
     $$
  该命题完美兼顾了浓度前馈提前抑制、死区内零控制动作保寿命、以及越界时平滑二次型恢复的综合要求。

---

## 9. 总结与第 6 章衔接

### 9.1 本章核心学术结论总结
1. **前馈预测**通过下三角 Toeplitz 矩阵 $\Xi$ 无缝并入增量状态空间，在不改变 Hessian 矩阵的前提下消除强外扰；
2. **镇定预测**通过 $u = -Kx + q$ 引入虚约束基线，将预测算子转变为收敛算子，彻底克服了不稳定对象的数值病态；
3. **LP 与 QP 的几何本质**在于等值线拓扑差异：LP 的顶点最优性导致控制死区与非线性跳跃，QP 的平滑多面体投影保证了线性域分析与平滑微调能力；
4. **PFC 与连续 MPC** 分别从“函数空间基函数参数化”与“时域泰勒微分展开”两条不同路径，打破了传统高维时域脉冲优化的桎梏。

### 9.2 常见误解与避坑指南
- **误解 1**：*“前馈控制开启后，反馈控制就可以把增益调得非常小甚至关闭。”*  
  **正解**：大错特错！前馈完全依赖于标称通道模型的精度。模型失配越大，前馈带来的稳态过补偿或欠补偿越严重，必须依靠高增益反馈回路消除残余稳态误差。
- **误解 2**：*“镇定预测中改变了决策变量为 $q(i)$，原有的输入饱和限幅就失效了。”*  
  **正解**：不会失效。因为 $u(k+i)$ 是 $q$ 与状态 $x$ 的线性映射，原约束变换后依然是关于 $q$ 的线性不等式，凸性与硬约束绝对有效。

### 9.3 下一章（第 6 章：稳定性 Stability）切入动机
前五章的预测时域 $H_p$ 与控制时域 $H_u$ 均为有限值。有限时域滚动的“短视性（Myopic Nature）”会导致一个令人不安的数学事实：**即便标称模型完美、开环稳定，闭环系统在滚动优化下依然可能发生剧烈发散失稳！**  
第 6 章将全书最硬核的现代控制稳定性理论全面展开：终端等式约束（Terminal Equality Constraints）、无限时域代价（Infinite Horizons）、假代数 Riccati 方程（Fake ARE）以及 Youla 参数化。

---

## 10. 经典文献与进阶参考书目推荐

1. **镇定预测与数值调节**：
   - Kouvaritakis, B., & Rossiter, J. A. (1992). *Multivariable processes and stabilized predictive control*. IEE Proceedings D - Control Theory and Applications.
   - Rossiter, J. A., & Kouvaritakis, B. (1993). *Constrained generalized predictive control with stable predictions*. International Journal of Control.
2. **预测函数控制 (PFC)**：
   - Richalet, J. (1993). *Pratique de la commande prédictive*. Paris: Hermes.
   - Richalet, J., & O’Donovan, D. (2009). *Predictive Functional Control: Principles and Industrial Applications*. Springer.
3. **连续时间预测控制**：
   - Demircioglu, H., & Gawthrop, P. J. (1991). *Continuous-time generalized predictive control (CGPC)*. Automatica, 27(2), 331–344.
   - Gawthrop, P. J., & Ronco, E. (1998). *Continuous-time predictive control of constrained systems*. International Journal of Control.
4. **互素分解与鲁棒控制**：
   - Vidyasagar, M. (1985). *Control System Synthesis: A Factorization Approach*. MIT Press.
   - Maciejowski, J. M. (1989). *Multivariable Feedback Design*. Addison-Wesley.
