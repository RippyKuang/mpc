# 第 6 章 稳定性（Stability）深度导读与全推导指南

> **书目**：J. M. Maciejowski, *Predictive Control with Constraints*, Prentice Hall, 2002  
> **章节**：Chapter 6: Stability (pp. 167–187)  
> **定位**：面向零现代控制理论、零最优控制、零非线性与 Lyapunov 理论基础的读者。以极其严谨、代数显式展开、绝无跳步的形式，从经典单输入单输出传递函数、零极点与差分方程出发，彻底透视模型预测控制（MPC）的闭环稳定性机制。

---

## 1. 章节元信息

- **章号**：第 6 章（Chapter 6）
- **标题**：稳定性（Stability）
- **核心主题**：
  1. 为什么“在每一个采样步都最优”的开环优化控制，施加到闭环系统后可能导致严重的**闭环发散失稳**（Example 6.1）？
  2. 终端等式约束法（Terminal Equality Constraint, $x(k+H_p|k) = 0$）与 Keerthi-Gilbert 定理（Theorem 6.1）：如何把开环最优目标函数直接作为闭环离散 Lyapunov 函数证明渐近稳定性？
  3. 无限预测时域法（Infinite Horizons）：如何通过离散 Lyapunov 方程 $A^T \bar{Q} A - \bar{Q} + C^T Q C = 0$ 将无限时域尾部代价压缩为一个精确的有限时域终端二次型权重 $\bar{Q}$？
  4. 不稳定对象的模态分解（Modal Decomposition）：如何利用 Jordan/特征模态分解将系统解耦为稳定与不稳定子空间，并对不稳定模态施加终端等式约束 $W_u x(k+H_u) = 0$？
  5. 伪代数 Riccati 方程法（Fake Algebraic Riccati Equations, FARE）：倒向 Riccati 差分方程的单调性，以及状态加权矩阵 $Q \ge 0$ 的充分性与非必要性（Exercise 6.7）。
  6. Youla 参数化（Youla Parametrization / $Q$-Parameterization）：基于内部模型控制（IMC）结构，将所有稳定控制器参数化为有限冲激响应（FIR）滤波器，将 MPC 优化问题转化为全局内部稳定无忧的凸二次规划（Convex QP）。
- **配套程序**：本章包含 7 个独立、可直接运行的 Python 3 仿真与验证脚本，涵盖本章全部例题、图表、定理与习题。

---

## 2. 符号、假设与预备知识

为确保零现代控制理论基础的读者完全理解，本节建立直观数学对应桥梁。

### 2.1 极点与稳定性直观对照

对于经典传递函数 $G(z) = \frac{Y(z)}{U(z)}$：
- 传递函数的极点是特征多项式 $A(z) = 0$ 的根 $\lambda$。
- 在时域中，离散单极点响应为几何级数（指数衰减或发散）：
  $$y(k) \propto \lambda^k$$
- 若 $|\lambda| < 1$（极点严格位于复平面单位圆内）：当 $k \to \infty$ 时，$\lambda^k \to 0$，系统是**渐近稳定的（Asymptotically Stable）**。
- 若 $|\lambda| > 1$（极点位于单位圆外）：当 $k \to \infty$ 时，$|\lambda|^k \to \infty$，系统**发散失稳（Unstable）**。
- 若 $|\lambda| = 1$（单位圆上的单极点）：产生等幅振荡或等幅持续响应（临界稳定），若有重极点则线性发散。
- 若 $\lambda = 0$：$\lambda^k = 0$（对所有 $k \ge 1$），系统在有限步内精确到达原点，称为**死拍响应（Deadbeat Response）**。

在离散状态空间模型中：
$$x(k+1) = A x(k) + B u(k), \quad y(k) = C x(k)$$
- 系统的自主动力学由状态转移矩阵 $A$ 决定。
- $A$ 的特征值 $\lambda_i(A)$（即满足 $\det(\lambda I - A) = 0$ 的标量）完全等价于传递函数的开环极点！
- 当施加线性状态反馈 $u(k) = -K x(k)$ 时，闭环状态方程为：
  $$x(k+1) = (A - B K) x(k) = A_{cl} x(k)$$
- 闭环系统的稳定性完全由闭环矩阵 $A_{cl} = A - B K$ 的特征值决定。所有特征值模长严格小于 1（谱半径 $\rho(A_{cl}) < 1$）是闭环渐近稳定的充要条件。

### 2.2 二次型与矩阵正定性

对于向量 $x = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$ 与对称矩阵 $Q \in \mathbb{R}^{n \times n}$：
- 二次型定义为标量：
  $$x^T Q x = \sum_{i=1}^n \sum_{j=1}^n Q_{ij} x_i x_j$$
- **严格正定（Positive Definite, 记作 $Q > 0$）**：对所有非零向量 $x \neq 0$，均有 $x^T Q x > 0$；等价于 $Q$ 的所有特征值严格大于 0，或者顺序主子式全大于 0。
- **半正定（Positive Semi-Definite, 记作 $Q \ge 0$）**：对所有向量 $x$，均有 $x^T Q x \ge 0$；等价于 $Q$ 的所有特征值均非负。
- **不定（Indefinite）**：存在向量使得二次型为正，同时也存在向量使得二次型为负；等价于 $Q$ 既有正特征值也有负特征值。

---

## 3. 逐节精读与推导

### 3.1 核心问题：为什么有限时域 MPC 可能失稳？

在前面各章中，我们看到 MPC 的优化问题定义在一个**有限时间窗口（Prediction Horizon $H_p$）**上：
$$V(k) = \sum_{j=1}^{H_p} \|y(k+j|k) - r(k+j)\|_Q^2 + \sum_{j=0}^{H_u-1} \|\Delta u(k+j|k)\|_R^2$$
在每个采样时刻 $k$，控制器求解一个使未来 $H_p$ 步性能最优的输入序列，并只将第一步 $u(k) = u(k|k)$ 施加给真实对象。

**致命的盲区（Short-Sightedness / Myopia）**：
优化器只关心时域窗口 $[k+1, k+H_p]$ 内的指标。它对第 $k+H_p+1$ 步及以后的事情**一无所知、毫不关心**！  
为了让当前窗口内的代价 $V(k)$ 极小化，控制器可能会采取某种急功近利的控制动作：在窗口内把输出强行拉低，却在窗口末端将系统状态推向一个极具破坏性、极不稳定的边缘状态（例如速度极大、剧烈震荡），只要这一恶果落在 $k+H_p$ 之后即可。而到了下一采样时刻 $k+1$，窗口向前滑动一步，控制器才发现“大祸临头”，不得不采取更加极端的控制动作进行补救，最终导致闭环状态彻底发散！

---

### 3.2 例子 6.1（Example 6.1）：纯延时系统的 MPC 致命失稳

Maciejowski 在第 167 页给出了一个极其经典、震撼的例子：**一个开环本身极其稳定（有限步死拍收敛）的系统，使用非对角正定加权矩阵的短时域 MPC 进行控制，闭环竟然严重发散！**

#### 3.2.1 被控对象与开环模型
考虑两个串联的纯离散延迟环节（Two pure delays in series）：
$$x_1(k+1) = u(k)$$
$$x_2(k+1) = x_1(k)$$
输出等于第二个延迟状态：$y(k) = x_2(k)$。  
写成状态空间矩阵形式：
$$A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 1 \end{bmatrix}$$
计算开环特征多项式：
$$\det(\lambda I - A) = \det\begin{bmatrix} \lambda & 0 \\ -1 & \lambda \end{bmatrix} = \lambda^2 - 0 = \lambda^2 = 0 \implies \lambda_1 = 0, \lambda_2 = 0$$
两个开环极点全在原点 $\lambda = 0$。如果令控制输入 $u(k) \equiv 0$，则：
$$x(1) = \begin{bmatrix} 0 \\ x_1(0) \end{bmatrix}, \quad x(2) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
系统在 2 步内完全死拍归零！开环极其稳定。

#### 3.2.2 MPC 控制器设置与目标函数
设置超短预测时域与控制时域：
$$H_p = 1, \quad H_u = 1$$
不惩罚控制量（$R = 0$），但选取一个严格正定的非对角状态加权矩阵 $Q$：
$$Q = \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix}$$
验证 $Q$ 的正定性：
- 顺序主子式 1：$\Delta_1 = 1 > 0$
- 顺序主子式 2：$\Delta_2 = \det(Q) = 1 \times 6 - 2 \times 2 = 6 - 4 = 2 > 0$
- 特征值满足 $\lambda^2 - 7\lambda + 2 = 0 \implies \lambda = \frac{7 \pm \sqrt{41}}{2} > 0$。  
$Q$ 是无可争议的严格对称正定矩阵！

在时刻 $k$，预测未来第 1 步的状态：
$$x(k+1|k) = A x(k) + B u(k) = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x_1(k) \\ x_2(k) \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} u(k) = \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix}$$

将预测状态代入单步性能指标二次型 $V(k) = x(k+1|k)^T Q x(k+1|k)$：
$$V(k) = \begin{bmatrix} u(k) & x_1(k) \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix} \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix}$$
逐步展开标量乘法：
$$\begin{aligned}
V(k) &= \begin{bmatrix} u(k) & x_1(k) \end{bmatrix} \begin{bmatrix} 1 \cdot u(k) + 2 \cdot x_1(k) \\ 2 \cdot u(k) + 6 \cdot x_1(k) \end{bmatrix} \\
&= u(k) \left( u(k) + 2 x_1(k) \right) + x_1(k) \left( 2 u(k) + 6 x_1(k) \right) \\
&= u(k)^2 + 2 u(k) x_1(k) + 2 u(k) x_1(k) + 6 x_1(k)^2 \\
&= u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2
\end{aligned}$$

#### 3.2.3 无约束最优解求导与闭环极点分析
为了最小化 $V(k)$，对未知决策变量 $u(k)$ 求一阶导数并令其为 0：
$$\frac{\partial V(k)}{\partial u(k)} = 2 u(k) + 4 x_1(k) = 0$$
解得最优控制律：
$$2 u(k) = -4 x_1(k) \implies u^*(k) = -2 x_1(k)$$
注意：二阶导数 $\frac{\partial^2 V}{\partial u(k)^2} = 2 > 0$，确实是严格极小值。

将最优控制律 $u(k) = -2 x_1(k)$ 代回真实工厂状态方程：
$$x_1(k+1) = u(k) = -2 x_1(k)$$
$$x_2(k+1) = x_1(k)$$
写成闭环状态方程：
$$x(k+1) = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix} x(k) = A_{cl} x(k)$$
计算闭环矩阵 $A_{cl}$ 的特征值：
$$\det(\lambda I - A_{cl}) = \det\begin{bmatrix} \lambda + 2 & 0 \\ -1 & \lambda \end{bmatrix} = (\lambda + 2)\lambda - 0 = \lambda(\lambda + 2) = 0$$
特征值为：
$$\lambda_1 = -2, \quad \lambda_2 = 0$$
**模长 $|\lambda_1| = |-2| = 2 > 1$！**  
闭环极点落在了单位圆之外！设初始状态为 $x(0) = [1, 0]^T$：
- $k = 0$: $x_1(0) = 1, x_2(0) = 0, u(0) = -2$
- $k = 1$: $x_1(1) = -2, x_2(1) = 1, u(1) = +4$
- $k = 2$: $x_1(2) = +4, x_2(2) = -2, u(2) = -8$
- $k = 3$: $x_1(3) = -8, x_2(3) = +4, u(3) = +16$
- $k = m$: $x_1(m) = (-2)^m \to \pm \infty$！

状态不仅剧烈正负交替振荡，而且幅值以公比 $2$ 呈指数几何级数发散！  
原书 Example 6.1 深刻揭示：**单纯的有限时域最优性，完全无法保证闭环稳定性。**

---

### 3.3 Mini-Tutorial 6：离散时间系统的 Lyapunov 稳定性理论

在现代控制理论中，证明系统稳定性的终极武器是俄国数学家 A. M. Lyapunov 于 1892 年提出的**第二方法（直接法）**。为了让零基础读者彻底理解，我们用最质朴的物理直觉展开。

#### 3.3.1 物理直觉：碗底的小球与广义能量
想象一个碗，碗底位于坐标原点 $(0,0)$。一个小球在碗内滚动：
1. **小球的高度（势能）**：除了碗底原点处势能为零外，在碗内的任何其他位置，高度和势能都严格大于零（正定性）。
2. **小球的摩擦损耗（能量耗散）**：随着小球运动，摩擦力不断消耗机械能。因此，在每一个时间步，系统的总机械能都必须**严格减少**（能量导数/差分为负）。
3. **必然归宿**：如果能量始终大于零且随时间单调递减，能量最终只能趋向于零，小球必然停在碗底原点。

在数学上，这个“广义能量”被称为 **Lyapunov 函数**，记作 $V(x)$。

#### 3.3.2 离散时间 Lyapunov 渐近稳定性定理
考虑自主非线性或线性离散时间系统：
$$x(k+1) = f(x(k)), \quad f(0) = 0$$
**定理（Discrete-Time Lyapunov Stability）**：  
若存在一个标量函数 $V(x): \mathbb{R}^n \to \mathbb{R}$，满足以下三个条件：
1. **连续正定性（Positive Definite）**：
   $$V(0) = 0, \quad \text{且对所有 } x \neq 0, \; V(x) > 0$$
   并存在正常数 $\alpha_1, \alpha_2$ 使得 $\alpha_1 \|x\|^2 \le V(x) \le \alpha_2 \|x\|^2$。
2. **差分严格负定（Strictly Negative Definite Difference）**：沿着系统的真实运动轨迹，定义单步差分：
   $$\Delta V(x(k)) \triangleq V(x(k+1)) - V(x(k)) = V(f(x(k))) - V(x(k))$$
   对所有 $x(k) \neq 0$，均有：
   $$\Delta V(x(k)) \le - \gamma(\|x(k)\|) < 0$$
   其中 $\gamma(\cdot)$ 为严格正定函数（例如 $\gamma(\|x\|) = c \|x\|^2, c > 0$）。
3. **径向无界（Radially Unbounded）**：当 $\|x\| \to \infty$ 时，$V(x) \to \infty$。

**结论**：系统的平衡点 $x = 0$ 是**全局渐近稳定的（Globally Asymptotically Stable）**，即对任意初始状态 $x(0)$，均有：
$$\lim_{k \to \infty} x(k) = 0$$

#### 3.3.3 MPC 稳定性证明的绝妙构想
在受约束的 MPC 中，由于存在控制量饱和和状态约束，闭环系统本质上是一个**高度复杂的非线性切换系统**，无法直接用传递函数求极点。  
先驱学者们（Keerthi & Gilbert 1988, Mayne 等）提出了一个石破天惊的思想：  
**何不直接将 MPC 求解出来的“最优目标函数值” $V^*(x(k))$ 作为系统的 Lyapunov 函数？！**
- 显然，只要加权矩阵 $Q > 0$，就有 $V^*(x(k)) > 0$（正定）。
- 如果我们能证明：随着时间从 $k$ 推进到 $k+1$，下一个时刻求解出的最优值 $V^*(x(k+1))$ 一定比当前时刻的最优值 $V^*(x(k))$ 小，即：
  $$V^*(x(k+1)) - V^*(x(k)) \le - x(k)^T Q x(k) < 0$$
  那么根据 Lyapunov 定理，MPC 闭环系统就**必然渐近稳定**！

---

### 3.4 第 6.1 节：终端等式约束法与 Keerthi-Gilbert 定理

为了确保上述 Lyapunov 差分 $\Delta V^*$ 严格小于零，最直接、最早被严格证明的方法是在优化问题中强制加入一个**终端等式约束（Terminal Equality Constraint）**。

#### 3.4.1 控制问题描述与约束设定
在采样时刻 $k$，被控对象为：
$$x(k+1) = A x(k) + B u(k)$$
状态和控制受凸集约束：$u(k+j|k) \in \mathbb{U}, x(k+j|k) \in \mathbb{X}$。  
优化目标函数：
$$V(k) = \sum_{j=0}^{H_p-1} \left[ x(k+j|k)^T Q x(k+j|k) + u(k+j|k)^T R u(k+j|k) \right]$$
其中 $Q > 0, R > 0$。  
**施加终端等式约束**：
$$x(k+H_p|k) = 0$$
即要求控制器规划的未来控制序列，必须在预测窗口末端将系统状态**精确镇定到原点**！

#### 3.4.2 定理 6.1（Keerthi-Gilbert 1988）的严密代数推导

**【推导目标】**：证明受控闭环系统渐近稳定，且最优代价函数 $V^*(k)$ 单调递减。

**步骤 1：分析时刻 $k$ 的最优解**  
在时刻 $k$，求解该受约束 QP 问题，得到最优控制序列：
$$\mathbf{u}^*(k) = \{ u^*(k|k), u^*(k+1|k), u^*(k+2|k), \dots, u^*(k+H_p-1|k) \}$$
对应的最优状态轨迹为：
$$\mathbf{x}^*(k) = \{ x^*(k|k), x^*(k+1|k), \dots, x^*(k+H_p|k) \}$$
其中 $x^*(k|k) = x(k)$ 是当前测量状态，且满足终端约束：
$$x^*(k+H_p|k) = 0$$
对应的最优目标函数值记为 $V^*(k)$：
$$V^*(k) = \sum_{j=0}^{H_p-1} \left[ {x^*(k+j|k)}^T Q x^*(k+j|k) + {u^*(k+j|k)}^T R u^*(k+j|k) \right]$$
根据 receding horizon 原理，真实施加给系统的控制量为第一步：
$$u(k) = u^*(k|k)$$
系统到达下一真实状态：
$$x(k+1) = A x(k) + B u^*(k|k) \equiv x^*(k+1|k)$$

**步骤 2：在时刻 $k+1$ 构造一个“次优但绝对可行”的候选控制序列**  
时间推进到 $k+1$，真实状态变为 $x(k+1)$。控制器需要求解一个新的优化问题，窗口向前滑动到 $[k+1, k+1+H_p]$。  
此时，我们构造一个**候选控制序列（Feasible Candidate Sequence）** $\tilde{\mathbf{u}}(k+1)$：  
将时刻 $k$ 算出的后 $H_p-1$ 个控制量原封不动地继承下来，而在最后补上一个零输入（因为原点是平衡点，$u=0$ 保持 $x=0$）：
$$\tilde{u}(k+1+j | k+1) \triangleq \begin{cases}
u^*(k+1+j | k), & j = 0, 1, \dots, H_p-2 \\
0, & j = H_p - 1
\end{cases}$$
将该候选控制序列代入状态方程预测未来状态 $\tilde{x}$：
- 对于 $j = 0, \dots, H_p-1$：
  $$\tilde{x}(k+1+j | k+1) = x^*(k+1+j | k)$$
- 对于终端状态 $j = H_p$：
  $$\tilde{x}(k+1+H_p | k+1) = A \tilde{x}(k+H_p | k+1) + B \tilde{u}(k+H_p | k+1) = A \cdot 0 + B \cdot 0 = 0$$
**绝妙结论**：这个构造出来的候选控制序列**完全满足所有输入约束和终端等式约束**！它是一个合法的可行解。

**步骤 3：计算候选控制序列对应的代价 $\tilde{V}(k+1)$**  
将候选序列代入目标函数公式：
$$\begin{aligned}
\tilde{V}(k+1) &= \sum_{j=0}^{H_p-1} \left[ \tilde{x}(k+1+j|k+1)^T Q \tilde{x}(k+1+j|k+1) + \tilde{u}(k+1+j|k+1)^T R \tilde{u}(k+1+j|k+1) \right] \\
&= \sum_{j=0}^{H_p-2} \left[ {x^*(k+1+j|k)}^T Q x^*(k+1+j|k) + {u^*(k+1+j|k)}^T R u^*(k+1+j|k) \right] \\
&\quad + \underbrace{\tilde{x}(k+H_p|k+1)^T Q \tilde{x}(k+H_p|k+1)}_{0^T Q 0 = 0} + \underbrace{\tilde{u}(k+H_p|k+1)^T R \tilde{u}(k+H_p|k+1)}_{0^T R 0 = 0}
\end{aligned}$$
因此最后一项精确为 0！展开式只剩下前 $H_p-1$ 项：
$$\tilde{V}(k+1) = \sum_{j=0}^{H_p-2} \left[ {x^*(k+1+j|k)}^T Q x^*(k+1+j|k) + {u^*(k+1+j|k)}^T R u^*(k+1+j|k) \right]$$

**步骤 4：对比 $\tilde{V}(k+1)$ 与 $V^*(k)$**  
回顾时刻 $k$ 的最优代价 $V^*(k)$：
$$V^*(k) = \left[ {x^*(k|k)}^T Q x^*(k|k) + {u^*(k|k)}^T R u^*(k|k) \right] + \sum_{j=1}^{H_p-1} \left[ {x^*(k+j|k)}^T Q x^*(k+j|k) + {u^*(k+j|k)}^T R u^*(k+j|k) \right]$$
将求和下标变量替换为 $i = j-1$（即 $j = i+1$）：
$$\sum_{j=1}^{H_p-1} [\dots] = \sum_{i=0}^{H_p-2} \left[ {x^*(k+1+i|k)}^T Q x^*(k+1+i|k) + {u^*(k+1+i|k)}^T R u^*(k+1+i|k) \right]$$
这正好就是我们算出的 $\tilde{V}(k+1)$！  
因此，两者的精确代数关系为：
$$V^*(k) = \left[ x(k)^T Q x(k) + u^*(k|k)^T R u^*(k|k) \right] + \tilde{V}(k+1)$$
移项得到：
$$\tilde{V}(k+1) = V^*(k) - \left[ x(k)^T Q x(k) + u^*(k|k)^T R u^*(k|k) \right]$$

**步骤 5：利用最优性原理建立单调递减不等式**  
在时刻 $k+1$，真实的 MPC 控制器会求解 QP 寻找**全局最优值** $V^*(k+1)$。  
因为 $V^*(k+1)$ 是所有可行解中的最小值，而 $\tilde{V}(k+1)$ 只是一个特定的可行解，所以**最优值必然小于或等于任何可行解的代价值**：
$$V^*(k+1) \le \tilde{V}(k+1)$$
将代数关系代入：
$$V^*(k+1) \le V^*(k) - \left[ x(k)^T Q x(k) + u^*(k|k)^T R u^*(k|k) \right]$$
整理成 Lyapunov 差分形式：
$$\Delta V^*(k) \triangleq V^*(k+1) - V^*(k) \le - \left[ x(k)^T Q x(k) + u^*(k|k)^T R u^*(k|k) \right]$$
因为 $Q > 0, R > 0$：
- 当 $x(k) \neq 0$ 时，$x(k)^T Q x(k) > 0$，故 $\Delta V^*(k) < 0$！
- 目标函数的最优值 $V^*(k)$ 随着采样步的推进**严格单调递减**！
- 根据离散 Lyapunov 稳定性定理，$x(k) \to 0$ 当 $k \to \infty$。  
**证毕！**

#### 3.4.3 终端等式约束的代价与局限性（图 6.1 分析）
原书 Figure 6.1 指出，虽然终端等式约束在理论上极其优美，但在实际工程中存在严重缺陷：
1. **可行域急剧缩小（Controllable Basin Contraction）**：  
   要在短短 $H_p$ 步内将状态强行压制到原点 $x(k+H_p) = 0$，而且控制量还受到饱和约束 $|u| \le u_{max}$，这要求初始状态必须离原点非常近！如果初始状态稍大，在 $H_p$ 步内无论如何施加最大控制量也无法到达原点，导致 QP 问题**无解（Infeasible）**。
2. **控制动作过于激进（Excessive Control Activity）**：  
   强行要求末端为零会导致控制量在窗口前端频繁打到极限幅值，系统鲁棒性较差。

---

### 3.5 第 6.2 节：无限预测时域法（Infinite Horizons）

为了摆脱终端等式约束带来的可行域狭窄问题，Rawlings 和 Muske (1993) 提出了**无限预测时域法**：令预测时域 $H_p \to \infty$。

#### 3.5.1 稳定对象（Section 6.2.1）：离散 Lyapunov 方程与等价有限时域
考虑稳定的被控对象（所有特征值 $|\lambda_i(A)| < 1$）。  
设控制时域有限为 $H_u$。即在 $H_u$ 步之后，不再施加任何新的控制动作：
$$u(k+j|k) = 0, \quad \forall j \ge H_u$$
（若为增量模型，则 $\Delta u(k+j|k) = 0 \implies u(k+j|k) = u(k+H_u-1|k)$）。

目标函数为无限时域求和：
$$V(k) = \sum_{j=1}^{\infty} y(k+j|k)^T Q y(k+j|k) + \sum_{j=0}^{H_u-1} u(k+j|k)^T R u(k+j|k)$$
将求和拆分为两部分：前 $H_u$ 步的主动控制区，以及 $H_u$ 步之后的**自主衰减尾部（Autonomous Tail）**：
$$V(k) = \sum_{j=1}^{H_u} y(k+j|k)^T Q y(k+j|k) + \sum_{j=0}^{H_u-1} u(k+j|k)^T R u(k+j|k) + V_{tail}(k)$$
尾部代价定义为：
$$V_{tail}(k) \triangleq \sum_{i=H_u}^{\infty} x(k+i+1|k)^T C^T Q C x(k+i+1|k)$$
在 $i \ge H_u$ 之后，由于 $u=0$，状态方程退化为自发衰减方程：
$$x(k+H_u + m | k) = A^m x(k+H_u|k), \quad m = 1, 2, \dots$$
将自主轨迹代入尾部代价：
$$\begin{aligned}
V_{tail}(k) &= \sum_{m=1}^{\infty} \left[ A^m x(k+H_u|k) \right]^T C^T Q C \left[ A^m x(k+H_u|k) \right] \\
&= x(k+H_u|k)^T \left( \sum_{m=1}^{\infty} (A^m)^T C^T Q C A^m \right) x(k+H_u|k) \\
&\triangleq x(k+H_u|k)^T \bar{Q} x(k+H_u|k)
\end{aligned}$$
其中终端权重矩阵定义为无穷矩阵级数：
$$\bar{Q} \triangleq \sum_{m=1}^{\infty} (A^m)^T C^T Q C A^m = A^T \left( \sum_{l=0}^\infty (A^l)^T C^T Q C A^l \right) A$$

**离散 Lyapunov 方程的推导**：  
令 $P = \sum_{l=0}^\infty (A^l)^T C^T Q C A^l$。考虑差分：
$$\begin{aligned}
A^T P A - P &= A^T \left( \sum_{l=0}^\infty (A^l)^T C^T Q C A^l \right) A - \sum_{l=0}^\infty (A^l)^T C^T Q C A^l \\
&= \sum_{l=0}^\infty (A^{l+1})^T C^T Q C A^{l+1} - \left( C^T Q C + \sum_{l=1}^\infty (A^l)^T C^T Q C A^l \right) \\
&= \sum_{m=1}^\infty (A^m)^T C^T Q C A^m - C^T Q C - \sum_{l=1}^\infty (A^l)^T C^T Q C A^l \\
&= - C^T Q C
\end{aligned}$$
移项即得著名的**离散 Lyapunov 方程（Discrete Lyapunov Equation）**：
$$A^T P A - P + C^T Q C = 0$$
注意到 $\bar{Q} = A^T P A = P - C^T Q C$，因此 $\bar{Q}$ 满足：
$$A^T \bar{Q} A - \bar{Q} + A^T C^T Q C A = 0$$
（在原书中，若定义尾部从 $i=H_u$ 计入，则 $\bar{Q}$ 直接满足 $A^T \bar{Q} A - \bar{Q} + C^T Q C = 0$）。

**震撼结论**：  
无限时域优化问题**完全等价于**一个预测时域仅为 $H_p = H_u$ 的**有限时域 QP 问题**，只需在末端加上由离散 Lyapunov 方程精确算出的终端权重 $\bar{Q}$：
$$V(k) = \sum_{j=1}^{H_u} y(k+j|k)^T Q y(k+j|k) + \sum_{j=0}^{H_u-1} u(k+j|k)^T R u(k+j|k) + x(k+H_u|k)^T \bar{Q} x(k+H_u|k)$$
这样既获得了无限时域的天然稳定性保证，又只需要求解极小规模的有限变量 QP！

---

#### 3.5.2 不稳定对象（Section 6.2.2）：模态分解与不稳定模态终端等式约束
如果被控对象本身是开环不稳定的（$A$ 存在特征值 $|\lambda| \ge 1$），则自主尾部矩阵级数 $\sum_{m=1}^\infty (A^m)^T C^T Q C A^m$ 会彻底发散至无穷大！  
Maciejowski 在第 176–178 页给出了精妙的**模态分解法（Modal Decomposition）**（原书 Figure 6.2）。

**步骤 1：Jordan/特征模态解耦**  
对状态矩阵 $A$ 进行特征值分解：
$$A = W J W^{-1}$$
其中 $W$ 的列向量为特征向量。将特征值按照稳定模态（$|\lambda| < 1$）与不稳定模态（$|\lambda| \ge 1$）排序分块：
$$J = \begin{bmatrix} J_s & 0 \\ 0 & J_u \end{bmatrix}, \quad W = \begin{bmatrix} W_s & W_u \end{bmatrix}, \quad W^{-1} = \begin{bmatrix} (W^{-1})_s \\ (W^{-1})_u \end{bmatrix}$$
定义模态坐标：
$$z(k) = W^{-1} x(k) = \begin{bmatrix} z_s(k) \\ z_u(k) \end{bmatrix}$$
状态方程在模态坐标下完全解耦：
$$z_s(k+1) = J_s z_s(k) + (W^{-1})_s B u(k)$$
$$z_u(k+1) = J_u z_u(k) + (W^{-1})_u B u(k)$$

**步骤 2：对不稳定模态施加终端等式约束**  
对于不稳定子系统 $z_u$，在 $H_u$ 步之后 $u=0$，其动态为 $z_u(k+H_u+m) = J_u^m z_u(k+H_u)$。若 $z_u(k+H_u) \neq 0$，则当 $m \to \infty$ 时，$J_u^m \to \infty$，目标函数必然发散！  
为了让无限时域目标函数有限，控制器**必须且只需**在 $k+H_u$ 处将不稳定模态精确清零：
$$z_u(k+H_u|k) = 0 \iff (W^{-1})_u x(k+H_u|k) = 0$$

**步骤 3：稳定模态子系统的离散 Lyapunov 终端权重**  
在不稳定模态被强制清零后，剩余的稳定模态 $z_s$ 自然衰减，其无限尾部代价满足稳定的离散 Lyapunov 方程：
$$J_s^T \bar{\Pi} J_s - \bar{\Pi} + Q_{modal, s} = 0$$
然后反变换回原始状态坐标：
$$\bar{Q}_s = (W^{-1})_s^T \bar{\Pi} (W^{-1})_s$$
最终，受约束 MPC 优化问题为：
$$\min_{\mathbf{u}} \sum_{j=1}^{H_u} y(k+j|k)^T Q y(k+j|k) + \sum_{j=0}^{H_u-1} u(k+j|k)^T R u(k+j|k) + x(k+H_u|k)^T \bar{Q}_s x(k+H_u|k)$$
$$\text{s.t.} \quad (W^{-1})_u x(k+H_u|k) = 0, \quad u \in \mathbb{U}$$
这完美兼顾了不稳定对象的严格镇定与稳定模态的大范围可行域！

---

### 3.6 第 6.3 节：伪代数 Riccati 方程法（FARE）

在未加约束的无限时域线性二次型（LQ）问题中，最优状态反馈增益为：
$$K_\infty = (R + B^T P_\infty B)^{-1} B^T P_\infty A$$
其中 $P_\infty$ 是离散代数 Riccati 方程（DARE）的唯一正定解：
$$P_\infty = A^T P_\infty A - A^T P_\infty B (R + B^T P_\infty B)^{-1} B^T P_\infty A + Q$$
而在有限时域问题中，$P_j$ 服从向后迭代的 Riccati 差分方程：
$$P_{j+1} = A^T P_j A - A^T P_j B (R + B^T P_j B)^{-1} B^T P_j A + Q$$

Bitmead, Gevers 和 Wertz (1990) 提出了**伪代数 Riccati 方程法（Fake Algebraic Riccati Equation, FARE）**。  
定义虚拟加权矩阵：
$$Q_j \triangleq Q - (P_{j+1} - P_j)$$
代入差分方程后，形式上变成了代数 Riccati 方程：
$$P_j = A^T P_j A - A^T P_j B (R + B^T P_j B)^{-1} B^T P_j A + Q_j$$
**单调性定理（Monotonicity Property）**：  
若起始终端权重 $P_0$ 选得足够大，使得 $P_1 \le P_0$，则对于所有步数均有单调递减：
$$P_{j+1} \le P_j \implies Q_j \ge Q > 0$$
此时虚拟 Riccati 方程保证由有限步算出的增益 $K_j$ 施加给闭环系统后必定渐近稳定！  
特别地，若令 $P_0 \to \infty$（信息滤波形式中取 $P_0^{-1} = 0$），恰好等价于施加终端等式约束 $x(k+H_p) = 0$！

---

### 3.7 第 6.4 节：Youla 参数化方法（Dual Youla / Q-Parameterization）

如果被控对象本身是稳定的，控制理论中存在一个极为深刻的结论：**所有使闭环稳定的控制器的全体集合，可以用一个稳定的自由传递函数参数 $Q(z)$ 完全线性参数化！**（原书 Figure 6.3）。

#### 3.7.1 内部模型控制（IMC）与 Youla 参数结构
如图 6.3 所示：
- 真实工厂传递函数为 $P(z)$。
- 控制器内部包含工厂的名义模型 $P(z)$。
- 测量输出为 $y(z) = P(z) u(z) + d(z)$，其中 $d(z)$ 为未建模扰动。
- 模型预测输出为 $y_m(z) = P(z) u(z)$。
- 两者相减得到创新扰动估计：
  $$e(z) = y(z) - y_m(z) = d(z)$$
- Youla 控制器根据该扰动生成控制动作：
  $$u(z) = - Q(z) e(z) = - Q(z) [y(z) - P(z) u(z)]$$

#### 3.7.2 闭环传递函数的仿射性（Affinity）
整理控制方程：
$$u(z) = - Q(z) y(z) + Q(z) P(z) u(z) \implies [I - Q(z) P(z)] u(z) = - Q(z) y(z)$$
$$u(z) = - [I - Q(z) P(z)]^{-1} Q(z) y(z) \triangleq - K(z) y(z)$$
将 $u(z) = - Q(z) d(z)$ 代回输出方程：
$$y(z) = P(z) u(z) + d(z) = - P(z) Q(z) d(z) + d(z) = [I - P(z) Q(z)] d(z)$$
**惊人的数学特性**：闭环系统从扰动 $d$ 到输出 $y$ 以及控制 $u$ 的传递函数：
$$T_{yd}(z) = I - P(z) Q(z), \quad T_{ud}(z) = - Q(z)$$
关于未知参数 $Q(z)$ 是**完全线性的（仿射的，Affine）**！

#### 3.7.3 FIR 参数化与凸 QP 优化（Example 6.4）
将无限维的稳定传递函数 $Q(z)$ 用阶数为 $n$ 的有限冲激响应（FIR）滤波器截断参数化：
$$Q(z) = H(0) + H(1) z^{-1} + H(2) z^{-2} + \dots + H(n) z^{-n}$$
时域差分方程为：
$$u(k+j|k) = - \sum_{i=0}^n H(i) d(k+j-i|k)$$
- 决策变量不再是每个时刻的控制量 $u$，而是 FIR 滤波器的脉冲响应系数 $\mathbf{H} = [H(0), H(1), \dots, H(n)]^T \in \mathbb{R}^{n+1}$！
- 由于 $d(k+j-i|k)$ 仅由当前和过去的测量值决定，与未来的 $H(i)$ 独立，因此未来的预测状态和输出是决策变量 $\mathbf{H}$ 的**纯线性函数**。
- MPC 的二次型目标函数代入后，化为一个关于 $\mathbf{H}$ 的**严格凸二次规划（Strictly Convex QP）**！
- **稳定性终极保证**：因为工厂 $P(z)$ 本身稳定，而 FIR 滤波器 $Q(z)$ 的所有极点均位于原点 $z=0$（绝对稳定），所以**无论优化算法选出什么样的参数 $H(i)$，无论控制量如何打到饱和约束，闭环系统在结构上绝对保证内部渐近稳定！**

---

## 4. 本章综合推导链

本章所有定理与方法之间的逻辑演化链条如下图所示：

```
                    【核心矛盾】
      短时域 MPC 只顾眼前最优，导致终端状态恶化
               (Example 6.1: 闭环极点 lambda = -2)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
【方案一：空间硬截断】            【方案二：时间轴延伸】
强制要求末端到达原点             将时域拓展至无穷大
  x(k+Hp|k) = 0                     Hp -> infinity
         │                               │
         ▼                               ▼
【Keerthi-Gilbert 定理】         【解耦两段式处理】
候选序列尾部补零可行            前 Hu 步优化，后部 u = 0 自主衰减
V*(k+1) - V*(k) <= -x^T Q x              │
严格充当 Lyapunov 函数                   │
保证渐近稳定                             ├──────────────────────────┐
(缺陷：可行域极小)                       │                          │
                                         ▼                          ▼
                                   【稳定对象】               【不稳定对象】
                               自主尾部收敛              自主尾部发散
                                     │                          │
                                     ▼                          ▼
                           【离散 Lyapunov 方程】     【模态分解解耦】
                           A^T Q_bar A - Q_bar + Q = 0   z = W^{-1} x = [z_s; z_u]
                           将无限尾部等价折算为终端权重        │
                           V_term = x^T Q_bar x               ├─────────────┐
                           有限 QP，自然稳定                  │             │
                                                              ▼             ▼
                                                        【不稳定模态】 【稳定模态】
                                                        终端等式清零  Lyapunov 权重
                                                        W_u x = 0     Q_bar_s
                                                              │             │
                                                              └──────┬──────┘
                                                                     ▼
                                                        兼具大可行域与严格稳定
                                                                     │
         ┌───────────────────────────────────────────────────────────┘
         ▼
【方案三：Riccati 倒向演化】
  伪代数 Riccati 方程 (FARE)
  P_{j+1} <= P_j 保证 Q_j >= Q > 0 单调收敛
  (Exercise 6.7: 证明 Q >= 0 仅为充分条件而非必要条件)
         │
         ▼
【方案四：系统构型变换 (Youla 参数化)】
  IMC 双回路控制，自由参数 Q(z) 采用 FIR 截断
  闭环对 Q(z) 严格仿射，QP 保持凸性
  结构性保证内部绝对稳定！
```

---

## 5. Python 实现与代码架构解析

本章全部代码均保存在目录 `/home/kuang/Desktop/mpc/ch06_stability/` 下，共包含 7 个独立 Python 脚本。每个脚本均配有完备的数学推导注释与可视化图表生成。

### 脚本索引与功能对照表

| 脚本文件名 | 对应小节 / 例题 | 核心算法与理论要点 | 生成图表文件 |
| :--- | :--- | :--- | :--- |
| `ch6_example6_1_instability.py` | Section 6.1 / Ex 6.1 | 纯延迟系统短时域 MPC 失稳复现；极点 $\lambda = -2$ 指数发散；加入终端约束死拍修复 | `ch6_example6_1_instability.png` |
| `ch6_terminal_equality_constraint.py` | Section 6.1 / Theorem 6.1 | 双积分系统终端等式约束 MPC；严格验证 $\Delta V^* \le -\ell(x,u)$ 离散 Lyapunov 单调递减；单纯形可控可行域边界计算 | `ch6_terminal_constraint_lyapunov.png` |
| `ch6_infinite_horizon_lyapunov.py` | Section 6.2.1 / Ex 6.5 | 离散 Lyapunov 方程 $A^T \bar{Q} A - \bar{Q} + C^T Q C = 0$ 求解；精确终端权重 $\bar{Q} = 100/19$；死拍增益 $u=-1.8x$；有限时域精确等价 | `ch6_infinite_horizon_lyapunov.png` |
| `ch6_unstable_infinite_horizon.py` | Section 6.2.2 / Ex 6.4 | 不稳定系统模态分解 $A = W J W^{-1}$；不稳定模态终端等式约束 $W_u x(k+H_u) = 0$；稳定模态 Lyapunov 权重；相平面轨迹收敛 | `ch6_unstable_infinite_horizon.png` |
| `ch6_fake_riccati_fare.py` | Section 6.3 / Ex 6.7 | 倒向 Riccati 迭代与 FARE 方程；增益 $K = [0, 32/17]$，闭环极点 $\pm 0.4851$ 稳定；推导不定矩阵 $Q_{N-1} = \text{diag}(-48, 12.24)$（证明充分非必要性） | `ch6_fake_riccati_fare.png` |
| `ch6_youla_parameterization.py` | Section 6.4 / Ex 6.4 | 稳定对象 Youla-Kucera 参数化；FIR 脉冲响应系数 $\mathbf{H}$ 优化；凸 QP 求解；阶跃扰动抑制；结构内部绝对稳定 | `ch6_youla_parameterization.png` |
| `ch6_exercise6_2_and_6_3.py` | Exercises 6.2 & 6.3 | 纯延迟系统时域拓展对比：(a) $H_u=1 \implies \lambda=-1/6$；(b) $H_u=2 \implies \lambda=-2/3$；对角加权矩阵 $Q=\text{diag}(1,6) \implies u=0$ 死拍归零 | `ch6_exercise6_2_and_6_3.png` |

---

## 6. 数值实验与验证

### 实验 1：短时域 MPC 失稳与终端约束修复（基于 `ch6_example6_1_instability.py`）
- **实验目的**：复现原书 Example 6.1，观察正定非对角加权矩阵导致开环稳定系统发生闭环指数级正负振荡发散的现象；验证终端等式约束 $x(k+H_p) = 0$ 的镇定效果。
- **仿真参数**：$A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}, B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, Q = \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix}, H_p = 1, H_u = 1$，初始状态 $x(0) = [1, 0]^T$。
- **实验结果**：
  - 未加终端约束时，控制增益为 $u(k) = -2 x_1(k)$，闭环极点为 $\lambda = -2, 0$。状态序列为 $x_1 = [1, -2, 4, -8, 16, \dots]$，在 10 步内幅值暴涨至超过 500！
  - 施加终端约束 $x(k+1|k) = 0$ 时，由于 $x_1(k+1) = u(k) = 0$，强制要求 $u(k) = 0$。闭环极点变为双重原点 $\lambda = 0, 0$，系统在 2 步内完全死拍归零！

### 实验 2：Lyapunov 函数单调递减验证（基于 `ch6_terminal_equality_constraint.py`）
- **实验目的**：对带输入饱和约束 $|u| \le 1.0$ 的双积分系统，逐采样步检验 Keerthi-Gilbert 定理的不等式 $\Delta V^*(k) \le - [x^T Q x + u^T R u]$ 是否恒成立。
- **实验结果**：在全部仿真步长中，布尔判定变量 `is_lyapunov_decreasing` 均为 `True`。随着时间推进，最优目标代价值 $V^*(k)$ 呈现严格单调递减曲线，光滑收敛至 0。

### 实验 3：模态解耦镇定不稳定对象（基于 `ch6_unstable_infinite_horizon.py`）
- **实验目的**：对包含开环不稳定极点 $\lambda_1 = 1.25$ 与稳定极点 $\lambda_2 = 0.60$ 的二阶系统，验证模态分解终端约束 $W_u x(k+H_u) = 0$ 的镇定能力。
- **实验结果**：
  - 朴素短时域 MPC（无终端约束）无法压制不稳定模态，状态 $x_1(k)$ 在 10 步后彻底发散至无穷大。
  - 模态解耦 MPC 成功将不稳定模态坐标 $z_u(k)$ 引导收敛至 0，稳定模态在 Lyapunov 终端权重 $\bar{Q}_s$ 作用下平滑归零，相平面轨迹完美收敛至原点。

### 实验 4：FARE 条件的充分非必要性检验（基于 `ch6_fake_riccati_fare.py`）
- **实验目的**：复现 Exercise 6.7，验证存在负特征值的不定加权矩阵 $Q_{N-1}$ 是否仍能产生渐近稳定的闭环系统。
- **实验结果**：算出的状态加权矩阵为 $Q_{N-1} = \text{diag}(-48, 12.2353)$，特征值为 $-48$ 和 $+12.24$（严格不定）。但闭环矩阵特征值为 $\pm \sqrt{4/17} \approx \pm 0.4851$，谱半径远小于 1，闭环系统严格渐近稳定！数值上无可辩驳地证实了 Lyapunov/Riccati 条件的“非必要性”。

---

## 7. 课后全部习题详细解答（Exercise 6.1 至 6.7）

本节严格按照用户要求的 **3-Part Template（【解题思路】、【逐步推导】、【最终结论】）**，对第 185–187 页全部 7 道习题进行**全宇宙最详尽、零跳步**的逐行代数推导。

---

### Exercise 6.1

#### 【解题思路】
本题要求使用 MATLAB/Python 仿真 Example 6.1 中给出的模型与控制律，验证闭环系统的不稳定性。  
核心方法是建立闭环状态空间模型 $A_{cl} = A - B K$，并求解在非零初始状态（如 $x(0) = [1, 0]^T$）下的自由零输入响应。

#### 【逐步推导】
1. **开环系统矩阵**：
   $$A = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad D = [0]$$
2. **最优无约束反馈增益**：
   在 Example 6.1 中，通过最小化单步预测代价 $V(k) = x(k+1)^T \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix} x(k+1)$，求导得到：
   $$u(k) = -2 x_1(k) = - \begin{bmatrix} 2 & 0 \end{bmatrix} x(k) \implies K = \begin{bmatrix} 2 & 0 \end{bmatrix}$$
3. **闭环状态转移矩阵计算**：
   $$A_{cl} = A - B K = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} - \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} - \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix}$$
4. **特征值与谱半径分析**：
   $$\det(\lambda I - A_{cl}) = \det\begin{bmatrix} \lambda + 2 & 0 \\ -1 & \lambda \end{bmatrix} = \lambda(\lambda + 2) = 0$$
   特征值为 $\lambda_1 = -2, \lambda_2 = 0$。  
   由于 $|\lambda_1| = 2 > 1$，闭环系统是不稳定的。
5. **时域状态递推展开**：
   取初始条件 $x(0) = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$：
   - $k=1$: $x(1) = A_{cl} x(0) = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}$
   - $k=2$: $x(2) = A_{cl} x(1) = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} -2 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \end{bmatrix}$
   - $k=3$: $x(3) = A_{cl} x(2) = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 4 \\ -2 \end{bmatrix} = \begin{bmatrix} -8 \\ 4 \end{bmatrix}$
   - 一般通项：$x_1(k) = (-2)^k, \quad x_2(k) = (-2)^{k-1}$ ($k \ge 1$)。  
   振荡发散速度为 $2^k$，验证完毕。

#### 【最终结论】
闭环系统矩阵为 $A_{cl} = \begin{bmatrix} -2 & 0 \\ 1 & 0 \end{bmatrix}$，极点为 $\lambda = -2$ 和 $\lambda = 0$。时域响应呈正负交替的几何指数发散，严格证实了短时域 MPC 的闭环失稳。

---

### Exercise 6.2

#### 【解题思路】
仍考虑 Example 6.1 中的纯延迟系统与加权矩阵 $Q = \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix}$，但将预测时域延长为 $H_p = 2$：
$$V(k) = x(k+1|k)^T Q x(k+1|k) + x(k+2|k)^T Q x(k+2|k)$$
分别在以下两种控制时域设定下推导无约束最优控制律，并判定闭环稳定性：
- (a) $H_u = 1$：假设未来输入恒定，$u(k+1|k) = u(k|k)$。
- (b) $H_u = 2$：未来两个输入 $u(k|k)$ 和 $u(k+1|k)$ 独立自由优化。

#### 【逐步推导】

**预测状态公式准备**：
已知 $x_1(k+1) = u(k), x_2(k+1) = x_1(k)$。
- 未来第 1 步状态：
  $$x(k+1|k) = \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix}$$
- 未来第 2 步状态：
  $$x_1(k+2|k) = u(k+1|k), \quad x_2(k+2|k) = x_1(k+1|k) = u(k)$$
  $$x(k+2|k) = \begin{bmatrix} u(k+1|k) \\ u(k) \end{bmatrix}$$

第一项代价计算（与 Example 6.1 相同）：
$$x(k+1|k)^T Q x(k+1|k) = u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2$$

第二项代价的一般二次型展开：
$$\begin{aligned}
x(k+2|k)^T Q x(k+2|k) &= \begin{bmatrix} u(k+1|k) & u(k) \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 2 & 6 \end{bmatrix} \begin{bmatrix} u(k+1|k) \\ u(k) \end{bmatrix} \\
&= u(k+1|k)^2 + 4 u(k+1|k) u(k) + 6 u(k)^2
\end{aligned}$$

---

**求解 (a)：当 $H_u = 1$ 时**  
此时约束 $u(k+1|k) = u(k)$。将此代入第二项代价：
$$\begin{aligned}
x(k+2|k)^T Q x(k+2|k) &= u(k)^2 + 4 u(k) \cdot u(k) + 6 u(k)^2 \\
&= (1 + 4 + 6) u(k)^2 = 11 u(k)^2
\end{aligned}$$
总目标函数为两项相加：
$$\begin{aligned}
V(k) &= \left( u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2 \right) + 11 u(k)^2 \\
&= 12 u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2
\end{aligned}$$
对单个决策变量 $u(k)$ 求一阶导数并令其为 0：
$$\frac{\partial V(k)}{\partial u(k)} = 24 u(k) + 4 x_1(k) = 0$$
解得控制律：
$$24 u(k) = -4 x_1(k) \implies u(k) = - \frac{4}{24} x_1(k) = - \frac{1}{6} x_1(k)$$
将 $u(k) = - \frac{1}{6} x_1(k)$ 代入闭环系统：
$$x_1(k+1) = - \frac{1}{6} x_1(k), \quad x_2(k+1) = x_1(k)$$
闭环状态转移矩阵为：
$$A_{cl} = \begin{bmatrix} -1/6 & 0 \\ 1 & 0 \end{bmatrix}$$
计算特征值：
$$\det(\lambda I - A_{cl}) = \lambda \left( \lambda + \frac{1}{6} \right) = 0 \implies \lambda_1 = - \frac{1}{6} \approx -0.1667, \quad \lambda_2 = 0$$
由于 $|\lambda_1| = \frac{1}{6} < 1$，**闭环系统严格渐近稳定！**

---

**求解 (b)：当 $H_u = 2$ 时**  
此时 $u(k)$ 与 $u(k+1)$ 均为独立决策变量。总目标函数为：
$$V(k) = u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2 + u(k+1)^2 + 4 u(k+1) u(k) + 6 u(k)^2$$
合并关于 $u(k)$ 的同类项（$u(k)^2 + 6 u(k)^2 = 7 u(k)^2$）：
$$V(k) = 7 u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2 + u(k+1)^2 + 4 u(k+1) u(k)$$

分别对 $u(k+1)$ 和 $u(k)$ 求偏导数并令其为 0：
1. 先对未来的输入 $u(k+1)$ 求偏导：
   $$\frac{\partial V}{\partial u(k+1)} = 2 u(k+1) + 4 u(k) = 0 \implies u(k+1) = -2 u(k)$$
2. 将 $u(k+1) = -2 u(k)$ 代回总代价函数消去 $u(k+1)$：
   $$\begin{aligned}
   u(k+1)^2 + 4 u(k+1) u(k) &= (-2 u(k))^2 + 4(-2 u(k)) u(k) \\
   &= 4 u(k)^2 - 8 u(k)^2 = -4 u(k)^2
   \end{aligned}$$
   代回后目标函数化简为仅含当前输入 $u(k)$ 的二次函数：
   $$V(k) = 7 u(k)^2 + (-4 u(k)^2) + 4 u(k) x_1(k) + 6 x_1(k)^2 = 3 u(k)^2 + 4 u(k) x_1(k) + 6 x_1(k)^2$$
3. 对当前输入 $u(k)$ 求一阶导数并令其为 0：
   $$\frac{\partial V}{\partial u(k)} = 6 u(k) + 4 x_1(k) = 0$$
   解得最优当前控制律：
   $$6 u(k) = -4 x_1(k) \implies u^*(k) = - \frac{4}{6} x_1(k) = - \frac{2}{3} x_1(k)$$
将 $u^*(k) = - \frac{2}{3} x_1(k)$ 代入闭环系统：
$$x_1(k+1) = - \frac{2}{3} x_1(k), \quad x_2(k+1) = x_1(k)$$
闭环状态转移矩阵为：
$$A_{cl} = \begin{bmatrix} -2/3 & 0 \\ 1 & 0 \end{bmatrix}$$
计算特征值：
$$\det(\lambda I - A_{cl}) = \lambda \left( \lambda + \frac{2}{3} \right) = 0 \implies \lambda_1 = - \frac{2}{3} \approx -0.6667, \quad \lambda_2 = 0$$
由于 $|\lambda_1| = \frac{2}{3} < 1$，**闭环系统同样严格渐近稳定！**

#### 【最终结论】
- 当 $H_p = 2, H_u = 1$ 时，控制律为 $u(k) = -\frac{1}{6} x_1(k)$，闭环极点为 $\lambda = -1/6$，系统稳定。
- 当 $H_p = 2, H_u = 2$ 时，控制律为 $u(k) = -\frac{2}{3} x_1(k)$，闭环极点为 $\lambda = -2/3$，系统稳定。  
这表明：**只要将预测时域延长至 $H_p \ge 2$，无论 $H_u$ 设为 1 还是 2，都能将原本失稳的系统彻底拉回稳定区域！**

---

### Exercise 6.3

#### 【解题思路】
本题要求将 Example 6.1 和 Exercise 6.2 中的加权矩阵替换为对角矩阵：
$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 6 \end{bmatrix}$$
分别求解以下三种时域组合下的预测控制律，并分析闭环行为：
1. $H_p = 1, H_u = 1$
2. $H_p = 2, H_u = 1$
3. $H_p = 2, H_u = 2$

#### 【逐步推导】

**对角加权矩阵下的预测状态二次型计算**：
对于任意状态 $x = [x_a, x_b]^T$：
$$x^T Q x = \begin{bmatrix} x_a & x_b \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 6 \end{bmatrix} \begin{bmatrix} x_a \\ x_b \end{bmatrix} = x_a^2 + 6 x_b^2$$
关键代数特征：**交叉项系数精确为 0！**

**情况 1：$H_p = 1, H_u = 1$**
$$x(k+1|k) = \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix} \implies V(k) = u(k)^2 + 6 x_1(k)^2$$
求导：
$$\frac{\partial V}{\partial u(k)} = 2 u(k) = 0 \implies u^*(k) = 0$$

**情况 2：$H_p = 2, H_u = 1$（$u(k+1) = u(k)$）**
$$x(k+1|k) = \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix}, \quad x(k+2|k) = \begin{bmatrix} u(k) \\ u(k) \end{bmatrix}$$
代入二次型：
$$V(k) = \left( u(k)^2 + 6 x_1(k)^2 \right) + \left( u(k)^2 + 6 u(k)^2 \right) = 8 u(k)^2 + 6 x_1(k)^2$$
求导：
$$\frac{\partial V}{\partial u(k)} = 16 u(k) = 0 \implies u^*(k) = 0$$

**情况 3：$H_p = 2, H_u = 2$（$u(k+1)$ 自由）**
$$x(k+1|k) = \begin{bmatrix} u(k) \\ x_1(k) \end{bmatrix}, \quad x(k+2|k) = \begin{bmatrix} u(k+1) \\ u(k) \end{bmatrix}$$
代入二次型：
$$V(k) = \left( u(k)^2 + 6 x_1(k)^2 \right) + \left( u(k+1)^2 + 6 u(k)^2 \right) = 7 u(k)^2 + u(k+1)^2 + 6 x_1(k)^2$$
求偏导：
$$\frac{\partial V}{\partial u(k+1)} = 2 u(k+1) = 0 \implies u^*(k+1) = 0$$
$$\frac{\partial V}{\partial u(k)} = 14 u(k) = 0 \implies u^*(k) = 0$$

**闭环系统响应分析**：
在所有三种情况下，最优控制律均完全相同，恒为：
$$u^*(k) \equiv 0$$
将 $u(k) = 0$ 代回真实工厂状态方程：
$$x_1(k+1) = 0, \quad x_2(k+1) = x_1(k)$$
- $k=1$: $x_1(1) = 0, \quad x_2(1) = x_1(0)$
- $k=2$: $x_1(2) = 0, \quad x_2(2) = x_1(1) = 0$
系统在正好 2 步内完全死拍（Deadbeat）衰减到绝对零点！闭环极点全为 $\lambda = 0$。

#### 【最终结论】
对于对角状态加权矩阵，由于不存在 $u(k)$ 与当前状态 $x_1(k)$ 的交叉耦合惩罚，无论时域如何选择，最优解恒为零输入 $u^*(k) = 0$。闭环系统表现为开环自主死拍收敛，极点位于原点，绝对稳定。这也反衬出：**Example 6.1 的失稳完全是由非对角权重的交叉耦合项诱发的！**

---

### Exercise 6.4

#### 【解题思路】
本题要求严格证明对于不稳定对象，式 (6.30) 与 (6.31) 给出的矩阵形式。  
核心方法是利用特征值分解将状态矩阵 $A$ 相似变换为块对角 Jordan 形，分离出严格稳定子空间与不稳定子空间；通过施加不稳定模态终端清零约束，证明无限时域剩余代价仅取决于稳定子空间，并由稳定子空间的离散 Lyapunov 方程唯一确定。

#### 【逐步推导】

1. **模态分解与坐标变换**：  
   设状态矩阵 $A \in \mathbb{R}^{n \times n}$ 具有 $n_s$ 个稳定特征值（$|\lambda| < 1$）和 $n_u$ 个不稳定特征值（$|\lambda| \ge 1$），$n_s + n_u = n$。  
   存在非奇异特征向量矩阵 $W$ 使得：
   $$A = W J W^{-1}, \quad J = \begin{bmatrix} J_s & 0 \\ 0 & J_u \end{bmatrix}$$
   其中 $J_s \in \mathbb{R}^{n_s \times n_s}$ 满足 $\rho(J_s) < 1$，$J_u \in \mathbb{R}^{n_u \times n_u}$ 满足 $|\lambda(J_u)| \ge 1$。  
   定义模态变换状态：
   $$z(k) = W^{-1} x(k) = \begin{bmatrix} z_s(k) \\ z_u(k) \end{bmatrix}, \quad W^{-1} = \begin{bmatrix} W_s^{-1} \\ W_u^{-1} \end{bmatrix}$$
   其中 $W_u^{-1} \in \mathbb{R}^{n_u \times n}$ 为提取不稳定模态的矩阵分块。

2. **自主尾部收敛的充要条件**：  
   在时域 $k+H_u$ 之后，控制增量或控制量设为 0。系统进入自主演化：
   $$x(k+H_u+m) = A^m x(k+H_u) = W J^m W^{-1} x(k+H_u) = W \begin{bmatrix} J_s^m z_s(k+H_u) \\ J_u^m z_u(k+H_u) \end{bmatrix}$$
   无限时域状态加权代价为：
   $$V_{tail} = \sum_{m=0}^\infty x(k+H_u+m)^T C^T Q C x(k+H_u+m)$$
   若 $z_u(k+H_u) \neq 0$，由于 $J_u$ 不稳定，当 $m \to \infty$ 时 $\|J_u^m z_u\| \to \infty$，级数必然发散至无穷大。  
   因此，代价有限的**充要条件**是强制不稳定模态在终端为零：
   $$z_u(k+H_u) = 0 \iff W_u^{-1} x(k+H_u) = 0$$
   这严格导出了终端约束方程（式 6.30）。

3. **稳定子空间尾部代价的收敛与 Lyapunov 方程**：  
   在 $z_u(k+H_u) = 0$ 约束下，$x(k+H_u+m) = W_s J_s^m z_s(k+H_u)$。  
   将此代入无限求和：
   $$\begin{aligned}
   V_{tail} &= \sum_{m=0}^\infty \left[ W_s J_s^m z_s(k+H_u) \right]^T C^T Q C \left[ W_s J_s^m z_s(k+H_u) \right] \\
   &= z_s(k+H_u)^T \left( \sum_{m=0}^\infty (J_s^m)^T (W_s^T C^T Q C W_s) J_s^m \right) z_s(k+H_u) \\
   &\triangleq z_s(k+H_u)^T \bar{\Pi} z_s(k+H_u)
   \end{aligned}$$
   由于 $\rho(J_s) < 1$，矩阵级数绝对收敛，且其和 $\bar{\Pi}$ 满足稳定的离散 Lyapunov 方程：
   $$J_s^T \bar{\Pi} J_s - \bar{\Pi} + W_s^T C^T Q C W_s = 0$$
4. **反变换回原始状态空间**：  
   因为 $z_s(k+H_u) = W_s^{-1} x(k+H_u)$，代入得：
   $$V_{tail} = x(k+H_u)^T \left[ (W_s^{-1})^T \bar{\Pi} W_s^{-1} \right] x(k+H_u) \triangleq x(k+H_u)^T \bar{Q}_s x(k+H_u)$$
   这完全精确地验证了式 (6.31)。

#### 【最终结论】
通过模态分解，不稳定模态被终端等式约束 $W_u^{-1} x(k+H_u) = 0$ 强制清零以阻断发散；稳定模态则通过求解降阶离散 Lyapunov 方程 $J_s^T \bar{\Pi} J_s - \bar{\Pi} + W_s^T C^T Q C W_s = 0$ 得到精确的二次型终端权重 $\bar{Q}_s = (W_s^{-1})^T \bar{\Pi} W_s^{-1}$。

---

### Exercise 6.5

#### 【解题思路】
考虑一阶标量系统：
$$x(k+1) = 0.9 x(k) + 0.5 u(k), \quad y(k) = x(k)$$
设权重为 $Q = 1, R = 0$，预测时域为无限长 $H_p = \infty$，控制时域为 $H_u = 2$。  
要求：
1. 建立并求解离散 Lyapunov 方程，求出终端加权标量 $\bar{Q}$。
2. 写出等价的有限时域目标函数。
3. 推导无约束最优预测控制律。

#### 【逐步推导】

1. **自主尾部动力学与 Lyapunov 方程求解**：  
   当 $H_u = 2$ 时，在 $j \ge 2$ 之后控制量为零（$u(k+j) = 0$）。系统状态满足：
   $$x(k+j+1) = 0.9 x(k+j)$$
   无限尾部代价为：
   $$V_{tail} = \sum_{m=1}^\infty x(k+2+m)^2 = \sum_{m=1}^\infty (0.9^m x(k+2))^2 = x(k+2)^2 \sum_{m=1}^\infty (0.9^2)^m$$
   根据等比级数求和公式（首项 $0.81$，公比 $0.81$）：
   $$\sum_{m=1}^\infty 0.81^m = \frac{0.81}{1 - 0.81} = \frac{0.81}{0.19} = \frac{81}{19}$$
   若计入时刻 $k+2$ 处的代价（原书 Lyapunov 方程约定 $a^2 \bar{Q} - \bar{Q} + 1 = 0$）：
   $$0.9^2 \bar{Q} - \bar{Q} + 1 = 0 \implies 0.81 \bar{Q} - \bar{Q} + 1 = 0$$
   $$(1 - 0.81) \bar{Q} = 1 \implies 0.19 \bar{Q} = 1 \implies \bar{Q} = \frac{1}{0.19} = \frac{100}{19} \approx 5.263158$$

2. **等价有限时域目标函数**：  
   无限时域代价转化为前 2 步加权和加上终端权重 $\bar{Q}$：
   $$V(k) = x(k+1|k)^2 + \bar{Q} x(k+2|k)^2 = x(k+1)^2 + \frac{100}{19} x(k+2)^2$$
3. **状态预测显式代数表达式**：
   $$x(k+1) = 0.9 x(k) + 0.5 u(k)$$
   $$x(k+2) = 0.9 x(k+1) + 0.5 u(k+1) = 0.9 (0.9 x(k) + 0.5 u(k)) + 0.5 u(k+1) = 0.81 x(k) + 0.45 u(k) + 0.5 u(k+1)$$
4. **求导寻找最优控制序列**：
   - 首先对未来的自由决策变量 $u(k+1)$ 求偏导：
     $$\frac{\partial V}{\partial u(k+1)} = 2 \bar{Q} x(k+2) \frac{\partial x(k+2)}{\partial u(k+1)} = 2 \cdot \frac{100}{19} x(k+2) \cdot 0.5 = \frac{100}{19} x(k+2) = 0$$
     因为 $\bar{Q} > 0$，导数为 0 必然要求：
     $$x(k+2) = 0$$
     即：最优决策必定使系统在第 2 步实现**死拍清零**！
   - 将 $x(k+2) = 0$ 代回总目标函数：
     $$V(k) = x(k+1)^2 + \frac{100}{19} \cdot 0^2 = x(k+1)^2$$
   - 现在最小化 $V(k) = x(k+1)^2$：
     $$\frac{\partial V}{\partial u(k)} = 2 x(k+1) \cdot 0.5 = x(k+1) = 0$$
     要让 $x(k+1) = 0$，将预测方程代入：
     $$0.9 x(k) + 0.5 u(k) = 0 \implies 0.5 u(k) = -0.9 x(k) \implies u^*(k) = - \frac{0.9}{0.5} x(k) = -1.8 x(k)$$
5. **闭环状态与极点验证**：
   将最优控制律 $u(k) = -1.8 x(k)$ 代回真实系统：
   $$x(k+1) = 0.9 x(k) + 0.5(-1.8 x(k)) = 0.9 x(k) - 0.9 x(k) = 0$$
   闭环极点为 $\lambda = 0$。系统在第 1 步就精确死拍归零！

#### 【最终结论】
离散 Lyapunov 终端权重为 $\bar{Q} = \frac{100}{19} \approx 5.2632$；等价有限目标函数为 $V(k) = x(k+1)^2 + \frac{100}{19} x(k+2)^2$；最优预测控制律为 $u(k) = -1.8 x(k)$，闭环为死拍系统，极点在原点，严格渐近稳定。

---

### Exercise 6.6

#### 【解题思路】
无限时域 LQ 问题保证稳定的经典充要条件是：矩阵对 $(A, B)$ 是**可镇定的（Stabilizable）**，且矩阵对 $(A, Q^{1/2})$ 或 $(A, C)$ 是**可检测的（Detectable）**。  
在采用控制增量 $\Delta u$ 的扩展增广模型中：
$$\bar{x}(k) = \begin{bmatrix} \Delta x(k) \\ y(k) \end{bmatrix}, \quad \bar{A} = \begin{bmatrix} A & 0 \\ C A & I \end{bmatrix}, \quad \bar{B} = \begin{bmatrix} B \\ C B \end{bmatrix}$$
本题要求利用 Popov-Belevitch-Hautus (PBH) 秩判据，严格证明增广系统可镇定性与可检测性的条件。

#### 【逐步推导】

**1. 可镇定性的 PBH 判据分析**：  
PBH 判据定理：矩阵对 $(\bar{A}, \bar{B})$ 可镇定，当且仅当对所有不稳定复数 $|\lambda| \ge 1$，均满足满行秩条件：
$$\text{rank} \begin{bmatrix} \lambda I - \bar{A} & \bar{B} \end{bmatrix} = n + p$$
展开分块矩阵：
$$\begin{bmatrix} \lambda I - \bar{A} & \bar{B} \end{bmatrix} = \begin{bmatrix} \lambda I - A & 0 & B \\ -CA & (\lambda - 1)I & CB \end{bmatrix}$$
分两种情况讨论特征值 $\lambda$：
- **情况 1：$\lambda \neq 1$ 且 $|\lambda| \ge 1$**：  
  由于 $\lambda \neq 1$，右下角子块 $(\lambda - 1)I$ 是满秩的可逆对角阵。我们可以做初等列变换将 $-CA$ 消去，矩阵的秩等于：
  $$\text{rank} \begin{bmatrix} \lambda I - \bar{A} & \bar{B} \end{bmatrix} = p + \text{rank} \begin{bmatrix} \lambda I - A & B \end{bmatrix}$$
  因此，满行秩条件完全退化为：对所有 $|\lambda| \ge 1 (\lambda \neq 1)$，原系统 $\text{rank} \begin{bmatrix} \lambda I - A & B \end{bmatrix} = n$，即**原系统 $(A, B)$ 自身必须可镇定**！
- **情况 2：$\lambda = 1$（积分器极点）**：  
  代入 $\lambda = 1$：
  $$\begin{bmatrix} I - \bar{A} & \bar{B} \end{bmatrix} = \begin{bmatrix} I - A & 0 & B \\ -CA & 0 & CB \end{bmatrix}$$
  将第 1 行左乘 $C$ 加到第 2 行：
  $$C(I - A) + (-CA) = C - CA - CA = C - CA \dots$$
  更为标准的初等块行变换是：利用系统方程稳态增益矩阵：
  $$\begin{bmatrix} I - A & B \\ C & 0 \end{bmatrix}$$
  该矩阵满行秩要求系统在 $z=1$ 处不存在传递函数零点！即：**被控对象在 $z=1$ 处无不变零点（No transmission zeros at z=1）**。如果系统在 $z=1$ 处有零点，则积分控制无法起作用。

**2. 可检测性的 PBH 判据分析**：  
PBH 判据定理：矩阵对 $(\bar{A}, \bar{C})$ 可检测，当且仅当对所有 $|\lambda| \ge 1$：
$$\text{rank} \begin{bmatrix} \lambda I - \bar{A} \\ \bar{C} \end{bmatrix} = n + p$$
在标准 MPC 中，输出加权测量为 $y(k)$，故 $\bar{C} = \begin{bmatrix} 0 & I \end{bmatrix}$。代入得：
$$\begin{bmatrix} \lambda I - \bar{A} \\ \bar{C} \end{bmatrix} = \begin{bmatrix} \lambda I - A & 0 \\ -CA & (\lambda - 1)I \\ 0 & I \end{bmatrix}$$
利用最底部的 $I$ 可以彻底消去中间行的 $(\lambda - 1)I$。消去后矩阵等价于：
$$\begin{bmatrix} \lambda I - A \\ -CA \\ I \end{bmatrix}$$
显然最上方的 $\lambda I - A$ 对所有 $|\lambda| \ge 1$ 必须与测量矩阵保持满列秩，这直接等价于**原系统 $(A, C)$ 必须是可检测的**。

#### 【最终结论】
增广系统在无限时域保持可镇定和可检测的充要条件为：
1. 原系统 $(A, B)$ 可镇定；
2. 原系统 $(A, C)$ 可检测；
3. 原系统在单位圆 $z=1$ 处无传输零点（确保能够通过积分作用消除稳态误差）。

---

### Exercise 6.7

#### 【解题思路】
考虑有限时域 LQ 问题，对象参数为：
$$A = \begin{bmatrix} 0 & 2 \\ 2 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad P_{N-1} = \begin{bmatrix} 16 & 0 \\ 0 & 16 \end{bmatrix}, \quad R = 1$$
要求：
1. 计算最优反馈增益矩阵 $K_{N-1} = (R + B^T P_{N-1} B)^{-1} B^T P_{N-1} A$。
2. 验证闭环矩阵 $A - B K_{N-1}$ 的特征值并判定其稳定性。
3. 从假代数 Riccati 方程（FARE）推导出隐含的单步状态加权矩阵 $Q_{N-1}$，验证其是否正定，并阐明其控制理论深层含义。

#### 【逐步推导】

**步骤 1：精确计算增益矩阵 $K_{N-1}$**
1. 计算标量项 $B^T P_{N-1} B$：
   $$B^T P_{N-1} B = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 16 & 0 \\ 0 & 16 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 16 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 16$$
2. 计算逆矩阵项 $(R + B^T P_{N-1} B)^{-1}$：
   $$(R + B^T P_{N-1} B)^{-1} = (1 + 16)^{-1} = 17^{-1} = \frac{1}{17}$$
3. 计算行向量项 $B^T P_{N-1} A$：
   $$B^T P_{N-1} = \begin{bmatrix} 16 & 0 \end{bmatrix}$$
   $$B^T P_{N-1} A = \begin{bmatrix} 16 & 0 \end{bmatrix} \begin{bmatrix} 0 & 2 \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 16 \times 0 + 0 \times 2 & 16 \times 2 + 0 \times 0 \end{bmatrix} = \begin{bmatrix} 0 & 32 \end{bmatrix}$$
4. 增益矩阵：
   $$K_{N-1} = \frac{1}{17} \begin{bmatrix} 0 & 32 \end{bmatrix} = \begin{bmatrix} 0 & \frac{32}{17} \end{bmatrix} \approx \begin{bmatrix} 0 & 1.882353 \end{bmatrix}$$

---

**步骤 2：计算闭环矩阵及其特征值**
$$B K_{N-1} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 0 & \frac{32}{17} \end{bmatrix} = \begin{bmatrix} 0 & \frac{32}{17} \\ 0 & 0 \end{bmatrix}$$
闭环状态转移矩阵：
$$A_{cl} = A - B K_{N-1} = \begin{bmatrix} 0 & 2 \\ 2 & 0 \end{bmatrix} - \begin{bmatrix} 0 & \frac{32}{17} \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 2 - \frac{32}{17} \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 0 & \frac{2}{17} \\ 2 & 0 \end{bmatrix}$$
计算其特征方程：
$$\det(\lambda I - A_{cl}) = \det\begin{bmatrix} \lambda & -\frac{2}{17} \\ -2 & \lambda \end{bmatrix} = \lambda^2 - (-2)\left(-\frac{2}{17}\right) = \lambda^2 - \frac{4}{17} = 0$$
解特征值：
$$\lambda^2 = \frac{4}{17} \implies \lambda = \pm \sqrt{\frac{4}{17}} \approx \pm \frac{2}{4.1231} \approx \pm 0.485071$$
计算特征值模长：
$$|\lambda| = \sqrt{\frac{4}{17}} \approx 0.485071 < 1.0$$
因为两个闭环极点均严格位于复平面单位圆内，**闭环系统严格渐近稳定！**

---

**步骤 3：反推隐含状态加权矩阵 $Q_{N-1}$**  
回顾假代数 Riccati 方程（式 6.49）：
$$P_{N-1} = A^T P_{N-1} A - A^T P_{N-1} B (R + B^T P_{N-1} B)^{-1} B^T P_{N-1} A + Q_{N-1}$$
注意到后半部分正好就是 $A^T P_{N-1} B K_{N-1}$，移项解出 $Q_{N-1}$：
$$Q_{N-1} = P_{N-1} - A^T P_{N-1} A + A^T P_{N-1} B K_{N-1}$$
逐项显式计算：
1. $P_{N-1} = \begin{bmatrix} 16 & 0 \\ 0 & 16 \end{bmatrix}$
2. $A^T P_{N-1} A$：
   $$A^T P_{N-1} = \begin{bmatrix} 0 & 2 \\ 2 & 0 \end{bmatrix} \begin{bmatrix} 16 & 0 \\ 0 & 16 \end{bmatrix} = \begin{bmatrix} 0 & 32 \\ 32 & 0 \end{bmatrix}$$
   $$A^T P_{N-1} A = \begin{bmatrix} 0 & 32 \\ 32 & 0 \end{bmatrix} \begin{bmatrix} 0 & 2 \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 64 & 0 \\ 0 & 64 \end{bmatrix}$$
3. $A^T P_{N-1} B K_{N-1}$：
   $$A^T P_{N-1} B = \begin{bmatrix} 0 & 32 \\ 32 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 32 \end{bmatrix}$$
   $$A^T P_{N-1} B K_{N-1} = \begin{bmatrix} 0 \\ 32 \end{bmatrix} \begin{bmatrix} 0 & \frac{32}{17} \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & \frac{1024}{17} \end{bmatrix} \approx \begin{bmatrix} 0 & 0 \\ 0 & 60.235294 \end{bmatrix}$$
4. 综合求和计算 $Q_{N-1}$：
   $$\begin{aligned}
   Q_{N-1} &= \begin{bmatrix} 16 & 0 \\ 0 & 16 \end{bmatrix} - \begin{bmatrix} 64 & 0 \\ 0 & 64 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 0 & \frac{1024}{17} \end{bmatrix} \\
   &= \begin{bmatrix} 16 - 64 & 0 \\ 0 & 16 - 64 + \frac{1024}{17} \end{bmatrix} \\
   &= \begin{bmatrix} -48 & 0 \\ 0 & -48 + \frac{1024}{17} \end{bmatrix}
   \end{aligned}$$
   计算 $(2,2)$ 元素：
   $$-48 + \frac{1024}{17} = \frac{-48 \times 17 + 1024}{17} = \frac{-816 + 1024}{17} = \frac{208}{17} \approx +12.235294$$
   因此：
   $$Q_{N-1} = \begin{bmatrix} -48 & 0 \\ 0 & \frac{208}{17} \end{bmatrix}$$
5. **正定性判定**：
   $Q_{N-1}$ 是对角矩阵，其特征值直接就是对角线元素：
   $$\lambda_1(Q_{N-1}) = -48, \quad \lambda_2(Q_{N-1}) = \frac{208}{17} \approx +12.2353$$
   因为存在严格负特征值 $-48$，$Q_{N-1}$ **严格不定（Indefinite）**，绝非正定或半正定矩阵！

#### 【最终结论】
1. 反馈增益为 $K_{N-1} = \begin{bmatrix} 0 & \frac{32}{17} \end{bmatrix}$。
2. 闭环矩阵特征值为 $\lambda = \pm \sqrt{4/17} \approx \pm 0.4851$，谱半径小于 1，闭环严格稳定。
3. 隐含状态加权矩阵为 $Q_{N-1} = \text{diag}\left(-48, \frac{208}{17}\right)$，具有负特征值，是不定矩阵。
4. **控制理论核心洞见**：标准 LQ 与 Lyapunov 稳定性定理要求加权矩阵 $Q \ge 0$ 只是保证闭环稳定的**充分条件**，而**绝非必要条件**！即使目标函数中某状态权重大幅为负，系统在 Riccati 反馈下依然能够保持严格渐近稳定。

---

## 8. 本章自测题

### 8.1 概念题（共 5 题）
1. 为什么“每一拍都求解全局最优”的有限时域 MPC，施加到真实闭环系统中可能导致发散失稳？其失稳的根本几何机制是什么？
2. 试述 Keerthi-Gilbert 定理中，构造时刻 $k+1$ 可行候选控制序列的“补零法”核心思想。为什么最后一步补零能够自然满足终端等式约束？
3. 什么是离散 Lyapunov 方程 $A^T \bar{Q} A - \bar{Q} + C^T Q C = 0$？为什么将终端权重设为 $\bar{Q}$ 能够让有限时域 MPC 在数学上精确等价于无限预测时域 MPC？
4. 当被控对象具有开环不稳定极点时，为什么不能直接求解离散 Lyapunov 方程？模态分解是如何将问题解耦为“等式约束”与“Lyapunov 权重”两部分的？
5. 简述基于 Youla 参数化（$Q$-Parameterization）的 MPC 结构。为什么即使在控制量发生严重饱和限制时，该架构依然能无条件保证闭环内部稳定性？

### 8.2 推导题（共 3 题）
1. 考虑一阶被控对象 $x(k+1) = a x(k) + b u(k)$，其中 $|a| < 1$。目标函数为无限时域 $V = \sum_{j=1}^\infty x(k+j)^2$。设控制时域为 $H_u = 1$。利用等比级数求和公式，完整推导终端权重 $\bar{Q}$ 的标量解析表达式，并求出最优反馈控制律 $u^*(k) = K x(k)$。
2. 设开环系统为 $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$（离散双积分器）。试求满足终端等式约束 $x(k+2|k) = 0$ 的可控状态集合 $\mathbb{X}_{reach}(2)$ 在控制约束 $|u| \le 1$ 下的多面体不等式边界。
3. 给定标量 Riccati 差分方程 $P_{j+1} = a^2 P_j - \frac{a^2 b^2 P_j^2}{R + b^2 P_j} + Q$。试证明：若初值选取满足 $P_1 \le P_0$，则序列 $\{P_j\}$ 严格单调不增，且有下界，因而必收敛于代数 Riccati 方程的唯一正定解。

### 8.3 编程题（共 2 题）
1. 编写 Python 程序，针对二阶欠阻尼振荡系统 $G(z) = \frac{0.1 z + 0.08}{z^2 - 1.5 z + 0.7}$，分别实现带终端 Lyapunov 权重 $\bar{Q}$ 的 MPC 与不带终端权重的朴素 MPC。在采样步长内绘制闭环极点分布图与跟踪误差响应曲线。
2. 编写 Python 程序，实现 Example 6.4 的 Youla 参数化预测控制器。针对带有纯滞后的非最小相位对象 $P(z) = \frac{z - 1.2}{(z - 0.6)(z - 0.8)}$，将 Youla 参数设为 3 阶 FIR 滤波器，求解在输入幅值约束 $|u| \le 1.5$ 下的最佳 FIR 抽头系数。

### 8.4 综合题（共 1 题）
某化学反应器存在一个不稳定热失控模态（开环极点 $\lambda = 1.4$）和一个稳定反应物浓度模态（开环极点 $\lambda = 0.5$）。已知执行器阀门存在严格的行程限制 $0 \le u(k) \le 2$ 与速率限制 $|\Delta u(k)| \le 0.5$。请设计一套完整的“模态解耦 + 无限时域等价有限 QP”的 MPC 架构：
1. 写出坐标变换矩阵 $W$ 与解耦状态方程；
2. 建立带状态与速率约束的标准化 QP 求解矩阵；
3. 说明当由于扰动导致初始状态超出可控盆地（Controllable Basin）时，如何通过软化终端约束（Soft Constraints / Slack Variables）保证求解器始终可行且不失稳。

---

## 9. 总结与下一章衔接

### 9.1 本章核心结论
1. **短视必危**：有限时域优化并不包含稳定性保证。非对角权重或不恰当的时域长度可能将系统推向不稳定，必须在设计层面显式引入稳定性约束或终端惩罚。
2. **四大镇定法宝**：
   - **终端等式约束**：结构最纯粹，通过将最优化目标函数直接用作 Lyapunov 函数证明渐近稳定，但代价是初始可行域偏小。
   - **无限时域 Lyapunov 权重**：对稳定对象将无限时域压缩为有限 QP，计算开销极低且可行域巨大。
   - **模态解耦法**：对不稳定对象对症下药，仅对不稳定模态施加等式约束，兼顾可行域与镇定性。
   - **Youla 参数化**：IMC 结构下的凸优化范式，从底层系统拓扑上彻底消除了发散风险。
3. **充分不等于必要**：Riccati/Lyapunov 理论中的 $Q \ge 0$ 是充分条件。实际工程中某些状态权重为负并不必然导致闭环失稳。

### 9.2 常见误区与避坑指南
- **误区 1：只要矩阵 $Q$ 和 $R$ 是正定的，MPC 就一定会稳定。**  
  *纠错*：Example 6.1 无情打破了这一幻想。$Q > 0$ 只能保证优化问题有唯一极小值，无法保证滚动执行后的闭环极点落在单位圆内。
- **误区 2：无限时域 MPC 必须在计算机里预测成百上千步。**  
  *纠错*：完全不需要！离散 Lyapunov 方程 $A^T \bar{Q} A - \bar{Q} + C^T Q C = 0$ 允许我们在 $H_u$ 步（通常仅 2~5 步）后直接用一个二次型矩阵 $\bar{Q}$ 精确封闭尾部。
- **误区 3：终端等式约束可以随意加给任何初始状态。**  
  *纠错*：终端约束 $x(k+H_p) = 0$ 严重压缩了可行域。若初始状态不在 $H_p$ 步饱和执行器的可达集合内，QP 求解器会直接报告无解（Infeasible）。

### 9.3 下一章（Chapter 7: Tuning）衔接
在第 6 章中，我们解决了预测控制最关心的**生存底线问题——稳定性（Stability）**。  
然而，一个稳定的控制器并不意味着它是一个性能优秀的控制器。在实际工业过程中：
- 预测时域 $H_p$ 和控制时域 $H_u$ 究竟选多大才合适？
- 状态权重 $Q$、输出权重 $Q_y$、控制权重 $R$ 和增量权重 $R_{\Delta u}$ 应该如何系统性整定？
- 观测器（Kalman 滤波 / DMC 观测器）的滤波极点应当如何与 MPC 控制极点匹配？  
这些问题将在 **第 7 章 整定（Tuning）** 中全面展开！

---

## 10. 推荐经典教程与拓展阅读

1. **Keerthi, S. S. & Gilbert, E. G. (1988)**. *Optimal, infinite-horizon feedback laws for a general class of constrained discrete-time systems: Stability and moving-horizon approximations*. Journal of Optimization Theory and Applications, 57(2), 265–293.  
   *(奠基之作：首次严格提出终端等式约束与 Lyapunov 函数单调递减证明)*
2. **Rawlings, J. B. & Muske, K. R. (1993)**. *The stability of constrained receding horizon control*. IEEE Transactions on Automatic Control, 38(10), 1512–1516.  
   *(无限预测时域法与模态分解解耦处理不稳定对象的权威文献)*
3. **Mayne, D. Q., Rawlings, J. B., Rao, C. V., & Scokaert, P. O. (2000)**. *Constrained model predictive control: Stability and optimality*. Automatica, 36(6), 789–814.  
   *(MPC 稳定性理论世纪之交的最权威经典综述)*
4. **Bitmead, R. R., Gevers, M., & Wertz, V. (1990)**. *Adaptive Optimal Control: The Thinking Man's GPC*. Prentice Hall.  
   *(伪代数 Riccati 方程 FARE 与单调性收敛理论的专著)*
5. **Maciejowski, J. M. (1989)**. *Multivariable Feedback Design*. Addison-Wesley.  
   *(深入理解 Youla-Kucera 参数化与多变量频域鲁棒性的经典专著)*
