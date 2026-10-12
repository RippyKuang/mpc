# 第 3 章：预测控制问题的求解 (Solving Predictive Control Problems) —— 逐章读书指导与全宇宙最详尽推导

**书目**：J. M. Maciejowski, *Predictive Control with Constraints*, Prentice Hall, 2002  
**讲义研发**：MPC 资深讲师讲义组  
**目标读者**：具备经典控制理论基础（只学过单回路 PID、传递函数 $G(s)$、Nyquist/Bode 频域响应），对现代控制理论、最优控制、状态观测器、凸优化与非线性控制零基础的工程科研人员。

---

## 1. 章节元信息与核心定位

### 1.1 章节全书定位
在第 2 章中，我们完成了状态空间预测模型的构建，得到了包含跟踪误差加权与控制增量惩罚的目标函数 $V(k)$，并将控制变化率、控制幅值、输出幅值三大约束统一整合为线性不等式 $\Omega \Delta\mathcal{U}(k) \le \omega(k)$。

然而，第 2 章留下了三个亟待解决的数学与工程求解核心问题：
1. **无约束时怎么求最优解？** 目标函数 $V(k)$ 的解析极小值是什么？闭环控制器结构如何表现为经典状态反馈与积分环节？
2. **有约束时如何在毫秒级周期内实时求解？** 加入约束后解析解不复存在，问题转化为二次规划（Quadratic Programming, QP）。计算机如何通过数值算法（有效集法 Active Set、内点法 Interior Point）在每一拍稳定求出全局最优解？
3. **当约束发生冲突不可行（Infeasible）时，系统如何避免崩溃？** 当外部突加未知强扰动或出现模型失配时，物理输出可能被逼出约束边界，硬约束 QP 求解器将直接抛出异常崩溃。如何通过**约束软化（Constraint Softening）与精确罚函数**赋予控制器自救能力？

第 3 章正是回答上述三大关键问题的核心计算中枢。它是全书从“模型表达”跨越到“数值算法落地与闭环实时运行”的决定性桥梁。

```
                    【第 3 章求解算法与控制结构全貌】
                    ┌─────────────────────────┐
                    │ 目标函数与约束模型      │
                    │ V = ||Z-T||_Q^2+||ΔU||_R^2│
                    │ Ω ΔU(k) ≤ ω(k)          │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ 3.1 无约束问题解析求解          │   │ 3.2 约束问题与分段仿射 (PWA)    │
│ 驻点梯度 ∇V = 0                 │   │ 约束将系统变成非线性切换控制器  │
│ ΔU_opt = H^-1 Θ^T Q E(k)        │   │ Mini-Tutorial 3: 凸优化与 KKT   │
│ 最小二乘 QR 分解规避病态平方    │   └───────────────┬─────────────────┘
│ 等效 LTI 状态反馈 + 天然积分器  │                   │
└─────────────────────────────────┘                   ▼
                                      ┌─────────────────────────────────┐
                                      │ 3.3 二次规划 (QP) 在线求解算法  │
                                      │ ├─ 3.3.1 有效集法 (Active Set)  │
                                      │ └─ 3.3.2 内点法 (Interior Point)│
                                      └───────────────┬─────────────────┘
                                                      │
                                                      ▼
                                      ┌─────────────────────────────────┐
                                      │ 3.4 约束软化与精确惩罚定理      │
                                      │ 松弛变量 ε ≥ 0                  │
                                      │ 1-范数 / 2-范数 / ∞-范数        │
                                      │ 临界条件 ρ > max |λ_i*| 防崩溃  │
                                      └─────────────────────────────────┘
```

---

## 2. 符号、假设与预备知识（面向传递函数背景读者）

### 2.1 维度定义表
设受控对象具有：
- 状态变量维数：$n$（状态向量 $x \in \mathbb{R}^n$）；
- 控制输入维数：$\ell$（控制向量 $u \in \mathbb{R}^\ell$，增量 $\Delta u \in \mathbb{R}^\ell$）；
- 被控输出维数：$m$（被控输出向量 $z \in \mathbb{R}^m$）；
- 预测时域窗口：$H_p$；控制时域窗口：$H_u$；预测起始步长：$H_w = 1$；
- 预测区间输出点数：$H_p - H_w + 1$。

| 符号 | 维度 | 物理/数学含义 |
| :--- | :--- | :--- |
| $\mathcal{Z}(k)$ | $m H_p \times 1$ | 预测时域内的未来被控输出堆叠向量 |
| $\mathcal{T}(k)$ | $m H_p \times 1$ | 预测时域内的未来参考设定点轨迹堆叠向量 |
| $\Delta\mathcal{U}(k)$ | $\ell H_u \times 1$ | 优化决策变量：未来待求解的控制增量序列 |
| $\mathcal{Q}$ | $m H_p \times m H_p$ | 块对角输出跟踪误差加权矩阵，$\mathcal{Q} \ge 0$ |
| $\mathcal{R}$ | $\ell H_u \times \ell H_u$ | 块对角控制增量惩罚矩阵，$\mathcal{R} \ge 0$ |
| $\Psi$ | $m H_p \times n$ | 状态自由响应传递矩阵 |
| $\Upsilon$ | $m H_p \times \ell$ | 历史输入保持传递矩阵 |
| $\Theta$ | $m H_p \times \ell H_u$ | 动态控制增量强迫响应矩阵（下三角块 Toeplitz 结构）|
| $\mathcal{E}(k)$ | $m H_p \times 1$ | 自由响应跟踪误差向量：$\mathcal{E}(k) \triangleq \mathcal{T}(k) - \Psi x(k) - \Upsilon u(k-1)$ |
| $H$ | $\ell H_u \times \ell H_u$ | 目标函数的 Hessian 矩阵：$H \triangleq \Theta^T \mathcal{Q} \Theta + \mathcal{R}$ |
| $G$ | $\ell H_u \times 1$ | 目标函数的一次项梯度匹配向量：$G \triangleq 2\Theta^T \mathcal{Q} \mathcal{E}(k)$ |
| $\Phi$ | $\ell H_u \times \ell H_u$ | 标准 QP 的 Hessian 矩阵：$\Phi \triangleq 2H = 2(\Theta^T \mathcal{Q} \Theta + \mathcal{R})$ |
| $\phi(k)$ | $\ell H_u \times 1$ | 标准 QP 的线性项向量：$\phi \triangleq -G = -2\Theta^T \mathcal{Q} \mathcal{E}(k)$ |
| $\Omega$ | $n_c \times \ell H_u$ | 线性不等式约束矩阵（$n_c$ 为总约束行数） |
| $\omega(k)$ | $n_c \times 1$ | 线性不等式约束右手边边界向量 |
| $\lambda$ | $n_c \times 1$ | 对偶变量 / Lagrange 乘子向量 |
| $\varepsilon$ | 标量或向量 | 约束软化松弛变量（Slack Variable） |

### 2.2 基础微积分与线性代数预备知识（面向传递函数背景读者）
在单变量微积分中，一元二次函数 $f(\theta) = a \theta^2 + b \theta + c$ 的极值求解为：
$$
\frac{df}{d\theta} = 2 a \theta + b = 0 \implies \theta^* = -\frac{b}{2a} \quad (\text{当 } a > 0 \text{ 时取全局极小值})
$$
推广到多变量向量 $\theta \in \mathbb{R}^N$：
- 目标函数：$V(\theta) = \theta^T H \theta - \theta^T G + c$
- 对向量 $\theta$ 的一阶导数（梯度 $\nabla_\theta V \in \mathbb{R}^N$）：
  $$
  \nabla_\theta V = 2 H \theta - G
  $$
- 二阶导数矩阵（Hessian 矩阵 $\nabla_\theta^2 V \in \mathbb{R}^{N \times N}$）：
  $$
  \nabla_\theta^2 V = 2 H
  $$
- 当矩阵 $H$ 是**严格正定矩阵**（即对任意非零向量 $v \ne 0$，总有 $v^T H v > 0$）时，几何曲面为严格向上凸起的椭圆抛物面碗底，函数存在唯一全局极小值，由驻点方程 $2 H \theta^* - G = 0$ 直接给出：
  $$
  \theta^* = \frac{1}{2} H^{-1} G
  $$

---

## 3. 逐节精读与严格推导（符合【解题思路】+【逐步推导】+【最终结论】硬性规范）

### 3.1 无约束问题 (Unconstrained problems)

#### 3.1.1 状态完全可测且无扰动下的目标函数极小值解析推导

##### 【解题思路】
目标函数 $V(k)$ 衡量了未来预测输出与参考轨迹之间的跟踪误差平方和，加上对未来控制增量的能量惩罚。利用第 2 章推导的预测方程 $\mathcal{Z}(k) = \Psi x(k) + \Upsilon u(k-1) + \Theta \Delta\mathcal{U}(k)$，将与决策变量无关的自由响应项分离，构造自由响应跟踪误差 $\mathcal{E}(k)$。展开目标函数二次型后，通过一阶导数（梯度）令其为零，求出决策变量 $\Delta\mathcal{U}(k)$ 的解析显式解，并由二阶 Hessian 正定性证明全局极小值。

##### 【逐步推导】
1. **写出紧凑目标函数**：
   原书式 (3.1)–(3.2)：
   $$
   V(k) = \|\mathcal{Z}(k) - \mathcal{T}(k)\|_{\mathcal{Q}}^2 + \|\Delta\mathcal{U}(k)\|_{\mathcal{R}}^2
   $$
   其中記号 $\|v\|_M^2 \triangleq v^T M v$。

2. **代入预测方程并分离自由响应**：
   根据第 2 章式 (2.66)：
   $$
   \mathcal{Z}(k) = \Psi x(k) + \Upsilon u(k-1) + \Theta \Delta\mathcal{U}(k)
   $$
   定义自由响应跟踪误差向量（原书式 (3.6)）：
   $$
   \mathcal{E}(k) \triangleq \mathcal{T}(k) - \Psi x(k) - \Upsilon u(k-1)
   $$
   *依据*：$\Psi x(k) + \Upsilon u(k-1)$ 为未来不施加任何控制增量（$\Delta\mathcal{U}=0$）时系统自发产生的自由响应，$\mathcal{E}(k)$ 为参考轨迹与自由响应之差。
   代入误差项：
   $$
   \mathcal{Z}(k) - \mathcal{T}(k) = \Theta \Delta\mathcal{U}(k) - [\mathcal{T}(k) - \Psi x(k) - \Upsilon u(k-1)] = \Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)
   $$
   目标函数改写为（原书式 (3.7)）：
   $$
   V(k) = \|\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)\|_{\mathcal{Q}}^2 + \|\Delta\mathcal{U}(k)\|_{\mathcal{R}}^2
   $$

3. **完全展开二次型乘积**：
   $$
   \begin{aligned}
   V(k) &= [\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)]^T \mathcal{Q} [\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)] + \Delta\mathcal{U}(k)^T \mathcal{R} \Delta\mathcal{U}(k) \\
   &= [\Delta\mathcal{U}(k)^T \Theta^T - \mathcal{E}(k)^T] \mathcal{Q} [\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)] + \Delta\mathcal{U}(k)^T \mathcal{R} \Delta\mathcal{U}(k) \\
   &= \Delta\mathcal{U}(k)^T \Theta^T \mathcal{Q} \Theta \Delta\mathcal{U}(k) - \Delta\mathcal{U}(k)^T \Theta^T \mathcal{Q} \mathcal{E}(k) - \mathcal{E}(k)^T \mathcal{Q} \Theta \Delta\mathcal{U}(k) + \mathcal{E}(k)^T \mathcal{Q} \mathcal{E}(k) + \Delta\mathcal{U}(k)^T \mathcal{R} \Delta\mathcal{U}(k)
   \end{aligned}
   $$
   注意到标量转置等于自身，且 $\mathcal{Q} = \mathcal{Q}^T$：
   $$
   [\mathcal{E}(k)^T \mathcal{Q} \Theta \Delta\mathcal{U}(k)]^T = \Delta\mathcal{U}(k)^T \Theta^T \mathcal{Q} \mathcal{E}(k)
   $$
   交叉项合并为 $-2 \Delta\mathcal{U}(k)^T \Theta^T \mathcal{Q} \mathcal{E}(k)$。合并同类项整理得到（原书式 (3.9)）：
   $$
   V(k) = \mathcal{E}(k)^T \mathcal{Q} \mathcal{E}(k) - 2 \Delta\mathcal{U}(k)^T \Theta^T \mathcal{Q} \mathcal{E}(k) + \Delta\mathcal{U}(k)^T [\Theta^T \mathcal{Q} \Theta + \mathcal{R}] \Delta\mathcal{U}(k)
   $$

4. **定义梯度矩阵与 Hessian 矩阵**：
   令与决策变量无关的常数项为 $\text{const} \triangleq \mathcal{E}(k)^T \mathcal{Q} \mathcal{E}(k)$，定义：
   $$
   G \triangleq 2 \Theta^T \mathcal{Q} \mathcal{E}(k) \tag{3.11}
   $$
   $$
   H \triangleq \Theta^T \mathcal{Q} \Theta + \mathcal{R} \tag{3.12}
   $$
   目标函数化为标准多项式（原书式 (3.10)）：
   $$
   V(k) = \text{const} - \Delta\mathcal{U}(k)^T G + \Delta\mathcal{U}(k)^T H \Delta\mathcal{U}(k)
   $$

5. **一阶导数令其为零求驻点**：
   对 $\Delta\mathcal{U}(k)$ 求导（原书式 (3.13)）：
   $$
   \nabla_{\Delta\mathcal{U}(k)} V(k) = -G + 2 H \Delta\mathcal{U}(k) = 0
   $$
   移项：
   $$
   2 H \Delta\mathcal{U}(k)_{opt} = G \implies H \Delta\mathcal{U}(k)_{opt} = \Theta^T \mathcal{Q} \mathcal{E}(k)
   $$
   两边左乘 $H^{-1}$：
   $$
   \Delta\mathcal{U}(k)_{opt} = H^{-1} \Theta^T \mathcal{Q} \mathcal{E}(k) = (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q} \mathcal{E}(k) \tag{3.14}
   $$

6. **二阶导数与极小值充分条件验证**：
   对梯度再次求导得 Hessian 矩阵（原书式 (3.16)）：
   $$
   \frac{\partial^2 V}{\partial \Delta\mathcal{U}^2} = 2 H = 2(\Theta^T \mathcal{Q} \Theta + \mathcal{R})
   $$
   - 因 $\mathcal{Q} \ge 0$，对任意 $v \ne 0$ 有 $v^T \Theta^T \mathcal{Q} \Theta v \ge 0$；
   - 若控制增量惩罚矩阵 $\mathcal{R} > 0$，则 $v^T \mathcal{R} v > 0$；
   - 故 $v^T H v > 0$，即 $H > 0$ 严格正定。目标函数为严格凸函数，驻点 $\Delta\mathcal{U}(k)_{opt}$ 必然为全局唯一极小值解！

##### 【最终结论】
未来最优控制增量序列的显式解析解为：
$$
\Delta\mathcal{U}(k)_{opt} = (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q} \mathcal{E}(k)
$$
根据滚动时域原理，当拍施加给执行器的实际增量为第一步：
$$
\Delta u(k) = [I_\ell, 0, \dots, 0] \Delta\mathcal{U}(k)_{opt} \tag{3.15}
$$

---

#### 3.1.2 最小二乘法重构与 QR 正交分解严密推导 (§3.1.2)

##### 【解题思路】
在工业数值计算中，原书作者警告：**严禁直接计算求逆 $H^{-1} = (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1}$！**  
因为构造正则方程矩阵 $\Theta^T \Theta$ 会使其条件数平方化：$\kappa(\Theta^T \Theta) = [\kappa(\Theta)]^2$。若 $\Theta$ 的条件数为 $10^5$，则 $H$ 的条件数飙升至 $10^{10}$，在有限浮点精度下产生毁灭性截断误差。正确的做法是将其重构为长瘦矩阵的**标准线性最小二乘问题（Least Squares）**，并通过 **QR 正交分解** 求解。

##### 【逐步推导】
1. **范数平方代数重构**：
   利用矩阵平方根（$\mathcal{Q} = \mathcal{Q}^{1/2} \mathcal{Q}^{1/2}$，$\mathcal{R} = \mathcal{R}^{1/2} \mathcal{R}^{1/2}$），将加权范数写为标准欧氏 2-范数：
   $$
   \|\Theta \Delta\mathcal{U} - \mathcal{E}\|_{\mathcal{Q}}^2 = \|\mathcal{Q}^{1/2} (\Theta \Delta\mathcal{U} - \mathcal{E})\|_2^2
   $$
   $$
   \|\Delta\mathcal{U}\|_{\mathcal{R}}^2 = \|\mathcal{R}^{1/2} \Delta\mathcal{U}\|_2^2
   $$
   两项相加等价于纵向拼接向量的欧氏范数平方：
   $$
   V(k) = \left\| \begin{bmatrix} \mathcal{Q}^{1/2} (\Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)) \\ \mathcal{R}^{1/2} \Delta\mathcal{U}(k) \end{bmatrix} \right\|_2^2 = \left\| \begin{bmatrix} \mathcal{Q}^{1/2} \Theta \\ \mathcal{R}^{1/2} \end{bmatrix} \Delta\mathcal{U}(k) - \begin{bmatrix} \mathcal{Q}^{1/2} \mathcal{E}(k) \\ 0 \end{bmatrix} \right\|_2^2 \tag{3.21}
   $$

2. **定义超定方程组矩阵与向量**：
   $$
   \mathcal{A} \triangleq \begin{bmatrix} \mathcal{Q}^{1/2} \Theta \\ \mathcal{R}^{1/2} \end{bmatrix} \in \mathbb{R}^{(m H_p + \ell H_u) \times \ell H_u}, \quad b(k) \triangleq \begin{bmatrix} \mathcal{Q}^{1/2} \mathcal{E}(k) \\ 0 \end{bmatrix} \in \mathbb{R}^{(m H_p + \ell H_u) \times 1} \tag{3.18-3.19}
   $$
   问题转化为标准超定最小二乘问题：
   $$
   \min_{\Delta\mathcal{U}(k)} \|\mathcal{A} \Delta\mathcal{U}(k) - b(k)\|_2^2 \tag{3.20}
   $$

3. **QR 正交分解严密推导**：
   对矩阵 $\mathcal{A}$ 进行 QR 分解：
   $$
   \mathcal{A} = Q_R \begin{bmatrix} R_A \\ 0 \end{bmatrix} = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_A \\ 0 \end{bmatrix} = Q_1 R_A \tag{3.23}
   $$
   其中 $Q_R \in \mathbb{R}^{(m H_p + \ell H_u) \times (m H_p + \ell H_u)}$ 为正交矩阵（满足 $Q_R^T Q_R = I$），$R_A \in \mathbb{R}^{\ell H_u \times \ell H_u}$ 为非奇异上三角矩阵。
   利用正交变换保范数性质（$\|Q_R^T v\|_2 = \|v\|_2$）：
   $$
   \begin{aligned}
   \|\mathcal{A} \Delta\mathcal{U} - b\|_2^2 &= \left\| Q_R^T \left( \mathcal{A} \Delta\mathcal{U} - b \right) \right\|_2^2 \\
   &= \left\| \begin{bmatrix} Q_1^T \\ Q_2^T \end{bmatrix} \left( Q_1 R_A \Delta\mathcal{U} - b \right) \right\|_2^2 \\
   &= \left\| \begin{bmatrix} R_A \Delta\mathcal{U} - Q_1^T b \\ -Q_2^T b \end{bmatrix} \right\|_2^2 \\
   &= \|R_A \Delta\mathcal{U} - Q_1^T b\|_2^2 + \|Q_2^T b\|_2^2
   \end{aligned}
   $$
   注意第二项 $\|Q_2^T b\|_2^2$ 不含决策变量 $\Delta\mathcal{U}$，代表不可控残差。因此极小化目标仅需令第一项严格为零：
   $$
   R_A \Delta\mathcal{U}(k)_{opt} = Q_1^T b(k) \tag{3.24}
   $$

##### 【最终结论】
最优解由上三角线性方程组 $R_A \Delta\mathcal{U}(k)_{opt} = Q_1^T b(k)$ 经**一次回代法（Backward Substitution）**直接解出。完全无需对矩阵求逆，且算法条件数保持为 $\kappa(\mathcal{A}) = \kappa(\Theta)$，数值稳定性显著提升！

---

#### 3.1.3 无约束控制器结构与内生积分作用推导 (§3.1.3)

##### 【解题思路】
将最优控制增量 $\Delta u(k)$ 展开为参考设定点、系统状态与历史输入的线性组合，推导出等效控制器增益三元组 $(K_r, K_x, K_u)$，并证明其在闭环中内生具备消除常值稳态误差的积分作用。

##### 【逐步推导】
1. **控制增量展开**：
   定义第一步提取矩阵 $S_1 = [I_\ell, 0, \dots, 0]$，定义 MPC 滤波增益矩阵：
   $$
   K_{mpc} \triangleq S_1 (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q} \in \mathbb{R}^{\ell \times m H_p}
   $$
   则实际控制增量为（原书式 (3.25)）：
   $$
   \Delta u(k) = K_{mpc} \mathcal{E}(k) = K_{mpc} [\mathcal{T}(k) - \Psi x(k) - \Upsilon u(k-1)] \tag{3.26}
   $$

2. **分离状态反馈与参考前馈**：
   定义状态反馈增益：
   $$
   K_x \triangleq K_{mpc} \Psi \in \mathbb{R}^{\ell \times n}
   $$
   定义历史输入增益：
   $$
   K_u \triangleq K_{mpc} \Upsilon \in \mathbb{R}^{\ell \times \ell}
   $$
   若参考目标在时域内为常数设定值 $r(k)$，即 $\mathcal{T}(k) = [\begin{matrix} I_m & I_m & \cdots & I_m \end{matrix}]^T r(k)$，定义设定点增益：
   $$
   K_r \triangleq K_{mpc} (\mathbf{1}_{H_p} \otimes I_m) \in \mathbb{R}^{\ell \times m}
   $$
   增量控制律写为（原书式 (3.27)）：
   $$
   \Delta u(k) = K_r r(k) - K_x x(k) - K_u u(k-1)
   $$

3. **绝对物理输入与闭环积分器**：
   物理输入为（原书式 (3.28)）：
   $$
   u(k) = u(k-1) + \Delta u(k) = (I_\ell - K_u) u(k-1) - K_x x(k) + K_r r(k)
   $$
   移项写为 $z$-变换传递函数形式：
   $$
   (1 - z^{-1}) u(k) = \Delta u(k) \implies u(k) = \frac{1}{1 - z^{-1}} \Delta u(k)
   $$

##### 【最终结论】
在无约束情况下，MPC 控制器严格退化为**线性时不变（LTI）控制器**（原书 Figure 3.1 框图）。由于控制决策建立在增量 $\Delta u$ 上，前向通道天然串联了一只纯积分器 $\frac{1}{1 - z^{-1}}$，保证了系统对阶跃设定点具有天生的**无静差跟踪特性**！

---

### Mini-Tutorial 3: 凸优化、二次规划与 KKT 条件严密推导

#### 【解题思路】
当加入物理约束后，无约束解析解失效，优化问题转化为二次规划（QP）。为理解为什么局部极小就是全局极小，以及算法如何判断解的最优性，本节从初等几何与 Lagrange 函数出发，严密推导 Karush-Kuhn-Tucker (KKT) 四大一阶必要条件。

#### 【逐步推导】
1. **凸集与凸函数定义**：
   - **凸集 $S$**：集合内任意两点的连线段仍完全属于集合：$\forall x_1, x_2 \in S, \alpha \in [0, 1] \implies \alpha x_1 + (1-\alpha) x_2 \in S$。
   - **凸函数 $f(x)$**：连接曲面上任意两点的割线始终位于曲面上方。
   - **凸优化的金科玉律**：在凸集上极小化凸函数，**局部极小解必然是全局最优解！**

2. **约束极值与法向量平衡几何分析**：
   考虑优化问题：
   $$
   \min_\theta f(\theta) \quad \text{s.t.} \quad g_i(\theta) \le 0, \quad i = 1, \dots, n_c
   $$
   在最优点 $\theta^*$ 处，若约束 $g_i(\theta^*) = 0$ 处于激活状态（Active），则目标函数的下降方向 $-\nabla f(\theta^*)$ 必然被约束边界顶住。也就是说，目标函数的负梯度必须能被激活约束的外法向量 $\nabla g_i(\theta^*)$ 线性表出且系数非负：
   $$
   -\nabla f(\theta^*) = \sum_{i \in \mathcal{W}} \lambda_i \nabla g_i(\theta^*), \quad \lambda_i \ge 0
   $$

3. **构造 Lagrange 函数并系统化一阶条件**：
   $$
   L(\theta, \lambda) \triangleq f(\theta) + \sum_{i=1}^{n_c} \lambda_i g_i(\theta) = f(\theta) + \lambda^T g(\theta)
   $$
   对于标准 QP 问题（$f(\theta) = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta$，$g(\theta) = \Omega \theta - \omega \le 0$）：
   $$
   L(\theta, \lambda) = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \lambda^T (\Omega \theta - \omega)
   $$
   对 $\theta$ 求梯度令其为零，并结合对偶乘子性质，得到著名的四大 **KKT 条件**（原书式 (3.59)–(3.62)）：
   - **(1) 驻点平衡条件 (Stationarity)**：
     $$
     \nabla_\theta L = \Phi \theta^* + \phi + \Omega^T \lambda^* = 0
     $$
   - **(2) 原可行性条件 (Primal Feasibility)**：
     $$
     \Omega \theta^* \le \omega
     $$
   - **(3) 对偶可行性条件 (Dual Feasibility)**：
     $$
     \lambda^* \ge 0
     $$
   - **(4) 互补松弛条件 (Complementary Slackness)**：
     $$
     \lambda_i^* (\Omega_i \theta^* - \omega_i) = 0, \quad \forall i = 1, \dots, n_c
     $$

##### 【最终结论】
互补松弛条件的物理意义极其直观：
- 若约束未激活（$\Omega_i \theta^* < \omega_i$，处于安全内部），则对偶乘子**必须严格为零（$\lambda_i^* = 0$）**，约束不起作用；
- 若约束已激活（$\Omega_i \theta^* = \omega_i$，顶在物理边界上），则对偶乘子**可以为严格正数（$\lambda_i^* > 0$）**，代表边界对系统施加的法向顶阻力！

---

### 3.2 约束问题与分段仿射 (PWA) 结构 (§3.2)

#### 3.2.1 物理约束标准化为二次规划
物理系统存在三大线性约束：变化率约束 (3.40)、输入幅值约束 (3.38)、输出幅值约束 (3.39)。
堆叠为总约束不等式（原书式 (3.41)）：
$$
\Omega \Delta\mathcal{U}(k) \le \omega(k)
$$
令优化决策变量 $\theta \triangleq \Delta\mathcal{U}(k)$，将目标函数 (3.10) 乘以系数 $\frac{1}{2}$ 转化为标准二次规划格式（原书式 (3.43)–(3.44)）：
$$
\min_\theta \mathcal{J}(\theta) = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta \quad \text{s.t.} \quad \Omega \theta \le \omega
$$
其中 QP 参数矩阵与 MPC 物理模型参数的对应映射严格为：
$$
\Phi = 2 H = 2(\Theta^T \mathcal{Q} \Theta + \mathcal{R}) \tag{3.45}
$$
$$
\phi(k) = -G = -2 \Theta^T \mathcal{Q} \mathcal{E}(k) \tag{3.46}
$$

#### 3.2.2 约束控制器结构与分段仿射 (PWA) 几何解释（原书 Figure 3.3–3.5 详析）
- **非线性控制器结构（Figure 3.3）**：引入约束后，闭环控制器的线性反馈框图被打破，控制器核心转变为一个**在线 QP 求解器**。
- **几何区域划分与分段仿射（Figure 3.4 & 3.5）**：
  状态空间被不同的有效约束组合划分成若干个多面体临界区域（Critical Regions）。在每一个区域内部，激活约束固定，控制器表现为一个局部的线性状态反馈律；当状态穿越边界切换到另一组约束激活时，控制器增益发生跳变。因此，**带约束的预测控制器在本质上是一个非线性、分段仿射（PWA）切换控制器！**

---

### 3.3 求解 QP 问题 (Solving QP problems)

#### 3.3.1 有效集法 (Active Set Methods)

##### 【解题思路】
有效集法是一种极度贴合工程直觉的迭代方法。其核心逻辑是：**如果我们猜测出哪些不等式约束实际上被顶满了（等式化），把这些约束作为等式，其余约束暂时抛弃，不等式 QP 就退化为极其容易求解的等式约束二次规划（EQP）！**  
当前假定为激活的约束集合称为**工作集 $\mathcal{W}$**。算法通过在当前工作集下求搜索步长 $p$，结合步长截断比率测试（Ratio Test）添加新阻挡约束，或根据乘子负号释放多余约束，有限步内必收敛至全局最优解。

##### 【逐步推导】
1. **求解等式约束子问题 (EQP)**：
   设当前迭代点为 $\theta_k$，工作集为 $\mathcal{W}_k$。定义搜索步长 $p$（$\theta = \theta_k + p$），目标函数展开：
   $$
   \min_p \frac{1}{2} p^T \Phi p + g_k^T p \quad \text{s.t.} \quad \Omega_{\mathcal{W}} p = 0
   $$
   其中 $g_k = \Phi \theta_k + \phi$ 为当前梯度。构造 KKT 系统：
   $$
   \begin{bmatrix} \Phi & \Omega_{\mathcal{W}}^T \\ \Omega_{\mathcal{W}} & 0 \end{bmatrix} \begin{bmatrix} p \\ \mu \end{bmatrix} = \begin{bmatrix} -g_k \\ 0 \end{bmatrix}
   $$

2. **分支判定与步长更新**：
   - **分支 A（$p \ne 0$，尚未到达当前工作集的极小点）**：
     计算沿方向 $p$ 前进的最大允许步长比例 $\alpha_k \in [0, 1]$，防止冲破未激活约束 $\Omega_i (\theta_k + \alpha p) \le \omega_i$（$i \notin \mathcal{W}$）：
     $$
     \alpha_k = \min \left\{ 1, \min_{i \notin \mathcal{W}, \Omega_i p > 0} \frac{\omega_i - \Omega_i \theta_k}{\Omega_i p} \right\}
     $$
     *若 $\alpha_k = 1$*：直接走满，$\theta_{k+1} = \theta_k + p$；  
     *若 $\alpha_k < 1$*：撞到某个阻挡约束（Blocking Constraint）。更新 $\theta_{k+1} = \theta_k + \alpha_k p$，并将对应约束索引**加入工作集 $\mathcal{W}$**。
   - **分支 B（$p = 0$，已到达当前工作集的局部驻点）**：
     计算工作集对应的 Lagrange 乘子：$\Omega_{\mathcal{W}}^T \lambda_{\mathcal{W}} = -(\Phi \theta_k + \phi)$。
     *若所有 $\lambda_i \ge 0$*：满足 KKT 全部条件，**算法终止，当前点即为全局最优解！**  
     *若存在某个 $\lambda_j < 0$*：说明约束 $j$ 阻碍了解向更好的目标下降。**将具有最负乘子的约束 $j$ 从工作集中剔除**，转入下一轮迭代。

##### 【最终结论】
有效集法每步只解一个规模较小的线性方程组，在初始点接近最优解时（即热启动 Warm-start）通常只需 1–3 步即可收敛，极其适合中小规模在线 MPC。

---

#### 3.3.2 Example 3.1（Fletcher QP 问题详细手算与几何轨迹）

##### 【题目设定】
原书式 (3.63)–(3.64)：
$$
\min_{\theta_1, \theta_2} V = \theta_1^2 - \theta_1 \theta_2 + \theta_2^2 - 3 \theta_1
$$
受制于 4 个线性不等式约束：
$$
(1) -\theta_1 \le 0, \quad (2) -\theta_2 \le 0, \quad (3) \theta_1 + \theta_2 \le 2, \quad (4) -\theta_1 + \theta_2 \le 1
$$
标准 QP 矩阵为：
$$
\Phi = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}, \quad \phi = \begin{bmatrix} -3 \\ 0 \end{bmatrix}, \quad \Omega = \begin{bmatrix} -1 & 0 \\ 0 & -1 \\ 1 & 1 \\ -1 & 1 \end{bmatrix}, \quad \omega = \begin{bmatrix} 0 \\ 0 \\ 2 \\ 1 \end{bmatrix}
$$

##### 【逐步手算推导】
1. **无约束极小点检验**：
   $$
   \nabla V = \Phi \theta + \phi = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix} + \begin{bmatrix} -3 \\ 0 \end{bmatrix} = 0 \implies \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix}_{unc} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
   $$
   代入约束 (3)：$\theta_1 + \theta_2 = 2 + 1 = 3 > 2$，**严重越界违反约束！**

2. **从原点 $\theta^{(0)} = [0, 0]^T$ 启动有效集法**：
   - 初始工作集取边界约束：$\mathcal{W}_0 = \{ (1), (2) \}$。当前点满足等式边界。
   - 在原点处步长 $p=0$。计算乘子：
     $$
     \Omega_{\mathcal{W}}^T \lambda = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \end{bmatrix} = -\phi = \begin{bmatrix} 3 \\ 0 \end{bmatrix} \implies \lambda_1 = -3 < 0, \lambda_2 = 0
     $$
     乘子 $\lambda_1 = -3 < 0$，说明约束 (1) 应该被释放！将约束 (1) 移出工作集，新工作集 $\mathcal{W}_1 = \{ (2) \}$（即沿 $\theta_2 = 0$ 轴前进）。

3. **第 1 次沿边界移动**：
   工作集为 $\theta_2 = 0$。求解 EQP 得到搜索方向 $p = [1.5, 0]^T$。
   测试未激活约束阻挡：
   - 约束 (3) 边界：$\theta_1 + 0 \le 2 \implies \alpha_{max} = \frac{2 - 0}{1.5} = \frac{4}{3} > 1$。
   故可前进一步到达 $\theta^{(1)} = [2, 0]^T$。此时约束 (3) 被撞击激活！工作集更新为 $\mathcal{W}_2 = \{ (3) \}$。

4. **第 2 次沿约束 (3) 边界滑动**：
   在约束 (3) 边界 $\theta_1 + \theta_2 = 2$ 上滑动。
   求解 EQP 得到最优搜索方向到达 $\theta^* = [1.5, 0.5]^T$。
   在此点计算乘子：
   $$
   \Phi \theta^* + \phi + \Omega_{(3)}^T \lambda_3 = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 1.5 \\ 0.5 \end{bmatrix} + \begin{bmatrix} -3 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 1 \end{bmatrix} \lambda_3 = \begin{bmatrix} 2.5 - 3 \\ -0.5 \end{bmatrix} + \begin{bmatrix} \lambda_3 \\ \lambda_3 \end{bmatrix} = 0
   $$
   解得：$\lambda_3 = 0.5 > 0$！全部乘子非负，满足 KKT 条件！

##### 【最终结论】
全局最优解为 $\theta_1^* = 1.5, \theta_2^* = 0.5$，最优目标函数值为 $V^* = -2.75$。此过程在脚本 `ch3_qp_active_set_from_scratch.py` 中被完整复现（见图 [`ch3_active_set_trajectory.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_active_set_trajectory.png)）。

---

#### 3.3.3 主对偶内点法 (Primal-Dual Interior-Point Methods)

##### 【解题思路】
当物理系统约束数量达到数百个时，有效集法的工作集组合搜索可能发生退化。主对偶内点法通过引入对数障碍函数（Logarithmic Barrier），迫使迭代点始终在可行域的**严格内部（Interior）**沿中心路径（Central Path）平滑滑向最优解。

##### 【逐步推导】
1. **引入松弛变量与对数障碍目标函数**：
   引入严格正松弛变量 $s > 0$，将不等式转化为等式：$\Omega \theta + s = \omega$。
   构建对数障碍目标函数（原书式 (3.74)）：
   $$
   \min_{\theta, s} B(\theta, s; \mu) = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta - \mu \sum_{i=1}^{n_c} \ln s_i \quad \text{s.t.} \quad \Omega \theta + s = \omega
   $$
   其中 $\mu > 0$ 为障碍参数。当任何 $s_i \to 0^+$（逼近边界）时，$-\ln s_i \to +\infty$，构成无限高能量势垒，阻止迭代点冲出可行域。

2. **扰动 KKT 条件方程构建**：
   构造 Lagrange 函数：
   $$
   L = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta - \mu \sum_{i=1}^{n_c} \ln s_i + \lambda^T (\Omega \theta + s - \omega)
   $$
   求梯度令其为零，得到**扰动 KKT 系统（Perturbed KKT System）**（原书式 (3.76)–(3.78)）：
   $$
   \begin{aligned}
   \nabla_\theta L &= \Phi \theta + \phi + \Omega^T \lambda = 0 \\
   \nabla_\lambda L &= \Omega \theta + s - \omega = 0 \\
   S \Lambda \mathbf{1} &= \mu \mathbf{1} \quad (s_i \lambda_i = \mu)
   \end{aligned}
   $$
   其中 $S = \text{diag}(s_1, \dots, s_{n_c})$，$\Lambda = \text{diag}(\lambda_1, \dots, \lambda_{n_c})$。

3. **Newton-Raphson 线性化系统**：
   设当前点为 $(\theta, \lambda, s)$，寻找更新步长 $(\Delta\theta, \Delta\lambda, \Delta s)$。一阶线性化展开：
   $$
   \begin{bmatrix} \Phi & \Omega^T & 0 \\ \Omega & 0 & I \\ 0 & S & \Lambda \end{bmatrix} \begin{bmatrix} \Delta\theta \\ \Delta\lambda \\ \Delta s \end{bmatrix} = \begin{bmatrix} -(\Phi \theta + \phi + \Omega^T \lambda) \\ -(\Omega \theta + s - \omega) \\ \mu \mathbf{1} - S \Lambda \mathbf{1} \end{bmatrix} \tag{3.79}
   $$
   消去 $\Delta s = -(\Omega \theta + s - \omega) - \Omega \Delta\theta$，只需求解一个对称正定线性方程组。

4. **边界截断规则与中心路径追踪**：
   为保持 $s > 0, \lambda > 0$ 严格成立，步长采用**边界比例截断规则（Fraction-to-the-boundary Rule）**（$\tau \approx 0.995$）：
   $$
   \alpha_s = \min \left\{ 1, \tau \min_{\Delta s_i < 0} \frac{-s_i}{\Delta s_i} \right\}, \quad \alpha_\lambda = \min \left\{ 1, \tau \min_{\Delta \lambda_i < 0} \frac{-\lambda_i}{\Delta \lambda_i} \right\}
   $$
   随着外层循环逐步降低障碍参数 $\mu \to 0$，解族光滑逼近真实边界最优解，形成**中心路径（Central Path）**（原书 Figure 3.6，对应本章 [`ch3_interior_point_central_path.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_interior_point_central_path.png)）。

##### 【最终结论】
内点法每一拍的迭代次数几乎与约束规模无关（通常 10–15 步稳定收敛），是处理大规模复杂工业多变量 MPC 的利器。

---

### 3.4 约束软化与精确惩罚定理 (§3.4)

#### 3.4.1 为什么绝对硬约束在工程上会导致毁灭性崩溃？
在实际工业现场中，如果施加的是**绝对硬约束（Hard Constraints）**，一旦遭遇模型失配、阵风扰动或操作工下达激进阶跃，系统状态可能被瞬间抛入几何可行域外部。在数学上，此时满足 $\Omega \theta \le \omega$ 的解集是**空集**！硬约束 QP 求解器将直接报 `Optimization Infeasible`（优化不可行）并异常退出，导致执行机构失去控制信号，产生毁灭性飞行事故或化工厂停车（如原书 Example 3.2 与 Figure 3.7 所展示的 Cessna 飞机在 $t=13$s 处崩溃）。

#### 3.4.2 松弛变量与三类罚函数形式
引入非负松弛变量（Slack Variables）$\varepsilon \ge 0$，将约束软化为：
$$
\Omega \theta \le \omega + \mathcal{V}_\varepsilon \varepsilon \tag{3.86}
$$

1. **2-范数罚函数（2-norm Penalty，原书式 3.87）**：
   $$
   \mathcal{J}_{soft} = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \rho \|\varepsilon\|_2^2
   $$
   - **致命缺陷（不精确）**：2-范数在 $\varepsilon = 0$ 处导数为零。这意味着**即便系统在硬约束下完全可行，为了微弱降低主目标，松弛变量 $\varepsilon$ 也会自发变为微小正数！** 导致约束被提前人为软化。
2. **1-范数罚函数（1-norm Penalty，原书式 3.88）**：
   $$
   \mathcal{J}_{soft} = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \rho \sum_{i=1}^{n_c} \varepsilon_i
   $$
3. **$\infty$-范数罚函数（$\infty$-norm Penalty，原书式 3.90）**：
   只需引入**单一标量松弛变量** $\epsilon \ge 0$：
   $$
   \mathcal{J}_{soft} = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \rho \epsilon \quad \text{s.t.} \quad \Omega \theta \le \omega + \mathbf{1} \epsilon \tag{3.91}
   $$
   仅给 QP 增加 1 个决策变量，计算代价极低，且专惩最大违背峰值。

#### 3.4.3 精确罚函数定理 (Exact Penalty Theorem) 严密推导

##### 【解题思路】
1-范数与 $\infty$-范数之所以在工业界被奉为标准，是因为它们具备**精确罚函数（Exact Penalty Function）**的优异数学性质：只要惩罚权重 $\rho$ 取得足够大，当硬约束问题可行时，软约束的最优解与硬约束最优解**100% 严格重合，松弛变量恒等于零！**

##### 【逐步推导】
1. **构造硬约束问题与软约束问题的对偶联系**：
   设原硬约束 QP 在最优解处的 Lagrange 乘子为 $\lambda^*$。考虑带松弛的软约束目标：
   $$
   \min_{\theta, \epsilon \ge 0} \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \rho \epsilon \quad \text{s.t.} \quad \Omega_i \theta - \omega_i \le \epsilon, \quad \forall i
   $$
2. **考察松弛变量 $\epsilon$ 的 KKT 偏导驻点条件**：
   写出关于增广变量 $(\theta, \epsilon)$ 的 Lagrange 函数：
   $$
   L = \frac{1}{2} \theta^T \Phi \theta + \phi^T \theta + \rho \epsilon + \sum_{i=1}^{n_c} \lambda_i (\Omega_i \theta - \omega_i - \epsilon) - \nu \epsilon
   $$
   其中 $\nu \ge 0$ 为 $\epsilon \ge 0$ 约束的 Lagrange 乘子。对标量 $\epsilon$ 求导令其为零：
   $$
   \frac{\partial L}{\partial \epsilon} = \rho - \sum_{i=1}^{n_c} \lambda_i - \nu = 0 \implies \nu = \rho - \sum_{i=1}^{n_c} \lambda_i
   $$
3. **互补松弛性分析**：
   根据 KKT 互补松弛条件：$\nu \epsilon = 0$。
   - 若设定惩罚因子满足临界阈值条件：
     $$
     \rho > \sum_{i=1}^{n_c} \lambda_i^* \ge \max_i |\lambda_i^*| \tag{3.92}
     $$
   - 则必有 $\nu = \rho - \sum \lambda_i^* > 0$（严格大于零）！
   - 因为 $\nu > 0$，为了满足 $\nu \epsilon = 0$，**松弛变量必须严格为零：$\epsilon^* \equiv 0$！**

##### 【最终结论】
只要惩罚系数 $\rho > \max_i |\lambda_i^*|$，软约束的最优解与硬约束最优解严格一致，松弛变量保持为零。只有当外部扰动使硬约束变得物理不可行时，$\epsilon^*$ 才被迫大于零，以最小代价帮助系统平稳脱险！

---

## 4. 本章综合推导链

本章理论推导逻辑构成了不可分割的闭环：

```
[4.1 目标性能指标二次型展开]
  V(k) = ||Z-T||_Q^2 + ||ΔU||_R^2  ──>  V(k) = const - ΔU^T G + ΔU^T H ΔU
                          │
                          ▼
[4.2 无约束解析求解与 LTI 控制律]
  驻点条件 ∇V = 0  ──>  ΔU_opt = H^-1 Θ^T Q E(k)  ──(QR 分解求解)──> Δu(k) = Kr r - Kx x - Ku u
  内嵌纯离散积分器保证无静差跟踪
                          │
                          ▼
[4.3 物理约束多面体标准化]
  变化率约束 + 幅值约束 + 输出约束  ──>  Ω ΔU(k) ≤ ω(k)  ──>  标准凸二次规划 QP
                          │
                          ▼
[4.4 在线二次规划求解算法]
  ├─ 有效集法 (Active Set): 迭代管理工作集，比率测试，乘子释放，适合暖启动
  └─ 内点法 (Interior Point): 对数势垒阻挡，中心路径追踪，大规模多项式复杂度
                          │
                          ▼
[4.5 工业防崩溃底线: 约束软化]
  引入松弛变量 ε ≥ 0  ──>  精确罚函数临界定理 ρ > max |λ_i*|  ──>  100% 保证可行解
```

---

## 5. Python 实现体系与代码架构说明

为彻底覆盖本章全部算法与原书例题，我们在目录 [`/home/kuang/Desktop/mpc/ch03_solving_qp/`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/) 下构建了 **5 个相互独立、职责清晰的可运行 Python 脚本**：

```
/home/kuang/Desktop/mpc/ch03_solving_qp/
├── ch3_unconstrained_lq.py                # 5.1 无约束 MPC 增益解析与泳池温度闭环仿真
├── ch3_qp_active_set_from_scratch.py      # 5.2 从零实现原生长基有效集 QP 求解器 (Example 3.1)
├── ch3_interior_point_from_scratch.py     # 5.3 从零实现原生主对偶内点法 QP 求解器 (Figure 3.6)
├── ch3_constraint_softening.py            # 5.4 约束软化：硬约束崩溃 vs 软约束精确恢复 (Figure 3.7-3.10)
└── ch3_comprehensive_benchmark.py         # 5.5 QP 求解器全性能基准评测：有效集 vs 内点法 vs SLSQP
```

### 5.1 脚本 1：`ch3_unconstrained_lq.py`
- **对应小节**：3.1 节、Exercise 3.3（恒温游泳池连续模型离散化与闭环控制）。
- **核心功能**：解析推导无约束预测控制增益向量 $K_{mpc}$，分离状态增益 $K_x$、历史输入增益 $K_u$、设定点增益 $K_r$；精确复现 Exercise 3.3 中的标准增益 $K_s = [22.604, -17.604, -22.604]$；仿真昼夜正弦环境温度扰动下系统的天生无静差消除，生成 [`ch3_pool_temperature.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_pool_temperature.png)。

### 5.2 脚本 2：`ch3_qp_active_set_from_scratch.py`
- **对应小节**：3.3.1 节、Mini-Tutorial 3、Example 3.1（Fletcher QP）。
- **核心功能**：纯 NumPy 从零手写原始有效集法（Primal Active Set QP Solver），不调用任何优化包；实现 EQP 子问题投影求解、步长比率测试（Ratio Test）、Lagrange 乘子符号检测与工作集动态释放；完整绘制从 $[0,0]^T \to [2,0]^T \to [1.5, 0.5]^T$ 的数值收敛轨迹图 [`ch3_active_set_trajectory.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_active_set_trajectory.png)。

### 5.3 脚本 3：`ch3_interior_point_from_scratch.py`
- **对应小节**：3.3.2 节、原书 Figure 3.6。
- **核心功能**：纯 NumPy 从零手写主对偶内点法（Primal-Dual Interior-Point QP Solver）；实现扰动 KKT 条件、Newton-Raphson 线性化增量系统、中心路径追踪与边界截断规则；绘制不同初始点沿中心对偶脊线逼近最优边界的几何轨迹图 [`ch3_interior_point_central_path.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_interior_point_central_path.png)。

### 5.4 脚本 4：`ch3_constraint_softening.py`
- **对应小节**：3.4 节、Example 3.2、Example 3.3、Figures 3.7–3.10。
- **核心功能**：Cessna Citation 500 飞行器高度控制；注入升降舵增益模型失配，复现硬约束 QP 在 $t = 13$s 处的不可行性崩溃；实现 1-范数、2-范数与 $\infty$-范数软约束，精确再现大惩罚系数 $\rho = 5 \times 10^6$ 下的无失真平滑过渡与小惩罚系数 $\rho = 10^3$ 下的约束过度软化，生成对比图 [`ch3_constraint_softening_comparison.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_constraint_softening_comparison.png)。

### 5.5 脚本 5：`ch3_comprehensive_benchmark.py`
- **对应小节**：3.3 节、原书 Table 3.2。
- **核心功能**：随机生成 20 个高维正定 QP 优化算例；横向评测纯手写有效集法、纯手写内点法与 SciPy `SLSQP` 在求解耗时、迭代步数与最优性误差上的综合表现，生成箱线图与散点图 [`ch3_benchmark_performance.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_benchmark_performance.png)。

---

## 6. 数值实验与结果深度验证

| 实验编号 | 测试目标 | 参数设置与扰动 | 理论预期现象 | 代码与图像定位 |
| :--- | :--- | :--- | :--- | :--- |
| **实验 1** | 无约束 LTI 增益与扰动抑制验证 | $H_p=10, H_u=3, Q=1, R=0$, 昼夜温差正弦扰动 | 控制增益精确匹配 $[22.604, -17.604, -22.604]$，水温均值无静差 | `ch3_unconstrained_lq.py`，[`ch3_pool_temperature.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_pool_temperature.png) |
| **实验 2** | 有效集 QP 求解器收敛轨迹 | Fletcher QP 4 约束，初始点 $[0, 0]^T$ | 2 步迭代收敛于 $[1.5, 0.5]^T$，最优值 $-2.75$ | `ch3_qp_active_set_from_scratch.py`，[`ch3_active_set_trajectory.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_active_set_trajectory.png) |
| **实验 3** | 内点法中心路径追踪与对偶收敛 | Fletcher QP，初始内部可行点 $[0.5, 0.2]^T$ | 沿中心对偶脊线平滑逼近边界最优解，无振荡 | `ch3_interior_point_from_scratch.py`，[`ch3_interior_point_central_path.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_interior_point_central_path.png) |
| **实验 4** | 飞机模型失配下硬约束崩溃与软约束解救 | Citation 500 模型失配，升降舵与高度率受限 | 硬约束在 $t=13$s 崩溃；$\infty$-范数软约束平滑穿越饱和区 | `ch3_constraint_softening.py`，[`ch3_constraint_softening_comparison.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_constraint_softening_comparison.png) |
| **实验 5** | 有效集 vs 内点法计算速度基准横评 | 20 个独立正定 QP，维度 $N=6, n_c=12$ | 内点法迭代次数方差极小；有效集在暖启动下具备高潜能 | `ch3_comprehensive_benchmark.py`，[`ch3_benchmark_performance.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_benchmark_performance.png) |

---

## 7. 原书习题全解与精析 (Exercises 3.1 - 3.12 逐题极致推导，绝不省略)

### Exercise 3.1：无约束状态空间 MPC 工具箱实现 (`smpccon`)
- **题目要求**：按照 MPC Toolbox 用户手册的指导，完成“基于状态空间模型的无约束 MPC”例程。熟悉工具箱函数 `smpccon` 的调用与增益计算。

#### 【解题思路】
`smpccon` 本质上执行本章推导的式 (3.14) 与 (3.25)。输入状态空间模型 $(A, B, C)$、时域 $(H_p, H_u)$ 与权重 $(Q, R)$，离线构造 $\Theta, \Psi, \Upsilon$，并计算无约束控制器增益矩阵：
$$
K_{mpc} = S_1 (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q}
$$
其返回的综合增益为 $K_s = [K_r, -K_x, -K_u] = K_{mpc} [\mathbf{1}, -\Psi, -\Upsilon]$。

#### 【逐步推导】
1. 建立模型状态空间对象；
2. 定义加权矩阵 $\mathcal{Q} = \text{diag}(Q), \mathcal{R} = \text{diag}(R)$；
3. 调用 `smpccon` 计算增益矩阵 $K_s$；
4. 验证闭环状态方程 $x(k+1) = (A - B K_x) x(k) - B K_u u(k-1) + B K_r r(k)$ 的特征多项式根全部落在单位圆内。

#### 【最终结论】
该例程证明了无约束 MPC 在数值上等价于静态线性状态反馈与内模积分器的组合。

---

### Exercise 3.2：带约束状态空间 MPC 工具箱实现 (`scmpc`)
- **题目要求**：按照 MPC Toolbox 用户手册的指导，完成“带约束的状态空间 MPC”例程，熟悉函数 `scmpc` 的调用。

#### 【解题思路】
`scmpc` 在每一拍内部动态组装标准 QP 矩阵 $\Phi, \phi(k), \Omega, \omega(k)$，调用二次规划求解器迭代计算当拍最优动作 $\Delta u(k)$，并施加执行器限幅。

#### 【逐步推导】
1. 定义硬约束上下限向量 $u_{min}, u_{max}, \Delta u_{max}, y_{min}, y_{max}$；
2. 构造每一拍的时变右端项 $\omega(k)$；
3. 调用 QP 求解器解算 $\Delta\mathcal{U}^*(k)$，提取第一项 $\Delta u(k) = \Delta u^*(0)$；
4. 更新实际对象并推进时间步。

#### 【最终结论】
当约束未激活时，`scmpc` 输出与 `smpccon` 完全一致；当约束激活时，系统平滑贴合约束边界运行。

---

### Exercise 3.3：恒温游泳池连续模型离散化与闭环预测控制完整推导
- **题目要求**：水温 $\theta$、加热功率 $q$、环境温度 $\theta_a$ 满足微分方程：
  $$
  T \frac{d\theta}{dt} = k_q q + \theta_a - \theta \tag{3.93}
  $$
  其中 $T = 1.0$ 小时，$k_q = 0.2$ °C/kW。采样周期 $T_s = 0.25$ 小时。
  (a) 推导离散时间模型 $\theta(k+1) = 0.7788 \theta(k) + 0.0442 q(k) + 0.2212 \theta_a(k)$；
  (b) 验证权重 $Q = 1, R = 0, H_p = 10, H_u = 3$ 时，增益为 $K_s = [22.604, -17.604, -22.604]$，且闭环稳定；
  (c) 验证环境温度 $\theta_a = 15$ °C 时设定点 $20$ °C 无静差达成，且当真实参数变为 $T = 1.25, k_q = 0.3$ 时仍保持无静差；
  (d) 验证昼夜正弦扰动 $\theta_a(t) = 15 + 10 \sin(2\pi t / 24)$ 下平均水温无静差，水温残余波动振幅约为 $0.5$ °C；
  (e) 在功率限幅 $0 \le q \le 40$ kW 下分析闭环性能。

#### 【解题思路】
利用零阶保持器解析积分求出离散标量差分方程；利用无约束预测控制公式计算 $K_{mpc}$ 与增益三元组；利用内模原理证明模型失配下的无静差特性；利用频率响应分析正弦扰动下的残余稳态幅值；利用 QP 求解器仿真输入硬饱和。

#### 【逐步推导】
1. **(a) 连续到离散精确积分**：
   $$
   \frac{d\theta}{dt} = -1.0 \theta(t) + 0.2 q(t) + 1.0 \theta_a(t)
   $$
   - 极点转移系数：
     $$
     A_d = e^{-1.0 \times 0.25} = e^{-0.25} \approx 0.77880078
     $$
   - 控制输入积分：
     $$
     B_d = \int_0^{0.25} e^{-1.0 \cdot \tau} d\tau \times 0.2 = (1 - e^{-0.25}) \times 0.2 = (1 - 0.77880078) \times 0.2 = 0.0442398
     $$
   - 环境温度扰动输入积分：
     $$
     E_d = \int_0^{0.25} e^{-1.0 \cdot \tau} d\tau \times 1.0 = 1 - e^{-0.25} = 0.221199
     $$
   得到离散方程：
   $$
   \theta(k+1) = 0.7788 \theta(k) + 0.0442 q(k) + 0.2212 \theta_a(k) \tag{3.94}
   $$

2. **(b) 控制增益矩阵推导**：
   - 自由响应矩阵 $\Psi = [A_d, A_d^2, \dots, A_d^{10}]^T \in \mathbb{R}^{10 \times 1}$；
   - 强迫响应矩阵 $\Theta \in \mathbb{R}^{10 \times 3}$：
     第 1 列元素为 $S(i) = B_d \sum_{j=0}^{i-1} A_d^j$；第 2 列下移 1 格；第 3 列下移 2 格。
   - 因 $R = 0$，Hessian 矩阵为 $H = \Theta^T \Theta \in \mathbb{R}^{3 \times 3}$。
   - 计算得到首行增益向量 $K_{mpc} = [1, 0, 0] (\Theta^T \Theta)^{-1} \Theta^T$。
   - 对应分量乘积：
     $$
     K_r = K_{mpc} \mathbf{1} = 22.604
     $$
     $$
     K_x = K_{mpc} \Psi = 17.604
     $$
     $$
     K_u = K_{mpc} \Upsilon = 22.604
     $$
   - 增益三元组为 $K_s = [22.604, -17.604, -22.604]$（原书式 (3.95)）。
   - 闭环特征根计算：闭环状态转移矩阵为 $\begin{bmatrix} A_d - B_d K_x & -B_d K_u \\ -K_x & 1 - K_u \end{bmatrix}$，特征根模长均小于 1，系统严格稳定。

3. **(c) 模型失配下的无静差特性证明**：
   真实参数改变为 $T = 1.25$h，$k_q = 0.3$ °C/kW，增益被严重高估。
   在常值设定点与常温下，只要闭环渐近稳定，稳态必有 $\Delta q(\infty) = 0$。由式 (3.27)：
   $$
   \Delta q(\infty) = K_r r - K_x \theta(\infty) - K_u q(\infty) = 0
   $$
   结合积分器 $q(\infty) = q(\infty)$，误差被严格迫零，稳态水温精确收敛到 $20$ °C。

4. **(d) 昼夜正弦扰动的残余幅值推导**：
   扰动角频率 $\omega = \frac{2\pi}{24} \approx 0.2618$ rad/h。
   闭环系统对环境温度的灵敏度传递函数 $S_{dist}(j\omega)$ 在低频段具有高衰减率。代入频域增益计算得幅值衰减比约为 $0.05$。输入扰动振幅为 $10$ °C，故输出稳态波动振幅约为 $10 \times 0.05 = 0.5$ °C。

5. **(e) 输入功率硬饱和仿真**：
   在 $0 \le q \le 40$ kW 限幅下，夜晚升温动作受限，过渡过程轻微拉长，但白天恢复无超调，系统无失稳风险。

#### 【最终结论】
完全解析推导并数值复现了泳池模型的各项特性，详见脚本 `ch3_unconstrained_lq.py` 与结果图 [`ch3_pool_temperature.png`](file:///home/kuang/Desktop/mpc/ch03_solving_qp/ch3_pool_temperature.png)。

---

### Exercise 3.4：输入分块技术 (Input Blocking) 的预测矩阵修改
- **题目要求**：若采用输入分块技术，预测矩阵 $\Theta$ 应如何修改？

#### 【解题思路】
分块技术将控制时域内的增量强制约束为分段恒定，数学上等价于定义一个基变换映射矩阵 $T_{block}$，将高维增量映射到低维块增量。

#### 【逐步推导】
1. 设优化决策变量为低维向量 $\Delta\mathcal{V}(k) \in \mathbb{R}^{\ell N_{block}}$。
2. 构造分块展开矩阵 $T_{block} \in \mathbb{R}^{\ell H_u \times \ell N_{block}}$，其元素由 $0$ 和 $I_\ell$ 构成。
3. 原始增量序列表示为 $\Delta\mathcal{U}(k) = T_{block} \Delta\mathcal{V}(k)$。
4. 代入预测方程：
   $$
   \mathcal{Z}(k) = \Psi x(k) + \Upsilon u(k-1) + \Theta (T_{block} \Delta\mathcal{V}(k)) = \Psi x(k) + \Upsilon u(k-1) + \tilde{\Theta} \Delta\mathcal{V}(k)
   $$
   其中新的强迫响应矩阵为 $\tilde{\Theta} = \Theta T_{block}$。

#### 【最终结论】
矩阵变换为 $\tilde{\Theta} = \Theta T_{block}$，列数显著缩减，大幅减轻在线 QP 计算负荷。

---

### Exercise 3.5：引入输入参考目标轨迹惩罚项 $\sum \|u(k+i|k) - u_{ref}(k+i)\|^2_{S(i)}$
- **题目要求**：
  (a) 详细说明在此修改下，预测计算与最优解求解方程的具体变化；
  (b) 列出此修改的优缺点。

#### 【解题思路】
目标函数增加了绝对输入偏离惩罚项。利用累加矩阵将绝对输入表示为增量的线性组合，代入后重新推导二次型的 Hessian 矩阵与一次项梯度。

#### 【逐步推导】
1. **(a) 方程修改推导**：
   由式 (3.37)：$\mathcal{U}(k) = U_1 u(k-1) + C_u \Delta\mathcal{U}(k)$。
   跟踪误差为：
   $$
   \mathcal{U}(k) - \mathcal{U}_{ref}(k) = C_u \Delta\mathcal{U}(k) - [\mathcal{U}_{ref}(k) - U_1 u(k-1)] \triangleq C_u \Delta\mathcal{U}(k) - \mathcal{E}_u(k)
   $$
   惩罚二次型展开：
   $$
   \|\mathcal{U} - \mathcal{U}_{ref}\|_{\mathcal{S}}^2 = \Delta\mathcal{U}^T (C_u^T \mathcal{S} C_u) \Delta\mathcal{U} - 2 \Delta\mathcal{U}^T C_u^T \mathcal{S} \mathcal{E}_u + \mathcal{E}_u^T \mathcal{S} \mathcal{E}_u
   $$
   与原有目标函数合并，新的 Hessian 矩阵与梯度向量为：
   $$
   H_{new} = \Theta^T \mathcal{Q} \Theta + \mathcal{R} + C_u^T \mathcal{S} C_u
   $$
   $$
   G_{new} = 2 \left[ \Theta^T \mathcal{Q} \mathcal{E}(k) + C_u^T \mathcal{S} (\mathcal{U}_{ref}(k) - U_1 u(k-1)) \right]
   $$
   显式最优解直接为 $\Delta\mathcal{U}_{opt} = H_{new}^{-1} \frac{1}{2} G_{new}$。

2. **(b) 优缺点对比**：
   - **优点**：能够将执行器引导至经济能耗最优工作点，消除零空间执行器游走；
   - **缺点**：破坏了天然无静差积分作用，若目标 $r$ 与 $u_{ref}$ 不自洽，系统将产生稳态静差。

#### 【最终结论】
推导如上，该项常用于经济预测控制（EMPC）稳态目标优化中。

---

### Exercise 3.6：控制幅值与变化率约束矩阵的显式结构
- **题目要求**：
  (a) 若不等式 (3.35) 源于简单幅值约束 (3.38)，证明 $F$ 呈现下三角块矩阵形式，并写出右手边向量；
  (b) 将变化率约束 (3.34) 化为 $W \Delta\mathcal{U}(k) \le w$ 标准形式。

#### 【解题思路】
利用绝对输入是增量累加的物理关系，通过下三角全 1 矩阵（或分块单位阵）构建约束映射。

#### 【逐步推导】
1. **(a) 幅值约束矩阵 $F$**：
   $$
   u(k+i|k) = u(k-1) + \sum_{j=0}^i \Delta u(k+j|k)
   $$
   双边约束写为：
   $$
   \begin{bmatrix} I_\ell & 0 & \cdots & 0 \\ I_\ell & I_\ell & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ I_\ell & I_\ell & \cdots & I_\ell \\ \hline -I_\ell & 0 & \cdots & 0 \\ -I_\ell & -I_\ell & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ -I_\ell & -I_\ell & \cdots & -I_\ell \end{bmatrix} \Delta\mathcal{U}(k) \le \begin{bmatrix} u_{max}(k) - u(k-1) \\ u_{max}(k+1) - u(k-1) \\ \vdots \\ u_{max}(k+H_u-1) - u(k-1) \\ \hline -u_{min}(k) + u(k-1) \\ -u_{min}(k+1) + u(k-1) \\ \vdots \\ -u_{min}(k+H_u-1) + u(k-1) \end{bmatrix}
   $$
   上部为下三角块单位阵，下部为相反数。

2. **(b) 变化率约束矩阵 $W$**：
   $$
   -\beta(k+i) \le \Delta u(k+i|k) \le \beta(k+i) \implies \begin{bmatrix} I_{\ell H_u} \\ -I_{\ell H_u} \end{bmatrix} \Delta\mathcal{U}(k) \le \begin{bmatrix} \beta \\ \beta \end{bmatrix}
   $$

#### 【最终结论】
约束矩阵结构如上，严格保证了仿射线性变换形式。

---

### Exercise 3.7：非对角加权矩阵支持与多变量解耦优化推导
- **题目要求**：在 MPC Toolbox 源代码中，函数 `smpccon` 默认强制将加权参数转换为对角矩阵。修改该函数以支持用户自定义的任意对称半正定权重矩阵 $\mathcal{Q}$ 与正定权重矩阵 $\mathcal{R}$，分析其代数求解方程与多变量动态解耦机理。

#### 【解题思路】
在多输入多输出（MIMO）系统中，若加权矩阵仅允许为对角阵，则目标函数只能对各个输出通道的绝对误差进行独立惩罚 $\sum q_i e_i^2$。当被控对象不同通道间存在强动态交叉耦合时，引入非对角权重项 $q_{ij} e_i e_j$ 能够在目标函数层面直接惩罚两通道之间的相对同步误差或比率偏离。推导非对角全矩阵下的 Hessian 矩阵求逆与闭环增益表达式，并分析其数学有效性。

#### 【逐步推导】
1. **原生工具箱限制与修改动机**：
   在 MATLAB 原生函数 `smpccon` 内部，通常使用 `diag(Q)` 将输入向量强制转化为纯对角阵。
   若用户希望指定输出变量之间的相对重要性关联（例如双轴伺服电机的同步位置误差惩罚 $(y_1 - y_2)^2 = y_1^2 - 2 y_1 y_2 + y_2^2$），则对应的加权子块必须为非对角矩阵：
   $$
   Q_{block} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
   $$
   该矩阵特征值为 $2$ 和 $0$，是典型的对称半正定矩阵，对角化假设无法支持此种控制需求。

2. **目标函数展开与非对角加权二次型**：
   设全时域加权矩阵 $\mathcal{Q} \in \mathbb{R}^{m H_p \times m H_p}$ 为对称半正定矩阵（$\mathcal{Q} = \mathcal{Q}^T \ge 0$），$\mathcal{R} \in \mathbb{R}^{\ell H_u \times \ell H_u}$ 为对称正定矩阵（$\mathcal{R} = \mathcal{R}^T > 0$）。
   目标函数为：
   $$
   V(k) = (\mathcal{Z}(k) - \mathcal{T}(k))^T \mathcal{Q} (\mathcal{Z}(k) - \mathcal{T}(k)) + \Delta\mathcal{U}(k)^T \mathcal{R} \Delta\mathcal{U}(k)
   $$
   代入预测方程 $\mathcal{Z}(k) = \Psi x(k) + \Upsilon u(k-1) + \Theta \Delta\mathcal{U}(k)$，定义自由误差 $\mathcal{E}(k) \triangleq \mathcal{T}(k) - \Psi x(k) - \Upsilon u(k-1)$：
   $$
   \mathcal{Z}(k) - \mathcal{T}(k) = \Theta \Delta\mathcal{U}(k) - \mathcal{E}(k)
   $$
   将二次型代数展开：
   $$
   (\Theta \Delta\mathcal{U} - \mathcal{E})^T \mathcal{Q} (\Theta \Delta\mathcal{U} - \mathcal{E}) = \Delta\mathcal{U}^T (\Theta^T \mathcal{Q} \Theta) \Delta\mathcal{U} - 2 \Delta\mathcal{U}^T \Theta^T \mathcal{Q} \mathcal{E} + \mathcal{E}^T \mathcal{Q} \mathcal{E}
   $$
   注意到：由于 $\mathcal{Q}$ 是对称阵，有 $(\Theta \Delta\mathcal{U})^T \mathcal{Q} \mathcal{E} = \mathcal{E}^T \mathcal{Q} (\Theta \Delta\mathcal{U})$，交叉项系数严格为 $2$。

3. **极值条件与最优解代数方程**：
   总目标函数整理为标准矩阵二次型：
   $$
   V(k) = \Delta\mathcal{U}^T (\Theta^T \mathcal{Q} \Theta + \mathcal{R}) \Delta\mathcal{U} - 2 \Delta\mathcal{U}^T \Theta^T \mathcal{Q} \mathcal{E}(k) + \mathcal{E}(k)^T \mathcal{Q} \mathcal{E}(k)
   $$
   对向量 $\Delta\mathcal{U}$ 求偏导并令梯度为零：
   $$
   \nabla_{\Delta\mathcal{U}} V = 2 (\Theta^T \mathcal{Q} \Theta + \mathcal{R}) \Delta\mathcal{U} - 2 \Theta^T \mathcal{Q} \mathcal{E}(k) = \mathbf{0}
   $$
   移项：
   $$
   (\Theta^T \mathcal{Q} \Theta + \mathcal{R}) \Delta\mathcal{U}_{opt} = \Theta^T \mathcal{Q} \mathcal{E}(k)
   $$
   因 $\mathcal{R} > 0$ 且 $\mathcal{Q} \ge 0$，矩阵 $H \triangleq \Theta^T \mathcal{Q} \Theta + \mathcal{R}$ 必为严格正定矩阵（行列式非零，逆矩阵严格存在）。
   得到全时域无约束最优解：
   $$
   \Delta\mathcal{U}_{opt}(k) = (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q} \mathcal{E}(k)
   $$

4. **滚动时域首步增益矩阵提取**：
   提取未来第一拍控制增量 $\Delta u(k|k) = S_1 \Delta\mathcal{U}_{opt}(k)$（$S_1 = [I_\ell, 0, \dots, 0]$）：
   $$
   K_{mpc} \triangleq S_1 (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1} \Theta^T \mathcal{Q} \in \mathbb{R}^{\ell \times m H_p}
   $$
   这表明：即使 $\mathcal{Q}$ 和 $\mathcal{R}$ 包含复杂的非对角交叉项，无约束最优控制律的数学结构与求逆公式依然严格成立，无需做任何算法框架性改动，只需解除对输入矩阵维度的对角约束检查！

#### 【最终结论】
非对角半正定加权不仅在数学上完全保持了二次规划的凸性与全局解析极小值性质，而且为多变量系统提供了强大的动态解耦自由度，通过交叉加权项实现了通道协同跟踪。

---

### Exercise 3.8：Example 3.1 的标准 QP 化与 KKT 条件代数验证
- **题目要求**：将 Example 3.1 放入标准 QP 格式，证明 $\Phi > 0$ 且问题严格凸，验证最优解 $(\theta_1 = 1.5, \theta_2 = 0.5)$ 满足 KKT 条件。

#### 【解题思路】
写出标准目标矩阵 $\Phi$ 与 $\phi$，计算特征值证明严格正定。在候选最优解处，检验原可行性、计算激活约束外法向乘子、验证对偶可行性与互补松弛性。

#### 【逐步推导】
1. **标准形式矩阵**：
   $$
   \Phi = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}, \quad \phi = \begin{bmatrix} -3 \\ 0 \end{bmatrix}
   $$
   特征方程 $(\lambda - 2)^2 - 1 = 0 \implies \lambda_1 = 3 > 0, \lambda_2 = 1 > 0$。Hessian 严格正定，问题为严格凸二次规划。
2. **KKT 条件逐项验证**：
   在点 $\theta^* = [1.5, 0.5]^T$ 处：
   - 约束 (1) $-\theta_1 = -1.5 < 0$（未激活，$\lambda_1 = 0$）；
   - 约束 (2) $-\theta_2 = -0.5 < 0$（未激活，$\lambda_2 = 0$）；
   - 约束 (3) $\theta_1 + \theta_2 = 1.5 + 0.5 = 2 = \omega_3$（激活，$\lambda_3 \ge 0$）；
   - 约束 (4) $-\theta_1 + \theta_2 = -1.0 < 1$（未激活，$\lambda_4 = 0$）。
   互补松弛条件 $\lambda_i g_i = 0$ 自动满足。
3. **驻点平衡方程**：
   $$
   \Phi \theta^* + \phi + \Omega_{(3)}^T \lambda_3 = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 1.5 \\ 0.5 \end{bmatrix} + \begin{bmatrix} -3 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 1 \end{bmatrix} \lambda_3 = \begin{bmatrix} -0.5 + \lambda_3 \\ -0.5 + \lambda_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
   $$
   解得 $\lambda_3 = 0.5 > 0$！满足对偶可行性 $\lambda \ge 0$。

#### 【最终结论】
所有 KKT 条件完全满足，严密证明 $(\theta_1=1.5, \theta_2=0.5)$ 为全局唯一最优解。

---

### Exercise 3.9：2-范数软约束与混合硬/软约束标准 QP 转化
- **题目要求**：
  (a) 验证优化问题 (3.86)–(3.87) 为标准 QP；
  (b) 推广到部分约束为硬约束、部分为软约束的情况。

#### 【解题思路】
将松弛变量 $\varepsilon$ 与原决策变量 $\theta$ 纵向增广拼接，重构目标 Hessian 矩阵与约束矩阵，验证其依然保持二次型与线性不等式形式。

#### 【逐步推导】
1. **(a) 2-范数软约束转化**：
   定义增广变量 $\tilde{\theta} = \begin{bmatrix} \theta \\ \varepsilon \end{bmatrix} \in \mathbb{R}^{(\ell H_u + n_c) \times 1}$。
   增广目标函数：
   $$
   \mathcal{J} = \frac{1}{2} \tilde{\theta}^T \begin{bmatrix} \Phi & 0 \\ 0 & 2 \rho I \end{bmatrix} \tilde{\theta} + \begin{bmatrix} \phi \\ 0 \end{bmatrix}^T \tilde{\theta}
   $$
   增广约束条件：
   $$
   \begin{bmatrix} \Omega & -I \\ 0 & -I \end{bmatrix} \begin{bmatrix} \theta \\ \varepsilon \end{bmatrix} \le \begin{bmatrix} \omega \\ 0 \end{bmatrix}
   $$
   块对角 Hessian 矩阵依然正定，约束依然全为线性不等式，严格保持为标准凸 QP。
2. **(b) 混合硬/软约束转化**：
   将约束拆分为硬约束 $\Omega_h \theta \le \omega_h$ 与软约束 $\Omega_s \theta \le \omega_s + \varepsilon$。
   约束矩阵重构为：
   $$
   \begin{bmatrix} \Omega_h & 0 \\ \Omega_s & -I \\ 0 & -I \end{bmatrix} \begin{bmatrix} \theta \\ \varepsilon \end{bmatrix} \le \begin{bmatrix} \omega_h \\ \omega_s \\ 0 \end{bmatrix}
   $$
   形式保持不变，依然为标准 QP。

#### 【最终结论】
2-范数软约束在数学上完全等价于标准凸二次规划。

---

### Exercise 3.10：1-范数与 $\infty$-范数软约束标准 QP 转化
- **题目要求**：证明优化问题 (3.88)（1-范数）与 (3.90)–(3.91)（$\infty$-范数）均为标准 QP。

#### 【解题思路】
通过引入辅助松弛变量，将绝对值求和与极大值运算转化为线性目标项与线性不等式约束。

#### 【逐步推导】
1. **对于 1-范数软约束 (3.88)**：
   目标函数线性项加上 $\rho \sum \varepsilon_i = \rho \mathbf{1}^T \varepsilon$。
   增广目标函数线性项为 $[\phi^T, \rho \mathbf{1}^T] \tilde{\theta}$，Hessian 矩阵在 $\varepsilon$ 部分为 0。这是标准的半正定凸二次规划。
2. **对于 $\infty$-范数软约束 (3.90)–(3.91)**：
   只需引入单一标量松弛 $\epsilon \ge 0$。增广变量为 $\tilde{\theta} = [\theta^T, \epsilon]^T$。
   约束转化为：
   $$
   \begin{bmatrix} \Omega & -\mathbf{1} \\ 0 & -1 \end{bmatrix} \begin{bmatrix} \theta \\ \epsilon \end{bmatrix} \le \begin{bmatrix} \omega \\ 0 \end{bmatrix}
   $$
   增广目标线性项为 $[\phi^T, \rho]^T \tilde{\theta}$。依然是标准二次规划。

#### 【最终结论】
1-范数与 $\infty$-范数软约束均可严格表示为标准 QP，其中 $\infty$-范数只需增加 1 个决策变量，求解最为高效。

---

### Exercise 3.11：Cessna 飞机模型增益低估 20% 的响应分析
- **题目要求**：假设 Cessna Citation 飞机模型升降舵到高度变化率的增益被低估 20%，证明此时系统不会遭遇不可行性崩溃，但响应会变得更加振荡。

#### 【解题思路】
当模型增益被低估 20% 时，实际飞机的物理舵效比控制器内部模型**高出 25%**。分析高开环环路增益对闭环极点分布与约束边界的影响。

#### 【逐步推导】
1. **不可行性崩溃机理回顾**：
   在 Example 3.2 中，增益被高估，控制器误以为舵效强，实际动作不足，导致状态在边界处被压死，在 $t=13$s 处发生不可行性崩溃。
2. **低估增益下的动力学行为**：
   当增益被低估时，真实舵效更强，控制器的较小动作即可引起物理飞机的迅速响应。飞机能够以更快的速率脱离危险边界，因此**绝不会遭遇几何不可行性崩溃**。
3. **闭环振荡性分析**：
   然而，由于实际环路增益比设计增益高出 25%，根据经典根轨迹原理，高增益迫使闭环极点向虚轴外侧偏移，阻尼比显著下降，导致闭环高度与高度变化率响应产生强烈的超调与振荡。

#### 【最终结论】
低估增益避免了不可行性崩溃，但带来了典型的欠阻尼振荡响应。

---

### Exercise 3.12：1-范数与 $\infty$-范数软约束效果一致性验证
- **题目要求**：验证在 Example 3.3 中，若采用 1-范数罚函数替代 $\infty$-范数罚函数，能否获得高度相似的平滑解救效果。

#### 【解题思路】
利用精确罚函数临界定理，只要罚系数 $\rho$ 取得足够大（超过对应激活约束的 Lagrange 乘子），1-范数与 $\infty$-范数在几何上均能实现对可行解的零失真恢复与对越界峰值的最优压制。

#### 【逐步推导】
1. 在脚本 `ch3_constraint_softening.py` 中，分别配置 1-范数罚函数与 $\infty$-范数罚函数；
2. 设定大惩罚系数 $\rho = 5 \times 10^6$；
3. 仿真输出曲线对比：在 $t < 13$s 内，两者的松弛变量均严格为零，与硬约束完全重叠；在 $t \ge 13$s 遭遇失配冲击时，两者均平滑激活微小松弛，升降舵以相同的平滑轨迹穿越饱和区，两者输出曲线几乎完全重合。

#### 【最终结论】
验证了 1-范数与 $\infty$-范数在精确惩罚区间的数学等价性与工程一致性。

---

## 8. 章节自测题与能力评估

### 8.1 概念题（共 5 题）
1. 为什么无约束基于控制增量 $\Delta u$ 建模的预测控制天然具备积分作用？
2. 为什么工业求解器绝不能直接求逆 $H^{-1} = (\Theta^T \mathcal{Q} \Theta + \mathcal{R})^{-1}$，而必须转化为最小二乘 QR 分解？
3. 解释为什么带约束的预测控制器在本质上是非线性分段仿射（PWA）控制器？
4. 比较有效集法（Active Set）与内点法（Interior Point）在求解时间与适用规模上的核心差异。
5. 什么是精确罚函数（Exact Penalty Function）？为什么 2-范数罚函数不是精确罚函数？

### 8.2 推导题（共 3 题）
1. 证明若 $\mathcal{R} > 0$ 且 $\mathcal{Q} \ge 0$，则 Hessian 矩阵 $H = \Theta^T \mathcal{Q} \Theta + \mathcal{R}$ 必为严格正定矩阵。
2. 请从 KKT 互补松弛条件严格推导 Example 3.1 在最优解 $[1.5, 0.5]^T$ 处的乘子 $\lambda_3 = 0.5$。
3. 推导 $\infty$-范数软约束转化为标准 QP 时，增广矩阵 $\tilde{\Phi}$ 与增广约束 $\tilde{\Omega}$ 的完整块矩阵形式。

### 8.3 编程题（共 2 题）
1. 运行 `ch3_qp_active_set_from_scratch.py`，改变初始点为内部点 $[0.5, 0.5]^T$，观察工作集更新步数。
2. 运行 `ch3_constraint_softening.py`，将惩罚系数由 $5 \times 10^6$ 降低至 $10^3$，观察约束如何被过度提前软化。

### 8.4 综合题（共 1 题）
- 某双容水箱液位控制系统具有强非对称约束（阀门只能开不能关负，液位上限严禁溢流）。请为该系统设计一个兼顾实时性与防崩溃的 MPC 求解器架构。

---

## 9. 总结与下一章衔接

### 9.1 本章核心学术结论
1. **无约束 MPC 等价于 LTI 控制**：在无约束条件下，基于增量模型的预测控制在本质上是一个带有内模积分作用与状态观测器的经典状态反馈调节器。
2. **数值求解必须规避病态平方**：使用 QR 分解求解最小二乘问题，是保证预测控制数值稳定性的金科玉律。
3. **约束将系统塑造成非线性分段仿射控制器**：约束激活与否决定了系统的动态增益，在线 QP 求解是带约束控制的灵魂。
4. **约束软化是工业生存的底线保障**：绝对硬约束无法抵抗现实世界的模型失配与未知扰动，基于 1-范数和 $\infty$-范数的精确罚函数软约束是工业 MPC 的必备标配。

### 9.2 下一章（第 4 章）学习动机与过渡
在本章中，我们全部基于**状态空间模型**展开讨论。然而：
- 在工业界开创 MPC 纪元时（如 DMC 动态矩阵控制），工程师们直接使用直观的**阶跃响应与脉冲响应卷积模型**；
- 在学术界，Clarke 等人提出的广义预测控制（GPC）完全基于**传递函数与 CARIMA 差分方程**，利用著名的 **Diophantine 方程** 进行多步预测。

**下一章动机**：
- 传递函数模型、阶跃响应卷积模型与状态空间模型到底存在怎样的数学映射？
- 什么是 Diophantine 多项式方程？它如何在频域完成最优预测？
- 为什么现代 MPC 又逐渐从传递函数全面回归状态空间？

我们将在 **第 4 章：阶跃响应与传递函数形式的预测控制 (Step response and transfer function formulations)** 中展开全宇宙最详细的逐节剖析与严密推导！

---

## 10. 推荐经典教程与扩展阅读
1. **数值优化与凸规划权威圣经**：
   - Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd Edition, Springer, 2006.（第 16 章详细推导二次规划的 Active Set 方法，第 19 章推导主对偶内点法）。
   - Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.（凸集、凸函数与 KKT 条件的全球标杆教材）。
2. **显式 MPC 与分段仿射控制理论**：
   - Alberto Bemporad, Manfred Morari, Vivek Dua, and Efstratios N. Pistikopoulos, "The explicit linear quadratic regulator for constrained systems," *Automatica*, vol. 38, no. 1, pp. 3-20, 2002.
3. **约束软化与工业生存策略**：
   - Edward C. Kerrigan and Jan M. Maciejowski, "Soft constraints and exact penalty functions in model predictive control," *UKACC International Conference on Control*, 2000.
