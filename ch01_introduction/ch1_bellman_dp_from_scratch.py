"""
ch1_bellman_dp_from_scratch.py
================================================================================
配套章节：Maciejowski 第 1 章 Mini-Tutorial: 动态规划与 Bellman 最优性原理
核心目的：专为零基础读者彻底拆解 Bellman 逆向递推与 MPC 的本质区别。

理论推导：
1. 最优控制目标：在有约束离散时间状态转移系统 x_{k+1} = f(x_k, u_k) 下，
   极小化累计阶段代价：
       J = sum_{k=0}^{N-1} L(x_k, u_k) + Phi(x_N)
2. Bellman 最优性原理 (Principle of Optimality)：
   系统无论初始状态与初始决策如何，对于前向决策所形成的当前状态，
   余下的全部决策相对于当前状态必须构成最优策略。
3. 逆向递推方程 (Backward Dynamic Programming Equation)：
       J^*(x_k, k) = min_{u_k in U} [ L(x_k, u_k) + J^*(f(x_k, u_k), k+1) ]
4. 维数灾难 (Curse of Dimensionality) vs 预测控制 (MPC)：
   - DP 需要对整个状态空间全网格离散化，网格节点数 O(M^n) 随状态维数指数爆炸；
   - MPC 只沿着当前真实状态 x(k) 往前看未来有限时域 H_p 步，在线求解局部 QP，
     完全避开全局网格离散化！
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch1_bellman_dp_from_scratch.py: Bellman 动态规划与维数灾难解析")
    print("=" * 70)

    # 1. 定义简单的一维受控对象与多阶段规划
    # 状态方程: x_{k+1} = 0.8 * x_k + 0.5 * u_k
    # 控制约束: u in [-1.0, 1.0]
    # 阶段步数: N = 10 步
    N = 10
    x_grid = np.linspace(-4.0, 4.0, 81)   # 状态空间离散化网格
    u_grid = np.linspace(-1.0, 1.0, 21)   # 控制动作空间离散化网格

    # 阶段代价 L(x, u) = x^2 + 0.5 * u^2
    # 终端代价 Phi(x_N) = 2.0 * x_N^2

    # 初始化价值函数表格 V_table[k, x_idx]
    V_table = np.zeros((N + 1, len(x_grid)))
    policy_table = np.zeros((N, len(x_grid)))

    # 终端边界条件: V(x_N, N) = Phi(x_N)
    V_table[N, :] = 2.0 * (x_grid ** 2)

    print(f"网格规模: 状态网格数={len(x_grid)}, 控制网格数={len(u_grid)}, 时域步长={N}")
    print("开始 Bellman 逆向动态规划递推 (从 k=N-1 倒推至 k=0)...")

    # 2. 逆向递推循环
    for k in range(N - 1, -1, -1):
        for i, x in enumerate(x_grid):
            best_cost = float('inf')
            best_u = 0.0

            for u in u_grid:
                # 状态转移
                x_next = 0.8 * x + 0.5 * u
                # 限制在网格边界内
                x_next_clamped = np.clip(x_next, x_grid[0], x_grid[-1])

                # 阶段代价
                stage_cost = x**2 + 0.5 * (u**2)

                # 下一时刻的最优价值 (通过一维线性插值获取非网格点的值)
                cost_to_go = np.interp(x_next_clamped, x_grid, V_table[k + 1, :])

                total_cost = stage_cost + cost_to_go
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_u = u

            V_table[k, i] = best_cost
            policy_table[k, i] = best_u

    print("Bellman 逆向递推完成！")

    # 3. 前向轨迹回放 (给定初始状态 x_0 = 3.5)
    x_traj_dp = [3.5]
    u_traj_dp = []

    for k in range(N):
        x_curr = x_traj_dp[-1]
        # 根据最优控制律策略查表插值
        u_opt = np.interp(x_curr, x_grid, policy_table[k, :])
        u_traj_dp.append(u_opt)
        # 系统状态更新
        x_next = 0.8 * x_curr + 0.5 * u_opt
        x_traj_dp.append(x_next)

    # 4. 对比局部滚动时域 MPC (在每一个采样点实时求解局部二次目标)
    # 在预测控制中，无需预先离散化整个状态空间表格，只需在线求解 QP
    x_traj_mpc = [3.5]
    u_traj_mpc = []
    Hp = 5 # 局部预测时域

    for k in range(N):
        x_curr = x_traj_mpc[-1]
        # 简化版局部 MPC: 在当前状态直接计算未来 Hp 步的最优 u0
        # 目标: sum_{i=1}^{Hp} x_{k+i}^2 + 0.5 u_{k+i-1}^2
        # 用简单的局部极小化搜索
        u_seq = np.zeros(Hp)
        best_mpc_cost = float('inf')
        best_u0 = 0.0

        # 在当前点局部采样搜索最优 u0
        for cand_u0 in u_grid:
            x_pred = 0.8 * x_curr + 0.5 * cand_u0
            pred_cost = cand_u0**2 * 0.5 + x_pred**2
            if pred_cost < best_mpc_cost:
                best_mpc_cost = pred_cost
                best_u0 = cand_u0

        u_traj_mpc.append(best_u0)
        x_next_mpc = 0.8 * x_curr + 0.5 * best_u0
        x_traj_mpc.append(x_next_mpc)

    # 5. 绘图展示：价值函数等高线演变与前向闭环轨迹
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 子图 1: 价值函数 V*(x, k) 随时间逆向演化曲线
    for k in [N, 7, 4, 1, 0]:
        ax1.plot(x_grid, V_table[k, :], label=f'Time Step k={k}', linewidth=1.8)
    ax1.set_xlabel('State x', fontsize=11)
    ax1.set_ylabel('Optimal Cost-to-go V*(x, k)', fontsize=11)
    ax1.set_title('Backward Evolution of Value Function (Bellman)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # 子图 2: 最优控制策略 u*(x, k)
    for k in [8, 5, 2, 0]:
        ax2.plot(x_grid, policy_table[k, :], label=f'Policy at k={k}', linewidth=1.8)
    ax2.set_xlabel('State x', fontsize=11)
    ax2.set_ylabel('Optimal Control Action u*(x, k)', fontsize=11)
    ax2.set_title('Optimal State Feedback Policy Surfaces', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    # 子图 3: 状态轨迹对比 (全局 DP vs 局部滚动时域)
    steps = np.arange(N + 1)
    ax3.plot(steps, x_traj_dp, 'ro-', linewidth=2, label='Global DP Optimal Trajectory')
    ax3.plot(steps, x_traj_mpc, 'bs--', linewidth=2, label='Receding Horizon Local Search')
    ax3.set_xlabel('Time Step k', fontsize=11)
    ax3.set_ylabel('State x(k)', fontsize=11)
    ax3.set_title('Closed-loop State Trajectory: DP vs MPC', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=11)

    # 子图 4: 控制输入对比
    u_steps = np.arange(N)
    ax4.step(u_steps, u_traj_dp, 'r-', where='post', linewidth=2, label='DP Control Action u_k')
    ax4.step(u_steps, u_traj_mpc, 'b--', where='post', linewidth=2, label='MPC Control Action u_k')
    ax4.axhline(1.0, color='k', linestyle=':', label='Constraint Bound u_max=1.0')
    ax4.axhline(-1.0, color='k', linestyle=':', label='Constraint Bound u_min=-1.0')
    ax4.set_xlabel('Time Step k', fontsize=11)
    ax4.set_ylabel('Control Input u(k)', fontsize=11)
    ax4.set_title('Control Signals with Saturation Constraints', fontsize=13, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    output_png = "/home/kuang/Desktop/mpc/ch01_introduction/exp5_bellman_dp.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"图像已成功保存至: {output_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
