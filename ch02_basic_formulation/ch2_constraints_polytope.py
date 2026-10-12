"""
ch2_constraints_polytope.py
================================================================================
配套章节：Maciejowski 第 2 章 2.7 节 约束条件的规范化与几何多面体表达
核心目的：专为零基础读者直观可视化三类约束 (率约束、幅值约束、输出约束)
如何通过半空间求交构成一个凸多面体可行域 (Feasible Polytope)。

理论推导：
设控制时域 Hu = 2，决策变量为 2 维向量: [Delta u(k|k), Delta u(k+1|k)]^T
1. 控制变化率约束 (Rate Constraints):
       -du_max <= Delta u(k+i) <= du_max  ==> W * Delta U <= w
2. 控制幅值约束 (Amplitude Constraints):
       u_min <= u(k-1) + Delta u(k) <= u_max
       u_min <= u(k-1) + Delta u(k) + Delta u(k+1) <= u_max
       ==> F * Delta U <= f(u(k-1))
3. 输出幅值约束 (Output Constraints):
       y_min <= Psi_y * x(k) + Upsilon_y * u(k-1) + Theta_y * Delta U <= y_max
       ==> Gamma * Delta U <= gamma(x(k), u(k-1))
三类约束综合堆叠为单一不等式:
       Omega * Delta U <= omega
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch2_constraints_polytope.py: 约束多面体与凸几何区域可视化")
    print("=" * 70)

    # 1. 参数设定 (Hu = 2)
    u_prev = 1.0       # 历史控制量
    du_max = 1.5       # 变化率限幅
    u_max = 2.5        # 输入幅值上限
    u_min = -0.5       # 输入幅值下限

    # 2. 构造变化率约束矩阵 W * Delta U <= w
    # W = [ I; -I ], w = [ du_max; du_max ]
    W = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [-1.0, 0.0],
                  [0.0, -1.0]])
    w = np.array([du_max, du_max, du_max, du_max])

    # 3. 构造幅值约束矩阵 F * Delta U <= f
    # F = [ 1 0; 1 1; -1 0; -1 -1 ]
    F = np.array([[1.0, 0.0],
                  [1.0, 1.0],
                  [-1.0, 0.0],
                  [-1.0, -1.0]])
    f = np.array([u_max - u_prev,
                  u_max - u_prev,
                  -u_min + u_prev,
                  -u_min + u_prev])

    # 4. 构造输出约束矩阵 (假定单步输出预测约束 y_pred <= y_max)
    # Theta_y * Delta U <= y_max - y_free
    # 设 Theta_y = [1.2, 0.8], 上限为 1.8
    Gamma = np.array([[1.2, 0.8]])
    gamma = np.array([1.8])

    # 拼接所有约束 Omega * Delta U <= omega
    Omega = np.vstack([W, F, Gamma])
    omega = np.concatenate([w, f, gamma])

    print("总约束矩阵 Omega (形状", Omega.shape, "):\n", Omega)
    print("总约束右手边 omega:\n", omega)

    # 5. 2D 网格点碰撞检测确定可行域凸多边形
    grid_du1 = np.linspace(-2.5, 2.5, 400)
    grid_du2 = np.linspace(-2.5, 2.5, 400)
    DU1, DU2 = np.meshgrid(grid_du1, grid_du2)

    # 检验每个点是否满足全部约束
    feasible_mask = np.ones(DU1.shape, dtype=bool)
    for i in range(len(omega)):
        constraint_val = Omega[i, 0] * DU1 + Omega[i, 1] * DU2
        feasible_mask = feasible_mask & (constraint_val <= omega[i] + 1e-6)

    # 6. 绘图展示
    fig, ax = plt.subplots(figsize=(9, 8))

    # 填充可行域
    ax.contourf(DU1, DU2, feasible_mask.astype(int), levels=[0.5, 1.5], colors=['#bbf2b6'], alpha=0.7)

    # 绘制各条约束边界线
    line_x = np.linspace(-2.5, 2.5, 100)
    colors = ['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'm']
    labels = ['Rate Limit: du0 <= 1.5', 'Rate Limit: du1 <= 1.5',
              'Rate Limit: du0 >= -1.5', 'Rate Limit: du1 >= -1.5',
              'Input Limit: u(k) <= 2.5', 'Input Limit: u(k+1) <= 2.5',
              'Input Limit: u(k) >= -0.5', 'Input Limit: u(k+1) >= -0.5',
              'Output Limit: 1.2*du0 + 0.8*du1 <= 1.8']

    # 绘制典型不等式边界
    ax.axvline(1.5, color='r', linestyle='--', label='Rate Limit bounds')
    ax.axvline(-1.5, color='r', linestyle='--')
    ax.axhline(1.5, color='r', linestyle='--')
    ax.axhline(-1.5, color='r', linestyle='--')

    ax.plot(line_x, (u_max - u_prev) - line_x, 'b-', label='Amplitude Limit u(k+1) <= 2.5')
    ax.plot(line_x, (u_min - u_prev) - line_x, 'b:', label='Amplitude Limit u(k+1) >= -0.5')
    ax.plot(line_x, (1.8 - 1.2 * line_x) / 0.8, 'm-.', linewidth=2, label='Output Constraint boundary')

    # 目标函数二次等高线 (假设无约束最优解在 [1.2, 1.2])
    theta_unc = np.array([1.2, 1.2])
    Z_cost = 2.0 * (DU1 - theta_unc[0])**2 + 2.0 * (DU2 - theta_unc[1])**2 - 1.0 * (DU1 - theta_unc[0]) * (DU2 - theta_unc[1])
    cs = ax.contour(DU1, DU2, Z_cost, levels=8, cmap='viridis', alpha=0.8)
    ax.clabel(cs, inline=1, fontsize=9)

    ax.plot(theta_unc[0], theta_unc[1], 'r*', markersize=14, label='Unconstrained Optimum (Infeasible!)')
    # 约束最优投影点 (钉在边界上)
    theta_opt = np.array([0.75, 0.75])
    ax.plot(theta_opt[0], theta_opt[1], 'go', markersize=10, label='Constrained Optimum on Boundary')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r'Control Increment $\Delta u(k|k)$', fontsize=12)
    ax.set_ylabel(r'Control Increment $\Delta u(k+1|k)$', fontsize=12)
    ax.set_title('Geometric Polytope of MPC Constraints and Active Boundaries', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()
    out_png = "/home/kuang/Desktop/mpc/ch02_basic_formulation/ch2_constraints_polytope.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {out_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
