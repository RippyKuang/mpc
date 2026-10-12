"""
ch1_nonminimum_phase_mpc.py
================================================================================
配套章节：Maciejowski 第 1 章 1.3 节、Example 1.3
核心目的：专为经典控制读者演示非最小相位系统 (Non-minimum phase, 右半平面零点)
在 PID 控制 vs 预测控制 (MPC) 下的根本响应差异。

理论背景：
连续传递函数：
    G(s) = (1 - s) / ((s + 1)(2s + 1)) = (-s + 1) / (2s^2 + 3s + 1)
特征：
1. 存在右半平面零点 z = +1 (Non-minimum phase zero)；
2. 阶跃响应具有典型的“逆向响应” (Inverse Response / 下冲)：
   控制量增加时，输出首先往下掉，随后才上升！
3. 经典高增益反馈 (PID) 极易在此类系统上发散或产生严重振荡；
4. MPC 利用未来预测能力，在优化中权衡未来步长，预见下冲并平滑控制。
================================================================================
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch1_nonminimum_phase_mpc.py: 非最小相位系统逆向响应与预测控制")
    print("=" * 70)

    # 1. 连续传递函数离散化
    # G(s) = (-s + 1) / (2s^2 + 3s + 1)
    Ts = 0.2
    # 状态空间能控型
    A_c = np.array([[0.0, 1.0],
                    [-0.5, -1.5]])
    B_c = np.array([[0.0],
                    [1.0]])
    C_c = np.array([[0.5, -0.5]])
    D_c = np.array([[0.0]])

    sys_c = signal.StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(Ts, method='zoh')
    A = sys_d.A
    B = sys_d.B
    C = sys_d.C
    D = sys_d.D

    print(f"采样周期 Ts = {Ts}s, 离散系统矩阵:")
    print("A =\n", A)
    print("B =\n", B)
    print("C =\n", C)

    # 2. 预测控制参数
    Hp = 15       # 预测时域
    Hu = 5        # 控制时域
    Q = 10.0      # 跟踪误差惩罚
    R = 1.0       # 控制增量惩罚
    u_max = 2.0   # 执行机构硬约束
    u_min = -2.0

    # 仿真步数
    sim_steps = 60
    t_sim = np.arange(sim_steps) * Ts
    r = np.ones(sim_steps) # 设定值 = 1.0

    # 闭环状态与输入记录
    x_mpc = np.zeros((2, 1))
    u_prev_mpc = 0.0
    y_hist_mpc = []
    u_hist_mpc = []

    print("\n开始 MPC 闭环滚动时域仿真...")
    for k in range(sim_steps):
        # 构造二次型目标函数与约束
        # 决策变量: du = [du_0, du_1, ..., du_{Hu-1}]
        def mpc_obj(du):
            cost = 0.0
            x_k = x_mpc.copy()
            u_k = u_prev_mpc
            for i in range(Hp):
                if i < Hu:
                    u_k = u_k + du[i]
                x_k = A @ x_k + B * u_k
                y_k = (C @ x_k + D * u_k).item()
                cost += Q * ((y_k - r[k]) ** 2)
                if i < Hu:
                    cost += R * (du[i] ** 2)
            return cost

        # 绝对输入幅值约束
        # u(k+i) = u_prev + sum_{j=0}^i du_j in [u_min, u_max]
        constraints = []
        for i in range(Hu):
            def make_c_max(idx):
                return lambda du: u_max - (u_prev_mpc + np.sum(du[:idx+1]))
            def make_c_min(idx):
                return lambda du: (u_prev_mpc + np.sum(du[:idx+1])) - u_min
            constraints.append({'type': 'ineq', 'fun': make_c_max(i)})
            constraints.append({'type': 'ineq', 'fun': make_c_min(i)})

        du_init = np.zeros(Hu)
        res = minimize(mpc_obj, du_init, method='SLSQP', constraints=constraints)

        du0 = res.x[0]
        u_curr = u_prev_mpc + du0

        # 对象状态演进
        y_curr = (C @ x_mpc + D * u_curr).item()
        y_hist_mpc.append(y_curr)
        u_hist_mpc.append(u_curr)

        x_mpc = A @ x_mpc + B * u_curr
        u_prev_mpc = u_curr

    # 3. 经典离散 PID 控制器对比
    Kp = 0.8
    Ki = 0.2 * Ts
    Kd = 0.1 / Ts
    x_pid = np.zeros((2, 1))
    y_hist_pid = []
    u_hist_pid = []
    e_int = 0.0
    e_prev = 0.0

    for k in range(sim_steps):
        y_curr = (C @ x_pid).item()
        y_hist_pid.append(y_curr)

        err = r[k] - y_curr
        e_int += err
        e_diff = (err - e_prev)
        e_prev = err

        u_pid = Kp * err + Ki * e_int + Kd * e_diff
        u_pid = np.clip(u_pid, u_min, u_max)
        u_hist_pid.append(u_pid)

        x_pid = A @ x_pid + B * u_pid

    # 4. 绘图对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t_sim, y_hist_mpc, 'b-', linewidth=2.2, label='Model Predictive Control (MPC)')
    ax1.plot(t_sim, y_hist_pid, 'r--', linewidth=1.8, label='Classical PID Feedback')
    ax1.axhline(1.0, color='k', linestyle=':', label='Setpoint r=1.0')
    ax1.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Output y(t)', fontsize=12)
    ax1.set_title('Non-Minimum Phase System Control: MPC vs Classical PID', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=11)

    # 标注逆向响应
    ax1.annotate('Initial Undershoot\n(Inverse Response due to RHP Zero)',
                 xy=(0.6, -0.15), xytext=(1.8, -0.4),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', color='darkred')

    ax2.step(t_sim, u_hist_mpc, 'b-', where='post', linewidth=2.0, label='MPC Control Action u(t)')
    ax2.step(t_sim, u_hist_pid, 'r--', where='post', linewidth=1.8, label='PID Control Action u(t)')
    ax2.axhline(u_max, color='k', linestyle=':', label='Actuator Limits [-2, 2]')
    ax2.axhline(u_min, color='k', linestyle=':')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Control Input u(t)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    output_png = "/home/kuang/Desktop/mpc/ch01_introduction/exp6_nonminimum_phase.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {output_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
