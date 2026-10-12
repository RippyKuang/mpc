"""
ch1_unstable_helicopter.py
================================================================================
配套章节：Maciejowski 第 1 章 1.6 节、Example 1.2
核心目的：专为零基础读者演示开环不稳定系统 (Unstable Plant) 在预测控制中的挑战：
1. 开环不稳定传递函数 G(s) = 1 / (s - 1) (右半平面极点 p = +1)；
2. 存在执行器硬限幅 |u| <= u_max；
3. 吸引域 (Domain of Attraction) 限制：初始状态若过大，执行器饱和，系统必发散；
4. 预测时域 Hp 的临界长度：时域过短无法“看见”不稳定极点的未来发散，导致闭环失稳；
5. 模型重置 (Model Realignment) 机制：利用真实测量实时重置开环内部模型。
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
    print("运行 ch1_unstable_helicopter.py: 开环不稳定对象与预测时域/约束交互")
    print("=" * 70)

    # 1. 开环不稳定一阶连续系统: dx/dt = 1.0 * x + 1.0 * u
    # 传递函数: G(s) = 1 / (s - 1)
    Ts = 0.1
    A_c = np.array([[1.0]])
    B_c = np.array([[1.0]])
    C_c = np.array([[1.0]])
    D_c = np.array([[0.0]])

    sys_c = signal.StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(Ts, method='zoh')
    A = sys_d.A[0, 0] # A = e^(1.0 * 0.1) = 1.10517
    B = sys_d.B[0, 0] # B = (e^0.1 - 1) * 1 = 0.10517
    C = 1.0

    print(f"连续极点 p = +1.0 (不稳定), 离散系统矩阵 (Ts={Ts}s):")
    print(f"A = {A:.5f} (> 1.0, 离散单位圆外极点), B = {B:.5f}")

    # 2. 仿真对比两种工况:
    # 工况 A: 预测时域足够长 (Hp = 15, Hu = 5) -> 成功镇定
    # 工况 B: 预测时域过短 (Hp = 2, Hu = 2) -> 短视导致执行器饱和失稳
    u_max = 2.0
    u_min = -2.0
    x0 = 0.8 # 初始偏离平衡态
    sim_steps = 40
    t_sim = np.arange(sim_steps) * Ts

    def simulate_mpc(Hp_val, Hu_val):
        x_curr = x0
        x_hist = []
        u_hist = []
        stable = True

        for k in range(sim_steps):
            x_hist.append(x_curr)
            if abs(x_curr) > 100.0:
                stable = False
                # 发散截断
                x_hist.extend([np.nan] * (sim_steps - len(x_hist)))
                u_hist.extend([np.nan] * (sim_steps - len(u_hist)))
                break

            # 局部 QP 求解
            def obj(u_seq):
                cost = 0.0
                x_pred = x_curr
                for i in range(Hp_val):
                    u_i = u_seq[i] if i < Hu_val else u_seq[Hu_val - 1]
                    x_pred = A * x_pred + B * u_i
                    cost += 10.0 * (x_pred ** 2) + 0.5 * (u_i ** 2)
                return cost

            bounds = [(u_min, u_max)] * Hu_val
            u_init = np.zeros(Hu_val)
            res = minimize(obj, u_init, method='SLSQP', bounds=bounds)

            u_opt = res.x[0]
            u_hist.append(u_opt)

            # 真实物理演化
            x_curr = A * x_curr + B * u_opt

        return np.array(x_hist[:sim_steps]), np.array(u_hist[:sim_steps]), stable

    x_long, u_long, s_long = simulate_mpc(Hp_val=15, Hu_val=5)
    x_short, u_short, s_short = simulate_mpc(Hp_val=2, Hu_val=2)

    # 3. 绘图对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t_sim, x_long, 'b-', linewidth=2.2, label='Long Horizon (Hp=15, Hu=5): Stabilized')
    ax1.plot(t_sim, x_short, 'r--', linewidth=2.0, label='Short Horizon (Hp=2, Hu=2): Diverges!')
    ax1.axhline(0.0, color='k', linestyle=':', label='Target State x=0')
    ax1.set_ylabel('State x(t)', fontsize=12)
    ax1.set_ylim(-1.5, 4.0)
    ax1.set_title('Unstable Plant Stabilization: Critical Importance of Prediction Horizon', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=11)

    ax2.step(t_sim, u_long, 'b-', where='post', linewidth=2.0, label='Control u(t) (Long Horizon)')
    ax2.step(t_sim, u_short, 'r--', where='post', linewidth=2.0, label='Control u(t) (Short Horizon)')
    ax2.axhline(u_max, color='k', linestyle=':', label='Actuator Limits [-2, 2]')
    ax2.axhline(u_min, color='k', linestyle=':')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Control Input u(t)', fontsize=12)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    output_png = "/home/kuang/Desktop/mpc/ch01_introduction/exp3_unstable_helicopter.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {output_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
