"""
ch2_observer_disturbance_from_scratch.py
================================================================================
配套章节：Maciejowski 第 2 章 Mini-Tutorial 2: 状态观测器与扰动估计器
核心目的：专为经典控制读者彻底拆解状态观测器 (Luenberger Observer)
与步进输出扰动估计器的工作原理与误差动态方程。

理论推导：
1. 真实物理对象 (含输出扰动 d_k)：
       x(k+1) = A * x(k) + B * u(k)
       y(k)   = C * x(k) + d(k)
2. 增广状态空间 (将未知恒定扰动作为状态变量 d(k+1) = d(k))：
       [ x(k+1) ]   [ A   0 ] [ x(k) ]   [ B ]
       [ d(k+1) ] = [ 0   I ] [ d(k) ] + [ 0 ] * u(k)
       y(k)       = [ C   I ] [ x(k); d(k) ]
3. 预测-更新型离散观测器方程：
       e(k)       = y(k) - (C * x_hat(k|k-1) + d_hat(k|k-1))  (新息残差)
       x_hat(k|k) = x_hat(k|k-1) + L_x * e(k)
       d_hat(k|k) = d_hat(k|k-1) + L_d * e(k)
       x_hat(k+1|k) = A * x_hat(k|k) + B * u(k)
       d_hat(k+1|k) = d_hat(k|k)
4. 误差动态方程与稳定性证明：
       误差系统特征值由 (A - L_x * C) 决定，配置极点在单位圆内保证渐近收敛。
================================================================================
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch2_observer_disturbance_from_scratch.py: 状态与扰动观测器仿真")
    print("=" * 70)

    # 1. 连续振荡系统离散化 (二阶系统)
    Ts = 0.1
    A_c = np.array([[0.0, 1.0],
                    [-4.0, -1.0]])
    B_c = np.array([[0.0],
                    [2.0]])
    C_c = np.array([[1.0, 0.0]]) # 只能测量第一个状态 x1 (例如位移)，无法测量 x2 (速度)

    sys_c = signal.StateSpace(A_c, B_c, C_c, 0.0)
    sys_d = sys_c.to_discrete(Ts, method='zoh')
    A = sys_d.A
    B = sys_d.B
    C = sys_d.C

    print("离散系统模型:")
    print("A =\n", np.round(A, 4))
    print("B =\n", np.round(B, 4))
    print("C =\n", np.round(C, 4))

    # 2. 观测器增益设计 (采用极点配置法)
    # 系统开环极点:
    eig_open = np.linalg.eigvals(A)
    print("\n系统开环离散极点:", np.round(eig_open, 4))

    # 设计观测器极点，使其比系统极点响应更快 (放置在 0.4 +/- 0.1j)
    desired_poles = np.array([0.4 + 0.1j, 0.4 - 0.1j])
    place_res = signal.place_poles(A.T, C.T, desired_poles)
    Lx = place_res.gain_matrix.T # (2 x 1) 状态观测器增益
    Ld = 0.5                     # 扰动估计器增益

    print("\n观测器状态增益 Lx =\n", np.round(Lx, 4))
    print(f"扰动增益 Ld = {Ld}")

    # 3. 仿真过程 (在 t=2.0s 施加未知恒定输出阶跃扰动 d=1.5)
    sim_steps = 60
    t_sim = np.arange(sim_steps) * Ts

    # 真实对象初始状态 [1.5, -2.0]
    x_true = np.array([[1.5], [-2.0]])
    # 观测器对初始状态完全一无所知，从 [0.0, 0.0] 开始估计
    x_hat = np.array([[0.0], [0.0]])
    d_hat = 0.0

    x_true_hist = []
    x_hat_hist = []
    d_true_hist = []
    d_hat_hist = []

    for k in range(sim_steps):
        # 物理控制输入 (正弦激励)
        u_k = np.sin(0.5 * k * Ts)

        # 扰动在 t >= 2.0s (第 20 步) 突加 1.5
        d_true = 1.5 if k >= 20 else 0.0

        # 传感器实际测量值
        y_meas = (C @ x_true).item() + d_true

        # 记录
        x_true_hist.append(x_true.flatten())
        x_hat_hist.append(x_hat.flatten())
        d_true_hist.append(d_true)
        d_hat_hist.append(d_hat)

        # --- 观测器更新步 (Innovation update) ---
        y_pred = (C @ x_hat).item() + d_hat
        residual = y_meas - y_pred # 新息

        # 修正状态与扰动
        x_hat = x_hat + Lx * residual
        d_hat = d_hat + Ld * residual

        # --- 真实对象与观测器前向时间推进步 ---
        x_true = A @ x_true + B * u_k
        x_hat = A @ x_hat + B * u_k

    x_true_hist = np.array(x_true_hist)
    x_hat_hist = np.array(x_hat_hist)

    # 4. 绘图对比真实状态 vs 观测状态
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax1.plot(t_sim, x_true_hist[:, 0], 'k-', linewidth=2.0, label='True State x1 (Position)')
    ax1.plot(t_sim, x_hat_hist[:, 0], 'r--', linewidth=2.0, label='Observer Estimate x1_hat')
    ax1.set_ylabel('State x1', fontsize=11)
    ax1.set_title('Discrete State & Disturbance Observer from Scratch', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=10)

    ax2.plot(t_sim, x_true_hist[:, 1], 'k-', linewidth=2.0, label='True State x2 (Velocity, Unmeasured)')
    ax2.plot(t_sim, x_hat_hist[:, 1], 'b--', linewidth=2.0, label='Observer Estimate x2_hat')
    ax2.set_ylabel('State x2', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)

    ax3.plot(t_sim, d_true_hist, 'k-', linewidth=2.0, label='True Output Disturbance d(t)')
    ax3.plot(t_sim, d_hat_hist, 'm--', linewidth=2.0, label='Disturbance Estimate d_hat(t)')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Disturbance d', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out_png = "/home/kuang/Desktop/mpc/ch02_basic_formulation/ch2_observer_tracking.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {out_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
