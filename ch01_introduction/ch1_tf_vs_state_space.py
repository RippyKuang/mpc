"""
ch1_tf_vs_state_space.py
================================================================================
配套章节：Maciejowski 第 1 章 引言 (Introduction)
核心目的：专为只懂传递函数的读者设计。
从经典传递函数 G(s) 逐步推导离散状态空间模型，并在时域和频域完成严格对比仿真。

理论内容：
1. 连续时间二阶系统传递函数：
       G(s) = Y(s) / U(s) = omega_n^2 / (s^2 + 2 * zeta * omega_n * s + omega_n^2)
2. 转换为连续状态空间模型 (能控标准型)：
       dx1/dt = x2
       dx2/dt = -omega_n^2 * x1 - 2*zeta*omega_n * x2 + omega_n^2 * u
       y = x1
3. 零阶保持器 (ZOH) 精确离散化：
       x(k+1) = A_d * x(k) + B_d * u(k)
       y(k)   = C_d * x(k) + D_d * u(k)
4. 数值仿真验证连续传递函数阶跃响应与离散状态空间步进响应完全吻合。
================================================================================
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch1_tf_vs_state_space.py: 传递函数与离散状态空间等价性验证")
    print("=" * 70)

    # 1. 定义二阶典型欠阻尼系统参数
    zeta = 0.5       # 阻尼比 (欠阻尼，存在超调)
    omega_n = 2.0    # 自然振荡角频率 (rad/s)

    num = [omega_n**2]
    den = [1.0, 2.0 * zeta * omega_n, omega_n**2]
    sys_tf = signal.TransferFunction(num, den)

    print(f"连续传递函数: G(s) = {omega_n**2} / (s^2 + {2.0*zeta*omega_n}s + {omega_n**2})")

    # 2. 转换为能控标准型连续状态空间
    # 令 x1 = y, x2 = dy/dt
    A_c = np.array([[0.0, 1.0],
                    [-omega_n**2, -2.0 * zeta * omega_n]])
    B_c = np.array([[0.0],
                    [omega_n**2]])
    C_c = np.array([[1.0, 0.0]])
    D_c = np.array([[0.0]])

    print("\n连续状态空间矩阵 (能控标准型):")
    print("A_c =\n", A_c)
    print("B_c =\n", B_c)
    print("C_c =\n", C_c)

    # 3. 零阶保持器 (ZOH) 离散化，采样周期 Ts = 0.1s
    Ts = 0.1
    sys_c = signal.StateSpace(A_c, B_c, C_c, D_c)
    sys_d = sys_c.to_discrete(Ts, method='zoh')

    A_d = sys_d.A
    B_d = sys_d.B
    C_d = sys_d.C
    D_d = sys_d.D

    print(f"\n离散化矩阵 (Ts = {Ts}s):")
    print("A_d =\n", A_d)
    print("B_d =\n", B_d)
    print("C_d =\n", C_d)

    # 4. 时域阶跃响应对比仿真
    t_span = np.linspace(0, 6.0, 601)
    t_cont, y_cont = signal.step(sys_tf, T=t_span)

    # 离散状态空间步进一步仿真 (从零状态出发)
    N_steps = int(6.0 / Ts) + 1
    t_disc = np.arange(N_steps) * Ts
    x_k = np.zeros((2, 1))
    y_disc = []

    for k in range(N_steps):
        u_k = 1.0  # 单位阶跃输入
        y_val = (C_d @ x_k + D_d * u_k).item()
        y_disc.append(y_val)
        # 状态递推
        x_k = A_d @ x_k + B_d * u_k

    y_disc = np.array(y_disc)

    # 计算离散采样点上的最大绝对误差
    y_cont_sampled = np.interp(t_disc, t_cont, y_cont)
    max_err = np.max(np.abs(y_disc - y_cont_sampled))
    print(f"\n连续传递函数与离散状态空间仿真最大误差: {max_err:.2e}")

    # 5. 绘图对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t_cont, y_cont, 'b-', label='Continuous Transfer Function G(s)', linewidth=2.0)
    ax1.step(t_disc, y_disc, 'r--', where='post', label='Discrete State-Space (ZOH)', linewidth=1.8)
    ax1.axhline(1.0, color='k', linestyle=':', label='Setpoint r=1.0')
    ax1.set_ylabel('Output y(t)', fontsize=12)
    ax1.set_title('Equivalence of Transfer Function and Discrete State Space Models', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=11)

    # 误差曲线
    ax2.plot(t_disc, np.abs(y_disc - y_cont_sampled), 'm-', linewidth=1.5, label='Absolute Discretization Error')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Error |y_disc - y_cont|', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    output_png = "/home/kuang/Desktop/mpc/ch01_introduction/exp4_tf_to_state_space.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {output_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
