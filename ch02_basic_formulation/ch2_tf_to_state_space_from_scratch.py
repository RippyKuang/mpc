"""
ch2_tf_to_state_space_from_scratch.py
================================================================================
配套章节：Maciejowski 第 2 章 预备知识与 2.1 节
核心目的：专为只懂传递函数 G(s) 的读者打造。
从零推导连续传递函数到状态空间能控型/能观型的代数变换，并实现零阶保持器 (ZOH) 离散化。

理论推导：
1. 连续传递函数：
       G(s) = Y(s) / U(s) = (b_1 s + b_0) / (s^2 + a_1 s + a_0)
2. 微分方程表达：
       d^2 y/dt^2 + a_1 dy/dt + a_0 y = b_1 du/dt + b_0 u
3. 能控标准型状态变量定义与矩阵结构：
       dx1/dt = x2
       dx2/dt = -a_0 x1 - a_1 x2 + u
       y      = b_0 x1 + b_1 x2
4. 矩阵指数与 ZOH 精确离散化算法：
       A_d = expm(A_c * Ts)
       B_d = A_c^(-1) * (A_d - I) * B_c  (当 A_c 可逆时)
================================================================================
"""

import numpy as np
import scipy.signal as signal
from scipy.linalg import expm, inv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("运行 ch2_tf_to_state_space_from_scratch.py: 传递函数向状态空间转化推导")
    print("=" * 70)

    # 1. 给定典型二阶振荡环节传递函数
    # G(s) = (2 s + 5) / (s^2 + 3 s + 6)
    b1, b0 = 2.0, 5.0
    a1, a0 = 3.0, 6.0

    print(f"连续传递函数: G(s) = ({b1}s + {b0}) / (s^2 + {a1}s + {a0})")

    # 2. 从零构造能控标准型 (Controllable Canonical Form)
    A_c = np.array([[0.0, 1.0],
                    [-a0, -a1]])
    B_c = np.array([[0.0],
                    [1.0]])
    C_c = np.array([[b0, b1]])
    D_c = np.array([[0.0]])

    print("\n连续状态空间矩阵 (能控标准型):")
    print("A_c =\n", A_c)
    print("B_c =\n", B_c)
    print("C_c =\n", C_c)

    # 3. 采样周期 Ts = 0.1s，从零实现零阶保持器 (ZOH) 离散化
    Ts = 0.1
    # 状态转移矩阵 A_d = expm(A_c * Ts)
    A_d = expm(A_c * Ts)
    # 输入矩阵 B_d = inv(A_c) @ (A_d - I) @ B_c
    I_n = np.eye(2)
    B_d = inv(A_c) @ (A_d - I_n) @ B_c
    C_d = C_c.copy()
    D_d = D_c.copy()

    print(f"\n手算精确 ZOH 离散化矩阵 (Ts = {Ts}s):")
    print("A_d =\n", A_d)
    print("B_d =\n", B_d)
    print("C_d =\n", C_d)

    # 4. 验证与 SciPy 内置函数一致性
    sys_c = signal.StateSpace(A_c, B_c, C_c, D_c)
    sys_d_scipy = sys_c.to_discrete(Ts, method='zoh')
    diff_A = np.max(np.abs(A_d - sys_d_scipy.A))
    diff_B = np.max(np.abs(B_d - sys_d_scipy.B))
    print(f"矩阵 A_d 与 SciPy 对比最大差异: {diff_A:.2e}")
    print(f"矩阵 B_d 与 SciPy 对比最大差异: {diff_B:.2e}")

    # 5. 阶跃响应对比验证
    t_span = np.linspace(0, 5.0, 501)
    sys_tf = signal.TransferFunction([b1, b0], [1.0, a1, a0])
    t_c, y_c = signal.step(sys_tf, T=t_span)

    # 离散步进
    N_steps = int(5.0 / Ts) + 1
    t_d = np.arange(N_steps) * Ts
    x_k = np.zeros((2, 1))
    y_d = []
    for k in range(N_steps):
        u_k = 1.0
        y_val = (C_d @ x_k + D_d * u_k).item()
        y_d.append(y_val)
        x_k = A_d @ x_k + B_d * u_k

    # 6. 绘图
    plt.figure(figsize=(9, 5))
    plt.plot(t_c, y_c, 'b-', linewidth=2.0, label='Continuous TF G(s) Step Response')
    plt.step(t_d, y_d, 'r--', where='post', linewidth=1.8, label='Discrete State-Space Simulation')
    plt.axhline(b0 / a0, color='k', linestyle=':', label=f'Steady State y_ss={b0/a0:.3f}')
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.ylabel('Output y(t)', fontsize=11)
    plt.title('Step Response Verification: Transfer Function to Discrete State-Space', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_png = "/home/kuang/Desktop/mpc/ch02_basic_formulation/ch2_tf_to_state_space.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"图像已保存至: {out_png}")
    print("=" * 70)

if __name__ == '__main__':
    main()
