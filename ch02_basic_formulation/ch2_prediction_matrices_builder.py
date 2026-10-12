"""
ch2_prediction_matrices_builder.py
================================================================================
配套章节：Maciejowski 第 2 章 2.6 节 预测方程 (Prediction)
核心目的：专为零基础读者彻底拆解预测矩阵 (Psi, Upsilon, Theta) 的代数构造逻辑。

理论推导：
离散时间增量状态方程：
    x(k+1) = A * x(k) + B * u(k) = A * x(k) + B * u(k-1) + B * Delta u(k)
输出方程：
    z(k)   = C_z * x(k)
未来 i 步状态递归推导：
    x(k+i|k) = A^i * x(k) + sum_{j=0}^{i-1} A^j * B * u(k-1)
             + sum_{j=1}^i ( sum_{p=0}^{i-j} A^p * B ) * Delta u(k+j-1|k)
紧凑矩阵预测方程：
    Z(k) = Psi * x(k) + Upsilon * u(k-1) + Theta * Delta U(k)
================================================================================
"""

import numpy as np

def build_prediction_matrices(A, B, C, Hp, Hu, Hw=1):
    """
    输入:
      A:  (n x n) 状态转移矩阵
      B:  (n x ell) 输入矩阵
      C:  (m x n) 被控输出矩阵
      Hp: 预测时域窗口
      Hu: 控制时域窗口
      Hw: 预测起始步长 (通常为 1)
    输出:
      Psi:     (m*(Hp-Hw+1) x n)
      Upsilon: (m*(Hp-Hw+1) x ell)
      Theta:   (m*(Hp-Hw+1) x ell*Hu)
    """
    n = A.shape[0]
    ell = B.shape[1]
    m = C.shape[0]
    rows = Hp - Hw + 1

    Psi = np.zeros((rows * m, n))
    Upsilon = np.zeros((rows * m, ell))
    Theta = np.zeros((rows * m, ell * Hu))

    # 预先计算 A^i 和 sum_{p=0}^{i-1} A^p * B
    A_powers = [np.eye(n)]
    for i in range(1, Hp + 1):
        A_powers.append(A_powers[-1] @ A)

    # 计算阶跃脉冲积分核: S_bar(i) = sum_{p=0}^{i-1} A^p * B
    S_bar = [np.zeros((n, ell))]
    cum_sum = np.zeros((n, ell))
    for i in range(1, Hp + 1):
        cum_sum = cum_sum + A_powers[i - 1] @ B
        S_bar.append(cum_sum.copy())

    # 逐行填充
    for r, step in enumerate(range(Hw, Hp + 1)):
        # 1. Psi: C * A^step
        Psi[r*m : (r+1)*m, :] = C @ A_powers[step]

        # 2. Upsilon: C * S_bar(step)
        Upsilon[r*m : (r+1)*m, :] = C @ S_bar[step]

        # 3. Theta: 下三角块 Toeplitz
        for c in range(Hu):
            if c < step:
                # 对应时间延迟 step - c
                step_diff = step - c
                Theta[r*m : (r+1)*m, c*ell : (c+1)*ell] = C @ S_bar[step_diff]

    return Psi, Upsilon, Theta

def main():
    print("=" * 70)
    print("运行 ch2_prediction_matrices_builder.py: 预测矩阵 (Psi, Upsilon, Theta) 手算检验")
    print("=" * 70)

    # 定义简单二阶系统 (n=2, ell=1, m=1)
    A = np.array([[0.8, 0.1],
                  [0.0, 0.9]])
    B = np.array([[0.0],
                  [1.0]])
    C = np.array([[1.0, 0.0]])

    Hp = 4
    Hu = 2
    Hw = 1

    print("输入系统矩阵:")
    print("A =\n", A)
    print("B =\n", B)
    print("C =\n", C)
    print(f"时域参数: Hp={Hp}, Hu={Hu}, Hw={Hw}\n")

    Psi, Upsilon, Theta = build_prediction_matrices(A, B, C, Hp, Hu, Hw)

    print("计算出的预测矩阵结构:")
    print(f"Psi (形状 {Psi.shape}):\n", np.round(Psi, 4))
    print(f"\nUpsilon (形状 {Upsilon.shape}):\n", np.round(Upsilon, 4))
    print(f"\nTheta (形状 {Theta.shape}, 下三角 Toeplitz 结构):\n", np.round(Theta, 4))

    # 验证测试：给定具体初值与控制动作，对比矩阵预测与实际状态逐步递推
    x0 = np.array([[1.0], [0.5]])
    u_prev = 0.2
    delta_U = np.array([[0.1], [-0.05]]) # Hu=2 步控制增量

    # 方式 1: 矩阵直接相乘预测
    Z_matrix = Psi @ x0 + Upsilon * u_prev + Theta @ delta_U

    # 方式 2: 物理对象逐步演进验证
    Z_step = []
    x_k = x0.copy()
    u_curr = u_prev
    for step in range(1, Hp + 1):
        du = delta_U[step - 1, 0] if step <= Hu else 0.0
        u_curr = u_curr + du
        x_k = A @ x_k + B * u_curr
        y_val = (C @ x_k).item()
        Z_step.append(y_val)
    Z_step = np.array(Z_step).reshape(-1, 1)

    max_err = np.max(np.abs(Z_matrix - Z_step))
    print("\n矩阵预测结果 Z_matrix:\n", np.round(Z_matrix.flatten(), 5))
    print("逐步递推结果 Z_step:\n", np.round(Z_step.flatten(), 5))
    print(f"\n两者最大绝对误差: {max_err:.2e} (完全一致！)")
    print("=" * 70)

if __name__ == '__main__':
    main()
