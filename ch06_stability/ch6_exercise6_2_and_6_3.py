"""
Chapter 6 Stability: Exercises 6.2 & 6.3
Analytical Derivations and Simulation of Horizon Effects and Weight Structure
==============================================================================
This script implements:
1. Exercise 6.2: Non-diagonal weight Q = [[1, 2], [2, 6]], Hp = 2
   (a) Hu = 1 (u(k+1) = u(k)):
       Cost: V(k) = 12 u(k)^2 + 4 u(k) x1(k) + 6 x1(k)^2
       Optimal law: u(k) = -1/6 * x1(k)
       Closed-loop pole: lambda = -1/6 ~= -0.1667 (STABLE!)
   (b) Hu = 2 (u(k+1) optimized independently):
       Partial dV/du(k+1) = 0 => u(k+1) = -2 u(k)
       Reduced cost: V(k) = 3 u(k)^2 + 4 u(k) x1(k) + 6 x1(k)^2
       Optimal law: u*(k) = -2/3 * x1(k)
       Closed-loop pole: lambda = -2/3 ~= -0.6667 (STABLE!)
2. Exercise 6.3: Diagonal weight Q = diag(1, 6)
   - Cross-terms vanish: dV/du(k) = 2 * alpha * u(k) = 0 => u*(k) = 0
   - System settles deadbeat to origin in exactly 2 steps!
3. Comprehensive simulation and side-by-side comparison:
   - Example 6.1: Hp=1, Hu=1, Q=[[1,2],[2,6]] -> Unstable (lambda = -2.0)
   - Ex 6.2(a):   Hp=2, Hu=1, Q=[[1,2],[2,6]] -> Stable   (lambda = -0.1667)
   - Ex 6.2(b):   Hp=2, Hu=2, Q=[[1,2],[2,6]] -> Stable   (lambda = -0.6667)
   - Ex 6.3:      Hp=2, Hu=2, Q=diag(1,6)     -> Deadbeat (lambda = 0.0)
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_simulation():
    print("=" * 75)
    print("Chapter 6: Exercises 6.2 & 6.3 - Analytical Verification & Simulation")
    print("=" * 75)

    # 1. System matrices: Two pure delays
    A = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    B = np.array([[1.0],
                  [0.0]])

    # 2. Control Laws to Compare
    # Law 1: Example 6.1 (Hp=1, Hu=1, Q non-diag) => u = -2 * x1
    K_ex6_1 = np.array([2.0, 0.0])
    A_cl_ex6_1 = A - B @ K_ex6_1.reshape(1, 2)
    poles_ex6_1 = la.eigvals(A_cl_ex6_1)

    # Law 2: Exercise 6.2(a) (Hp=2, Hu=1, Q non-diag) => u = -1/6 * x1
    K_ex6_2a = np.array([1.0 / 6.0, 0.0])
    A_cl_ex6_2a = A - B @ K_ex6_2a.reshape(1, 2)
    poles_ex6_2a = la.eigvals(A_cl_ex6_2a)

    # Law 3: Exercise 6.2(b) (Hp=2, Hu=2, Q non-diag) => u = -2/3 * x1
    K_ex6_2b = np.array([2.0 / 3.0, 0.0])
    A_cl_ex6_2b = A - B @ K_ex6_2b.reshape(1, 2)
    poles_ex6_2b = la.eigvals(A_cl_ex6_2b)

    # Law 4: Exercise 6.3 (Hp=2, Hu=2, Q=diag(1, 6)) => u = 0
    K_ex6_3 = np.array([0.0, 0.0])
    A_cl_ex6_3 = A - B @ K_ex6_3.reshape(1, 2)
    poles_ex6_3 = la.eigvals(A_cl_ex6_3)

    print("Analytical Control Laws and Closed-Loop Poles:")
    print(f"1. Example 6.1 (Hp=1, Hu=1): u = -2.0000 x1,  Poles: {poles_ex6_1} (UNSTABLE)")
    print(f"2. Ex 6.2(a)   (Hp=2, Hu=1): u = -0.1667 x1,  Poles: {poles_ex6_2a} (STABLE)")
    print(f"3. Ex 6.2(b)   (Hp=2, Hu=2): u = -0.6667 x1,  Poles: {poles_ex6_2b} (STABLE)")
    print(f"4. Ex 6.3      (Q diagonal): u =  0.0000 x1,  Poles: {poles_ex6_3} (DEADBEAT)")

    # 3. Simulate Responses from Initial Condition x0 = [1.0, 0.0]^T
    sim_steps = 12
    time_steps = np.arange(sim_steps + 1)
    time_u = np.arange(sim_steps)
    x0 = np.array([1.0, 0.0])

    def simulate_linear_loop(A_cl, B, K, steps):
        x = np.zeros((steps + 1, 2))
        u = np.zeros(steps)
        x[0] = x0
        for k in range(steps):
            u[k] = - K @ x[k]
            x[k+1] = A_cl @ x[k]
        return x, u

    x_1, u_1 = simulate_linear_loop(A_cl_ex6_1, B, K_ex6_1, sim_steps)
    x_2a, u_2a = simulate_linear_loop(A_cl_ex6_2a, B, K_ex6_2a, sim_steps)
    x_2b, u_2b = simulate_linear_loop(A_cl_ex6_2b, B, K_ex6_2b, sim_steps)
    x_3, u_3 = simulate_linear_loop(A_cl_ex6_3, B, K_ex6_3, sim_steps)

    # -------------------------------------------------------------
    # 4. Plotting Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): State x1(k) Responses
    ax = axes[0, 0]
    ax.plot(time_steps, x_1[:, 0], 'r--x', linewidth=1.5, label=r'Ex 6.1 ($H_p=1, H_u=1$, $\lambda=-2$) [Unstable]')
    ax.plot(time_steps, x_2a[:, 0], 'b-o', linewidth=2, label=r'Ex 6.2(a) ($H_p=2, H_u=1$, $\lambda=-1/6$) [Stable]')
    ax.plot(time_steps, x_2b[:, 0], 'g-s', linewidth=2, label=r'Ex 6.2(b) ($H_p=2, H_u=2$, $\lambda=-2/3$) [Stable]')
    ax.plot(time_steps, x_3[:, 0], 'm-^', linewidth=2, label=r'Ex 6.3 (Diag Q, $u=0$, Deadbeat)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_ylim(-1.5, 3.0)
    ax.set_title('State x1(k) Evolution (Initial x0 = [1, 0])', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('State x1(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (b): State x2(k) Responses
    ax = axes[0, 1]
    ax.plot(time_steps, x_1[:, 1], 'r--x', linewidth=1.5, label=r'Ex 6.1 [Unstable]')
    ax.plot(time_steps, x_2a[:, 1], 'b-o', linewidth=2, label=r'Ex 6.2(a) [Stable]')
    ax.plot(time_steps, x_2b[:, 1], 'g-s', linewidth=2, label=r'Ex 6.2(b) [Stable]')
    ax.plot(time_steps, x_3[:, 1], 'm-^', linewidth=2, label=r'Ex 6.3 [Deadbeat]')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_ylim(-1.5, 3.0)
    ax.set_title('State x2(k) Evolution (Delayed State)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('State x2(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (c): Control Action u(k)
    ax = axes[1, 0]
    ax.step(time_u, u_1, 'r--x', where='post', label=r'Ex 6.1: $u(k) = -2 x_1(k)$')
    ax.step(time_u, u_2a, 'b-o', where='post', label=r'Ex 6.2(a): $u(k) = -1/6 x_1(k)$')
    ax.step(time_u, u_2b, 'g-s', where='post', label=r'Ex 6.2(b): $u(k) = -2/3 x_1(k)$')
    ax.step(time_u, u_3, 'm-^', where='post', label=r'Ex 6.3: $u(k) = 0$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title('Control Action Comparison u(k)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Control Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (d): Complex Plane Closed-Loop Pole Locations
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle ($|z|=1$)')
    ax.scatter(np.real(poles_ex6_1), np.imag(poles_ex6_1), color='red', marker='x', s=120, linewidth=3, label=r'Ex 6.1 Pole: $\lambda=-2.0$ (Unstable)')
    ax.scatter(np.real(poles_ex6_2a), np.imag(poles_ex6_2a), color='blue', marker='o', s=100, linewidth=2, label=r'Ex 6.2(a) Pole: $\lambda=-1/6$')
    ax.scatter(np.real(poles_ex6_2b), np.imag(poles_ex6_2b), color='green', marker='s', s=100, linewidth=2, label=r'Ex 6.2(b) Pole: $\lambda=-2/3$')
    ax.scatter(np.real(poles_ex6_3), np.imag(poles_ex6_3), color='purple', marker='^', s=120, linewidth=2, label=r'Ex 6.3 Pole: $\lambda=0.0$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Closed-Loop Poles in the Complex z-Plane', fontsize=11, fontweight='bold')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch06_stability/ch6_exercise6_2_and_6_3.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
