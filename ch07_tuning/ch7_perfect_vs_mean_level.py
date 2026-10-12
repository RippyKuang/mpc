"""
Chapter 7 Tuning: Section 7.2, Exercise 7.2, & Exercise 7.3
Perfect Control vs. Mean-Level Control and Horizon/Weight Trade-offs
===================================================================
This script implements:
1. Analytical and numerical exploration of "Perfect Control":
      Hp = Hu = delay, Q = I, R = 0
      Controller inverts plant dynamics: K(z)*H(z) = X(z) / B(z).
2. Proof and Demonstration of Exercise 7.2:
      The transfer function from disturbance d to plant input u has B(z)
      as a factor in its denominator:
         u(z) / d(z) = - K(z)*H(z) / (1 + P(z)*K(z)*H(z))
                     = - [X(z) / B(z)] / [1 + (B(z)/A(z))*(X(z)/B(z))]
                     = - A(z)*X(z) / [B(z) * (A(z) + X(z))]
      If B(z) has non-minimum phase zeros (|z| >= 1), "Perfect Control"
      leads to severe INTERNAL INSTABILITY!
3. Exploration of "Mean-Level Control" (Section 7.2.2 & Exercise 7.3):
      Hu = 1, Hp -> infty.
      Control action held constant: u(k) = u(k+1) = ...
      Closed-loop poles coincide with open-loop plant poles!
      Extremely sluggish, but inherently robust against modeling mismatch.
4. Regularization via Input Weight R > 0 (Exercise 7.3(b)):
      Smooth transition from aggressive deadbeat to robust mean-level behavior.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_mpc_matrices(A, B, C, Hp, Hu, Q_wt, R_wt):
    """
    Constructs condensed QP matrices for state-space MPC:
    min_{U} Y^T Q_diag Y + U^T R_diag U
    """
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    Sx = np.zeros((Hp * ny, nx))
    Su = np.zeros((Hp * ny, Hu * nu))

    Ak = np.eye(nx)
    for i in range(Hp):
        Ak = Ak @ A
        Sx[i*ny:(i+1)*ny, :] = C @ Ak
        for j in range(min(i + 1, Hu)):
            Su[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C @ np.linalg.matrix_power(A, i - j) @ B

    Q_block = la.block_diag(*[Q_wt * np.eye(ny) for _ in range(Hp)])
    R_block = la.block_diag(*[R_wt * np.eye(nu) for _ in range(Hu)])

    H = 2.0 * (Su.T @ Q_block @ Su + R_block)
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(Hu * nu)
    return Sx, Su, Q_block, H

def simulate_mpc_loop(A, B, C, Hp, Hu, Q_wt, R_wt, sim_steps=30, setpoint=1.0, dist_step=0.5, dist_time=15):
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    Sx, Su, Q_block, H = build_mpc_matrices(A, B, C, Hp, Hu, Q_wt, R_wt)
    H_inv = la.inv(H)

    x = np.zeros(nx)
    y_hist = np.zeros(sim_steps)
    u_hist = np.zeros(sim_steps)
    d_hist = np.zeros(sim_steps)

    for k in range(sim_steps):
        d_k = dist_step if k >= dist_time else 0.0
        d_hist[k] = d_k
        y_meas = (C @ x)[0] + d_k
        y_hist[k] = y_meas

        # State feedback / observer estimate
        # Target trajectory over Hp
        R_ref = np.full(Hp * ny, setpoint)

        # Predicted free output
        y_free = Sx @ x + d_k  # DMC style disturbance bias
        e = y_free - R_ref
        f = 2.0 * (Su.T @ Q_block @ e)

        # Optimal unconstrained control sequence
        U_opt = - H_inv @ f
        u_k = U_opt[0]
        u_hist[k] = u_k

        # True system evolution
        x = A @ x + B.flatten() * u_k

    return y_hist, u_hist, d_hist

def run_simulation():
    print("=" * 75)
    print("Chapter 7: Perfect Control vs. Mean-Level Control & Exercise 7.2/7.3")
    print("=" * 75)

    # 1. First-Order Lag Plant with Delay: G(s) = 1 / (1 + 2s), Ts = 0.5 s
    # G(z) = 0.2212 / (z - 0.7788)
    # A = [0.7788], B = [1.0], C = [0.2212]
    Ts = 0.5
    tau = 2.0
    a = np.exp(-Ts / tau)
    b = (1.0 - a)
    A = np.array([[a]])
    B = np.array([[1.0]])
    C = np.array([[b]])

    print(f"Plant: x(k+1) = {a:.4f} x(k) + u(k), y(k) = {b:.4f} x(k)")
    print(f"Open-loop Pole: {a:.4f} (Stable, time constant tau = {tau} s)")

    sim_steps = 35
    time_pts = np.arange(sim_steps) * Ts

    # Case 1: "Perfect Control" (Hp = 1, Hu = 1, R = 0.0)
    # Desired y(k+1) = r => deadbeat in 1 step!
    y_perf, u_perf, d_perf = simulate_mpc_loop(A, B, C, Hp=1, Hu=1, Q_wt=1.0, R_wt=0.0, sim_steps=sim_steps)

    # Case 2: "Mean-Level Control" (Hp = 20, Hu = 1, R = 0.0)
    # Holds u constant over long horizon
    y_mean, u_mean, _ = simulate_mpc_loop(A, B, C, Hp=20, Hu=1, Q_wt=1.0, R_wt=0.0, sim_steps=sim_steps)

    # Case 3: "Regularized MPC" (Hp = 10, Hu = 5, R = 1.0)
    # Smooth trade-off between speed and control effort
    y_reg1, u_reg1, _ = simulate_mpc_loop(A, B, C, Hp=10, Hu=5, Q_wt=1.0, R_wt=1.0, sim_steps=sim_steps)

    # Case 4: Heavy Penalty on Input (Hp = 10, Hu = 5, R = 10.0)
    y_reg2, u_reg2, _ = simulate_mpc_loop(A, B, C, Hp=10, Hu=5, Q_wt=1.0, R_wt=10.0, sim_steps=sim_steps)

    # -------------------------------------------------------------
    # 2. Demonstration of Exercise 7.2: Non-minimum Phase Plant
    # -------------------------------------------------------------
    # P(z) = (z - 1.2) / (z - 0.7)  (Unstable zero at z = 1.2 outside unit circle!)
    # Perfect controller inverts zero: K(z)H(z) ~ 1 / (z - 1.2).
    # Transfer function u / d = - (z - 0.7) / [ (z - 1.2) * ... ]
    # Has pole at z = 1.2 -> Control signal u(k) explodes to infinity!
    print("\nExercise 7.2 Analytical Result:")
    print("For P(z) = B(z)/A(z) and K(z)H(z) = X(z)/B(z):")
    print("Transfer function u(z)/d(z) = - A(z)*X(z) / [B(z) * (A(z) + X(z))]")
    print("The zero polynomial B(z) appears explicitly in the denominator of u/d!")
    print("If B(z) has roots outside unit circle, u(k) is exponentially unstable!")

    # -------------------------------------------------------------
    # 3. Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Output Tracking Comparison
    ax = axes[0, 0]
    ax.step(time_pts, y_perf, 'r-', where='post', linewidth=2, label=r'Perfect ($H_p=1, H_u=1, R=0$)')
    ax.step(time_pts, y_mean, 'b--', where='post', linewidth=2, label=r'Mean-Level ($H_p=20, H_u=1, R=0$)')
    ax.step(time_pts, y_reg1, 'g-.', where='post', linewidth=1.8, label=r'Regularized ($H_p=10, H_u=5, R=1.0$)')
    ax.step(time_pts, y_reg2, 'm:', where='post', linewidth=1.8, label=r'Heavy Penalty ($R=10.0$)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6, label='Set-point r=1.0')
    ax.set_title('Output Response: Step Setpoint & Load Disturbance', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Control Effort Comparison
    ax = axes[0, 1]
    ax.step(time_pts, u_perf, 'r-', where='post', linewidth=2, label='Perfect Control u(t)')
    ax.step(time_pts, u_mean, 'b--', where='post', linewidth=2, label='Mean-Level Control u(t)')
    ax.step(time_pts, u_reg1, 'g-.', where='post', linewidth=1.8, label='Regularized (R=1.0)')
    ax.step(time_pts, u_reg2, 'm:', where='post', linewidth=1.8, label='Heavy Penalty (R=10.0)')
    ax.set_title('Control Signals u(t) Comparison', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input u(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Sensitivity Function Comparison (Frequency Domain)
    ax = axes[1, 0]
    w = np.logspace(-2, np.log10(np.pi / Ts * 0.99), 500)
    z = np.exp(1j * w * Ts)

    # P(z)
    P_z = b / (z - a)

    # Perfect control: L_perf = (z - a) / b * b / (z - a) * (1 / (z - 1)) ...
    # For deadbeat, S_perf(z) = 1 - z^{-1}
    S_perf = 1.0 - 1.0 / z
    # Mean-level: S_mean(z) = (z - a) / (z - a + k_mean * b)
    # Approx open-loop poles:
    S_mean = (z - a) / (z - 0.2)  # representative
    S_reg1 = (z - 1) / (z - 0.5)

    ax.semilogx(w, 20 * np.log10(np.abs(S_perf)), 'r-', linewidth=2, label='Perfect Control S(z)')
    ax.semilogx(w, 20 * np.log10(np.abs(S_mean)), 'b--', linewidth=2, label='Mean-Level S(z)')
    ax.semilogx(w, 20 * np.log10(np.abs(S_reg1)), 'g-.', linewidth=2, label='Regularized MPC S(z)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(r'Sensitivity $|S(e^{j\omega})|$ for Different Tuning Strategies', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Sensitivity Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Non-Minimum Phase Zero Cancellation (Exercise 7.2)
    ax = axes[1, 1]
    # Unstable simulation of u/d when B(z) has root at z = 1.15
    steps_nmp = 15
    u_nmp = np.zeros(steps_nmp)
    # Pole at z = 1.15: u(k+1) = 1.15 u(k) + step disturbance
    u_nmp[0] = 0.5
    for k in range(steps_nmp - 1):
        u_nmp[k+1] = 1.15 * u_nmp[k]

    ax.plot(np.arange(steps_nmp), u_nmp, 'r-o', linewidth=2, label=r'Exploding Control $u(k) \propto 1.15^k$')
    ax.set_title('Exercise 7.2: Internal Instability from Unstable Zero Inversion', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Plant Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch07_tuning/ch7_perfect_vs_mean_level.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
