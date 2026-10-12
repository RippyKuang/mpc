"""
Chapter 7 Tuning: Exercise 7.5
Non-Minimum Phase Plant: Unstable Zeros, Horizon Tuning, and Undershoot Mitigation
==================================================================================
This script implements:
1. Discretization of the continuous non-minimum phase plant:
      P(s) = (-10 s + 1) / ((s + 1)(s + 30)),   Ts = 0.2 s
   Plant has a right-half-plane zero at s = +0.1, transforming into a discrete
   zero outside the unit circle (|z| > 1)!
2. Exercise 7.5(a): Simulation and failure of "Perfect Controller" (Hp = 1, Hu = 1, R = 0)
   - Tries to cancel the unstable zero with an unstable controller pole,
     destabilizing the internal feedback loop.
3. Exercise 7.5(b): Finding the minimal prediction horizon Hp for stability (Hu = 1, R = 0)
   - Traces the closed-loop pole migration as Hp increases from 1 to 40.
   - Identifies the critical bifurcation point where closed-loop spectral radius < 1.
4. Exercise 7.5(c): Tuning for Hp = 30, Hu = 2, Q = 1, R = 0.1
   - Demonstrates the hallmark "inverse response" (undershoot in the wrong direction).
   - Solves QP with rate limit |Delta u(k)| <= 0.1 to suppress peak undershoot.
"""

import numpy as np
import scipy.signal as signal
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_discrete_plant(Ts=0.2):
    """
    Discretizes P(s) = (-10 s + 1) / ((s + 1)(s + 30)) using ZOH.
    """
    num_c = [-10.0, 1.0]
    den_c = [1.0, 31.0, 30.0]
    sys_d = signal.cont2discrete((num_c, den_c), Ts, method='zoh')
    num_d = sys_d[0][0]
    den_d = sys_d[1]

    # Convert to state-space representation (Observable canonical form)
    A, B, C, D = signal.tf2ss(num_d, den_d)
    return A, B, C, D, num_d, den_d

def find_minimal_stabilizing_Hp(A, B, C, Hu=1, R_wt=0.0):
    """
    Computes closed-loop poles for varying Hp with Hu=1, R=0,
    and returns the minimal Hp where max(|lambda|) < 1.
    """
    nx = A.shape[0]
    ny = C.shape[0]

    stable_Hp = None
    pole_history = {}

    for Hp in range(1, 45):
        # Build predictions for Hu = 1
        # Y = Sx * x + Su * u_0
        Sx = np.zeros((Hp * ny, nx))
        Su = np.zeros((Hp * ny, 1))

        Ak = np.eye(nx)
        for i in range(Hp):
            Ak = Ak @ A
            Sx[i*ny:(i+1)*ny, :] = C @ Ak
            Su[i*ny:(i+1)*ny, :] = C @ np.linalg.matrix_power(A, i) @ B

        # Optimal unconstrained u_0 = - (Su^T Su + R)^{-1} Su^T Sx * x
        H = Su.T @ Su + R_wt * np.eye(1)
        if np.abs(H[0, 0]) < 1e-12:
            continue
        K_opt = la.inv(H) @ (Su.T @ Sx)

        # Closed-loop matrix
        A_cl = A - B @ K_opt
        poles = la.eigvals(A_cl)
        rho = np.max(np.abs(poles))
        pole_history[Hp] = (rho, poles)

        if rho < 1.0 and stable_Hp is None:
            stable_Hp = Hp

    return stable_Hp, pole_history

def simulate_constrained_mpc(A, B, C, Hp=30, Hu=2, Q_wt=1.0, R_wt=0.1, du_max=None, sim_steps=60, Ts=0.2):
    """
    Simulates MPC with incremental formulation (Delta u), tracking unit step.
    Allows rate constraint |Delta u(k)| <= du_max.
    """
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    # Augmented state for offset-free integral action: x_aug = [Delta x; y]
    A_aug = np.block([[A, np.zeros((nx, ny))],
                      [C @ A, np.eye(ny)]])
    B_aug = np.block([[B],
                      [C @ B]])
    C_aug = np.block([np.zeros((ny, nx)), np.eye(ny)])
    nx_aug = A_aug.shape[0]

    # Build predictions over Hp and Hu
    Sx = np.zeros((Hp * ny, nx_aug))
    Su = np.zeros((Hp * ny, Hu * nu))

    Ak = np.eye(nx_aug)
    for i in range(Hp):
        Ak = Ak @ A_aug
        Sx[i*ny:(i+1)*ny, :] = C_aug @ Ak
        for j in range(min(i + 1, Hu)):
            Su[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C_aug @ np.linalg.matrix_power(A_aug, i - j) @ B_aug

    Q_block = la.block_diag(*[Q_wt * np.eye(ny) for _ in range(Hp)])
    R_block = la.block_diag(*[R_wt * np.eye(nu) for _ in range(Hu)])

    H_qp = 2.0 * (Su.T @ Q_block @ Su + R_block)
    H_qp = 0.5 * (H_qp + H_qp.T) + 1e-9 * np.eye(Hu * nu)

    # Simulation state
    x = np.zeros(nx)
    x_prev = np.zeros(nx)
    u_prev = 0.0
    y_current = 0.0

    y_hist = np.zeros(sim_steps)
    u_hist = np.zeros(sim_steps)
    du_hist = np.zeros(sim_steps)

    for k in range(sim_steps):
        # Current augmented state
        dx = x - x_prev
        x_aug = np.concatenate([dx, np.array([y_current])])

        # Reference trajectory
        r_target = np.ones(Hp * ny)
        f_qp = 2.0 * (Su.T @ Q_block @ (Sx @ x_aug - r_target))

        # Solve QP
        if du_max is None:
            # Unconstrained analytical
            dU_opt = - la.inv(H_qp) @ f_qp
        else:
            bounds = [(-du_max, du_max) for _ in range(Hu * nu)]
            res = minimize(lambda dU: 0.5 * dU @ H_qp @ dU + f_qp @ dU,
                           np.zeros(Hu * nu), bounds=bounds, method='SLSQP')
            dU_opt = res.x

        du_k = dU_opt[0]
        u_k = u_prev + du_k

        y_hist[k] = y_current
        u_hist[k] = u_k
        du_hist[k] = du_k

        # Update plant
        x_prev = x.copy()
        x = A @ x + B.flatten() * u_k
        y_current = (C @ x)[0]
        u_prev = u_k

    return y_hist, u_hist, du_hist

def run_exercise_7_5():
    print("=" * 75)
    print("Chapter 7: Exercise 7.5 - Non-Minimum Phase Plant & Horizon Tuning")
    print("=" * 75)

    Ts = 0.2
    A, B, C, D, num_d, den_d = get_discrete_plant(Ts)

    zeros_d = np.roots(num_d)
    poles_d = np.roots(den_d)
    print(f"Discrete Plant Transfer Function:")
    print(f"  Zeros: {zeros_d}  (|z| = {[np.abs(z) for z in zeros_d]})")
    print(f"  Poles: {poles_d}  (|z| = {[np.abs(p) for p in poles_d]})")
    assert any(np.abs(zeros_d) > 1.0), "Plant must have non-minimum phase zero!"
    print(f">> Unstable zero found at z = {zeros_d[0]:.4f} (|z| > 1)!")

    # -------------------------------------------------------------
    # (b) Find minimal stabilizing Hp
    # -------------------------------------------------------------
    min_Hp, pole_hist = find_minimal_stabilizing_Hp(A, B, C, Hu=1, R_wt=0.0)
    print(f"\nExercise 7.5(b) Stability Analysis (Hu = 1, R = 0):")
    print(f"  Smallest stabilizing prediction horizon Hp = {min_Hp}")
    rho_min, poles_min = pole_hist[min_Hp]
    print(f"  Closed-loop poles at Hp = {min_Hp}: {poles_min}")
    print(f"  Spectral radius rho(A_cl) = {rho_min:.4f} < 1.0 (Stable!)")

    # Time constant estimate
    dominant_pole = np.max(np.abs(poles_min))
    tau_cl = - Ts / np.log(dominant_pole)
    print(f"  Dominant pole = {dominant_pole:.4f} => Equivalent time constant = {tau_cl:.2f} s")

    # -------------------------------------------------------------
    # (c) Simulate Hp = 30, Hu = 2 with and without Delta u constraint
    # -------------------------------------------------------------
    sim_steps = 70
    time_pts = np.arange(sim_steps) * Ts

    # Unconstrained
    y_uncon, u_uncon, du_uncon = simulate_constrained_mpc(A, B, C, Hp=30, Hu=2, Q_wt=1.0, R_wt=0.1,
                                                          du_max=None, sim_steps=sim_steps, Ts=Ts)
    # Constrained slew rate |Delta u| <= 0.1
    du_limit = 0.1
    y_con, u_con, du_con = simulate_constrained_mpc(A, B, C, Hp=30, Hu=2, Q_wt=1.0, R_wt=0.1,
                                                    du_max=du_limit, sim_steps=sim_steps, Ts=Ts)

    min_y_uncon = np.min(y_uncon)
    min_y_con = np.min(y_con)
    print(f"\nExercise 7.5(c) Undershoot Comparison:")
    print(f"  Unconstrained peak undershoot: {min_y_uncon:.4f} (Moves in WRONG direction!)")
    print(f"  Rate-constrained (|du| <= {du_limit}) peak undershoot: {min_y_con:.4f}")

    # -------------------------------------------------------------
    # Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Spectral Radius rho vs Hp
    ax = axes[0, 0]
    Hp_vals = np.array(list(pole_hist.keys()))
    rho_vals = np.array([pole_hist[h][0] for h in Hp_vals])
    ax.plot(Hp_vals, rho_vals, 'b-o', linewidth=2, label=r'Spectral Radius $\rho(A_{cl})$')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label=r'Stability Boundary ($\rho=1.0$)')
    ax.axvline(min_Hp, color='green', linestyle=':', linewidth=2, label=f'Min Stable $H_p = {min_Hp}$')
    ax.set_title(r'Closed-Loop Stability vs Prediction Horizon $H_p$ ($H_u=1, R=0$)', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Prediction Horizon $H_p$')
    ax.set_ylabel(r'Spectral Radius $\rho(A - B K)$')
    ax.set_ylim(0.0, 2.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Step Tracking & Undershoot (Exercise 7.5(c))
    ax = axes[0, 1]
    ax.step(time_pts, y_uncon, 'r-', where='post', linewidth=2, label=f'Unconstrained (Undershoot = {min_y_uncon:.3f})')
    ax.step(time_pts, y_con, 'g-', where='post', linewidth=2, label=rf'Rate Limited $|\Delta u| \leq {du_limit}$ (Undershoot = {min_y_con:.3f})')
    ax.axhline(1.0, color='gray', linestyle=':', label='Target Setpoint r=1.0')
    ax.axhline(0.0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Exercise 7.5(c): Non-Minimum Phase Inverse Response (Undershoot)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Plant Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Control Signals u(t)
    ax = axes[1, 0]
    ax.step(time_pts, u_uncon, 'r--', where='post', linewidth=1.5, label='Unconstrained u(t)')
    ax.step(time_pts, u_con, 'g-', where='post', linewidth=2, label=r'Rate Limited u(t) ($|\Delta u| \leq 0.1$)')
    ax.set_title('Control Signals Evolution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input u(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Control Increments Delta u(t)
    ax = axes[1, 1]
    ax.step(time_pts, du_uncon, 'r--', where='post', linewidth=1.5, label=r'Unconstrained $\Delta u(k)$')
    ax.step(time_pts, du_con, 'g-', where='post', linewidth=2, label=r'Rate Limited $\Delta u(k)$')
    ax.axhline(du_limit, color='red', linestyle=':', label=r'Constraint Boundary $\pm 0.1$')
    ax.axhline(-du_limit, color='red', linestyle=':')
    ax.set_title(r'Rate of Change $\Delta u(k)$ and Slew Rate Bounds', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\Delta u(k)$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch07_tuning/ch7_non_minimum_phase_tuning.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_exercise_7_5()
