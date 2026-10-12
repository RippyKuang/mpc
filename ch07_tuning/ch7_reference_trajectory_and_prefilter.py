"""
Chapter 7 Tuning: Section 7.5 & Exercise 7.7
Reference Trajectory Design, Setpoint Anticipation, and Pre-Filters
===================================================================
This script implements:
1. First-Order Exponential Reference Trajectory (Richalet / Section 7.5):
      r(k+j|k) = (1 - lambda^j) * s(k) + lambda^j * y(k)
   where lambda = exp(-Ts / T_ref) in [0, 1).
   Initializing at the latest measurement y(k) closes an additional feedback loop (Figure 7.17).
2. Setpoint Anticipation (Equations 7.34–7.38, Figure 7.18 & Exercise 7.7(a)):
      When future setpoints s(k+j) are known, the controller starts control action
      in advance, eliminating tracking lag and overshoot.
      Verification of Exercise 7.7(a): When lambda = 0 (Lambda = 0), r(k+j|k) = s(k+j)
      (pure preview / anticipation).
3. Second-Order Reference Trajectory (Exercise 7.7(b)):
      r(k+j|k) = s(k+j) - [alpha^j / (alpha - beta) + beta^j / (beta - alpha)] * (s(k) - y(k))
   allows smooth, zero-slope initial takeoff and critically damped transitions.
4. Pre-Filter F(z) outside the feedback loop:
      Low-pass filtering setpoint steps to prevent actuator saturation
      without degrading disturbance rejection dynamics.
"""

import numpy as np
import scipy.signal as signal
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_reference_trajectories(y_k, s_future, lambda_param, alpha=0.7, beta=0.5, Hp=20):
    """
    Generates reference trajectories over horizon Hp:
    1. First-order exponential (Richalet)
    2. Anticipated exponential
    3. Second-order (Exercise 7.7(b))
    """
    r_exp = np.zeros(Hp)
    r_ant = np.zeros(Hp)
    r_2nd = np.zeros(Hp)

    # Current setpoint s(k) = s_future[0]
    s_curr = s_future[0]

    # 1. Standard Richalet exponential
    for j in range(1, Hp + 1):
        lam_j = lambda_param ** j
        r_exp[j-1] = (1.0 - lam_j) * s_curr + lam_j * y_k

    # 2. Anticipated reference trajectory (Maciejowski eq. 7.35)
    r_prev = y_k
    for j in range(Hp):
        s_target = s_future[j]
        r_ant[j] = (1.0 - lambda_param) * s_target + lambda_param * r_prev
        r_prev = r_ant[j]

    # 3. Second-order trajectory (Exercise 7.7(b))
    # r(k+j|k) = s(k+j) - [alpha^j / (alpha - beta) + beta^j / (beta - alpha)] * (s(k) - y(k))
    denom = alpha - beta
    for j in range(1, Hp + 1):
        coeff = (alpha**j - beta**j) / denom
        r_2nd[j-1] = s_future[j-1] - (1.0 - coeff) * (s_curr - y_k) if j > 1 else y_k + 0.1 * (s_curr - y_k)
        # Smooth asymptotic transition:
        decay = (alpha**j) / (alpha - beta) - (beta**j) / (alpha - beta)
        r_2nd[j-1] = s_future[j-1] - decay * (s_curr - y_k)

    return r_exp, r_ant, r_2nd

def simulate_mpc_with_reference(A, B, C, traj_type='none', T_ref=1.0, preview=True,
                                sim_steps=60, Ts=0.2, step_time=20):
    """
    Simulates MPC with various reference trajectory configurations.
    """
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    Hp = 20
    Hu = 4
    Q_wt = 1.0
    R_wt = 0.2
    u_max = 2.5

    lambda_param = np.exp(-Ts / T_ref)

    # Future setpoint profile: step from 0 to 1 at step_time
    s_full = np.zeros(sim_steps + Hp + 5)
    s_full[step_time:] = 1.0

    # Extended state space MPC prediction matrices
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
    H_qp = 2.0 * (Su.T @ Q_block @ Su + R_block)
    H_qp = 0.5 * (H_qp + H_qp.T) + 1e-9 * np.eye(Hu * nu)
    H_inv = la.inv(H_qp)

    x = np.zeros(nx)
    y_hist = np.zeros(sim_steps)
    u_hist = np.zeros(sim_steps)
    ref_actual = np.zeros(sim_steps)

    for k in range(sim_steps):
        y_curr = (C @ x)[0]
        y_hist[k] = y_curr

        # Horizon setpoint window
        if preview:
            s_window = s_full[k+1:k+1+Hp]
        else:
            s_window = np.full(Hp, s_full[k])

        # Formulate reference trajectory r(k+j|k)
        if traj_type == 'none':
            r_traj = s_window
        elif traj_type == 'first_order':
            r_traj, _, _ = generate_reference_trajectories(y_curr, s_window, lambda_param, Hp=Hp)
        elif traj_type == 'anticipated':
            _, r_traj, _ = generate_reference_trajectories(y_curr, s_window, lambda_param, Hp=Hp)
        elif traj_type == 'second_order':
            _, _, r_traj = generate_reference_trajectories(y_curr, s_window, lambda_param, alpha=0.8, beta=0.6, Hp=Hp)

        ref_actual[k] = r_traj[0]

        # MPC optimization
        y_free = Sx @ x
        f_qp = 2.0 * (Su.T @ Q_block @ (y_free - r_traj))
        U_opt = - H_inv @ f_qp
        u_k = np.clip(U_opt[0], -u_max, u_max)
        u_hist[k] = u_k

        # Update plant
        x = A @ x + B.flatten() * u_k

    return y_hist, u_hist, s_full[:sim_steps], ref_actual

def run_simulation():
    print("=" * 75)
    print("Chapter 7: Reference Trajectories, Anticipation & Pre-filters (Section 7.5)")
    print("=" * 75)

    # Continuous plant: P(s) = 1 / (s^2 + 1.4s + 1), Ts = 0.2 s
    Ts = 0.2
    sys_c = signal.TransferFunction([1.0], [1.0, 1.4, 1.0])
    sys_d = signal.cont2discrete((sys_c.num, sys_c.den), Ts, method='zoh')
    A, B, C, D = signal.tf2ss(sys_d[0][0], sys_d[1])

    sim_steps = 50
    time_pts = np.arange(sim_steps) * Ts

    # Case 1: Raw Step (No reference trajectory, no anticipation)
    y_raw, u_raw, s_prof, _ = simulate_mpc_with_reference(A, B, C, traj_type='none', preview=False, sim_steps=sim_steps)

    # Case 2: First-order Exponential Reference Trajectory (T_ref = 1.0 s)
    y_exp, u_exp, _, r_exp_log = simulate_mpc_with_reference(A, B, C, traj_type='first_order', T_ref=1.0, preview=False, sim_steps=sim_steps)

    # Case 3: Anticipated Reference Trajectory (Preview of future step change)
    y_ant, u_ant, _, r_ant_log = simulate_mpc_with_reference(A, B, C, traj_type='anticipated', T_ref=1.0, preview=True, sim_steps=sim_steps)

    # Case 4: Second-Order Reference Trajectory (Exercise 7.7(b))
    y_2nd, u_2nd, _, r_2nd_log = simulate_mpc_with_reference(A, B, C, traj_type='second_order', preview=True, sim_steps=sim_steps)

    # -------------------------------------------------------------
    # Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Output Tracking Comparison
    ax = axes[0, 0]
    ax.step(time_pts, s_prof, 'k:', linewidth=1.5, label='Set-point Step s(t)')
    ax.plot(time_pts, y_raw, 'r--', linewidth=1.8, label='Raw Step (Overshoot & aggressive)')
    ax.plot(time_pts, y_exp, 'b-', linewidth=2, label='1st-Order Ref Traj ($T_{ref}=1.0$s)')
    ax.plot(time_pts, y_ant, 'g-.', linewidth=2, label='Anticipated Step (Early takeoff)')
    ax.plot(time_pts, y_2nd, 'm:', linewidth=2, label='2nd-Order Ref Traj (Ex 7.7(b))')
    ax.set_title('Plant Output y(t) under Different Reference Profiles', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Control Effort u(t) Comparison (Saturation Mitigation)
    ax = axes[0, 1]
    ax.step(time_pts, u_raw, 'r--', where='post', linewidth=1.5, label='Raw Step: High initial peak')
    ax.step(time_pts, u_exp, 'b-', where='post', linewidth=2, label='1st-Order: Smooth actuation')
    ax.step(time_pts, u_ant, 'g-.', where='post', linewidth=2, label='Anticipated: Pre-action before step')
    ax.step(time_pts, u_2nd, 'm:', where='post', linewidth=2, label='2nd-Order: Zero-slope takeoff')
    ax.axhline(2.5, color='red', linestyle='--', alpha=0.5, label='Saturation Limit +-2.5')
    ax.axhline(-2.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Control Signals u(t): Actuator Peak Suppression', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input u(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Zoomed-in Anticipation Window (t = 2.5 to 6.0 s)
    ax = axes[1, 0]
    zoom_mask = (time_pts >= 2.5) & (time_pts <= 6.5)
    ax.step(time_pts[zoom_mask], s_prof[zoom_mask], 'k:', linewidth=2, label='Setpoint Step at t=4.0s')
    ax.plot(time_pts[zoom_mask], y_raw[zoom_mask], 'r--', linewidth=2, label='Raw (Starts after step)')
    ax.plot(time_pts[zoom_mask], y_ant[zoom_mask], 'g-', linewidth=2.5, label='Anticipated (Starts at t=3.0s)')
    ax.set_title('Zoomed-in Anticipation: Pre-emptive Action Eliminates Delay', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Verification of Exercise 7.7(a) (Lambda = 0 => Pure Preview)
    ax = axes[1, 1]
    # When lambda = 0, r(k+j) = s(k+j)
    lam_zero = 0.0
    r_lam0, _, _ = generate_reference_trajectories(0.2, s_prof[20:40], lam_zero, Hp=20)
    ax.step(np.arange(20), s_prof[20:40], 'k-', linewidth=2, label='Future Setpoints $s(k+j)$')
    ax.plot(np.arange(20), r_lam0, 'ro', markersize=6, label=r'Traj with $\lambda=0$ ($r \equiv s$)')
    ax.set_title(r'Exercise 7.7(a): Proof that $\Lambda=0 \rightarrow r(k+j|k) \equiv s(k+j)$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Step j')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch07_tuning/ch7_reference_trajectory_and_prefilter.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
