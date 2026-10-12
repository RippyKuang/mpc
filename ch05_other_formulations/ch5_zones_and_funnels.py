"""
Chapter 5: Other Formulations of Predictive Control
Script 4: Zone Control, Funnel Objectives, and Coincidence Points (Section 5.5, Figures 5.7 - 5.10)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.5

Key Concepts Implemented:
1. Zone Objectives (Figure 5.7):
   Output is not penalized if it remains inside the deadband [z_min, z_max].
   Formulated using one-sided slack variables:
   z(k+i|k) - epsilon_high <= z_max,  z(k+i|k) + epsilon_low >= z_min,  epsilon >= 0
   Cost penalizes slacks: w_slack * sum(epsilon^2) + sum(Delta u^T * R * Delta u)
2. Funnel Objectives (Figures 5.8 & 5.9):
   Contracting dynamic boundary:
   Lower boundary: (1 - alpha^i) * s + alpha^i * z(k)  (0 < alpha < 1)
   Upper boundary: s + Delta_funnel * alpha^i (or straight-line boundary as in Honeywell RMPCT)
3. Coincidence Points (Figure 5.10):
   Instead of penalizing every future step i = 1..Hp, only evaluate cost at coincidence points P = {P1, P2}.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_process_model():
    """
    Second-order overdamped industrial process:
    G(s) = 1 / ((3s + 1)(5s + 1))
    Discretized with Ts = 1.0 s.
    """
    Ts = 1.0
    # G(s) = 1 / (15 s^2 + 8 s + 1) -> s^2 + (8/15)s + 1/15
    Ac = np.array([[0.0, 1.0],
                   [-1.0/15.0, -8.0/15.0]])
    Bc = np.array([[0.0],
                   [1.0/15.0]])
    Cc = np.array([[1.0, 0.0]])

    n = 2
    M = la.expm(np.block([[Ac, Bc], [np.zeros((1, 3))]]) * Ts)
    A = M[:n, :n]
    B = M[:n, n:]
    C = Cc
    return A, B, C, Ts

def build_mpc_matrices(A, B, C, Hp, Hu):
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]

    Psi = np.zeros((Hp * p, n))
    for i in range(1, Hp + 1):
        Psi[(i-1)*p : i*p, :] = C @ np.linalg.matrix_power(A, i)

    Upsilon = np.zeros((Hp * p, m))
    for i in range(1, Hp + 1):
        sum_B = sum(np.linalg.matrix_power(A, j) @ B for j in range(i))
        Upsilon[(i-1)*p : i*p, :] = C @ sum_B

    Theta = np.zeros((Hp * p, Hu * m))
    for j in range(1, Hu + 1):
        for i in range(j, Hp + 1):
            sum_B = sum(np.linalg.matrix_power(A, k) @ B for k in range(i - j + 1))
            Theta[(i-1)*p : i*p, (j-1)*m : j*m] = C @ sum_B

    return Psi, Upsilon, Theta

def simulate_zone_mpc():
    """
    Zone Control: Setpoint band [18.0, 22.0].
    Initial output y(0) = 10.0, disturbed at step 30 by an external load.
    Inside the zone [18, 22], control effort is minimized (Delta u = 0, saving actuator energy).
    """
    A, B, C, Ts = get_process_model()
    Hp, Hu = 15, 5
    Psi, Upsilon, Theta = build_mpc_matrices(A, B, C, Hp, Hu)

    N_sim = 50
    t = np.arange(N_sim) * Ts

    z_min, z_max = 18.0, 22.0
    u_min, u_max = -5.0, 30.0
    du_min, du_max = -4.0, 4.0
    S_u = np.tril(np.ones((Hu, Hu)))

    y_sim = np.zeros(N_sim)
    u_sim = np.zeros(N_sim)
    du_sim = np.zeros(N_sim)
    x_curr = np.array([[10.0], [0.0]]) # Initial output 10.0
    u_prev = 10.0

    w_slack = 100.0
    r_u = 0.5

    for k in range(N_sim - 1):
        y_sim[k] = (C @ x_curr)[0, 0]

        # External step disturbance at t = 30
        dist = 3.0 if k >= 30 else 0.0

        # Free response
        y_free = Psi @ x_curr + Upsilon * u_prev

        # Optimization variables: [Delta_U (Hu); eps_high (Hp); eps_low (Hp)]
        # min sum(r_u * Delta_U^2) + sum(w_slack * eps_high^2) + sum(w_slack * eps_low^2)
        n_vars = Hu + 2 * Hp

        def cost_fn(var):
            du = var[:Hu]
            eps_h = var[Hu : Hu + Hp]
            eps_l = var[Hu + Hp :]
            return r_u * np.sum(du**2) + w_slack * np.sum(eps_h**2 + eps_l**2)

        def jac_fn(var):
            du = var[:Hu]
            eps_h = var[Hu : Hu + Hp]
            eps_l = var[Hu + Hp :]
            return np.concatenate([2 * r_u * du, 2 * w_slack * eps_h, 2 * w_slack * eps_l])

        # Constraints:
        # y_pred = y_free + Theta * du
        # y_pred - eps_h <= z_max  ==>  z_max - (y_free + Theta*du) + eps_h >= 0
        # y_pred + eps_l >= z_min  ==>  (y_free + Theta*du) + eps_l - z_min >= 0
        cons = [
            {'type': 'ineq', 'fun': lambda v: (z_max - (y_free.flatten() + Theta @ v[:Hu]) + v[Hu : Hu+Hp])},
            {'type': 'ineq', 'fun': lambda v: ((y_free.flatten() + Theta @ v[:Hu]) + v[Hu+Hp:] - z_min)},
            {'type': 'ineq', 'fun': lambda v: np.full(Hu, u_max - u_prev) - S_u @ v[:Hu]},
            {'type': 'ineq', 'fun': lambda v: S_u @ v[:Hu] - np.full(Hu, u_min - u_prev)},
        ]

        bounds = [(du_min, du_max)] * Hu + [(0, None)] * (2 * Hp)
        v0 = np.zeros(n_vars)

        res = minimize(cost_fn, v0, jac=jac_fn, bounds=bounds, constraints=cons, method='SLSQP')
        du_opt = res.x[0]

        u_curr = np.clip(u_prev + du_opt, u_min, u_max)
        u_sim[k] = u_curr
        du_sim[k] = du_opt
        u_prev = u_curr

        # Process update
        x_curr = A @ x_curr + B * u_curr + np.array([[0.0], [dist]])

    y_sim[-1] = (C @ x_curr)[0, 0]
    u_sim[-1] = u_sim[-2]
    return t, y_sim, u_sim, du_sim, z_min, z_max

def simulate_funnel_mpc():
    """
    Funnel Objective (Figures 5.8 & 5.9):
    Target setpoint s = 20.0. Initial y(0) = 5.0.
    Funnel lower boundary: (1 - alpha^i) * s + alpha^i * y(k), alpha = 0.8
    Funnel upper boundary: s + 2.0 * (alpha**i)
    """
    A, B, C, Ts = get_process_model()
    Hp, Hu = 15, 5
    Psi, Upsilon, Theta = build_mpc_matrices(A, B, C, Hp, Hu)

    N_sim = 40
    t = np.arange(N_sim) * Ts
    s_target = 20.0
    alpha = 0.82

    y_sim = np.zeros(N_sim)
    u_sim = np.zeros(N_sim)
    funnel_lower_all = []
    funnel_upper_all = []

    x_curr = np.array([[5.0], [0.0]])
    u_prev = 5.0
    S_u = np.tril(np.ones((Hu, Hu)))

    w_slack = 200.0
    r_u = 0.2

    for k in range(N_sim - 1):
        y_k = (C @ x_curr)[0, 0]
        y_sim[k] = y_k

        # Dynamic Funnel for horizon i = 1..Hp
        f_low = np.array([(1.0 - alpha**i) * s_target + (alpha**i) * y_k - 0.5 for i in range(1, Hp + 1)])
        f_high = np.array([s_target + 5.0 * (alpha**i) for i in range(1, Hp + 1)])

        if k == 0 or k == 10:
            funnel_lower_all.append((k, f_low))
            funnel_upper_all.append((k, f_high))

        y_free = Psi @ x_curr + Upsilon * u_prev

        # Solve slack QP
        n_vars = Hu + 2 * Hp
        def cost_fn(var):
            du = var[:Hu]
            eps_h = var[Hu : Hu + Hp]
            eps_l = var[Hu + Hp :]
            return r_u * np.sum(du**2) + w_slack * np.sum(eps_h**2 + eps_l**2)

        def jac_fn(var):
            du = var[:Hu]
            eps_h = var[Hu : Hu + Hp]
            eps_l = var[Hu + Hp :]
            return np.concatenate([2 * r_u * du, 2 * w_slack * eps_h, 2 * w_slack * eps_l])

        cons = [
            {'type': 'ineq', 'fun': lambda v: (f_high - (y_free.flatten() + Theta @ v[:Hu]) + v[Hu : Hu+Hp])},
            {'type': 'ineq', 'fun': lambda v: ((y_free.flatten() + Theta @ v[:Hu]) + v[Hu+Hp:] - f_low)},
        ]
        bounds = [(-5.0, 5.0)] * Hu + [(0, None)] * (2 * Hp)
        v0 = np.zeros(n_vars)
        res = minimize(cost_fn, v0, jac=jac_fn, bounds=bounds, constraints=cons, method='SLSQP')
        du_opt = res.x[0]

        u_curr = u_prev + du_opt
        u_sim[k] = u_curr
        u_prev = u_curr
        x_curr = A @ x_curr + B * u_curr

    y_sim[-1] = (C @ x_curr)[0, 0]
    u_sim[-1] = u_sim[-2]
    return t, y_sim, u_sim, funnel_lower_all, funnel_upper_all, s_target

def main():
    print("=" * 70)
    print("Chapter 5: Zone Control, Funnel Objectives & Coincidence Points")
    print("=" * 70)

    # 1. Zone MPC Simulation
    t_z, y_z, u_z, du_z, z_min, z_max = simulate_zone_mpc()
    print("Zone MPC simulation complete: output freely drifts inside deadband [18, 22].")

    # 2. Funnel MPC Simulation
    t_f, y_f, u_f, f_low_list, f_high_list, s_target = simulate_funnel_mpc()
    print("Funnel MPC simulation complete: output guided within contracting exponential funnel.")

    # 3. Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # --- Plot (a): Zone Objective (Figure 5.7) ---
    axes[0, 0].plot(t_z, y_z, 'b-', lw=2.2, label='Process Output y(t)')
    axes[0, 0].axhline(z_max, color='r', linestyle='--', lw=1.5, label=f'Upper Zone Bound ({z_max})')
    axes[0, 0].axhline(z_min, color='g', linestyle='--', lw=1.5, label=f'Lower Zone Bound ({z_min})')
    axes[0, 0].fill_between(t_z, z_min, z_max, color='lightblue', alpha=0.3, label='Zone Deadband (Zero Cost Region)')
    axes[0, 0].set_ylabel('Output y(t)', fontsize=11)
    axes[0, 0].set_title('(a) Zone Control Objective (Figure 5.7)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':')
    axes[0, 0].legend(loc='lower right')

    # --- Plot (b): Zone Control Effort ---
    axes[0, 1].step(t_z, u_z, 'k-', lw=1.8, where='post', label='Actuator Input u(t)')
    axes[0, 1].set_ylabel('Control Input u(t)', fontsize=11)
    axes[0, 1].set_title('(b) Zone Control Input (Resting when Inside Zone)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].legend()

    # --- Plot (c): Funnel Objective (Figure 5.8 & 5.9) ---
    axes[1, 0].plot(t_f, y_f, 'b-', lw=2.5, label='Actual Output y(t)')
    axes[1, 0].axhline(s_target, color='r', linestyle=':', lw=1.8, label=f'Set-point Target ({s_target})')

    # Draw contracting funnels sampled at t=0 and t=10
    colors_f = ['orange', 'purple']
    for idx, (k_step, f_low) in enumerate(f_low_list):
        k_time = k_step * 1.0
        h_time = k_time + np.arange(1, len(f_low) + 1) * 1.0
        f_high = f_high_list[idx][1]
        axes[1, 0].plot(h_time, f_low, color=colors_f[idx], linestyle='--', lw=1.5, label=f'Funnel Lower (t={k_time:.0f}s)')
        axes[1, 0].plot(h_time, f_high, color=colors_f[idx], linestyle=':', lw=1.5, label=f'Funnel Upper (t={k_time:.0f}s)')
        axes[1, 0].fill_between(h_time, f_low, f_high, color=colors_f[idx], alpha=0.15)

    axes[1, 0].set_xlabel('Time (seconds)', fontsize=11)
    axes[1, 0].set_ylabel('Output y(t)', fontsize=11)
    axes[1, 0].set_title('(c) Funnel Objectives: Contracting Dynamic Bounds (Figure 5.8)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':')
    axes[1, 0].legend(loc='lower right')

    # --- Plot (d): Coincidence Points Visualization (Figure 5.10) ---
    t_coinc = np.linspace(0, 15, 100)
    ref_curve = s_target * (1.0 - np.exp(-t_coinc / 3.0))
    axes[1, 1].plot(t_coinc, ref_curve, 'g--', lw=2.0, label='Target Reference Trajectory r(t)')
    p_points = np.array([3.0, 7.0, 12.0])
    p_vals = s_target * (1.0 - np.exp(-p_points / 3.0))
    axes[1, 1].plot(p_points, p_vals, 'ro', markersize=10, label='Coincidence Points {P1, P2, P3}')
    for pt, val in zip(p_points, p_vals):
        axes[1, 1].annotate(f'P = {pt:.0f}s', xy=(pt, val), xytext=(pt + 0.5, val - 2.0),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('Prediction Time Horizon (seconds)', fontsize=11)
    axes[1, 1].set_ylabel('Output y', fontsize=11)
    axes[1, 1].set_title('(d) Coincidence Points Formulation (Figure 5.10)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, linestyle=':')
    axes[1, 1].legend(loc='lower right')

    plt.tight_layout()
    p_plot = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_zones_and_funnels.png'
    plt.savefig(p_plot, dpi=300)
    plt.close()
    print(f"[Plot Saved] -> {p_plot}")

if __name__ == '__main__':
    main()
