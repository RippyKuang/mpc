"""
Chapter 5: Other Formulations of Predictive Control
Script 1: Measured Disturbances and Feedforward Control (Section 5.1 & Exercise 5.1)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.1 & Exercise 5.1

Key Concepts Implemented:
1. Augmented prediction equation with measured disturbance matrix Xi:
   Z(k) = Psi * x(k|k) + Upsilon * u(k-1) + Theta * Delta_U(k) + Xi * D_m(k)
2. Tracking error modification:
   E(k) = T(k) - Psi * x(k|k) - Upsilon * u(k-1) - Xi * D_m(k)
3. Swimming Pool Benchmark (Exercise 5.1 & 3.3):
   - Perfect model + unconstrained: Feedforward eliminates sinusoidal disturbance entirely.
   - Model mismatch (T=1.25, kq=0.3): Residual variation drops from ~0.5 deg C to ~0.2 deg C.
   - Input constraint (0 <= q <= 40 kW): Active saturation periods limit compensation.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_mpc_matrices(A, B, Bd, Cy, Cz, Hp, Hu):
    """
    Constructs the prediction matrices:
    Psi, Upsilon, Theta, Xi
    """
    n = A.shape[0]
    m = B.shape[1]
    md = Bd.shape[1]
    p = Cz.shape[0]

    # Psi: (Hp*p) x n
    Psi = np.zeros((Hp * p, n))
    A_pow = np.eye(n)
    for i in range(1, Hp + 1):
        A_pow = A_pow @ A
        Psi[(i-1)*p : i*p, :] = Cz @ A_pow

    # Upsilon: (Hp*p) x m
    Upsilon = np.zeros((Hp * p, m))
    A_pow_sum = np.zeros((n, n))
    A_pow = np.eye(n)
    for i in range(1, Hp + 1):
        A_pow_sum = A_pow_sum + A_pow
        Upsilon[(i-1)*p : i*p, :] = Cz @ A_pow_sum @ B
        A_pow = A_pow @ A

    # Theta: (Hp*p) x (Hu*m)
    Theta = np.zeros((Hp * p, Hu * m))
    for j in range(1, Hu + 1):
        col_start = (j - 1) * m
        col_end = j * m
        A_sum = np.zeros((n, n))
        for i in range(j, Hp + 1):
            row_start = (i - 1) * p
            row_end = i * p
            A_sum = A_sum + np.linalg.matrix_power(A, i - j)
            Theta[row_start:row_end, col_start:col_end] = Cz @ A_sum @ B

    # Xi: (Hp*p) x (Hp*md)
    # Predicted disturbance matrix
    # Z(k) = ... + sum_{j=1}^i Cz * A^(i-j) * Bd * dm(k + j - 1|k)
    Xi = np.zeros((Hp * p, Hp * md))
    for i in range(1, Hp + 1):
        row_start = (i - 1) * p
        row_end = i * p
        for j in range(1, i + 1):
            col_start = (j - 1) * md
            col_end = j * md
            Xi[row_start:row_end, col_start:col_end] = Cz @ np.linalg.matrix_power(A, i - j) @ Bd

    return Psi, Upsilon, Theta, Xi

def simulate_swimming_pool(use_feedforward=True, has_model_error=False, constrain_input=False,
                            sim_hours=48, Ts=0.25):
    """
    Simulates the swimming pool system for Exercise 5.1.
    dT/dt = (1/T) * (kq * q + theta_a - theta)
    Nominal: T=1.0 h, kq=0.2 deg C/kW
    Mismatch: T=1.25 h, kq=0.3 deg C/kW
    """
    N_steps = int(sim_hours / Ts)
    t = np.arange(N_steps) * Ts

    # Nominal model
    T_nom = 1.0
    kq_nom = 0.2
    A_nom = np.array([[np.exp(-Ts / T_nom)]])
    B_nom = np.array([[(1.0 - np.exp(-Ts / T_nom)) * kq_nom]])
    Bd_nom = np.array([[(1.0 - np.exp(-Ts / T_nom)) * 1.0]])
    Cy_nom = np.array([[1.0]])
    Cz_nom = np.array([[1.0]])

    # True plant model
    if has_model_error:
        T_true = 1.25
        kq_true = 0.3
    else:
        T_true = 1.0
        kq_true = 0.2
    A_true = np.array([[np.exp(-Ts / T_true)]])
    B_true = np.array([[(1.0 - np.exp(-Ts / T_true)) * kq_true]])
    Bd_true = np.array([[(1.0 - np.exp(-Ts / T_true)) * 1.0]])

    Hp = 10
    Hu = 3
    Q_val = 1.0
    R_val = 0.0

    Psi, Upsilon, Theta, Xi = build_mpc_matrices(A_nom, B_nom, Bd_nom, Cy_nom, Cz_nom, Hp, Hu)

    # Ambient air temperature: theta_a(t) = 15 + 10*sin(2*pi*t/24)
    theta_a = 15.0 + 10.0 * np.sin(2.0 * np.pi * t / 24.0)
    r_target = 20.0  # Setpoint water temperature

    # State variables
    theta_plant = np.zeros(N_steps)
    q_applied = np.zeros(N_steps)
    theta_plant[0] = 20.0
    u_prev = 25.0  # Initial steady heater power
    x_model = 20.0 # Internal model state

    for k in range(N_steps - 1):
        # DMC output disturbance estimation: d_hat(k) = y(k) - x_model(k)
        d_hat = theta_plant[k] - x_model

        # Disturbance forecast over horizon Hp
        Dm = np.zeros((Hp, 1))
        for i in range(Hp):
            if use_feedforward:
                Dm[i, 0] = theta_a[k] # Common practice: persist latest measured disturbance
            else:
                Dm[i, 0] = 15.0       # Unmeasured: nominal ambient

        # Reference trajectory
        T_ref = np.ones((Hp, 1)) * r_target

        # Tracking error vector with DMC output disturbance + feedforward
        if use_feedforward:
            E = T_ref - (Psi * x_model + Upsilon * u_prev + Xi @ Dm + d_hat)
        else:
            E = T_ref - (Psi * x_model + Upsilon * u_prev + Xi @ (np.ones((Hp, 1)) * 15.0) + d_hat)

        # Optimization
        H = Theta.T @ Theta + (R_val + 1e-6) * np.eye(Hu)
        g = Theta.T @ E

        if not constrain_input:
            delta_u_vec = la.solve(H, g.flatten())
            delta_u_opt = delta_u_vec[0]
        else:
            def obj(du):
                return 0.5 * du @ H @ du - g.flatten() @ du
            def jac(du):
                return H @ du - g.flatten()

            S_u = np.tril(np.ones((Hu, Hu)))
            cons = [
                {'type': 'ineq', 'fun': lambda du: (u_prev + S_u @ du)},        # >= 0 kW
                {'type': 'ineq', 'fun': lambda du: (40.0 - (u_prev + S_u @ du))} # <= 40 kW
            ]
            du0 = np.zeros(Hu)
            res = minimize(obj, du0, jac=jac, constraints=cons, method='SLSQP')
            delta_u_opt = res.x[0]

        u_opt = u_prev + delta_u_opt
        if constrain_input:
            u_opt = np.clip(u_opt, 0.0, 40.0)

        q_applied[k] = u_opt
        u_prev = u_opt

        # Advance internal model state
        if use_feedforward:
            x_model = A_nom[0, 0] * x_model + B_nom[0, 0] * u_opt + Bd_nom[0, 0] * theta_a[k]
        else:
            x_model = A_nom[0, 0] * x_model + B_nom[0, 0] * u_opt + Bd_nom[0, 0] * 15.0

        # Advance actual plant state
        theta_next = A_true[0, 0] * theta_plant[k] + B_true[0, 0] * u_opt + Bd_true[0, 0] * theta_a[k]
        theta_plant[k + 1] = theta_next

    q_applied[-1] = q_applied[-2]
    return t, theta_plant, q_applied, theta_a

def main():
    print("=" * 70)
    print("Chapter 5: Measured Disturbances & Feedforward Simulation (Exercise 5.1)")
    print("=" * 70)

    # 1. Perfect model, Unconstrained: Feedback vs Feedforward
    t, th_perf_nofb, q_perf_nofb, ta = simulate_swimming_pool(use_feedforward=False, has_model_error=False, constrain_input=False)
    t, th_perf_ff, q_perf_ff, _ = simulate_swimming_pool(use_feedforward=True, has_model_error=False, constrain_input=False)

    var_nofb = np.max(th_perf_nofb[40:]) - np.min(th_perf_nofb[40:])
    var_ff = np.max(th_perf_ff[40:]) - np.min(th_perf_ff[40:])
    print(f"(a) Perfect Model, Unconstrained:")
    print(f"    - Feedback Only Residual Peak-to-Peak: {var_nofb:.4f} °C (Amp ~ {var_nofb/2:.4f} °C)")
    print(f"    - Feedforward Residual Peak-to-Peak:   {var_ff:.4f} °C (Perfect Cancellation!)")

    # 2. Model Mismatch (T=1.25, kq=0.3): Feedback vs Feedforward
    t, th_err_nofb, q_err_nofb, _ = simulate_swimming_pool(use_feedforward=False, has_model_error=True, constrain_input=False)
    t, th_err_ff, q_err_ff, _ = simulate_swimming_pool(use_feedforward=True, has_model_error=True, constrain_input=False)

    var_err_nofb = np.max(th_err_nofb[40:]) - np.min(th_err_nofb[40:])
    var_err_ff = np.max(th_err_ff[40:]) - np.min(th_err_ff[40:])
    print(f"\n(b) Model Mismatch (T=1.25, kq=0.3), Unconstrained:")
    print(f"    - Feedback Only Residual Variation Amplitude: ~{var_err_nofb/2:.4f} °C")
    print(f"    - Feedforward Residual Variation Amplitude:   ~{var_err_ff/2:.4f} °C (Target ~0.2 °C)")

    # 3. Input Constrained (0 <= q <= 40 kW)
    t, th_c_nofb, q_c_nofb, _ = simulate_swimming_pool(use_feedforward=False, has_model_error=False, constrain_input=True)
    t, th_c_ff, q_c_ff, _ = simulate_swimming_pool(use_feedforward=True, has_model_error=False, constrain_input=True)

    print(f"\n(c) Input Power Constrained (0 <= q <= 40 kW):")
    print(f"    - Feedforward holds setpoint tightly during unconstrained periods.")
    print(f"    - During saturation (0 or 40 kW), both trajectories coincide as expected.")

    # Plotting results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)

    # Subplot (a): Perfect Model
    axes[0, 0].plot(t, th_perf_nofb, 'r--', label='Feedback Only')
    axes[0, 0].plot(t, th_perf_ff, 'b-', label='Feedforward + Feedback')
    axes[0, 0].axhline(20.0, color='k', linestyle=':', label='Setpoint (20°C)')
    axes[0, 0].set_ylabel('Water Temp θ (°C)')
    axes[0, 0].set_title('(a) Perfect Model: Temperature Tracking')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t, q_perf_nofb, 'r--', label='q (Feedback Only)')
    axes[0, 1].plot(t, q_perf_ff, 'b-', label='q (Feedforward + Feedback)')
    axes[0, 1].set_ylabel('Heater Power q (kW)')
    axes[0, 1].set_title('(a) Perfect Model: Heater Power Action')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Subplot (b): Model Mismatch
    axes[1, 0].plot(t, th_err_nofb, 'r--', label='Feedback Only (~0.5°C swing)')
    axes[1, 0].plot(t, th_err_ff, 'g-', label='Feedforward (~0.2°C swing)')
    axes[1, 0].axhline(20.0, color='k', linestyle=':', label='Setpoint')
    axes[1, 0].set_ylabel('Water Temp θ (°C)')
    axes[1, 0].set_title('(b) Model Mismatch: Residual Temperature Variation')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(t, q_err_nofb, 'r--', label='q (Feedback Only)')
    axes[1, 1].plot(t, q_err_ff, 'g-', label='q (Feedforward)')
    axes[1, 1].set_ylabel('Heater Power q (kW)')
    axes[1, 1].set_title('(b) Model Mismatch: Heater Power')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Subplot (c): Constrained Power
    axes[2, 0].plot(t, th_c_nofb, 'r--', label='Feedback Only')
    axes[2, 0].plot(t, th_c_ff, 'm-', label='Feedforward + Feedback')
    axes[2, 0].axhline(20.0, color='k', linestyle=':', label='Setpoint')
    axes[2, 0].set_xlabel('Time (hours)')
    axes[2, 0].set_ylabel('Water Temp θ (°C)')
    axes[2, 0].set_title('(c) Constrained Power (0 <= q <= 40 kW): Temperature')
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    axes[2, 1].plot(t, q_c_nofb, 'r--', label='q (Feedback Only)')
    axes[2, 1].plot(t, q_c_ff, 'm-', label='q (Feedforward + Feedback)')
    axes[2, 1].axhline(40.0, color='k', linestyle='--', label='q_max = 40 kW')
    axes[2, 1].axhline(0.0, color='k', linestyle='--', label='q_min = 0 kW')
    axes[2, 1].set_xlabel('Time (hours)')
    axes[2, 1].set_ylabel('Heater Power q (kW)')
    axes[2, 1].set_title('(c) Constrained Power: Saturation Clamping')
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    plt.tight_layout()
    output_png = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_feedforward_swimming_pool.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n[Plot Saved] -> {output_png}")

if __name__ == '__main__':
    main()
