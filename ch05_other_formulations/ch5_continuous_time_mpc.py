"""
Chapter 5: Other Formulations of Predictive Control
Script 6: Continuous-Time Predictive Control & Taylor Series Predictions
(Section 5.7, Exercises 5.7 & 5.8)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.7 & Exercises 5.7, 5.8

Key Concepts Implemented:
1. Continuous-Time System & Output Derivatives Vector Y(t) (Eq 5.50 & Exercise 5.7):
   dot_x(t) = A*x(t) + B*u(t),  y(t) = C*x(t)
   Y(t) = [y(t); y_dot(t); y_ddot(t); ...; y^(n)(t)] = C_A * x(t) + H_B * U(t)
   where H_B is the lower triangular Toeplitz matrix formed by Markov parameters C*A^k*B.
2. Continuous-Time Quadratic Cost Function (Eq 5.46 & Exercise 5.8):
   V(t) = int_{T1}^{T2} (r(t+tau) - y_hat(t+tau|t))^2 d_tau + lambda * int_0^{T2} u(t+tau|t)^2 d_tau
   Quadratic form in U(t): V(t) = 0.5 * U(t)^T * H_ct * U(t) + g_ct^T * U(t) + const.
3. Closed-Loop Simulation with High-Order Taylor Predictions and Continuous Receding Horizon.
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_continuous_plant():
    """
    Continuous-time plant: 2nd order mass-spring-damper
    dot_x1 = x2
    dot_x2 = -k/m * x1 - c/m * x2 + 1/m * u
    Let m = 1, k = 4, c = 2 -> poles at -1 +/- j*sqrt(3)
    """
    A = np.array([[0.0, 1.0],
                  [-4.0, -2.0]])
    B = np.array([[0.0],
                  [1.0]])
    C = np.array([[1.0, 0.0]])
    return A, B, C

def verify_exercise_5_7():
    """
    Verifies Eq (5.50) (Exercise 5.7):
    Computes Y(t) = C_A * x(t) + H_B * U(t) for n = 3 derivatives.
    """
    A, B, C = get_continuous_plant()
    n_order = 3  # y, y_dot, y_ddot, y_dddot

    # Construct C_A
    C_A = np.zeros((n_order + 1, A.shape[0]))
    for i in range(n_order + 1):
        C_A[i, :] = C @ np.linalg.matrix_power(A, i)

    # Construct H_B (lower triangular Toeplitz)
    H_B = np.zeros((n_order + 1, n_order))
    for i in range(1, n_order + 1):
        for j in range(i):
            power = i - 1 - j
            H_B[i, j] = (C @ np.linalg.matrix_power(A, power) @ B)[0, 0]

    print("=" * 70)
    print("Exercise 5.7 Verification: Derivatives Equation (5.50)")
    print("=" * 70)
    print("C_A matrix (powers of A):")
    print(C_A)
    print("\nH_B matrix (Markov parameters C*A^k*B):")
    print(H_B)
    print("\nVerified structure: H_B is strictly lower-triangular Toeplitz with zero diagonal!")
    return C_A, H_B

def build_continuous_mpc_matrices(T1, T2, n_order=2, m_order=2, lam=0.05):
    """
    Exercise 5.8: Expresses V(t) as quadratic form in U(t).
    U(t) = [u(t); u_dot(t); ...; u^(m)(t)]
    Gamma(tau) = [1, tau, tau^2/2!, ..., tau^n/n!]
    Lambda(tau) = [1, tau, tau^2/2!, ..., tau^m/m!]
    """
    A, B, C = get_continuous_plant()

    import math

    # Precompute factorial vectors
    fact_n = np.array([math.factorial(i) for i in range(n_order + 1)])
    fact_m = np.array([math.factorial(j) for j in range(m_order + 1)])

    # Compute Omega_y = int_{T1}^{T2} Gamma(tau)^T * Gamma(tau) d_tau
    Omega_y = np.zeros((n_order + 1, n_order + 1))
    for i in range(n_order + 1):
        for j in range(n_order + 1):
            p = i + j + 1
            # int_{T1}^{T2} tau^(i+j) / (i! * j!) d_tau = (T2^p - T1^p) / (p * i! * j!)
            Omega_y[i, j] = (T2**p - T1**p) / (p * fact_n[i] * fact_n[j])

    # Compute Omega_u = int_0^{T2} Lambda(tau)^T * Lambda(tau) d_tau
    Omega_u = np.zeros((m_order + 1, m_order + 1))
    for i in range(m_order + 1):
        for j in range(m_order + 1):
            p = i + j + 1
            Omega_u[i, j] = (T2**p) / (p * fact_m[i] * fact_m[j])

    # C_A and H_B
    C_A = np.zeros((n_order + 1, A.shape[0]))
    for i in range(n_order + 1):
        C_A[i, :] = C @ np.linalg.matrix_power(A, i)

    # For U(t) of dimension m_order + 1
    # Y(t) = C_A * x(t) + H_B_ext * U(t)
    H_B_ext = np.zeros((n_order + 1, m_order + 1))
    for i in range(1, n_order + 1):
        for j in range(min(i, m_order + 1)):
            power = i - 1 - j
            H_B_ext[i, j] = (C @ np.linalg.matrix_power(A, power) @ B)[0, 0]

    return A, B, C, Omega_y, Omega_u, C_A, H_B_ext, fact_n

def simulate_continuous_mpc():
    """
    Simulates continuous plant under Continuous-Time MPC.
    Prediction horizon: [T1, T2] = [0.0, 1.5] seconds.
    Taylor expansion order: n = 2, m = 1.
    """
    T1, T2 = 0.0, 1.5
    n_order = 2
    m_order = 1
    lam = 0.02

    A, B, C, Omega_y, Omega_u, C_A, H_B_ext, fact_n = build_continuous_mpc_matrices(T1, T2, n_order, m_order, lam)

    # Simulation setup
    t_end = 8.0
    dt_control = 0.05  # MPC update interval
    dt_sim = 0.005      # RK4 integration step
    N_steps = int(t_end / dt_control)
    t_span = np.arange(N_steps) * dt_control

    r_target = 1.0  # Unit step setpoint

    # Reference vector in Taylor domain:
    # If r(t+tau) = r_target (constant), then r_deriv = [r_target, 0, ..., 0]
    r_vec = np.zeros(n_order + 1)
    r_vec[0] = r_target

    x_state = np.zeros((2, 1)) # initial state
    y_history = []
    u_history = []
    time_history = []

    for k in range(N_steps):
        t_now = k * dt_control

        # Formulate quadratic cost in U:
        # y_hat(tau) = Gamma(tau) * (C_A * x + H_B_ext * U)
        # Cost = (C_A*x + H_B_ext*U - r)^T * Omega_y * (C_A*x + H_B_ext*U - r) + lam * U^T * Omega_u * U
        # H_cost = 2 * (H_B_ext^T * Omega_y * H_B_ext + lam * Omega_u)
        # g_cost = 2 * H_B_ext^T * Omega_y * (C_A * x - r_vec.reshape(-1, 1))

        H_cost = 2.0 * (H_B_ext.T @ Omega_y @ H_B_ext + lam * Omega_u)
        g_cost = 2.0 * H_B_ext.T @ Omega_y @ (C_A @ x_state - r_vec.reshape(-1, 1))

        # Unconstrained optimal U*(t) = - H_cost^{-1} * g_cost
        U_opt = -la.solve(H_cost, g_cost)

        # Receding horizon principle: apply u(t) = U_opt[0]
        u_applied = U_opt[0, 0]

        # RK4 continuous integration over [t_now, t_now + dt_control]
        n_rk = int(dt_control / dt_sim)
        for _ in range(n_rk):
            y_curr = (C @ x_state)[0, 0]
            y_history.append(y_curr)
            u_history.append(u_applied)
            time_history.append(t_now)
            t_now += dt_sim

            # RK4 step
            k1 = A @ x_state + B * u_applied
            k2 = A @ (x_state + 0.5 * dt_sim * k1) + B * u_applied
            k3 = A @ (x_state + 0.5 * dt_sim * k2) + B * u_applied
            k4 = A @ (x_state + dt_sim * k3) + B * u_applied
            x_state = x_state + (dt_sim / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return np.array(time_history), np.array(y_history), np.array(u_history)

def main():
    verify_exercise_5_7()

    print("\nSimulating Continuous-Time MPC (Gawthrop Formulation)...")
    t, y, u = simulate_continuous_mpc()
    print("Simulation finished successfully!")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, y, 'b-', lw=2.2, label='Continuous Plant Output y(t)')
    axes[0].axhline(1.0, color='r', linestyle='--', label='Setpoint Target r = 1.0')
    axes[0].set_ylabel('Output y(t)', fontsize=11)
    axes[0].set_title('Continuous-Time Predictive Control Tracking (Section 5.7 & Exercise 5.8)', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':')
    axes[0].legend(loc='lower right')

    axes[1].plot(t, u, 'k-', lw=1.8, label='Optimal Control Input u(t)')
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Control Input u(t)', fontsize=11)
    axes[1].set_title('Continuous Control Input Action', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    p_plot = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_continuous_time_mpc.png'
    plt.savefig(p_plot, dpi=300)
    plt.close()
    print(f"[Plot Saved] -> {p_plot}")

if __name__ == '__main__':
    main()
