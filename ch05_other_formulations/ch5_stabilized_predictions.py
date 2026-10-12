"""
Chapter 5: Other Formulations of Predictive Control
Script 2: Stabilized Predictions for Unstable Systems (Section 5.2)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.2

Key Concepts Implemented:
1. Open-loop prediction breakdown for unstable systems:
   A^i blows up exponentially -> Condition number kappa(Theta) diverges.
2. Stabilizing pre-feedback parametrization:
   u(k+i|k) = -K * x_hat(k+i|k) + q(i)
   Closed-loop state matrix: A_cl = (A - B*K) with all eigenvalues inside unit disk.
   State prediction becomes:
   x_hat(k+i|k) = A_cl^i * x_hat(k|k) + sum_{j=0}^{i-1} A_cl^{i-1-j} * B * q(j)
   Since A_cl^i -> 0, predictions remain bounded and numerically well-conditioned.
3. Linear relationship between q(i) and Delta u(k+i|k) -> Preserves standard convex QP.
4. Numerical conditioning benchmark on an unstable plant:
   Plots condition number kappa vs prediction horizon Hp.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_unstable_plant():
    """
    Continuous-to-discrete unstable plant (e.g. linearized inverted pendulum or unstable drone):
    Continuous: p(s) = 2 / (s^2 - 1.5 s - 1)
    Discrete with Ts = 0.1 s:
    Poles outside unit disk.
    """
    Ts = 0.1
    # State-space realization:
    # dot_x1 = x2
    # dot_x2 = 1.0 * x1 + 1.5 * x2 + 2.0 * u
    Ac = np.array([[0.0, 1.0],
                   [1.0, 1.5]])
    Bc = np.array([[0.0],
                   [2.0]])
    Cc = np.array([[1.0, 0.0]])

    # Discretize using matrix exponential
    n = Ac.shape[0]
    M = la.expm(np.block([[Ac, Bc], [np.zeros((1, n + 1))]]) * Ts)
    A = M[:n, :n]
    B = M[:n, n:]
    C = Cc
    return A, B, C, Ts

def design_stabilizing_gain(A, B):
    """
    Designs a discrete LQR gain K such that eig(A - B*K) are strictly within unit circle.
    """
    Q_lqr = np.eye(A.shape[0]) * 5.0
    R_lqr = np.eye(B.shape[1]) * 1.0
    P = la.solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = la.inv(R_lqr + B.T @ P @ B) @ (B.T @ P @ A)
    return K

def build_open_loop_prediction_matrices(A, B, C, Hp, Hu):
    """
    Standard open-loop prediction matrices Psi_ol, Theta_ol.
    """
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]

    Psi_ol = np.zeros((Hp * p, n))
    for i in range(1, Hp + 1):
        Psi_ol[(i-1)*p : i*p, :] = C @ np.linalg.matrix_power(A, i)

    Theta_ol = np.zeros((Hp * p, Hu * m))
    for j in range(1, Hu + 1):
        for i in range(j, Hp + 1):
            Theta_ol[(i-1)*p : i*p, (j-1)*m : j*m] = C @ np.linalg.matrix_power(A, i - j) @ B

    return Psi_ol, Theta_ol

def build_stabilized_prediction_matrices(A, B, C, K, Hp, Hu):
    """
    Stabilized prediction matrices Psi_stab, Theta_stab using A_cl = A - B*K.
    State evolution under u(k+i|k) = -K*x(k+i|k) + q(i):
    x(k+i|k) = A_cl^i * x(k|k) + sum_{j=0}^{i-1} A_cl^{i-1-j} * B * q(j)
    y(k+i|k) = C * x(k+i|k)
    """
    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]
    A_cl = A - B @ K

    Psi_stab = np.zeros((Hp * p, n))
    for i in range(1, Hp + 1):
        Psi_stab[(i-1)*p : i*p, :] = C @ np.linalg.matrix_power(A_cl, i)

    Theta_stab = np.zeros((Hp * p, Hu * m))
    for j in range(1, Hu + 1):
        for i in range(j, Hp + 1):
            Theta_stab[(i-1)*p : i*p, (j-1)*m : j*m] = C @ np.linalg.matrix_power(A_cl, i - j) @ B

    return Psi_stab, Theta_stab

def condition_number_benchmark():
    """
    Compares condition number of Theta_ol vs Theta_stab as Hp increases from 5 to 35.
    """
    A, B, C, Ts = get_unstable_plant()
    K = design_stabilizing_gain(A, B)

    eigs_open = np.linalg.eigvals(A)
    eigs_cl = np.linalg.eigvals(A - B @ K)
    print(f"Open-loop eigenvalues: {eigs_open} (Unstable: max |lambda| = {np.max(np.abs(eigs_open)):.3f} > 1)")
    print(f"Closed-loop eigenvalues: {eigs_cl} (Stable: max |lambda| = {np.max(np.abs(eigs_cl)):.3f} < 1)")

    horizons = np.arange(5, 35, 2)
    cond_ol = []
    cond_stab = []

    for hp in horizons:
        hu = min(hp, 5)
        _, Theta_ol = build_open_loop_prediction_matrices(A, B, C, hp, hu)
        _, Theta_stab = build_stabilized_prediction_matrices(A, B, C, K, hp, hu)

        cond_ol.append(np.linalg.cond(Theta_ol))
        cond_stab.append(np.linalg.cond(Theta_stab))

    return horizons, cond_ol, cond_stab

def simulate_closed_loop():
    """
    Simulates closed-loop step response with input constraints: -2.0 <= u <= 2.0
    Comparing open-loop vs stabilized formulation.
    """
    A, B, C, Ts = get_unstable_plant()
    K = design_stabilizing_gain(A, B)

    Hp = 20
    Hu = 5
    N_sim = 60
    t = np.arange(N_sim) * Ts
    r = 1.0  # Setpoint step

    # Precompute prediction matrices
    Psi_ol, Theta_ol = build_open_loop_prediction_matrices(A, B, C, Hp, Hu)
    Psi_stab, Theta_stab = build_stabilized_prediction_matrices(A, B, C, K, Hp, Hu)

    # Simulation with Stabilized MPC
    x_stab_sim = np.zeros((N_sim, 2))
    u_stab_sim = np.zeros(N_sim)
    y_stab_sim = np.zeros(N_sim)
    x_curr = np.zeros((2, 1))

    Q_bar = np.eye(Hp) * 1.0
    R_bar = np.eye(Hu) * 0.1

    H_q = Theta_stab.T @ Q_bar @ Theta_stab + R_bar

    u_limit = 2.0

    for k in range(N_sim - 1):
        y_stab_sim[k] = (C @ x_curr)[0, 0]
        x_stab_sim[k, :] = x_curr.flatten()

        # Free response: y_free = Psi_stab * x_curr
        T_ref = np.ones((Hp, 1)) * r
        E = T_ref - Psi_stab @ x_curr

        g_q = Theta_stab.T @ Q_bar @ E

        # Constraints on input:
        # u(k+i|k) = -K * x_hat(k+i|k) + q(i)
        # For step 0: u(k) = -K * x(k) + q(0)
        # -u_lim <= -K * x(k) + q(0) <= u_lim
        def obj(q):
            return 0.5 * q @ H_q @ q - g_q.flatten() @ q
        def jac(q):
            return H_q @ q - g_q.flatten()

        q0 = np.zeros(Hu)
        # Bound on q(0) to satisfy |u(k)| <= u_limit
        K_x = float((K @ x_curr)[0, 0])
        lb = -u_limit + K_x
        ub =  u_limit + K_x
        bounds = [(lb, ub)] + [(None, None)] * (Hu - 1)

        res = minimize(obj, q0, jac=jac, bounds=bounds, method='SLSQP')
        q_opt = res.x
        u_opt = -K_x + q_opt[0]
        u_opt = np.clip(u_opt, -u_limit, u_limit)

        u_stab_sim[k] = u_opt
        x_curr = A @ x_curr + B * u_opt

    y_stab_sim[-1] = (C @ x_curr)[0, 0]
    x_stab_sim[-1, :] = x_curr.flatten()
    u_stab_sim[-1] = u_stab_sim[-2]

    return t, y_stab_sim, u_stab_sim

def main():
    print("=" * 70)
    print("Chapter 5: Stabilized Predictions for Unstable Plants (Section 5.2)")
    print("=" * 70)

    horizons, cond_ol, cond_stab = condition_number_benchmark()

    print("\nHorizon Hp vs Condition Number kappa(Theta):")
    print(f"{'Hp':>4} | {'Open-Loop kappa':>18} | {'Stabilized kappa':>18} | {'Ratio':>12}")
    print("-" * 60)
    for hp, col, cst in zip(horizons, cond_ol, cond_stab):
        print(f"{hp:4d} | {col:18.2e} | {cst:18.2e} | {col/cst:12.1e}")

    t, y_sim, u_sim = simulate_closed_loop()

    # Plot figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Condition Number vs Horizon (Log scale)
    axes[0, 0].semilogy(horizons, cond_ol, 'ro-', lw=2, label='Open-Loop Predictions (A^i blows up)')
    axes[0, 0].semilogy(horizons, cond_stab, 'bs-', lw=2, label='Stabilized Predictions (A_cl^i -> 0)')
    axes[0, 0].set_xlabel('Prediction Horizon Hp')
    axes[0, 0].set_ylabel('Condition Number kappa(Theta)')
    axes[0, 0].set_title('(a) Numerical Conditioning: Condition Number vs Horizon')
    axes[0, 0].grid(True, which='both', linestyle=':')
    axes[0, 0].legend()

    # (b) Condition Number Ratio
    axes[0, 1].plot(horizons, np.array(cond_ol) / np.array(cond_stab), 'g^-', lw=2)
    axes[0, 1].set_xlabel('Prediction Horizon Hp')
    axes[0, 1].set_ylabel('Condition Number Ratio kappa_ol / kappa_stab')
    axes[0, 1].set_title('(b) Ill-Conditioning Growth Factor')
    axes[0, 1].grid(True)

    # (c) Closed Loop Tracking on Unstable Plant
    axes[1, 0].plot(t, y_sim, 'b-', lw=2, label='Plant Output y(t)')
    axes[1, 0].axhline(1.0, color='r', linestyle='--', label='Setpoint r = 1.0')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Output y')
    axes[1, 0].set_title('(c) Stabilized MPC Closed-Loop Tracking (Unstable Plant)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # (d) Control Input with Hard Constraints
    axes[1, 1].plot(t, u_sim, 'k-', lw=1.8, label='Applied Input u(t)')
    axes[1, 1].axhline(2.0, color='r', linestyle=':', label='u_max = 2.0')
    axes[1, 1].axhline(-2.0, color='r', linestyle=':', label='u_min = -2.0')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Control Input u')
    axes[1, 1].set_title('(d) Constrained Input Trajectory')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    output_png = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_stabilized_predictions.png'
    plt.savefig(output_png, dpi=300)
    print(f"\n[Plot Saved] -> {output_png}")

if __name__ == '__main__':
    main()
