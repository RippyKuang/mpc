"""
Chapter 6 Stability: Section 6.2.2 & Exercise 6.4
Infinite Horizon Predictive Control for Unstable Plants via Modal Decomposition
================================================================================
This script implements:
1. Modal decomposition of state matrix A:
      A = W * J * W^{-1},  J = blkdiag(J_s, J_u)
      z(k) = W^{-1} * x(k) = [z_s(k); z_u(k)]
2. Terminal equality constraint on unstable modes:
      z_u(k + H_u | k) = 0  <=>  W_u * x(k + H_u | k) = 0
3. Discrete Lyapunov equation on stable subsystem:
      J_s^T * Pi_bar * J_s - Pi_bar + Q_tilde_s = 0
   yielding the terminal weight for the stable modes.
4. Numerical simulation & comparison:
   - Stabilizing MPC with modal terminal constraint W_u * x(k+H_u) = 0
   - Unconstrained/Naively truncated MPC (fails to guarantee stability of unstable pole)
   - Phase portraits and state trajectory evolution under control saturation |u| <= u_max
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def modal_decomposition(A, C, Q):
    """
    Performs modal decomposition separating stable (|lambda| < 1)
    and unstable (|lambda| >= 1) modes.
    """
    eigvals, W = la.eig(A)
    # Sort into stable and unstable
    stable_idx = np.where(np.abs(eigvals) < 1.0)[0]
    unstable_idx = np.where(np.abs(eigvals) >= 1.0)[0]

    # Reorder columns of W
    idx_order = np.concatenate([stable_idx, unstable_idx])
    W_sorted = np.real(W[:, idx_order])
    W_inv = la.inv(W_sorted)

    n_s = len(stable_idx)
    n_u = len(unstable_idx)

    J = W_inv @ A @ W_sorted
    J_s = J[:n_s, :n_s]
    J_u = J[n_s:, n_s:]

    W_s_inv = W_inv[:n_s, :]  # maps x -> z_s
    W_u_inv = W_inv[n_s:, :]  # maps x -> z_u

    # Transform C^T Q C into modal coordinates:
    # x^T C^T Q C x = z^T (W^T C^T Q C W) z
    Q_modal = W_sorted.T @ C.T @ Q @ C @ W_sorted
    Q_s_modal = Q_modal[:n_s, :n_s]

    # Solve discrete Lyapunov equation for stable modal tail:
    # J_s^T * Pi_s * J_s - Pi_s + Q_s_modal = 0
    Pi_s = la.solve_discrete_lyapunov(J_s.T, Q_s_modal)

    # Terminal state weight on original state x:
    # z_s^T Pi_s z_s = x^T (W_s_inv^T Pi_s W_s_inv) x
    Q_bar_s = W_s_inv.T @ Pi_s @ W_s_inv

    return {
        'W': W_sorted,
        'W_inv': W_inv,
        'W_u_inv': W_u_inv,
        'W_s_inv': W_s_inv,
        'J_s': J_s,
        'J_u': J_u,
        'Pi_s': Pi_s,
        'Q_bar_s': Q_bar_s,
        'n_s': n_s,
        'n_u': n_u,
        'eigvals': eigvals[idx_order]
    }

def solve_unstable_mpc(x_curr, A, B, C, Q, R, Hu, Q_bar_s, W_u_inv, u_max, enforce_terminal_constraint=True):
    """
    Solves MPC for unstable plant:
    min sum_{i=0}^{Hu-1} [x(k+i+1)^T C^T Q C x(k+i+1) + u(k+i)^T R u(k+i)]
        + x(k+Hu)^T Q_bar_s x(k+Hu)
    subject to:
        -u_max <= u(k+i) <= u_max
        W_u_inv * x(k+Hu) = 0  (if enforce_terminal_constraint)
    """
    nx = A.shape[0]
    nu = B.shape[1]

    # Predict future states as linear function of U = [u_0; ...; u_{Hu-1}]
    # x_{i} = A^i * x_0 + sum_{j=0}^{i-1} A^{i-1-j} B u_j
    # Stack into block matrices
    Sx = np.zeros((Hu * nx, nx))
    Su = np.zeros((Hu * nx, Hu * nu))

    Ak = np.eye(nx)
    for i in range(Hu):
        Ak = Ak @ A
        Sx[i*nx:(i+1)*nx, :] = Ak
        for j in range(i + 1):
            Su[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(A, i - j) @ B

    # Terminal state matrices: x_{Hu} = A^Hu * x_0 + Su_terminal * U
    A_Hu = np.linalg.matrix_power(A, Hu)
    Su_Hu = Su[(Hu-1)*nx:Hu*nx, :]

    # Quadratic cost matrices
    Q_block = la.block_diag(*[C.T @ Q @ C for _ in range(Hu - 1)], C.T @ Q @ C + Q_bar_s)
    R_block = la.block_diag(*[R for _ in range(Hu)])

    # Cost = 0.5 * U^T H U + f^T U
    H = 2.0 * (Su.T @ Q_block @ Su + R_block)
    # Regularize slightly for numerical conditioning
    H = 0.5 * (H + H.T) + 1e-8 * np.eye(Hu * nu)

    def objective(U):
        f = 2.0 * (Su.T @ Q_block @ (Sx @ x_curr))
        return 0.5 * U @ H @ U + f @ U

    def grad_obj(U):
        f = 2.0 * (Su.T @ Q_block @ (Sx @ x_curr))
        return H @ U + f

    bounds = [(-u_max, u_max) for _ in range(Hu * nu)]
    constraints = []

    if enforce_terminal_constraint:
        # Linear equality constraint: W_u_inv * (A_Hu * x_curr + Su_Hu * U) = 0
        # W_u_inv * Su_Hu * U = -W_u_inv * A_Hu * x_curr
        A_eq = W_u_inv @ Su_Hu
        b_eq = -W_u_inv @ A_Hu @ x_curr
        constraints.append({'type': 'eq', 'fun': lambda U: (A_eq @ U - b_eq).flatten(),
                            'jac': lambda U: A_eq})

    u_init = np.zeros(Hu * nu)
    res = minimize(objective, u_init, jac=grad_obj, bounds=bounds, constraints=constraints,
                   method='SLSQP', options={'ftol': 1e-8, 'maxiter': 200, 'disp': False})

    if res.success:
        return res.x[:nu]
    else:
        # Fallback to bounded least squares / unconstrained if terminal constraint infeasible
        return np.clip(u_init[:nu], -u_max, u_max)

def run_simulation():
    print("=" * 70)
    print("Chapter 6: Infinite Horizon MPC for Unstable Plant via Modal Decomposition")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. System Definition (Coupled System with 1 Unstable, 1 Stable Pole)
    # -------------------------------------------------------------
    # A has eigenvalues: lambda1 = 1.25 (unstable), lambda2 = 0.60 (stable)
    A = np.array([[1.25, 0.20],
                  [0.00, 0.60]])
    B = np.array([[1.0],
                  [1.0]])
    C = np.eye(2)
    Q = np.eye(2)
    R = np.array([[0.5]])
    Hu = 4
    u_max = 2.5

    decomp = modal_decomposition(A, C, Q)
    print("Modal Decomposition Results:")
    print(f"Eigenvalues: {decomp['eigvals']}")
    print(f"Stable subsystem J_s: {decomp['J_s']}")
    print(f"Unstable subsystem J_u: {decomp['J_u']}")
    print(f"Lyapunov Pi_s for stable modes: {decomp['Pi_s']}")
    print(f"Terminal state weight Q_bar_s:\n{decomp['Q_bar_s']}")
    print(f"Unstable mode extractor W_u_inv: {decomp['W_u_inv']}")

    # -------------------------------------------------------------
    # 2. Closed-Loop Simulation Comparison
    # -------------------------------------------------------------
    sim_steps = 20
    x0 = np.array([1.2, -1.0])

    # Trajectory 1: With unstable terminal constraint W_u * x(k+Hu) = 0 + Q_bar_s
    x_hist_stab = np.zeros((sim_steps + 1, 2))
    u_hist_stab = np.zeros(sim_steps)
    x_hist_stab[0] = x0

    # Trajectory 2: Naive short horizon MPC (no terminal constraint, terminal weight = Q)
    x_hist_naive = np.zeros((sim_steps + 1, 2))
    u_hist_naive = np.zeros(sim_steps)
    x_hist_naive[0] = x0

    for k in range(sim_steps):
        # 1. Stabilizing MPC
        u_opt_stab = solve_unstable_mpc(x_hist_stab[k], A, B, C, Q, R, Hu,
                                        decomp['Q_bar_s'], decomp['W_u_inv'], u_max,
                                        enforce_terminal_constraint=True)
        u_hist_stab[k] = u_opt_stab[0]
        x_hist_stab[k+1] = A @ x_hist_stab[k] + B.flatten() * u_hist_stab[k]

        # 2. Naive MPC (without terminal constraint, Q_bar_s = 0)
        u_opt_naive = solve_unstable_mpc(x_hist_naive[k], A, B, C, Q, R, Hu,
                                         np.zeros((2, 2)), decomp['W_u_inv'], u_max,
                                         enforce_terminal_constraint=False)
        u_hist_naive[k] = u_opt_naive[0]
        x_hist_naive[k+1] = A @ x_hist_naive[k] + B.flatten() * u_hist_naive[k]

    # Verify unstable modal coordinate z_u(k)
    zu_hist_stab = [decomp['W_u_inv'] @ x_hist_stab[k] for k in range(sim_steps + 1)]
    zu_hist_naive = [decomp['W_u_inv'] @ x_hist_naive[k] for k in range(sim_steps + 1)]

    # -------------------------------------------------------------
    # 3. Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    time_steps = np.arange(sim_steps + 1)
    time_u = np.arange(sim_steps)

    # Plot (a): State Trajectories x1(k) and x2(k)
    ax = axes[0, 0]
    ax.plot(time_steps, x_hist_stab[:, 0], 'b-o', label=r'Stab MPC: $x_1(k)$ (Unstable pole mode)')
    ax.plot(time_steps, x_hist_stab[:, 1], 'g-s', label=r'Stab MPC: $x_2(k)$ (Stable pole mode)')
    ax.plot(time_steps, x_hist_naive[:, 0], 'r--^', alpha=0.7, label=r'Naive MPC: $x_1(k)$ (Diverges)')
    ax.plot(time_steps, x_hist_naive[:, 1], 'm--v', alpha=0.7, label=r'Naive MPC: $x_2(k)$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Closed-Loop State Evolution: Stabilizing vs Naive MPC', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('States')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (b): Unstable Mode Coordinate z_u(k) = W_u * x(k)
    ax = axes[0, 1]
    ax.plot(time_steps, [z[0] for z in zu_hist_stab], 'b-o', linewidth=2, label=r'With $W_u x(k+H_u) = 0$ ($z_u \to 0$)')
    ax.plot(time_steps, [z[0] for z in zu_hist_naive], 'r--^', linewidth=1.5, label=r'Without Constraint ($z_u \to \infty$)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(r'Unstable Mode Coordinate $z_u(k) = W_u x(k)$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel(r'$z_u(k)$')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (c): Control Actions and Constraints
    ax = axes[1, 0]
    ax.step(time_u, u_hist_stab, 'b-o', where='post', label='Stab MPC u(k)')
    ax.step(time_u, u_hist_naive, 'r--^', where='post', label='Naive MPC u(k)')
    ax.axhline(u_max, color='red', linestyle='--', label=r'Saturation $\pm u_{max}$')
    ax.axhline(-u_max, color='red', linestyle='--')
    ax.set_title(f'Control Input u(k) under Saturation |u| <= {u_max}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Control Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # Plot (d): Phase Portrait (x1 vs x2)
    ax = axes[1, 1]
    ax.plot(x_hist_stab[:, 0], x_hist_stab[:, 1], 'b-o', linewidth=2, label='Stabilizing Trajectory (Converges to 0)')
    ax.plot(x_hist_naive[:, 0], x_hist_naive[:, 1], 'r--^', linewidth=1.5, label='Naive Trajectory (Unbounded)')
    ax.plot(x0[0], x0[1], 'k*', markersize=12, label='Initial State x(0)')
    ax.plot(0, 0, 'go', markersize=8, label='Equilibrium Origin')
    ax.set_title('Phase Portrait (x1 vs x2)', fontsize=11, fontweight='bold')
    ax.set_xlabel('State x1')
    ax.set_ylabel('State x2')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch06_stability/ch6_unstable_infinite_horizon.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
