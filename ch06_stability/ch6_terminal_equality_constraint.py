"""
Chapter 6: Stability
Script 2: Terminal Equality Constraints & Keerthi-Gilbert Lyapunov Proof (Theorem 6.1)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 6.1, Theorem 6.1

Key Concepts Implemented:
1. Terminal Equality Constraint Formulation:
   min sum_{i=1}^{Hp} [ x(k+i)^T * Q * x(k+i) + u(k+i-1)^T * R * u(k+i-1) ]
   s.t.  u_min <= u(k+i-1) <= u_max
         x(k+Hp|k) = 0   (Terminal equality constraint)
2. Lyapunov Monotonicity Verification (Keerthi-Gilbert 1988):
   Optimal candidate at step k+1 constructed by shifting previous optimal tail and appending u = 0:
   V*(k+1) - V*(k) <= - [ x(k)^T * Q * x(k) + u*(k)^T * R * u*(k) ] <= 0
3. Feasible Region Shrinkage (The Price of Terminal Equality):
   Computes the controllable sets X_reach(Hp) for Hp = 2, 3, 5 under actuator saturation |u| <= 1.0.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_double_integrator():
    """
    Discrete-time double integrator:
    x1(k+1) = x1(k) + Ts * x2(k)
    x2(k+1) = x2(k) + Ts * u(k)
    Ts = 0.5 s
    """
    Ts = 0.5
    A = np.array([[1.0, Ts],
                  [0.0, 1.0]])
    B = np.array([[0.5 * Ts**2],
                  [Ts]])
    Q = np.eye(2) * 2.0
    R = np.eye(1) * 0.5
    return A, B, Q, R, Ts

def solve_terminal_mpc(x_init, Hp, A, B, Q, R, u_max=1.0):
    """
    Solves QP with terminal constraint x(k+Hp|k) = 0.
    Decision variables: U = [u(0), u(1), ..., u(Hp-1)] (Hp vars).
    """
    n = A.shape[0]
    m = B.shape[1]

    # Prediction matrices
    # x(i) = A^i * x0 + sum_{j=0}^{i-1} A^(i-1-j) * B * u(j)
    def cost_fn(U_vec):
        cost = 0.0
        x = x_init.copy()
        for i in range(Hp):
            u = U_vec[i:i+1].reshape(-1, 1)
            x = A @ x + B @ u
            cost += float((x.T @ Q @ x)[0, 0] + (u.T @ R @ u)[0, 0])
        return cost

    # Terminal equality constraint: x(Hp) == 0 (2 equality constraints)
    def terminal_con(U_vec):
        x = x_init.copy()
        for i in range(Hp):
            u = U_vec[i:i+1].reshape(-1, 1)
            x = A @ x + B @ u
        return x.flatten() # == 0

    bounds = [(-u_max, u_max)] * Hp
    cons = [{'type': 'eq', 'fun': terminal_con}]

    U0 = np.zeros(Hp)
    res = minimize(cost_fn, U0, bounds=bounds, constraints=cons, method='SLSQP')
    if res.success:
        return res.x[0], res.fun, res.success
    else:
        return 0.0, np.nan, False

def verify_lyapunov_decay():
    """
    Simulates closed-loop trajectory and verifies:
    Delta V*(k) = V*(k+1) - V*(k) <= - (x^T Q x + u^T R u)
    """
    A, B, Q, R, Ts = get_double_integrator()
    Hp = 5
    u_max = 1.0
    N_sim = 15

    x_curr = np.array([[0.4], [0.1]])
    V_history = []
    stage_cost_history = []
    x_history = []
    u_history = []

    print("=" * 70)
    print("Theorem 6.1: Lyapunov Monotonicity Verification (Keerthi-Gilbert 1988)")
    print("=" * 70)
    print(f"{'Step k':>6} | {'V*(k)':>12} | {'V*(k+1)-V*(k)':>16} | {'-ell(x,u)':>12} | {'Lyapunov Decrease Valid?':>25}")
    print("-" * 75)

    for k in range(N_sim):
        x_history.append(x_curr.flatten())
        u_opt, V_opt, success = solve_terminal_mpc(x_curr, Hp, A, B, Q, R, u_max)
        if not success:
            print(f"Optimization finished or terminal state reached at step {k}!")
            break

        V_history.append(V_opt)
        x_next = A @ x_curr + B * u_opt
        stage_cost = float((x_next.T @ Q @ x_next)[0, 0] + u_opt**2 * R[0, 0])
        stage_cost_history.append(stage_cost)
        u_history.append(u_opt)

        if k > 0:
            diff_V = V_history[k] - V_history[k-1]
            bound = -stage_cost_history[k-1]
            is_valid = (diff_V <= bound + 1e-4)
            print(f"{k-1:6d} | {V_history[k-1]:12.4f} | {diff_V:16.4f} | {bound:12.4f} | {str(is_valid):>25}")

        # Update state
        x_curr = x_next
        if la.norm(x_curr) < 1e-4:
            break

    return np.array(x_history), np.array(u_history), np.array(V_history), np.array(stage_cost_history)

def compute_reachability_set(Hp_list=[2, 3, 5]):
    """
    Computes controllable sets X_reach(Hp) analytically via linprog:
    C_Hp * U = -A^Hp * x0 with U in [-1, 1]^Hp.
    """
    from scipy.optimize import linprog
    A, B, Q, R, Ts = get_double_integrator()
    grid_x1 = np.linspace(-1.5, 1.5, 40)
    grid_x2 = np.linspace(-1.5, 1.5, 40)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)

    feas_maps = {}
    for hp in Hp_list:
        # Build controllability matrix C_hp: [A^(hp-1)*B, ..., B] (2 x hp)
        C_hp = np.zeros((2, hp))
        for col in range(hp):
            p = hp - 1 - col
            C_hp[:, col] = (np.linalg.matrix_power(A, p) @ B).flatten()
        A_hp = np.linalg.matrix_power(A, hp)

        c_dummy = np.zeros(hp)
        bounds = [(-1.0, 1.0)] * hp

        mask = np.zeros(X1.shape, dtype=bool)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                x0 = np.array([[X1[i, j]], [X2[i, j]]])
                b_eq = (-A_hp @ x0).flatten()
                res = linprog(c_dummy, A_eq=C_hp, b_eq=b_eq, bounds=bounds, method='highs')
                mask[i, j] = res.success
        feas_maps[hp] = mask
    return X1, X2, feas_maps

def main():
    x_hist, u_hist, V_hist, ell_hist = verify_lyapunov_decay()
    X1, X2, feas_maps = compute_reachability_set()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Lyapunov Function Monotone Decay
    t_v = np.arange(len(V_hist))
    axes[0, 0].plot(t_v, V_hist, 'ro-', lw=2, label='Optimal Cost V*(k)')
    axes[0, 0].set_xlabel('Time Step k', fontsize=11)
    axes[0, 0].set_ylabel('Lyapunov Function Value V*(k)', fontsize=11)
    axes[0, 0].set_title('(a) Strict Monotonic Decay of Lyapunov Function V*(k)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':')
    axes[0, 0].legend()

    # (b) Difference V*(k+1) - V*(k) vs Stage Cost Bound -ell(x,u)
    diff_V = np.diff(V_hist)
    bound_ell = -ell_hist[:-1]
    t_diff = np.arange(len(diff_V))
    axes[0, 1].plot(t_diff, diff_V, 'b-s', lw=2, label='Actual Decay: V*(k+1) - V*(k)')
    axes[0, 1].plot(t_diff, bound_ell, 'r--', lw=2, label='Theoretical Upper Bound: -ell(x,u)')
    axes[0, 1].set_xlabel('Time Step k', fontsize=11)
    axes[0, 1].set_ylabel('Cost Difference', fontsize=11)
    axes[0, 1].set_title('(b) Keerthi-Gilbert Inequality: V*(k+1) - V*(k) <= -ell(x,u)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].legend()

    # (c) Closed-Loop State Trajectories
    t_x = np.arange(len(x_hist))
    axes[1, 0].plot(t_x, x_hist[:, 0], 'b-o', lw=2, label='Position x1(k)')
    axes[1, 0].plot(t_x, x_hist[:, 1], 'g-s', lw=2, label='Velocity x2(k)')
    axes[1, 0].set_xlabel('Time Step k', fontsize=11)
    axes[1, 0].set_ylabel('State Values', fontsize=11)
    axes[1, 0].set_title('(c) Asymptotic Convergence to Origin under |u| <= 1.0', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':')
    axes[1, 0].legend()

    # (d) Feasible Basin of Attraction Shrinkage (The Price of Terminal Constraint)
    colors = {2: 'lightcoral', 3: 'khaki', 5: 'lightgreen'}
    for hp in [5, 3, 2]:
        axes[1, 1].contourf(X1, X2, feas_maps[hp].astype(int), levels=[0.5, 1.5], colors=[colors[hp]], alpha=0.6)
        axes[1, 1].contour(X1, X2, feas_maps[hp].astype(int), levels=[0.5], colors=['k'], linewidths=1.2)
    axes[1, 1].plot(x_hist[:, 0], x_hist[:, 1], 'b-o', lw=2, label='Simulation Trajectory')
    axes[1, 1].plot(0, 0, 'r*', markersize=14, label='Origin (0,0)')
    axes[1, 1].set_xlabel('Initial State x1', fontsize=11)
    axes[1, 1].set_ylabel('Initial State x2', fontsize=11)
    axes[1, 1].set_title('(d) Controllable Basin X_reach(Hp): Green(Hp=5) > Yellow(3) > Red(2)', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, linestyle=':')
    axes[1, 1].legend()

    plt.tight_layout()
    output_png = '/home/kuang/Desktop/mpc/ch06_stability/ch6_terminal_constraint_lyapunov.png'
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n[Plot Saved] -> {output_png}")

if __name__ == '__main__':
    main()
