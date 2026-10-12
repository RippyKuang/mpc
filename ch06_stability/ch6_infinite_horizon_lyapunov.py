"""
Chapter 6 Stability: Section 6.2.1 & Exercise 6.5
Infinite Horizon Predictive Control via Discrete Lyapunov Equation Terminal Weight
====================================================================================
This script implements:
1. Derivation and numerical solution of the discrete Lyapunov equation:
      A^T * Q_bar * A - Q_bar + C^T * Q * C = 0
2. Exact equivalence between an infinite prediction horizon (H_p = infinity) with
   input horizon H_u, and a finite-horizon MPC with terminal weight Q_bar.
3. Reproduction of Exercise 6.5:
      x(k+1) = 0.9 x(k) + 0.5 u(k)
      Q = 1, R = 0, H_u = 2, H_p = infinity
      Lyapunov equation: 0.9^2 * Q_bar - Q_bar + 1 = 0 => Q_bar = 100/19 ~= 5.263158
      Optimal deadbeat feedback gain: u(k) = -1.8 x(k)
4. Comparison of:
   - Infinite horizon MPC (terminal weight Q_bar) -> Guaranteed asymptotic stability
   - Truncated finite horizon MPC (Q_bar = 0 or Q_N = 1) -> Suboptimal / less stable
   - Constrained infinite horizon MPC (|u| <= u_max) via QP (scipy.optimize)
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def solve_discrete_lyapunov(A, Q_state):
    """
    Solves A^T * Q_bar * A - Q_bar + Q_state = 0
    using scipy.linalg.solve_discrete_lyapunov.
    Note: scipy solves A * X * A^T - X + Q = 0, so we pass A^T.
    """
    return la.solve_discrete_lyapunov(A.T, Q_state)

def solve_qp_mpc(x_current, a, b, Q_term, R_weight, u_max, Hu=2):
    """
    Solves 2-step MPC QP with box constraint |u_i| <= u_max:
    min_{u_0, u_1} x_1^2 + Q_term * x_2^2 + R * (u_0^2 + u_1^2)
    subject to:
      x_1 = a * x_current + b * u_0
      x_2 = a * x_1 + b * u_1
      -u_max <= u_0, u_1 <= u_max
    """
    def objective(u):
        x1 = a * x_current + b * u[0]
        x2 = a * x1 + b * u[1]
        return x1**2 + Q_term * (x2**2) + R_weight * (u[0]**2 + u[1]**2)

    def grad_obj(u):
        x1 = a * x_current + b * u[0]
        x2 = a * x1 + b * u[1]
        # dx1/du0 = b, dx2/du0 = a*b, dx2/du1 = b
        g0 = 2 * x1 * b + 2 * Q_term * x2 * (a * b) + 2 * R_weight * u[0]
        g1 = 2 * Q_term * x2 * b + 2 * R_weight * u[1]
        return np.array([g0, g1])

    bounds = [(-u_max, u_max), (-u_max, u_max)]
    u0_init = np.zeros(Hu)
    res = minimize(objective, u0_init, jac=grad_obj, bounds=bounds, method='SLSQP',
                   options={'ftol': 1e-9, 'disp': False})
    return res.x

def run_simulation():
    print("=" * 70)
    print("Chapter 6: Infinite Horizon Predictive Control via Lyapunov Terminal Weight")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. System Definition (Exercise 6.5)
    # -------------------------------------------------------------
    # x(k+1) = a * x(k) + b * u(k)
    a = 0.9
    b = 0.5
    A = np.array([[a]])
    B = np.array([[b]])
    C = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = 0.0  # control weight = 0 as in Exercise 6.5
    Hu = 2

    print(f"Plant: x(k+1) = {a} x(k) + {b} u(k)")
    print(f"Horizon: Hu = {Hu}, Hp = infinity, Q = {Q[0,0]}, R = {R}")

    # -------------------------------------------------------------
    # 2. Solve Discrete Lyapunov Equation for Autonomous Tail
    # -------------------------------------------------------------
    # After Hu = 2, u(k+i) = 0, so x(k+i+1) = A * x(k+i)
    # Tail cost: sum_{i=Hu}^{inf} x(k+i+1)^T C^T Q C x(k+i+1) = x(k+Hu)^T Q_bar x(k+Hu)
    # A^T Q_bar A - Q_bar + C^T Q C = 0
    # For scalar: a^2 * Q_bar - Q_bar + 1 = 0 => (1 - 0.81) * Q_bar = 1 => 0.19 * Q_bar = 1
    Q_bar_exact = 1.0 / (1.0 - a**2)
    Q_bar_scipy = solve_discrete_lyapunov(A, C.T @ Q @ C)

    print(f"Analytical Q_bar = 1 / (1 - {a}^2) = 100/19 = {Q_bar_exact:.6f}")
    print(f"Numerical  Q_bar = {Q_bar_scipy[0,0]:.6f}")
    assert np.isclose(Q_bar_exact, Q_bar_scipy[0,0])

    # -------------------------------------------------------------
    # 3. Equivalent Finite Horizon Cost Function
    # -------------------------------------------------------------
    # V(k) = x(k+1)^2 + x(k+2)^2 + ... (infinite)
    #      = x(k+1)^2 + Q_bar * x(k+2)^2
    # Predictions:
    # x(k+1) = 0.9 x(k) + 0.5 u(k)
    # x(k+2) = 0.9 x(k+1) + 0.5 u(k+1) = 0.81 x(k) + 0.45 u(k) + 0.5 u(k+1)
    #
    # Unconstrained minimization:
    # dV / du(k+1) = 2 * Q_bar * x(k+2) * (0.5) = 0 => x(k+2) = 0
    # => 0.5 u(k+1) = -0.9 x(k+1) => deadbeat at step k+2!
    # Substituting x(k+2) = 0 into V gives V = x(k+1)^2.
    # To minimize x(k+1)^2: x(k+1) = 0.9 x(k) + 0.5 u(k) = 0 => u(k) = -1.8 x(k).
    # Then closed-loop pole is at 0 (deadbeat in 1 step)!
    K_deadbeat = -a / b
    print(f"Optimal deadbeat feedback gain: u(k) = {K_deadbeat:.4f} x(k)")

    # -------------------------------------------------------------
    # 4. Simulation Comparison: Unconstrained vs Constrained
    # -------------------------------------------------------------
    sim_steps = 15
    x0 = 5.0
    u_max = 3.0  # Input constraint: |u| <= u_max
    R_comp = 0.5 # Add control penalty for realistic regularized comparison

    # Solve discrete algebraic Riccati equation (DARE) for infinite horizon with R
    P_inf = la.solve_discrete_are(A, B, C.T @ Q @ C, np.array([[R_comp]]))
    K_lqr = -la.inv(R_comp + B.T @ P_inf @ B) @ (B.T @ P_inf @ A)
    print(f"\nWith R = {R_comp}:")
    print(f"Infinite-horizon DARE Riccati P = {P_inf[0,0]:.4f}, Gain K_lqr = {K_lqr[0,0]:.4f}")

    # Case A: Infinite horizon unconstrained (Deadbeat u = -1.8 x)
    x_inf_uncon = np.zeros(sim_steps + 1)
    u_inf_uncon = np.zeros(sim_steps)
    x_inf_uncon[0] = x0
    for k in range(sim_steps):
        u_inf_uncon[k] = K_deadbeat * x_inf_uncon[k]
        x_inf_uncon[k+1] = a * x_inf_uncon[k] + b * u_inf_uncon[k]

    # Case C: Constrained MPC with Lyapunov terminal weight Q_bar (Hu = 2, Hp = inf)
    x_mpc_lyap = np.zeros(sim_steps + 1)
    u_mpc_lyap = np.zeros(sim_steps)
    x_mpc_lyap[0] = x0

    # Case D: Constrained MPC without terminal weight (Hp = 2, terminal weight = Q = 1)
    x_mpc_noterm = np.zeros(sim_steps + 1)
    u_mpc_noterm = np.zeros(sim_steps)
    x_mpc_noterm[0] = x0

    for k in range(sim_steps):
        # Solve with Lyapunov terminal weight Q_bar
        u_opt_lyap = solve_qp_mpc(x_mpc_lyap[k], a, b, Q_bar_exact, R_comp, u_max, Hu=2)
        u_mpc_lyap[k] = u_opt_lyap[0]
        x_mpc_lyap[k+1] = a * x_mpc_lyap[k] + b * u_mpc_lyap[k]

        # Solve without Lyapunov terminal weight (terminal weight Q = 1)
        u_opt_no = solve_qp_mpc(x_mpc_noterm[k], a, b, 1.0, R_comp, u_max, Hu=2)
        u_mpc_noterm[k] = u_opt_no[0]
        x_mpc_noterm[k+1] = a * x_mpc_noterm[k] + b * u_mpc_noterm[k]

    # -------------------------------------------------------------
    # 5. Plotting and Visualization
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    time_steps = np.arange(sim_steps + 1)
    time_u = np.arange(sim_steps)

    # Plot (a): Exercise 6.5 Unconstrained Deadbeat Response
    ax = axes[0, 0]
    ax.step(time_steps, x_inf_uncon, 'b-o', where='post', label=r'State $x(k)$')
    ax.step(time_u, u_inf_uncon, 'r--s', where='post', label=r'Input $u(k) = -1.8 x(k)$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Exercise 6.5: Unconstrained Deadbeat ($Q_{bar} = 100/19$, R=0)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (b): Constrained Response Comparison (State)
    ax = axes[0, 1]
    ax.step(time_steps, x_mpc_lyap, 'g-o', where='post', linewidth=2, label=r'With Lyapunov $Q_{bar} = 5.26$')
    ax.step(time_steps, x_mpc_noterm, 'm--^', where='post', linewidth=1.5, label=r'Without Lyapunov ($Q_N = 1$)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f'Constrained State Response (|u| <= {u_max}, R={R_comp})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('State x(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (c): Constrained Control Action Comparison
    ax = axes[1, 0]
    ax.step(time_u, u_mpc_lyap, 'g-o', where='post', linewidth=2, label=r'$u(k)$ with $Q_{bar}$')
    ax.step(time_u, u_mpc_noterm, 'm--^', where='post', linewidth=1.5, label=r'$u(k)$ without $Q_{bar}$')
    ax.axhline(u_max, color='red', linestyle='--', label=r'Constraint $\pm u_{max}$')
    ax.axhline(-u_max, color='red', linestyle='--')
    ax.set_title('Control Action under Input Saturation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Control Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (d): Cumulative Stage Cost Evolution
    ax = axes[1, 1]
    cum_cost_lyap = np.cumsum(x_mpc_lyap[:-1]**2 + R_comp * u_mpc_lyap**2)
    cum_cost_noterm = np.cumsum(x_mpc_noterm[:-1]**2 + R_comp * u_mpc_noterm**2)
    ax.plot(time_u, cum_cost_lyap, 'g-o', linewidth=2, label=r'Cumulative Cost (with $Q_{bar}$)')
    ax.plot(time_u, cum_cost_noterm, 'm--^', linewidth=1.5, label=r'Cumulative Cost (without $Q_{bar}$)')
    ax.set_title('Cumulative Closed-Loop Cost Performance', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel(r'$\sum (x^2 + R u^2)$')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch06_stability/ch6_infinite_horizon_lyapunov.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
