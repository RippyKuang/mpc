"""
Chapter 5: Other Formulations of Predictive Control
Script 3: Non-Quadratic Costs, LP vs QP Geometry & 1-Norm/Inf-Norm Formulations
(Section 5.4, Figures 5.5, 5.6, and Exercise 5.2)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.4 & Exercise 5.2

Key Concepts Implemented:
1. Geometric Duality & Characteristic Differences (Figures 5.5 & 5.6):
   - LP: Linear contours, solution strictly on polytope vertices (corners).
   - QP: Ellipsoidal contours, solution can be in interior, on single constraint, or vertex.
   - Weight variation in QP moves operating point continuously along boundary (Figure 5.6).
2. Transformation of Absolute-Value (1-Norm) and Min-Max (Inf-Norm) Costs to Standard LP:
   - 1-Norm: Introduce slack vector y >= |C*theta - d| -> Linear objective 1^T y.
   - Inf-Norm: Introduce scalar slack gamma >= |C*theta - d| -> Linear objective gamma.
3. Closed-Loop Comparison: 2-Norm (QP) vs 1-Norm (LP) vs Inf-Norm (LP) MPC.
   - 1-Norm induces deadband and sparse control moves (Delta u = 0 for small deviations).
   - Inf-Norm minimizes peak worst-case deviation.
   - 2-Norm provides smooth, energy-optimal tracking.
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import linprog, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_geometry_fig5_5_fig5_6():
    """
    Recreates the exact geometric comparison of Figures 5.5 and 5.6.
    Feasible region defined by linear inequalities:
    A_ineq * x <= b_ineq
    """
    # 2D polytope
    # 1: x1 + 2*x2 <= 8
    # 2: 2*x1 + x2 <= 8
    # 3: -x1 <= -0.5  (x1 >= 0.5)
    # 4: -x2 <= -0.5  (x2 >= 0.5)
    A_ineq = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ])
    b_ineq = np.array([8.0, 8.0, -0.5, -0.5])

    # Grid for contour plotting
    x1_vals = np.linspace(0, 5, 200)
    x2_vals = np.linspace(0, 5, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Feasibility mask
    feasible = (X1 + 2*X2 <= 8.0) & (2*X1 + X2 <= 8.0) & (X1 >= 0.5) & (X2 >= 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Subplot 1: LP Geometry (Figure 5.5 Left) ---
    # Cost: min c1*x1 + c2*x2 (or max, cost increases upwards)
    c_lp = np.array([1.2, 1.8])
    # Solve LP
    res_lp = linprog(c_lp, A_ub=A_ineq, b_ub=b_ineq, bounds=[(0, None), (0, None)], method='highs')
    x_lp_opt = res_lp.x

    Z_lp = c_lp[0] * X1 + c_lp[1] * X2
    cs1 = axes[0].contour(X1, X2, Z_lp, levels=15, colors='steelblue', linestyles='--')
    axes[0].clabel(cs1, inline=True, fontsize=8)
    axes[0].imshow(feasible.astype(int), extent=(0, 5, 0, 5), origin='lower', cmap='Greys', alpha=0.15)
    # Draw boundary lines
    axes[0].plot([0.5, 0.5], [0.5, 3.75], 'k-', lw=2.5)
    axes[0].plot([0.5, 3.75], [0.5, 0.5], 'k-', lw=2.5)
    axes[0].plot([0.5, 2.667], [3.75, 2.667], 'k-', lw=2.5)
    axes[0].plot([2.667, 3.75], [2.667, 0.5], 'k-', lw=2.5)
    axes[0].plot(x_lp_opt[0], x_lp_opt[1], 'ko', markersize=10, label=f'LP Optimum: ({x_lp_opt[0]:.2f}, {x_lp_opt[1]:.2f})')
    axes[0].set_title('Figure 5.5 (Left): LP Contours & Vertex Optimum', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_xlim([0, 5])
    axes[0].set_ylim([0, 5])
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':')

    # --- Subplot 2: QP Geometry (Figure 5.5 Right / Figure 5.6 Left) ---
    # Target outside feasible region: x_des = [3.5, 3.5]
    # W = diag([1.0, 3.0]) -> x1 deviations cheaper than x2
    x_des = np.array([3.5, 3.5])
    W1 = np.diag([1.0, 3.0])
    def qp1_cost(x): return 0.5 * (x - x_des) @ W1 @ (x - x_des)
    res_qp1 = minimize(qp1_cost, [1.0, 1.0], constraints=[{'type': 'ineq', 'fun': lambda x: b_ineq - A_ineq @ x}], method='SLSQP')
    x_qp1_opt = res_qp1.x

    Z_qp1 = 0.5 * (W1[0, 0]*(X1 - x_des[0])**2 + W1[1, 1]*(X2 - x_des[1])**2)
    cs2 = axes[1].contour(X1, X2, Z_qp1, levels=12, colors='darkorange', linestyles='--')
    axes[1].clabel(cs2, inline=True, fontsize=8)
    axes[1].imshow(feasible.astype(int), extent=(0, 5, 0, 5), origin='lower', cmap='Greys', alpha=0.15)
    axes[1].plot([0.5, 0.5], [0.5, 3.75], 'k-', lw=2.5)
    axes[1].plot([0.5, 3.75], [0.5, 0.5], 'k-', lw=2.5)
    axes[1].plot([0.5, 2.667], [3.75, 2.667], 'k-', lw=2.5)
    axes[1].plot([2.667, 3.75], [2.667, 0.5], 'k-', lw=2.5)
    axes[1].plot(x_des[0], x_des[1], 'r*', markersize=12, label=f'Desired Target: ({x_des[0]}, {x_des[1]})')
    axes[1].plot(x_qp1_opt[0], x_qp1_opt[1], 'ko', markersize=10, label=f'QP Optimum: ({x_qp1_opt[0]:.2f}, {x_qp1_opt[1]:.2f})')
    axes[1].set_title('Figure 5.6 (Left): QP (x1 cheaper -> wide horizontal ellipse)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    axes[1].set_xlim([0, 5])
    axes[1].set_ylim([0, 5])
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle=':')

    # --- Subplot 3: QP Weight Shift (Figure 5.6 Right) ---
    # W = diag([3.0, 1.0]) -> x2 deviations cheaper than x1
    W2 = np.diag([3.0, 1.0])
    def qp2_cost(x): return 0.5 * (x - x_des) @ W2 @ (x - x_des)
    res_qp2 = minimize(qp2_cost, [1.0, 1.0], constraints=[{'type': 'ineq', 'fun': lambda x: b_ineq - A_ineq @ x}], method='SLSQP')
    x_qp2_opt = res_qp2.x

    Z_qp2 = 0.5 * (W2[0, 0]*(X1 - x_des[0])**2 + W2[1, 1]*(X2 - x_des[1])**2)
    cs3 = axes[2].contour(X1, X2, Z_qp2, levels=12, colors='purple', linestyles='--')
    axes[2].clabel(cs3, inline=True, fontsize=8)
    axes[2].imshow(feasible.astype(int), extent=(0, 5, 0, 5), origin='lower', cmap='Greys', alpha=0.15)
    axes[2].plot([0.5, 0.5], [0.5, 3.75], 'k-', lw=2.5)
    axes[2].plot([0.5, 3.75], [0.5, 0.5], 'k-', lw=2.5)
    axes[2].plot([0.5, 2.667], [3.75, 2.667], 'k-', lw=2.5)
    axes[2].plot([2.667, 3.75], [2.667, 0.5], 'k-', lw=2.5)
    axes[2].plot(x_des[0], x_des[1], 'r*', markersize=12, label=f'Desired Target: ({x_des[0]}, {x_des[1]})')
    axes[2].plot(x_qp2_opt[0], x_qp2_opt[1], 'ko', markersize=10, label=f'QP Optimum: ({x_qp2_opt[0]:.2f}, {x_qp2_opt[1]:.2f})')
    axes[2].set_title('Figure 5.6 (Right): QP (x2 cheaper -> tall vertical ellipse)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    axes[2].set_xlim([0, 5])
    axes[2].set_ylim([0, 5])
    axes[2].legend(loc='lower left')
    axes[2].grid(True, linestyle=':')

    plt.tight_layout()
    p_out = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_fig5_5_fig5_6_geometry.png'
    plt.savefig(p_out, dpi=300)
    plt.close()
    print(f"[Plot Saved] -> {p_out}")

def solve_1norm_mpc_lp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0, r_wt=0.1):
    """
    Solves 1-norm MPC (Eq 5.25 & Exercise 5.2) as an LP using SciPy linprog.
    Cost: min sum_i q_i |z(k+i|k) - r(k+i)| + sum_i r_i |Delta u(k+i)|
    Variables:
    theta = [Delta_U (Hu); y_err (Hp); y_du (Hu)]
    Dimension: Hu + Hp + Hu
    """
    Hp = Theta.shape[0]
    Hu = Theta.shape[1]

    # Objective: 0*Delta_U + q_wt*1^T*y_err + r_wt*1^T*y_du
    c_lp = np.concatenate([np.zeros(Hu), np.full(Hp, q_wt), np.full(Hu, r_wt)])

    # Constraints:
    # 1. Tracking error slack:
    #    y_err >= Theta * Delta_U - E  ==>  Theta * Delta_U - y_err <= E
    #    y_err >= -(Theta * Delta_U - E) ==> -Theta * Delta_U - y_err <= -E
    A1 = np.hstack([Theta, -np.eye(Hp), np.zeros((Hp, Hu))])
    b1 = E.flatten()

    A2 = np.hstack([-Theta, -np.eye(Hp), np.zeros((Hp, Hu))])
    b2 = -E.flatten()

    # 2. Control move slack:
    #    y_du >= Delta_U  ==>  Delta_U - y_du <= 0
    #    y_du >= -Delta_U ==> -Delta_U - y_du <= 0
    A3 = np.hstack([np.eye(Hu), np.zeros((Hu, Hp)), -np.eye(Hu)])
    b3 = np.zeros(Hu)

    A4 = np.hstack([-np.eye(Hu), np.zeros((Hu, Hp)), -np.eye(Hu)])
    b4 = np.zeros(Hu)

    # 3. Input rate limits: du_min <= Delta_U <= du_max
    # 4. Input amplitude limits: u_min <= u_prev + S_u * Delta_U <= u_max
    #    S_u * Delta_U <= u_max - u_prev
    #   -S_u * Delta_U <= -(u_min - u_prev)
    A5 = np.hstack([S_u, np.zeros((Hu, Hp)), np.zeros((Hu, Hu))])
    b5 = np.full(Hu, u_max - u_prev)

    A6 = np.hstack([-S_u, np.zeros((Hu, Hp)), np.zeros((Hu, Hu))])
    b6 = np.full(Hu, -(u_min - u_prev))

    A_ub = np.vstack([A1, A2, A3, A4, A5, A6])
    b_ub = np.concatenate([b1, b2, b3, b4, b5, b6])

    # Variable bounds:
    # Delta_U in [du_min, du_max], y_err >= 0, y_du >= 0
    bounds = [(du_min, du_max)] * Hu + [(0, None)] * Hp + [(0, None)] * Hu

    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x[0]
    else:
        return 0.0

def solve_infnorm_mpc_lp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0):
    """
    Solves Inf-norm MPC (Eq 5.33 - 5.36) as an LP using SciPy linprog.
    Cost: min gamma
    s.t. -gamma <= Theta * Delta_U - E <= gamma
    Variables: [Delta_U (Hu); gamma (1)]
    """
    Hp = Theta.shape[0]
    Hu = Theta.shape[1]

    c_lp = np.concatenate([np.zeros(Hu), [1.0]])

    # Constraints:
    # Theta * Delta_U - gamma <= E
    # -Theta * Delta_U - gamma <= -E
    ones_col = np.ones((Hp, 1))
    A1 = np.hstack([Theta, -ones_col])
    b1 = E.flatten()

    A2 = np.hstack([-Theta, -ones_col])
    b2 = -E.flatten()

    # Input constraints
    A3 = np.hstack([S_u, np.zeros((Hu, 1))])
    b3 = np.full(Hu, u_max - u_prev)

    A4 = np.hstack([-S_u, np.zeros((Hu, 1))])
    b4 = np.full(Hu, -(u_min - u_prev))

    A_ub = np.vstack([A1, A2, A3, A4])
    b_ub = np.concatenate([b1, b2, b3, b4])

    bounds = [(du_min, du_max)] * Hu + [(0, None)]

    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x[0]
    else:
        return 0.0

def solve_2norm_mpc_qp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0, r_wt=0.1):
    """
    Standard 2-Norm QP solver for comparison.
    """
    Hp = Theta.shape[0]
    Hu = Theta.shape[1]

    H = Theta.T @ (np.eye(Hp) * q_wt) @ Theta + np.eye(Hu) * r_wt
    g = Theta.T @ (np.eye(Hp) * q_wt) @ E

    def obj(du): return 0.5 * du @ H @ du - g.flatten() @ du
    def jac(du): return H @ du - g.flatten()

    cons = [
        {'type': 'ineq', 'fun': lambda du: np.full(Hu, u_max - u_prev) - S_u @ du},
        {'type': 'ineq', 'fun': lambda du: S_u @ du - np.full(Hu, u_min - u_prev)}
    ]
    bounds = [(du_min, du_max)] * Hu
    res = minimize(obj, np.zeros(Hu), jac=jac, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x[0]

def simulate_norm_comparisons():
    """
    Simulates a SISO system tracking a step reference using 1-Norm, Inf-Norm, and 2-Norm MPC.
    Plant: G(s) = 2 / (s + 1)^2 -> Discrete with Ts = 0.5 s.
    """
    Ts = 0.5
    # Continuous: G(s) = 2 / (s^2 + 2s + 1)
    A_c = np.array([[0.0, 1.0], [-1.0, -2.0]])
    B_c = np.array([[0.0], [2.0]])
    C_c = np.array([[1.0, 0.0]])

    n = 2
    M = la.expm(np.block([[A_c, B_c], [np.zeros((1, 3))]]) * Ts)
    A = M[:n, :n]
    B = M[:n, n:]
    C = C_c

    Hp = 12
    Hu = 4

    # Build prediction matrices
    Psi = np.zeros((Hp, n))
    for i in range(1, Hp + 1):
        Psi[i-1, :] = C @ np.linalg.matrix_power(A, i)

    Upsilon = np.zeros((Hp, 1))
    for i in range(1, Hp + 1):
        sum_val = sum(np.linalg.matrix_power(A, j) @ B for j in range(i))
        Upsilon[i-1, 0] = (C @ sum_val)[0, 0]

    Theta = np.zeros((Hp, Hu))
    for j in range(1, Hu + 1):
        for i in range(j, Hp + 1):
            sum_val = sum(np.linalg.matrix_power(A, k) @ B for k in range(i - j + 1))
            Theta[i-1, j-1] = (C @ sum_val)[0, 0]

    S_u = np.tril(np.ones((Hu, Hu)))

    u_min, u_max = -1.5, 1.5
    du_min, du_max = -0.6, 0.6

    N_sim = 40
    t = np.arange(N_sim) * Ts
    r = 1.0

    controllers = ['1-Norm (LP)', 'Inf-Norm (LP)', '2-Norm (QP)']
    results = {}

    for ctype in controllers:
        y_sim = np.zeros(N_sim)
        u_sim = np.zeros(N_sim)
        du_sim = np.zeros(N_sim)
        x_curr = np.zeros((2, 1))
        u_prev = 0.0

        for k in range(N_sim - 1):
            y_sim[k] = (C @ x_curr)[0, 0]
            T_ref = np.ones((Hp, 1)) * r
            E = T_ref - Psi @ x_curr - Upsilon * u_prev

            if ctype == '1-Norm (LP)':
                du_opt = solve_1norm_mpc_lp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0, r_wt=0.05)
            elif ctype == 'Inf-Norm (LP)':
                du_opt = solve_infnorm_mpc_lp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0)
            else:
                du_opt = solve_2norm_mpc_qp(Theta, E, S_u, u_prev, u_min, u_max, du_min, du_max, q_wt=1.0, r_wt=0.05)

            u_curr = np.clip(u_prev + du_opt, u_min, u_max)
            du_sim[k] = du_opt
            u_sim[k] = u_curr
            u_prev = u_curr

            x_curr = A @ x_curr + B * u_curr

        y_sim[-1] = (C @ x_curr)[0, 0]
        u_sim[-1] = u_sim[-2]
        du_sim[-1] = 0.0
        results[ctype] = {'y': y_sim, 'u': u_sim, 'du': du_sim}

    # Plot simulation comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    colors = {'1-Norm (LP)': 'darkorange', 'Inf-Norm (LP)': 'purple', '2-Norm (QP)': 'steelblue'}
    styles = {'1-Norm (LP)': '--', 'Inf-Norm (LP)': '-.', '2-Norm (QP)': '-'}

    for ctype in controllers:
        axes[0].plot(t, results[ctype]['y'], color=colors[ctype], linestyle=styles[ctype], lw=2.2, label=ctype)
        axes[1].step(t, results[ctype]['u'], color=colors[ctype], linestyle=styles[ctype], lw=1.8, where='post', label=ctype)
        axes[2].step(t, results[ctype]['du'], color=colors[ctype], linestyle=styles[ctype], lw=1.5, where='post', label=ctype)

    axes[0].axhline(r, color='k', linestyle=':', label='Target Setpoint r=1.0')
    axes[0].set_ylabel('Output y(t)', fontsize=11)
    axes[0].set_title('Closed-Loop Tracking: 1-Norm (LP) vs Inf-Norm (LP) vs 2-Norm (QP)', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':')
    axes[0].legend(loc='lower right')

    axes[1].axhline(u_max, color='r', linestyle='--', label='u_max = 1.5')
    axes[1].axhline(u_min, color='r', linestyle='--', label='u_min = -1.5')
    axes[1].set_ylabel('Control Input u(t)', fontsize=11)
    axes[1].set_title('Control Input Action', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':')
    axes[1].legend(loc='lower right')

    axes[2].axhline(du_max, color='gray', linestyle=':', label='du_max = 0.6')
    axes[2].axhline(du_min, color='gray', linestyle=':', label='du_min = -0.6')
    axes[2].set_xlabel('Time (seconds)', fontsize=11)
    axes[2].set_ylabel('Control Move Δu(t)', fontsize=11)
    axes[2].set_title('Control Moves: Note Deadband/Sparsity of 1-Norm vs Smooth 2-Norm', fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle=':')
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    p_sim = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_lp_vs_qp_mpc.png'
    plt.savefig(p_sim, dpi=300)
    plt.close()
    print(f"[Plot Saved] -> {p_sim}")

def main():
    print("=" * 70)
    print("Chapter 5: Non-Quadratic Costs, LP vs QP Geometry & Formulations")
    print("=" * 70)
    plot_geometry_fig5_5_fig5_6()
    simulate_norm_comparisons()
    print("Verification complete! LP and QP geometric and tracking plots generated.")

if __name__ == '__main__':
    main()
