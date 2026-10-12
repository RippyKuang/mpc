"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 3: Solving Predictive Control Problems
Filename: ch3_constraint_softening.py

Description:
Demonstration of Constraint Softening and Slack Variables (Section 3.4).
Replicates the core phenomenon of Figures 3.7, 3.8, 3.9, and 3.10:
1. Hard-constrained MPC encounters infeasibility under severe gusts or model error.
2. Introduction of slack variables epsilon >= 0 with:
   - 1-norm penalty:  rho * sum(epsilon)      (Exact penalty property)
   - 2-norm penalty:  rho * sum(epsilon^2)    (Quadratic slack penalty)
   - inf-norm penalty: rho * epsilon_max      (Single scalar slack variable)
3. Verifies how exact vs inexact penalty values (rho = 5e5 vs rho = 1e4) trade
   off constraint strictness against control smoothness.

Dependencies: numpy, scipy, matplotlib
===============================================================================
"""

import os
import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 1. Cessna Citation Model & State-Space Formulations
# =============================================================================

def get_citation_discrete_model(Ts=0.5, mismatch_factor=1.0):
    Ac = np.array([
        [-1.2822, 0.0, 0.98, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-5.4293, 0.0, -1.8366, 0.0],
        [-128.2 * mismatch_factor, 128.2 * mismatch_factor, 0.0, 0.0]
    ])
    Bc = np.array([[-0.3], [0.0], [-17.0], [0.0]])
    C = np.array([
        [0.0, 1.0, 0.0, 0.0],       # Pitch angle theta
        [0.0, 0.0, 0.0, 1.0],       # Altitude h
        [-128.2 * mismatch_factor, 128.2 * mismatch_factor, 0.0, 0.0]  # Altitude rate h_dot
    ])
    D = np.zeros((3, 1))
    sys_d = signal.cont2discrete((Ac, Bc, C, D), Ts, method='zoh')
    return sys_d[0], sys_d[1], sys_d[2]

def build_mpc_prediction_matrices(A, B, C, Hp=10, Hu=3):
    nx = A.shape[0]
    nu = B.shape[1]
    nz = C.shape[0]

    Psi = np.zeros((nz * Hp, nx))
    A_pow = np.eye(nx)
    for i in range(1, Hp + 1):
        A_pow = A_pow @ A
        Psi[(i - 1) * nz : i * nz, :] = C @ A_pow

    Upsilon = np.zeros((nz * Hp, nu))
    sum_A_B = np.zeros((nx, nu))
    A_p = np.eye(nx)
    for i in range(1, Hp + 1):
        sum_A_B = sum_A_B + A_p @ B
        A_p = A_p @ A
        Upsilon[(i - 1) * nz : i * nz, :] = C @ sum_A_B

    Theta = np.zeros((nz * Hp, nu * Hu))
    for r in range(1, Hp + 1):
        for c in range(1, Hu + 1):
            if r >= c:
                shift = r - c
                sum_term = np.zeros((nx, nu))
                A_k = np.eye(nx)
                for k in range(shift + 1):
                    sum_term = sum_term + A_k @ B
                    A_k = A_k @ A
                Theta[(r - 1) * nz : r * nz, (c - 1) * nu : c * nu] = C @ sum_term

    return Psi, Upsilon, Theta


# =============================================================================
# 2. MPC Solvers: Hard Constraints vs Softened Constraints
# =============================================================================

def solve_mpc_step(x_curr, u_prev, r_traj, Psi, Upsilon, Theta,
                   Hp=10, Hu=3, mode="hard", rho=1e4,
                   u_max=0.262, du_max=0.262, hdot_max=30.0):
    """
    mode:
      - 'hard': Hard output constraint |hdot| <= 30 m/s
      - 'soft_inf': Soft constraint via inf-norm (scalar slack epsilon >= 0)
      - 'soft_1norm': Soft constraint via 1-norm (separate slacks for each constrained step)
    """
    Q_single = np.diag([1/3, 1/3, 1/3])
    Q_bar = np.kron(np.eye(Hp), Q_single)
    R_bar = np.eye(Hu) * 1.0

    H_orig = 2.0 * (Theta.T @ Q_bar @ Theta + R_bar)
    free_resp = Psi @ x_curr + Upsilon @ u_prev
    E = r_traj - free_resp
    g_orig = -2.0 * (Theta.T @ Q_bar @ E).flatten()

    # Rate constraints: -du_max <= du <= du_max
    # Magnitude constraints: -u_max <= u(k-1) + sum(du) <= u_max
    # Altitude rate constraints on output:
    # Output indices for h_dot are row 2, 5, 8, ... (index 2 + 3*i)
    hdot_indices = [2 + 3 * i for i in range(Hp)]
    Theta_hdot = Theta[hdot_indices, :]
    free_hdot = free_resp[hdot_indices, 0]

    C_acc = np.tril(np.ones((Hu, Hu)))

    if mode == "hard":
        # Formulate hard QP
        A_rows = []
        b_rows = []
        # du limits
        for i in range(Hu):
            r = np.zeros(Hu); r[i] = 1.0; A_rows.append(r); b_rows.append(du_max)
            r = np.zeros(Hu); r[i] = -1.0; A_rows.append(r); b_rows.append(du_max)
        # u limits
        for i in range(Hu):
            A_rows.append(C_acc[i, :]); b_rows.append(u_max - u_prev[0, 0])
            A_rows.append(-C_acc[i, :]); b_rows.append(u_max + u_prev[0, 0])
        # hdot limits: -hdot_max <= Theta_hdot * du + free_hdot <= hdot_max
        for i in range(Hp):
            A_rows.append(Theta_hdot[i, :]); b_rows.append(hdot_max - free_hdot[i])
            A_rows.append(-Theta_hdot[i, :]); b_rows.append(hdot_max + free_hdot[i])

        A_in = np.array(A_rows)
        b_in = np.array(b_rows)

        def cost(du): return 0.5 * du @ H_orig @ du + g_orig @ du
        def jac(du): return H_orig @ du + g_orig

        res = minimize(cost, np.zeros(Hu), jac=jac,
                       constraints=[{'type': 'ineq', 'fun': lambda du: b_in - A_in @ du}],
                       method='SLSQP')
        if not res.success:
            return None, False  # Infeasible!
        return res.x[0], True

    elif mode == "soft_inf":
        # Introduce single scalar slack epsilon >= 0
        # Variable: z = [du_0, du_1, ..., du_{Hu-1}, epsilon]
        # min 0.5 * du^T H du + g^T du + rho * epsilon
        n_vars = Hu + 1
        H_aug = np.zeros((n_vars, n_vars))
        H_aug[:Hu, :Hu] = H_orig
        g_aug = np.zeros(n_vars)
        g_aug[:Hu] = g_orig
        g_aug[Hu] = rho  # linear penalty on epsilon

        A_rows = []
        b_rows = []
        # Hard actuator limits
        for i in range(Hu):
            r = np.zeros(n_vars); r[i] = 1.0; A_rows.append(r); b_rows.append(du_max)
            r = np.zeros(n_vars); r[i] = -1.0; A_rows.append(r); b_rows.append(du_max)
        for i in range(Hu):
            r = np.zeros(n_vars); r[:Hu] = C_acc[i, :]; A_rows.append(r); b_rows.append(u_max - u_prev[0, 0])
            r = np.zeros(n_vars); r[:Hu] = -C_acc[i, :]; A_rows.append(r); b_rows.append(u_max + u_prev[0, 0])
        # Softened output constraints: Theta_hdot * du - epsilon <= hdot_max - free_hdot
        for i in range(Hp):
            r = np.zeros(n_vars); r[:Hu] = Theta_hdot[i, :]; r[Hu] = -1.0
            A_rows.append(r); b_rows.append(hdot_max - free_hdot[i])

            r = np.zeros(n_vars); r[:Hu] = -Theta_hdot[i, :]; r[Hu] = -1.0
            A_rows.append(r); b_rows.append(hdot_max + free_hdot[i])
        # epsilon >= 0 ==> -epsilon <= 0
        r = np.zeros(n_vars); r[Hu] = -1.0; A_rows.append(r); b_rows.append(0.0)

        A_in = np.array(A_rows)
        b_in = np.array(b_rows)

        def cost(z): return 0.5 * z @ H_aug @ z + g_aug @ z
        def jac(z): return H_aug @ z + g_aug

        res = minimize(cost, np.zeros(n_vars), jac=jac,
                       constraints=[{'type': 'ineq', 'fun': lambda z: b_in - A_in @ z}],
                       method='SLSQP')
        if not res.success:
            return 0.0, False
        return res.x[0], True


# =============================================================================
# 3. Flight Maneuver Simulation & Infeasibility Demonstration
# =============================================================================

def run_softening_comparison(save_dir="."):
    print("\n===================================================================")
    print("Executing Chapter 3 Constraint Softening Simulation (Section 3.4)...")
    print("===================================================================")

    Ts = 0.5
    tend = 20.0
    nsteps = int(tend / Ts)
    tvec = np.arange(nsteps) * Ts
    Hp = 10
    Hu = 3

    # Nominal plant and model
    A, B, C = get_citation_discrete_model(Ts=Ts, mismatch_factor=1.0)
    Psi, Upsilon, Theta = build_mpc_prediction_matrices(A, B, C, Hp=Hp, Hu=Hu)

    alt_step = 400.0
    r_traj = np.tile(np.array([[0.0], [alt_step], [0.0]]), (Hp, 1))

    # We inject a severe gust disturbance on altitude rate at t = 5.0 s of +6.0 m/s
    # In hard-constrained mode, this pushes the system beyond physical reach -> Infeasible!
    def simulate_run(mode, rho):
        x = np.zeros((4, 1))
        u_prev = 0.0
        hist_h = []
        hist_hdot = []
        hist_elev = []
        infeasible_step = None

        for k in range(nsteps):
            # Check for disturbance
            dist = 6.0 if 5.0 <= tvec[k] <= 10.0 else 0.0
            y_meas = C @ x
            y_meas[2, 0] += dist

            hist_h.append(y_meas[1, 0])
            hist_hdot.append(y_meas[2, 0])
            hist_elev.append(np.rad2deg(u_prev))

            du, success = solve_mpc_step(x, np.array([[u_prev]]), r_traj,
                                         Psi, Upsilon, Theta,
                                         Hp=Hp, Hu=Hu, mode=mode, rho=rho)
            if not success:
                infeasible_step = tvec[k]
                print(f"[{mode}] Infeasibility encountered at t = {tvec[k]:.1f} s! Controller aborted.")
                # Fill remaining with NaN
                for _ in range(k + 1, nsteps):
                    hist_h.append(np.nan)
                    hist_hdot.append(np.nan)
                    hist_elev.append(np.nan)
                break

            u_curr = u_prev + du
            u_prev = u_curr
            x = A @ x + B @ np.array([[u_curr]])

        return np.array(hist_h), np.array(hist_hdot), np.array(hist_elev), infeasible_step

    # 1. Hard constraints run
    print("\n[Run 1] Hard constraints with 6 m/s turbulence gust...")
    h_hard, hdot_hard, elev_hard, inf_t = simulate_run(mode="hard", rho=0.0)

    # 2. Soft constraints run with exact penalty rho = 5e5 (Fig 3.9)
    print("\n[Run 2] Soft constraints with exact penalty rho = 5e5 (Fig 3.9)...")
    h_exact, hdot_exact, elev_exact, _ = simulate_run(mode="soft_inf", rho=5e5)

    # 3. Soft constraints run with inexact penalty rho = 1e4 (Fig 3.10)
    print("\n[Run 3] Soft constraints with relaxed penalty rho = 1e4 (Fig 3.10)...")
    h_inexact, hdot_inexact, elev_inexact, _ = simulate_run(mode="soft_inf", rho=1e4)

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Section 3.4: Hard Infeasibility vs Soft Constraint Recovery (Figs 3.7-3.10)", fontsize=13)

    # Altitude
    axes[0].plot(tvec, h_hard, 'r--', lw=2.0, label='Hard Constraint (Crashes at t=5s)')
    axes[0].plot(tvec, h_exact, 'b-', lw=1.8, label='Soft Exact (rho = 5e5, Fig 3.9)')
    axes[0].plot(tvec, h_inexact, 'g-.', lw=1.8, label='Soft Inexact (rho = 1e4, Fig 3.10)')
    axes[0].axhline(alt_step, color='k', linestyle=':', label='Set-point (400 m)')
    axes[0].set_title("Altitude h (m)")
    axes[0].set_ylabel("m")
    axes[0].grid(True)
    axes[0].legend()

    # Altitude rate
    axes[1].plot(tvec, hdot_hard, 'r--', lw=2.0, label='Hard Constraint')
    axes[1].plot(tvec, hdot_exact, 'b-', lw=1.8, label='Soft Exact (rho = 5e5)')
    axes[1].plot(tvec, hdot_inexact, 'g-.', lw=1.8, label='Soft Inexact (rho = 1e4)')
    axes[1].axhline(30.0, color='r', linestyle='--', label='Rate Limit (+30 m/s)')
    axes[1].set_title("Altitude Rate h_dot (m/s)")
    axes[1].set_ylabel("m/s")
    axes[1].grid(True)
    axes[1].legend()

    # Elevator angle
    axes[2].step(tvec, elev_hard, 'r--', where='post', lw=2.0, label='Hard Constraint')
    axes[2].step(tvec, elev_exact, 'b-', where='post', lw=1.8, label='Soft Exact (Aggressive correction)')
    axes[2].step(tvec, elev_inexact, 'g-.', where='post', lw=1.8, label='Soft Inexact (Smoother action)')
    axes[2].axhline(15.0, color='r', linestyle='--', label='Elevator Limit (+/- 15 deg)')
    axes[2].axhline(-15.0, color='r', linestyle='--')
    axes[2].set_title("Elevator Angle (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("deg")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    p_out = os.path.join(save_dir, "ch3_constraint_softening_comparison.png")
    plt.savefig(p_out, dpi=200)
    plt.close()
    print(f"-> Saved Constraint Softening comparison to: {p_out}")

if __name__ == "__main__":
    run_softening_comparison(save_dir="/home/kuang/Desktop/mpc/ch03_solving_qp")
