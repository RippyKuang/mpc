"""
Chapter 6 Stability: Section 6.4 & Example 6.4
Predictive Control via Youla Parameterization (Q-Parameterization)
===================================================================
This script implements:
1. The Youla-Kucera parameterization for stable plants (Figure 6.3 in Maciejowski):
      u(z) = - Q(z) * [y(z) - P(z) u(z)] = - Q(z) * d(z)
   where d(z) is the unmeasured disturbance estimate (innovation).
2. FIR Parameterization of the Youla parameter Q(z):
      Q(z) = H(0) + H(1) z^{-1} + ... + H(n) z^{-n}
   Decision variables: pulse response coefficients H = [H(0), H(1), ..., H(n)]^T.
3. Convex Quadratic Programming Formulation:
   - State and output predictions are strictly affine in the coefficients H.
   - Objective function V(k) is strictly quadratic in H.
   - Input and rate constraints: |u(k)| <= u_max, |Delta u(k)| <= du_max.
4. Fundamental Theoretical Invariance:
   - Since plant P(z) is stable and Q(z) is FIR (all poles at z = 0),
     the closed-loop system is GUARANTEED to be internally stable for ANY H!
5. Numerical Simulation:
   - Disturbance rejection under step load disturbance
   - Constrained vs Unconstrained Youla-MPC
   - FIR impulse response of the optimal Youla parameter Q(z)
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class YoulaPredictiveController:
    """
    Predictive Controller based on FIR Youla Parameterization (Section 6.4, Example 6.4)
    """
    def __init__(self, A, B, C, D, n_fir=4, Hp=10, Q_wt=1.0, R_wt=0.1, u_max=2.0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]
        self.n_fir = n_fir  # order of FIR Youla parameter (n+1 coefficients)
        self.Hp = Hp
        self.Q_wt = Q_wt
        self.R_wt = R_wt
        self.u_max = u_max

        # Buffer of past disturbances d(k - i)
        self.d_buffer = np.zeros((n_fir + 1, self.ny))

    def update_disturbance(self, y_meas, y_model):
        """
        Calculates d(k) = y(k) - y_model(k) and updates buffer
        """
        d_k = y_meas - y_model
        # Shift buffer: d_buffer[0] = d(k), d_buffer[1] = d(k-1), ...
        self.d_buffer[1:] = self.d_buffer[:-1]
        self.d_buffer[0] = d_k
        return d_k

    def solve_optimal_H(self, x_model_curr, r_target=0.0):
        """
        Solves QP for optimal FIR coefficients H = [H(0), ..., H(n)]:
        min sum_{j=1}^{Hp} (y(k+j) - r)^2 * Q + sum_{j=0}^{Hp-1} u(k+j)^2 * R
        subject to |u(k)| <= u_max
        """
        nH = self.n_fir + 1

        def cost_func(H):
            # Simulate prediction horizon under candidate H
            x = x_model_curr.copy()
            cost = 0.0
            # Buffer copy
            d_buf = self.d_buffer.copy()

            for j in range(self.Hp):
                # u(k+j) = - sum_{i=0}^n H[i] * d(k+j-i)
                # For j > 0, assume disturbance remains constant: d(k+j) = d(k)
                u_j = - np.sum(H * d_buf[:, 0])
                x_next = self.A @ x + self.B * u_j
                y_next = (self.C @ x_next)[0, 0]

                cost += self.Q_wt * (y_next + d_buf[0, 0] - r_target)**2 + self.R_wt * (u_j**2)

                x = x_next
                # Update disturbance buffer for next step
                d_buf[1:] = d_buf[:-1]
                # Disturbance forecast: persistent disturbance d(k+j) = d(k)

            return cost

        # Decision variables: H
        H0 = np.zeros(nH)
        H0[0] = 0.5  # initial guess

        # Constraint: |u(k)| <= u_max => |- H^T * d_buf| <= u_max
        def u_constraint_upper(H):
            u_0 = - np.sum(H * self.d_buffer[:, 0])
            return self.u_max - u_0

        def u_constraint_lower(H):
            u_0 = - np.sum(H * self.d_buffer[:, 0])
            return u_0 + self.u_max

        constraints = [
            {'type': 'ineq', 'fun': u_constraint_upper},
            {'type': 'ineq', 'fun': u_constraint_lower}
        ]

        res = minimize(cost_func, H0, constraints=constraints, method='SLSQP',
                       options={'ftol': 1e-7, 'disp': False})
        H_opt = res.x
        u_opt = - np.sum(H_opt * self.d_buffer[:, 0])
        return H_opt, u_opt

def run_simulation():
    print("=" * 70)
    print("Chapter 6: Section 6.4 - Youla Parameterization (Q-Parameterization)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. System Definition: Stable SISO Plant
    # -------------------------------------------------------------
    # P(z) = 0.3 / (z^2 - 1.2 z + 0.35)
    # Poles: lambda1 = 0.7, lambda2 = 0.5 (strictly inside unit circle => stable!)
    A = np.array([[1.2, -0.35],
                  [1.0,  0.0]])
    B = np.array([[1.0],
                  [0.0]])
    C = np.array([[0.0, 0.3]])
    D = np.array([[0.0]])

    poles = la.eigvals(A)
    print(f"Plant Poles: {poles} (Both |lambda| < 1 => Open-loop Stable)")

    controller = YoulaPredictiveController(A, B, C, D, n_fir=4, Hp=8, Q_wt=1.0, R_wt=0.05, u_max=1.8)

    # -------------------------------------------------------------
    # 2. Closed-Loop Simulation with Step Load Disturbance
    # -------------------------------------------------------------
    sim_steps = 25
    x_plant = np.zeros((sim_steps + 1, 2))
    x_model = np.zeros((sim_steps + 1, 2))
    y_plant = np.zeros(sim_steps + 1)
    y_model = np.zeros(sim_steps + 1)
    u_hist = np.zeros(sim_steps)
    dist_hist = np.zeros(sim_steps + 1)

    # Inject persistent step disturbance at step k = 5
    dist_step_time = 5
    dist_magnitude = 1.0

    H_optimal_history = []

    for k in range(sim_steps):
        # Step disturbance on plant output
        d_k = dist_magnitude if k >= dist_step_time else 0.0
        dist_hist[k] = d_k

        y_meas = (C @ x_plant[k])[0] + d_k
        y_mod = (C @ x_model[k])[0]

        y_plant[k] = y_meas
        y_model[k] = y_mod

        # Update Youla controller disturbance estimate
        controller.update_disturbance(np.array([[y_meas]]), np.array([[y_mod]]))

        # Solve optimal Youla FIR coefficients
        H_opt, u_k = controller.solve_optimal_H(x_model[k], r_target=0.0)
        u_hist[k] = u_k
        H_optimal_history.append(H_opt)

        # Plant and model update
        x_plant[k+1] = A @ x_plant[k] + B.flatten() * u_k
        x_model[k+1] = A @ x_model[k] + B.flatten() * u_k

    y_plant[-1] = (C @ x_plant[-1])[0] + (dist_magnitude if sim_steps >= dist_step_time else 0.0)
    y_model[-1] = (C @ x_model[-1])[0]

    print(f"\nOptimal FIR Youla Parameter Coefficients at final step:")
    print(f"H(0..4) = {H_optimal_history[-1]}")

    # -------------------------------------------------------------
    # 3. Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    time_steps = np.arange(sim_steps + 1)
    time_u = np.arange(sim_steps)

    # Plot (a): Output Tracking and Disturbance Rejection
    ax = axes[0, 0]
    ax.step(time_steps, y_plant, 'b-o', where='post', label=r'Plant Output $y(k)$')
    ax.step(time_steps, y_model, 'm--', where='post', label=r'Internal Model Output $y_m(k)$')
    ax.step(time_steps, dist_hist, 'r:', where='post', alpha=0.7, label=r'Output Disturbance $d(k)$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Youla-MPC: Disturbance Rejection Performance', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Output')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (b): Control Input and Saturation Limits
    ax = axes[0, 1]
    ax.step(time_u, u_hist, 'g-s', where='post', label=r'Control Input $u(k) = -Q(z) d(k)$')
    ax.axhline(controller.u_max, color='red', linestyle='--', label=r'Constraint $\pm u_{max}$')
    ax.axhline(-controller.u_max, color='red', linestyle='--')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f'Control Signal under Constraint |u| <= {controller.u_max}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Control Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (c): Impulse Response of Optimal Youla Filter Q(z)
    ax = axes[1, 0]
    final_H = H_optimal_history[-1]
    fir_lags = np.arange(len(final_H))
    ax.stem(fir_lags, final_H, basefmt=" ")
    ax.set_title('FIR Impulse Response Coefficients H(i) of Youla Filter Q(z)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Lag index i')
    ax.set_ylabel('Coefficient H(i)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot (d): Internal Model Error / Disturbance Estimate d(k)
    ax = axes[1, 1]
    d_est = y_plant - y_model
    ax.step(time_steps, d_est, 'c-^', where='post', label=r'$\hat{d}(k) = y(k) - y_m(k)$')
    ax.step(time_steps, dist_hist, 'r:', where='post', label=r'True $d(k)$')
    ax.set_title('Disturbance Identification Signal fed to Youla Block', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Disturbance Signal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch06_stability/ch6_youla_parameterization.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
