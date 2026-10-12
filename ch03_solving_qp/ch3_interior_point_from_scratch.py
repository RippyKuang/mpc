"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 3: Solving Predictive Control Problems
Filename: ch3_interior_point_from_scratch.py

Description:
Pure, standalone implementation of the Primal-Dual Interior-Point Quadratic
Programming (QP) solver from first principles using only basic NumPy linear algebra.
Solves:
    min  0.5 * theta^T * H * theta + g^T * theta
    s.t. A_in * theta <= b_in

Features:
- Logarithmic barrier parameter mu reduction along the central path
- Perturbed KKT system Newton-Raphson step computation
- Fraction-to-the-boundary rule for strictly positive slack and dual variables
- Duality gap tracking and convergence logging
- Replicates Example 3.1 and Figure 3.6 (Maciejowski, p. 96) showing the
  analytic center, central path trajectory, and convergence to (1.5, 0.5).

Dependencies: numpy, matplotlib
===============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PrimalDualInteriorPointQPSolver:
    """
    Primal-Dual Interior Point QP Solver from First Principles.
    """
    def __init__(self, H, g, A_in, b_in, tol=1e-7, max_iter=60):
        self.H = np.array(H, dtype=float)
        self.g = np.array(g, dtype=float).flatten()
        self.A_in = np.array(A_in, dtype=float)
        self.b_in = np.array(b_in, dtype=float).flatten()
        self.tol = tol
        self.max_iter = max_iter
        self.n = len(self.g)
        self.m = len(self.b_in)

    def solve(self, theta0=None, s0=None, lam0=None):
        """
        Runs the primal-dual interior point algorithm.
        """
        # Initialize strictly feasible primal and dual slack variables
        if theta0 is None:
            theta = np.zeros(self.n)
        else:
            theta = np.array(theta0, dtype=float).flatten()

        # Initialize slacks s > 0
        if s0 is None:
            res = self.b_in - self.A_in @ theta
            s = np.where(res > 0.1, res, 1.0)
        else:
            s = np.array(s0, dtype=float).flatten()

        # Initialize dual multipliers lambda > 0
        if lam0 is None:
            lam = np.ones(self.m)
        else:
            lam = np.array(lam0, dtype=float).flatten()

        history = [theta.copy()]
        duality_gaps = []

        print(f"[InteriorPoint] Starting at theta0 = {theta}")
        print(f"[InteriorPoint] Initial slacks s = {s}, multipliers lambda = {lam}")

        for it in range(1, self.max_iter + 1):
            # Compute duality gap: eta = s^T * lambda / m
            gap = np.dot(s, lam) / self.m
            duality_gaps.append(gap)

            # Compute residuals
            r_dual = self.H @ theta + self.g + self.A_in.T @ lam
            r_primal = self.A_in @ theta + s - self.b_in

            res_norm = np.linalg.norm(r_dual) + np.linalg.norm(r_primal)
            if gap < self.tol and res_norm < self.tol:
                print(f"[InteriorPoint] Converged in {it} iterations! Gap = {gap:.2e}, Residuals = {res_norm:.2e}")
                return theta, history, duality_gaps, 0.5 * theta @ self.H @ theta + self.g @ theta, lam

            # Centering parameter sigma and barrier target mu
            sigma = 0.2
            mu = sigma * gap

            # Perturbed complementarity residual: r_cent = S * lam - mu * 1
            r_cent = s * lam - mu

            # Solve Newton system via block elimination:
            # S * Delta_lam + Lam * Delta_s = -r_cent
            # ==> Delta_s = -r_primal - A_in * Delta_theta
            # ==> Delta_lam = S^(-1) * (-r_cent - Lam * Delta_s)
            #               = S^(-1) * (-r_cent + Lam * r_primal + Lam * A_in * Delta_theta)
            # Plug Delta_lam into: H * Delta_theta + A_in^T * Delta_lam = -r_dual
            # (H + A_in^T * S^(-1) * Lam * A_in) Delta_theta = -r_dual - A_in^T * S^(-1) * (-r_cent + Lam * r_primal)
            diag_ratio = lam / s
            H_mod = self.H + self.A_in.T @ np.diag(diag_ratio) @ self.A_in
            rhs_theta = -r_dual - self.A_in.T @ (( -r_cent + lam * r_primal ) / s)

            d_theta = np.linalg.solve(H_mod, rhs_theta)
            d_s = -r_primal - self.A_in @ d_theta
            d_lam = (-r_cent - lam * d_s) / s

            # Fraction-to-the-boundary step lengths to ensure s > 0, lam > 0
            alpha_p = 1.0
            for i in range(self.m):
                if d_s[i] < 0:
                    alpha_p = min(alpha_p, -0.995 * s[i] / d_s[i])

            alpha_d = 1.0
            for i in range(self.m):
                if d_lam[i] < 0:
                    alpha_d = min(alpha_d, -0.995 * lam[i] / d_lam[i])

            # Update iterates
            theta = theta + alpha_p * d_theta
            s = s + alpha_p * d_s
            lam = lam + alpha_d * d_lam
            history.append(theta.copy())

            if it % 5 == 0 or it == 1:
                print(f"[InteriorPoint] Iter {it:2d}: Gap = {gap:.4e}, alpha_p = {alpha_p:.3f}, theta = {theta}")

        print("[InteriorPoint] Reached max iterations.")
        return theta, history, duality_gaps, 0.5 * theta @ self.H @ theta + self.g @ theta, lam


def demonstrate_example_3_1_interior_point(save_dir="."):
    """
    Replicates Example 3.1 & Figure 3.6 (Maciejowski, Section 3.3.2, p. 95-96):
        min V(theta) = theta_1^2 - theta_1 * theta_2 + theta_2^2 - 3 * theta_1
        s.t.
            -theta_1 <= 0
            -theta_2 <= 0
             theta_1 + theta_2 <= 2
            -theta_1 + theta_2 <= 1
    """
    print("\n===================================================================")
    print("Executing Example 3.1 via Custom Primal-Dual Interior-Point Solver...")
    print("===================================================================")

    H = np.array([
        [2.0, -1.0],
        [-1.0, 2.0]
    ])
    g = np.array([-3.0, 0.0])

    A_in = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0]
    ])
    b_in = np.array([0.0, 0.0, 2.0, 1.0])

    # Start near analytic center (approx (2/3, 2/3) = (0.667, 0.667))
    solver = PrimalDualInteriorPointQPSolver(H, g, A_in, b_in)
    theta_opt, history, gaps, cost_opt, lams = solver.solve(theta0=[0.67, 0.67])

    print(f"\nFinal Result:")
    print(f"  Optimal theta* = {theta_opt}")
    print(f"  Optimal Cost   = {cost_opt:.5f} (Theoretical: -11/4 = -2.75000)")
    print(f"  Lagrange Multipliers = {lams}")

    assert np.allclose(theta_opt, [1.5, 0.5], atol=1e-4), "Failed to match theoretical optimum (1.5, 0.5)!"

    # Trace central path points for different fixed mu values to replicate Fig 3.6
    central_path_points = []
    mu_values = np.logspace(-3, 2, 25)
    for mu_target in mu_values:
        # Solve centering problem with fixed mu
        t_temp = np.array([0.67, 0.67])
        s_temp = b_in - A_in @ t_temp
        l_temp = np.ones(len(b_in))
        for _ in range(15):
            r_d = H @ t_temp + g + A_in.T @ l_temp
            r_p = A_in @ t_temp + s_temp - b_in
            r_c = s_temp * l_temp - mu_target
            H_mod = H + A_in.T @ np.diag(l_temp / s_temp) @ A_in
            rhs_t = -r_d - A_in.T @ ((-r_c + l_temp * r_p) / s_temp)
            dt = np.linalg.solve(H_mod, rhs_t)
            ds = -r_p - A_in @ dt
            dl = (-r_c - l_temp * ds) / s_temp
            a_p = min(1.0, 0.95 * np.min([-s_temp[i]/ds[i] for i in range(len(ds)) if ds[i]<0] or [1.0]))
            a_d = min(1.0, 0.95 * np.min([-l_temp[i]/dl[i] for i in range(len(dl)) if dl[i]<0] or [1.0]))
            t_temp += a_p * dt
            s_temp += a_p * ds
            l_temp += a_d * dl
        central_path_points.append(t_temp.copy())

    central_path_points = np.array(central_path_points)
    history = np.array(history)

    # Visualization replicating Figure 3.6
    t1 = np.linspace(-0.5, 2.5, 300)
    t2 = np.linspace(-0.5, 2.5, 300)
    T1, T2 = np.meshgrid(t1, t2)
    V = T1**2 - T1 * T2 + T2**2 - 3.0 * T1

    plt.figure(figsize=(9, 8))
    contours = plt.contour(T1, T2, V, levels=np.linspace(-3.5, 5, 25), cmap='viridis', alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)

    polygon_x = [0.0, 2.0, 0.5, 0.0, 0.0]
    polygon_y = [0.0, 0.0, 1.5, 1.0, 0.0]
    plt.fill(polygon_x, polygon_y, color='lightgreen', alpha=0.3, label='Feasible Region')
    plt.plot(polygon_x, polygon_y, 'g-', lw=2.5, label='Constraint Boundaries')

    # Plot Central Path (dots) as in Figure 3.6
    plt.plot(central_path_points[:, 0], central_path_points[:, 1], 'm.', markersize=8, label='Central Path (gamma: 0.01 -> 100)')
    # Plot Analytic center
    plt.plot(2/3, 2/3, 'ko', markersize=10, markerfacecolor='none', markeredgewidth=2, label='Analytic Center (2/3, 2/3)')

    # Plot Primal-Dual algorithm Newton steps
    plt.plot(history[:, 0], history[:, 1], 'b.-', lw=1.8, markersize=7, label='Primal-Dual Newton Steps')

    # Highlight theoretical optimum (1.5, 0.5)
    plt.plot(1.5, 0.5, 'r*', markersize=16, label='Theoretical Optimum (3/2, 1/2)')

    plt.title("Figure 3.6 Replication: Primal-Dual Interior-Point Method & Central Path", fontsize=13)
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.xlim(-0.3, 2.4)
    plt.ylim(-0.3, 2.2)
    plt.grid(True)
    plt.legend(loc='upper right')

    p_out = os.path.join(save_dir, "ch3_interior_point_central_path.png")
    plt.savefig(p_out, dpi=200)
    plt.close()
    print(f"-> Saved Interior Point Central Path plot to: {p_out}")

if __name__ == "__main__":
    demonstrate_example_3_1_interior_point(save_dir="/home/kuang/Desktop/mpc/ch03_solving_qp")
