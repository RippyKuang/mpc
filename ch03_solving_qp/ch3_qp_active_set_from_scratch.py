"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 3: Solving Predictive Control Problems
Filename: ch3_qp_active_set_from_scratch.py

Description:
Pure, standalone implementation of the primal Active Set Quadratic Programming
(QP) solver from first principles using only basic NumPy linear algebra.
Solves:
    min  0.5 * theta^T * H * theta + g^T * theta
    s.t. A_in * theta <= b_in

Features:
- Solves equality-constrained subproblems via KKT system
- Ratio test to compute exact step size alpha avoiding constraint violations
- Lagrange multiplier sign checking for working set updates
- Step-by-step logging of working set additions/deletions
- Full visual trace on Example 3.1 (Fletcher problem / Maciejowski p. 95)

Dependencies: numpy, matplotlib
===============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ActiveSetQPSolver:
    """
    Primal Active Set Quadratic Programming Solver.
    """
    def __init__(self, H, g, A_in, b_in, max_iter=50, tol=1e-8):
        self.H = np.array(H, dtype=float)
        self.g = np.array(g, dtype=float).flatten()
        self.A_in = np.array(A_in, dtype=float)
        self.b_in = np.array(b_in, dtype=float).flatten()
        self.max_iter = max_iter
        self.tol = tol
        self.n = len(self.g)
        self.m = len(self.b_in)

    def solve(self, x0=None):
        """
        Solves the QP starting from a strictly feasible initial point x0.
        Returns: optimal_x, history, optimal_cost, multipliers
        """
        # If no initial point is given, find a feasible starting point
        if x0 is None:
            x = np.zeros(self.n)
            # Check feasibility
            if np.any(self.A_in @ x > self.b_in + self.tol):
                raise ValueError("Default x0=0 is not feasible. Please provide a feasible starting point.")
        else:
            x = np.array(x0, dtype=float).flatten()
            if np.any(self.A_in @ x > self.b_in + self.tol):
                raise ValueError("Provided x0 is not strictly feasible!")

        # Determine initial active set (working set W)
        residuals = self.b_in - self.A_in @ x
        W = [i for i in range(self.m) if abs(residuals[i]) < self.tol]

        history = [x.copy()]
        print(f"[ActiveSet] Starting at x0 = {x}, Initial Working Set W = {W}")

        for it in range(1, self.max_iter + 1):
            grad = self.H @ x + self.g

            # Build equality-constrained KKT system for step p:
            # [ H       A_W^T ] [ p      ] = [ -grad ]
            # [ A_W       0   ] [ lambda ]   [   0   ]
            if len(W) == 0:
                p = -np.linalg.solve(self.H, grad)
                lam_W = np.array([])
            else:
                A_W = self.A_in[W, :]
                n_W = len(W)
                KKT_mat = np.block([
                    [self.H, A_W.T],
                    [A_W, np.zeros((n_W, n_W))]
                ])
                rhs = np.concatenate([-grad, np.zeros(n_W)])
                sol = np.linalg.solve(KKT_mat, rhs)
                p = sol[:self.n]
                lam_W = sol[self.n:]

            p_norm = np.linalg.norm(p)

            if p_norm < self.tol:
                # p = 0: x is optimal on the current working set W!
                # Now check Lagrange multipliers to see if any constraint should be dropped.
                if len(W) == 0:
                    print(f"[ActiveSet] Step {it}: Unconstrained minimum reached at x = {x}")
                    return x, history, 0.5 * x @ self.H @ x + self.g @ x, lam_W

                min_lam_idx = np.argmin(lam_W)
                min_lam = lam_W[min_lam_idx]

                if min_lam >= -self.tol:
                    # All multipliers >= 0 -> KKT conditions satisfied!
                    print(f"[ActiveSet] Optimal solution found in {it} iterations! x* = {x}")
                    full_multipliers = np.zeros(self.m)
                    for idx, w_idx in enumerate(W):
                        full_multipliers[w_idx] = lam_W[idx]
                    return x, history, 0.5 * x @ self.H @ x + self.g @ x, full_multipliers
                else:
                    # Drop constraint with most negative multiplier
                    dropped_constraint = W[min_lam_idx]
                    del W[min_lam_idx]
                    print(f"[ActiveSet] Step {it}: Multiplier lambda_{dropped_constraint} = {min_lam:.4f} < 0. Dropping constraint {dropped_constraint}. New W = {W}")
            else:
                # Step p is nonzero: compute step length alpha via ratio test
                alpha = 1.0
                blocking_constraint = None

                for i in range(self.m):
                    if i not in W:
                        a_i_p = np.dot(self.A_in[i, :], p)
                        if a_i_p > self.tol:
                            dist = (self.b_in[i] - np.dot(self.A_in[i, :], x)) / a_i_p
                            if dist < alpha:
                                alpha = dist
                                blocking_constraint = i

                # Update state
                x = x + alpha * p
                history.append(x.copy())
                print(f"[ActiveSet] Step {it}: Moving alpha = {alpha:.4f}, p = {p}, new x = {x}")

                if alpha < 1.0 and blocking_constraint is not None:
                    W.append(blocking_constraint)
                    print(f"[ActiveSet] Step {it}: Hit boundary! Adding constraint {blocking_constraint} to W. New W = {W}")

        print("[ActiveSet] Warning: Maximum iterations reached without full convergence.")
        return x, history, 0.5 * x @ self.H @ x + self.g @ x, None


def demonstrate_example_3_1(save_dir="."):
    """
    Replicates Example 3.1 (Maciejowski, Section 3.3.2, p. 95):
        min V(theta) = theta_1^2 - theta_1 * theta_2 + theta_2^2 - 3 * theta_1
        s.t.
            theta_1 >= 0           ==>  -theta_1 <= 0
            theta_2 >= 0           ==>  -theta_2 <= 0
            theta_1 + theta_2 <= 2 ==>   theta_1 + theta_2 <= 2
    """
    print("\n===================================================================")
    print("Executing Example 3.1 via Custom Active Set QP Solver...")
    print("===================================================================")

    # Cost: V(theta) = 0.5 * theta^T * H * theta + g^T * theta
    # H = [[2, -1], [-1, 2]],  g = [-3, 0]
    H = np.array([
        [2.0, -1.0],
        [-1.0, 2.0]
    ])
    g = np.array([-3.0, 0.0])

    # Constraints:
    # 0: -theta_1 <= 0
    # 1: -theta_2 <= 0
    # 2:  theta_1 + theta_2 <= 2
    # 3: -theta_1 + theta_2 <= 1 (from text triangle boundary)
    A_in = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0]
    ])
    b_in = np.array([0.0, 0.0, 2.0, 1.0])

    # Start from interior point x0 = [0.5, 0.5]
    solver = ActiveSetQPSolver(H, g, A_in, b_in)
    x_opt, history, cost_opt, lams = solver.solve(x0=[0.5, 0.5])

    print(f"\nFinal Result:")
    print(f"  Optimal theta* = {x_opt}")
    print(f"  Optimal Cost   = {cost_opt:.5f} (Theoretical: -11/4 = -2.75000)")
    print(f"  Active Multipliers = {lams}")

    assert np.allclose(x_opt, [1.5, 0.5], atol=1e-4), "Failed to match theoretical optimum (1.5, 0.5)!"

    # Visualize feasible set, cost contours, and optimization path
    history = np.array(history)
    t1 = np.linspace(-0.5, 2.5, 300)
    t2 = np.linspace(-0.5, 2.5, 300)
    T1, T2 = np.meshgrid(t1, t2)
    V = T1**2 - T1 * T2 + T2**2 - 3.0 * T1

    plt.figure(figsize=(9, 8))
    # Plot contours
    contours = plt.contour(T1, T2, V, levels=np.linspace(-3.5, 5, 25), cmap='viridis', alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8)

    # Plot feasible region boundary
    # Triangle: (0,0) -> (2,0) -> (0.5, 1.5) -> (0, 1) -> (0,0)
    polygon_x = [0.0, 2.0, 0.5, 0.0, 0.0]
    polygon_y = [0.0, 0.0, 1.5, 1.0, 0.0]
    plt.fill(polygon_x, polygon_y, color='lightgreen', alpha=0.3, label='Feasible Region')
    plt.plot(polygon_x, polygon_y, 'g-', lw=2.5, label='Constraint Boundaries')

    # Plot Active Set Iteration steps
    plt.plot(history[:, 0], history[:, 1], 'ro-', lw=2.5, markersize=8, label='Active Set Iteration Steps')
    for idx, pt in enumerate(history):
        plt.text(pt[0] + 0.04, pt[1] + 0.04, f"Step {idx}", fontsize=11, fontweight='bold', color='darkred')

    # Highlight theoretical optimum (1.5, 0.5)
    plt.plot(1.5, 0.5, 'k*', markersize=16, label='Theoretical Optimum (3/2, 1/2)')
    # Highlight unconstrained minimum (2.0, 1.0)
    plt.plot(2.0, 1.0, 'bX', markersize=12, label='Unconstrained Minimum (2, 1)')

    plt.title("Example 3.1: Active Set QP Optimization Trajectory (From Scratch)", fontsize=13)
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.xlim(-0.3, 2.4)
    plt.ylim(-0.3, 2.2)
    plt.grid(True)
    plt.legend(loc='upper right')

    p_out = os.path.join(save_dir, "ch3_active_set_trajectory.png")
    plt.savefig(p_out, dpi=200)
    plt.close()
    print(f"-> Saved Active Set trajectory plot to: {p_out}")

if __name__ == "__main__":
    demonstrate_example_3_1(save_dir="/home/kuang/Desktop/mpc/ch03_solving_qp")
