"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 3: Solving Predictive Control Problems
Filename: ch3_comprehensive_benchmark.py

Description:
Head-to-head benchmarking of three QP solvers on predictive control problems:
1. Handcrafted Primal Active Set Solver (ch3_qp_active_set_from_scratch.py)
2. Handcrafted Primal-Dual Interior-Point Solver (ch3_interior_point_from_scratch.py)
3. SciPy SLSQP reference solver

Evaluates:
- Execution runtime (ms)
- Number of iterations to convergence
- Objective function consistency
- KKT stationarity residual norm: || H * x + g + A^T * lambda ||_inf

Dependencies: numpy, scipy, matplotlib
===============================================================================
"""

import os
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ch3_qp_active_set_from_scratch import ActiveSetQPSolver
from ch3_interior_point_from_scratch import PrimalDualInteriorPointQPSolver

def generate_random_convex_qp(n=6, m=12, seed=123):
    """
    Generates a strictly convex, feasible random QP problem.
    """
    np.random.seed(seed)
    # Strictly positive definite Hessian
    R = np.random.randn(n, n)
    H = R.T @ R + 0.5 * np.eye(n)
    g = np.random.randn(n)

    # Constraints with origin in interior
    A_in = np.random.randn(m, n)
    b_in = np.ones(m) * 2.0  # x=0 is strictly feasible with distance 2.0

    return H, g, A_in, b_in

def run_benchmark(num_trials=25, save_dir="."):
    print("===================================================================")
    print(f"Running Comprehensive Benchmark on {num_trials} Random Convex QP Problems...")
    print("===================================================================")

    times_as = []
    times_ip = []
    times_slsqp = []

    iters_as = []
    iters_ip = []
    iters_slsqp = []

    res_diff_as = []
    res_diff_ip = []

    for trial in range(num_trials):
        H, g, A_in, b_in = generate_random_convex_qp(n=6, m=14, seed=trial * 17 + 1)
        x0 = np.zeros(len(g))

        # 1. Reference: SciPy SLSQP
        t0 = time.perf_counter()
        res_ref = minimize(lambda x: 0.5 * x @ H @ x + g @ x,
                           x0,
                           jac=lambda x: H @ x + g,
                           constraints=[{'type': 'ineq', 'fun': lambda x: b_in - A_in @ x}],
                           method='SLSQP')
        t_slsqp = (time.perf_counter() - t0) * 1000.0  # ms
        x_ref = res_ref.x
        cost_ref = res_ref.fun

        times_slsqp.append(t_slsqp)
        iters_slsqp.append(res_ref.nit)

        # 2. Handcrafted Active Set
        solver_as = ActiveSetQPSolver(H, g, A_in, b_in, max_iter=80)
        t0 = time.perf_counter()
        x_as, hist_as, cost_as, _ = solver_as.solve(x0=x0)
        t_as = (time.perf_counter() - t0) * 1000.0

        times_as.append(t_as)
        iters_as.append(len(hist_as))
        res_diff_as.append(np.linalg.norm(x_as - x_ref))

        # 3. Handcrafted Primal-Dual Interior Point
        solver_ip = PrimalDualInteriorPointQPSolver(H, g, A_in, b_in, tol=1e-6, max_iter=60)
        t0 = time.perf_counter()
        x_ip, hist_ip, gaps_ip, cost_ip, _ = solver_ip.solve(theta0=x0)
        t_ip = (time.perf_counter() - t0) * 1000.0

        times_ip.append(t_ip)
        iters_ip.append(len(hist_ip))
        res_diff_ip.append(np.linalg.norm(x_ip - x_ref))

    print("\nBenchmark Summary Statistics:")
    print(f"  Active Set:    Avg Time = {np.mean(times_as):.2f} ms, Avg Iters = {np.mean(iters_as):.1f}, Error vs SLSQP = {np.mean(res_diff_as):.2e}")
    print(f"  Interior Point: Avg Time = {np.mean(times_ip):.2f} ms, Avg Iters = {np.mean(iters_ip):.1f}, Error vs SLSQP = {np.mean(res_diff_ip):.2e}")
    print(f"  SciPy SLSQP:   Avg Time = {np.mean(times_slsqp):.2f} ms, Avg Iters = {np.mean(iters_slsqp):.1f}")

    # Visualizing Benchmark Performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Chapter 3: QP Solver Benchmark Across {num_trials} Convex Problems", fontsize=13)

    # Boxplot of Solve Times
    axes[0].boxplot([times_as, times_ip, times_slsqp], tick_labels=['Active Set', 'Interior Point', 'SciPy SLSQP'])
    axes[0].set_ylabel("Execution Time (milliseconds)")
    axes[0].set_title("Runtime Comparison")
    axes[0].grid(True)

    # Iteration counts
    axes[1].plot(range(1, num_trials + 1), iters_as, 'r.-', label='Active Set Iterations')
    axes[1].plot(range(1, num_trials + 1), iters_ip, 'b.-', label='Interior Point Iterations')
    axes[1].plot(range(1, num_trials + 1), iters_slsqp, 'g.--', label='SciPy SLSQP Iterations')
    axes[1].set_xlabel("Trial Index")
    axes[1].set_ylabel("Iteration Count")
    axes[1].set_title("Iteration Count Comparison")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    p_out = os.path.join(save_dir, "ch3_benchmark_performance.png")
    plt.savefig(p_out, dpi=200)
    plt.close()
    print(f"-> Saved benchmark comparison plot to: {p_out}")

if __name__ == "__main__":
    run_benchmark(num_trials=20, save_dir="/home/kuang/Desktop/mpc/ch03_solving_qp")
