"""
Chapter 6: Stability
Script 1: Example 6.1 & Exercise 6.1 - Closed-Loop Instability of Short-Horizon MPC
and Rescue via Terminal Equality Constraint (Section 6.1)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 6.1 & Exercise 6.1

Key Concepts Implemented:
1. Plant Model (Two Delays in Series, Deadbeat Stable Open-Loop):
   x1(k+1) = u(k),  x2(k+1) = x1(k)
   A = [[0, 0], [1, 0]],  B = [[1], [0]],  C = [[1, 0]]
2. Short Horizon Hp = 1 with Positive-Definite Non-Diagonal Weight Q:
   Q = [[1, 2], [2, 6]]  (det = 2 > 0, trace = 7 > 0)
   V(k) = x(k+1)^T * Q * x(k+1) = u(k)^2 + 4*u(k)*x1(k) + 6*x1(k)^2
   Optimal unconstrained control: u(k) = -2 * x1(k)
3. Destabilization Mechanism:
   A_cl = A - B*K = [[-2, 0], [1, 0]] -> Eigenvalues at lambda = -2, 0!
   |lambda| = 2 > 1 -> Severe exponential divergence!
4. Stabilization via Terminal Constraint (Theorem 6.1):
   Imposing x(k+Hp|k) = 0 enforces u(k) = 0 -> A_cl = [[0, 0], [1, 0]] (Deadbeat Stable).
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simulate_example_6_1(N_steps=12, x0=np.array([[1.0], [0.0]])):
    """
    Simulates Example 6.1 under:
    1. Unconstrained MPC (Hp = 1, Hu = 1): u(k) = -2*x1(k) (Instability!)
    2. Terminal Constrained MPC (x(k+1) = 0): u(k) = 0 (Deadbeat Stability!)
    """
    A = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    B = np.array([[1.0],
                  [0.0]])
    Q = np.array([[1.0, 2.0],
                  [2.0, 6.0]])

    # Check open-loop eigenvalues
    eigs_ol = la.eigvals(A)
    print("=" * 70)
    print("Example 6.1 & Exercise 6.1: Closed-Loop Instability of Short-Horizon MPC")
    print("=" * 70)
    print(f"Open-loop plant matrix A:\n{A}")
    print(f"Open-loop eigenvalues: {eigs_ol.real} (Both at 0: Deadbeat Stable!)")
    print(f"State weight matrix Q:\n{Q}")
    print(f"Eigenvalues of Q: {la.eigvals(Q).real} (Strictly Positive Definite!)")

    # Analytical unconstrained gain:
    # dV/du = 2*u + 4*x1 = 0 -> u = -2*x1
    K_unconstrained = np.array([[2.0, 0.0]])
    A_cl_uncon = A - B @ K_unconstrained
    eigs_cl_uncon = la.eigvals(A_cl_uncon)
    print(f"\nUnconstrained Closed-Loop Matrix A_cl:\n{A_cl_uncon}")
    print(f"Closed-Loop Eigenvalues: {eigs_cl_uncon.real}")
    print(f"Max eigenvalue magnitude |lambda| = {np.max(np.abs(eigs_cl_uncon)):.1f} > 1.0 (UNSTABLE!)")

    # Terminal constrained gain: x(k+1) = 0 -> u(k) = 0
    K_terminal = np.array([[0.0, 0.0]])
    A_cl_term = A - B @ K_terminal
    eigs_cl_term = la.eigvals(A_cl_term)
    print(f"\nTerminal Constrained Closed-Loop Matrix A_cl:\n{A_cl_term}")
    print(f"Closed-Loop Eigenvalues: {eigs_cl_term.real} (Deadbeat Stable!)")

    # Time-domain simulation
    t = np.arange(N_steps)
    x_uncon = np.zeros((N_steps, 2))
    u_uncon = np.zeros(N_steps)
    x_term = np.zeros((N_steps, 2))
    u_term = np.zeros(N_steps)

    x_curr_u = x0.copy()
    x_curr_t = x0.copy()

    for k in range(N_steps):
        x_uncon[k, :] = x_curr_u.flatten()
        x_term[k, :] = x_curr_t.flatten()

        # Unconstrained action
        u_u = -2.0 * x_curr_u[0, 0]
        u_uncon[k] = u_u
        x_curr_u = A @ x_curr_u + B * u_u

        # Terminal constrained action
        u_t = 0.0
        u_term[k] = u_t
        x_curr_t = A @ x_curr_t + B * u_t

    return t, x_uncon, u_uncon, x_term, u_term

def main():
    t, x_uncon, u_uncon, x_term, u_term = simulate_example_6_1()

    print("\nSimulation trajectory table (Unconstrained vs Terminal Constrained):")
    print(f"{'Step k':>6} | {'Uncon x1':>12} | {'Uncon x2':>12} | {'Uncon u':>12} || {'Term x1':>10} | {'Term x2':>10}")
    print("-" * 75)
    for k in range(min(7, len(t))):
        print(f"{k:6d} | {x_uncon[k, 0]:12.2f} | {x_uncon[k, 1]:12.2f} | {u_uncon[k]:12.2f} || {x_term[k, 0]:10.2f} | {x_term[k, 1]:10.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Unconstrained Divergent State Trajectory
    axes[0, 0].plot(t, x_uncon[:, 0], 'r-o', lw=2, label='State x1(k)')
    axes[0, 0].plot(t, x_uncon[:, 1], 'b-s', lw=2, label='State x2(k)')
    axes[0, 0].set_xlabel('Time Step k', fontsize=11)
    axes[0, 0].set_ylabel('State Magnitude', fontsize=11)
    axes[0, 0].set_title('(a) Example 6.1: Exponential Divergence (Eigenvalue = -2)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':')
    axes[0, 0].legend()

    # (b) Unconstrained Divergent Control Input
    axes[0, 1].step(t, u_uncon, 'k-o', lw=1.8, where='post', label='u(k) = -2*x1(k)')
    axes[0, 1].set_xlabel('Time Step k', fontsize=11)
    axes[0, 1].set_ylabel('Control Input u(k)', fontsize=11)
    axes[0, 1].set_title('(b) Control Input Explosive Oscillation', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].legend()

    # (c) Terminal Constrained Stable State Trajectory
    axes[1, 0].plot(t, x_term[:, 0], 'g-o', lw=2.2, label='State x1(k) (Deadbeat -> 0)')
    axes[1, 0].plot(t, x_term[:, 1], 'm-s', lw=2.2, label='State x2(k) (Deadbeat -> 0)')
    axes[1, 0].set_xlabel('Time Step k', fontsize=11)
    axes[1, 0].set_ylabel('State Magnitude', fontsize=11)
    axes[1, 0].set_title('(c) Theorem 6.1: Rescue via Terminal Equality Constraint x(k+1)=0', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':')
    axes[1, 0].legend()

    # (d) Phase Plane Comparison (x1 vs x2)
    axes[1, 1].plot(x_uncon[:, 0], x_uncon[:, 1], 'r--o', lw=1.5, label='Unconstrained (Escapes to Inf)')
    axes[1, 1].plot(x_term[:, 0], x_term[:, 1], 'g-s', lw=2.5, label='Terminal Constrained (Converges to Origin)')
    axes[1, 1].plot(0, 0, 'k*', markersize=15, label='Target Origin (0,0)')
    axes[1, 1].set_xlabel('State x1', fontsize=11)
    axes[1, 1].set_ylabel('State x2', fontsize=11)
    axes[1, 1].set_title('(d) Phase Plane Trajectory Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, linestyle=':')
    axes[1, 1].legend()

    plt.tight_layout()
    output_png = '/home/kuang/Desktop/mpc/ch06_stability/ch6_example6_1_instability.png'
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"\n[Plot Saved] -> {output_png}")

if __name__ == '__main__':
    main()
