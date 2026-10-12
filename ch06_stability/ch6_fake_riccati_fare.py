"""
Chapter 6 Stability: Section 6.3 & Exercise 6.7
The Fake Algebraic Riccati Equation (FARE) and Sufficiency vs Necessity of Q >= 0
==================================================================================
This script implements:
1. Exact reproduction and analytical verification of Exercise 6.7:
      A = [[0, 2], [2, 0]],  B = [[1], [0]],  P_{N-1} = 16 * I_2,  R = 1
      Optimal gain: K_{N-1} = (R + B^T P B)^{-1} B^T P A = [0, 32/17] = [0, 1.8824]
      Closed-loop matrix: A_cl = A - B * K_{N-1} = [[0, 2/17], [2, 0]]
      Eigenvalues: lambda = +- sqrt(4/17) ~= +- 0.48507  (|lambda| < 1 => STABLE!)
2. Calculation of implied stage weight matrix Q_{N-1}:
      Q_{N-1} = P_{N-1} - A^T P A + A^T P B K_{N-1} = diag(-48, 12.2353)
      Eigenvalues of Q_{N-1}: {-48, +12.2353} -> INDEFINITE (contains negative eigenvalue)!
3. Key Theoretical Takeaway:
      Lyapunov / Riccati condition Q >= 0 is SUFFICIENT for stability, but NOT NECESSARY!
      A system can be stabilized by a Riccati iteration even when Q is indefinite.
4. Numerical simulation:
      - Closed loop trajectory from initial state x(0)
      - Complex plane pole plot showing open loop vs closed loop
      - Riccati difference equation backwards propagation over N steps
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_exercise_6_7():
    print("=" * 75)
    print("Chapter 6: Exercise 6.7 - Riccati Iteration with Indefinite State Weight Q")
    print("=" * 75)

    # 1. System matrices
    A = np.array([[0.0, 2.0],
                  [2.0, 0.0]])
    B = np.array([[1.0],
                  [0.0]])
    P_prev = 16.0 * np.eye(2)
    R = 1.0

    print("Plant Matrices:")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"P_{{N-1}} =\n{P_prev}")
    print(f"R = {R}")

    # Open-loop poles:
    ol_poles = la.eigvals(A)
    print(f"\nOpen-loop eigenvalues of A: {ol_poles} (Unstable! |lambda| = 2 > 1)")

    # 2. Optimal gain K_{N-1}
    # K_{N-1} = (R + B^T P B)^{-1} B^T P A
    # B^T P B = [1, 0] * 16 * [1; 0] = 16
    # R + B^T P B = 1 + 16 = 17
    # B^T P A = [16, 0] * [[0, 2], [2, 0]] = [0, 32]
    # K = [0, 32/17]
    BPB = (B.T @ P_prev @ B)[0, 0]
    BPA = B.T @ P_prev @ A
    K = BPA / (R + BPB)
    print(f"\nComputed Optimal Gain K_{{N-1}} = {K}")
    print(f"Exact Analytical K_{{N-1}}    = [0, 32/17] = [0, {32/17:.6f}]")

    # 3. Closed-loop matrix and eigenvalues
    A_cl = A - B @ K
    cl_poles = la.eigvals(A_cl)
    exact_cl_pole = np.sqrt(4.0 / 17.0)
    print(f"\nClosed-loop Matrix A_cl = A - B * K_{{N-1}}:\n{A_cl}")
    print(f"Closed-loop Eigenvalues: {cl_poles}")
    print(f"Exact Analytical Eigenvalues: +- sqrt(4/17) = +- {exact_cl_pole:.6f}")
    print(f"Spectral radius rho(A_cl) = {np.max(np.abs(cl_poles)):.6f} < 1.0 -> STRICTLY ASYMPTOTICALLY STABLE!")

    # 4. Implied stage weight matrix Q_{N-1}
    # Riccati step: P_{N-1} = Q_{N-1} + A^T P_{N-1} A - A^T P_{N-1} B (R + B^T P B)^{-1} B^T P A
    # => Q_{N-1} = P_{N-1} - A^T P A + A^T P B K
    AT_P_A = A.T @ P_prev @ A
    AT_P_B_K = A.T @ P_prev @ B @ K
    Q_implied = P_prev - AT_P_A + AT_P_B_K
    q_eigvals = la.eigvals(Q_implied)
    print(f"\nImplied Stage Weight Matrix Q_{{N-1}}:\n{Q_implied}")
    print(f"Eigenvalues of Q_{{N-1}}: {q_eigvals}")
    print("Notice: Eigenvalues are -48.0 and +12.2353. Q_{N-1} is strictly INDEFINITE!")
    print("Conclusion: Standard LQ/Lyapunov theorem requires Q >= 0 for SUFFICIENCY.")
    print("            Exercise 6.7 demonstrates Q >= 0 is NOT NECESSARY for stability!")

    return A, B, K, A_cl, ol_poles, cl_poles, Q_implied

def simulate_and_plot(A, B, K, A_cl, ol_poles, cl_poles):
    sim_steps = 15
    x0 = np.array([2.0, -1.5])

    # Simulate closed-loop system
    x_cl = np.zeros((sim_steps + 1, 2))
    u_cl = np.zeros(sim_steps)
    x_cl[0] = x0

    for k in range(sim_steps):
        u_cl[k] = - (K @ x_cl[k])[0]
        x_cl[k+1] = A @ x_cl[k] + B.flatten() * u_cl[k]

    # Riccati difference equation backwards propagation
    N_horizon = 20
    P_traj = np.zeros((N_horizon, 2, 2))
    K_traj = np.zeros((N_horizon, 2))
    P_curr = 16.0 * np.eye(2)
    R = 1.0
    Q_fixed = np.eye(2)  # For comparison: standard Riccati with Q > 0

    for i in range(N_horizon):
        P_traj[i] = P_curr
        BPB = (B.T @ P_curr @ B)[0, 0]
        BPA = B.T @ P_curr @ A
        K_i = BPA / (R + BPB)
        K_traj[i] = K_i.flatten()
        P_curr = Q_fixed + A.T @ P_curr @ A - A.T @ P_curr @ B @ K_i

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    time_steps = np.arange(sim_steps + 1)
    time_u = np.arange(sim_steps)

    # Plot (a): Closed-loop State Trajectories
    ax = axes[0, 0]
    ax.plot(time_steps, x_cl[:, 0], 'b-o', label=r'$x_1(k)$')
    ax.plot(time_steps, x_cl[:, 1], 'g-s', label=r'$x_2(k)$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(r'Exercise 6.7: Closed-Loop Trajectories with Indefinite $Q_{N-1}$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('States')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (b): Control Signal u(k)
    ax = axes[0, 1]
    ax.step(time_u, u_cl, 'r-d', where='post', label=r'$u(k) = -K_{N-1} x(k)$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Control Input Evolution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel('Control Input u(k)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (c): Complex Plane Pole Map (Open Loop vs Closed Loop)
    ax = axes[1, 0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle ($|z| = 1$)')
    ax.scatter(np.real(ol_poles), np.imag(ol_poles), color='red', marker='x', s=120, linewidth=3, label=r'Open-loop Poles $\pm 2.0$ (Unstable)')
    ax.scatter(np.real(cl_poles), np.imag(cl_poles), color='green', marker='o', s=100, linewidth=2, label=r'Closed-loop Poles $\pm 0.485$ (Stable)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Complex Plane Pole Locations', fontsize=11, fontweight='bold')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot (d): Riccati Matrix P Convergence
    ax = axes[1, 1]
    steps_r = np.arange(N_horizon)
    ax.plot(steps_r, P_traj[:, 0, 0], 'b-o', label=r'$P_{11}$')
    ax.plot(steps_r, P_traj[:, 1, 1], 'm-s', label=r'$P_{22}$')
    ax.plot(steps_r, P_traj[:, 0, 1], 'g-^', label=r'$P_{12} = P_{21}$')
    ax.set_title('Riccati Difference Equation Backward Iteration', fontsize=11, fontweight='bold')
    ax.set_xlabel('Backward Step index')
    ax.set_ylabel('Matrix Elements of P')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch06_stability/ch6_fake_riccati_fare.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    A, B, K, A_cl, ol_poles, cl_poles, Q_implied = analyze_exercise_6_7()
    simulate_and_plot(A, B, K, A_cl, ol_poles, cl_poles)
