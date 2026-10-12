"""
Chapter 8 Robust Predictive Control: Section 8.3 & Exercise 8.3
LQG/LTR (Loop Transfer Recovery) Tuning & Kalman Filter Asymptotic Recovery
============================================================================
This script implements:
1. Target Feedback Loop (TFL) Design:
      Target state-feedback loop gain: L_target(z) = K_c * (z*I - A)^{-1} * B
      possessing guaranteed robust stability margins.
2. The Doyle-Stein Loop Transfer Recovery (LTR) Procedure:
      Augments process noise covariance: W(q) = W_0 + q * B * B^T
      As recovery parameter q -> infty (or measurement noise V -> 0, Exercise 8.3),
      the full observer-based loop transfer function:
         L_obs(z) = K_c * (z*I - A + B*K_c + L*C)^{-1} * L * C * (z*I - A)^{-1} * B
      asymptotically recovers the target loop shape L_target(z) everywhere!
3. Proof and Verification of Exercise 8.3:
      When V = cov{v(k)} -> 0, the Kalman filter gain partition satisfies L_{xw} -> I.
4. Bode and Singular Value Recovery Visualization across recovery parameter q.
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def solve_discrete_lqr(A, B, Q, R):
    """
    Solves discrete LQR: u = - K * x
    """
    P = la.solve_discrete_are(A, B, Q, R)
    K = la.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def solve_discrete_kalman(A, C, W, V):
    """
    Solves discrete Kalman filter observer gain L.
    A * P * A^T - P + W - A * P * C^T * (C * P * C^T + V)^{-1} * C * P * A^T = 0
    L = A * P * C^T * (C * P * C^T + V)^{-1}
    """
    P = la.solve_discrete_are(A.T, C.T, W, V)
    L = A @ P @ C.T @ la.inv(C @ P @ C.T + V)
    return L, P

def run_simulation():
    print("=" * 75)
    print("Chapter 8: Section 8.3 & Exercise 8.3 - LQG/LTR Loop Transfer Recovery")
    print("=" * 75)

    # 1. Discrete Plant (Ts = 0.2 s)
    # P(s) = 2 / (s^2 + 1.2s + 1)
    Ts = 0.2
    A = np.array([
        [0.9763, -0.1905],
        [0.1905,  0.7477]
    ])
    B = np.array([[0.0381], [0.1809]])
    C = np.array([[1.0, 0.0]])
    nx = A.shape[0]

    # 2. Target Feedback Loop (TFL) Design via State Feedback
    Q_tfl = np.eye(nx)
    R_tfl = np.array([[0.1]])
    K_target, _ = solve_discrete_lqr(A, B, Q_tfl, R_tfl)
    print(f"Target State-Feedback Gain K_target = {K_target}")

    # 3. LTR Observer Design for varying recovery parameter q
    W0 = 0.1 * np.eye(nx)
    V0 = np.array([[1.0]])
    q_values = [0.01, 1.0, 100.0, 10000.0]

    # Frequency grid up to Nyquist frequency
    w_nyq = np.pi / Ts
    w = np.logspace(-2, np.log10(w_nyq * 0.999), 400)

    # Target Loop Gain: L_target(e^{jw}) = K * (z*I - A)^{-1} * B
    L_target_mag = np.zeros(len(w))
    for i, wi in enumerate(w):
        z = np.exp(1j * wi * Ts)
        Phi_inv = la.inv(z * np.eye(nx) - A)
        G_tfl = K_target @ Phi_inv @ B
        L_target_mag[i] = np.abs(G_tfl[0, 0])

    # Observer Loop Gains: L_obs(e^{jw})
    L_obs_mags = {}
    observer_gains = {}

    for q in q_values:
        W_q = W0 + q * (B @ B.T)
        L_q, _ = solve_discrete_kalman(A, C, W_q, V0)
        observer_gains[q] = L_q

        mag_q = np.zeros(len(w))
        for i, wi in enumerate(w):
            z = np.exp(1j * wi * Ts)
            # Controller transfer function K_ctrl(z) = K_target * (z*I - A + B*K + L*C)^{-1} * L
            A_ctrl = A - B @ K_target - L_q @ C
            ctrl_tf = K_target @ la.inv(z * np.eye(nx) - A_ctrl) @ L_q
            # Plant P(z) = C * (z*I - A)^{-1} * B
            plant_tf = C @ la.inv(z * np.eye(nx) - A) @ B
            loop_tf = ctrl_tf @ plant_tf
            mag_q[i] = np.abs(loop_tf[0, 0])

        L_obs_mags[q] = mag_q

    # 4. Numerical Proof of Exercise 8.3: As V -> 0, L_xw -> I
    print("\nExercise 8.3 Verification (Measurement Noise V -> 0):")
    # Plant model with state and disturbance partition: x = [x_p; w]
    # x(k+1) = [A, B_w; 0, I] x(k) + [B; 0] u(k)
    A_ext = np.block([[A, B], [np.zeros((1, nx)), 1.0]])
    C_ext = np.block([[C, np.zeros((1, 1))]])
    W_ext = np.eye(nx + 1)

    for V_test in [1.0, 1e-2, 1e-4, 1e-7]:
        P_ext = la.solve_discrete_are(A_ext.T, C_ext.T, W_ext, np.array([[V_test]]))
        L_ext = A_ext @ P_ext @ C_ext.T @ la.inv(C_ext @ P_ext @ C_ext.T + np.array([[V_test]]))
        print(f"  V = {V_test:1.0e} => Kalman Gain L_ext = {L_ext.flatten()[:2]}, L_xw = {L_ext[2,0]:.6f}")

    # -------------------------------------------------------------
    # 5. Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Loop Transfer Recovery (Bode Magnitude)
    ax = axes[0, 0]
    ax.semilogx(w, 20 * np.log10(L_target_mag), 'k-', linewidth=3, label='Target Loop Shape $L_{target}(z)$')
    styles = ['r:', 'm-.', 'b--', 'g-']
    for q, style in zip(q_values, styles):
        ax.semilogx(w, 20 * np.log10(L_obs_mags[q]), style, linewidth=2, label=f'Recovered Loop ($q = {q}$)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Section 8.3: LQG/LTR Loop Transfer Recovery', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Loop Gain Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Recovery Error |L_target - L_obs| vs Frequency
    ax = axes[0, 1]
    for q, style in zip(q_values, styles):
        err_mag = np.abs(L_target_mag - L_obs_mags[q])
        ax.semilogx(w, err_mag, style, linewidth=2, label=f'Error ($q = {q}$)')
    ax.set_title(r'Recovery Error $|L_{target}(e^{j\omega}) - L_{obs}(e^{j\omega})|$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Absolute Error')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Nyquist Paths showing Robust Margins Recovery
    ax = axes[1, 0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, label='Unit Circle')
    ax.plot(-1, 0, 'r+', markersize=14, markeredgewidth=3, label='Critical Point (-1, 0)')

    # Nyquist curve for q = 0.01 and q = 10000
    for q, col in zip([0.01, 10000.0], ['red', 'green']):
        L_q = observer_gains[q]
        A_ctrl = A - B @ K_target - L_q @ C
        nyq_vals = []
        for wi in w:
            z = np.exp(1j * wi * Ts)
            ctrl_tf = (K_target @ la.inv(z * np.eye(nx) - A_ctrl) @ L_q)[0, 0]
            plant_tf = (C @ la.inv(z * np.eye(nx) - A) @ B)[0, 0]
            nyq_vals.append(ctrl_tf * plant_tf)
        nyq_vals = np.array(nyq_vals)
        ax.plot(np.real(nyq_vals), np.imag(nyq_vals), color=col, linewidth=2, label=f'Nyquist ($q={q}$)')

    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_title('Nyquist Path: Safe Distance from (-1, 0) Recovered', fontsize=11, fontweight='bold')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Step Response under Observer Recovery
    ax = axes[1, 1]
    # Simulate step response of target vs recovered loops
    sim_steps = 40
    t_pts = np.arange(sim_steps) * Ts

    # Target loop state space: A_cl = A - B * K_target
    x_tfl = np.zeros(nx)
    y_tfl = np.zeros(sim_steps)
    for k in range(sim_steps):
        y_tfl[k] = (C @ x_tfl)[0]
        u_k = - (K_target @ x_tfl)[0] + 1.0 / (C @ la.inv(np.eye(nx) - A + B @ K_target) @ B)[0, 0]
        x_tfl = A @ x_tfl + B.flatten() * u_k

    ax.step(t_pts, y_tfl, 'k-', where='post', linewidth=2.5, label='Target State-Feedback Response')

    for q, style in zip([0.01, 10000.0], ['r--', 'g-']):
        L_q = observer_gains[q]
        A_ctrl = A - B @ K_target - L_q @ C
        x_p = np.zeros(nx)
        x_h = np.zeros(nx)
        y_rec = np.zeros(sim_steps)
        for k in range(sim_steps):
            y_rec[k] = (C @ x_p)[0]
            innov = y_rec[k] - (C @ x_h)[0]
            u_k = - (K_target @ x_h)[0] + 1.0 / (C @ la.inv(np.eye(nx) - A + B @ K_target) @ B)[0, 0]
            x_h = A @ x_h + B.flatten() * u_k + L_q.flatten() * innov
            x_p = A @ x_p + B.flatten() * u_k
        ax.step(t_pts, y_rec, style, where='post', linewidth=2, label=f'Observer Loop ($q={q}$)')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Step Responses: Observer Loop Matches Target Loop', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch08_robust_mpc/ch8_lqg_ltr_recovery.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
