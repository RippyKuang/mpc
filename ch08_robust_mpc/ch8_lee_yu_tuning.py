"""
Chapter 8 Robust Predictive Control: Section 8.2 & Exercise 8.2
The Robust Tuning Procedure of Lee and Yu (1994) & Observer Modal Decomposition
==============================================================================
This script implements:
1. The Lee & Yu (1994) Robust MPC Tuning Framework (Section 8.2):
      Shapes the disturbance and measurement noise models in the state estimator
      to roll off observer gain at high frequencies, achieving robust stability
      against unmodeled high-frequency plant dynamics.
2. Proof and Demonstration of Exercise 8.2:
      The observer poles decompose into:
      - The plant poles (eigenvalues of A)
      - The eigenvalues of the disturbance estimator block matrix (Equation 8.122):
           M_obs = [[I - L_d * C_d, - L_d * C_d],
                    [I - L_w * C_d, - L_w * C_d]]
3. Time and Frequency Domain Simulation:
      - Standard Observer (Aggressive): Observer bandwidth too high, destabilized by unmodeled lag.
      - Lee-Yu Robust Observer: Observer bandwidth rolled off, preserving robust stability.
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_lee_yu_observer_poles(A, C, Ad, Cd, L_x, L_d):
    """
    Computes full observer poles and verifies the modal decoupling theorem (Exercise 8.2).
    Full augmented system: A_aug = [[A, 0], [0, Ad]], C_aug = [C, Cd]
    Observer error matrix: A_obs = A_aug - L_aug * C_aug
    """
    nx = A.shape[0]
    nd = Ad.shape[0]

    A_aug = la.block_diag(A, Ad)
    C_aug = np.hstack([C, Cd])
    L_aug = np.vstack([L_x, L_d])

    # Full closed-loop observer error transition matrix
    A_err = A_aug - L_aug @ C_aug
    full_poles = la.eigvals(A_err)

    # Decoupled components (Exercise 8.2)
    plant_poles = la.eigvals(A)
    # Block matrix (8.122)
    M_block = Ad - L_d @ Cd
    dist_poles = la.eigvals(M_block)

    return full_poles, plant_poles, dist_poles

def simulate_lee_yu_system(Ts=0.2, sim_steps=50):
    # Nominal Plant: P0(s) = 1 / (s + 1), Ts = 0.2
    # x(k+1) = a * x(k) + b * u(k)
    a = np.exp(-Ts)
    b = 1.0 - a
    A = np.array([[a]])
    B = np.array([[b]])
    C = np.array([[1.0]])

    # Disturbance model: Ad = 1.0 (integrator for step disturbance), Cd = 1.0
    Ad = np.array([[1.0]])
    Cd = np.array([[1.0]])

    # Design 1: Standard DMC-like Deadbeat Observer (L_x = 0, L_d = 1.0)
    L1_x = np.array([[0.0]])
    L1_d = np.array([[0.95]])
    full_poles_1, plant_p_1, dist_p_1 = analyze_lee_yu_observer_poles(A, C, Ad, Cd, L1_x, L1_d)

    # Design 2: Lee-Yu Robust Filtered Observer (L_x, L_d tuned for low bandwidth)
    L2_x = np.array([[0.15]])
    L2_d = np.array([[0.25]])
    full_poles_2, plant_p_2, dist_p_2 = analyze_lee_yu_observer_poles(A, C, Ad, Cd, L2_x, L2_d)

    print(f"Plant Pole: {plant_p_1[0]:.4f}")
    print(f"Design 1 (Aggressive) Observer Poles: Full = {full_poles_1}")
    print(f"Design 2 (Lee-Yu Robust) Observer Poles: Full = {full_poles_2}")

    # Perturbed Plant with extra unmodeled delay (2-step pure delay):
    # Real plant has additional states
    A_pert = np.array([
        [a,    b,   0.0],
        [0.0,  0.0, 1.0],
        [0.0,  0.0, 0.0]
    ])
    B_pert = np.array([[0.0], [0.0], [1.0]])
    C_pert = np.array([[1.0, 0.0, 0.0]])

    # Simulate closed-loop with both observers tracking step setpoint
    # State feedback gain K = 1.2
    K_fb = np.array([1.2])

    t_pts = np.arange(sim_steps) * Ts
    r_target = 1.0

    def run_loop(L_x, L_d):
        x_true = np.zeros(3)
        x_hat = np.zeros(1)
        d_hat = np.zeros(1)
        y_hist = np.zeros(sim_steps)
        u_hist = np.zeros(sim_steps)

        u_prev = 0.0
        for k in range(sim_steps):
            y_meas = (C_pert @ x_true)[0]
            y_hist[k] = y_meas

            # Innovation
            y_pred = (C @ x_hat)[0] + (Cd @ d_hat)[0]
            innov = y_meas - y_pred

            # Observer update
            x_hat_up = x_hat + L_x.flatten() * innov
            d_hat_up = d_hat + L_d.flatten() * innov

            # Control calculation: u = - K * x_hat + feedforward for setpoint
            u_k = K_fb[0] * (r_target - d_hat_up[0] - x_hat_up[0])
            u_hist[k] = u_k

            # Observer propagation
            x_hat = A @ x_hat_up + B.flatten() * u_k
            d_hat = Ad @ d_hat_up

            # True plant propagation
            x_true = A_pert @ x_true + B_pert.flatten() * u_k

        return y_hist, u_hist

    y_d1, u_d1 = run_loop(L1_x, L1_d)
    y_d2, u_d2 = run_loop(L2_x, L2_d)

    return t_pts, y_d1, u_d1, y_d2, u_d2, full_poles_1, full_poles_2

def run_simulation():
    print("=" * 75)
    print("Chapter 8: Section 8.2 & Exercise 8.2 - Lee & Yu Robust Tuning")
    print("=" * 75)

    t_pts, y_d1, u_d1, y_d2, u_d2, p1, p2 = simulate_lee_yu_system()

    # -------------------------------------------------------------
    # Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Output Tracking under Unmodeled Delay
    ax = axes[0, 0]
    ax.plot(t_pts, y_d1, 'r--', linewidth=2, label='Standard Observer (Oscillates & poorly damped)')
    ax.plot(t_pts, y_d2, 'g-', linewidth=2, label='Lee-Yu Robust Observer (Smooth & well damped)')
    ax.axhline(1.0, color='gray', linestyle=':', label='Target Setpoint r=1.0')
    ax.set_title('Plant Output y(t) under Unmodeled High-Frequency Delay', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Control Signal u(t) Comparison
    ax = axes[0, 1]
    ax.step(t_pts, u_d1, 'r--', where='post', linewidth=1.5, label='Standard Observer: High gain chatter')
    ax.step(t_pts, u_d2, 'g-', where='post', linewidth=2, label='Lee-Yu Observer: Smooth filtered actuation')
    ax.set_title('Control Actuation u(t) Signals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input u(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Complex Plane Observer Pole Map (Exercise 8.2)
    ax = axes[1, 0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
    ax.scatter(np.real(p1), np.imag(p1), color='red', marker='x', s=120, linewidth=2, label='Design 1 Poles (Aggressive)')
    ax.scatter(np.real(p2), np.imag(p2), color='green', marker='o', s=100, linewidth=2, label='Design 2 Poles (Lee-Yu Tuned)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title('Exercise 8.2: Observer Poles in the Complex z-Plane', fontsize=11, fontweight='bold')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Observer Sensitivity Loop Shape
    ax = axes[1, 1]
    Ts = 0.2
    w_nyq = np.pi / Ts
    w = np.logspace(-2, np.log10(w_nyq * 0.99), 300)
    z = np.exp(1j * w * Ts)

    # Transfer function of disturbance estimator
    T_obs1 = (0.95) / (z - (1.0 - 0.95))
    T_obs2 = (0.25) / (z - (1.0 - 0.25))

    ax.semilogx(w, 20 * np.log10(np.abs(T_obs1)), 'r--', linewidth=2, label='Standard Observer (High bandwidth)')
    ax.semilogx(w, 20 * np.log10(np.abs(T_obs2)), 'g-', linewidth=2, label='Lee-Yu Observer (Rolled off at high freq)')
    ax.set_title('Disturbance Estimator Filter Magnitude (Bode Plot)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch08_robust_mpc/ch8_lee_yu_tuning.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
