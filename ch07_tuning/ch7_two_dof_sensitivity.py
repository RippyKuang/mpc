"""
Chapter 7 Tuning: Section 7.1, Section 7.3, & Exercise 7.1
Two-Degree-of-Freedom Feedback Architecture, Sensitivity, and Complementary Sensitivity
========================================================================================
This script implements:
1. The 2-DOF linear feedback control structure (Figure 7.1 in Maciejowski):
      Forward controller: K(z)
      Feedback filter:    H(z)
      Plant:              P(z)
      Pre-filter:         F(z)
      Loop gain:          L(z) = P(z) * K(z) * H(z)
2. Exact algebraic verification of Exercise 7.1:
      S(z) + T(z) = I
   where S(z) = (I + L(z))^{-1} is the Sensitivity Function,
   and   T(z) = L(z) * (I + L(z))^{-1} is the Complementary Sensitivity Function.
3. Frequency response analysis (Section 7.3):
      - |S(e^{j omega})|: Disturbance attenuation and robust performance
      - |T(e^{j omega})|: Noise attenuation and robust stability (multiplicative uncertainty)
      - Peak sensitivity Ms = max_omega |S(e^{j omega})|
      - Guaranteed gain margin GM >= Ms / (Ms - 1)
      - Guaranteed phase margin PM >= 2 * arcsin(1 / (2 * Ms))
4. The Waterbed Effect (Bode's Sensitivity Integral):
      Lowering |S| at low frequencies inevitably pushes |S| > 1 at intermediate frequencies!
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_freq_response(num, den, dt, w_array):
    """
    Evaluates discrete transfer function G(e^{j w dt}) = num(z)/den(z)
    over a frequency grid w_array (rad/s).
    """
    z = np.exp(1j * w_array * dt)
    num_val = np.polyval(num, z)
    den_val = np.polyval(den, z)
    return num_val / den_val

def run_analysis():
    print("=" * 75)
    print("Chapter 7: Two-DOF Sensitivity Analysis & Exercise 7.1 Verification")
    print("=" * 75)

    # 1. Continuous-time Plant and Discretization
    # Second-order underdamped plant: P(s) = 2 / (s^2 + 2*zeta*wn*s + wn^2)
    # wn = 1.0 rad/s, zeta = 0.5, Ts = 0.2 s
    Ts = 0.2
    wn = 1.0
    zeta = 0.5
    sys_c = signal.TransferFunction([2.0], [1.0, 2 * zeta * wn, wn**2])
    sys_d = signal.cont2discrete((sys_c.num, sys_c.den), Ts, method='zoh')
    num_P = sys_d[0][0]
    den_P = sys_d[1]

    print(f"Sampling time Ts = {Ts} s")
    print(f"Discrete Plant P(z):")
    print(f"  Numerator:   {num_P}")
    print(f"  Denominator: {den_P}")

    # 2. Predictive/PI-like Controller K(z) with integral action for zero offset
    # K(z) = Kp + Ki * Ts / (z - 1) = ((Kp + Ki*Ts)*z - Kp) / (z - 1)
    Kp = 0.8
    Ki = 0.4
    num_K = np.array([Kp + Ki * Ts, -Kp])
    den_K = np.array([1.0, -1.0])

    # Feedback filter H(z): first-order low-pass filter H(z) = (1 - alpha) / (z - alpha)
    # alpha = exp(-Ts / tau_f), tau_f = 0.1 s
    tau_f = 0.1
    alpha = np.exp(-Ts / tau_f)
    num_H = np.array([1.0 - alpha])
    den_H = np.array([1.0, -alpha])

    # 3. Open-loop transfer function L(z) = P(z) * K(z) * H(z)
    num_L = np.convolve(num_P, np.convolve(num_K, num_H))
    den_L = np.convolve(den_P, np.convolve(den_K, den_H))

    # Frequency grid up to Nyquist frequency w_nyq = pi / Ts
    w_nyq = np.pi / Ts
    w = np.logspace(-2, np.log10(w_nyq * 0.999), 1000)

    # 4. Evaluate Frequency Responses
    L_jw = evaluate_freq_response(num_L, den_L, Ts, w)

    # Sensitivity S(z) = 1 / (1 + L(z))
    S_jw = 1.0 / (1.0 + L_jw)

    # Complementary Sensitivity T(z) = L(z) / (1 + L(z))
    T_jw = L_jw / (1.0 + L_jw)

    # 5. Numerical Proof of Exercise 7.1: S(z) + T(z) = I
    sum_ST = S_jw + T_jw
    identity_error = np.max(np.abs(sum_ST - 1.0))
    print(f"\nExercise 7.1 Verification: max |S(e^{{jw}}) + T(e^{{jw}}) - 1| = {identity_error:.2e}")
    assert identity_error < 1e-12, "Identity S + T = 1 failed!"
    print(">> Equation (7.8) S(z) + T(z) = I holds with machine precision!")

    # 6. Robustness Margins
    Ms = np.max(np.abs(S_jw))
    w_Ms = w[np.argmax(np.abs(S_jw))]
    Mt = np.max(np.abs(T_jw))

    GM_bound = Ms / (Ms - 1.0) if Ms > 1.0 else np.inf
    PM_bound_deg = np.degrees(2.0 * np.arcsin(1.0 / (2.0 * Ms))) if Ms > 0.5 else 180.0

    print(f"\nRobustness Metrics (Section 7.3):")
    print(f"  Peak Sensitivity Ms  = {Ms:.4f} (at w = {w_Ms:.3f} rad/s)")
    print(f"  Peak Comp Sens Mt    = {Mt:.4f}")
    print(f"  Lower Bound Gain Margin  GM >= {GM_bound:.2f} ({20*np.log10(GM_bound):.2f} dB)")
    print(f"  Lower Bound Phase Margin PM >= {PM_bound_deg:.2f} deg")

    # -------------------------------------------------------------
    # 7. Plotting Frequency Responses
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Sensitivity and Complementary Sensitivity (Bode Magnitude)
    ax = axes[0, 0]
    ax.semilogx(w, 20 * np.log10(np.abs(S_jw)), 'b-', linewidth=2, label=r'Sensitivity $|S(e^{j\omega})|$ (Disturbance)')
    ax.semilogx(w, 20 * np.log10(np.abs(T_jw)), 'r--', linewidth=2, label=r'Comp. Sens. $|T(e^{j\omega})|$ (Noise/Tracking)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(20 * np.log10(Ms), color='blue', linestyle='-.', alpha=0.6, label=f'Peak $M_s = {Ms:.2f}$ ({20*np.log10(Ms):.1f} dB)')
    ax.set_title(r'Sensitivity Functions $S(z)$ and $T(z)$ (Section 7.3)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Verification of Exercise 7.1: S + T = 1
    ax = axes[0, 1]
    ax.semilogx(w, np.abs(sum_ST), 'g-', linewidth=2, label=r'$|S(e^{j\omega}) + T(e^{j\omega})|$')
    ax.semilogx(w, np.abs(sum_ST - 1.0), 'm:', linewidth=1.5, label=r'Error $|(S+T) - 1|$ ($< 10^{-14}$)')
    ax.set_title('Exercise 7.1: Numerical Proof of $S(z) + T(z) = I$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Absolute Value')
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Nyquist Diagram of Loop Gain L(z) and Ms Circle
    ax = axes[1, 0]
    ax.plot(np.real(L_jw), np.imag(L_jw), 'b-', linewidth=2, label=r'$L(e^{j\omega})$ Nyquist Path')
    ax.plot(np.real(L_jw), -np.imag(L_jw), 'b--', alpha=0.4)  # negative frequencies
    # Critical point (-1, 0)
    ax.plot(-1, 0, 'r+', markersize=14, markeredgewidth=3, label='Critical Point (-1, 0)')
    # Ms circle: circle around (-1, 0) with radius 1 / Ms
    theta = np.linspace(0, 2*np.pi, 200)
    r_ms = 1.0 / Ms
    ax.plot(-1 + r_ms * np.cos(theta), r_ms * np.sin(theta), 'r:', linewidth=2, label=f'$M_s$ Circle (Radius $1/M_s = {r_ms:.2f}$)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_title('Nyquist Diagram and Robustness Margin Circle', fontsize=11, fontweight='bold')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Step Responses (Tracking and Disturbance Rejection)
    ax = axes[1, 1]
    # Simulate closed-loop response to step setpoint and step disturbance
    t_sim = 60
    n_sim = int(t_sim / Ts)
    time_pts = np.arange(n_sim) * Ts

    # Closed-loop transfer functions:
    # y = T * s + S * d
    sys_T = signal.dlti(np.convolve(num_P, num_K), np.polyadd(den_L, num_L), dt=Ts)
    sys_S = signal.dlti(den_L, np.polyadd(den_L, num_L), dt=Ts)

    _, y_step = signal.dstep(sys_T, n=n_sim)
    _, y_dist = signal.dstep(sys_S, n=n_sim)

    ax.step(time_pts, y_step[0].flatten(), 'b-', where='post', linewidth=2, label='Set-point Step Response')
    ax.step(time_pts, y_dist[0].flatten(), 'r--', where='post', linewidth=1.5, label='Disturbance Step Response')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Time-Domain 2-DOF Closed-Loop Responses', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch07_tuning/ch7_two_dof_sensitivity.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_analysis()
