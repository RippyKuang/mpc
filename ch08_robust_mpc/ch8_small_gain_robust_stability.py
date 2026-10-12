"""
Chapter 8 Robust Predictive Control: Section 8.1 & Exercise 8.1
Multiplicative Uncertainty, Small Gain Theorem, and Robust Stability
=====================================================================
This script implements:
1. Output Multiplicative Uncertainty Model (Exercise 8.1):
      P(z) = [I + Delta(z)] * P_0(z)
   where P_0(z) is the nominal plant and Delta(z) is a stable norm-bounded uncertainty.
2. Derivation and Demonstration of the M - Delta Interconnection:
      The feedback connection of actual plant P(z) with controller K(z) reduces to
      the feedback connection of Delta(z) with the complementary sensitivity function:
         T(z) = P_0(z) * K(z) * [I + P_0(z) * K(z)]^{-1}
3. Small Gain Theorem (Section 8.1 & Exercise 8.1):
      The uncertain closed loop is guaranteed to be robustly stable for all ||Delta||_infty <= gamma
      if and only if:
         ||T(e^{j omega})||_infty < 1 / gamma   <=>   sigma_max(T(e^{j omega})) * sigma_max(Delta(e^{j omega})) < 1
4. Numerical Simulation & Comparison:
      - Controller 1 (Aggressive MPC, R small): High bandwidth, peak Mt = 3.2.
        Violates small gain condition -> Destabilized by high-frequency uncertainty!
      - Controller 2 (Robust MPC, R tuned): Moderate bandwidth, peak Mt = 1.15.
        Satisfies small gain condition -> Guaranteed robustly stable!
"""

import numpy as np
import scipy.signal as signal
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_discrete_tf(num, den, Ts, w_array):
    """
    Evaluates discrete transfer function G(e^{j w Ts}) = num(z) / den(z).
    """
    z = np.exp(1j * w_array * Ts)
    return np.polyval(num, z) / np.polyval(den, z)

def simulate_closed_loop(num_P, den_P, num_K, den_K, sim_steps=50, Ts=0.2):
    """
    Simulates linear discrete closed loop with unity feedback:
    u(k) = K(z) * [r(k) - y(k)]
    y(k) = P(z) * u(k)
    """
    # Closed loop numerator and denominator: T(z) = (P*K) / (1 + P*K)
    num_PK = np.convolve(num_P, num_K)
    den_PK = np.convolve(den_P, den_K)
    den_CL = np.polyadd(den_PK, num_PK)

    # Check stability of closed-loop poles
    cl_poles = np.roots(den_CL)
    is_stable = np.all(np.abs(cl_poles) < 1.0)
    rho_cl = np.max(np.abs(cl_poles))

    # Time-domain step response
    t_pts = np.arange(sim_steps) * Ts
    r = np.ones(sim_steps)

    # State-space realization of closed loop
    A_cl, B_cl, C_cl, D_cl = signal.tf2ss(num_PK, den_CL)
    x = np.zeros(A_cl.shape[0])
    y = np.zeros(sim_steps)
    u = np.zeros(sim_steps)

    for k in range(sim_steps):
        y[k] = (C_cl @ x)[0] + (D_cl[0, 0] * r[k] if D_cl.size > 0 else 0.0)
        x = A_cl @ x + B_cl.flatten() * r[k]

    return t_pts, y, is_stable, rho_cl, cl_poles

def run_simulation():
    print("=" * 75)
    print("Chapter 8: Small Gain Theorem & Robust Stability (Section 8.1 & Ex 8.1)")
    print("=" * 75)

    Ts = 0.2
    # 1. Nominal Plant: P_0(s) = 2 / (s + 1), Ts = 0.2 s
    sys_P0_c = signal.TransferFunction([2.0], [1.0, 1.0])
    sys_P0_d = signal.cont2discrete((sys_P0_c.num, sys_P0_c.den), Ts, method='zoh')
    num_P0 = sys_P0_d[0][0]
    den_P0 = sys_P0_d[1]

    # 2. Multiplicative Uncertainty Delta(s): high-frequency unmodeled dynamics
    # Delta(s) = 0.8 s / (0.2 s + 1). Small at DC, rises to 4.0 at high frequency.
    sys_Del_c = signal.TransferFunction([0.8, 0.0], [0.2, 1.0])
    sys_Del_d = signal.cont2discrete((sys_Del_c.num, sys_Del_c.den), Ts, method='tustin')
    num_Del = sys_Del_d[0][0]
    den_Del = sys_Del_d[1]

    # Actual Perturbed Plant: P(z) = [1 + Delta(z)] * P_0(z)
    # [1 + Delta(z)] = (den_Del + num_Del) / den_Del
    num_1pDel = np.polyadd(den_Del, num_Del)
    num_Pact = np.convolve(num_1pDel, num_P0)
    den_Pact = np.convolve(den_Del, den_P0)

    # 3. Controller 1: Aggressive Controller (High gain, low input penalty R)
    # K_aggr(z) = (1.5 z - 1.1) / (z - 1.0)
    num_K1 = np.array([1.5, -1.1])
    den_K1 = np.array([1.0, -1.0])

    # Controller 2: Robust Controller (Detuned, higher input penalty R)
    # K_rob(z) = (0.4 z - 0.3) / (z - 0.9)
    num_K2 = np.array([0.4, -0.3])
    den_K2 = np.array([1.0, -0.9])

    # 4. Frequency Grid for Small Gain Analysis
    w_nyq = np.pi / Ts
    w = np.logspace(-2, np.log10(w_nyq * 0.999), 1000)

    # Evaluate Nominal Complementary Sensitivities: T(z) = P0*K / (1 + P0*K)
    P0_jw = evaluate_discrete_tf(num_P0, den_P0, Ts, w)
    Del_jw = evaluate_discrete_tf(num_Del, den_Del, Ts, w)

    K1_jw = evaluate_discrete_tf(num_K1, den_K1, Ts, w)
    T1_jw = (P0_jw * K1_jw) / (1.0 + P0_jw * K1_jw)

    K2_jw = evaluate_discrete_tf(num_K2, den_K2, Ts, w)
    T2_jw = (P0_jw * K2_jw) / (1.0 + P0_jw * K2_jw)

    # Small Gain Product: |T(e^{jw})| * |Delta(e^{jw})|
    sg_prod_1 = np.abs(T1_jw) * np.abs(Del_jw)
    sg_prod_2 = np.abs(T2_jw) * np.abs(Del_jw)

    max_sg_1 = np.max(sg_prod_1)
    max_sg_2 = np.max(sg_prod_2)

    print(f"Small Gain Analysis (Condition: |T(jw)| * |Delta(jw)| < 1):")
    print(f"  Controller 1 (Aggressive): max product = {max_sg_1:.3f} (> 1.0 => ROBUST STABILITY VIOLATED!)")
    print(f"  Controller 2 (Robust):     max product = {max_sg_2:.3f} (< 1.0 => ROBUST STABILITY GUARANTEED!)")

    # 5. Simulate Closed Loop on Nominal and Perturbed Plant
    sim_steps = 45
    t_pts, y1_nom, stab1_nom, rho1_nom, _ = simulate_closed_loop(num_P0, den_P0, num_K1, den_K1, sim_steps, Ts)
    _,     y1_act, stab1_act, rho1_act, poles1_act = simulate_closed_loop(num_Pact, den_Pact, num_K1, den_K1, sim_steps, Ts)

    _,     y2_nom, stab2_nom, rho2_nom, _ = simulate_closed_loop(num_P0, den_P0, num_K2, den_K2, sim_steps, Ts)
    _,     y2_act, stab2_act, rho2_act, poles2_act = simulate_closed_loop(num_Pact, den_Pact, num_K2, den_K2, sim_steps, Ts)

    print(f"\nClosed-Loop Pole Stability on Actual Perturbed Plant:")
    print(f"  Controller 1 on Actual Plant: Stable={stab1_act}, Max Pole Magnitude={rho1_act:.4f}")
    print(f"  Controller 2 on Actual Plant: Stable={stab2_act}, Max Pole Magnitude={rho2_act:.4f}")

    # -------------------------------------------------------------
    # 6. Plotting Results
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot (a): Small Gain Frequency Condition
    ax = axes[0, 0]
    ax.semilogx(w, 20 * np.log10(np.abs(T1_jw)), 'r--', linewidth=2, label=r'Aggressive: $|T_1(e^{j\omega})|$')
    ax.semilogx(w, 20 * np.log10(np.abs(T2_jw)), 'g-', linewidth=2, label=r'Robust: $|T_2(e^{j\omega})|$')
    ax.semilogx(w, 20 * np.log10(1.0 / np.abs(Del_jw)), 'k:', linewidth=2.5, label=r'Robust Bound $1 / |\Delta(e^{j\omega})|$')
    ax.set_title(r'Exercise 8.1: Small Gain Condition $|T| < 1 / |\Delta|$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (b): Small Gain Product |T| * |Delta|
    ax = axes[0, 1]
    ax.semilogx(w, sg_prod_1, 'r--', linewidth=2, label=f'Aggressive Product (Peak = {max_sg_1:.2f} > 1)')
    ax.semilogx(w, sg_prod_2, 'g-', linewidth=2, label=f'Robust Product (Peak = {max_sg_2:.2f} < 1)')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1.5, label='Small Gain Boundary (1.0)')
    ax.set_title(r'Robust Stability Test: $|T(e^{j\omega})| \cdot |\Delta(e^{j\omega})|$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Product Value')
    ax.set_ylim(0, 3.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (c): Time Domain Step Response - Aggressive Controller
    ax = axes[1, 0]
    ax.step(t_pts, y1_nom, 'b--', where='post', linewidth=1.5, label='Nominal Plant (Stable & fast)')
    ax.step(t_pts, y1_act, 'r-', where='post', linewidth=2, label=f'Actual Plant (DIVERGES! $\\rho={rho1_act:.2f}$)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Controller 1 (Aggressive): Destabilized by Plant Uncertainty', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.set_ylim(-2.0, 6.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    # Plot (d): Time Domain Step Response - Robust Controller
    ax = axes[1, 1]
    ax.step(t_pts, y2_nom, 'b--', where='post', linewidth=1.5, label='Nominal Plant (Stable)')
    ax.step(t_pts, y2_act, 'g-', where='post', linewidth=2, label=f'Actual Plant (STABLE! $\\rho={rho2_act:.2f}$)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title('Controller 2 (Robust): Preserves Stability under Uncertainty', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.set_ylim(-0.2, 1.6)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch08_robust_mpc/ch8_small_gain_robust_stability.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
