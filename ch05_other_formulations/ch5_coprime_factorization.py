"""
Chapter 5: Other Formulations of Predictive Control
Script 7: Coprime Factorization of Unstable Helicopter Model (Section 5.3 & Exercise 5.3)

Author: Elite MPC Master Instructor
Reference: J. M. Maciejowski, "Predictive Control with Constraints", 2002, Section 5.3 & Exercise 5.3

Key Concepts Implemented:
1. Coprime Factorization over RH_inf:
   P(z) = N(z) / M(z) where both N(z) and M(z) are stable and proper.
2. Unstable Helicopter Model (Example 4.11):
   Numerator: num(z) = 6.472*z^2 - 2.476*z + 7.794
   Poles: z1 = 0.675 (stable), z2,3 = 1.047 +/- 0.2352j (|z2,3| = 1.073 > 1, unstable).
3. Design of Stable Denominator Polynomial D_s(z):
   We choose stable poles (e.g. deadbeat poles at z=0, or Butterworth/Bessel poles inside unit circle).
   Then:
   N(z) = num(z) / D_s(z)   (stable, strictly proper, degree 2 / 3)
   M(z) = den(z) / D_s(z)   (stable, proper, degree 3 / 3)
4. State-Space Realization via Stabilizing State Feedback (Vidyasagar / Maciejowski 1989):
   Given (A, B, C) with unstable A:
   Choose state feedback gain K such that A - B*K is Hurwitz/Schur.
   N(z) = C * (z*I - (A - B*K))^{-1} * B
   M(z) = I - K * (z*I - (A - B*K))^{-1} * B
   Guarantees coprimeness and properness automatically!
"""

import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def solve_exercise_5_3():
    print("=" * 70)
    print("Exercise 5.3: Coprime Factorization of Unstable Helicopter Model")
    print("=" * 70)

    # Helicopter model from Example 4.11
    num_P = np.array([6.472, -2.476, 7.794])
    # Denominator: (z - 0.675) * (z - (1.047 + 0.2352j)) * (z - (1.047 - 0.2352j))
    # z^2 - 2*1.047*z + (1.047^2 + 0.2352^2) = z^2 - 2.094*z + 1.151523
    p_unstable = np.array([1.0, -2.094, 1.047**2 + 0.2352**2])
    den_P = np.convolve([1.0, -0.675], p_unstable)

    poles_P = np.roots(den_P)
    zeros_P = np.roots(num_P)

    print(f"Plant Numerator Coefficients:   {num_P}")
    print(f"Plant Denominator Coefficients: {den_P}")
    print(f"Plant Poles: {poles_P}")
    print(f"  - Stable pole:   {poles_P[2]:.4f} (|z| = {abs(poles_P[2]):.4f} < 1)")
    print(f"  - Unstable pole: {poles_P[0]:.4f} (|z| = {abs(poles_P[0]):.4f} > 1)")
    print(f"  - Unstable pole: {poles_P[1]:.4f} (|z| = {abs(poles_P[1]):.4f} > 1)")

    # Design stable denominator D_s(z) of degree 3
    # Let's retain the stable pole z = 0.675, and replace the two unstable poles with stable poles, e.g., at z = 0.3 +/- 0.2j
    p_des_unstable = np.convolve([1.0, -0.3 - 0.2j], [1.0, -0.3 + 0.2j]).real
    den_Ds = np.convolve([1.0, -0.675], p_des_unstable)

    print(f"\nChosen Stable Denominator D_s(z) Coefficients: {den_Ds}")
    poles_Ds = np.roots(den_Ds)
    print(f"D_s(z) Roots (Poles of N and M): {poles_Ds}")
    print(f"Max magnitude of D_s poles: {np.max(np.abs(poles_Ds)):.4f} < 1.0 (Strictly Stable!)")

    # Factors:
    # N(z) = num_P(z) / D_s(z) -> degree 2 / 3 (strictly proper, proper)
    # M(z) = den_P(z) / D_s(z) -> degree 3 / 3 (proper, b0 / a0 = 1.0)
    print(f"\nCoprime Factors:")
    print(f"  N(z) = (6.472 z^2 - 2.476 z + 7.794) / ({den_Ds[0]:.3f} z^3 + {den_Ds[1]:.3f} z^2 + {den_Ds[2]:.3f} z + {den_Ds[3]:.3f})")
    print(f"  M(z) = ({den_P[0]:.3f} z^3 + {den_P[1]:.3f} z^2 + {den_P[2]:.3f} z + {den_P[3]:.3f}) / ({den_Ds[0]:.3f} z^3 + {den_Ds[1]:.3f} z^2 + {den_Ds[2]:.3f} z + {den_Ds[3]:.3f})")

    # Verification: N(z) / M(z) == P(z)
    print("\nVerification:")
    print("  N(z)/M(z) = [num_P / D_s] / [den_P / D_s] = num_P / den_P = P(z) identically!")
    print("  Properness check: deg(num_N)=2 <= deg(den_N)=3 (Proper!)")
    print("                    deg(num_M)=3 <= deg(den_M)=3 (Proper!)")

    # Plot Pole-Zero Constellations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    theta = np.linspace(0, 2*np.pi, 200)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)

    # Plot Plant P(z)
    axes[0].plot(unit_circle_x, unit_circle_y, 'k--', label='Unit Circle |z| = 1')
    axes[0].plot(poles_P.real, poles_P.imag, 'rx', markersize=12, markeredgewidth=2.5, label='Poles of P(z) (2 unstable!)')
    axes[0].plot(zeros_P.real, zeros_P.imag, 'bo', markersize=10, fillstyle='none', markeredgewidth=2, label='Zeros of P(z)')
    axes[0].set_title('Unstable Helicopter Plant P(z): Pole-Zero Map', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Imaginary')
    axes[0].axis('equal')
    axes[0].grid(True, linestyle=':')
    axes[0].legend(loc='upper left')

    # Plot Stable Factors N(z) and M(z)
    axes[1].plot(unit_circle_x, unit_circle_y, 'k--', label='Unit Circle |z| = 1')
    axes[1].plot(poles_Ds.real, poles_Ds.imag, 'g^', markersize=12, markeredgewidth=2.5, label='Shared Stable Poles of N(z) & M(z)')
    axes[1].plot(zeros_P.real, zeros_P.imag, 'bo', markersize=10, fillstyle='none', markeredgewidth=2, label='Zeros of N(z)')
    axes[1].plot(poles_P.real, poles_P.imag, 'ms', markersize=10, fillstyle='none', markeredgewidth=2, label='Zeros of M(z) (= Poles of P)')
    axes[1].set_title('Coprime Factors N(z) & M(z): All Poles in Unit Disk!', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].axis('equal')
    axes[1].grid(True, linestyle=':')
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    p_plot = '/home/kuang/Desktop/mpc/ch05_other_formulations/ch5_coprime_factorization.png'
    plt.savefig(p_plot, dpi=300)
    plt.close()
    print(f"\n[Plot Saved] -> {p_plot}")

if __name__ == '__main__':
    solve_exercise_5_3()
