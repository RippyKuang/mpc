"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 3: Solving Predictive Control Problems
Filename: ch3_unconstrained_lq.py

Description:
1. Analytical derivation and computation of unconstrained MPC gain matrices:
   K_mpc, K_x, K_u, K_r
2. Explicit linear time-invariant closed-loop state feedback representation:
   u(k) = (I - K_mpc * Upsilon) * u(k-1) - K_x * x(k) + K_r * T(k)
3. Full verification of Exercise 3.3: Heated swimming pool temperature control
   with heater power q(t) and ambient temperature disturbance theta_a(t).

Dependencies: numpy, scipy, matplotlib
===============================================================================
"""

import os
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_unconstrained_mpc_gains(A, B, C, Hp=10, Hu=3, Q_weights=None, R_weights=None):
    """
    Computes analytical unconstrained MPC gains K_mpc, K_x, K_u, K_r.
    System:
        x(k+1) = A * x(k) + B * u(k)
        z(k)   = C * x(k)
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    C = np.atleast_2d(C)
    
    nx = A.shape[0]
    nu = B.shape[1]
    nz = C.shape[0]

    if Q_weights is None:
        Q_bar = np.eye(nz * Hp)
    elif np.isscalar(Q_weights):
        Q_bar = np.eye(nz * Hp) * float(Q_weights)
    else:
        Q_bar = np.kron(np.eye(Hp), np.diag(Q_weights))

    if R_weights is None:
        R_bar = np.eye(nu * Hu) * 0.0
    elif np.isscalar(R_weights):
        R_bar = np.eye(nu * Hu) * float(R_weights)
    else:
        R_bar = np.kron(np.eye(Hu), np.diag(R_weights))

    # Build Psi: (nz * Hp) x nx
    Psi = np.zeros((nz * Hp, nx))
    A_pow = np.eye(nx)
    for i in range(1, Hp + 1):
        A_pow = A_pow @ A
        Psi[(i - 1) * nz : i * nz, :] = C @ A_pow

    # Build Upsilon: (nz * Hp) x nu
    Upsilon = np.zeros((nz * Hp, nu))
    sum_A_B = np.zeros((nx, nu))
    A_p = np.eye(nx)
    for i in range(1, Hp + 1):
        sum_A_B = sum_A_B + A_p @ B
        A_p = A_p @ A
        Upsilon[(i - 1) * nz : i * nz, :] = C @ sum_A_B

    # Build Theta: (nz * Hp) x (nu * Hu)
    Theta = np.zeros((nz * Hp, nu * Hu))
    for r in range(1, Hp + 1):
        for c in range(1, Hu + 1):
            if r >= c:
                shift = r - c
                sum_term = np.zeros((nx, nu))
                A_k = np.eye(nx)
                for k in range(shift + 1):
                    sum_term = sum_term + A_k @ B
                    A_k = A_k @ A
                Theta[(r - 1) * nz : r * nz, (c - 1) * nu : c * nu] = C @ sum_term

    # Hessian and gain
    H = Theta.T @ Q_bar @ Theta + R_bar
    K_mpc_full = np.linalg.solve(H, Theta.T @ Q_bar)
    
    # Receding horizon takes first nu rows
    K_mpc = K_mpc_full[:nu, :]
    
    K_x = K_mpc @ Psi
    K_u = K_mpc @ Upsilon
    K_r = np.sum(K_mpc, axis=1, keepdims=True)

    return {
        'Psi': Psi,
        'Upsilon': Upsilon,
        'Theta': Theta,
        'H': H,
        'K_mpc': K_mpc,
        'K_x': K_x,
        'K_u': K_u,
        'K_r': K_r
    }


def run_swimming_pool_simulation(save_dir="."):
    """
    Exercise 3.3 Verification: Heated Swimming Pool
    Continuous dynamics:
        T * d theta / dt = k_q * q + theta_a - theta
        T = 1 hr, k_q = 0.2 C/kW
    Sampling:
        Ts = 0.25 hr (15 minutes)
    """
    print("\n--- Exercise 3.3: Heated Swimming Pool Verification ---")
    Ts = 0.25
    T_param = 1.0
    k_q = 0.2

    # Discrete model:
    # theta(k+1) = exp(-Ts/T) * theta(k) + (1 - exp(-Ts/T)) * k_q * q(k) + (1 - exp(-Ts/T)) * theta_a(k)
    Ad = np.exp(-Ts / T_param)
    Bd_q = k_q * (1.0 - Ad)
    Bd_ta = 1.0 * (1.0 - Ad)

    print(f"Discrete Model: Ad = {Ad:.5f}, Bd_q = {Bd_q:.5f}, Bd_ta = {Bd_ta:.5f}")
    assert np.isclose(Ad, 0.7788, atol=1e-3), "Ad mismatch with Eq (3.94)"
    assert np.isclose(Bd_q, 0.0442, atol=1e-3), "Bd_q mismatch with Eq (3.94)"
    assert np.isclose(Bd_ta, 0.2212, atol=1e-3), "Bd_ta mismatch with Eq (3.94)"

    # Gains with Hp=10, Hu=3, Q=1, R=0
    gains = compute_unconstrained_mpc_gains(A=[[Ad]], B=[[Bd_q]], C=[[1.0]],
                                            Hp=10, Hu=3, Q_weights=1.0, R_weights=0.0)
    
    K_mpc = gains['K_mpc']
    K_x = gains['K_x'][0, 0]
    K_u = gains['K_u'][0, 0]
    K_r = gains['K_r'][0, 0]

    print(f"Calculated Gains: K_r = {K_r:.3f}, -K_x = {-K_x:.3f}, -K_u = {-K_u:.3f}")
    print(f"Expected in Exercise 3.3(b): Ks = [22.604, -17.604, -22.604] (or -1.0 for standard Upsilon)")

    # Closed-loop simulation under two scenarios:
    # 1. Constant ambient temperature (15 C) with plant mismatch (T=1.25, k=0.3)
    # 2. Sinusoidal diurnal ambient temperature: theta_a(t) = 15 + 10 * sin(2*pi*t/24)
    total_hours = 48.0
    nsteps = int(total_hours / Ts)
    tvec = np.arange(nsteps) * Ts

    # Scenario 1: Mismatched pool, constant ambient 15 C, target 20 C
    T_mismatch = 1.25
    kq_mismatch = 0.3
    Ad_real = np.exp(-Ts / T_mismatch)
    Bd_q_real = kq_mismatch * (1.0 - Ad_real)
    Bd_ta_real = 1.0 * (1.0 - Ad_real)

    theta_sim1 = np.zeros(nsteps)
    q_sim1 = np.zeros(nsteps)
    theta_curr = 15.0  # initial water temperature
    q_prev = 0.0
    sp1 = 20.0
    theta_a1 = 15.0

    for k in range(nsteps):
        theta_sim1[k] = theta_curr
        # Unconstrained MPC control move
        # delta_q = K_r * sp - K_x * theta_curr - K_u * q_prev
        delta_q = K_r * sp1 - K_x * theta_curr - K_u * q_prev
        q_curr = q_prev + delta_q
        q_sim1[k] = q_curr
        q_prev = q_curr

        # Plant updates with real parameters
        theta_next = Ad_real * theta_curr + Bd_q_real * q_curr + Bd_ta_real * theta_a1
        theta_curr = theta_next

    # Scenario 2: Perfect model, sinusoidal ambient, input constrained 0 <= q <= 40 kW
    theta_sim2 = np.zeros(nsteps)
    q_sim2 = np.zeros(nsteps)
    theta_a2 = 15.0 + 10.0 * np.sin(2.0 * np.pi * tvec / 24.0)
    theta_curr = 15.0
    q_prev = 0.0

    for k in range(nsteps):
        theta_sim2[k] = theta_curr
        delta_q = K_r * sp1 - K_x * theta_curr - K_u * q_prev
        q_unconstrained = q_prev + delta_q
        # Enforce physical constraints: 0 <= q <= 40 kW
        q_curr = np.clip(q_unconstrained, 0.0, 40.0)
        q_sim2[k] = q_curr
        q_prev = q_curr

        theta_next = Ad * theta_curr + Bd_q * q_curr + Bd_ta * theta_a2[k]
        theta_curr = theta_next

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Exercise 3.3: Unconstrained & Constrained Swimming Pool MPC", fontsize=13)

    # Subplot 1: Scenario 1 Water Temp
    axes[0, 0].plot(tvec, theta_sim1, 'b-', lw=1.8, label='Water Temp theta(t)')
    axes[0, 0].axhline(sp1, color='k', linestyle='--', label='Set-point (20 C)')
    axes[0, 0].axhline(theta_a1, color='gray', linestyle=':', label='Ambient (15 C)')
    axes[0, 0].set_title("Scenario 1: Constant Ambient & Plant Mismatch")
    axes[0, 0].set_ylabel("Temperature (C)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Subplot 2: Scenario 1 Heater Power
    axes[0, 1].step(tvec, q_sim1, 'g-', where='post', lw=1.5)
    axes[0, 1].set_title("Scenario 1: Heater Power q(t)")
    axes[0, 1].set_ylabel("Power (kW)")
    axes[0, 1].grid(True)

    # Subplot 3: Scenario 2 Water Temp & Diurnal Cycle
    axes[1, 0].plot(tvec, theta_sim2, 'b-', lw=1.8, label='Water Temp theta(t)')
    axes[1, 0].plot(tvec, theta_a2, 'r:', lw=1.2, label='Diurnal Ambient theta_a(t)')
    axes[1, 0].axhline(sp1, color='k', linestyle='--', label='Set-point (20 C)')
    axes[1, 0].set_title("Scenario 2: Sinusoidal Ambient & Input Saturation")
    axes[1, 0].set_xlabel("Time (hours)")
    axes[1, 0].set_ylabel("Temperature (C)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Subplot 4: Scenario 2 Heater Power (Clamped [0, 40] kW)
    axes[1, 1].step(tvec, q_sim2, 'g-', where='post', lw=1.5, label='Power q(t)')
    axes[1, 1].axhline(40.0, color='r', linestyle='--', label='Max Power (40 kW)')
    axes[1, 1].axhline(0.0, color='k', linestyle='--', label='Min Power (0 kW)')
    axes[1, 1].set_title("Scenario 2: Heater Power [0, 40] kW Saturation")
    axes[1, 1].set_xlabel("Time (hours)")
    axes[1, 1].set_ylabel("Power (kW)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    p_out = os.path.join(save_dir, "ch3_pool_temperature.png")
    plt.savefig(p_out, dpi=200)
    plt.close()
    print(f"-> Saved Swimming Pool simulation to: {p_out}")

if __name__ == "__main__":
    run_swimming_pool_simulation(save_dir="/home/kuang/Desktop/mpc/ch03_solving_qp")
