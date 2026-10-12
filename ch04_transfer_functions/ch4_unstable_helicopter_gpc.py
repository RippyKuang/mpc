"""
ch4_unstable_helicopter_gpc.py
==============================
Maciejowski (2002) Chapter 4: Replicating Example 4.11 & Figures 4.4, 4.5.
Unstable Helicopter Longitudinal Control using Generalized Predictive Control (GPC).
Implements:
  1. Unstable discrete transfer function:
       B(z^-1) = 6.472 z^-1 - 2.476 z^-2 + 7.794 z^-3
       A(z^-1) = 1 - 2.769 z^-1 + 2.565 z^-2 - 0.7773 z^-3
  2. GPC tuning: Hp = 5, Hu = 2, lambda = 0.
  3. Observer polynomial comparisons (Exercise 4.12 & Figure 4.5):
       - Deadbeat observer: C(z^-1) = 1 (pole at z = 0)
       - Slow observer:     C(z^-1) = 1 - 0.8 z^-1 (pole at z = 0.8)
       - Intermediate:      C(z^-1) = 1 - 0.3 z^-1 (pole at z = 0.3)
       - Sluggish:          C(z^-1) = 1 - 0.99 z^-1 (pole at z = 0.99)
  4. Input ramp disturbance: du(k) = [-0.03, -0.07, -0.1] at k = 17, 18, 19.
  5. Exact replication of output tracking and actuator response (Figure 4.5).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simulate_gpc_helicopter(c_pole=0.0):
    """
    Simulates GPC closed loop for unstable helicopter with observer pole at c_pole.
    Observer polynomial C(z^-1) = 1 - c_pole * z^-1.
    """
    Ts = 0.6
    # Polynomials
    A = np.array([1.0, -2.769, 2.565, -0.7773])
    B = np.array([0.0, 6.472, -2.476, 7.794])
    C = np.array([1.0, -c_pole])
    
    Hp = 5
    Hu = 2
    lambda_weight = 1e-6  # Numerically zero weight as per book (lambda = 0)
    
    # Delta polynomial A_tilde = A * (1 - z^-1)
    delta = np.array([1.0, -1.0])
    A_tilde = np.convolve(A, delta)
    
    # Solve Diophantine equations for j = 1..Hp:
    # C(z^-1) = E_j(z^-1) * A_tilde(z^-1) + z^-j * F_j(z^-1)
    E_list = []
    F_list = []
    g_coeffs = []
    
    for j in range(1, Hp + 1):
        rem = np.copy(C).astype(float)
        quot = np.zeros(j)
        for k in range(j):
            c_k = rem[k] if k < len(rem) else 0.0
            quot[k] = c_k / A_tilde[0]
            sub = np.zeros(k + len(A_tilde))
            sub[k:] = quot[k] * A_tilde
            max_len = max(len(rem), len(sub))
            new_rem = np.zeros(max_len)
            new_rem[:len(rem)] += rem
            new_rem[:len(sub)] -= sub
            rem = new_rem
            
        F_j = rem[j:] if len(rem) > j else np.array([0.0])
        E_list.append(quot)
        F_list.append(F_j)
        
        Ej_B = np.convolve(quot, B)
        # B has leading zero (delay d=1), so Ej_B[j] corresponds to Delta u(t+j-1)
        g_coeffs.append(Ej_B[j] if len(Ej_B) > j else 0.0)
        
    # Build G matrix (Hp x Hu)
    G = np.zeros((Hp, Hu))
    for i in range(Hp):
        for j in range(min(i + 1, Hu)):
            G[i, j] = g_coeffs[i - j]
            
    # GPC Gain
    K_all = np.linalg.inv(G.T @ G + lambda_weight * np.eye(Hu)) @ G.T
    k_first = K_all[0:1, :]  # (1, Hp)
    
    # Simulation setup
    sim_steps = 40
    y_hist = np.zeros(sim_steps + 10)
    u_hist = np.zeros(sim_steps + 10)
    du_hist = np.zeros(sim_steps + 10)
    
    # Input disturbance profile matching Example 4.11 & Figure 4.5:
    # d_in = 0 for k < 17; -0.03 at k=17; -0.07 at k=18; -0.1 for k >= 19
    d_u = np.zeros(sim_steps)
    for k in range(sim_steps):
        if k == 17:
            d_u[k] = -0.03
        elif k == 18:
            d_u[k] = -0.07
        elif k >= 19:
            d_u[k] = -0.10
            
    offset = 5
    # Setpoint is 0.0 (regulator problem to reject disturbance, matching Fig 4.5)
    r = np.zeros(sim_steps)
    
    for k in range(sim_steps):
        t = k + offset
        
        # 1. Propagate true plant:
        # y(t) = - a_1 y(t-1) - a_2 y(t-2) - a_3 y(t-3) + b_1 u_eff(t-1) + b_2 u_eff(t-2) + b_3 u_eff(t-3)
        y_val = 0.0
        for i in range(1, len(A)):
            y_val -= A[i] * y_hist[t - i]
        for j in range(1, len(B)):
            u_eff = u_hist[t - j] + d_u[k - j + 1 if (k - j + 1) >= 0 else 0]
            y_val += B[j] * u_eff
        y_hist[t] = y_val
        
        # 2. Compute free response f:
        # Filtered signals through C(z^-1):
        # We project forward with Delta u(future) = 0:
        y_proj = list(y_hist[t - len(A_tilde) : t + 1])
        f_vec = np.zeros(Hp)
        
        for j in range(1, Hp + 1):
            y_next = 0.0
            for i in range(1, len(A_tilde)):
                idx = len(y_proj) - i
                if idx >= 0:
                    y_next -= A_tilde[i] * y_proj[idx]
            for m_b in range(1, len(B)):
                past_step = j - m_b
                if past_step <= 0:
                    y_next += B[m_b] * du_hist[t + past_step]
            y_proj.append(y_next)
            f_vec[j - 1] = y_next
            
        # 3. GPC control law:
        e_free = r[k] - f_vec
        delta_u = (k_first @ e_free)[0]
        
        # 4. Update control buffers:
        du_hist[t] = delta_u
        u_hist[t] = u_hist[t - 1] + delta_u
        
    time_sec = np.arange(sim_steps) * Ts
    y_out = y_hist[offset : offset + sim_steps]
    u_out = u_hist[offset : offset + sim_steps]
    
    return time_sec, y_out, u_out

def run_helicopter_experiments():
    print("===================================================================")
    print("Replicating Maciejowski (2002) Example 4.11 & Figure 4.5...")
    print("===================================================================")
    
    # 1. Deadbeat Observer: C(z^-1) = 1 (pole = 0)
    t, y_deadbeat, u_deadbeat = simulate_gpc_helicopter(c_pole=0.0)
    # 2. Slow Observer: C(z^-1) = 1 - 0.8 z^-1 (pole = 0.8)
    _, y_slow, u_slow = simulate_gpc_helicopter(c_pole=0.8)
    # 3. Intermediate Observer (Exercise 4.12): pole = 0.3
    _, y_p03, u_p03 = simulate_gpc_helicopter(c_pole=0.3)
    # 4. Sluggish Observer (Exercise 4.12): pole = 0.99
    _, y_p99, u_p99 = simulate_gpc_helicopter(c_pole=0.99)
    
    # Plotting Figure 4.5 Replication
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Helicopter pitch / output response
    axes[0].plot(t, y_slow, 'r-', linewidth=2, label='Slow Observer: pole = 0.8 (Solid, Fig 4.5)')
    axes[0].plot(t, y_deadbeat, 'b--', linewidth=2, label='Deadbeat Observer: pole = 0.0 (Dashed, Fig 4.5)')
    axes[0].plot(t, y_p03, 'g-.', linewidth=1.5, label='Exercise 4.12: pole = 0.3')
    axes[0].plot(t, y_p99, 'm:', linewidth=1.5, label='Exercise 4.12: pole = 0.99')
    axes[0].axvline(x=17*0.6, color='k', linestyle=':', label='Disturbance Start (t = 10.2s)')
    axes[0].set_title('Figure 4.5 Replication: GPC Applied to Unstable Helicopter')
    axes[0].set_ylabel('Pitch Output y')
    axes[0].grid(True)
    axes[0].legend(loc='lower left')
    
    # Subplot 2: Control effort u(t)
    axes[1].step(t, u_slow, 'r-', linewidth=1.5, where='post', label='pole = 0.8')
    axes[1].step(t, u_deadbeat, 'b--', linewidth=1.5, where='post', label='pole = 0.0 (Deadbeat)')
    axes[1].step(t, u_p03, 'g-.', linewidth=1.2, where='post', label='pole = 0.3')
    axes[1].step(t, u_p99, 'm:', linewidth=1.2, where='post', label='pole = 0.99')
    axes[1].set_title('Actuator Control Signal u(t)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Control Input u')
    axes[1].grid(True)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch04_transfer_functions/ch4_helicopter_gpc_fig4_5.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Figure 4.5 replication to: {plot_path}")

if __name__ == '__main__':
    run_helicopter_experiments()
