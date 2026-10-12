"""
ch4_gpc_simulation.py
=====================
Maciejowski (2002) Chapter 4: Generalized Predictive Control (GPC) from scratch.
Implements:
  1. CARIMA model simulation: A(q^-1) y(k) = B(q^-1) u(k-1) + C(q^-1)/Delta * e(k).
  2. Multi-step predictor via Diophantine recursion:
       y_hat(k+j|k) = G_j Delta u(k+j-1) + f_j(k)
  3. Dynamic matrix G construction (lower-triangular Toeplitz).
  4. Free response vector f(k) from past inputs and outputs.
  5. Unconstrained GPC closed-loop control law:
       Delta u(k) = [1, 0, ..., 0] * (G^T G + lambda * I)^(-1) G^T (w - f)
  6. Closed-loop simulation under colored noise and input step load disturbance.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def solve_diophantine_all(A, B, C, Hp):
    """
    Computes Diophantine polynomials E_j, F_j, G_j for j = 1, ..., Hp.
    C(q^-1) = E_j(q^-1) * A_tilde(q^-1) + q^-j * F_j(q^-1)
    where A_tilde = A * (1 - q^-1).
    E_j * B = G_j * C + q^-j * Gamma_j
    Returns:
      G_mat: (Hp, Hp) lower triangular matrix of dynamic step-response coefficients
      E_list, F_list, G_list
    """
    delta = np.array([1.0, -1.0])
    A_tilde = np.convolve(A, delta)
    
    E_list = []
    F_list = []
    g_coeffs = []  # Step response impulse weights g_0, g_1, ...
    
    # Solve C = E_j * A_tilde + q^-j * F_j
    for j in range(1, Hp + 1):
        rem = np.copy(C).astype(float)
        quot = np.zeros(j)
        for k in range(j):
            c_k = rem[k] if k < len(rem) else 0.0
            quot[k] = c_k / A_tilde[0]
            sub = np.zeros(k + len(A_tilde))
            sub[k:] = quot[k] * A_tilde
            # Subtract
            max_len = max(len(rem), len(sub))
            new_rem = np.zeros(max_len)
            new_rem[:len(rem)] += rem
            new_rem[:len(sub)] -= sub
            rem = new_rem
            
        F_j = rem[j:] if len(rem) > j else np.array([0.0])
        E_list.append(quot)
        F_list.append(F_j)
        
        # Multiply E_j * B to get impulse coefficients for G_j
        Ej_B = np.convolve(quot, B)
        # The j-th coefficient g_{j-1} is Ej_B[j-1]
        g_coeffs.append(Ej_B[j - 1] if len(Ej_B) >= j else 0.0)
        
    # Build G matrix (lower-triangular Toeplitz)
    G_mat = np.zeros((Hp, Hp))
    for i in range(Hp):
        for j in range(i + 1):
            G_mat[i, j] = g_coeffs[i - j]
            
    return G_mat, E_list, F_list, g_coeffs

def run_gpc_simulation():
    print("===================================================================")
    print("Running Generalized Predictive Control (GPC) Simulation...")
    print("===================================================================")
    
    # Benchmark discrete plant:
    # A(z^-1) = 1 - 1.5 z^-1 + 0.7 z^-2
    # B(z^-1) = 0.4 + 0.2 z^-1
    # C(z^-1) = 1 - 0.4 z^-1 (Colored noise polynomial / observer tuning)
    A = np.array([1.0, -1.5, 0.7])
    B = np.array([0.4, 0.2])
    C = np.array([1.0, -0.4])
    
    Hp = 10
    Hu = 3
    lambda_weight = 0.5
    
    # Solve Diophantine equations to build G matrix
    G_full, E_list, F_list, g_coeffs = solve_diophantine_all(A, B, C, Hp)
    G = G_full[:, :Hu]  # Dimension: (Hp, Hu)
    
    print("GPC Dynamic Matrix G (first 4 rows):\n", np.round(G[:4, :], 4))
    
    # Calculate GPC Gain vector:
    # Delta_U = (G^T G + lambda * I)^(-1) G^T (w - f)
    K_gpc_all = np.linalg.inv(G.T @ G + lambda_weight * np.eye(Hu)) @ G.T  # (Hu, Hp)
    k_gpc = K_gpc_all[0:1, :]  # (1, Hp)
    print("GPC Feedback Gain k_gpc:\n", np.round(k_gpc, 4))
    
    # Simulation settings
    sim_steps = 70
    r = np.zeros(sim_steps)
    r[10:40] = 1.0
    r[40:] = 2.5
    
    # Injected load disturbance at plant input at step 45
    d_load = np.zeros(sim_steps)
    d_load[45:] = -0.4
    
    # Plant buffers
    na = len(A) - 1
    nb = len(B)
    nc = len(C) - 1
    
    y_hist = np.zeros(sim_steps + 10)
    u_hist = np.zeros(sim_steps + 10)
    du_hist = np.zeros(sim_steps + 10)
    e_noise = np.random.normal(0, 0.05, sim_steps + 10)
    
    # Delta operator polynomial A_tilde = A * (1 - z^-1)
    delta = np.array([1.0, -1.0])
    A_tilde = np.convolve(A, delta)
    
    # Offset index for past values
    offset = 5
    
    for k in range(sim_steps):
        t = k + offset
        
        # 1. Simulate physical plant CARIMA difference equation:
        # y(t) = sum_{i=1}^na (-a_i y(t-i)) + sum_{j=0}^{nb-1} b_j (u(t-1-j) + d_load) + noise
        y_val = 0.0
        for i in range(1, len(A)):
            y_val -= A[i] * y_hist[t - i]
        for j in range(len(B)):
            u_in = u_hist[t - 1 - j] + d_load[k - j if k >= j else 0]
            y_val += B[j] * u_in
            
        # Add filtered noise: C(q^-1) e(t)
        noise_val = e_noise[t]
        for i in range(1, len(C)):
            noise_val += C[i] * e_noise[t - i]
            
        y_hist[t] = y_val
        
        # 2. Compute Free Response f(k+j|k) using CARIMA past history:
        # Optimal predictor identity:
        # y_hat(t+j|t) = G_j Delta u + F_j * (y_f) + ...
        # For simulation, we compute f via forward recursion with Delta u = 0 for future:
        y_sim = np.copy(y_hist[t - len(A_tilde) : t + 1])
        du_sim = np.copy(du_hist[t - len(B) : t + 1])
        
        f_vec = np.zeros(Hp)
        # Recursively project future y under Delta u(t+j) = 0:
        y_proj = list(y_hist[t - len(A_tilde) : t + 1])
        u_proj = u_hist[t - 1]
        
        for j in range(1, Hp + 1):
            # Difference equation with Delta u = 0:
            # y(t+j) = - sum_{i=1}^deg(A_tilde) A_tilde_i * y(t+j-i) + sum B_i * Delta u(past)
            y_next = 0.0
            for i in range(1, len(A_tilde)):
                idx = len(y_proj) - i
                if idx >= 0:
                    y_next -= A_tilde[i] * y_proj[idx]
            for m_b in range(len(B)):
                # Future Delta u are zero, only past Delta u contribute
                past_step = j - 1 - m_b
                if past_step < 0:
                    hist_idx = t + past_step
                    y_next += B[m_b] * du_hist[hist_idx]
            y_proj.append(y_next)
            f_vec[j - 1] = y_next
            
        # 3. GPC Control law:
        w_future = np.full(Hp, r[k])
        error_free = w_future - f_vec
        delta_u = (k_gpc @ error_free)[0]
        
        # 4. Apply control action
        du_hist[t] = delta_u
        u_hist[t] = u_hist[t - 1] + delta_u
        
    # Extract results
    time_axis = np.arange(sim_steps)
    y_out = y_hist[offset : offset + sim_steps]
    u_out = u_hist[offset : offset + sim_steps]
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    axes[0].plot(time_axis, r, 'k--', label='Set-point w(t)')
    axes[0].plot(time_axis, y_out, 'b-', linewidth=2, label='GPC Output y(t)')
    axes[0].axvline(x=45, color='r', linestyle=':', label='Input Load Disturbance (-0.4)')
    axes[0].set_title('Generalized Predictive Control (GPC) with CARIMA Noise Model')
    axes[0].set_ylabel('Output y')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].step(time_axis, u_out, 'g-', where='post', label='Control Signal u(t)')
    axes[1].set_title('GPC Control Effort u(t)')
    axes[1].set_xlabel('Sample Step k')
    axes[1].set_ylabel('Input u')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch04_transfer_functions/ch4_gpc_tracking_colored_noise.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved GPC simulation figure to: {plot_path}")

if __name__ == '__main__':
    run_gpc_simulation()
