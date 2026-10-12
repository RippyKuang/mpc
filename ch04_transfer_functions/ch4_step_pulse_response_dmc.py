"""
ch4_step_pulse_response_dmc.py
==============================
Maciejowski (2002) Chapter 4: Step and Pulse Response Models & DMC.
Implements:
  1. ss2step: Computes step response matrices S(0), ..., S(N) of a state-space system (Exercise 4.2a).
  2. step2pulse: Converts step response sequence S(t) to pulse response sequence H(t) (Exercise 4.1).
  3. step2ups & step2the: Assembles prediction matrices Upsilon and Theta from step response (Exercise 4.2b).
  4. DMC Controller: Dynamic Matrix Control closed-loop simulation with unmeasured step disturbance.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ss2step(A, B, C, D, M, N):
    """
    Computes step response matrices S(M), S(M+1), ..., S(N) of a state-space model.
    S_ij(t) is stored in S[i, j, t - M].
    Dimension: (m, ell, N - M + 1)
    """
    m, n = C.shape[0], A.shape[0]
    ell = B.shape[1]
    num_steps = N - M + 1
    S = np.zeros((m, ell, num_steps))
    
    # Simulate step response for each input channel independently
    for j in range(ell):
        u_step = np.zeros((ell, 1))
        u_step[j, 0] = 1.0
        x = np.zeros((n, 1))
        for t in range(N + 1):
            y = C @ x + D @ u_step
            if t >= M:
                S[:, j, t - M] = y.flatten()
            x = A @ x + B @ u_step
            
    return S

def step2pulse(S):
    """
    Converts step response matrices S(0), ..., S(N) to pulse response matrices H(0), ..., H(N).
    Formula:
      H(0) = S(0)
      H(t) = S(t) - S(t-1) for t >= 1
    (Exercise 4.1)
    """
    m, ell, N_plus_1 = S.shape
    H = np.zeros_like(S)
    H[:, :, 0] = S[:, :, 0]
    for t in range(1, N_plus_1):
        H[:, :, t] = S[:, :, t] - S[:, :, t - 1]
    return H

def step2ups(S, Hw, Hp):
    """
    Computes matrix Upsilon from step response sequence (Exercise 4.2b).
    Upsilon = [S(Hw)^T, S(Hw+1)^T, ..., S(Hp)^T]^T
    Dimension: (m * (Hp - Hw + 1), ell)
    """
    m, ell, _ = S.shape
    rows = []
    for t in range(Hw, Hp + 1):
        rows.append(S[:, :, t])
    return np.vstack(rows)

def step2the(S, Hw, Hp, Hu):
    """
    Computes matrix Theta (Dynamic Matrix A_dyn) from step response sequence (Exercise 4.2b).
    Theta has block lower-triangular Toeplitz structure:
      Theta[i, j] = S(Hw + i - j) if Hw + i - j >= 0 else 0
    Dimension: (m * (Hp - Hw + 1), ell * Hu)
    """
    m, ell, _ = S.shape
    num_p = Hp - Hw + 1
    Theta = np.zeros((m * num_p, ell * Hu))
    
    for i in range(num_p):
        pred_step = Hw + i
        for j in range(Hu):
            step_idx = pred_step - j
            if step_idx >= 0:
                Theta[i*m : (i+1)*m, j*ell : (j+1)*ell] = S[:, :, step_idx]
            else:
                Theta[i*m : (i+1)*m, j*ell : (j+1)*ell] = np.zeros((m, ell))
                
    return Theta

def run_dmc_simulation():
    print("===================================================================")
    print("Running DMC Closed-Loop Simulation with Step-Response Model...")
    print("===================================================================")
    
    # Define a stable second-order continuous plant: G(s) = 2 / (s^2 + 3s + 2)
    # Poles at s = -1, -2. DC Gain = 1.0.
    # State-space with Ts = 0.2s:
    # A_c = [[0, 1], [-2, -3]], B_c = [[0], [2]], C_c = [[1, 0]], D_c = [[0]]
    Ts = 0.2
    Ac = np.array([[0.0, 1.0], [-2.0, -3.0]])
    Bc = np.array([[0.0], [2.0]])
    Cc = np.array([[1.0, 0.0]])
    Dc = np.array([[0.0]])
    
    # Exact ZOH Discretization
    from scipy.linalg import expm
    M_mat = np.block([[Ac, Bc], [np.zeros((1, 2)), 0.0]])
    eM = expm(M_mat * Ts)
    Ad = eM[0:2, 0:2]
    Bd = eM[0:2, 2:3]
    Cd = Cc
    Dd = Dc
    
    # Step response truncation length N = 30 (corresponding to 6.0 seconds)
    N = 30
    S_seq = ss2step(Ad, Bd, Cd, Dd, 0, N)
    H_seq = step2pulse(S_seq)
    
    print(f"Computed step response S(0) = {S_seq[0, 0, 0]:.4f}, S(N) = {S_seq[0, 0, N]:.4f}")
    print(f"Computed pulse response H(0) = {H_seq[0, 0, 0]:.4f}, H(1) = {H_seq[0, 0, 1]:.4f}")
    
    # MPC tuning parameters
    Hw = 1
    Hp = 15
    Hu = 4
    lambda_r = 0.1
    
    Theta = step2the(S_seq, Hw, Hp, Hu)
    Upsilon = step2ups(S_seq, Hw, Hp)
    
    # DMC Controller Gain: Delta_U = (Theta^T Theta + lambda * I)^(-1) Theta^T E
    H_qp = Theta.T @ Theta + lambda_r * np.eye(Hu)
    K_dmc = np.linalg.inv(H_qp) @ Theta.T  # Dimension: (Hu, Hp)
    k_first = K_dmc[0:1, :]  # Dimension: (1, Hp)
    
    # Closed-loop simulation setup
    sim_steps = 60
    time = np.arange(sim_steps) * Ts
    r = np.ones(sim_steps)  # Set-point step to 1.0
    r[sim_steps // 2:] = 2.0  # Set-point step to 2.0 at half-time
    
    # Unmeasured load disturbance at plant input starting at t = 5.0s (step 25)
    d_input = np.zeros(sim_steps)
    d_input[25:] = -0.5
    
    # History buffers
    past_du = np.zeros(N)  # past Delta u: [Delta u(k-1), Delta u(k-2), ...]
    y_meas = np.zeros(sim_steps)
    u_applied = np.zeros(sim_steps)
    x_true = np.zeros((2, 1))
    u_prev = 0.0
    d_hat = 0.0  # bias disturbance estimate
    
    for k in range(sim_steps):
        # 1. Measure physical plant output
        u_eff = np.array([[u_prev + d_input[k]]])
        y_curr = (Cd @ x_true + Dd @ u_eff)[0, 0]
        y_meas[k] = y_curr
        
        # 2. Predicted model output at current step based on past moves:
        # y_model(k) = sum_{i=1}^N S(i) * past_du(i-1) (with u(-infty)=0)
        # DMC bias correction:
        # y_free(k+j|k) = S(j) * u(k-1) + sum_{i=1}^{N-j} (S(j+i) - S(i)) * past_du(i-1) + d_hat(k)
        # Or standard DMC shift calculation:
        # At step k, compute bias:
        y_free_k = np.sum([S_seq[0, 0, min(i+1, N)] * past_du[i] for i in range(N-1)])
        d_hat = y_curr - y_free_k
        
        # 3. Predict free response over horizon Hp
        y_free = np.zeros(Hp)
        for j in range(Hp):
            pred_step = j + 1
            # convolution of past moves shifted forward by pred_step
            past_sum = 0.0
            for i in range(N - 1):
                idx = min(pred_step + i + 1, N)
                past_sum += S_seq[0, 0, idx] * past_du[i]
            y_free[j] = past_sum + d_hat
            
        # 4. Reference trajectory over horizon
        r_future = np.full(Hp, r[k])
        e_vec = r_future - y_free
        
        # 5. Compute optimal move
        delta_u = (k_first @ e_vec)[0]
        u_curr = u_prev + delta_u
        u_applied[k] = u_curr
        
        # 6. Update past input moves buffer
        past_du[1:] = past_du[:-1]
        past_du[0] = delta_u
        u_prev = u_curr
        
        u_input_total = np.array([[u_curr + d_input[k]]])
        x_true = Ad @ x_true + Bd @ u_input_total
        
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    
    # Subplot 1: Step & Pulse Responses
    t_resp = np.arange(N + 1) * Ts
    axes[0].step(t_resp, S_seq[0, 0, :], 'b-', where='post', label='Step Response S(t)')
    axes[0].step(t_resp, H_seq[0, 0, :], 'r--', where='post', label='Pulse Response H(t)')
    axes[0].set_title('Plant Open-Loop Step and Pulse Response (FIR Truncation N=30)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    axes[0].legend()
    
    # Subplot 2: Closed-Loop Output Tracking
    axes[1].plot(time, r, 'k--', label='Set-point r(t)')
    axes[1].plot(time, y_meas, 'b-', linewidth=2, label='DMC Output y(t)')
    axes[1].axvline(x=25*Ts, color='r', linestyle=':', label='Input Disturbance Injected (-0.5)')
    axes[1].set_title('DMC Closed-Loop Step Tracking & Disturbance Rejection')
    axes[1].set_ylabel('Output y')
    axes[1].grid(True)
    axes[1].legend()
    
    # Subplot 3: Manipulated Variable u(t)
    axes[2].step(time, u_applied, 'g-', where='post', label='Manipulated Input u(t)')
    axes[2].set_title('Control Action u(t)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Input u')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch04_transfer_functions/ch4_dmc_simulation.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved DMC simulation figure to: {plot_path}")

if __name__ == '__main__':
    run_dmc_simulation()
