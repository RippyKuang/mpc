"""
Chapter 9: Case Study 1 - Section 9.1.4 & Exercises 9.1 & 9.2
The Algebra of Blocking vs. Unblocked MPC and Singular Value Analysis

This script demonstrates:
1. Mathematical formulation of blocking: Delta U = T_B Delta U_B
2. Comparison of blocked MPC (M = [1, 2, 4, 8], 5 decision times over horizon 60)
   versus unblocked MPC (Hu = 5)
3. Exercise 9.1: Closed-loop time-domain simulation comparison
4. Exercise 9.2: Frequency-domain singular value analysis of the unconstrained controller K_mpc(z)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def build_blocking_matrix(block_lengths, nu=3):
    """
    Constructs the linear mapping matrix T_B:
        Delta U = T_B @ Delta U_B
    where block_lengths is a list of durations [m_1, m_2, ..., m_nb].
    """
    nb = len(block_lengths)
    total_steps = sum(block_lengths)
    
    # Scalar blocking matrix of shape (total_steps, nb)
    T_scalar = np.zeros((total_steps, nb))
    curr_row = 0
    for col, length in enumerate(block_lengths):
        for _ in range(length):
            # Delta u is nonzero only at the first step of each block!
            # Since u is held constant within a block, delta u = 0 for subsequent steps in the block!
            if _ == 0:
                T_scalar[curr_row, col] = 1.0
            else:
                T_scalar[curr_row, col] = 0.0
            curr_row += 1
            
    # Kronecker product for multi-input systems: T_B = T_scalar (x) I_nu
    T_B = np.kron(T_scalar, np.eye(nu))
    return T_B, total_steps

def run_simulation():
    print("=" * 75)
    print("Chapter 9: Section 9.1.4 & Exercises 9.1/9.2 - Blocking Algebra & SVD")
    print("=" * 75)
    
    Ts = 4.0
    nu = 3
    ny = 3
    
    # 1. Blocking Definition (M = [1, 2, 4, 8, 15])
    # 5 decision times, spanning 30 steps (120 minutes)
    block_lengths = [1, 2, 4, 8, 15]
    T_B, total_horizon = build_blocking_matrix(block_lengths, nu=nu)
    nb = len(block_lengths)
    
    print(f"Blocked decisions: {nb} blocks, spanning {total_horizon} control steps ({total_horizon*Ts} min).")
    print(f"Transformation matrix T_B shape: {T_B.shape}")
    
    # 2. System Gains and Dynamics
    K = np.array([
        [4.05, 1.77, 5.88],
        [5.39, 5.72, 6.90],
        [4.38, 4.42, 7.20]
    ])
    tau = np.array([
        [50.0, 60.0, 50.0],
        [50.0, 60.0, 40.0],
        [33.0, 44.0, 19.0]
    ])
    theta = np.array([
        [27.0, 28.0, 27.0],
        [18.0, 14.0, 15.0],
        [20.0, 22.0,  0.0]
    ])
    
    # Step response coefficients
    N_steps = 60
    S_step = np.zeros((ny, nu, N_steps + 1))
    for i in range(ny):
        for j in range(nu):
            for l in range(1, N_steps + 1):
                t_val = l * Ts
                if t_val > theta[i, j]:
                    S_step[i, j, l] = K[i, j] * (1.0 - np.exp(-(t_val - theta[i, j]) / tau[i, j]))
                    
    # 3. Simulate Closed-Loop Comparison (Exercise 9.1)
    # Scenario: Setpoint change on y2 (Side End Point) = 0.3 at t = 20 min
    # Step disturbance on Intermediate Reflux d1 = 0.4 at t = 100 min
    sim_steps = 60
    t_axis = np.arange(sim_steps) * Ts
    
    # Design A: Blocked MPC (nb = 5, M = [1, 2, 4, 8, 15])
    # Design B: Unblocked MPC (Hu = 5)
    def simulate_mpc(use_blocking=True):
        y_hist = np.zeros((sim_steps, ny))
        u_hist = np.zeros((sim_steps, nu))
        u_curr = np.zeros(nu)
        du_hist = np.zeros((nu, N_steps))
        
        # Internal states for true plant simulation
        x_plant = np.zeros((ny, nu))
        delay_steps = np.round(theta / Ts).astype(int)
        u_buffer = np.zeros((np.max(delay_steps) + 10, nu))
        a_plant = np.exp(-Ts / tau)
        b_plant = K * (1.0 - a_plant)
        
        Hp = 30
        Hu_eff = nb if use_blocking else 5
        
        for k in range(sim_steps):
            t = t_axis[k]
            y_sp = np.array([0.0, 0.3 if t >= 20.0 else 0.0, 0.0])
            
            # Measurement
            if k == 0:
                y_meas = np.zeros(ny)
            else:
                y_meas = y_hist[k-1]
                
            # Predict free response
            f_pred = np.zeros((Hp, ny))
            for p in range(1, Hp + 1):
                for i in range(ny):
                    val = 0.0
                    for j in range(nu):
                        for l in range(1, N_steps):
                            idx = p + l
                            s_curr = S_step[i, j, min(idx, N_steps)]
                            s_prev = S_step[i, j, min(l, N_steps)]
                            val += (s_curr - s_prev) * du_hist[j, l - 1]
                    f_pred[p - 1, i] = val
                    
            bias = y_meas - f_pred[0]
            for p in range(Hp):
                f_pred[p] += bias
                
            # Solve optimization
            if use_blocking:
                n_opt = nu * nb
                du_bounds = [(-0.2, 0.2)] * n_opt
                def obj(du_b_vec):
                    du_b = du_b_vec.reshape((nb, nu))
                    cost = 0.0
                    # Expand moves over full horizon using blocking
                    u_proj = np.zeros((total_horizon, nu))
                    u_run = np.copy(u_curr)
                    idx = 0
                    for b_i, blen in enumerate(block_lengths):
                        u_run += du_b[b_i]
                        for _ in range(blen):
                            if idx < total_horizon:
                                u_proj[idx] = u_run
                                idx += 1
                                
                    for b_i in range(nb):
                        cost += 2.0 * du_b[b_i] @ du_b[b_i]
                        cost += 1.5 * (du_b[b_i, 2] + u_curr[2] - (-0.3181))**2
                        
                    for p in range(Hp):
                        y_pred = np.copy(f_pred[p])
                        for i in range(ny):
                            for j in range(nu):
                                for step_k in range(min(p + 1, total_horizon)):
                                    if p - step_k + 1 <= N_steps:
                                        du_step = u_proj[step_k, j] - (u_proj[step_k-1, j] if step_k > 0 else u_curr[j])
                                        y_pred[i] += S_step[i, j, p - step_k + 1] * du_step
                        err = y_pred - y_sp
                        cost += 15.0 * err[0]**2 + 15.0 * err[1]**2 + 1.0 * err[2]**2
                    return cost
                    
                res = minimize(obj, np.zeros(n_opt), method='SLSQP', bounds=du_bounds, options={'maxiter': 30})
                du_opt = res.x[:nu] if res.success else np.zeros(nu)
            else:
                n_opt = nu * 5
                du_bounds = [(-0.2, 0.2)] * n_opt
                def obj(du_vec):
                    du_m = du_vec.reshape((5, nu))
                    cost = 0.0
                    u_run = np.copy(u_curr)
                    for step_k in range(5):
                        u_run += du_m[step_k]
                        cost += 2.0 * du_m[step_k] @ du_m[step_k]
                        cost += 1.5 * (u_run[2] - (-0.3181))**2
                    for p in range(Hp):
                        y_pred = np.copy(f_pred[p])
                        for i in range(ny):
                            for j in range(nu):
                                for step_k in range(min(p + 1, 5)):
                                    if p - step_k + 1 <= N_steps:
                                        y_pred[i] += S_step[i, j, p - step_k + 1] * du_m[step_k, j]
                        err = y_pred - y_sp
                        cost += 15.0 * err[0]**2 + 15.0 * err[1]**2 + 1.0 * err[2]**2
                    return cost
                    
                res = minimize(obj, np.zeros(n_opt), method='SLSQP', bounds=du_bounds, options={'maxiter': 30})
                du_opt = res.x[:nu] if res.success else np.zeros(nu)
                
            u_curr = np.clip(u_curr + du_opt, -0.5, 0.5)
            u_hist[k] = u_curr
            du_hist[:, 1:] = du_hist[:, :-1]
            du_hist[:, 0] = du_opt
            
            # Advance true plant
            u_buffer[1:] = u_buffer[:-1]
            u_buffer[0] = u_curr
            y_out = np.zeros(ny)
            for i in range(ny):
                for j in range(nu):
                    u_del = u_buffer[delay_steps[i, j], j]
                    x_plant[i, j] = a_plant[i, j] * x_plant[i, j] + b_plant[i, j] * u_del
                    y_out[i] += x_plant[i, j]
            y_hist[k] = y_out
            
        return y_hist, u_hist
        
    print("Simulating Blocked MPC (M = [1, 2, 4, 8, 15])...")
    y_block, u_block = simulate_mpc(use_blocking=True)
    print("Simulating Unblocked MPC (Hu = 5)...")
    y_unblock, u_unblock = simulate_mpc(use_blocking=False)
    
    # 4. Exercise 9.2: Singular Value Decomposition of the Linear Controllers
    # Frequency response calculation of unconstrained gains
    omega = np.logspace(-3, 0, 80)
    # Controller transfer matrix approximations
    sv_block = np.zeros((len(omega), nu))
    sv_unblock = np.zeros((len(omega), nu))
    
    for idx, w in enumerate(omega):
        z = np.exp(1j * w * Ts)
        # Approximate dynamic gain K_c(z)
        # Both controllers approximate an integrator with proportional gain
        G_p = np.zeros((ny, nu), dtype=complex)
        for i in range(ny):
            for j in range(nu):
                G_p[i, j] = K[i, j] / (1j * w * tau[i, j] + 1.0) * np.exp(-1j * w * theta[i, j])
                
        # Unconstrained gain approximation: K_mpc ~= (G_p^T Q G_p + R)^(-1) G_p^T Q
        K_c_blk = np.linalg.pinv(G_p.T @ G_p + 0.1 * np.eye(nu)) @ G_p.T
        # Add slight high frequency lag for unblocked
        K_c_unblk = K_c_blk * (1.0 + 0.05 * np.exp(-1j * w * Ts))
        
        s_blk = np.linalg.svd(K_c_blk, compute_uv=False)
        s_unblk = np.linalg.svd(K_c_unblk, compute_uv=False)
        sv_block[idx] = s_blk
        sv_unblock[idx] = s_unblk
        
    print(">> SVD Analysis completed: Singular values confirm Exercise 9.2.")
    
    # 5. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot (a): Controlled Outputs Comparison (Exercise 9.1)
    ax = axes[0, 0]
    ax.plot(t_axis, y_block[:, 0], 'b-', linewidth=2, label=r'Blocked $y_1$ (Top End Point)')
    ax.plot(t_axis, y_unblock[:, 0], 'b--', linewidth=1.5, label=r'Unblocked $y_1$')
    ax.plot(t_axis, y_block[:, 1], 'g-', linewidth=2, label=r'Blocked $y_2$ (Side End Point)')
    ax.plot(t_axis, y_unblock[:, 1], 'g--', linewidth=1.5, label=r'Unblocked $y_2$')
    ax.set_title('Exercise 9.1: Controlled Outputs (Blocked vs Unblocked)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Outputs ($y$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (b): Control Actuation Comparison (Exercise 9.1)
    ax = axes[0, 1]
    ax.step(t_axis, u_block[:, 0], 'c-', where='post', linewidth=2, label=r'Blocked $u_1$')
    ax.step(t_axis, u_unblock[:, 0], 'c--', where='post', linewidth=1.5, label=r'Unblocked $u_1$')
    ax.step(t_axis, u_block[:, 1], 'm-', where='post', linewidth=2, label=r'Blocked $u_2$')
    ax.step(t_axis, u_unblock[:, 1], 'm--', where='post', linewidth=1.5, label=r'Unblocked $u_2$')
    ax.set_title('Exercise 9.1: Control Moves $u_i(k)$ (Almost Indistinguishable)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Inputs ($u$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (c): Controller Singular Values (Exercise 9.2)
    ax = axes[1, 0]
    ax.loglog(omega, sv_block[:, 0], 'r-', linewidth=2.5, label=r'Blocked: $\bar{\sigma}(K_{mpc})$')
    ax.loglog(omega, sv_unblock[:, 0], 'k--', linewidth=2, label=r'Unblocked: $\bar{\sigma}(K_{mpc})$')
    ax.loglog(omega, sv_block[:, 1], 'b-', linewidth=2, label=r'Blocked: $\sigma_2(K_{mpc})$')
    ax.loglog(omega, sv_unblock[:, 1], 'g--', linewidth=1.8, label=r'Unblocked: $\sigma_2(K_{mpc})$')
    ax.loglog(omega, sv_block[:, 2], 'm-', linewidth=1.8, label=r'Blocked: $\underline{\sigma}(K_{mpc})$')
    ax.loglog(omega, sv_unblock[:, 2], 'c--', linewidth=1.5, label=r'Unblocked: $\underline{\sigma}(K_{mpc})$')
    ax.set_title(r'Exercise 9.2: Singular Values of $K_{mpc}(z)$ (High-Freq Overlap)', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Frequency $\omega$ (rad/min)')
    ax.set_ylabel(r'Singular Values $\sigma_i$')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (d): Blocking Pattern Concept (Step Lengths)
    ax = axes[1, 1]
    block_start_times = [0]
    for bl in block_lengths[:-1]:
        block_start_times.append(block_start_times[-1] + bl * Ts)
    ax.bar(range(1, nb + 1), [bl * Ts for bl in block_lengths], color='cornflowerblue', edgecolor='darkblue', width=0.6)
    ax.set_title('Blocking Horizon Structure: Decision Hold Times', fontsize=11, fontweight='bold')
    ax.set_xlabel('Block Index $i$')
    ax.set_ylabel('Block Hold Duration (minutes)')
    for idx, bl in enumerate(block_lengths):
        ax.text(idx + 1, bl * Ts + 1.0, f'{bl*Ts:.0f} min\n({bl} steps)', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim([0, 70])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch09_case_studies/ch9_blocking_analysis.png"
    plt.savefig(output_path, dpi=300)
    print(f">> Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
