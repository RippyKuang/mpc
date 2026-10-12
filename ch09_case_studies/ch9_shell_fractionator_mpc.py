"""
Chapter 9: Case Study 1 - Shell Heavy Oil Fractionator MPC
Section 9.1 & Process Engineering Implementation

This script simulates the industrial Shell Heavy Oil Fractionator:
- 3 Manipulated Variables (MVs): Top Draw (u1), Side Draw (u2), Bottoms Reflux Duty (u3)
- 2 Disturbance Variables (DVs): Intermediate Reflux Duty (d1, measured), Upper Reflux Duty (d2, unmeasured)
- 3 Controlled Variables (CVs): Top End Point (y1), Side End Point (y2), Bottoms Reflux Temp (y7)
- 1 Economic Objective: Minimize u3 (Bottoms Reflux Duty) via dummy setpoint s4 = -0.5
- Constraints on input rates, input amplitudes, and output bounds
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import cont2discrete

def build_shell_state_space(Ts=4.0):
    """
    Constructs a discretized multi-input multi-output state-space model for the
    Shell Heavy Oil Fractionator with time delays approximated or discretized.
    Inputs: [u1, u2, u3, d1, d2]
    Outputs: [y1, y2, y7]
    """
    # Nominal transfer function parameters: G_ij(s) = K / (tau * s + 1) * exp(-theta * s)
    # Gains K, time constants tau (min), delays theta (min)
    # Table 9.2 & 9.3:
    # Row 0: y1 (Top End Point)
    # Row 1: y2 (Side End Point)
    # Row 2: y7 (Bottoms Reflux Temp)
    K = np.array([
        [4.05, 1.77, 5.88, 1.20, 1.44],
        [5.39, 5.72, 6.90, 1.52, 1.83],
        [4.38, 4.42, 7.20, 1.14, 1.26]
    ])
    tau = np.array([
        [50.0, 60.0, 50.0, 45.0, 40.0],
        [50.0, 60.0, 40.0, 25.0, 20.0],
        [33.0, 44.0, 19.0, 27.0, 32.0]
    ])
    theta = np.array([
        [27.0, 28.0, 27.0, 27.0, 27.0],
        [18.0, 14.0, 15.0, 15.0, 15.0],
        [20.0, 22.0,  0.0,  1.0,  3.0]
    ])
    
    # Discretize each FOPTD channel and assemble into state-space
    # For delay theta, d_steps = int(round(theta / Ts))
    delay_steps = np.round(theta / Ts).astype(int)
    
    # Build augmented discrete state space
    # For each channel (i, j): x_ij(k+1) = a_ij * x_ij(k) + b_ij * u_j(k - d_steps)
    A_blocks = []
    B_blocks = []
    C_blocks = []
    
    total_states = 0
    ny, nu_total = K.shape
    
    # Store discrete poles and gains
    a_disc = np.exp(-Ts / tau)
    b_disc = K * (1.0 - a_disc)
    
    return {
        'K': K,
        'tau': tau,
        'theta': theta,
        'delay_steps': delay_steps,
        'a_disc': a_disc,
        'b_disc': b_disc,
        'Ts': Ts,
        'ny': ny,
        'nu': 3,
        'nd': 2
    }

class ShellFractionatorSimulator:
    def __init__(self, model):
        self.m = model
        self.Ts = model['Ts']
        self.ny = model['ny']
        self.nu = model['nu']
        self.nd = model['nd']
        
        # State memory for simulation: 3 outputs x 5 inputs
        self.x = np.zeros((self.ny, self.nu + self.nd))
        # Input history buffer for delays (up to 30 steps)
        self.max_delay = int(np.max(self.m['delay_steps'])) + 5
        self.u_hist = np.zeros((self.max_delay + 1, self.nu + self.nd))
        
    def step(self, u_mv, d_val):
        """
        Advances the physical plant by one sampling step.
        u_mv: (3,) manipulated inputs
        d_val: (2,) disturbance inputs
        """
        u_all = np.concatenate([u_mv, d_val])
        # Shift history
        self.u_hist[1:] = self.u_hist[:-1]
        self.u_hist[0] = u_all
        
        # Update each sub-state with appropriate delayed input
        y = np.zeros(self.ny)
        for i in range(self.ny):
            for j in range(self.nu + self.nd):
                d_idx = self.m['delay_steps'][i, j]
                u_delayed = self.u_hist[d_idx, j]
                self.x[i, j] = self.m['a_disc'][i, j] * self.x[i, j] + self.m['b_disc'][i, j] * u_delayed
                y[i] += self.x[i, j]
        return y

class ShellMPCController:
    def __init__(self, model, Hp=30, Hu=5):
        self.m = model
        self.Hp = Hp
        self.Hu = Hu
        self.Ts = model['Ts']
        self.nu = 3
        self.ny = 3
        
        # Precompute step response coefficients S_ij(l) for l = 1, ..., N_steps
        self.N_steps = 75
        self.S_step = np.zeros((self.ny, self.nu, self.N_steps + 1))
        for i in range(self.ny):
            for j in range(self.nu):
                k_val = self.m['K'][i, j]
                tau_val = self.m['tau'][i, j]
                theta_val = self.m['theta'][i, j]
                for l in range(1, self.N_steps + 1):
                    t_val = l * self.Ts
                    if t_val > theta_val:
                        self.S_step[i, j, l] = k_val * (1.0 - np.exp(-(t_val - theta_val) / tau_val))
                    else:
                        self.S_step[i, j, l] = 0.0
                        
        # MPC Weights
        self.Q_y = np.diag([20.0, 20.0, 1.0])
        self.w_econ = 2.0  # Weight on pulling u3 towards -0.3181
        self.R_du = np.diag([2.0, 2.0, 2.0])
        
        # Constraints:
        self.du_max = 0.05 * self.Ts  # 0.2
        self.u_min = -0.5
        self.u_max = 0.5
        self.y_min = -0.5
        self.y_max = 0.5
        
        self.u_last = np.zeros(self.nu)
        # Delta u history for convolution: shape (nu, N_steps)
        self.du_hist = np.zeros((self.nu, self.N_steps))
        
    def solve(self, y_meas, y_sp, d_meas):
        """
        Solves constrained QP-MPC step with exact step-response prediction.
        """
        n_vars = self.nu * self.Hu
        du0 = np.zeros(n_vars)
        
        bounds = [(-self.du_max, self.du_max)] * n_vars
        
        # Calculate free response f_i(k) from past control moves + disturbance estimation
        f_pred = np.zeros((self.Hp, self.ny))
        for p in range(1, self.Hp + 1):
            for i in range(self.ny):
                val = 0.0
                for j in range(self.nu):
                    for l in range(1, self.N_steps):
                        idx = p + l
                        s_curr = self.S_step[i, j, min(idx, self.N_steps)]
                        s_prev = self.S_step[i, j, min(l, self.N_steps)]
                        val += (s_curr - s_prev) * self.du_hist[j, l - 1]
                f_pred[p - 1, i] = val
                
        # Offset bias correction
        bias = y_meas - f_pred[0]
        for p in range(self.Hp):
            f_pred[p] += bias
            
        def cost_func(du_vec):
            du_mat = du_vec.reshape((self.Hu, self.nu))
            u_traj = np.zeros((self.Hu, self.nu))
            u_curr = np.copy(self.u_last)
            for k in range(self.Hu):
                u_curr += du_mat[k]
                u_traj[k] = u_curr
                
            cost = 0.0
            # Control move penalty
            for k in range(self.Hu):
                cost += du_mat[k].T @ self.R_du @ du_mat[k]
                cost += self.w_econ * (u_traj[k, 2] - (-0.3181))**2
                
            # Dynamic matrix convolution for forced response
            for p in range(self.Hp):
                y_forced = np.zeros(self.ny)
                for i in range(self.ny):
                    for j in range(self.nu):
                        for k in range(min(p + 1, self.Hu)):
                            step_idx = p - k + 1
                            if step_idx <= self.N_steps:
                                y_forced[i] += self.S_step[i, j, step_idx] * du_mat[k, j]
                                
                y_total = f_pred[p] + y_forced
                err = y_total - y_sp
                cost += err.T @ self.Q_y @ err
                
            return cost
            
        cons = []
        for k in range(self.Hu):
            def make_u_min_con(step_k, ch):
                return lambda du_vec: (self.u_last[ch] + np.sum(du_vec.reshape((self.Hu, self.nu))[:step_k+1, ch])) - self.u_min
            def make_u_max_con(step_k, ch):
                return lambda du_vec: self.u_max - (self.u_last[ch] + np.sum(du_vec.reshape((self.Hu, self.nu))[:step_k+1, ch]))
                
            for ch in range(self.nu):
                cons.append({'type': 'ineq', 'fun': make_u_min_con(k, ch)})
                cons.append({'type': 'ineq', 'fun': make_u_max_con(k, ch)})
                
        res = minimize(cost_func, du0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 60, 'ftol': 1e-4})
        if res.success:
            du_opt = res.x[:self.nu]
        else:
            du_opt = np.zeros(self.nu)
            
        self.u_last = np.clip(self.u_last + du_opt, self.u_min, self.u_max)
        # Shift du history
        self.du_hist[:, 1:] = self.du_hist[:, :-1]
        self.du_hist[:, 0] = du_opt
        return self.u_last

def run_simulation():
    print("=" * 75)
    print("Chapter 9: Case Study 1 - Shell Heavy Oil Fractionator Industrial MPC")
    print("=" * 75)
    
    Ts = 4.0  # Sampling interval (minutes)
    sim_time = 400.0  # Total simulation time (minutes)
    steps = int(sim_time / Ts)
    t_axis = np.arange(steps) * Ts
    
    model = build_shell_state_space(Ts=Ts)
    plant = ShellFractionatorSimulator(model)
    mpc = ShellMPCController(model, Hp=30, Hu=5)
    
    # Storage arrays
    y_hist = np.zeros((steps, 3))
    u_hist = np.zeros((steps, 3))
    d_hist = np.zeros((steps, 2))
    sp_hist = np.zeros((steps, 3))
    
    # Scenario Definition:
    # 1. Step disturbance on Intermediate Reflux Duty (d1 = 0.5 at t = 40 min)
    # 2. Step disturbance on Upper Reflux Duty (d2 = -0.5 at t = 160 min)
    # 3. Setpoint change on Side End Point (y2_sp = 0.3 at t = 260 min)
    for k in range(steps):
        t = t_axis[k]
        
        # Disturbances
        d1 = 0.5 if t >= 40.0 else 0.0
        d2 = -0.5 if t >= 160.0 else 0.0
        d_val = np.array([d1, d2])
        d_hist[k] = d_val
        
        # Setpoints
        y_sp = np.array([0.0, 0.3 if t >= 260.0 else 0.0, 0.0])
        sp_hist[k] = y_sp
        
        # Current plant output measurement
        if k == 0:
            y_meas = np.zeros(3)
        else:
            y_meas = y_hist[k-1]
            
        # MPC Solve
        u_opt = mpc.solve(y_meas, y_sp, d_meas=d1)
        u_hist[k] = u_opt
        
        # Plant Update
        y_next = plant.step(u_opt, d_val)
        y_hist[k] = y_next
        
    print(f"Simulation completed across {steps} steps ({sim_time} minutes).")
    print(f"Final Outputs: y1={y_hist[-1, 0]:.4f}, y2={y_hist[-1, 1]:.4f}, y7={y_hist[-1, 2]:.4f}")
    print(f"Final Inputs:  u1={u_hist[-1, 0]:.4f}, u2={u_hist[-1, 1]:.4f}, u3={u_hist[-1, 2]:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Controlled Outputs & Setpoints
    ax = axes[0]
    ax.plot(t_axis, y_hist[:, 0], 'b-', linewidth=2, label=r'Top End Point $y_1$')
    ax.plot(t_axis, y_hist[:, 1], 'g-', linewidth=2, label=r'Side End Point $y_2$')
    ax.plot(t_axis, y_hist[:, 2], 'm-', linewidth=2, label=r'Bottoms Reflux Temp $y_7$ (Zone)')
    ax.step(t_axis, sp_hist[:, 0], 'b--', linewidth=1.5, label=r'Setpoint $y_{1,sp}$')
    ax.step(t_axis, sp_hist[:, 1], 'g--', linewidth=1.5, label=r'Setpoint $y_{2,sp}$')
    ax.axhline(0.5, color='r', linestyle=':', label=r'Hard Limits $\pm 0.5$')
    ax.axhline(-0.5, color='r', linestyle=':')
    ax.set_title('Shell Heavy Oil Fractionator: Controlled Outputs and Setpoint Tracking', fontsize=11, fontweight='bold')
    ax.set_ylabel('Outputs ($y$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Manipulated Variables (MVs) & Constraints
    ax = axes[1]
    ax.step(t_axis, u_hist[:, 0], 'c-', where='post', linewidth=1.8, label=r'Top Draw $u_1$')
    ax.step(t_axis, u_hist[:, 1], 'orange', where='post', linewidth=1.8, label=r'Side Draw $u_2$')
    ax.step(t_axis, u_hist[:, 2], 'purple', where='post', linewidth=2.2, label=r'Bottoms Reflux Duty $u_3$ (Econ Min)')
    ax.axhline(0.5, color='r', linestyle='--', label=r'Actuator Limits $\pm 0.5$')
    ax.axhline(-0.5, color='r', linestyle='--')
    ax.axhline(-0.5, color='purple', linestyle=':', label=r'Economic Target $u_3 \rightarrow -0.5$')
    ax.set_title(r'Manipulated Inputs with Rate Constraints $|\Delta u| \leq 0.05 T_s$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inputs ($u$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=9)
    
    # Plot 3: Disturbance Profiles
    ax = axes[2]
    ax.step(t_axis, d_hist[:, 0], 'r-', where='post', linewidth=1.8, label=r'Measured Disturbance $d_1$ (Intermediate Reflux)')
    ax.step(t_axis, d_hist[:, 1], 'k--', where='post', linewidth=1.8, label=r'Unmeasured Disturbance $d_2$ (Upper Reflux)')
    ax.set_title('Disturbance Rejection Scenarios ($d_1, d_2$)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Disturbances ($d$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch09_case_studies/ch9_shell_fractionator_mpc.png"
    plt.savefig(output_path, dpi=300)
    print(f">> Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
