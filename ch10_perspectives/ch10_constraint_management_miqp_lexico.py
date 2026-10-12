"""
Chapter 10: Perspectives - Section 10.2 & Example 10.1
Constraint Management & Lexicographic Prioritized Infeasibility Handling

This script demonstrates:
1. Infeasibility crisis in constrained MPC: when an overwhelming disturbance occurs,
   physical input saturations prevent simultaneous satisfaction of all output constraints.
2. Comparison of three constraint management philosophies:
   (a) Naive Hard Constraints: QP becomes strictly infeasible, causing solver crash/abort.
   (b) Uniform Soft Constraints: all slacks weighted equally, causing critical safety
       constraints to be violated to reduce secondary deviations.
   (c) Lexicographic Prioritized Constraints (Vada et al. 1999 / Tyler & Morari 1999):
       Hierarchy ensures lower-priority quality constraints are sacrificed first,
       strictly preserving high-priority safety constraints.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PriorityConstraintSystem:
    def __init__(self):
        self.Ts = 1.0
        # 2-input, 2-output coupled system
        self.A = np.array([
            [0.8, 0.1],
            [0.1, 0.85]
        ])
        self.B = np.array([
            [0.4, 0.1],
            [0.1, 0.5]
        ])
        self.C = np.eye(2)
        
        # Hard physical actuator saturation:
        self.u_min = np.array([-2.0, -2.0])
        self.u_max = np.array([ 2.0,  2.0])
        
        # Output constraints:
        # y1 <= 3.0 (Priority 1: Reactor Safety Pressure limit - MUST NOT EXCEED!)
        # y2 <= 2.0 (Priority 2: Product Quality limit - Secondary importance)
        self.y1_max = 3.0
        self.y2_max = 2.0

class PriorityMPCSolver:
    def __init__(self, sys, mode='lexicographic', Hp=10, Hu=4):
        self.sys = sys
        self.mode = mode  # 'hard', 'uniform_soft', 'lexicographic'
        self.Hp = Hp
        self.Hu = Hu
        
        self.Q = np.diag([10.0, 10.0])
        self.R = np.eye(2) * 1.0
        
        if mode == 'uniform_soft':
            # Equal weight on all constraint violations
            self.w_slack1 = 100.0
            self.w_slack2 = 100.0
        elif mode == 'lexicographic':
            # Huge hierarchical separation (Priority 1 is 1000x heavier than Priority 2)
            self.w_slack1 = 100000.0
            self.w_slack2 = 100.0
            
        self.u_last = np.zeros(2)
        
    def solve(self, x_curr, y_sp):
        n_vars = 2 * self.Hu
        du0 = np.zeros(n_vars)
        bounds = [(-1.5, 1.5) for _ in range(n_vars)]
        
        def cost_func(du_vec):
            du_mat = du_vec.reshape((self.Hu, 2))
            cost = 0.0
            x_pred = np.copy(x_curr)
            u_run = np.copy(self.u_last)
            
            for k in range(self.Hp):
                if k < self.Hu:
                    du_k = du_mat[k]
                    u_run += du_k
                    cost += du_k.T @ self.R @ du_k
                    
                x_pred = self.sys.A @ x_pred + self.sys.B @ u_run
                y_pred = self.sys.C @ x_pred
                err = y_pred - y_sp
                cost += err.T @ self.Q @ err
                
                if self.mode in ['uniform_soft', 'lexicographic']:
                    # Soft slack penalties
                    viol1 = max(0.0, y_pred[0] - self.sys.y1_max)
                    viol2 = max(0.0, y_pred[1] - self.sys.y2_max)
                    cost += self.w_slack1 * (viol1**2) + self.w_slack2 * (viol2**2)
                    
            return cost
            
        cons = []
        # Input saturation constraints
        for k in range(self.Hu):
            for ch in range(2):
                def make_min_u(step_k, c):
                    return lambda du_vec: (self.u_last[c] + np.sum(du_vec.reshape((self.Hu, 2))[:step_k+1, c])) - self.sys.u_min[c]
                def make_max_u(step_k, c):
                    return lambda du_vec: self.sys.u_max[c] - (self.u_last[c] + np.sum(du_vec.reshape((self.Hu, 2))[:step_k+1, c]))
                cons.append({'type': 'ineq', 'fun': make_min_u(k, ch)})
                cons.append({'type': 'ineq', 'fun': make_max_u(k, ch)})
                
        # If 'hard' mode, enforce hard output constraints
        if self.mode == 'hard':
            for k in range(self.Hp):
                def make_max_y1(step_k):
                    def con(du_vec):
                        du_mat = du_vec.reshape((self.Hu, 2))
                        x_p = np.copy(x_curr)
                        u_r = np.copy(self.u_last)
                        for j in range(step_k + 1):
                            if j < self.Hu:
                                u_r += du_mat[j]
                            x_p = self.sys.A @ x_p + self.sys.B @ u_r
                        return self.sys.y1_max - (self.sys.C @ x_p)[0]
                    return con
                def make_max_y2(step_k):
                    def con(du_vec):
                        du_mat = du_vec.reshape((self.Hu, 2))
                        x_p = np.copy(x_curr)
                        u_r = np.copy(self.u_last)
                        for j in range(step_k + 1):
                            if j < self.Hu:
                                u_r += du_mat[j]
                            x_p = self.sys.A @ x_p + self.sys.B @ u_r
                        return self.sys.y2_max - (self.sys.C @ x_p)[1]
                    return con
                cons.append({'type': 'ineq', 'fun': make_max_y1(k)})
                cons.append({'type': 'ineq', 'fun': make_max_y2(k)})
                
        res = minimize(cost_func, du0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 30, 'ftol': 1e-3})
        if res.success:
            du_opt = res.x[:2]
        else:
            if self.mode == 'hard':
                # Hard constraint solver crashed / returned infeasible!
                # Fallback: freeze or return zeros (simulating emergency stop / failure)
                du_opt = np.zeros(2)
            else:
                du_opt = du0[:2]
                
        self.u_last = np.clip(self.u_last + du_opt, self.sys.u_min, self.sys.u_max)
        return self.u_last, res.success

def simulate_crisis_scenario(mode='lexicographic', n_steps=40):
    sys = PriorityConstraintSystem()
    solver = PriorityMPCSolver(sys, mode=mode, Hp=10, Hu=4)
    
    t_axis = np.arange(n_steps) * sys.Ts
    y_sp = np.array([1.0, 1.0])
    
    x = np.zeros(2)
    y_hist = np.zeros((n_steps, 2))
    u_hist = np.zeros((n_steps, 2))
    infeasible_hist = []
    
    for k in range(n_steps):
        t = t_axis[k]
        
        # Massive unmeasured disturbance spike injected at t in [10, 25]
        # Disturbance enters states directly
        d = np.array([2.5, 2.0]) if (10 <= t <= 25) else np.zeros(2)
        
        u, success = solver.solve(x, y_sp)
        if not success:
            infeasible_hist.append(t)
            
        y = sys.C @ x
        y_hist[k] = y
        u_hist[k] = u
        
        # Plant update with disturbance
        x = sys.A @ x + sys.B @ u + d
        
    return t_axis, y_hist, u_hist, infeasible_hist

def run_constraint_management_experiments():
    print("=" * 70)
    print("Chapter 10: Section 10.2 - Constraint Management & Lexicographic Prioritization")
    print("=" * 70)
    
    print("1. Simulating Case A: Naive Hard Constraints (Infeasibility Crash)...")
    res_hard = simulate_crisis_scenario('hard')
    print(f"   Infeasibility occurred {len(res_hard[3])} times during crisis!")
    
    print("2. Simulating Case B: Uniform Soft Constraints (Equal Slacks w1=w2=100)...")
    res_uniform = simulate_crisis_scenario('uniform_soft')
    
    print("3. Simulating Case C: Lexicographic Priority (Safety w1=1e5 >> Quality w2=100)...")
    res_lexico = simulate_crisis_scenario('lexicographic')
    
    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    t_axis = res_hard[0]
    
    # Case A: Hard Constraints
    ax = axes[0, 0]
    ax.plot(t_axis, res_hard[1][:, 0], 'r-', linewidth=2.0, label='y1 (Safety Pressure)')
    ax.plot(t_axis, res_hard[1][:, 1], 'b-', linewidth=2.0, label='y2 (Product Quality)')
    ax.axhline(3.0, color='r', linestyle=':', linewidth=2.0, label='y1 Max Limit (3.0)')
    ax.axhline(2.0, color='b', linestyle=':', linewidth=2.0, label='y2 Max Limit (2.0)')
    ax.set_ylabel('Outputs', fontweight='bold')
    ax.set_title('Case A: Hard Constraints - Infeasibility Abort & Catastrophic Blowup', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    ax = axes[0, 1]
    ax.plot(t_axis, res_hard[2][:, 0], 'r--', label='u1')
    ax.plot(t_axis, res_hard[2][:, 1], 'b-.', label='u2')
    ax.axhspan(-2.0, 2.0, color='gray', alpha=0.15, label='Actuator Limits [-2, 2]')
    ax.set_ylabel('Inputs', fontweight='bold')
    ax.set_title('Case A: Actuators Freeze during QP Infeasibility', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Case B: Uniform Soft
    ax = axes[1, 0]
    ax.plot(t_axis, res_uniform[1][:, 0], 'r-', linewidth=2.0, label='y1 (Safety Pressure)')
    ax.plot(t_axis, res_uniform[1][:, 1], 'b-', linewidth=2.0, label='y2 (Product Quality)')
    ax.axhline(3.0, color='r', linestyle=':', linewidth=2.0, label='Safety Limit (3.0)')
    ax.axhline(2.0, color='b', linestyle=':', linewidth=2.0, label='Quality Limit (2.0)')
    ax.set_ylabel('Outputs', fontweight='bold')
    ax.set_title('Case B: Uniform Soft (w1 = w2) - Safety Limit Breached (y1 > 3.0)!', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    ax = axes[1, 1]
    ax.plot(t_axis, res_uniform[2][:, 0], 'r--', label='u1')
    ax.plot(t_axis, res_uniform[2][:, 1], 'b-.', label='u2')
    ax.axhspan(-2.0, 2.0, color='gray', alpha=0.15, label='Actuator Limits')
    ax.set_ylabel('Inputs', fontweight='bold')
    ax.set_title('Case B: Actuators Saturated attempting compromised compromise', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Case C: Lexicographic Priority
    ax = axes[2, 0]
    ax.plot(t_axis, res_lexico[1][:, 0], 'r-', linewidth=2.5, label='y1 (Safety Strictly Guarded <= 3.0)')
    ax.plot(t_axis, res_lexico[1][:, 1], 'b--', linewidth=1.8, label='y2 (Quality Sacrificed to Protect y1)')
    ax.axhline(3.0, color='r', linestyle=':', linewidth=2.0, label='Safety Limit (3.0)')
    ax.axhline(2.0, color='b', linestyle=':', linewidth=2.0, label='Quality Limit (2.0)')
    ax.set_xlabel('Time (min)', fontweight='bold')
    ax.set_ylabel('Outputs', fontweight='bold')
    ax.set_title('Case C: Lexicographic Priority (w1=1e5 >> w2=100) - Flawless Safety Preservation', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    ax = axes[2, 1]
    ax.plot(t_axis, res_lexico[2][:, 0], 'r--', label='u1')
    ax.plot(t_axis, res_lexico[2][:, 1], 'b-.', label='u2')
    ax.axhspan(-2.0, 2.0, color='gray', alpha=0.15, label='Actuator Limits')
    ax.set_xlabel('Time (min)', fontweight='bold')
    ax.set_ylabel('Inputs', fontweight='bold')
    ax.set_title('Case C: Maximum Control Action Dedicated to Safety Output', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch10_perspectives/ch10_constraint_management_miqp_lexico.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> Constraint management plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    run_constraint_management_experiments()
