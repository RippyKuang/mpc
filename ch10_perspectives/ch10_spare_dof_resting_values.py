"""
Chapter 10: Perspectives - Section 10.1.1
Spare Degrees of Freedom & Ideal Resting Values in MPC

This script demonstrates:
1. Plant with more inputs than outputs (nu = 3, ny = 2), creating an input null space.
2. Static least-squares optimization for economic ideal resting values u_des.
3. Augmented dynamic MPC objective penalizing deviations ||u(k) - u_des||_{R_s}^2.
4. Comparison of:
   (a) Standard MPC (no resting value penalty, Rs = 0): inputs drift arbitrarily in null space.
   (b) Resting-value guided MPC (Rs > 0): uses fast expensive actuator dynamically,
       then smoothly transitions steady-state load to cheap baseload actuator.
   (c) Plant-model mismatch scenario: trade-off between resting value tracking and output offset.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SpareDofMPCSystem:
    def __init__(self):
        # Continuous-time state-space system (2 states, 3 inputs, 2 outputs)
        # States: x1 = temp, x2 = holdup
        # Inputs: u1 = expensive electric heat, u2 = cheap steam, u3 = coolant
        self.Ts = 0.5  # sampling time (min)
        
        # Discrete dynamics: x(k+1) = A x(k) + B u(k), y(k) = C x(k)
        self.A = np.array([
            [0.85, 0.05],
            [0.02, 0.90]
        ])
        self.B = np.array([
            [0.30, 0.40, -0.25],
            [0.15, 0.20,  0.10]
        ])
        self.C = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Steady-state gain matrix: S_inf = C (I - A)^(-1) B
        I = np.eye(2)
        self.S_inf = self.C @ np.linalg.inv(I - self.A) @ self.B
        
        # Economic weighting on resting inputs:
        # u1 (electric) has weight 10.0, u2 (steam) has weight 1.0, u3 (coolant) has weight 5.0
        self.W_u = np.diag([10.0, 1.0, 5.0])
        
    def compute_ideal_resting_values(self, y_sp):
        """
        Solves static least-squares:
        min ||W_u u_s||_2^2   s.t.  S_inf @ u_s = y_sp
        Analytical solution via Lagrange multipliers:
        u_s = W_u^(-2) S_inf^T (S_inf W_u^(-2) S_inf^T)^(-1) y_sp
        """
        W_inv2 = np.linalg.inv(self.W_u @ self.W_u)
        M = self.S_inf @ W_inv2 @ self.S_inf.T
        u_des = W_inv2 @ self.S_inf.T @ np.linalg.inv(M) @ y_sp
        return u_des

class ConstrainedMPCSolver:
    def __init__(self, sys, Hp=15, Hu=5, Rs_weight=0.0):
        self.sys = sys
        self.Hp = Hp
        self.Hu = Hu
        self.Rs_weight = Rs_weight
        
        self.Q = np.diag([100.0, 100.0])
        self.R = np.eye(3) * 0.5
        self.Rs = np.eye(3) * Rs_weight
        
        self.u_min = np.array([-2.0, -2.0, -2.0])
        self.u_max = np.array([5.0, 5.0, 5.0])
        self.du_max = np.array([1.5, 1.5, 1.5])
        
        self.u_last = np.zeros(3)
        
    def solve(self, x_curr, y_sp, u_des):
        n_vars = 3 * self.Hu
        du0 = np.zeros(n_vars)
        
        bounds = [(-self.du_max[ch], self.du_max[ch]) for _ in range(self.Hu) for ch in range(3)]
        
        def cost_func(du_vec):
            du_mat = du_vec.reshape((self.Hu, 3))
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
                
                err_y = y_pred - y_sp
                cost += err_y.T @ self.Q @ err_y
                
                if self.Rs_weight > 0.0:
                    err_u = u_run - u_des
                    cost += err_u.T @ self.Rs @ err_u
                    
            return cost
            
        cons = []
        for k in range(self.Hu):
            def make_min_con(step_k, ch):
                return lambda du_vec: (self.u_last[ch] + np.sum(du_vec.reshape((self.Hu, 3))[:step_k+1, ch])) - self.u_min[ch]
            def make_max_con(step_k, ch):
                return lambda du_vec: self.u_max[ch] - (self.u_last[ch] + np.sum(du_vec.reshape((self.Hu, 3))[:step_k+1, ch]))
            for ch in range(3):
                cons.append({'type': 'ineq', 'fun': make_min_con(k, ch)})
                cons.append({'type': 'ineq', 'fun': make_max_con(k, ch)})
                
        res = minimize(cost_func, du0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 25, 'ftol': 1e-3})
        du_opt = res.x[:3] if res.success else np.zeros(3)
        self.u_last = np.clip(self.u_last + du_opt, self.u_min, self.u_max)
        return self.u_last

def simulate_system(sys, Rs_weight=0.0, plant_mismatch=False, n_steps=60):
    solver = ConstrainedMPCSolver(sys, Hp=15, Hu=5, Rs_weight=Rs_weight)
    
    # Simulation trajectory
    t_axis = np.arange(n_steps) * sys.Ts
    y_sp = np.array([2.0, 1.5])
    u_des = sys.compute_ideal_resting_values(y_sp)
    
    # Plant matrices (with optional mismatch)
    B_plant = sys.B * 1.25 if plant_mismatch else sys.B
    
    x = np.zeros(2)
    y_hist = np.zeros((n_steps, 2))
    u_hist = np.zeros((n_steps, 3))
    
    for k in range(n_steps):
        # After step 30, change setpoint
        if k >= 30:
            y_sp_curr = np.array([3.0, 1.0])
        else:
            y_sp_curr = y_sp
        u_des_curr = sys.compute_ideal_resting_values(y_sp_curr)
        
        u = solver.solve(x, y_sp_curr, u_des_curr)
        y = sys.C @ x
        
        y_hist[k] = y
        u_hist[k] = u
        
        # Plant update
        x = sys.A @ x + B_plant @ u
        
    return t_axis, y_hist, u_hist, u_des

def run_spare_dof_experiments():
    print("=" * 70)
    print("Chapter 10: Section 10.1.1 - Spare Degrees of Freedom & Ideal Resting Values")
    print("=" * 70)
    
    sys = SpareDofMPCSystem()
    print("1. Steady-State Gain Matrix S(inf) (2 outputs, 3 inputs):")
    print(np.round(sys.S_inf, 4))
    
    y_target = np.array([2.0, 1.5])
    u_des = sys.compute_ideal_resting_values(y_target)
    print(f"2. For y_sp = {y_target}, ideal resting values u_des:")
    print(f"   u1 (expensive electric) = {u_des[0]:.4f}")
    print(f"   u2 (cheap steam)        = {u_des[1]:.4f}")
    print(f"   u3 (coolant)            = {u_des[2]:.4f}")
    print(f"   Check S(inf) @ u_des: {np.round(sys.S_inf @ u_des, 4)}")
    
    print("3. Running Case A: Standard MPC (Rs = 0, unguided inputs)...")
    res_A = simulate_system(sys, Rs_weight=0.0)
    
    print("4. Running Case B: Resting-Value Guided MPC (Rs = 2.0)...")
    res_B = simulate_system(sys, Rs_weight=2.0)
    
    print("5. Running Case C: Plant Mismatch (+25% gain) with Resting Values...")
    res_C = simulate_system(sys, Rs_weight=2.0, plant_mismatch=True)
    
    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    t_axis = res_A[0]
    
    # Row 0: Case A (Standard MPC)
    ax0 = axes[0, 0]
    ax0.plot(t_axis, res_A[1][:, 0], 'r-', linewidth=2.0, label='y1 (Temp)')
    ax0.plot(t_axis, res_A[1][:, 1], 'b-', linewidth=2.0, label='y2 (Flow)')
    ax0.plot([0, 15, 15, 30], [2.0, 2.0, 3.0, 3.0], 'r:', label='y1 SP')
    ax0.plot([0, 15, 15, 30], [1.5, 1.5, 1.0, 1.0], 'b:', label='y2 SP')
    ax0.set_ylabel('Outputs', fontweight='bold')
    ax0.set_title('Case A: Standard MPC (Rs = 0) - Outputs', fontweight='bold')
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.legend(loc='lower right', fontsize=8)
    
    ax1 = axes[0, 1]
    ax1.plot(t_axis, res_A[2][:, 0], 'r--', label='u1 (Expensive Electric)')
    ax1.plot(t_axis, res_A[2][:, 1], 'g-', label='u2 (Cheap Steam)')
    ax1.plot(t_axis, res_A[2][:, 2], 'b-.', label='u3 (Coolant)')
    ax1.set_ylabel('Inputs', fontweight='bold')
    ax1.set_title('Case A: Arbitrary Steady-State Input Distribution', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Row 1: Case B (Resting Guided)
    ax2 = axes[1, 0]
    ax2.plot(t_axis, res_B[1][:, 0], 'r-', linewidth=2.0, label='y1 (Temp)')
    ax2.plot(t_axis, res_B[1][:, 1], 'b-', linewidth=2.0, label='y2 (Flow)')
    ax2.set_ylabel('Outputs', fontweight='bold')
    ax2.set_title('Case B: Ideal Resting Values (Rs = 2.0) - Outputs', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8)
    
    ax3 = axes[1, 1]
    ax3.plot(t_axis, res_B[2][:, 0], 'r--', linewidth=2.0, label='u1 (Electric -> 0)')
    ax3.plot(t_axis, res_B[2][:, 1], 'g-', linewidth=2.0, label='u2 (Steam takes load)')
    ax3.plot(t_axis, res_B[2][:, 2], 'b-.', linewidth=1.5, label='u3 (Coolant)')
    ax3.set_ylabel('Inputs', fontweight='bold')
    ax3.set_title('Case B: Economic Re-allocation (u1 driven to near zero)', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)
    
    # Row 2: Case C (Plant Mismatch)
    ax4 = axes[2, 0]
    ax4.plot(t_axis, res_C[1][:, 0], 'r-', linewidth=2.0, label='y1 (Temp)')
    ax4.plot(t_axis, res_C[1][:, 1], 'b-', linewidth=2.0, label='y2 (Flow)')
    ax4.set_xlabel('Time (min)', fontweight='bold')
    ax4.set_ylabel('Outputs', fontweight='bold')
    ax4.set_title('Case C: Plant Mismatch (+25% gain) - Offset Vulnerability', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='lower right', fontsize=8)
    
    ax5 = axes[2, 1]
    ax5.plot(t_axis, res_C[2][:, 0], 'r--', label='u1 (Electric)')
    ax5.plot(t_axis, res_C[2][:, 1], 'g-', label='u2 (Steam)')
    ax5.plot(t_axis, res_C[2][:, 2], 'b-.', label='u3 (Coolant)')
    ax5.set_xlabel('Time (min)', fontweight='bold')
    ax5.set_ylabel('Inputs', fontweight='bold')
    ax5.set_title('Case C: Inputs under Mismatch Trade-off', fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch10_perspectives/ch10_spare_dof_resting_values.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> Plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    run_spare_dof_experiments()
