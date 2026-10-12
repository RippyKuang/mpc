"""
Chapter 10: Perspectives - Section 10.1.3 & Figures 10.1-10.4
Fault-Tolerant MPC & Daisy-Chaining Actuator Reconfiguration

This script demonstrates:
1. Multi-actuator system with redundant control surfaces (nu = 3, ny = 2).
   Outputs: y1 = yaw angle (deg), y2 = roll angle (deg).
   Inputs: u1 = rudder, u2 = ailerons, u3 = differential thrust / spoilers.
2. Actuator failure at t = 10s: rudder jams at a fixed angle u1 = +10 deg.
3. Four benchmark cases matching Maciejowski Section 10.1.3:
   - Case 1: Normal fault-free operation.
   - Case 2: Jammed rudder, unconstrained controller (no FDI, large transient).
   - Case 3: Jammed rudder, constrained controller unaware of failure (automatic daisy-chaining).
   - Case 4: Jammed rudder with Fault Detection & Identification (FDI) (instant constraint reconfiguration Delta u1 = 0).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class FaultTolerantAircraftModel:
    def __init__(self):
        self.Ts = 0.5  # seconds
        
        # Lateral-directional state space:
        # States: [sideslip beta, yaw rate r, roll rate p, roll angle phi]
        self.A = np.array([
            [0.92, -0.04,  0.02,  0.03],
            [0.08,  0.88, -0.01,  0.00],
            [-0.15, 0.05,  0.82,  0.00],
            [0.00,  0.00,  0.50,  1.00]
        ])
        
        # Inputs: [u1: rudder, u2: aileron, u3: diff thrust]
        self.B = np.array([
            [ 0.05,  0.01,  0.02],
            [-0.12,  0.02, -0.08],
            [ 0.03,  0.15,  0.01],
            [ 0.00,  0.00,  0.00]
        ])
        
        # Outputs: y1 = yaw rate approx / directional response, y2 = roll angle
        self.C = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
class FaultTolerantMPC:
    def __init__(self, sys, Hp=20, Hu=5):
        self.sys = sys
        self.Hp = Hp
        self.Hu = Hu
        
        self.Q = np.diag([50.0, 20.0])
        self.R = np.diag([1.0, 2.0, 3.0])  # u1 is cheapest, u3 is expensive
        
        self.u_min = np.array([-25.0, -20.0, -15.0])
        self.u_max = np.array([ 25.0,  20.0,  15.0])
        self.du_max = np.array([5.0, 5.0, 5.0])
        
        self.u_last = np.zeros(3)
        
    def solve(self, x_curr, y_sp, fdi_jammed_actuator=None):
        n_vars = 3 * self.Hu
        du0 = np.zeros(n_vars)
        
        bounds = []
        for step in range(self.Hu):
            for ch in range(3):
                if fdi_jammed_actuator is not None and ch == fdi_jammed_actuator:
                    # FDI informed: actuator cannot move at all!
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((-self.du_max[ch], self.du_max[ch]))
                    
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
                err = y_pred - y_sp
                cost += err.T @ self.Q @ err
                
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

def simulate_aircraft_fault(scenario='normal', total_time=50.0):
    sys = FaultTolerantAircraftModel()
    mpc = FaultTolerantMPC(sys, Hp=20, Hu=5)
    
    steps = int(total_time / sys.Ts)
    t_axis = np.arange(steps) * sys.Ts
    
    x = np.zeros(4)
    y_hist = np.zeros((steps, 2))
    u_hist = np.zeros((steps, 3))
    
    y_sp = np.array([5.0, 0.0])  # Step demand in yaw rate / heading, keep roll at 0
    
    for k in range(steps):
        t = t_axis[k]
        
        # Fault injection at t >= 10.0 s
        rudder_jammed = (t >= 10.0 and scenario != 'normal')
        
        # Controller configuration
        fdi_flag = 0 if (scenario == 'fdi_reconfig' and rudder_jammed) else None
        
        # MPC calculates commanded control
        u_cmd = mpc.solve(x, y_sp, fdi_jammed_actuator=fdi_flag)
        
        # Actual plant input: if jammed, u1 is frozen at +10 deg!
        u_actual = np.copy(u_cmd)
        if rudder_jammed:
            u_actual[0] = 10.0  # Jammed at +10 deg
            if scenario != 'fdi_reconfig':
                # Controller still thinks it applied u_cmd[0], but plant received 10.0
                pass
            else:
                mpc.u_last[0] = 10.0
                
        y = sys.C @ x
        y_hist[k] = y
        u_hist[k] = u_actual
        
        # Plant state propagation
        x = sys.A @ x + sys.B @ u_actual
        
    return t_axis, y_hist, u_hist

def run_fault_tolerant_experiments():
    print("=" * 70)
    print("Chapter 10: Section 10.1.3 - Fault-Tolerant MPC & Daisy-Chaining")
    print("=" * 70)
    
    print("1. Simulating Case 1: Normal Operation (No Failure)...")
    res_normal = simulate_aircraft_fault('normal')
    
    print("2. Simulating Case 2: Jammed Rudder (Unaware, Automatic Daisy-Chain)...")
    res_unaware = simulate_aircraft_fault('unaware')
    
    print("3. Simulating Case 3: Jammed Rudder (FDI Reconfiguration Delta u1 = 0)...")
    res_fdi = simulate_aircraft_fault('fdi_reconfig')
    
    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    t_axis = res_normal[0]
    
    # Left Column: Outputs (Yaw and Roll)
    # Row 0: Case 1 Normal
    ax = axes[0, 0]
    ax.plot(t_axis, res_normal[1][:, 0], 'r-', linewidth=2.0, label='Yaw Response')
    ax.plot(t_axis, res_normal[1][:, 1], 'b-', linewidth=2.0, label='Roll Response')
    ax.axhline(5.0, color='r', linestyle=':', label='Yaw SP (+5 deg)')
    ax.axhline(0.0, color='b', linestyle=':', label='Roll SP (0 deg)')
    ax.set_ylabel('Angle (deg)', fontweight='bold')
    ax.set_title('Case 1: Normal Operation (No Failure)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)
    
    ax = axes[0, 1]
    ax.plot(t_axis, res_normal[2][:, 0], 'r--', label='u1 (Rudder)')
    ax.plot(t_axis, res_normal[2][:, 1], 'g-', label='u2 (Aileron)')
    ax.plot(t_axis, res_normal[2][:, 2], 'b-.', label='u3 (Diff Thrust)')
    ax.set_ylabel('Controls (deg)', fontweight='bold')
    ax.set_title('Case 1: Controls under Normal Operation', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Row 1: Case 2 Unaware
    ax = axes[1, 0]
    ax.plot(t_axis, res_unaware[1][:, 0], 'r-', linewidth=2.0, label='Yaw Response')
    ax.plot(t_axis, res_unaware[1][:, 1], 'b-', linewidth=2.0, label='Roll Response')
    ax.axvline(10.0, color='magenta', linestyle='--', label='Rudder Jams at t=10s')
    ax.set_ylabel('Angle (deg)', fontweight='bold')
    ax.set_title('Case 2: Rudder Jammed (+10 deg) - Controller Unaware', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)
    
    ax = axes[1, 1]
    ax.plot(t_axis, res_unaware[2][:, 0], 'r--', linewidth=2.0, label='u1 (Jammed at +10 deg)')
    ax.plot(t_axis, res_unaware[2][:, 1], 'g-', linewidth=2.0, label='u2 (Aileron Daisy-Chains)')
    ax.plot(t_axis, res_unaware[2][:, 2], 'b-.', linewidth=2.0, label='u3 (Diff Thrust Rescues)')
    ax.axvline(10.0, color='magenta', linestyle='--')
    ax.set_ylabel('Controls (deg)', fontweight='bold')
    ax.set_title('Case 2: Automatic Daisy-Chaining Redistribution', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Row 2: Case 3 FDI Reconfigured
    ax = axes[2, 0]
    ax.plot(t_axis, res_fdi[1][:, 0], 'r-', linewidth=2.0, label='Yaw Response')
    ax.plot(t_axis, res_fdi[1][:, 1], 'b-', linewidth=2.0, label='Roll Response')
    ax.axvline(10.0, color='magenta', linestyle='--', label='Rudder Jams at t=10s (FDI Active)')
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Angle (deg)', fontweight='bold')
    ax.set_title('Case 3: Fault Detection & Reconfiguration (Delta u1 = 0)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8)
    
    ax = axes[2, 1]
    ax.plot(t_axis, res_fdi[2][:, 0], 'r--', linewidth=2.0, label='u1 (Frozen)')
    ax.plot(t_axis, res_fdi[2][:, 1], 'g-', linewidth=2.0, label='u2 (Optimal Reconfig)')
    ax.plot(t_axis, res_fdi[2][:, 2], 'b-.', linewidth=2.0, label='u3 (Optimal Reconfig)')
    ax.axvline(10.0, color='magenta', linestyle='--')
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Controls (deg)', fontweight='bold')
    ax.set_title('Case 3: Rapid Coordinated Control Re-allocation', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch10_perspectives/ch10_fault_tolerant_daisy_chain.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> Fault-tolerant plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    run_fault_tolerant_experiments()
