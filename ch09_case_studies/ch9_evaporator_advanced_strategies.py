"""
Chapter 9: Case Study 2 - Exercises 9.5 & 9.6
Advanced Evaporator Strategies: Zone Control & Event-Triggered Re-linearization

This script demonstrates:
1. Exercise 9.5: Zone control on Separator Level L2 in [0.5, 1.5] m (no setpoint)
   vs rigid setpoint tracking (L2 = 1.0 m), demonstrating reduced actuator wear on F2.
2. Exercise 9.6: Event-triggered adaptive re-linearization:
   Only re-linearize the internal model when product concentration |Delta X2| > 1.25% (5% of 25%),
   drastically reducing online Jacobian computations while preserving high control quality.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Import the nonlinear physics model from Script 4
from ch9_evaporator_nonlinear_mpc import NewellLeeEvaporator

class AdvancedEvaporatorMPC:
    def __init__(self, Ts=1.0, Hp=30, Hu=5, zone_level=False):
        self.Ts = Ts
        self.Hp = Hp
        self.Hu = Hu
        self.zone_level = zone_level
        
        # Base weights
        # If zone_level=True, level tracking weight inside [0.5, 1.5] is zero!
        self.w_level_track = 0.0 if zone_level else 1000.0
        self.w_zone_slack = 5000.0  # Heavy penalty for violating zone bounds [0.5, 1.5]
        self.Q = np.diag([self.w_level_track, 100.0, 100.0])
        self.R = np.eye(3) * 1.0
        
        self.u_min = np.array([0.0, 0.0, 0.0])
        self.u_max = np.array([4.0, 400.0, 400.0])
        
        self.u_last = np.array([2.0, 194.7, 208.0])
        self.x_op = np.array([1.0, 25.0, 50.5, 2.0, 194.7, 208.0])
        self.u_op = np.array([2.0, 194.7, 208.0])
        self.Ad = None
        self.Bd = None
        
    def update_model(self, Ad, Bd, x_op, u_op):
        self.Ad = np.copy(Ad)
        self.Bd = np.copy(Bd)
        self.x_op = np.copy(x_op)
        self.u_op = np.copy(u_op)
        
    def solve(self, x_meas, y_sp):
        n_vars = 3 * self.Hu
        du0 = np.zeros(n_vars)
        bounds = [(-20.0, 20.0) for _ in range(n_vars)]
        
        dx_0 = x_meas - self.x_op
        bias = x_meas[:3] - (self.x_op[:3] + dx_0[:3])
        
        def cost_func(du_vec):
            du_mat = du_vec.reshape((self.Hu, 3))
            cost = 0.0
            x_k = np.copy(dx_0)
            u_k = np.copy(self.u_last)
            
            for k in range(self.Hp):
                if k < self.Hu:
                    du_k = du_mat[k]
                    u_k = u_k + du_k
                    cost += du_k.T @ self.R @ du_k
                    
                delta_u = u_k - self.u_op
                x_k = self.Ad @ x_k + self.Bd @ delta_u
                
                y_pred = self.x_op[:3] + x_k[:3] + bias
                
                # Concentration and pressure tracking
                cost += self.Q[1, 1] * (y_pred[1] - y_sp[1])**2
                cost += self.Q[2, 2] * (y_pred[2] - y_sp[2])**2
                
                if self.zone_level:
                    # Zone penalty on L2: zero if 0.5 <= L2 <= 1.5, penalize violation
                    slack_low = max(0.0, 0.5 - y_pred[0])
                    slack_high = max(0.0, y_pred[0] - 1.5)
                    cost += self.w_zone_slack * (slack_low**2 + slack_high**2)
                else:
                    cost += self.Q[0, 0] * (y_pred[0] - y_sp[0])**2
                    
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
        du_opt = res.x[:3] if res.success else du0[:3]
        self.u_last = np.clip(self.u_last + du_opt, self.u_min, self.u_max)
        return self.u_last

def run_advanced_simulation():
    print("=" * 75)
    print("Chapter 9: Case Study 2 - Exercises 9.5 & 9.6: Zone Control & Event Adaptation")
    print("=" * 75)
    
    model = NewellLeeEvaporator()
    Ts = 1.0
    total_time = 100.0
    steps = int(total_time / Ts)
    t_axis = np.arange(steps) * Ts
    
    init_state = np.array([1.0, 25.0, 50.5, 2.0, 194.7, 208.0])
    init_u = np.array([2.0, 194.7, 208.0])
    Ad0, Bd0 = model.linearize(init_state, init_u, Ts=Ts)
    
    # -------------------------------------------------------------
    # 1. Run Exercise 9.5: Zone Control vs Setpoint Control on L2
    # -------------------------------------------------------------
    print("1. Running Level Setpoint Control (L2 = 1.0 m rigid)...")
    mpc_sp = AdvancedEvaporatorMPC(Ts=Ts, Hp=30, Hu=5, zone_level=False)
    mpc_sp.update_model(Ad0, Bd0, init_state, init_u)
    state_sp = np.copy(init_state)
    y_hist_sp = np.zeros((steps, 3))
    u_hist_sp = np.zeros((steps, 3))
    
    for k in range(steps):
        t = t_axis[k]
        sp_X2 = 25.0 if t < 20.0 else (25.0 - 0.5 * (t - 20.0) if t <= 40.0 else 15.0)
        sp_P2 = 50.5 if t < 20.0 else (50.5 + 0.975 * (t - 20.0) if t <= 40.0 else 70.0)
        y_sp = np.array([1.0, sp_X2, sp_P2])
        
        # Periodic update every 10 min
        if k > 0 and (k % 10 == 0):
            Ad_k, Bd_k = model.linearize(state_sp, mpc_sp.u_last, Ts=Ts)
            mpc_sp.update_model(Ad_k, Bd_k, state_sp, mpc_sp.u_last)
            
        u_opt = mpc_sp.solve(state_sp, y_sp)
        y_hist_sp[k] = state_sp[:3]
        u_hist_sp[k] = u_opt
        
        sol = solve_ivp(model.derivatives, (0, Ts), state_sp, args=(u_opt,), method='RK45', rtol=1e-5, atol=1e-7)
        state_sp = sol.y[:, -1]
        
    print("2. Running Level Zone Control (0.5 <= L2 <= 1.5 m buffer zone)...")
    mpc_zone = AdvancedEvaporatorMPC(Ts=Ts, Hp=30, Hu=5, zone_level=True)
    mpc_zone.update_model(Ad0, Bd0, init_state, init_u)
    state_zone = np.copy(init_state)
    y_hist_zone = np.zeros((steps, 3))
    u_hist_zone = np.zeros((steps, 3))
    
    for k in range(steps):
        t = t_axis[k]
        sp_X2 = 25.0 if t < 20.0 else (25.0 - 0.5 * (t - 20.0) if t <= 40.0 else 15.0)
        sp_P2 = 50.5 if t < 20.0 else (50.5 + 0.975 * (t - 20.0) if t <= 40.0 else 70.0)
        y_sp = np.array([1.0, sp_X2, sp_P2])
        
        if k > 0 and (k % 10 == 0):
            Ad_k, Bd_k = model.linearize(state_zone, mpc_zone.u_last, Ts=Ts)
            mpc_zone.update_model(Ad_k, Bd_k, state_zone, mpc_zone.u_last)
            
        u_opt = mpc_zone.solve(state_zone, y_sp)
        y_hist_zone[k] = state_zone[:3]
        u_hist_zone[k] = u_opt
        
        sol = solve_ivp(model.derivatives, (0, Ts), state_zone, args=(u_opt,), method='RK45', rtol=1e-5, atol=1e-7)
        state_zone = sol.y[:, -1]
        
    # -------------------------------------------------------------
    # 2. Run Exercise 9.6: Event-Triggered Adaptation (|Delta X2| > 1.25%)
    # -------------------------------------------------------------
    print("3. Running Event-Triggered Re-linearization (|Delta X2| > 1.25%)...")
    mpc_event = AdvancedEvaporatorMPC(Ts=Ts, Hp=30, Hu=5, zone_level=False)
    mpc_event.update_model(Ad0, Bd0, init_state, init_u)
    state_event = np.copy(init_state)
    y_hist_event = np.zeros((steps, 3))
    u_hist_event = np.zeros((steps, 3))
    
    last_relin_X2 = 25.0
    relin_events = []
    
    for k in range(steps):
        t = t_axis[k]
        sp_X2 = 25.0 if t < 20.0 else (25.0 - 0.5 * (t - 20.0) if t <= 40.0 else 15.0)
        sp_P2 = 50.5 if t < 20.0 else (50.5 + 0.975 * (t - 20.0) if t <= 40.0 else 70.0)
        y_sp = np.array([1.0, sp_X2, sp_P2])
        
        # Event trigger check: has X2 moved by more than 5% of base (1.25%) since last re-linearization?
        if abs(state_event[1] - last_relin_X2) >= 1.25:
            Ad_ev, Bd_ev = model.linearize(state_event, mpc_event.u_last, Ts=Ts)
            mpc_event.update_model(Ad_ev, Bd_ev, state_event, mpc_event.u_last)
            last_relin_X2 = state_event[1]
            relin_events.append(t)
            
        u_opt = mpc_event.solve(state_event, y_sp)
        y_hist_event[k] = state_event[:3]
        u_hist_event[k] = u_opt
        
        sol = solve_ivp(model.derivatives, (0, Ts), state_event, args=(u_opt,), method='RK45', rtol=1e-5, atol=1e-7)
        state_event = sol.y[:, -1]
        
    print(f">> Event-triggered re-linearizations occurred {len(relin_events)} times at t = {relin_events} min")
    print(">> Compare to periodic 10-min updates: 9 times!")
    
    # Calculate Total Variation (Actuator Wear) on F2: sum |F2(k) - F2(k-1)|
    tv_f2_sp = np.sum(np.abs(np.diff(u_hist_sp[:, 0])))
    tv_f2_zone = np.sum(np.abs(np.diff(u_hist_zone[:, 0])))
    print(f">> F2 Actuator Total Variation: Setpoint Control = {tv_f2_sp:.2f}, Zone Control = {tv_f2_zone:.2f}")
    print(f">> Zone Control achieved {(1.0 - tv_f2_zone/tv_f2_sp)*100:.1f}% reduction in valve cycling!")
    
    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # (0, 0): L2 Level - Zone vs Setpoint
    ax = axes[0, 0]
    ax.axhspan(0.5, 1.5, color='green', alpha=0.15, label='Permissible Level Zone [0.5, 1.5] m')
    ax.axhline(1.0, color='black', linestyle=':', label='Rigid Setpoint (L2=1.0 m)')
    ax.plot(t_axis, y_hist_sp[:, 0], 'r--', linewidth=2.0, label='Rigid Setpoint Control')
    ax.plot(t_axis, y_hist_zone[:, 0], 'b-', linewidth=2.0, label='Zone Control (Ex. 9.5)')
    ax.set_ylabel('Separator Level L2 (m)', fontsize=11, fontweight='bold')
    ax.set_title('Exercise 9.5: Separator Level - Zone vs Setpoint', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9)
    
    # (0, 1): F2 Manipulated Variable - Zone vs Setpoint
    ax = axes[0, 1]
    ax.plot(t_axis, u_hist_sp[:, 0], 'r--', linewidth=2.0, label=f'Setpoint Control (TV={tv_f2_sp:.2f})')
    ax.plot(t_axis, u_hist_zone[:, 0], 'b-', linewidth=2.0, label=f'Zone Control (TV={tv_f2_zone:.2f})')
    ax.set_ylabel('Product Flow Rate F2 (kg/min)', fontsize=11, fontweight='bold')
    ax.set_title('Exercise 9.5: Valve Action F2 - Reduced Cycling in Zone Control', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # (1, 0): X2 Concentration - Event-Triggered vs Periodic
    ax = axes[1, 0]
    sp_X2_line = [25.0 if t < 20.0 else (25.0 - 0.5 * (t - 20.0) if t <= 40.0 else 15.0) for t in t_axis]
    ax.plot(t_axis, sp_X2_line, 'k:', linewidth=2, label='Setpoint (25% -> 15%)')
    ax.plot(t_axis, y_hist_sp[:, 1], 'r--', linewidth=2.0, label='Periodic Update (Every 10 min, 9 times)')
    ax.plot(t_axis, y_hist_event[:, 1], 'g-', linewidth=2.0, label=f'Event-Triggered (|dX2|>5%, {len(relin_events)} times)')
    for evt_t in relin_events:
        ax.axvline(evt_t, color='magenta', linestyle=':', alpha=0.7)
    ax.plot([], [], 'm:', label='Re-linearization Event Trigger')
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Product Concentration X2 (%)', fontsize=11, fontweight='bold')
    ax.set_title('Exercise 9.6: Concentration Tracking under Event Adaptation', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # (1, 1): P2 Operating Pressure - Event-Triggered vs Periodic
    ax = axes[1, 1]
    sp_P2_line = [50.5 if t < 20.0 else (50.5 + 0.975 * (t - 20.0) if t <= 40.0 else 70.0) for t in t_axis]
    ax.plot(t_axis, sp_P2_line, 'k:', linewidth=2, label='Setpoint (50.5 -> 70.0 kPa)')
    ax.plot(t_axis, y_hist_sp[:, 2], 'r--', linewidth=2.0, label='Periodic Update')
    ax.plot(t_axis, y_hist_event[:, 2], 'g-', linewidth=2.0, label='Event-Triggered Adaptation')
    for evt_t in relin_events:
        ax.axvline(evt_t, color='magenta', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Operating Pressure P2 (kPa)', fontsize=11, fontweight='bold')
    ax.set_title('Exercise 9.6: Pressure Response under Event Adaptation', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch09_case_studies/ch9_evaporator_advanced_strategies.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> Advanced strategies plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    run_advanced_simulation()
    print("All advanced simulations completed successfully!")
