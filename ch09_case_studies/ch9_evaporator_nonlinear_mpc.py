"""
Chapter 9: Case Study 2 - Section 9.2 & Exercise 9.4
Newell & Lee Forced-Circulation Evaporator Nonlinear MPC

This script implements:
1. High-fidelity continuous-time nonlinear differential equations of the Newell & Lee (1989) evaporator.
2. Successive online re-linearization comparing:
   (a) Single fixed linear model (linearized at initial equilibrium: X2=25%, P2=50.5 kPa) -> Fig 9.18
   (b) Single re-linearization at t = 70 min -> Fig 9.19
   (c) Periodic re-linearization (every 10 minutes) -> Fig 9.20
3. Setpoint tracking scenario under ramp changes:
   - Product concentration X2 ramps from 25% down to 15% (t in [20, 40] min)
   - Operating pressure P2 ramps from 50.5 kPa up to 70.0 kPa (t in [20, 40] min)
   - Separator liquid level L2 maintained at 1.0 m
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

class NewellLeeEvaporator:
    """
    Nonlinear physics model of the Newell & Lee (1989) forced-circulation evaporator.
    States (6 states total):
        x[0] = L2: Separator liquid level (m)
        x[1] = X2: Product concentration (%)
        x[2] = P2: Operating pressure (kPa)
        x[3] = F2: Actual product flow rate (kg/min)
        x[4] = P100: Actual steam pressure (kPa)
        x[5] = F200: Actual cooling water flow rate (kg/min)
    Inputs (setpoints to valve servos):
        u = [F2_sp, P100_sp, F200_sp]
    Disturbances:
        F1 = 10.0 kg/min (feed flow rate)
        X1 = 5.0 % (feed concentration)
        T1 = 40.0 C (feed temperature)
        T200 = 25.0 C (cooling water inlet temp)
        F3 = 50.0 kg/min (circulating flow rate)
    """
    def __init__(self):
        # Physical constants
        self.rho_A = 20.0       # kg/m (liquid mass per unit level)
        self.C_p = 0.07         # kJ/(kg C) heat capacity of liquor
        self.C_pw = 0.07        # kJ/(kg C) heat capacity of cooling water
        self.lambda_vap = 38.5  # kJ/kg latent heat of vaporization
        self.C_press = 4.0      # kg/kPa vapor capacitance
        self.tau_act = 1.2      # min (actuator lag time constant)
        
        # Nominal disturbances (Table 9.10)
        self.F1 = 10.0          # kg/min feed flow rate
        self.X1 = 5.0           # % feed concentration
        self.T1 = 40.0          # C feed temperature
        self.T200 = 25.0        # C cooling water inlet temp
        self.F3 = 50.0          # kg/min circulating flow rate
        
    def derivatives(self, t, state, u_sp):
        L2, X2, P2, F2, P100, F200 = state
        F2_sp, P100_sp, F200_sp = u_sp
        
        # Saturations on actuators (Table 9.9)
        F2_sat = np.clip(F2, 0.0, 4.0)
        P100_sat = np.clip(P100, 0.0, 400.0)
        F200_sat = np.clip(F200, 0.0, 400.0)
        
        # Empirical saturated steam and boiling temperatures (Newell & Lee 1989)
        T100 = 0.1538 * P100_sat + 90.0
        T2 = 0.5616 * P2 + 0.3126 * X2 + 48.43
        T3 = 0.507 * P2 + 55.0
        
        # Heat transfer in heat exchanger (Eq. 9.16)
        UA1 = 0.16 * (self.F1 + self.F3)
        Q100 = UA1 * (T100 - T2)
        
        # Evaporation boil-up rate F4 (kg/min)
        F4 = (Q100 - self.C_p * self.F1 * (T2 - self.T1)) / self.lambda_vap
        F4 = max(F4, 0.0)
        
        # Heat transfer in condenser & condensation rate F5 (kg/min)
        denom = 1.0 + 6.84 / (0.14 * max(F200_sat, 1e-3))
        Q200 = 6.84 * (T3 - self.T200) / denom
        F5 = Q200 / self.lambda_vap
        
        # Dynamic differential equations (Equations from Newell & Lee 1989):
        # 1. Separator liquid level
        dL2_dt = (self.F1 - F4 - F2_sat) / self.rho_A
        # 2. Product concentration (material balance with holdup)
        mass_holdup = max(self.rho_A * L2, 1.0)
        dX2_dt = (self.F1 * self.X1 - F2_sat * X2 - X2 * (self.F1 - F4 - F2_sat)) / mass_holdup
        # 3. Operating pressure
        dP2_dt = (F4 - F5) / self.C_press
        
        # 4-6. Actuator lags (tau = 1.2 min)
        dF2_dt = (np.clip(F2_sp, 0.0, 4.0) - F2) / self.tau_act
        dP100_dt = (np.clip(P100_sp, 0.0, 400.0) - P100) / self.tau_act
        dF200_dt = (np.clip(F200_sp, 0.0, 400.0) - F200) / self.tau_act
        
        return [dL2_dt, dX2_dt, dP2_dt, dF2_dt, dP100_dt, dF200_dt]

    def linearize(self, state_op, u_op, Ts=1.0):
        """
        Computes 6-state continuous Jacobian (A6, B6) via finite differences,
        and discretizes using exact matrix exponential to form (Ad, Bd).
        """
        eps = 1e-5
        f0 = np.array(self.derivatives(0, state_op, u_op))
        A6 = np.zeros((6, 6))
        B6 = np.zeros((6, 3))
        
        for i in range(6):
            st_p = np.copy(state_op)
            st_p[i] += eps
            f_p = np.array(self.derivatives(0, st_p, u_op))
            A6[:, i] = (f_p - f0) / eps
            
        for j in range(3):
            u_p = np.copy(u_op)
            u_p[j] += eps
            f_p = np.array(self.derivatives(0, state_op, u_p))
            B6[:, j] = (f_p - f0) / eps
            
        # Exact zero-order hold discretization:
        M = np.block([[A6, B6], [np.zeros((3, 6)), np.zeros((3, 3))]])
        eM = expm(M * Ts)
        Ad = eM[:6, :6]
        Bd = eM[:6, 6:]
        return Ad, Bd

class EvaporatorMPC:
    def __init__(self, Ts=1.0, Hp=30, Hu=5):
        self.Ts = Ts
        self.Hp = Hp
        self.Hu = Hu
        
        # MPC Weights (Section 9.2, Equation 9.20)
        # Q = diag(1000, 100, 100) on [L2, X2, P2]
        self.Q = np.diag([1000.0, 100.0, 100.0])
        self.R = np.eye(3) * 1.0  # R(i) = I_3
        
        # Bounds on manipulated variables (Table 9.9)
        self.u_min = np.array([0.0, 0.0, 0.0])
        self.u_max = np.array([4.0, 400.0, 400.0])
        
        self.u_last = np.array([2.0, 194.7, 208.0])
        self.x_op = np.array([1.0, 25.0, 50.5, 2.0, 194.7, 208.0])
        self.u_op = np.array([2.0, 194.7, 208.0])
        self.Ad = None
        self.Bd = None
        
    def update_model(self, Ad, Bd, x_op, u_op):
        """Updates internal linear deviation model around current operating point."""
        self.Ad = np.copy(Ad)
        self.Bd = np.copy(Bd)
        self.x_op = np.copy(x_op)
        self.u_op = np.copy(u_op)
        
    def solve(self, x_meas, y_sp):
        """
        Solves constrained linear MPC using deviation variables:
        Delta x(k) = x(k) - x_op,  delta u(k) = u(k) - u_op
        """
        n_vars = 3 * self.Hu
        du0 = np.zeros(n_vars)
        
        # Deviation initial state
        dx_0 = x_meas - self.x_op
        bias = x_meas[:3] - (self.x_op[:3] + dx_0[:3])
        
        bounds = [(-20.0, 20.0) for _ in range(n_vars)]
        
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
                    
                # Linear propagation in deviation variables:
                delta_u = u_k - self.u_op
                x_k = self.Ad @ x_k + self.Bd @ delta_u
                
                # Output prediction in physical units:
                y_pred = self.x_op[:3] + x_k[:3] + bias
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
        if res.success:
            du_opt = res.x[:3]
        else:
            du_opt = du0[:3]
            
        self.u_last = np.clip(self.u_last + du_opt, self.u_min, self.u_max)
        return self.u_last

def simulate_evaporator_scenario(mode='fixed'):
    """
    Simulates the Newell-Lee evaporator under three distinct adaptation modes:
    - mode='fixed': Single linear model at t=0 (Fig 9.18)
    - mode='relinearize_70': Single re-linearization at t=70 min (Fig 9.19)
    - mode='adaptive_10': Periodic re-linearization every 10 min (Fig 9.20)
    """
    model = NewellLeeEvaporator()
    Ts = 1.0  # minute
    total_time = 100.0
    steps = int(total_time / Ts)
    t_axis = np.arange(steps) * Ts
    
    # Equilibrium operating point (Tables 9.8 - 9.10)
    init_state = np.array([1.0, 25.0, 50.5, 2.0, 194.7, 208.0])
    init_u = np.array([2.0, 194.7, 208.0])
    
    mpc = EvaporatorMPC(Ts=Ts, Hp=30, Hu=5)
    Ad0, Bd0 = model.linearize(init_state, init_u, Ts=Ts)
    mpc.update_model(Ad0, Bd0, init_state, init_u)
    
    state_curr = np.copy(init_state)
    
    y_hist = np.zeros((steps, 3))
    u_hist = np.zeros((steps, 3))
    sp_hist = np.zeros((steps, 3))
    
    for k in range(steps):
        t = t_axis[k]
        
        # Setpoint trajectories (Section 9.2):
        # L2 setpoint remains 1.0 m
        # X2 ramps down from 25% to 15% between t=20 and t=40
        # P2 ramps up from 50.5 kPa to 70.0 kPa between t=20 and t=40
        if t < 20.0:
            sp_X2 = 25.0
            sp_P2 = 50.5
        elif t <= 40.0:
            sp_X2 = 25.0 - (10.0 / 20.0) * (t - 20.0)
            sp_P2 = 50.5 + (19.5 / 20.0) * (t - 20.0)
        else:
            sp_X2 = 15.0
            sp_P2 = 70.0
        sp_L2 = 1.0
        
        y_sp = np.array([sp_L2, sp_X2, sp_P2])
        sp_hist[k] = y_sp
        
        # Model adaptation logic
        relinearize = False
        if mode == 'relinearize_70' and k == 70:
            relinearize = True
        elif mode == 'adaptive_10' and k > 0 and (k % 10 == 0):
            relinearize = True
            
        if relinearize:
            Ad_new, Bd_new = model.linearize(state_curr, mpc.u_last, Ts=Ts)
            mpc.update_model(Ad_new, Bd_new, state_curr, mpc.u_last)
            
        # MPC Solve
        u_opt = mpc.solve(state_curr, y_sp)
        
        # Record histories
        y_hist[k] = state_curr[:3]
        u_hist[k] = u_opt
        
        # High-precision ODE integration across sampling interval Ts
        sol = solve_ivp(model.derivatives, (0, Ts), state_curr, args=(u_opt,), method='RK45', rtol=1e-5, atol=1e-7)
        state_curr = sol.y[:, -1]
        
    return t_axis, y_hist, u_hist, sp_hist

def plot_evaporator_results(res_fixed, res_relin70, res_adapt10):
    t_axis = res_fixed[0]
    sp_hist = res_fixed[3]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    
    labels = ['Fixed Linear Model (Fig 9.18)', 'Re-linearized at t=70 min (Fig 9.19)', 'Adaptive Every 10 min (Fig 9.20)']
    styles = ['r--', 'g-.', 'b-']
    colors = ['tab:red', 'tab:green', 'tab:blue']
    
    datasets = [res_fixed, res_relin70, res_adapt10]
    
    # Row 0: L2 and F2
    ax = axes[0, 0]
    ax.plot(t_axis, sp_hist[:, 0], 'k:', linewidth=2, label='Setpoint (L2=1.0m)')
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[1][:, 0], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_ylabel('Separator Level L2 (m)', fontsize=11, fontweight='bold')
    ax.set_title('Separator Level Response (m)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9)
    
    ax = axes[0, 1]
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[2][:, 0], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_ylabel('F2 SP (kg/min)', fontsize=11, fontweight='bold')
    ax.set_title('Product Flow Rate Setpoint F2 SP (kg/min)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # Row 1: X2 and P100
    ax = axes[1, 0]
    ax.plot(t_axis, sp_hist[:, 1], 'k:', linewidth=2, label='Setpoint (25% -> 15%)')
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[1][:, 1], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_ylabel('Concentration X2 (%)', fontsize=11, fontweight='bold')
    ax.set_title('Product Concentration Response X2 (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    ax = axes[1, 1]
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[2][:, 1], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_ylabel('P100 SP (kPa)', fontsize=11, fontweight='bold')
    ax.set_title('Steam Pressure Setpoint P100 SP (kPa)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # Row 2: P2 and F200
    ax = axes[2, 0]
    ax.plot(t_axis, sp_hist[:, 2], 'k:', linewidth=2, label='Setpoint (50.5 -> 70.0 kPa)')
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[1][:, 2], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pressure P2 (kPa)', fontsize=11, fontweight='bold')
    ax.set_title('Operating Pressure Response P2 (kPa)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    
    ax = axes[2, 1]
    for idx, d in enumerate(datasets):
        ax.plot(t_axis, d[2][:, 2], styles[idx], label=labels[idx], linewidth=1.8)
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F200 SP (kg/min)', fontsize=11, fontweight='bold')
    ax.set_title('Cooling Water Flow Setpoint F200 SP (kg/min)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch09_case_studies/ch9_evaporator_nonlinear_mpc.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> Plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    print("=" * 75)
    print("Chapter 9: Case Study 2 - Newell & Lee Evaporator Nonlinear MPC")
    print("=" * 75)
    print("1. Running Fixed Linear Model MPC (Fig 9.18)...")
    res_fixed = simulate_evaporator_scenario(mode='fixed')
    print("2. Running Single Re-linearization at t=70 min MPC (Fig 9.19)...")
    res_relin70 = simulate_evaporator_scenario(mode='relinearize_70')
    print("3. Running Adaptive Periodic Re-linearized MPC (Fig 9.20, every 10 min)...")
    res_adapt10 = simulate_evaporator_scenario(mode='adaptive_10')
    
    print("Simulations complete! Generating comparison plots...")
    plot_evaporator_results(res_fixed, res_relin70, res_adapt10)
    print("Done!")
