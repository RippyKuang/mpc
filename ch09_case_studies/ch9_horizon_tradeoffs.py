"""
Chapter 9: Case Study 1 - Exercise 9.3
Prediction Horizon (Hp) Trade-off Study on the Shell Heavy Oil Fractionator

This script simulates the Shell Heavy Oil Fractionator with various prediction horizons:
Hp in [15, 30, 60, 90]
Investigates:
1. Impact on closed-loop dynamic performance (setpoint tracking and disturbance rejection)
2. Quantitative computation time scaling per QP step
3. Trade-off Pareto analysis between computation cost and tracking error (ISE)
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_horizon_study():
    print("=" * 75)
    print("Chapter 9: Exercise 9.3 - Prediction Horizon (Hp) Trade-off Analysis")
    print("=" * 75)
    
    Ts = 4.0
    nu = 3
    ny = 3
    
    # System gains, time constants, delays
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
    
    N_steps = 75
    S_step = np.zeros((ny, nu, N_steps + 1))
    for i in range(ny):
        for j in range(nu):
            for l in range(1, N_steps + 1):
                t_val = l * Ts
                if t_val > theta[i, j]:
                    S_step[i, j, l] = K[i, j] * (1.0 - np.exp(-(t_val - theta[i, j]) / tau[i, j]))
                    
    sim_steps = 50
    t_axis = np.arange(sim_steps) * Ts
    
    # Horizons to test
    Hp_list = [15, 30, 60, 90]
    Hu = 5
    
    results = {}
    
    for Hp in Hp_list:
        print(f"\nTesting Prediction Horizon Hp = {Hp}...")
        y_hist = np.zeros((sim_steps, ny))
        u_hist = np.zeros((sim_steps, nu))
        u_curr = np.zeros(nu)
        du_hist = np.zeros((nu, N_steps))
        
        # True plant state
        x_plant = np.zeros((ny, nu))
        delay_steps = np.round(theta / Ts).astype(int)
        u_buffer = np.zeros((np.max(delay_steps) + 10, nu))
        a_plant = np.exp(-Ts / tau)
        b_plant = K * (1.0 - a_plant)
        
        step_times = []
        
        for k in range(sim_steps):
            t = t_axis[k]
            # Setpoint change on Side End Point y2 at t = 20 min
            y_sp = np.array([0.0, 0.3 if t >= 20.0 else 0.0, 0.0])
            
            y_meas = np.zeros(ny) if k == 0 else y_hist[k-1]
            
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
                
            n_opt = nu * Hu
            du_bounds = [(-0.2, 0.2)] * n_opt
            
            t_start = time.perf_counter()
            
            def obj(du_vec):
                du_m = du_vec.reshape((Hu, nu))
                cost = 0.0
                u_run = np.copy(u_curr)
                for step_k in range(Hu):
                    u_run += du_m[step_k]
                    cost += 2.0 * du_m[step_k] @ du_m[step_k]
                    cost += 1.5 * (u_run[2] - (-0.3181))**2
                for p in range(Hp):
                    y_pred = np.copy(f_pred[p])
                    for i in range(ny):
                        for j in range(nu):
                            for step_k in range(min(p + 1, Hu)):
                                if p - step_k + 1 <= N_steps:
                                    y_pred[i] += S_step[i, j, p - step_k + 1] * du_m[step_k, j]
                    err = y_pred - y_sp
                    cost += 15.0 * err[0]**2 + 15.0 * err[1]**2 + 1.0 * err[2]**2
                return cost
                
            res = minimize(obj, np.zeros(n_opt), method='SLSQP', bounds=du_bounds, options={'maxiter': 25})
            t_end = time.perf_counter()
            step_times.append((t_end - t_start) * 1000.0)  # ms
            
            du_opt = res.x[:nu] if res.success else np.zeros(nu)
            u_curr = np.clip(u_curr + du_opt, -0.5, 0.5)
            u_hist[k] = u_curr
            du_hist[:, 1:] = du_hist[:, :-1]
            du_hist[:, 0] = du_opt
            
            # Plant update
            u_buffer[1:] = u_buffer[:-1]
            u_buffer[0] = u_curr
            y_out = np.zeros(ny)
            for i in range(ny):
                for j in range(nu):
                    u_del = u_buffer[delay_steps[i, j], j]
                    x_plant[i, j] = a_plant[i, j] * x_plant[i, j] + b_plant[i, j] * u_del
                    y_out[i] += x_plant[i, j]
            y_hist[k] = y_out
            
        # Performance metric: ISE on y2 after step change (t >= 20)
        target = np.array([0.3 if t >= 20.0 else 0.0 for t in t_axis])
        ise = np.sum((y_hist[:, 1] - target)**2) * Ts
        avg_cpu = np.mean(step_times)
        
        print(f"Hp = {Hp:2d}: Avg CPU = {avg_cpu:.2f} ms/step, ISE = {ise:.4f}")
        results[Hp] = {
            'y': y_hist,
            'u': u_hist,
            'avg_cpu': avg_cpu,
            'ise': ise
        }
        
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {15: 'red', 30: 'orange', 60: 'blue', 90: 'green'}
    
    # Plot (a): Output Tracking Comparison
    ax = axes[0, 0]
    target_line = np.array([0.3 if t >= 20.0 else 0.0 for t in t_axis])
    ax.step(t_axis, target_line, 'k--', linewidth=1.5, label='Setpoint Target')
    for Hp in Hp_list:
        ax.plot(t_axis, results[Hp]['y'][:, 1], color=colors[Hp], linewidth=2, label=f'Hp = {Hp}')
    ax.set_title('Exercise 9.3: Side End Point ($y_2$) Response for Various $H_p$', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Output $y_2$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (b): Control Actuation Move Comparison
    ax = axes[0, 1]
    for Hp in Hp_list:
        ax.step(t_axis, results[Hp]['u'][:, 1], color=colors[Hp], where='post', linewidth=1.8, label=f'Hp = {Hp}: $u_2(k)$')
    ax.axhline(0.5, color='gray', linestyle=':', label=r'Limits $\pm 0.5$')
    ax.axhline(-0.5, color='gray', linestyle=':')
    ax.set_title('Control Moves $u_2(k)$ (Longer $H_p$ Gives Smoother Action)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Input $u_2$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (c): Computation Time per Step
    ax = axes[1, 0]
    cpus = [results[Hp]['avg_cpu'] for Hp in Hp_list]
    bars = ax.bar([str(Hp) for Hp in Hp_list], cpus, color=['salmon', 'moccasin', 'lightskyblue', 'lightgreen'], edgecolor='black', width=0.5)
    ax.set_title('Average Solve Time per QP Step vs Prediction Horizon', fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Horizon $H_p$ (steps)')
    ax.set_ylabel('CPU Time (ms / step)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.1f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim([0, max(cpus) * 1.25])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Plot (d): Pareto Frontier (ISE vs CPU Time)
    ax = axes[1, 1]
    ises = [results[Hp]['ise'] for Hp in Hp_list]
    ax.plot(cpus, ises, 'mo-', linewidth=2, markersize=8)
    for Hp in Hp_list:
        ax.annotate(f'Hp = {Hp}', (results[Hp]['avg_cpu'], results[Hp]['ise']),
                    textcoords="offset points", xytext=(8, 5), ha='left', fontsize=9, fontweight='bold')
    ax.set_title('Pareto Frontier: Tracking Error (ISE) vs Computation Load', fontsize=11, fontweight='bold')
    ax.set_xlabel('Average CPU Time (ms)')
    ax.set_ylabel('Integral Squared Error (ISE)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch09_case_studies/ch9_horizon_tradeoffs.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n>> Plot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_horizon_study()
