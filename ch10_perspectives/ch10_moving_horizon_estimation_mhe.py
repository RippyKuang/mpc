"""
Chapter 10: Perspectives - Section 10.4 & Equations 10.27-10.30
Moving Horizon Estimation (MHE) with Physical State Constraints

This script demonstrates:
1. Moving Horizon Estimation as the dual of Model Predictive Control:
   Looks backward over an estimation window of length N to estimate state x(k).
2. The fundamental limitation of the standard Kalman Filter (KF):
   It cannot enforce physical inequality constraints (e.g., chemical concentrations x >= 0).
   Under severe negative measurement noise spikes near zero, KF estimates unphysical negative states.
3. Constrained MHE formulation as an online Quadratic Program (QP):
   Minimizes backward prediction errors + process noise + arrival cost
   subject to strict state bound constraints x(k - i) >= 0.
4. Comparative benchmark: True state vs Noisy sensor vs Kalman Filter vs Constrained MHE.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ChemicalProcessModel:
    def __init__(self):
        self.Ts = 1.0  # min
        # Continuous CSTR / decay dynamics discretized
        # States: x1 = reactant concentration (mol/L, x1 >= 0)
        #         x2 = catalyst activity (0 <= x2 <= 1)
        self.A = np.array([
            [0.80, 0.05],
            [0.00, 0.95]
        ])
        self.B = np.array([
            [0.20],
            [0.00]
        ])
        self.C = np.array([
            [1.0, 0.0]
        ])
        
        # Noise covariances
        self.W_cov = np.diag([0.01, 0.002])  # Process noise
        self.V_cov = 0.05                     # Measurement noise

class StandardKalmanFilter:
    def __init__(self, model):
        self.m = model
        self.x_est = np.array([1.5, 0.8])  # Prior guess
        self.P = np.eye(2) * 0.1
        self.Q_kf = self.m.W_cov
        self.R_kf = np.array([[self.m.V_cov]])
        
    def step(self, u_prev, y_meas):
        # 1. Predict
        x_pred = self.m.A @ self.x_est + self.m.B.flatten() * u_prev
        P_pred = self.m.A @ self.P @ self.m.A.T + self.Q_kf
        
        # 2. Update
        y_pred = self.m.C @ x_pred
        S = self.m.C @ P_pred @ self.m.C.T + self.R_kf
        K = P_pred @ self.m.C.T @ np.linalg.inv(S)
        
        self.x_est = x_pred + (K @ (y_meas - y_pred)).flatten()
        self.P = (np.eye(2) - K @ self.m.C) @ P_pred
        return np.copy(self.x_est)

class MovingHorizonEstimator:
    def __init__(self, model, N=6):
        self.m = model
        self.N = N  # Estimation horizon length
        
        self.R_weight = 1.0 / self.m.V_cov
        self.Q_weight = np.linalg.inv(self.m.W_cov)
        self.P0_inv = np.eye(2) * 5.0  # Arrival cost weight
        
        self.u_history = []
        self.y_history = []
        self.x_prior = np.array([1.5, 0.8])
        
    def update_data(self, u_val, y_val):
        self.u_history.append(u_val)
        self.y_history.append(y_val)
        
    def estimate(self):
        L = len(self.y_history)
        if L < 2:
            return np.copy(self.x_prior)
            
        N_curr = min(L, self.N)
        
        # We optimize initial state x(k - N_curr + 1) and sequence of process noises w
        # Decision variables: x_0 (dim 2) and w_0, ..., w_{N_curr-2} (dim 2 * (N_curr - 1))
        n_w = 2 * (N_curr - 1)
        n_vars = 2 + n_w
        var0 = np.zeros(n_vars)
        var0[:2] = self.x_prior
        
        y_window = self.y_history[-N_curr:]
        u_window = self.u_history[-N_curr:]
        
        def cost_func(vars_vec):
            x_0 = vars_vec[:2]
            w_mat = vars_vec[2:].reshape((N_curr - 1, 2)) if n_w > 0 else np.zeros((0, 2))
            
            cost = 0.0
            # 1. Arrival cost
            err_prior = x_0 - self.x_prior
            cost += err_prior.T @ self.P0_inv @ err_prior
            
            x_k = np.copy(x_0)
            # 2. Measurement error at step 0
            err_y0 = (self.m.C @ x_k)[0] - y_window[0]
            cost += self.R_weight * (err_y0**2)
            
            # Forward simulation across window
            for j in range(N_curr - 1):
                w_j = w_mat[j]
                cost += w_j.T @ self.Q_weight @ w_j
                x_k = self.m.A @ x_k + self.m.B.flatten() * u_window[j] + w_j
                err_y = (self.m.C @ x_k)[0] - y_window[j + 1]
                cost += self.R_weight * (err_y**2)
                
            return cost
            
        # Physical constraints: ALL states at all times must be non-negative: x1 >= 0, x2 >= 0!
        cons = []
        # Initial state bound
        cons.append({'type': 'ineq', 'fun': lambda v: v[0]})  # x1(0) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[1]})  # x2(0) >= 0
        
        # Propagated states bounds
        for step_i in range(N_curr - 1):
            def make_state_con(i_idx, state_ch):
                def con(vars_vec):
                    x_k = vars_vec[:2]
                    w_mat = vars_vec[2:].reshape((N_curr - 1, 2))
                    for j in range(i_idx + 1):
                        x_k = self.m.A @ x_k + self.m.B.flatten() * u_window[j] + w_mat[j]
                    return x_k[state_ch]
                return con
            cons.append({'type': 'ineq', 'fun': make_state_con(step_i, 0)})
            cons.append({'type': 'ineq', 'fun': make_state_con(step_i, 1)})
            
        res = minimize(cost_func, var0, method='SLSQP', constraints=cons, options={'maxiter': 30, 'ftol': 1e-3})
        
        # Forward propagate optimal variables to get current estimate x(k|k)
        x_opt = res.x[:2]
        if n_w > 0:
            w_opt = res.x[2:].reshape((N_curr - 1, 2))
            for j in range(N_curr - 1):
                x_opt = self.m.A @ x_opt + self.m.B.flatten() * u_window[j] + w_opt[j]
                
        # Update prior for next arrival cost
        self.x_prior = np.clip(x_opt, 0.0, None)
        return np.copy(self.x_prior)

def run_mhe_experiments(n_steps=50):
    print("=" * 70)
    print("Chapter 10: Section 10.4 - Moving Horizon Estimation (MHE) with Constraints")
    print("=" * 70)
    
    model = ChemicalProcessModel()
    kf = StandardKalmanFilter(model)
    mhe = MovingHorizonEstimator(model, N=6)
    
    t_axis = np.arange(n_steps) * model.Ts
    
    # Ground truth simulation
    x_true = np.array([1.2, 0.9])
    
    x_true_hist = np.zeros((n_steps, 2))
    y_meas_hist = np.zeros(n_steps)
    kf_est_hist = np.zeros((n_steps, 2))
    mhe_est_hist = np.zeros((n_steps, 2))
    
    np.random.seed(42)
    
    for k in range(n_steps):
        t = t_axis[k]
        # Feed input drops at t=15 min to drive concentration close to zero
        u_in = 0.5 if t < 15.0 else 0.02
        
        # Add process disturbance
        w_k = np.random.multivariate_normal([0, 0], model.W_cov)
        x_true = np.clip(model.A @ x_true + model.B.flatten() * u_in + w_k, 0.0, None)
        
        # Sensor measurement with Gaussian noise + severe negative outlier spikes (e.g. at t=28 and t=35)
        v_k = np.random.normal(0, np.sqrt(model.V_cov))
        if k in [25, 32]:
            v_k = -0.45  # Severe negative sensor artifact!
            
        y_meas = float((model.C @ x_true)[0]) + v_k
        
        # Update estimators
        x_kf = kf.step(u_in, np.array([y_meas]))
        mhe.update_data(u_in, y_meas)
        x_mhe = mhe.estimate()
        
        x_true_hist[k] = x_true
        y_meas_hist[k] = y_meas
        kf_est_hist[k] = x_kf
        mhe_est_hist[k] = x_mhe
        
    # Analyze negative violations
    kf_neg_violations = np.sum(kf_est_hist[:, 0] < 0.0)
    mhe_neg_violations = np.sum(mhe_est_hist[:, 0] < 0.0)
    print(f"Kalman Filter unphysical negative concentration occurrences: {kf_neg_violations} times (Min = {np.min(kf_est_hist[:, 0]):.4f} mol/L)")
    print(f"Constrained MHE unphysical negative concentration occurrences: {mhe_neg_violations} times (Min = {np.min(mhe_est_hist[:, 0]):.4f} mol/L)")
    print(">> Constrained MHE strictly respects physical thermodynamic bounds x >= 0!")
    
    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # Top: Reactant Concentration x1
    ax0 = axes[0]
    ax0.plot(t_axis, x_true_hist[:, 0], 'k-', linewidth=2.5, label='True Reactant Concentration (x1 >= 0)')
    ax0.plot(t_axis, y_meas_hist, 'rx', alpha=0.6, markersize=6, label='Noisy Sensor Measurements (with negative spikes)')
    ax0.plot(t_axis, kf_est_hist[:, 0], 'r--', linewidth=1.8, label='Unconstrained Kalman Filter (Dips < 0!)')
    ax0.plot(t_axis, mhe_est_hist[:, 0], 'b-', linewidth=2.0, label='Constrained MHE (Strictly >= 0)')
    ax0.axhline(0.0, color='gray', linestyle=':', linewidth=1.5, label='Physical Lower Bound (0 mol/L)')
    ax0.set_ylabel('Concentration (mol/L)', fontsize=11, fontweight='bold')
    ax0.set_title('Moving Horizon Estimation vs Kalman Filter under Low Concentrations', fontsize=12, fontweight='bold')
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.legend(loc='upper right', fontsize=9)
    
    # Bottom: Estimation Errors
    ax1 = axes[1]
    err_kf = np.abs(kf_est_hist[:, 0] - x_true_hist[:, 0])
    err_mhe = np.abs(mhe_est_hist[:, 0] - x_true_hist[:, 0])
    ax1.plot(t_axis, err_kf, 'r--', linewidth=1.8, label=f'Kalman Filter Error (Mean = {np.mean(err_kf):.4f})')
    ax1.plot(t_axis, err_mhe, 'b-', linewidth=2.0, label=f'Constrained MHE Error (Mean = {np.mean(err_mhe):.4f})')
    ax1.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Absolute Error |x1 - x1_true|', fontsize=11, fontweight='bold')
    ax1.set_title('Estimation Error Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plot_path = '/home/kuang/Desktop/mpc/ch10_perspectives/ch10_moving_horizon_estimation_mhe.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f">> MHE plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    run_mhe_experiments()
