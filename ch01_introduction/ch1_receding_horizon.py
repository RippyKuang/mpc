"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 1: Introduction - Receding Horizon Control Simulation Suite
Filename: ch1_receding_horizon.py

This script implements:
1. BasicMPC: Unconstrained receding horizon control using transfer function model
   (Faithful Python refactor of MATLAB program basicmpc.m, Appendix B)
2. TrackMPC: Incorporating estimated output disturbance d_hat(k) for offset-free tracking
   under plant-model mismatch and constant disturbances (trackmpc.m)
3. NoisyMPC: Simulating tracking in the presence of process noise, measurement noise,
   input disturbances, and output disturbances (noisympc.m)
4. UnstaMPC: Receding horizon control for open-loop unstable and non-minimum phase plants
   (e.g., longitudinal helicopter dynamics, Example 1.8) using model realignment

Dependencies: numpy, scipy, matplotlib
===============================================================================
"""

import os
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent display errors
import matplotlib.pyplot as plt

# Ensure reproducible figures
np.random.seed(42)

# =============================================================================
# Helper Utilities: Discrete Transfer Function Simulation & Step Response
# =============================================================================

class DiscreteTF:
    """
    Represents a discrete-time SISO transfer function:
        G(z) = B(z) / A(z)
    where:
        A(z) = z^n + a_1 z^{n-1} + ... + a_n
        B(z) = b_0 z^m + b_1 z^{m-1} + ... + b_m  (with b_0 = 0 for strictly proper)
    In difference equation form:
        y(k) = - sum_{i=1}^n a_i y(k-i) + sum_{j=1}^m b_j u(k-j)
    """
    def __init__(self, num, den, Ts):
        num = np.array(num, dtype=float).flatten()
        den = np.array(den, dtype=float).flatten()
        self.Ts = float(Ts)

        # Normalize so den[0] == 1.0
        if den[0] != 1.0:
            num = num / den[0]
            den = den / den[0]

        # Pad num to have the same length as den if necessary
        if len(num) < len(den):
            num = np.pad(num, (len(den) - len(num), 0), 'constant')
        elif len(den) < len(num):
            raise ValueError("Transfer function must be proper (degree(num) <= degree(den)).")

        self.num = num
        self.den = den
        self.order = len(den) - 1

        # Check strictly proper
        if abs(self.num[0]) > 1e-10:
            raise ValueError("Plant/Model must be strictly proper: num[0] == 0.")

    def step_response(self, N_steps):
        """
        Calculates step response S(0), S(1), ..., S(N_steps).
        For strictly proper system, S(0) = 0.
        """
        S = np.zeros(N_steps + 1)
        u = np.ones(N_steps + 1)
        for k in range(1, N_steps + 1):
            val = 0.0
            for j in range(1, len(self.num)):
                if k - j >= 0:
                    val += self.num[j] * u[k - j]
            for i in range(1, len(self.den)):
                if k - i >= 0:
                    val -= self.den[i] * S[k - i]
            S[k] = val
        return S

    def simulate_step(self, u_past, y_past):
        """
        One-step difference equation update:
        u_past: [u(k-1), u(k-2), ..., u(k - order)]
        y_past: [y(k-1), y(k-2), ..., y(k - order)]
        Returns y(k).
        """
        val_u = 0.0
        for j in range(1, len(self.num)):
            if j - 1 < len(u_past):
                val_u += self.num[j] * u_past[j - 1]
        val_y = 0.0
        for i in range(1, len(self.den)):
            if i - 1 < len(y_past):
                val_y += self.den[i] * y_past[i - 1]
        return val_u - val_y


# =============================================================================
# 1. BasicMPC: Baseline Formulation (Independent Model, No Disturbance Estimation)
# =============================================================================

def run_basicmpc(plant_tf, model_tf, P=[1], M=1, Tref=6.0, Ts=0.6, tend=60.0, setpoint_val=1.0):
    """
    Implements basic receding horizon control without constraints and without offset-free modification.
    Corresponds to basicmpc.m (Maciejowski, Appendix B).
    
    P: list of coincidence points, e.g. [1] or [1, 2] or [8]
    M: control horizon H_u
    """
    nsteps = int(np.floor(tend / Ts))
    tvec = np.arange(nsteps) * Ts

    P = np.array(P, dtype=int)
    max_P = int(np.max(P))
    len_P = len(P)

    # 1. Compute model step response up to max_P
    step_resp = model_tf.step_response(max_P)

    # 2. Build dynamic matrix Theta of size (len(P) x M)
    Theta = np.zeros((len_P, M))
    for j in range(len_P):
        pj = P[j]
        for i in range(1, M + 1):
            shift = pj - i + 1
            if 0 < shift < len(step_resp):
                Theta[j, i - 1] = step_resp[shift]
            else:
                Theta[j, i - 1] = 0.0

    # Exponential reference trajectory decay factor at coincidence points
    if Tref <= 1e-6:
        errfac = np.zeros(len_P)
    else:
        errfac = np.exp(-P * Ts / Tref)

    # State histories
    order_p = plant_tf.order
    order_m = model_tf.order
    ypast_plant = np.zeros(order_p)
    upast_plant = np.zeros(order_p)
    ypast_model = np.zeros(order_m)
    upast_model = np.zeros(order_m)

    y_plant_hist = np.zeros(nsteps)
    y_model_hist = np.zeros(nsteps)
    u_hist = np.zeros(nsteps)
    sp_hist = np.ones(nsteps) * setpoint_val

    u_prev = 0.0

    for k in range(nsteps):
        # Current plant output
        y_curr = ypast_plant[0] if k > 0 else 0.0
        y_plant_hist[k] = y_curr

        # Compute reference trajectory at coincidence points:
        # r(k + P_j | k) = s(k + P_j) - [s(k) - y(k)] * exp(-P_j * Ts / Tref)
        error_now = setpoint_val - y_curr
        r_P = setpoint_val - error_now * errfac

        # Free response of the internal model over prediction horizon
        ym_free = np.zeros(max_P + 1)
        yfpast = ypast_model.copy()
        ufpast = upast_model.copy()

        for step in range(1, max_P + 1):
            y_next = model_tf.simulate_step(ufpast, yfpast)
            ym_free[step] = y_next
            # update histories for next free step:
            yfpast = np.roll(yfpast, 1)
            yfpast[0] = y_next
            ufpast = np.roll(ufpast, 1)
            ufpast[0] = u_prev

        # ym_free at coincidence points P
        ym_free_P = ym_free[P]

        # Solve least-squares: Theta * Delta_U = r_P - ym_free_P
        rhs = r_P - ym_free_P
        delta_u_traj, _, _, _ = np.linalg.lstsq(Theta, rhs, rcond=None)

        # First control move: u(k) = u(k-1) + Delta_u(k|k)
        u_k = u_prev + delta_u_traj[0]
        u_hist[k] = u_k
        u_prev = u_k

        # Advance plant
        upast_plant = np.roll(upast_plant, 1)
        upast_plant[0] = u_k
        y_plant_next = plant_tf.simulate_step(upast_plant, ypast_plant)
        ypast_plant = np.roll(ypast_plant, 1)
        ypast_plant[0] = y_plant_next

        # Advance model
        upast_model = np.roll(upast_model, 1)
        upast_model[0] = u_k
        y_model_next = model_tf.simulate_step(upast_model, ypast_model)
        ypast_model = np.roll(ypast_model, 1)
        ypast_model[0] = y_model_next
        y_model_hist[k] = ypast_model[0]

    return {
        't': tvec,
        'y_plant': y_plant_hist,
        'y_model': y_model_hist,
        'u': u_hist,
        'setpoint': sp_hist
    }


# =============================================================================
# 2. TrackMPC: Offset-Free Predictive Control via Disturbance Estimation
# =============================================================================

def run_trackmpc(plant_tf, model_tf, P=[1], M=1, Tref=6.0, Ts=0.6, tend=60.0, setpoint_val=1.0):
    """
    Implements offset-free receding horizon control by estimating an output disturbance:
        d_hat(k) = y_plant(k) - y_model(k)
    and modifying the prediction:
        y_pred(k + P_j | k) = ym_free(k + P_j | k) + d_hat(k)
    This guarantees zero steady-state tracking error even with constant plant-model mismatch.
    """
    nsteps = int(np.floor(tend / Ts))
    tvec = np.arange(nsteps) * Ts

    P = np.array(P, dtype=int)
    max_P = int(np.max(P))
    len_P = len(P)

    step_resp = model_tf.step_response(max_P)

    Theta = np.zeros((len_P, M))
    for j in range(len_P):
        pj = P[j]
        for i in range(1, M + 1):
            shift = pj - i + 1
            if 0 < shift < len(step_resp):
                Theta[j, i - 1] = step_resp[shift]
            else:
                Theta[j, i - 1] = 0.0

    errfac = np.zeros(len_P) if Tref <= 1e-6 else np.exp(-P * Ts / Tref)

    order_p = plant_tf.order
    order_m = model_tf.order
    ypast_plant = np.zeros(order_p)
    upast_plant = np.zeros(order_p)
    ypast_model = np.zeros(order_m)
    upast_model = np.zeros(order_m)

    y_plant_hist = np.zeros(nsteps)
    y_model_hist = np.zeros(nsteps)
    u_hist = np.zeros(nsteps)
    d_hat_hist = np.zeros(nsteps)
    sp_hist = np.ones(nsteps) * setpoint_val

    u_prev = 0.0

    for k in range(nsteps):
        y_curr = ypast_plant[0] if k > 0 else 0.0
        y_plant_hist[k] = y_curr

        ym_curr = ypast_model[0] if k > 0 else 0.0
        y_model_hist[k] = ym_curr

        # Disturbance estimate (Eq 1.30): d_hat(k) = y(k) - y_m(k)
        d_hat = y_curr - ym_curr
        d_hat_hist[k] = d_hat

        # Reference trajectory
        error_now = setpoint_val - y_curr
        r_P = setpoint_val - error_now * errfac

        # Free response of model
        ym_free = np.zeros(max_P + 1)
        yfpast = ypast_model.copy()
        ufpast = upast_model.copy()

        for step in range(1, max_P + 1):
            y_next = model_tf.simulate_step(ufpast, yfpast)
            ym_free[step] = y_next
            yfpast = np.roll(yfpast, 1)
            yfpast[0] = y_next
            ufpast = np.roll(ufpast, 1)
            ufpast[0] = u_prev

        # Free response corrected by estimated disturbance (Eq 1.31):
        y_pred_free_P = ym_free[P] + d_hat

        # Solve least-squares: Theta * Delta_U = r_P - y_pred_free_P
        rhs = r_P - y_pred_free_P
        delta_u_traj, _, _, _ = np.linalg.lstsq(Theta, rhs, rcond=None)

        u_k = u_prev + delta_u_traj[0]
        u_hist[k] = u_k
        u_prev = u_k

        # Advance plant
        upast_plant = np.roll(upast_plant, 1)
        upast_plant[0] = u_k
        y_plant_next = plant_tf.simulate_step(upast_plant, ypast_plant)
        ypast_plant = np.roll(ypast_plant, 1)
        ypast_plant[0] = y_plant_next

        # Advance model
        upast_model = np.roll(upast_model, 1)
        upast_model[0] = u_k
        y_model_next = model_tf.simulate_step(upast_model, ypast_model)
        ypast_model = np.roll(ypast_model, 1)
        ypast_model[0] = y_model_next

    return {
        't': tvec,
        'y_plant': y_plant_hist,
        'y_model': y_model_hist,
        'u': u_hist,
        'd_hat': d_hat_hist,
        'setpoint': sp_hist
    }


# =============================================================================
# 3. NoisyMPC: Simulation with Disturbances and Measurement Noise
# =============================================================================

def run_noisympc(plant_tf, model_tf, P=[1], M=1, Tref=6.0, Ts=0.6, tend=60.0,
                 d_in_step_time=20.0, d_in_mag=0.2,
                 d_out_step_time=40.0, d_out_mag=0.3,
                 meas_noise_std=0.02):
    """
    Simulates system under:
    - Input disturbance d_in (e.g. load disturbance)
    - Output disturbance d_out (e.g. ambient temperature variation)
    - Measurement noise n (zero-mean Gaussian)
    Corresponds to Figure 1.8 and Figure 1.9 in Maciejowski.
    """
    nsteps = int(np.floor(tend / Ts))
    tvec = np.arange(nsteps) * Ts

    P = np.array(P, dtype=int)
    max_P = int(np.max(P))
    len_P = len(P)

    step_resp = model_tf.step_response(max_P)

    Theta = np.zeros((len_P, M))
    for j in range(len_P):
        pj = P[j]
        for i in range(1, M + 1):
            shift = pj - i + 1
            if 0 < shift < len(step_resp):
                Theta[j, i - 1] = step_resp[shift]
            else:
                Theta[j, i - 1] = 0.0

    errfac = np.zeros(len_P) if Tref <= 1e-6 else np.exp(-P * Ts / Tref)

    order_p = plant_tf.order
    order_m = model_tf.order
    ypast_plant = np.zeros(order_p)
    upast_plant = np.zeros(order_p)
    ypast_model = np.zeros(order_m)
    upast_model = np.zeros(order_m)

    y_true_hist = np.zeros(nsteps)
    y_meas_hist = np.zeros(nsteps)
    u_ctrl_hist = np.zeros(nsteps)
    u_plant_hist = np.zeros(nsteps)
    setpoint_val = 1.0
    sp_hist = np.ones(nsteps) * setpoint_val

    u_prev = 0.0

    for k in range(nsteps):
        t_curr = tvec[k]

        # True plant output from past state
        y_true = ypast_plant[0] if k > 0 else 0.0
        
        # Output disturbance
        d_out = d_out_mag if t_curr >= d_out_step_time else 0.0
        y_with_dist = y_true + d_out

        # Measurement noise
        noise = np.random.normal(0.0, meas_noise_std)
        y_meas = y_with_dist + noise

        y_true_hist[k] = y_with_dist
        y_meas_hist[k] = y_meas

        # Current model output
        ym_curr = ypast_model[0] if k > 0 else 0.0

        # Output disturbance estimate based on noisy measurement
        d_hat = y_meas - ym_curr

        # Reference trajectory
        error_now = setpoint_val - y_meas
        r_P = setpoint_val - error_now * errfac

        # Model free response
        ym_free = np.zeros(max_P + 1)
        yfpast = ypast_model.copy()
        ufpast = upast_model.copy()

        for step in range(1, max_P + 1):
            y_next = model_tf.simulate_step(ufpast, yfpast)
            ym_free[step] = y_next
            yfpast = np.roll(yfpast, 1)
            yfpast[0] = y_next
            ufpast = np.roll(ufpast, 1)
            ufpast[0] = u_prev

        y_pred_free_P = ym_free[P] + d_hat

        # Optimal move
        rhs = r_P - y_pred_free_P
        delta_u_traj, _, _, _ = np.linalg.lstsq(Theta, rhs, rcond=None)

        u_ctrl = u_prev + delta_u_traj[0]
        u_prev = u_ctrl

        # Input disturbance
        d_in = d_in_mag if t_curr >= d_in_step_time else 0.0
        u_actual_to_plant = u_ctrl + d_in

        u_ctrl_hist[k] = u_ctrl
        u_plant_hist[k] = u_actual_to_plant

        # Plant simulation step with u_actual_to_plant
        upast_plant = np.roll(upast_plant, 1)
        upast_plant[0] = u_actual_to_plant
        y_plant_next = plant_tf.simulate_step(upast_plant, ypast_plant)
        ypast_plant = np.roll(ypast_plant, 1)
        ypast_plant[0] = y_plant_next

        # Model simulation step with controller output u_ctrl
        upast_model = np.roll(upast_model, 1)
        upast_model[0] = u_ctrl
        y_model_next = model_tf.simulate_step(upast_model, ypast_model)
        ypast_model = np.roll(ypast_model, 1)
        ypast_model[0] = y_model_next

    return {
        't': tvec,
        'y_true': y_true_hist,
        'y_meas': y_meas_hist,
        'u_ctrl': u_ctrl_hist,
        'u_plant': u_plant_hist,
        'setpoint': sp_hist
    }


# =============================================================================
# 4. UnstaMPC: Receding Horizon Control for Unstable / Non-minimum Phase Plants
# =============================================================================

def run_unstampc(plant_tf, model_tf, P=[8], M=1, Tref=6.0, Ts=0.6, tend=50.0, setpoint_val=1.0):
    """
    Implements receding horizon control for unstable systems (e.g. Example 1.8 Helicopter).
    Key mechanism: Model Realignment.
    Because an open-loop model would diverge unbounded over time, the internal model's
    initial condition at step k is reset directly to the plant's current historical states:
        yfpast = ypast_plant.copy()
        ufpast = upast_plant.copy()
    Free response is then simulated forward for H_p steps from these realigned states.
    """
    nsteps = int(np.floor(tend / Ts))
    tvec = np.arange(nsteps) * Ts

    P = np.array(P, dtype=int)
    max_P = int(np.max(P))
    len_P = len(P)

    # Step response of model (computed by simulating model from zero initial state)
    step_resp = model_tf.step_response(max_P)

    Theta = np.zeros((len_P, M))
    for j in range(len_P):
        pj = P[j]
        for i in range(1, M + 1):
            shift = pj - i + 1
            if 0 < shift < len(step_resp):
                Theta[j, i - 1] = step_resp[shift]
            else:
                Theta[j, i - 1] = 0.0

    errfac = np.zeros(len_P) if Tref <= 1e-6 else np.exp(-P * Ts / Tref)

    order_p = plant_tf.order
    ypast_plant = np.zeros(order_p)
    upast_plant = np.zeros(order_p)

    y_plant_hist = np.zeros(nsteps)
    u_hist = np.zeros(nsteps)
    sp_hist = np.ones(nsteps) * setpoint_val

    u_prev = 0.0

    for k in range(nsteps):
        y_curr = ypast_plant[0] if k > 0 else 0.0
        y_plant_hist[k] = y_curr

        # Reference trajectory
        error_now = setpoint_val - y_curr
        r_P = setpoint_val - error_now * errfac

        # Model realignment: start free response directly from plant's true history!
        ym_free = np.zeros(max_P + 1)
        yfpast = ypast_plant.copy()
        ufpast = upast_plant.copy()

        for step in range(1, max_P + 1):
            y_next = model_tf.simulate_step(ufpast, yfpast)
            ym_free[step] = y_next
            yfpast = np.roll(yfpast, 1)
            yfpast[0] = y_next
            ufpast = np.roll(ufpast, 1)
            ufpast[0] = u_prev

        # ym_free at coincidence points P
        ym_free_P = ym_free[P]

        # Solve least-squares: Theta * Delta_U = r_P - ym_free_P
        rhs = r_P - ym_free_P
        delta_u_traj, _, _, _ = np.linalg.lstsq(Theta, rhs, rcond=None)

        u_k = u_prev + delta_u_traj[0]
        u_hist[k] = u_k
        u_prev = u_k

        # Simulate plant one step
        upast_plant = np.roll(upast_plant, 1)
        upast_plant[0] = u_k
        y_plant_next = plant_tf.simulate_step(upast_plant, ypast_plant)
        ypast_plant = np.roll(ypast_plant, 1)
        ypast_plant[0] = y_plant_next

    return {
        't': tvec,
        'y_plant': y_plant_hist,
        'u': u_hist,
        'setpoint': sp_hist
    }


# =============================================================================
# Numerical Experiments Execution & Plotting
# =============================================================================

def execute_all_experiments(save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    print("===================================================================")
    print("Executing Chapter 1 Numerical Experiments (Maciejowski 2002)...")
    print("===================================================================")

    # -------------------------------------------------------------------------
    # Experiment 1: Example 1.6 & Figures 1.5, 1.6, 1.7 (Offset-free tracking)
    # -------------------------------------------------------------------------
    print("\n[Experiment 1] Replicating Example 1.6 & Figures 1.5, 1.6, 1.7...")
    Ts = 0.6
    Tref = 6.0
    tend = 60.0
    P = [1]
    M = 1

    # In basicmpc.m:
    # nump = 1, denp = [1, -1.4, 0.45] -> tf(1, [1, -1.4, 0.45], Ts)
    # G(z) = 1 / (z^2 - 1.4z + 0.45) = z^{-2} / (1 - 1.4 z^{-1} + 0.45 z^{-2})
    # Denominator degree 2, numerator: [0, 0, 1]
    plant_exact = DiscreteTF(num=[0, 0, 1.0], den=[1.0, -1.4, 0.45], Ts=Ts)
    model_exact = DiscreteTF(num=[0, 0, 1.0], den=[1.0, -1.4, 0.45], Ts=Ts)

    # Mismatched model: den = [1, -1.4, 0.46] -> gain = 1 / (1 - 1.4 + 0.46) = 16.67
    model_mismatch = DiscreteTF(num=[0, 0, 1.0], den=[1.0, -1.4, 0.46], Ts=Ts)

    # 1. Nominal run (Figure 1.5)
    res_exact = run_basicmpc(plant_exact, model_exact, P=P, M=M, Tref=Tref, Ts=Ts, tend=tend)
    # 2. Mismatch without d_hat (Figure 1.6: Offset exists!)
    res_mismatch_basic = run_basicmpc(plant_exact, model_mismatch, P=P, M=M, Tref=Tref, Ts=Ts, tend=tend)
    # 3. Mismatch with d_hat (Figure 1.7: Offset-free tracking restored!)
    res_mismatch_track = run_trackmpc(plant_exact, model_mismatch, P=P, M=M, Tref=Tref, Ts=Ts, tend=tend)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Maciejowski Ch 1 - Experiment 1: Offset-free Tracking under Plant-Model Mismatch", fontsize=14)

    # Row 1: Exact model (Fig 1.5)
    axes[0, 0].plot(res_exact['t'], res_exact['y_plant'], 'b-', label='Plant Output $y(k)$', lw=1.8)
    axes[0, 0].plot(res_exact['t'], res_exact['setpoint'], 'k--', label='Setpoint $s(k)$')
    axes[0, 0].set_title("Exact Model: Perfect Tracking (Fig 1.5)")
    axes[0, 0].set_ylabel("Output")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].step(res_exact['t'], res_exact['u'], 'g-', where='post', lw=1.5)
    axes[0, 1].set_title("Exact Model: Control Input $u(k)$")
    axes[0, 1].set_ylabel("Input")
    axes[0, 1].grid(True)

    # Row 2: BasicMPC with Mismatch (Fig 1.6)
    axes[1, 0].plot(res_mismatch_basic['t'], res_mismatch_basic['y_plant'], 'r-', label='Plant Output $y(k)$', lw=1.8)
    axes[1, 0].plot(res_mismatch_basic['t'], res_mismatch_basic['setpoint'], 'k--', label='Setpoint $s(k)$')
    axes[1, 0].set_title("BasicMPC with Gain Mismatch: Nonzero Offset (Fig 1.6)")
    axes[1, 0].set_ylabel("Output")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(res_mismatch_basic['t'], res_mismatch_basic['u'], 'g-', where='post', lw=1.5)
    axes[1, 1].set_title("BasicMPC with Mismatch: Control Input $u(k)$")
    axes[1, 1].set_ylabel("Input")
    axes[1, 1].grid(True)

    # Row 3: TrackMPC with Mismatch (Fig 1.7)
    axes[2, 0].plot(res_mismatch_track['t'], res_mismatch_track['y_plant'], 'b-', label='Plant Output $y(k)$', lw=1.8)
    axes[2, 0].plot(res_mismatch_track['t'], res_mismatch_track['setpoint'], 'k--', label='Setpoint $s(k)$')
    axes[2, 0].set_title("TrackMPC with Disturbance Estimation: Offset-Free (Fig 1.7)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Output")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    axes[2, 1].step(res_mismatch_track['t'], res_mismatch_track['u'], 'g-', where='post', lw=1.5)
    axes[2, 1].set_title("TrackMPC with Mismatch: Control Input $u(k)$")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Input")
    axes[2, 1].grid(True)

    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "exp1_plant_mismatch.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"-> Saved Experiment 1 plot to: {fig1_path}")

    # -------------------------------------------------------------------------
    # Experiment 2: Example 1.7 & Figure 1.9 (Disturbances & Measurement Noise)
    # -------------------------------------------------------------------------
    print("\n[Experiment 2] Replicating Example 1.7 & Figure 1.9 (Disturbances & Noise)...")
    res_noisy = run_noisympc(plant_exact, model_mismatch, P=[1], M=1, Tref=6.0, Ts=0.6, tend=70.0,
                             d_in_step_time=20.0, d_in_mag=0.05,
                             d_out_step_time=45.0, d_out_mag=0.2,
                             meas_noise_std=0.02)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Maciejowski Ch 1 - Experiment 2: Noisy & Disturbed Plant (Fig 1.9 Replication)", fontsize=14)

    axes[0].plot(res_noisy['t'], res_noisy['y_true'], 'b-', label='Plant Output $y(t)$ (solid)', lw=1.8)
    axes[0].plot(res_noisy['t'], res_noisy['y_meas'], 'r:', label='Measured Output $y_m(t)$ (dotted)', alpha=0.7)
    axes[0].plot(res_noisy['t'], res_noisy['setpoint'], 'k--', label='Setpoint (dashed)')
    axes[0].set_title("Outputs with Input/Output Disturbances & Noise")
    axes[0].set_ylabel("Output")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(res_noisy['t'], res_noisy['u_ctrl'], 'b-', label='Controller Output $u(k)$ (solid)', lw=1.8)
    axes[1].plot(res_noisy['t'], res_noisy['u_plant'], 'g:', label='Plant Input with Disturbance $u(k)+d_{in}$ (dotted)', lw=1.5)
    axes[1].set_title("Control Inputs")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Input")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "exp2_noisy_disturbance.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"-> Saved Experiment 2 plot to: {fig2_path}")

    # -------------------------------------------------------------------------
    # Experiment 3: Example 1.8 & Figure 1.10 (Unstable & Non-minimum Phase Plant)
    # -------------------------------------------------------------------------
    print("\n[Experiment 3] Replicating Example 1.8 & Figure 1.10 (Unstable Helicopter)...")
    # G(s) = 9.8 * (s^2 - 0.5 s + 6.3) / [ (s + 0.6565) * (s^2 - 0.2366 s + 0.1493) ]
    num_s = [9.8, -4.9, 61.74]
    den_s = np.convolve([1.0, 0.6565], [1.0, -0.2366, 0.1493])
    
    Ts_heli = 0.6
    Tref_heli = 6.0
    P_heli = [8]
    M_heli = 1
    tend_heli = 40.0

    # Discretize via cont2discrete ZOH
    sys_d = signal.cont2discrete((num_s, den_s), Ts_heli, method='zoh')
    num_z = sys_d[0].flatten()
    den_z = sys_d[1].flatten()

    print(f"Helicopter discrete transfer function (Ts={Ts_heli}s):")
    print(f"  Numerator:   {num_z}")
    print(f"  Denominator: {den_z}")
    poles_z = np.roots(den_z)
    print(f"  Discrete poles: {poles_z} (Magnitudes: {np.abs(poles_z)})")
    print(f"  Unstable poles present: {any(np.abs(poles_z) > 1.0)}")

    plant_heli = DiscreteTF(num=num_z, den=den_z, Ts=Ts_heli)
    model_heli = DiscreteTF(num=num_z, den=den_z, Ts=Ts_heli)

    res_heli = run_unstampc(plant_heli, model_heli, P=P_heli, M=M_heli,
                           Tref=Tref_heli, Ts=Ts_heli, tend=tend_heli, setpoint_val=1.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Maciejowski Ch 1 - Experiment 3: Control of Unstable Helicopter (Fig 1.10 Replication)", fontsize=14)

    axes[0].plot(res_heli['t'], res_heli['y_plant'], 'b-', label='Helicopter Forward Speed $y(t)$', lw=1.8)
    axes[0].plot(res_heli['t'], res_heli['setpoint'], 'k--', label='Speed Setpoint')
    axes[0].set_title(f"Forward Speed Response ($T_{{ref}}=6s, T_s=0.6s, P_1=8, H_u=1$)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].step(res_heli['t'], res_heli['u'], 'g-', where='post', label='Rotor Angle $u(t)$', lw=1.5)
    axes[1].set_title("Control Input: Rotor Angle")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Rotor Angle (rad)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "exp3_unstable_helicopter.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"-> Saved Experiment 3 plot to: {fig3_path}")
    print("\nAll Chapter 1 experiments completed successfully!")

if __name__ == "__main__":
    execute_all_experiments(save_dir="/home/kuang/Desktop/mpc/ch01_introduction")
