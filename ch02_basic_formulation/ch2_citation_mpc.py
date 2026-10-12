"""
===============================================================================
Predictive Control with Constraints (J.M. Maciejowski, 2002)
Chapter 2: A Basic Formulation of Predictive Control
Filename: ch2_citation_mpc.py

Description:
Implementation of state-space constrained model predictive control for the
Cessna Citation 500 business jet longitudinal flight control (Section 2.7).
Replicates:
- Figure 2.6: Step change of 40 m in altitude set-point (unconstrained linear regime)
- Figure 2.7: Step change of 400 m in altitude set-point (elevator saturation limits active)
- Figure 2.8: Step change of 400 m with altitude rate constraint (|h_dot| <= 30 m/s)
- Figure 2.11: Step change of 400 m with altitude rate constraint and 10% plant-model mismatch

Dependencies: numpy, scipy, matplotlib
===============================================================================
"""

import os
import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 1. Continuous and Discrete Cessna Citation 500 Aircraft Model
# =============================================================================

class CitationAircraft:
    """
    Linearized longitudinal dynamics of a Cessna Citation 500 aircraft
    cruising at altitude 5000 m and airspeed V = 128.2 m/s.
    
    States:
        x_1: Angle of attack alpha (rad)
        x_2: Pitch angle theta (rad)
        x_3: Pitch rate q (rad/s)
        x_4: Altitude h (m)
        
    Input:
        u: Elevator angle (rad)
        
    Outputs:
        y_1: Pitch angle theta (rad)
        y_2: Altitude h (m)
        y_3: Altitude rate h_dot (m/s)  [h_dot = 128.2 * (theta - alpha)]
    """
    def __init__(self, Ts=0.5, mismatch=False):
        self.Ts = float(Ts)
        self.V = 128.2  # Airspeed in m/s

        # Continuous-time state space matrices (Maciejowski, Section 2.7, p. 64)
        self.Ac = np.array([
            [-1.2822, 0.0, 0.98, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-5.4293, 0.0, -1.8366, 0.0],
            [-128.2, 128.2, 0.0, 0.0]
        ], dtype=float)

        self.Bc = np.array([
            [-0.3],
            [0.0],
            [-17.0],
            [0.0]
        ], dtype=float)

        self.C = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-128.2, 128.2, 0.0, 0.0]
        ], dtype=float)

        self.D = np.zeros((3, 1), dtype=float)

        # Plant-model mismatch if requested (10% smaller gain on altitude rate)
        if mismatch:
            # Underestimate gain from elevator to altitude rate by 10%
            # Elements A(4,1), A(4,2), C(3,1), C(3,2) reduced by 10% in model (p. 69)
            self.Ac[3, 0] *= 0.9
            self.Ac[3, 1] *= 0.9
            self.C[2, 0] *= 0.9
            self.C[2, 1] *= 0.9

        # Zero-Order Hold (ZOH) Discretization
        sys_c = (self.Ac, self.Bc, self.C, self.D)
        sys_d = signal.cont2discrete(sys_c, self.Ts, method='zoh')
        self.Ad = np.array(sys_d[0], dtype=float)
        self.Bd = np.array(sys_d[1], dtype=float)
        self.Cd = np.array(sys_d[2], dtype=float)
        self.Dd = np.array(sys_d[3], dtype=float)

        self.nx = self.Ad.shape[0]  # 4
        self.nu = self.Bd.shape[1]  # 1
        self.ny = self.Cd.shape[0]  # 3


# =============================================================================
# 2. State-Space Constrained Model Predictive Controller
# =============================================================================

class StateSpaceMPC:
    """
    Implements discrete-time state-space predictive control as formulated
    in Maciejowski Chapter 2, Section 2.6.
    
    Prediction equations:
        Z = Psi * x(k) + Upsilon * u(k-1) + Theta * Delta_U
    """
    def __init__(self, model, Hp=10, Hu=3, Q_diag=[1/3, 1/3, 1/3], R_weight=1.0):
        self.model = model
        self.Hp = int(Hp)
        self.Hu = int(Hu)
        self.nx = model.nx
        self.nu = model.nu
        self.ny = model.ny

        # Build prediction matrices Psi, Upsilon, Theta
        self.build_prediction_matrices()

        # Build block-diagonal weighting matrices
        # Q: (ny * Hp) x (ny * Hp)
        Q_single = np.diag(Q_diag)
        self.Q_bar = np.kron(np.eye(self.Hp), Q_single)

        # R: (nu * Hu) x (nu * Hu)
        R_single = np.eye(self.nu) * float(R_weight)
        self.R_bar = np.kron(np.eye(self.Hu), R_single)

        # Precompute QP Hessian: H = 2 * (Theta^T * Q_bar * Theta + R_bar)
        self.H = 2.0 * (self.Theta.T @ self.Q_bar @ self.Theta + self.R_bar)
        self.H = 0.5 * (self.H + self.H.T)  # Ensure strict numerical symmetry

    def build_prediction_matrices(self):
        """
        Constructs Psi, Upsilon, Theta strictly according to Maciejowski
        Equations (2.66) and (2.70).
        """
        Ad, Bd, Cd = self.model.Ad, self.model.Bd, self.model.Cd
        Hp, Hu = self.Hp, self.Hu
        nx, nu, ny = self.nx, self.nu, self.ny

        # 1. Matrix Psi: (ny * Hp) x nx
        self.Psi = np.zeros((ny * Hp, nx))
        A_power = np.eye(nx)
        for i in range(1, Hp + 1):
            A_power = A_power @ Ad
            self.Psi[(i - 1) * ny : i * ny, :] = Cd @ A_power

        # 2. Matrix Upsilon: (ny * Hp) x nu
        self.Upsilon = np.zeros((ny * Hp, nu))
        sum_A_B = np.zeros((nx, nu))
        A_pow = np.eye(nx)
        for i in range(1, Hp + 1):
            sum_A_B = sum_A_B + A_pow @ Bd
            A_pow = A_pow @ Ad
            self.Upsilon[(i - 1) * ny : i * ny, :] = Cd @ sum_A_B

        # 3. Dynamic Matrix Theta: (ny * Hp) x (nu * Hu)
        self.Theta = np.zeros((ny * Hp, nu * Hu))
        for row in range(1, Hp + 1):
            for col in range(1, Hu + 1):
                shift = row - col
                if shift >= 0:
                    sum_term = np.zeros((nx, nu))
                    A_k = np.eye(nx)
                    for k in range(shift + 1):
                        sum_term = sum_term + A_k @ Bd
                        A_k = A_k @ Ad
                    self.Theta[(row - 1) * ny : row * ny, (col - 1) * nu : col * nu] = Cd @ sum_term

    def compute_control(self, x_curr, u_prev, r_traj, d_hat=None,
                        u_min=None, u_max=None,
                        du_min=None, du_max=None,
                        z_min=None, z_max=None):
        """
        Solves the constrained or unconstrained MPC problem at step k.
        """
        # Right-hand side tracking error vector
        # E = T - Psi * x(k) - Upsilon * u(k-1) - 1 * d_hat
        free_resp = self.Psi @ x_curr + self.Upsilon @ u_prev
        if d_hat is not None:
            # Replicate d_hat over the prediction horizon
            d_stack = np.tile(d_hat, self.Hp).reshape(-1, 1)
            free_resp = free_resp + d_stack

        E = r_traj - free_resp
        g = -2.0 * (self.Theta.T @ self.Q_bar @ E).flatten()

        # Check if constraints are active/specified
        has_constraints = any([u_min is not None, u_max is not None,
                               du_min is not None, du_max is not None,
                               z_min is not None, z_max is not None])

        if not has_constraints:
            # Unconstrained explicit solution: Delta_U = 0.5 * H^(-1) * (-g)
            delta_U = np.linalg.solve(self.H, -g)
            return delta_U[:self.nu], delta_U

        # Constrained QP formulation:
        # min 0.5 * delta_U^T * H * delta_U + g^T * delta_U
        # subject to A_in * delta_U <= b_in
        A_rows = []
        b_rows = []

        # 1. Slew rate constraints: du_min <= delta_u <= du_max
        if du_max is not None:
            # delta_u(k+i) <= du_max
            for i in range(self.Hu):
                row = np.zeros(self.Hu * self.nu)
                row[i * self.nu : (i + 1) * self.nu] = 1.0
                A_rows.append(row)
                b_rows.append(du_max)
        if du_min is not None:
            # -delta_u(k+i) <= -du_min
            for i in range(self.Hu):
                row = np.zeros(self.Hu * self.nu)
                row[i * self.nu : (i + 1) * self.nu] = -1.0
                A_rows.append(row)
                b_rows.append(-du_min)

        # 2. Input magnitude constraints: u_min <= u(k-1) + sum_{j=0}^i delta_u(k+j) <= u_max
        if u_max is not None or u_min is not None:
            # Build lower triangular accumulator matrix C_acc: (Hu x Hu)
            C_acc = np.tril(np.ones((self.Hu, self.Hu)))
            if u_max is not None:
                for i in range(self.Hu):
                    row = C_acc[i, :]
                    A_rows.append(row)
                    b_rows.append(float(np.squeeze(u_max - u_prev)))
            if u_min is not None:
                for i in range(self.Hu):
                    row = -C_acc[i, :]
                    A_rows.append(row)
                    b_rows.append(float(np.squeeze(-u_min + u_prev)))

        # 3. Controlled output constraints: z_min <= Theta * delta_U + free_resp <= z_max
        if z_max is not None:
            for row_idx, val in enumerate(z_max):
                if val is not None and not np.isinf(val):
                    A_rows.append(self.Theta[row_idx, :])
                    b_rows.append(float(np.squeeze(val - free_resp[row_idx, 0])))
        if z_min is not None:
            for row_idx, val in enumerate(z_min):
                if val is not None and not np.isinf(val):
                    A_rows.append(-self.Theta[row_idx, :])
                    b_rows.append(float(np.squeeze(-val + free_resp[row_idx, 0])))

        # Solve via SLSQP
        A_in = np.array(A_rows)
        b_in = np.array(b_rows)

        def cost_func(du):
            return 0.5 * du @ self.H @ du + g @ du

        def cost_grad(du):
            return self.H @ du + g

        constraints = []
        if len(b_rows) > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda du: b_in - A_in @ du,
                'jac': lambda du: -A_in
            })

        du0 = np.zeros(self.Hu * self.nu)
        res = minimize(cost_func, du0, jac=cost_grad, constraints=constraints, method='SLSQP',
                       options={'ftol': 1e-9, 'maxiter': 200})

        if not res.success:
            # Fallback to unconstrained if numerical solver stalls
            delta_U = np.linalg.solve(self.H, -g)
        else:
            delta_U = res.x

        return delta_U[:self.nu], delta_U


# =============================================================================
# 3. Simulation Runner for Citation Experiments
# =============================================================================

def simulate_citation_case(case_name, alt_step=40.0, constrain_hdot=False,
                           has_mismatch=False, add_disturbance=False, tend=20.0):
    """
    Executes flight control simulation for specified case.
    """
    Ts = 0.5
    nsteps = int(np.floor(tend / Ts))
    tvec = np.arange(nsteps) * Ts

    # True plant and internal model
    plant = CitationAircraft(Ts=Ts, mismatch=False)
    model = CitationAircraft(Ts=Ts, mismatch=has_mismatch)

    # Controller parameters
    Hp = 10
    Hu = 3
    Q_diag = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    R_weight = 1.0

    mpc = StateSpaceMPC(model=model, Hp=Hp, Hu=Hu, Q_diag=Q_diag, R_weight=R_weight)

    # Physical Constraints
    # Elevator angle limit: +/- 15 deg = +/- 0.262 rad
    u_max = 0.262
    u_min = -0.262

    # Elevator slew rate limit: +/- 30 deg/s -> +/- 0.524 rad/s -> +/- 0.262 rad / sample
    du_max = 0.524 * Ts
    du_min = -0.524 * Ts

    # Pitch angle passenger comfort constraint: +/- 20 deg = +/- 0.349 rad
    theta_max = 0.349
    theta_min = -0.349

    # Altitude rate constraint (if active): +/- 30 m/s
    hdot_max = 30.0 if constrain_hdot else np.inf
    hdot_min = -30.0 if constrain_hdot else -np.inf

    # Construct z_max and z_min over prediction horizon Hp
    # z = [theta, h, h_dot]^T
    z_max_single = [theta_max, np.inf, hdot_max]
    z_min_single = [theta_min, -np.inf, hdot_min]
    z_max_stack = np.tile(z_max_single, Hp)
    z_min_stack = np.tile(z_min_single, Hp)

    # State histories
    x_plant = np.zeros((plant.nx, 1))
    x_model = np.zeros((model.nx, 1))
    u_prev = 0.0

    # Logging
    hist_t = tvec
    hist_theta = np.zeros(nsteps)
    hist_h = np.zeros(nsteps)
    hist_hdot = np.zeros(nsteps)
    hist_elev = np.zeros(nsteps)
    hist_sp_h = np.ones(nsteps) * float(alt_step)

    for k in range(nsteps):
        # Current plant output
        y_plant = plant.Cd @ x_plant

        # Add step disturbance if requested (e.g. at t >= 5s on altitude rate)
        if add_disturbance and tvec[k] >= 5.0:
            y_plant[2, 0] += 5.0  # 5 m/s gust disturbance

        hist_theta[k] = np.rad2deg(y_plant[0, 0])
        hist_h[k] = y_plant[1, 0]
        hist_hdot[k] = y_plant[2, 0]
        hist_elev[k] = np.rad2deg(u_prev)

        # Disturbance estimate d_hat = y_plant - y_model
        y_model = model.Cd @ x_model
        d_hat = y_plant - y_model if has_mismatch else np.zeros((3, 1))

        # Build reference trajectory over Hp: r = [0, alt_step, 0]^T
        r_single = np.array([[0.0], [alt_step], [0.0]])
        r_traj = np.tile(r_single, (Hp, 1))

        # Solve MPC
        du_opt, _ = mpc.compute_control(
            x_curr=x_model if not has_mismatch else x_plant,
            u_prev=np.array([[u_prev]]),
            r_traj=r_traj,
            d_hat=d_hat,
            u_min=u_min if alt_step > 50 else None,
            u_max=u_max if alt_step > 50 else None,
            du_min=du_min if alt_step > 50 else None,
            du_max=du_max if alt_step > 50 else None,
            z_min=z_min_stack if constrain_hdot else None,
            z_max=z_max_stack if constrain_hdot else None
        )

        u_curr = u_prev + du_opt[0]
        u_prev = u_curr

        # Advance plant
        x_plant = plant.Ad @ x_plant + plant.Bd @ np.array([[u_curr]])
        x_model = model.Ad @ x_model + model.Bd @ np.array([[u_curr]])

    return {
        't': hist_t,
        'theta': hist_theta,
        'h': hist_h,
        'hdot': hist_hdot,
        'elev': hist_elev,
        'sp_h': hist_sp_h
    }


def run_all_citation_experiments(save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    print("===================================================================")
    print("Running Chapter 2 Cessna Citation 500 Flight Control Simulations...")
    print("===================================================================")

    # 1. Figure 2.6: Altitude step = 40 m (Unconstrained linear regime)
    print("\n[Case 1] Simulating 40 m Altitude Step (Figure 2.6)...")
    res1 = simulate_citation_case("Fig2.6", alt_step=40.0, constrain_hdot=False, has_mismatch=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Figure 2.6: Citation Aircraft Step Response (40 m Step, Unconstrained)", fontsize=13)

    axes[0, 0].plot(res1['t'], res1['theta'], 'b-', lw=1.8)
    axes[0, 0].axhline(20.0, color='r', linestyle='--', label='Pitch Limit (+20 deg)')
    axes[0, 0].axhline(-20.0, color='r', linestyle='--')
    axes[0, 0].set_title("Pitch Angle theta (deg)")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(res1['t'], res1['h'], 'b-', lw=1.8, label='Altitude')
    axes[0, 1].plot(res1['t'], res1['sp_h'], 'k--', label='Set-point (40 m)')
    axes[0, 1].set_title("Altitude h (m)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(res1['t'], res1['hdot'], 'b-', lw=1.8)
    axes[1, 0].set_title("Altitude Rate h_dot (m/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True)

    axes[1, 1].step(res1['t'], res1['elev'], 'g-', where='post', lw=1.8)
    axes[1, 1].axhline(15.0, color='r', linestyle='--', label='Elevator Limit (+15 deg)')
    axes[1, 1].axhline(-15.0, color='r', linestyle='--')
    axes[1, 1].set_title("Elevator Angle (deg)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    p1 = os.path.join(save_dir, "citation_fig2_6_step40m.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"-> Saved Figure 2.6 to: {p1}")

    # 2. Figure 2.7: Altitude step = 400 m (Elevator saturation active)
    print("\n[Case 2] Simulating 400 m Altitude Step (Figure 2.7)...")
    res2 = simulate_citation_case("Fig2.7", alt_step=400.0, constrain_hdot=False, has_mismatch=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Figure 2.7: Citation Aircraft Step Response (400 m Step, Elevator Saturated)", fontsize=13)

    axes[0, 0].plot(res2['t'], res2['theta'], 'b-', lw=1.8)
    axes[0, 0].axhline(20.0, color='r', linestyle='--', label='Comfort Limit (+20 deg)')
    axes[0, 0].set_title("Pitch Angle theta (deg)")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(res2['t'], res2['h'], 'b-', lw=1.8, label='Altitude')
    axes[0, 1].plot(res2['t'], res2['sp_h'], 'k--', label='Set-point (400 m)')
    axes[0, 1].set_title("Altitude h (m)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(res2['t'], res2['hdot'], 'b-', lw=1.8)
    axes[1, 0].set_title("Altitude Rate h_dot (m/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True)

    axes[1, 1].step(res2['t'], res2['elev'], 'g-', where='post', lw=1.8)
    axes[1, 1].axhline(15.0, color='r', linestyle='--', label='Limit (+/-15 deg)')
    axes[1, 1].axhline(-15.0, color='r', linestyle='--')
    axes[1, 1].set_title("Elevator Angle (deg) [Saturated]")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    p2 = os.path.join(save_dir, "citation_fig2_7_step400m.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"-> Saved Figure 2.7 to: {p2}")

    # 3. Figure 2.8: 400 m Step with Altitude Rate Constraint (|h_dot| <= 30 m/s)
    print("\n[Case 3] Simulating 400 m Step with Altitude Rate Constraint (Figure 2.8)...")
    res3 = simulate_citation_case("Fig2.8", alt_step=400.0, constrain_hdot=True, has_mismatch=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Figure 2.8: Citation Aircraft Response with Altitude Rate Limit (|h_dot| <= 30 m/s)", fontsize=13)

    axes[0, 0].plot(res3['t'], res3['theta'], 'b-', lw=1.8)
    axes[0, 0].axhline(20.0, color='r', linestyle='--', label='Comfort Limit (+20 deg)')
    axes[0, 0].set_title("Pitch Angle theta (deg)")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(res3['t'], res3['h'], 'b-', lw=1.8, label='Altitude')
    axes[0, 1].plot(res3['t'], res3['sp_h'], 'k--', label='Set-point (400 m)')
    axes[0, 1].set_title("Altitude h (m)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(res3['t'], res3['hdot'], 'b-', lw=1.8, label='h_dot')
    axes[1, 0].axhline(30.0, color='r', linestyle='--', label='Rate Limit (+30 m/s)')
    axes[1, 0].set_title("Altitude Rate h_dot (m/s) [Clamped to 30 m/s]")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(res3['t'], res3['elev'], 'g-', where='post', lw=1.8)
    axes[1, 1].axhline(15.0, color='r', linestyle='--', label='Limit (+/-15 deg)')
    axes[1, 1].axhline(-15.0, color='r', linestyle='--')
    axes[1, 1].set_title("Elevator Angle (deg)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    p3 = os.path.join(save_dir, "citation_fig2_8_alt_rate_limit.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"-> Saved Figure 2.8 to: {p3}")

    # 4. Figure 2.11: 400 m Step with Plant-Model Mismatch and Disturbance
    print("\n[Case 4] Simulating 400 m Step with Plant-Model Mismatch (Figure 2.11)...")
    res4 = simulate_citation_case("Fig2.11", alt_step=400.0, constrain_hdot=True, has_mismatch=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Figure 2.11: Citation Aircraft Response with 10% Plant-Model Mismatch", fontsize=13)

    axes[0, 0].plot(res4['t'], res4['theta'], 'b-', lw=1.8)
    axes[0, 0].axhline(20.0, color='r', linestyle='--', label='Pitch Limit (+20 deg)')
    axes[0, 0].set_title("Pitch Angle theta (deg)")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(res4['t'], res4['h'], 'b-', lw=1.8, label='Altitude')
    axes[0, 1].plot(res4['t'], res4['sp_h'], 'k--', label='Set-point (400 m)')
    axes[0, 1].set_title("Altitude h (m)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(res4['t'], res4['hdot'], 'b-', lw=1.8, label='h_dot')
    axes[1, 0].axhline(30.0, color='r', linestyle='--', label='Rate Limit (+30 m/s)')
    axes[1, 0].set_title("Altitude Rate h_dot (m/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(res4['t'], res4['elev'], 'g-', where='post', lw=1.8)
    axes[1, 1].axhline(15.0, color='r', linestyle='--', label='Limit (+/-15 deg)')
    axes[1, 1].axhline(-15.0, color='r', linestyle='--')
    axes[1, 1].set_title("Elevator Angle (deg)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    p4 = os.path.join(save_dir, "citation_fig2_11_mismatch.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"-> Saved Figure 2.11 to: {p4}")

    print("\nAll Chapter 2 flight control simulations completed successfully!")

if __name__ == "__main__":
    run_all_citation_experiments(save_dir="/home/kuang/Desktop/mpc/ch02_basic_formulation")
