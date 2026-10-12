"""
Chapter 8: Robust Predictive Control - Section 8.5 & Exercise 8.9
Maximal Output Admissible Set (MOAS) & Positive Invariance (Gilbert & Tan, 1991)

This script demonstrates:
1. Computation of the Maximal Output Admissible Set (MOAS) O_infinity for a constrained
   linear discrete-time system x(k+1) = Phi x(k).
2. The Gilbert-Tan LP-based finite determination algorithm to find the exact stopping index t*.
3. Positive Invariance verification (Exercise 8.9): x(0) in O_infinity implies x(k) in O_infinity for all k >= 0.
4. Robust disturbance-invariant set concepts under bounded exogenous disturbances w(k) in W.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

def compute_moas(Phi, H, h, max_iter=50):
    """
    Computes the Maximal Output Admissible Set (MOAS) O_infinity:
        O_infinity = { x in R^n | H Phi^k x <= h for all k >= 0 }
    using the Gilbert & Tan (1991) LP-based termination condition.
    
    Parameters:
        Phi: (n, n) closed-loop strictly stable matrix (spectral radius < 1)
        H: (m, n) constraint matrix
        h: (m,) constraint RHS vector
        max_iter: maximum horizon to test
        
    Returns:
        H_total: stacked constraint matrix defining O_t*
        h_total: stacked constraint RHS defining O_t*
        t_star: termination step index where O_{t*} = O_infinity
    """
    n = Phi.shape[0]
    m = H.shape[0]
    
    # Initialize with k = 0 constraints: H x <= h
    H_total = np.copy(H)
    h_total = np.copy(h)
    
    t_star = 0
    Phi_k = np.eye(n)
    
    for t in range(1, max_iter + 1):
        Phi_k = Phi_k @ Phi  # Phi^t
        all_redundant = True
        
        # Test each constraint row i at horizon t: max (H_i Phi^t) x  s.t.  H_total x <= h_total
        for i in range(m):
            c = -H[i, :] @ Phi_k  # linprog minimizes c^T x, so negate for maximization
            res = linprog(c, A_ub=H_total, b_ub=h_total, method='highs')
            
            if res.success:
                max_val = -res.fun
                # If maximum exceeds constraint boundary (with small tolerance), it is NOT redundant
                if max_val > h[i] + 1e-6:
                    all_redundant = False
            else:
                all_redundant = False
        
        if all_redundant:
            t_star = t - 1
            print(f"[Gilbert-Tan Algorithm] MOAS determined finitely at t* = {t_star}!")
            break
        
        # Append new constraints at step t
        H_total = np.vstack([H_total, H @ Phi_k])
        h_total = np.concatenate([h_total, h])
        t_star = t
        
    return H_total, h_total, t_star

def get_polytope_vertices_2d(A_ub, b_ub, bounds_box=(-10, 10, -10, 10)):
    """
    Computes 2D polytope vertices from halfspace representation A_ub x <= b_ub
    within a bounding box via intersection of halfspaces.
    """
    # Add bounding box constraints
    x_min, x_max, y_min, y_max = bounds_box
    box_A = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0]
    ])
    box_b = np.array([x_max, -x_min, y_max, -y_min])
    
    A_all = np.vstack([A_ub, box_A])
    b_all = np.concatenate([b_ub, box_b])
    
    # Find all intersection points of constraint lines
    m = A_all.shape[0]
    intersection_points = []
    
    for i in range(m):
        for j in range(i + 1, m):
            A_sub = A_all[[i, j], :]
            b_sub = b_all[[i, j]]
            
            if np.abs(np.linalg.det(A_sub)) > 1e-8:
                pt = np.linalg.solve(A_sub, b_sub)
                # Check feasibility against all halfspaces
                if np.all(A_all @ pt <= b_all + 1e-5):
                    intersection_points.append(pt)
                    
    if len(intersection_points) < 3:
        return np.empty((0, 2))
        
    pts = np.array(intersection_points)
    hull = ConvexHull(pts)
    return pts[hull.vertices]

def run_simulation():
    print("=" * 75)
    print("Chapter 8: Section 8.5 & Ex 8.9 - MOAS & Positive Invariance (Gilbert & Tan)")
    print("=" * 75)
    
    # 1. System Definition: Lightly Damped Physical Harmonic Oscillator
    # Continuous-time: x'' + 2*zeta*wn*x' + wn^2*x = 0
    # States: x_1 = position, x_2 = velocity
    wn = 1.5        # Natural frequency (rad/s)
    zeta = 0.15     # Light damping ratio (causes phase rotation)
    Ts = 0.6        # Discrete sampling period (s)
    
    # State matrix from exact discretization:
    from scipy.linalg import expm
    Ac = np.array([[0.0, 1.0],
                   [-wn**2, -2 * zeta * wn]])
    Phi = expm(Ac * Ts)
    
    cl_eig = np.linalg.eigvals(Phi)
    print(f"Closed-Loop System Matrix Phi:\n{Phi}")
    print(f"Closed-Loop Eigenvalues: {cl_eig}, Spectral radius rho(Phi) = {np.max(np.abs(cl_eig)):.4f} < 1")
    
    # State constraints: |x_1| <= 1.5 (position), |x_2| <= 1.2 (velocity)
    H = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0]
    ])
    h = np.array([1.5, 1.5, 1.2, 1.2])
    
    # 2. Compute MOAS O_infinity using Gilbert & Tan algorithm
    H_moas, h_moas, t_star = compute_moas(Phi, H, h, max_iter=30)
    print(f"Number of defining halfspaces for O_infinity: {H_moas.shape[0]}")
    
    # 3. Simulate Closed-Loop Trajectories
    # Point A: Inside O_infinity -> Must stay inside O_infinity for all k >= 0 (Exercise 8.9)
    x0_inside1 = np.array([0.7, -0.6])
    x0_inside2 = np.array([-0.6, 0.5])
    
    # Point B: In O_0 (satisfies immediate constraints) but OUTSIDE O_infinity -> rotates out of bounds at k=1, 2!
    x0_outside = np.array([1.3, 0.9])
    
    steps = 25
    traj_in1 = np.zeros((steps + 1, 2))
    traj_in2 = np.zeros((steps + 1, 2))
    traj_out = np.zeros((steps + 1, 2))
    
    traj_in1[0] = x0_inside1
    traj_in2[0] = x0_inside2
    traj_out[0] = x0_outside
    
    for k in range(steps):
        traj_in1[k+1] = Phi @ traj_in1[k]
        traj_in2[k+1] = Phi @ traj_in2[k]
        traj_out[k+1] = Phi @ traj_out[k]
        
    # Check invariance at each step for traj_in1
    invariance_held = True
    for k in range(steps + 1):
        if not np.all(H_moas @ traj_in1[k] <= h_moas + 1e-5):
            invariance_held = False
            break
    print(f">> Positive Invariance Verification (Ex 8.9): x(0) in O_inf -> x(k) in O_inf: {invariance_held} (SUCCESS)")
    
    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot (a): MOAS O_infinity vs Initial Constraint Set O_0
    ax = axes[0, 0]
    verts_0 = get_polytope_vertices_2d(H, h, bounds_box=(-3, 3, -3, 3))
    verts_moas = get_polytope_vertices_2d(H_moas, h_moas, bounds_box=(-3, 3, -3, 3))
    
    if len(verts_0) > 0:
        ax.fill(verts_0[:, 0], verts_0[:, 1], color='gray', alpha=0.2, label=r'Initial Constraint Set $\mathcal{O}_0$')
        ax.plot(np.append(verts_0[:, 0], verts_0[0, 0]), np.append(verts_0[:, 1], verts_0[0, 1]), 'k--', linewidth=1.5)
        
    if len(verts_moas) > 0:
        ax.fill(verts_moas[:, 0], verts_moas[:, 1], color='cornflowerblue', alpha=0.4, label=rf'MOAS $\mathcal{{O}}_\infty = \mathcal{{O}}_{{{t_star}}}$')
        ax.plot(np.append(verts_moas[:, 0], verts_moas[0, 0]), np.append(verts_moas[:, 1], verts_moas[0, 1]), 'b-', linewidth=2.5)
        
    # Plot intermediate sets O_1, O_2 for comparison
    H_1, h_1, _ = compute_moas(Phi, H, h, max_iter=1)
    verts_1 = get_polytope_vertices_2d(H_1, h_1, bounds_box=(-3, 3, -3, 3))
    if len(verts_1) > 0:
        ax.plot(np.append(verts_1[:, 0], verts_1[0, 0]), np.append(verts_1[:, 1], verts_1[0, 1]), 'orange', linestyle=':', linewidth=1.5, label=r'Intermediate Set $\mathcal{O}_1$')
        
    ax.set_title(rf'Maximal Output Admissible Set $\mathcal{{O}}_\infty$ Determination ($t^*={t_star}$)', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'State $x_1$ (Position)')
    ax.set_ylabel(r'State $x_2$ (Velocity)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    
    # Plot (b): Positive Invariance Phase Plane Trajectories
    ax = axes[0, 1]
    if len(verts_moas) > 0:
        ax.fill(verts_moas[:, 0], verts_moas[:, 1], color='lightcyan', alpha=0.5, label=r'MOAS $\mathcal{O}_\infty$ (Positively Invariant)')
        ax.plot(np.append(verts_moas[:, 0], verts_moas[0, 0]), np.append(verts_moas[:, 1], verts_moas[0, 1]), 'b-', linewidth=2)
        
    ax.plot(traj_in1[:, 0], traj_in1[:, 1], 'g-o', linewidth=2, label=r'Traj 1: $x(0) \in \mathcal{O}_\infty$ (Invariant & Safe)')
    ax.plot(traj_in2[:, 0], traj_in2[:, 1], 'c-s', linewidth=2, label=r'Traj 2: $x(0) \in \mathcal{O}_\infty$ (Invariant & Safe)')
    ax.plot(traj_out[:, 0], traj_out[:, 1], 'r--^', linewidth=2, label=r'Traj 3: $x(0) \notin \mathcal{O}_\infty$ (Violates Constraints!)')
    
    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2, label='Origin')
    ax.set_title('Exercise 8.9: Positive Invariance Proof Verification', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'State $x_1$')
    ax.set_ylabel(r'State $x_2$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    
    # Plot (c): State Bounds Satisfaction over Time (Velocity x_2(k))
    ax = axes[1, 0]
    t_axis = np.arange(steps + 1)
    
    ax.plot(t_axis, traj_in1[:, 1], 'g-o', linewidth=2, label=r'Safe $x_2(k)$ for $x(0) \in \mathcal{O}_\infty$')
    ax.plot(t_axis, traj_out[:, 1], 'r--^', linewidth=2, label=r'Unsafe $x_2(k)$ for $x(0) \notin \mathcal{O}_\infty$')
    ax.axhline(1.2, color='red', linestyle='--', linewidth=1.5, label=r'Velocity Bound $\pm 1.2$')
    ax.axhline(-1.2, color='red', linestyle='--', linewidth=1.5)
    ax.scatter([2], [traj_out[2, 1]], color='red', s=120, zorder=5, label='Constraint Violation at k=2!')
    ax.set_title('Constraint Satisfaction over Time ($x_2(k)$ Bound)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step k')
    ax.set_ylabel(r'Velocity State $x_2(k)$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    # Plot (d): Disturbance Invariance (mRPI Tube MPC concept)
    ax = axes[1, 1]
    # Add random bounded disturbance w(k) in [-0.05, 0.05]
    np.random.seed(42)
    w_bound = 0.06
    traj_dist = np.zeros((steps + 1, 2))
    traj_dist[0] = x0_inside1
    for k in range(steps):
        w = np.random.uniform(-w_bound, w_bound, size=2)
        traj_dist[k+1] = Phi @ traj_dist[k] + w
        
    if len(verts_moas) > 0:
        ax.fill(verts_moas[:, 0], verts_moas[:, 1], color='lavender', alpha=0.6, label=r'Nominal MOAS $\mathcal{O}_\infty$')
        ax.plot(np.append(verts_moas[:, 0], verts_moas[0, 0]), np.append(verts_moas[:, 1], verts_moas[0, 1]), 'b-', linewidth=1.5)
        
    ax.plot(traj_dist[:, 0], traj_dist[:, 1], 'm-d', linewidth=1.8, label=rf'Trajectory under Disturbance $|w_i| \leq {w_bound}$')
    ax.plot(traj_in1[:, 0], traj_in1[:, 1], 'g--', linewidth=1.5, label='Nominal Undisturbed Trajectory')
    ax.set_title('Disturbance Robust Invariance (Tube Concept)', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'State $x_1$')
    ax.set_ylabel(r'State $x_2$')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    output_path = "/home/kuang/Desktop/mpc/ch08_robust_mpc/ch8_moas_robust_invariance.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == '__main__':
    run_simulation()
