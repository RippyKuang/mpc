"""
ch4_hankel_svd_realization.py
=============================
Maciejowski (2002) Chapter 4: Subspace / Hankel-SVD Realization Algorithm.
Implements:
  1. build_hankel: Constructs Block-Hankel matrix from Markov parameters (pulse responses).
  2. ho_kalman_svd: Kung's SVD algorithm to extract minimal state-space (A, B, C, D).
  3. Example 4.1 & Example 4.2: 1-input 1-output deadbeat pulse response realization (3 states).
  4. Exercise 4.4: 2-input 1-output deadbeat step response realization (4 states).
  5. Singular value decay spectrum plot (replicating Figure 4.2).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_hankel(H_seq, r, c):
    """
    Constructs block-Hankel matrix H_{r, c} from pulse response matrices H(1), H(2), ...
    H_seq is a 3D array of shape (m, ell, N_total).
    H_{r, c} has shape (r * m, c * ell).
    """
    m, ell, N_total = H_seq.shape
    H_block = np.zeros((r * m, c * ell))
    
    for i in range(r):
        for j in range(c):
            # Markov parameter index: t = i + j + 1 (1-indexed pulse response)
            t = i + j + 1
            if t < N_total:
                H_block[i*m : (i+1)*m, j*ell : (j+1)*ell] = H_seq[:, :, t]
                
    return H_block

def ho_kalman_svd(H_seq, r, c, order=None, tol=1e-5):
    """
    Kung's SVD minimal realization algorithm (Ho-Kalman formulation).
    Given pulse response sequence H(0), H(1), ..., H(N):
      D = H(0)
      Builds Hankel H_{r, c} with Markov parameters H(1), ..., H(r+c-1).
      Computes SVD: H_{r, c} = U * Sigma * V^T.
      Determines minimal order n.
      Observability matrix O_r = U_n * Sigma_n^(1/2)
      Controllability matrix C_c = Sigma_n^(1/2) * V_n^T
      A = O_r(1:end-m, :)^dagger * O_r(m+1:end, :)
      B = C_c[:, 0:ell]
      C = O_r[0:m, :]
    """
    m, ell, _ = H_seq.shape
    D = H_seq[:, :, 0]
    
    H_block = build_hankel(H_seq, r, c)
    U, s, Vt = np.linalg.svd(H_block, full_matrices=False)
    
    if order is None:
        order = int(np.sum(s > tol * s[0]))
        
    print(f"-> Selected system order n = {order} (Singular values: {s[:min(len(s), 8)]})")
    
    U_n = U[:, :order]
    s_n = s[:order]
    Vt_n = Vt[:order, :]
    
    # Factorization into Observability and Controllability
    sqrt_s = np.diag(np.sqrt(s_n))
    O_r = U_n @ sqrt_s
    C_c = sqrt_s @ Vt_n
    
    # Extract C and B
    C = O_r[:m, :]
    B = C_c[:, :ell]
    
    # Shifted Hankel or shifted Observability matrix to extract A
    # O_r_up = O_r[:-m, :], O_r_down = O_r[m:, :]
    O_up = O_r[:-m, :]
    O_down = O_r[m:, :]
    A = np.linalg.pinv(O_up) @ O_down
    
    return A, B, C, D, s

def test_example_4_1_and_4_2():
    print("===================================================================")
    print("Replicating Example 4.1 & Example 4.2: 1-Input 1-Output Realization")
    print("===================================================================")
    # Step response: S(0) = 0, S(1) = 0, S(2) = -1, S(3) = 2, S(k) = 2 for k >= 3
    # Pulse response: H(0) = 0, H(1) = 0, H(2) = -1, H(3) = 3, H(k) = 0 for k > 3
    N = 10
    S = np.zeros((1, 1, N + 1))
    S[0, 0, 0] = 0.0
    S[0, 0, 1] = 0.0
    S[0, 0, 2] = -1.0
    S[0, 0, 3:] = 2.0
    
    # Pulse response H(t) = S(t) - S(t-1)
    H = np.zeros((1, 1, N + 1))
    H[0, 0, 0] = S[0, 0, 0]
    for t in range(1, N + 1):
        H[0, 0, t] = S[0, 0, t] - S[0, 0, t-1]
        
    print(f"Pulse response H(0..5): {[float(H[0, 0, t]) for t in range(6)]}")
    
    # Run Ho-Kalman SVD with r=5, c=5, order=3
    A, B, C, D, sv = ho_kalman_svd(H, r=5, c=5, order=3)
    
    print("\nIdentified Minimal State Space Model (order=3):")
    print("A matrix:\n", np.round(A, 4))
    print("B matrix:\n", np.round(B, 4))
    print("C matrix:\n", np.round(C, 4))
    print("D matrix:\n", np.round(D, 4))
    
    # Validate step response reproduction
    x = np.zeros((3, 1))
    S_reproduced = []
    for t in range(6):
        y = (C @ x + D * 1.0)[0, 0]
        S_reproduced.append(y)
        x = A @ x + B * 1.0
        
    print("\nOriginal S(0..5):   ", [float(S[0, 0, t]) for t in range(6)])
    print("Reproduced S(0..5): ", [round(val, 4) for val in S_reproduced])
    assert np.allclose([float(S[0, 0, t]) for t in range(6)], S_reproduced, atol=1e-4)
    print("-> SVD realization successfully reproduced Example 4.1/4.2 step response!")

def test_exercise_4_4():
    print("\n===================================================================")
    print("Solving Exercise 4.4: 2-Input 1-Output State-Space Realization")
    print("===================================================================")
    # S(0) = [0, 0], S(1) = [0, 2], S(2) = [1, 3], S(k) = [1, 3] for k >= 2
    N = 8
    S = np.zeros((1, 2, N + 1))
    S[0, :, 1] = [0.0, 2.0]
    for t in range(2, N + 1):
        S[0, :, t] = [1.0, 3.0]
    
    H = np.zeros((1, 2, N + 1))
    H[0, :, 0] = S[0, :, 0]
    for t in range(1, N + 1):
        H[0, :, t] = S[0, :, t] - S[0, :, t-1]
        
    print(f"H(0) = {H[0, :, 0]}, H(1) = {H[0, :, 1]}, H(2) = {H[0, :, 2]}, H(3) = {H[0, :, 3]}")
    
    # Run SVD realization with order=4 as required by Exercise 4.4
    A, B, C, D, sv = ho_kalman_svd(H, r=4, c=4, order=4)
    
    print("\nIdentified Exercise 4.4 State-Space (4 states):")
    print("A (4x4):\n", np.round(A, 4))
    print("B (4x2):\n", np.round(B, 4))
    print("C (1x4):\n", np.round(C, 4))
    
    # Simulate step response for u = [1, 0]^T and u = [0, 1]^T
    for ch, u_val in enumerate([[1.0, 0.0], [0.0, 1.0]]):
        u_step = np.array(u_val).reshape(2, 1)
        x = np.zeros((4, 1))
        reproduced = []
        for t in range(4):
            y = (C @ x + D @ u_step)[0, 0]
            reproduced.append(y)
            x = A @ x + B @ u_step
        target = [float(S[0, ch, t]) for t in range(4)]
        print(f"Channel {ch+1} Target S(0..3):     {target}")
        print(f"Channel {ch+1} Reproduced S(0..3): {[round(v, 4) for v in reproduced]}")
        assert np.allclose(target, reproduced, atol=1e-4)
    print("-> Exercise 4.4 4-state realization verified exactly!")
    
    # Plot singular value spectrum (Figure 4.2 replication)
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(sv) + 1), sv, 'ro-', linewidth=2, markersize=8)
    plt.axvline(x=4.5, color='b', linestyle='--', label='Truncation at order n=4')
    plt.title('Singular Value Spectrum of Block-Hankel Matrix (Exercise 4.4 / Fig 4.2)')
    plt.xlabel('Singular Value Index i')
    plt.ylabel('Singular Value sigma_i (log scale)')
    plt.grid(True, which='both')
    plt.legend()
    plot_path = '/home/kuang/Desktop/mpc/ch04_transfer_functions/ch4_hankel_svd_spectrum.png'
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Saved Hankel SVD spectrum plot to: {plot_path}")

if __name__ == '__main__':
    test_example_4_1_and_4_2()
    test_exercise_4_4()
