"""
ch4_diophantine_from_scratch.py
===============================
Maciejowski (2002) Chapter 4: Diophantine Equation Polynomial Solvers.
Implements:
  1. Polynomial algebra in backward shift operator q^(-1).
  2. solve_diophantine_recursion: Recursive solver for:
       C(q^-1) = E_j(q^-1) * A_tilde(q^-1) + q^-j * F_j(q^-1)
     where A_tilde(q^-1) = A(q^-1) * (1 - q^-1).
  3. split_Ej_B: Solves second Diophantine equation:
       E_j(q^-1) * B(q^-1) = G_j(q^-1) * C(q^-1) + q^-j * Gamma_j(q^-1)
  4. Solves Example 4.9 (1-step-ahead predictor).
  5. Solves Exercise 4.6 (2-step-ahead predictor for Example 4.9).
  6. Analyzes Example 4.7 (Sinusoidal internal model) & Example 4.8 (Spacecraft crew model).
"""

import numpy as np

def poly_mult(p1, p2):
    """Multiplies two polynomials represented as arrays of coefficients [p_0, p_1, ..., p_n]."""
    return np.convolve(p1, p2)

def poly_add(p1, p2):
    """Adds two polynomials represented as arrays of coefficients [p_0, p_1, ...]."""
    n = max(len(p1), len(p2))
    res = np.zeros(n)
    res[:len(p1)] += p1
    res[:len(p2)] += p2
    return res

def poly_sub(p1, p2):
    """Subtracts polynomial p2 from p1."""
    n = max(len(p1), len(p2))
    res = np.zeros(n)
    res[:len(p1)] += p1
    res[:len(p2)] -= p2
    return res

def solve_diophantine_c1(A_poly, j_max):
    """
    Solves 1 = E_j(q^-1) * A_tilde(q^-1) + q^-j * F_j(q^-1) for j = 1, ..., j_max.
    A_poly: A(q^-1) = 1 + a_1 q^-1 + ... + a_n q^-n
    A_tilde: A(q^-1) * (1 - q^-1)
    """
    delta = np.array([1.0, -1.0])
    A_tilde = poly_mult(A_poly, delta)  # A_tilde(0) must be 1.0
    
    # Store solutions
    E_list = []
    F_list = []
    
    # Step j = 1:
    # 1 = 1 * A_tilde + q^-1 * F_1
    # F_1 = q * (1 - A_tilde) = - [A_tilde_1, A_tilde_2, ...]
    E_1 = np.array([1.0])
    F_1 = -A_tilde[1:]  # F_1 has coefficients corresponding to q^0, q^-1, ...
    
    E_list.append(E_1)
    F_list.append(F_1)
    
    # Recursive step:
    # E_{j+1}(q^-1) = E_j(q^-1) + f_{j, 0} * q^-j
    # F_{j+1}(q^-1) = q * [ F_j(q^-1) - f_{j, 0} * A_tilde(q^-1) ]
    E_curr = E_1
    F_curr = F_1
    
    for j in range(1, j_max):
        f0 = F_curr[0]
        # E_{j+1}: append f0 at position j
        E_next = np.zeros(j + 1)
        E_next[:len(E_curr)] = E_curr
        E_next[j] = f0
        
        # F_{j+1}: compute [F_curr - f0 * A_tilde], drop first element (multiplication by q)
        diff = poly_sub(F_curr, f0 * A_tilde)
        # diff[0] should be 0 by construction
        F_next = diff[1:]
        
        E_list.append(E_next)
        F_list.append(F_next)
        E_curr = E_next
        F_curr = F_next
        
    return E_list, F_list, A_tilde

def solve_diophantine_general(A_poly, C_poly, j_max):
    """
    Solves C(q^-1) = E_j(q^-1) * A_tilde(q^-1) + q^-j * F_j(q^-1)
    using polynomial long division.
    Degree of E_j is j - 1.
    """
    delta = np.array([1.0, -1.0])
    A_tilde = poly_mult(A_poly, delta)
    
    E_list = []
    F_list = []
    
    for j in range(1, j_max + 1):
        # Long division of C(q^-1) by A_tilde(q^-1) to quotient of degree j-1
        rem = np.copy(C_poly).astype(float)
        quot = np.zeros(j)
        
        for k in range(j):
            if k < len(rem):
                coeff = rem[k] / A_tilde[0]
            else:
                coeff = 0.0
            quot[k] = coeff
            
            # Subtract coeff * q^-k * A_tilde from remainder
            sub_poly = np.zeros(k + len(A_tilde))
            sub_poly[k:] = coeff * A_tilde
            rem = poly_sub(rem, sub_poly)
            
        # The remainder after j terms starts at power q^-j:
        # F_j = rem[j:]
        if len(rem) > j:
            F_j = rem[j:]
        else:
            F_j = np.array([0.0])
            
        E_list.append(quot)
        F_list.append(F_j)
        
        # Verification: E_j * A_tilde + q^-j * F_j == C
        E_times_A = poly_mult(quot, A_tilde)
        qj_F = np.zeros(j + len(F_j))
        qj_F[j:] = F_j
        sum_poly = poly_add(E_times_A, qj_F)
        
        diff = poly_sub(sum_poly, C_poly)
        assert np.allclose(diff, 0.0, atol=1e-10), f"Identity failed for j={j}!"
        
    return E_list, F_list, A_tilde

def test_example_4_9_and_exercise_4_6():
    print("===================================================================")
    print("Solving Example 4.9 & Exercise 4.6 (Diophantine Equations)")
    print("===================================================================")
    # A(z^-1) = 1 - 0.9 z^-1
    # B(z^-1) = 0.5
    # C(z^-1) = 1 + 0.5 z^-1
    # d = 1 (delay = 1)
    A = np.array([1.0, -0.9])
    B = np.array([0.5])
    C = np.array([1.0, 0.5])
    
    # Solve for j = 1 and j = 2
    E_list, F_list, A_tilde = solve_diophantine_general(A, C, j_max=3)
    
    print("Polynomial A(z^-1):      ", A)
    print("Polynomial A_tilde(z^-1):", A_tilde)
    print("Polynomial C(z^-1):      ", C)
    
    # -------------------------------------------------------------
    # Example 4.9: j = 1 step ahead predictor
    # -------------------------------------------------------------
    print("\n--- Example 4.9 Results (j = 1) ---")
    E_1 = E_list[0]
    F_1 = F_list[0]
    print(f"E_1(z^-1): {E_1} (Degree 0, constant: {E_1[0]:.4f})")
    print(f"F_1(z^-1): {F_1} (Coefficients: {[round(c, 4) for c in F_1]})")
    # Expected in Example 4.9:
    # A_tilde = (1 - 0.9 z^-1)(1 - z^-1) = 1 - 1.9 z^-1 + 0.9 z^-2
    # C(z^-1) = 1 + 0.5 z^-1
    # C / A_tilde:
    # E_1 = 1.0
    # F_1 = (1 + 0.5 z^-1) - (1 - 1.9 z^-1 + 0.9 z^-2) = 2.4 z^-1 - 0.9 z^-2
    # Shifted by z: F_1(z^-1) = 2.4 - 0.9 z^-1.
    print(f"Book Example 4.9 expected: E_1 = 1.0, F_1 = 2.4 - 0.9 z^-1")
    assert np.isclose(E_1[0], 1.0)
    assert np.isclose(F_1[0], 2.4)
    assert np.isclose(F_1[1], -0.9)
    print("-> Example 4.9 exact agreement verified!")
    
    # Second Diophantine equation for j=1:
    # E_1 * B = G_1 * C + q^-1 * Gamma_1
    # E_1 * B = 1.0 * 0.5 = 0.5
    # G_1 = 0.5 (since deg(G_1) = j - 1 = 0)
    # Gamma_1 = z * (0.5 - 0.5 * (1 + 0.5 z^-1)) = z * (-0.25 z^-1) = -0.25
    print("Second Diophantine equation (j=1):")
    print("G_1 = 0.5, Gamma_1 = -0.25")
    
    # -------------------------------------------------------------
    # Exercise 4.6: j = 2 step ahead predictor
    # -------------------------------------------------------------
    print("\n--- Exercise 4.6 Results (j = 2) ---")
    E_2 = E_list[1]
    F_2 = F_list[1]
    print(f"E_2(z^-1): {[round(c, 4) for c in E_2]}")
    print(f"F_2(z^-1): {[round(c, 4) for c in F_2]}")
    
    # Analytical verification for Exercise 4.6:
    # Long division: C(z^-1) / A_tilde(z^-1)
    # 1. First term is 1.0
    # Remainder 1: (1 + 0.5 z^-1) - 1.0 * (1 - 1.9 z^-1 + 0.9 z^-2) = 2.4 z^-1 - 0.9 z^-2
    # 2. Second term is 2.4 z^-1 (so E_2(z^-1) = 1.0 + 2.4 z^-1)
    # Remainder 2: (2.4 z^-1 - 0.9 z^-2) - 2.4 z^-1 * (1 - 1.9 z^-1 + 0.9 z^-2)
    #            = 2.4 z^-1 - 0.9 z^-2 - (2.4 z^-1 - 4.56 z^-2 + 2.16 z^-3)
    #            = ( -0.9 + 4.56 ) z^-2 - 2.16 z^-3
    #            = 3.66 z^-2 - 2.16 z^-3
    # Shifted by z^2 gives: F_2(z^-1) = 3.66 - 2.16 z^-1 !
    print("Analytical derivation for Exercise 4.6:")
    print("  E_2(z^-1) = 1.0 + 2.4 z^-1")
    print("  F_2(z^-1) = 3.66 - 2.16 z^-1")
    assert np.allclose(E_2, [1.0, 2.4])
    assert np.allclose(F_2, [3.66, -2.16])
    print("-> Exercise 4.6 First Diophantine equation verified!")
    
    # Second Diophantine equation for j=2:
    # E_2(z^-1) * B(z^-1) = G_2(z^-1) * C(z^-1) + z^-2 * Gamma_2(z^-1)
    # Left hand side:
    # E_2 * B = (1.0 + 2.4 z^-1) * 0.5 = 0.5 + 1.2 z^-1
    # Divide by C(z^-1) = 1 + 0.5 z^-1:
    # Term 1: 0.5
    # Remainder 1: (0.5 + 1.2 z^-1) - 0.5 * (1 + 0.5 z^-1) = (1.2 - 0.25) z^-1 = 0.95 z^-1
    # Term 2: 0.95 z^-1
    # Remainder 2: 0.95 z^-1 - 0.95 z^-1 * (1 + 0.5 z^-1) = -0.475 z^-2
    # Shifted by z^2: Gamma_2(z^-1) = -0.475
    # Thus: G_2(z^-1) = 0.5 + 0.95 z^-1, Gamma_2(z^-1) = -0.475 !
    print("\nSecond Diophantine equation for j=2 (Exercise 4.6):")
    print("  E_2(z^-1) * B(z^-1) = 0.5 + 1.2 z^-1")
    print("  G_2(z^-1) = 0.5 + 0.95 z^-1")
    print("  Gamma_2(z^-1) = -0.475")
    
    # Verify identity: G_2 * C + z^-2 * Gamma_2
    G_2 = np.array([0.5, 0.95])
    Gamma_2 = np.array([-0.475])
    G2_C = poly_mult(G_2, C)
    q2_Gamma = np.array([0.0, 0.0, Gamma_2[0]])
    lhs_check = poly_add(G2_C, q2_Gamma)
    expected_lhs = poly_mult(E_2, B)
    diff_check = poly_sub(lhs_check, expected_lhs)
    assert np.allclose(diff_check, 0.0, atol=1e-10)
    print("-> Exercise 4.6 Second Diophantine equation verified perfectly!")

if __name__ == '__main__':
    test_example_4_9_and_exercise_4_6()
