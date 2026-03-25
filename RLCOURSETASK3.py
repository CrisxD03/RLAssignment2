import numpy as np
from math import factorial, exp

#  Task 3: Policy evaluation for "maintain at failure only" 
# We chose the method with the exact solution via V^π = (I - γP^π)^{-1} R^π (formula from slides)

#  Parameters 
gamma = 0.9          #Discount 
lam = 0.5           # Poisson degradation 
xi1, xi2 = 5, 7     # failure for M1, M2
n_x1 = xi1 + 1      # Deg states for M1: 0..5
n_x2 = xi2 + 1      # Deg states for M2: 0..7
n_E = 7              # Engineer phases: 0..6
n_states = n_x1 * n_x2 * n_E  # 6*8*7 = 336

#  State indexing: (x1, x2, E) -> flat index 
def idx(x1, x2, E):
    return x1 * n_x2 * n_E + x2 * n_E + E
    #We need this, because we are using a flat array to represent the state space and Every state gets a unique number from 0 to 335.

#  Truncated Poisson(0.5) degradation probabilities 
# p_i(x, x'): prob of going from state x to x', lumped at failure threshold
# We use the formula from our paper
def poisson_deg(x, xi):
    probs = np.zeros(xi + 1)
    for xp in range(x, xi):
        y = xp - x
        probs[xp] = exp(-lam) * lam**y / factorial(y)
    probs[xi] = 1.0 - probs.sum()  # lump all overflow at failure
    return probs

# precompute: deg1[x] = prob vector from x for M1, deg2[x] for M2
deg1 = np.array([poisson_deg(x, xi1) for x in range(n_x1)])  # (6, 6)
deg2 = np.array([poisson_deg(x, xi2) for x in range(n_x2)])  # (8, 8)

#  Helper function: fill joint degradation transitions into P row ,makes life easier,handles 
def fill_joint(P, s, d1_row, d2_row, target_E):
    """Fill P[s, :] with joint independent degradation probs going to target_E."""
    targets = [idx(x1p, x2p, target_E) for x1p in range(n_x1) for x2p in range(n_x2)]
    P[s, targets] = np.outer(d1_row, d2_row).flatten()

#  Build P^π (transition matrix) and R^π (cost vector) under the policy 
P = np.zeros((n_states, n_states))
R = np.zeros(n_states)

for x1 in range(n_x1):
    for x2 in range(n_x2):
        for E in range(n_E):
            s = idx(x1, x2, E)

            if E == 0:
                #  Decision point (depot) 
                if x1 < xi1 and x2 < xi2: #Both are below failure threshold
                    # do nothing: cost=0, both degrade, stay at depot
                    R[s] = 0
                    fill_joint(P, s, deg1[x1], deg2[x2], 0)

                elif x1 == xi1 and x2 < xi2: #M1 is at failure, M2 is below failure
                    # maintain M1: travel to M1. M1 stays at 5, M2 degrades
                    R[s] = 1  # M1 unavailable
                    for x2p in range(n_x2):
                        P[s, idx(xi1, x2p, 1)] = deg2[x2, x2p]

                elif x1 < xi1 and x2 == xi2: #M1 is below failure, M2 is at failure
                    # maintain M2: travel to M2. M2 stays at 7, M1 degrades
                    R[s] = 1  # M2 unavailable
                    for x1p in range(n_x1):
                        P[s, idx(x1p, xi2, 2)] = deg1[x1, x1p]

                else:  # both failed
                    # tiebreaker: maintain M2 first. No degradation (both at failure)
                    R[s] = 2  # both unavailable
                    P[s, idx(xi1, xi2, 2)] = 1.0

            elif E == 1:
                #  At M1, performing repair 
                if x1 == xi1:  # corrective (only reachable case under this policy)
                    R[s] = 6 + (1 if x2 == xi2 else 0)  # CM(5) + unavail_M1(1) + unavail_M2
                    for x2p in range(n_x2):
                        P[s, idx(xi1, x2p, 3)] = deg2[x2, x2p]  # M2 degrades
                else:
                    R[s] = 0; P[s, s] = 1.0  # unreachable, absorbing

            elif E == 3:
                #  M1 repair: phase 1 done 
                if x1 == xi1:  # CM continues to phase 2
                    R[s] = 1 + (1 if x2 == xi2 else 0)  # M1 still unavail + unavail_M2
                    for x2p in range(n_x2):
                        P[s, idx(xi1, x2p, 4)] = deg2[x2, x2p]
                else:
                    R[s] = 0; P[s, s] = 1.0  # unreachable

            elif E == 4:
                #  M1 CM done, travel back. M1 repaired, degrades from 0 
                if x1 == xi1:
                    R[s] = 1 if x2 == xi2 else 0  # only M2 unavail if failed
                    fill_joint(P, s, deg1[0], deg2[x2], 0)  # M1 fresh, both degrade
                else:
                    R[s] = 0; P[s, s] = 1.0

            elif E == 2:
                #  At M2, performing repair 
                if x2 == xi2:  # corrective (only reachable case)
                    R[s] = 6 + (1 if x1 == xi1 else 0)  # CM(5) + unavail_M2(1) + unavail_M1
                    for x1p in range(n_x1):
                        P[s, idx(x1p, xi2, 5)] = deg1[x1, x1p]  # M1 degrades
                else:
                    R[s] = 0; P[s, s] = 1.0

            elif E == 5:
                #  M2 repair: phase 1 done 
                if x2 == xi2:  # CM continues to phase 2
                    R[s] = 1 + (1 if x1 == xi1 else 0)  # M2 still unavail + unavail_M1
                    for x1p in range(n_x1):
                        P[s, idx(x1p, xi2, 6)] = deg1[x1, x1p] # M1 degrades
                else:
                    R[s] = 0; P[s, s] = 1.0

            elif E == 6:
                #  M2 CM done, travel back. M2 repaired, degrades from 0 
                if x2 == xi2:
                    R[s] = 1 if x1 == xi1 else 0  # only M1 unavail if failed
                    fill_joint(P, s, deg1[x1], deg2[0], 0)  # M2 fresh, both degrade
                else:
                    R[s] = 0; P[s, s] = 1.0

#fCompleted todo, sanity check below.
#Sanity check: each row of P must sum to 1 
row_sums = P.sum(axis=1)
assert np.allclose(row_sums, 1.0), f"Row sums deviate! Max error: {np.max(np.abs(row_sums - 1))}"
print("P rows all sum to 1 ")

#  Solve V^π = (I - γP)^{-1} R  (exact solution) 
I_mat = np.identity(n_states)
V = np.dot(np.linalg.inv(I_mat - np.multiply(gamma, P)), R)  # using same formula from hint

#  Output 
s0 = idx(0, 0, 0)
print(f"\nTotal expected discounted cost from healthy state:")
print(f"  V^π(0, 0, 0) = {V[s0]:.6f}")

