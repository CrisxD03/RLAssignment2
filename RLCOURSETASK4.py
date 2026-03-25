import numpy as np
from math import factorial, exp
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Task 4: Value Iteration 

# Same parameters as Task 3
gamma    = 0.9
lam      = 0.5
xi1, xi2 = 5, 7
n_x1, n_x2, n_E = xi1 + 1, xi2 + 1, 7  # 6 x 8 x 7 = 336 states
INF = 1e18  # big number assigned in order to penalise "do-nothing" action
eps = 1e-8  # L-inf convergence threshold

def idx(x1, x2, E):
    return x1 * n_x2 * n_E + x2 * n_E + E  #flatten as done in Task 3

def poisson_deg(x, xi):  #same as done in Task 3
    probs = np.zeros(xi + 1)
    for xp in range(x, xi):
        y = xp - x
        probs[xp] = exp(-lam) * lam**y / factorial(y)
    probs[xi] = 1.0 - probs.sum()
    return probs

# precompute degradation tables for both machines, same as Task 3
deg1 = np.array([poisson_deg(x, xi1) for x in range(n_x1)])  # (6, 6)
deg2 = np.array([poisson_deg(x, xi2) for x in range(n_x2)])  # (8, 8)

def Vmat(V, E):
    # pulls out V(x1, x2, E) for all (x1,x2) as a 2D matrix, makes matrix ops cleaner
    M = np.empty((n_x1, n_x2))
    for x1 in range(n_x1):
        for x2 in range(n_x2):
            M[x1, x2] = V[idx(x1, x2, E)]
    return M

#  INIT
V      = np.zeros(n_x1 * n_x2 * n_E)  # V_0 = 0 everywhere, doesn't matter for convergence
pi     = np.zeros((n_x1, n_x2), dtype=int)  # optimal action at E=0, gets updated each iteration
deltas = []  # track delta_k so we can plot convergence later
n_iter = 0

# VI main loop — synchronous backup, we use old V throughout and swap at the end
while True:
    V_new  = np.zeros_like(V)
    n_iter += 1

    VS = [Vmat(V, E) for E in range(n_E)]  # VS[E][x1, x2] = V(x1, x2, E)

    # precompute "travel home" expected values ,theyare reused at both PM-done and CM-done phases
    # home1[x2] = expected V*(x1',x2',0) when M1 just repaired (fresh at 0), M2 at x2, both degrade home
    # home2[x1] = symmetric for M2 just repaired
    home1 = deg1[0] @ VS[0] @ deg2.T  # shape (n_x2,)
    home2 = deg1    @ VS[0] @ deg2[0] # shape (n_x1,)

    #  E = 0: decision point — only place where the action is chosen
    # Q(a)[x1,x2] = immediate cost(a) + gamma * expected V at next E-phase
    Q0 = gamma * (deg1 @ VS[0] @ deg2.T)  # do nothing: both degrade, stay at depot E=0
    Q1 = gamma * (deg1 @ VS[1] @ deg2.T)  # go to M1: both degrade during travel, arrive at E=1
    Q2 = gamma * (deg1 @ VS[2] @ deg2.T)  # go to M2: both degrade during travel, arrive at E=2

    # unavailability cost during travel — 1 per failed machine while engineer is en route
    unavail = np.zeros((n_x1, n_x2))
    unavail[xi1, :] += 1  # M1 at failure
    unavail[:, xi2] += 1  # M2 at failure
    Q1 += unavail
    Q2 += unavail

    # do-nothing is forbidden when any machine is at failure (assignment rule)
    Q0[xi1, :] = INF
    Q0[:, xi2] = INF

    Qstack = np.stack([Q0, Q1, Q2], axis=-1)  # (n_x1, n_x2, 3) — all Q values side by side
    pi     = np.argmin(Qstack, axis=-1)        # (n_x1, n_x2) — pick cheapest action

    for x1 in range(n_x1):
        for x2 in range(n_x2):
            V_new[idx(x1, x2, 0)] = Qstack[x1, x2, pi[x1, x2]]

    #  E = 1: engineer arrived at M1, performs repair
    for x2 in range(n_x2):
        ua2 = 1 if x2 == xi2 else 0
        # PM (x1 < 5): costs 2 (travel+unavail_M1), M1 resets to 0, go to phase (0, x2', 3)
        pm1 = gamma * float(deg2[x2] @ VS[3][0, :])
        for x1 in range(xi1):
            V_new[idx(x1, x2, 1)] = (2 + ua2) + pm1
        # CM (x1 = 5): costs 6 (CM5 + unavail_M1), M1 stays at 5 through phase 1
        V_new[idx(xi1, x2, 1)] = (6 + ua2) + gamma * float(deg2[x2] @ VS[3][xi1, :])

    #  E = 3: M1 repair phase 1 done
    for x2 in range(n_x2):
        ua2 = 1 if x2 == xi2 else 0
        # PM done (x1=0): M1 is repaired, no M1 unavailability, travel home
        V_new[idx(0, x2, 3)]   = ua2 + gamma * float(home1[x2])
        # CM phase 2 (x1=5): M1 still being fixed, one more repair step to go
        V_new[idx(xi1, x2, 3)] = (1 + ua2) + gamma * float(deg2[x2] @ VS[4][xi1, :])
        # x1 in {1,2,3,4} are unreachable under any policy, V_new stays 0

    #  E = 4: M1 CM done, travel back to depot
    for x2 in range(n_x2):
        ua2 = 1 if x2 == xi2 else 0
        # M1 is now repaired (starts degrading from 0 on the way home), same structure as PM-done
        V_new[idx(xi1, x2, 4)] = ua2 + gamma * float(home1[x2])
        # x1 != 5 unreachable, stays 0

    #  E = 2: engineer arrived at M2, performs repair (symmetric to E=1)
    for x1 in range(n_x1):
        ua1 = 1 if x1 == xi1 else 0
        # PM (x2 < 7): M2 resets to 0, M1 degrades, next phase is (x1', 0, 5)
        pm2 = gamma * float(deg1[x1] @ VS[5][:, 0])
        for x2 in range(xi2):
            V_new[idx(x1, x2, 2)] = (2 + ua1) + pm2
        # CM (x2 = 7): M2 stays at 7 through phase 1
        V_new[idx(x1, xi2, 2)] = (6 + ua1) + gamma * float(deg1[x1] @ VS[5][:, xi2])

    #  E = 5: M2 repair phase 1 done (symmetric to E=3)
    for x1 in range(n_x1):
        ua1 = 1 if x1 == xi1 else 0
        # PM done (x2=0): M2 repaired, travel home
        V_new[idx(x1, 0, 5)]   = ua1 + gamma * float(home2[x1])
        # CM phase 2 (x2=7): M2 still being fixed
        V_new[idx(x1, xi2, 5)] = (1 + ua1) + gamma * float(deg1[x1] @ VS[6][:, xi2])
        # x2 in {1,...,6} unreachable, stays 0

    #  E = 6: M2 CM done, travel back to depot (symmetric to E=4)
    for x1 in range(n_x1):
        ua1 = 1 if x1 == xi1 else 0
        # M2 repaired (starts degrading from 0 on the way home)
        V_new[idx(x1, xi2, 6)] = ua1 + gamma * float(home2[x1])
        # x2 != 7 unreachable, stays 0

    #  convergence check
    delta = np.max(np.abs(V_new - V))
    deltas.append(delta)
    V = V_new
    if delta < eps:
        break

print(f"Value Iteration converged in {n_iter} iterations  (ε = {eps:.0e})")
print(f"\nV*(0, 0, 0) = {V[idx(0, 0, 0)]:.6f}")

print("\nOptimal action at E=0  (0=do nothing | 1=maintain M1 | 2=maintain M2):")
print("x2\\x1 |", "  ".join(str(x1) for x1 in range(n_x1)))
print("-" * 30)
for x2 in range(n_x2 - 1, -1, -1):
    row = "  ".join(str(pi[x1, x2]) for x1 in range(n_x1))
    print(f"  {x2}   | {row}")

# plots saved here
plot_dir = "/Users/yourUsername/Desktop/RLAssignment1Task4Plots_3"
os.makedirs(plot_dir, exist_ok=True)

# Figure 1: convergence — delta_k shrinks geometrically at rate ~gamma per iteration
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(deltas, color="steelblue", linewidth=1.5, label=r"$\delta_k = \|V_{k+1}-V_k\|_\infty$")
ax.axhline(eps, color="crimson", linestyle="--", linewidth=1.2, label=f"ε = {eps:.0e}")
ax.set_xlabel("Iteration $k$", fontsize=12)
ax.set_ylabel(r"$\|\,V_{k+1} - V_k\,\|_\infty$  (log scale)", fontsize=11)
ax.set_title(f"Value Iteration — Convergence  ({n_iter} iterations)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "convergence.png"), dpi=150)
plt.show()
print("Figure 1 saved: convergence.png")

# Figure 2: optimal policy heatmap — 3 colours, one per action
cmap_pol = mcolors.ListedColormap(["#4C72B0", "#DD8452", "#55A868"])
bounds   = [-0.5, 0.5, 1.5, 2.5]
norm_pol = mcolors.BoundaryNorm(bounds, cmap_pol.N)

fig, ax = plt.subplots(figsize=(6, 7))
ax.imshow(pi.T, origin="lower", cmap=cmap_pol, norm=norm_pol,
          extent=[-0.5, n_x1 - 0.5, -0.5, n_x2 - 0.5], aspect="auto")

ax.set_xlabel("$x_1$  (M1 degradation)", fontsize=12)
ax.set_ylabel("$x_2$  (M2 degradation)", fontsize=12)
ax.set_title(r"Optimal Policy $\pi^*(x_1, x_2)$ at $E=0$", fontsize=13)
ax.set_xticks(range(n_x1)); ax.set_xticklabels(range(n_x1))
ax.set_yticks(range(n_x2)); ax.set_yticklabels(range(n_x2))

# dashed line = M1 failure threshold, dotted = M2
ax.axvline(xi1 - 0.5, color="white", linestyle="--", linewidth=1.8, label=f"$\\xi_1 = {xi1}$")
ax.axhline(xi2 - 0.5, color="white", linestyle=":",  linewidth=1.8, label=f"$\\xi_2 = {xi2}$")

patches = [mpatches.Patch(color="#4C72B0", label="0: Do Nothing"),
           mpatches.Patch(color="#DD8452", label="1: Maintain M1"),
           mpatches.Patch(color="#55A868", label="2: Maintain M2")]
ax.legend(handles=patches, loc="upper left", fontsize=10,
          framealpha=0.85, edgecolor="grey")

for x1 in range(n_x1):
    for x2 in range(n_x2):
        ax.text(x1, x2, str(pi[x1, x2]),
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "optimal_policy.png"), dpi=150)
plt.show()
print("Figure 2 saved: optimal_policy.png")

# Figure 3: value function heatmap — shows how cost grows as degradation increases
V0_grid = np.array([[V[idx(x1, x2, 0)] for x2 in range(n_x2)]
                    for x1 in range(n_x1)])  # shape (n_x1, n_x2)

fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(V0_grid.T, origin="lower", cmap="YlOrRd",
               extent=[-0.5, n_x1 - 0.5, -0.5, n_x2 - 0.5], aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$V^*(x_1, x_2, 0)$", fontsize=11)

ax.set_xlabel("$x_1$  (M1 degradation)", fontsize=12)
ax.set_ylabel("$x_2$  (M2 degradation)", fontsize=12)
ax.set_title(r"Optimal Value Function $V^*(x_1, x_2, 0)$", fontsize=13)
ax.set_xticks(range(n_x1)); ax.set_xticklabels(range(n_x1))
ax.set_yticks(range(n_x2)); ax.set_yticklabels(range(n_x2))

ax.axvline(xi1 - 0.5, color="white", linestyle="--", linewidth=1.8)
ax.axhline(xi2 - 0.5, color="white", linestyle=":",  linewidth=1.8)

for x1 in range(n_x1):
    for x2 in range(n_x2):
        ax.text(x1, x2, f"{V[idx(x1, x2, 0)]:.2f}",
                ha="center", va="center", fontsize=7,
                color="black" if V0_grid[x1, x2] < 0.6 * V0_grid.max() else "white")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "value_function.png"), dpi=150)
plt.show()
print("Figure 3 saved: value_function.png")

print(f"\nAll figures saved to: {plot_dir}")
