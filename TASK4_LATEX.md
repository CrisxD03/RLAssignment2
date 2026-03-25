\section*{Task 4: Optimal Maintenance Policy via Value Iteration}

\subsection*{Algorithm Choice}

We have decided to use Value Iteration (VI) for the following reasons:
\begin{itemize}
    \item VI doesn't require an initial policy unlike PI. It starts from $V_0(s) = 0$ for all $s$, which is a valid and simple initialisation.
    \item Each iteration applies a single vectorised Bellman backup over all 336 states, with no inner linear system to solve.
    \item Convergence is measured directly via the $L_\infty$ norm on the value function, making it easy to visualise.
\end{itemize}

\subsection*{Algorithm and Convergence Criterion}

Starting from $V_0(s) = 0$ for all $s \in S$, each VI iteration applies the
synchronous Bellman optimality backup:
\[
    V_{k+1}(s) = \min_{a \in \mathcal{A}(s)}
    \left\{ R^a_s + \gamma \sum_{s' \in S} P^a_{s,s'}\, V_k(s') \right\},
    \quad s \in S.
\]
The old value function $V_k$ is used for all updates within iteration $k$
(synchronous backup). The algorithm terminates when
\[
    \delta_k = \|V_{k+1} - V_k\|_\infty
             = \max_{s \in S}\, |V_{k+1}(s) - V_k(s)| < \varepsilon,
\]
with $\varepsilon = 10^{-8}$ and $\gamma = 0.9 < 1$, being governed by the
Contraction Mapping Theorem (Lecture~4, Theorems~2 and~4), we know that
convergence to the unique $V^*$ is guaranteed. After termination, the optimal
policy is extracted as:
\[
    \pi^*(s) = \operatorname{argmin}_{a \in \mathcal{A}(s)}
    \left\{ R^a_s + \gamma \sum_{s'} P^a_{s,s'}\, V^*(s') \right\}.
\]

\subsection*{Implementation Details}

At $E = 0$ (the only decision state per our assumption), three Q-values are
computed for each $(x_1, x_2)$:
\begin{align*}
Q^0(x_1,x_2,0) &= \gamma \sum_{x_1'}\sum_{x_2'}
    p_1(x_1'|x_1)\,p_2(x_2'|x_2)\,V_k(x_1',x_2',0)
    \quad [\text{valid only if } x_1 < 5,\; x_2 < 7], \\[4pt]
Q^1(x_1,x_2,0) &= \bigl[\mathbb{I}(x_1{=}5)+\mathbb{I}(x_2{=}7)\bigr]
    + \gamma \sum_{x_1'}\sum_{x_2'}
    p_1(x_1'|x_1)\,p_2(x_2'|x_2)\,V_k(x_1',x_2',1), \\[4pt]
Q^2(x_1,x_2,0) &= \bigl[\mathbb{I}(x_1{=}5)+\mathbb{I}(x_2{=}7)\bigr]
    + \gamma \sum_{x_1'}\sum_{x_2'}
    p_1(x_1'|x_1)\,p_2(x_2'|x_2)\,V_k(x_1',x_2',2).
\end{align*}
The indicator terms are the unavailability costs incurred during the one-step
travel to the chosen machine. Do-nothing is forbidden when any machine is at
failure: $Q^0(x_1,x_2,0) = \infty$ if $x_1 = 5$ or $x_2 = 7$.

For states $E \in \{1,2,3,4,5,6\}$ there is no choice; the update follows the
BC equations from Task~2.

\subsection*{Convergence}

The algorithm converged in \textbf{174 iterations}. Figure~\ref{fig:convergence}
shows $\delta_k$ on a log scale. The decay is geometric at rate $\gamma = 0.9$,
consistent with the theoretical bound
$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$
(Lecture~4, Theorems~2 and~4).

\subsection*{Results}

VI converges to $V^*$, the unique fixed point of the Bellman optimality operator,
with residual error below $\varepsilon = 10^{-8}$.

\begin{table}[h]
\centering
\caption{Optimal expected discounted cost $V^*(x_1, x_2, 0)$ for selected depot states.}
\label{tab:t4_values}
\begin{tabular}{llc}
\toprule
$x_1$ & $x_2$ & $V^*(x_1,x_2,0)$ \\
\midrule
0 & 0 & 3.959117 \\
\bottomrule
\end{tabular}
\end{table}

The optimal expected discounted cost from the healthy depot state $(0, 0, 0)$
is $V^*(0,0,0) = \mathbf{3.959117}$.

Figure~\ref{fig:value} shows $V^*(x_1, x_2, 0)$ across all depot states. The
cost increases monotonically as both machines degrade, with the cheapest state
at $(0,0)$ and the most expensive at $(5,7)$ where both machines are at
failure. The jump in cost at $x_1 = 5$ and $x_2 = 7$ reflects the high cost
of corrective maintenance, which is consistent with the optimal policy
favouring preventive action before those states are reached.

\subsection*{Optimal Policy Structure}

Figure~\ref{fig:policy} shows $\pi^*(x_1, x_2)$ at $E = 0$ for all 48 depot
states. The policy has a threshold structure: the engineer maintains a machine
when its degradation level crosses a threshold, and does nothing when both
machines are below their respective thresholds.

The policy prescribes:
\begin{itemize}
    \item \textbf{Do nothing} when both machines are below their maintenance
          thresholds. The cost of travelling and repairing outweighs the
          expected future savings.
    \item \textbf{Maintain M1} when $x_1 \geq \tau_1 = 3$, provided $x_2$ is
          below its own threshold.
    \item \textbf{Maintain M2} when $x_2 \geq \tau_2 = 4$, provided $x_1$ is
          below its threshold.
    \item When both machines are above their thresholds, the policy chooses
          the machine with higher priority (M1 when $Q^1 < Q^2$, M2 otherwise).
    \item When a machine is at failure ($x_i = \xi_i$), corrective maintenance
          is mandatory (constraint from Task~1).
\end{itemize}

\section*{Appendix}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.75\textwidth]{figures/convergence.png}
    \caption{$L_\infty$ norm $\delta_k$ between successive value functions
             over VI iterations (log scale). The dashed red line marks
             $\varepsilon = 10^{-8}$.}
    \label{fig:convergence}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.65\textwidth]{figures/optimal_policy.png}
    \caption{Optimal action $\pi^*(x_1,x_2)$ at the depot ($E=0$).
             Blue: do nothing ($a=0$). Orange: maintain M1 ($a=1$).
             Green: maintain M2 ($a=2$). Dashed lines mark the failure
             thresholds $\xi_1 = 5$ and $\xi_2 = 7$.}
    \label{fig:policy}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.65\textwidth]{figures/value_function.png}
    \caption{Optimal value function $V^*(x_1, x_2, 0)$ at the depot.
             Darker cells indicate higher expected discounted cost.
             Dashed lines mark the failure thresholds $\xi_1 = 5$ and $\xi_2 = 7$.}
    \label{fig:value}
\end{figure}
