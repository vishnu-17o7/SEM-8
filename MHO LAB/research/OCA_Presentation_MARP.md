---
marp: true
theme: default 
class: lead
paginate: true
size: 16:9
style: |
  section h1 {
    text-align: center;
  }
  section h2 {
    text-align: center;
  }
title: The Overclocking Algorithm (OCA)
author: Vishn / MHO Lab
---

# The Overclocking Algorithm (OCA)
## A High-Performance Metaheuristic Inspired by CPU Architecture

**Course/Lab:** MHO Lab  
**Date:** March 2026

---

## Presentation Outline

1. Introduction
2. Characteristics of OCA
3. Flowchart
4. Initialization
5. Detailed Algorithm with Formulas
6. Termination Criteria
7. User-Defined Parameters of OCA
8. Pseudocode of OCA
9. Results Snapshot and Conclusion

---

## Introduction

- Metaheuristic optimization is essential for nonlinear, multimodal, and nonconvex problems.
- Many methods face a tradeoff between exploration power and computational cost.
- OCA introduces a CPU-inspired search process that balances:
  - global exploration,
  - local exploitation,
  - and runtime efficiency.
- Optimization objective:

\[
\mathbf{x}^* = \arg\min_{\mathbf{x}\in[LB,UB]^d} f(\mathbf{x})
\]

---

## Characteristics of OCA

- **CPU metaphor mapping**: core, voltage, throttling, cache miss, pipelining.
- **Tri-Core Leadership**: top 3 elite agents \((P_1, P_2, P_3)\) guide updates.
- **DVFS control**: adaptive transition from exploration to exploitation.
- **Instruction Pipelining (Momentum)**: improves convergence in curved valleys.
- **Cache Miss Reset**: mitigates stagnation in local optima.
- **Vectorized implementation**: practical runtime improvement.

---

## Flowchart of OCA

```text
START
  |
  v
Set parameters (N, d, Tmax, bounds, stall_limit, eps)
  |
  v
Initialize population X and velocity Vel
  |
  v
Evaluate fitness and select leaders P1, P2, P3
  |
  v
For t = 1 to Tmax
  |
  +--> Compute V(t), A, C
  |
  +--> Compute Tri-Core target X_target
  |
  +--> Update Vel and X
  |
  +--> Apply boundary handling and evaluate fitness
  |
  +--> Apply cache-miss reset for stagnated cores
  |
  +--> Update leaders P1, P2, P3
  |
  +--> Check termination criteria
  |
  v
Return best solution P1 and best fitness f(P1)
  |
  v
END
```

---

## Initialization

Population initialization for each agent \(i=1,\dots,N\):

\[
\mathbf{X}_i^{(0)} = \mathbf{LB} + \mathbf{r}_i \odot (\mathbf{UB} - \mathbf{LB}), \quad \mathbf{r}_i \sim U(0,1)^d
\]

Velocity initialization:

\[
\mathbf{Vel}_i^{(0)} = \mathbf{0}
\]

Initial fitness and leader selection:

\[
Fit_i^{(0)} = f\!\left(\mathbf{X}_i^{(0)}\right), \quad P_1,P_2,P_3 = \text{top-3 by fitness}
\]

---

## Detailed Algorithm (1): DVFS and Coefficients

Dynamic Voltage and Frequency Scaling (DVFS):

\[
V(t) = 2\left(1 - \frac{t}{T_{max}}\right)
\]

Coefficient vectors:

\[
\mathbf{A}_i(t) = 2V(t)\mathbf{r}_{1,i} - V(t), \quad \mathbf{C}_i(t) = 2\mathbf{r}_{2,i}
\]

Per-leader distance and candidate step for \(k\in\{1,2,3\}\):

\[
\mathbf{D}_{ik}(t)=\left|\mathbf{C}_i(t)\odot\mathbf{P}_k(t)-\mathbf{X}_i(t)\right|
\]

\[
\mathbf{Step}_{ik}(t)=\mathbf{P}_k(t)-\mathbf{A}_i(t)\odot\mathbf{D}_{ik}(t)
\]

---

## Detailed Algorithm (2): Target, Momentum, Bounds

Tri-Core target position:

\[
\mathbf{X}_{target,i}(t)=\frac{1}{3}\sum_{k=1}^{3}\mathbf{Step}_{ik}(t)
\]

Instruction-pipelining momentum update:

\[
\mathbf{Vel}_i(t+1)=w_i(t)\mathbf{Vel}_i(t)+\left(\mathbf{X}_{target,i}(t)-\mathbf{X}_i(t)\right)
\]

\[
\mathbf{X}_i(t+1)=\mathbf{X}_i(t)+\eta\,\mathbf{Vel}_i(t+1), \quad w_i(t)\sim U(0.5,0.9), \ \eta=0.5
\]

Boundary handling:

\[
\mathbf{X}_i(t+1)=\min\!\big(\max(\mathbf{X}_i(t+1),\mathbf{LB}),\mathbf{UB}\big)
\]

---

## Detailed Algorithm (3): Cache Miss and Elitism

Improvement tracking:

\[
\Delta f_i(t)=f\!\left(\mathbf{X}_i(t-1)\right)-f\!\left(\mathbf{X}_i(t)\right)
\]

Stagnation trigger:

\[
\Delta f_i(t) \le \epsilon \ \text{for consecutive iterations} \Rightarrow stall_i \uparrow
\]

Cache-miss reset condition and update:

\[
stall_i > stall_{limit} \Rightarrow \mathbf{X}_i \leftarrow \mathbf{P}_r + \sigma\,\mathcal{N}(0,\mathbf{I}), \ P_r\in\{P_1,P_2,P_3\}
\]

Elitism update each iteration:

\[
\{P_1,P_2,P_3\} \leftarrow \text{best-3 solutions in current population}
\]

---

## Termination Criteria

The run stops when any one of the following is true:

1. Maximum iteration reached:

\[
t \ge T_{max}
\]

2. Target fitness achieved (optional):

\[
f(P_1) \le f_{target}
\]

3. Global stagnation over patience window \(L\) (optional):

\[
\left|f_{best}(t)-f_{best}(t-L)\right| < \delta
\]

---

## User-Defined Parameters of OCA

| Parameter | Description | Typical Value/Range |
| :-- | :-- | :-- |
| \(N\) | Population size | 20 to 60 |
| \(d\) | Problem dimension | Problem-dependent |
| \(T_{max}\) | Max iterations | 100 to 1000 |
| \(LB,UB\) | Decision-variable bounds | Problem-dependent |
| \(w\) | Inertia weight | 0.5 to 0.9 |
| \(\eta\) | Velocity scaling factor | 0.5 |
| \(stall_{limit}\) | Cache miss trigger threshold | 8 to 15 |
| \(\epsilon\) | Improvement tolerance | \(10^{-8}\) to \(10^{-4}\) |
| \(\sigma\) | Reset noise scale | 0.01 to 0.2 of search span |
| \(f_{target}\) | Optional early-stop objective | User-defined |
| \(L,\delta\) | Optional patience and tolerance | User-defined |

---

## Pseudocode of OCA (Part 1)

```python
Input: f, N, d, LB, UB, Tmax, stall_limit, epsilon, eta, sigma
Output: Best solution P1 and best fitness f(P1)

Initialize X[i] = LB + rand(d) * (UB - LB) for i = 1..N
Initialize Vel[i] = zeros(d), stall[i] = 0
Evaluate fit[i] = f(X[i])
Select leaders P1, P2, P3 as best 3 solutions

for t in 1..Tmax:
    V = 2 * (1 - t / Tmax)

    for each agent i:
        r1, r2 = rand(d), rand(d)
        A = 2 * V * r1 - V
        C = 2 * r2

        D1 = abs(C * P1 - X[i]); Step1 = P1 - A * D1
        D2 = abs(C * P2 - X[i]); Step2 = P2 - A * D2
        D3 = abs(C * P3 - X[i]); Step3 = P3 - A * D3
```

---

## Pseudocode of OCA (Part 2)

```python
        X_target = (Step1 + Step2 + Step3) / 3
        w = uniform(0.5, 0.9)
        Vel[i] = w * Vel[i] + (X_target - X[i])
        X_new = X[i] + eta * Vel[i]
        X_new = clip(X_new, LB, UB)

        if f(X[i]) - f(X_new) <= epsilon:
            stall[i] += 1
        else:
            stall[i] = 0

        if stall[i] > stall_limit:
            Pr = random choice from {P1, P2, P3}
            X_new = Pr + sigma * normal(0, I)
            X_new = clip(X_new, LB, UB)
            stall[i] = 0

        X[i] = X_new

    Re-evaluate fitness for all agents
    Update leaders P1, P2, P3
    if termination condition met: break

return P1, f(P1)
```

---

## Results Snapshot (10D)

- OCA performs strongly on Rosenbrock and Schwefel.
- Benchmark runtime in your study:
  - OCA V7: about 0.25 s
  - GWO: about 0.70 s
- OCA shows a strong accuracy-speed tradeoff for difficult landscapes.

---

## Conclusion

- OCA is a CPU-inspired, formula-driven optimization framework.
- The key pipeline is: Tri-Core elitism + DVFS + momentum + cache-miss reset.
- It offers competitive solution quality with high runtime efficiency.
- It is suitable for future hybridization and high-dimensional extensions.

---

# Thank You
## Questions?
