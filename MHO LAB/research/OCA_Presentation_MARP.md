---
marp: true
theme: default
paginate: true
size: 16:9
title: The Overclocking Algorithm (OCA)
author: Vishn / MHO Lab
---

# The Overclocking Algorithm (OCA)
## A High-Performance Metaheuristic Inspired by CPU Architecture

**Course/Lab:** MHO Lab  
**Date:** February 2026

---

## Problem Motivation

- Real optimization problems are non-linear, multimodal, and high-dimensional.
- Classical methods are often too rigid for complex landscapes.
- Existing MHOs can be accurate but computationally expensive.
- Need: a method with **strong search ability + high execution speed**.

---

## Core Idea: CPU-Inspired Optimization

OCA models optimization behavior using CPU concepts:

- **Core** → search agent
- **Throughput (IPC)** → fitness
- **P-Cores** → elite leaders
- **DVFS** → exploration/exploitation control
- **Pipeline momentum** → stable, accelerated movement
- **Cache miss reset** → escape from stagnation

---

## OCA V7 (Lite) Architecture

- Population-based metaheuristic with vectorized updates.
- **Tri-Core leadership** using best 3 agents: \(P_1, P_2, P_3\).
- Full population guided by averaged elite suggestions.
- Lite design target: reduce loop-heavy overhead seen in standard implementations.

---

## Mathematical Formulation (1/2)

Initialization:

\[
X_i = LB + r \cdot (UB - LB)
\]

DVFS schedule:

\[
V(t) = 2\left(1 - \frac{t}{T_{max}}\right)
\]

Leader-guided components:

\[
D_k = |C\cdot P_k - X_i|, \quad Step_k = P_k - A\cdot D_k
\]

\[
X_{target} = \frac{1}{3}\sum_{k=1}^{3} Step_k
\]

---

## Mathematical Formulation (2/2)

Momentum (instruction pipelining):

\[
Vel_i(t+1)=w\,Vel_i(t)+(X_{target}-X_i(t))
\]

\[
X_i(t+1)=X_i(t)+0.5\,Vel_i(t+1)
\]

Stagnation escape (cache miss):

- If no improvement for threshold cycles, reset near random elite core.
- Promotes diversity and local-optima escape.

---

## Experimental Setup

- Benchmarks: Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, Griewank.
- Baselines: **PSO, GWO, GA, DE, FA**.
- Dimensions: **10D and 30D**.
- Population size: **30**.
- Iterations: **100**.
- Runs: **3 (averaged)**.

---

## Accuracy Highlights (10D)

- **Rosenbrock:** OCA outperforms GWO (better valley navigation).
- **Schwefel:** OCA significantly better than GWO (deceptive landscape handling).
- **Sphere/Ackley:** GWO remains highly competitive on unimodal precision.

Takeaway:

- OCA is especially effective on landscapes requiring momentum + reset diversity.

---

## Efficiency Highlights

Average runtime (10D):

- **OCA V7:** 0.25 s
- **GWO:** 0.70 s
- **FA:** 1.25 s

Key result:

- OCA is approximately **3x faster than GWO** while maintaining competitive or better accuracy on difficult functions.

---

## Why OCA Works

- **Tri-Core guidance** reduces dependence on a single leader.
- **DVFS decay** naturally transitions explore → exploit.
- **Momentum update** improves movement in narrow valleys.
- **Cache-miss reset** avoids long stagnation in local minima.
- **Vectorized implementation** improves practical runtime.

---

## Current Limitations

- Performance not uniformly best on all unimodal tasks.
- Parameter scheduling can still be problem-sensitive.
- Current experiments are moderate scale (10D/30D, limited runs).
- More statistical validation needed (larger run counts, significance tests).

---

## Future Work

- Hyper-threading style parallel sub-populations.
- Branch-prediction-inspired speculative search.
- 100D+ high-dimensional benchmarks.
- Real-time robotics path planning integration.
- Adaptive self-tuning parameters for reduced manual calibration.

---

## Conclusion

- OCA introduces a **non-biological, CPU-inspired** metaheuristic design.
- Combines **competitive solution quality** with **strong computational efficiency**.
- Most promising gains: deceptive and valley-shaped landscapes.
- OCA is a strong base for next-generation hybrid autonomous optimization.

---

# Thank You
## Questions?
