# Meta-Heuristic Optimization (MHO) Lab - Complete Guide

## Table of Contents
1. [Program 1: Exhaustive Search Optimization](#program-1-exhaustive-search-optimization)
2. [Program 2: Random Search Optimization](#program-2-random-search-optimization)
3. [Program 3: Ant Colony Optimization (ACO)](#program-3-ant-colony-optimization-aco)
4. [Program 4a: Tabu Search for N-Queens](#program-4a-tabu-search-for-n-queens)
5. [Program 4b: Simulated Annealing for TSP](#program-4b-simulated-annealing-for-tsp)
6. [Program 5: Particle Swarm Optimization (PSO)](#program-5-particle-swarm-optimization-pso)
7. [Program 6: Genetic Algorithm (GA)](#program-6-genetic-algorithm-ga)
8. [Program 7: Multi-Objective Genetic Algorithm (MOGA)](#program-7-multi-objective-genetic-algorithm-moga)
9. [Program 8: Adaptive Mutation in GA](#program-8-adaptive-mutation-in-ga)
10. [Program 9: Binary Genetic Algorithm](#program-9-binary-genetic-algorithm)
11. [Program 10: Continuous (Real-Coded) Genetic Algorithm](#program-10-continuous-real-coded-genetic-algorithm)

---

## Program 1: Exhaustive Search Optimization

### Algorithm Overview
Exhaustive Search (also called Brute Force) is the simplest optimization method that evaluates the objective function at every possible point in the search space to find the global optimum.

### Mathematical Formulation

**Optimization Problem:**
$$\min_{x \in S} f(x)$$

where $S$ is the discrete search space.

**Algorithm Steps:**
$$x^* = \arg\min_{x \in \{x_{min}, x_{min}+\Delta, x_{min}+2\Delta, ..., x_{max}\}} f(x)$$

where $\Delta$ is the step size.

### Detailed Algorithm

```
Algorithm: Exhaustive Search
Input: f(x) - objective function
       [x_min, x_max] - search bounds
       Δ - step size
Output: x* - optimal solution

1. Initialize: x* = x_min, f* = f(x_min)
2. FOR x = x_min TO x_max STEP Δ:
   a. Evaluate: f_current = f(x)
   b. IF f_current < f*:
      - Update: x* = x, f* = f_current
3. RETURN x*, f*
```

### Objective Function Used

**Quadratic Function:**
$$f(x) = x^2 - 4x + 5$$

**Finding the Analytical Minimum:**
$$\frac{df}{dx} = 2x - 4 = 0 \implies x^* = 2$$
$$f(2) = 4 - 8 + 5 = 1$$

This is a parabola with minimum at x = 2, f(2) = 1

### Parameter Analysis

| Parameter | Symbol | Description | Impact |
|-----------|--------|-------------|--------|
| **Lower Bound** | $x_{min}$ | Minimum value of search space | Must include potential optimum |
| **Upper Bound** | $x_{max}$ | Maximum value of search space | Must include potential optimum |
| **Step Size** | $\Delta$ | Increment between evaluations | Smaller = more accurate but slower |
| **Total Evaluations** | $N$ | $(x_{max} - x_{min})/\Delta + 1$ | Determines runtime |

### Key Characteristics
| Aspect | Description |
|--------|-------------|
| **Completeness** | Guaranteed to find global optimum |
| **Time Complexity** | O(n) for 1D, O(n^d) for d dimensions |
| **Space Complexity** | O(1) |
| **Scalability** | Poor - exponential with dimensions |

### VIVA Questions

**Q1: What is exhaustive search and when is it suitable?**
> A: Exhaustive search evaluates all possible solutions to find the global optimum. It's suitable when the search space is small, the function is expensive to evaluate only once, or when a guaranteed optimal solution is required.

**Q2: What are the limitations of exhaustive search?**
> A: The main limitations are:
> - Exponential time complexity in high dimensions (curse of dimensionality)
> - Impractical for continuous search spaces
> - Cannot handle large combinatorial problems

**Q3: How does the step size affect the search?**
> A: A smaller step size increases accuracy but exponentially increases computation time. A larger step size may miss the true optimum if it lies between sample points.

**Q4: What is the time complexity of exhaustive search for a 10-variable problem with 100 samples per variable?**
> A: O(100^10) = O(10^20) evaluations, which is computationally infeasible.

---

## Program 2: Random Search Optimization

### Algorithm Overview
Random Search is a stochastic optimization method that samples random points from the search space and keeps track of the best solution found.

### Mathematical Formulation

**Random Sampling:**
$$x_i \sim U(x_{min}, x_{max})$$

where $U$ denotes a uniform distribution.

**Best Solution Update:**
$$x^*_t = \arg\min_{x \in \{x_1, x_2, ..., x_t\}} f(x)$$

**Probability of Finding Optimum:**
If the optimal region occupies fraction $p$ of the search space, after $n$ samples:
$$P(\text{found}) = 1 - (1-p)^n$$

### Detailed Algorithm

```
Algorithm: Random Search
Input: f(x) - objective function
       [x_min, x_max] - search bounds  
       N - maximum iterations
Output: x* - best solution found

1. Initialize: x* = random(x_min, x_max), f* = f(x*)
2. FOR i = 1 TO N:
   a. Sample: x_i ~ Uniform(x_min, x_max)
   b. Evaluate: f_i = f(x_i)
   c. IF f_i < f*:
      - Update: x* = x_i, f* = f_i
3. RETURN x*, f*
```

### Parameter Analysis

| Parameter | Symbol | Description | Impact |
|-----------|--------|-------------|--------|
| **Lower Bound** | $x_{min}$ | Minimum search value | Defines search space |
| **Upper Bound** | $x_{max}$ | Maximum search value | Defines search space |
| **Max Iterations** | $N$ | Number of random samples | More iterations = higher probability of finding optimum |
| **Random Seed** | - | Seed for reproducibility | Same seed = same sequence of samples |

### Convergence Analysis

**Expected Number of Iterations to Find Optimum:**
$$E[n] = \frac{1}{p}$$

where $p$ is the fraction of search space containing acceptable solutions.

**Confidence Level:**
For 95% confidence of finding a solution occupying 1% of space:
$$n = \frac{\ln(1-0.95)}{\ln(1-0.01)} \approx 299 \text{ iterations}$$

### Key Characteristics
| Aspect | Description |
|--------|-------------|
| **Completeness** | No guarantee of finding global optimum |
| **Convergence** | Probabilistic - approaches optimum with more iterations |
| **Parallelizable** | Yes - samples are independent |
| **Scalability** | Better than exhaustive search |

### Advantages over Exhaustive Search
- Works with continuous search spaces
- Scales better to higher dimensions
- Can find good solutions quickly
- No assumptions about function shape

### VIVA Questions

**Q1: How does random search differ from exhaustive search?**
> A: Random search samples random points rather than systematically checking all points. It trades completeness for efficiency, working well in high dimensions but without optimality guarantees.

**Q2: What is the probability of finding the global optimum?**
> A: If the global optimum occupies a fraction p of the search space, after n independent samples, the probability of having sampled it at least once is 1 - (1-p)^n.

**Q3: How many iterations are needed to find a good solution?**
> A: This depends on the problem. For a solution occupying 1% of the space, approximately 300 iterations give ~95% probability of finding it.

**Q4: What are the stopping criteria for random search?**
> A: Common criteria include:
> - Maximum number of iterations
> - No improvement for N consecutive iterations
> - Solution quality threshold reached
> - Time limit

---

## Program 3: Ant Colony Optimization (ACO)

### Algorithm Overview
ACO is a swarm intelligence algorithm inspired by the foraging behavior of ants. Ants deposit pheromones on paths they traverse, and better paths accumulate more pheromones, guiding future ants.

### Mathematical Formulation

#### State Transition Rule (Probability of Moving from City i to j)

$$P_{ij}^k = \begin{cases} 
\frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta} & \text{if } j \in N_i^k \\
0 & \text{otherwise}
\end{cases}$$

where:
- $\tau_{ij}$ = pheromone intensity on edge (i, j)
- $\eta_{ij} = \frac{1}{d_{ij}}$ = heuristic desirability (inverse of distance)
- $N_i^k$ = set of unvisited cities for ant k at city i
- $\alpha$ = pheromone influence parameter
- $\beta$ = heuristic influence parameter

#### Pheromone Update Rule

**Evaporation:**
$$\tau_{ij}(t+1) = (1 - \rho) \cdot \tau_{ij}(t)$$

**Deposition:**
$$\tau_{ij}(t+1) = \tau_{ij}(t) + \sum_{k=1}^{m} \Delta\tau_{ij}^k$$

where:
$$\Delta\tau_{ij}^k = \begin{cases} 
\frac{Q}{L_k} & \text{if ant k used edge (i,j)} \\
0 & \text{otherwise}
\end{cases}$$

- $Q$ = pheromone deposit constant
- $L_k$ = total tour length of ant k
- $\rho$ = evaporation rate

### Detailed Algorithm

```
Algorithm: Ant Colony Optimization (Ant System)
Input: Distance matrix D, parameters α, β, ρ, m, max_iter
Output: Best tour and its length

1. INITIALIZE:
   - τij = τ0 for all edges (small positive value)
   - best_tour = ∅, best_length = ∞

2. FOR iteration = 1 TO max_iter:
   
   a. CONSTRUCT SOLUTIONS:
      FOR each ant k = 1 TO m:
         - Place ant at random starting city
         - WHILE unvisited cities remain:
            * Calculate Pij for all unvisited cities j
            * Select next city using roulette wheel on Pij
            * Move to selected city, mark as visited
         - Complete tour (return to start)
         - Calculate tour length Lk
   
   b. UPDATE PHEROMONES:
      - Evaporation: τij = (1-ρ) × τij for all edges
      - Deposition: τij += Σ Δτij^k for ants using edge (i,j)
   
   c. UPDATE BEST:
      IF min(Lk) < best_length:
         - best_tour = tour of best ant
         - best_length = min(Lk)

3. RETURN best_tour, best_length
```

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect of Increasing |
|-----------|--------|-------|-------------|----------------------|
| **Pheromone Influence** | $\alpha$ | 0-5 | Weight of pheromone in decision | More exploitation; follows learned paths |
| **Heuristic Influence** | $\beta$ | 1-5 | Weight of distance in decision | More greedy behavior; prefers short edges |
| **Evaporation Rate** | $\rho$ | 0.1-0.5 | Rate of pheromone decay per iteration | Faster forgetting; more exploration |
| **Number of Ants** | $m$ | 10-100 | Ants per iteration | Better exploration; more computation |
| **Initial Pheromone** | $\tau_0$ | 0.1-1 | Starting pheromone level | Higher = less initial bias |
| **Pheromone Constant** | $Q$ | 1-100 | Scaling for pheromone deposit | Higher = stronger reinforcement |

### Parameter Tuning Guidelines

**α (Pheromone Influence):**
- α = 0: Ignores pheromones → pure greedy algorithm
- α = 1: Balanced influence (recommended starting point)
- α > 2: Strong exploitation, may cause premature convergence

**β (Heuristic Influence):**
- β = 0: Ignores distances → random walk with pheromone bias
- β = 2-3: Recommended range for TSP
- β > 5: Very greedy, may miss global optimum

**ρ (Evaporation Rate):**
- ρ ≈ 0.1: Slow evaporation → long memory, slow adaptation
- ρ ≈ 0.5: Balanced (common choice)
- ρ ≈ 0.9: Fast evaporation → short memory, more exploration

### VIVA Questions

**Q1: Why does ACO use both pheromone and heuristic information?**
> A: Pheromone (τ) represents learned information from previous ants (exploitation), while heuristic (η = 1/distance) represents problem-specific knowledge (greedy guidance). Combining them balances exploration and exploitation.

**Q2: What happens if evaporation rate is too high or too low?**
> A: 
> - Too high (ρ → 1): Pheromones disappear quickly, losing learned information, behaving like random search
> - Too low (ρ → 0): Old paths dominate, may get stuck in local optima

**Q3: How does ACO avoid getting stuck in local optima?**
> A: Through:
> - Probabilistic path selection (not always choosing the best)
> - Pheromone evaporation (forgetting old paths)
> - Multiple ants exploring different paths simultaneously

**Q4: What is the difference between Ant System (AS) and Ant Colony System (ACS)?**
> A: ACS includes:
> - Local pheromone update during tour construction
> - Pseudo-random proportional rule (exploitation vs exploration)
> - Only the best ant deposits pheromones (elitism)

**Q5: Can ACO solve continuous optimization problems?**
> A: Standard ACO is designed for discrete/combinatorial problems. Extensions like ACO_R adapt it for continuous domains using probability density functions.

---

## Program 4a: Tabu Search for N-Queens

### Algorithm Overview
Tabu Search is a local search method that uses memory (tabu list) to avoid revisiting recent solutions, helping escape local optima.

### Mathematical Formulation

#### N-Queens Problem Definition

For an N×N chessboard, place N queens such that no two queens attack each other.

**Conflict Function (Objective to Minimize):**
$$f(S) = \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \mathbb{1}[\text{queens } i \text{ and } j \text{ attack each other}]$$

For permutation representation $S = [s_1, s_2, ..., s_N]$ where $s_i$ = row of queen in column $i$:

**Diagonal Attack Condition:**
$$\text{Attack}(i,j) = \begin{cases} 
1 & \text{if } |i - j| = |s_i - s_j| \\
0 & \text{otherwise}
\end{cases}$$

#### Neighborhood Structure

**Swap Neighborhood:**
$$N(S) = \{S' : S' = \text{swap}(S, i, j) \text{ for } 1 \leq i < j \leq N\}$$

**Neighborhood Size:**
$$|N(S)| = \binom{N}{2} = \frac{N(N-1)}{2}$$

### Detailed Algorithm

```
Algorithm: Tabu Search for N-Queens
Input: N (board size), max_iter, tabu_tenure
Output: Best solution found

1. INITIALIZE:
   - S_current = random_permutation(1..N)
   - S_best = S_current
   - f_best = conflicts(S_current)
   - TabuList = empty queue

2. FOR iteration = 1 TO max_iter:
   
   a. IF f_best = 0: RETURN S_best  (solution found)
   
   b. GENERATE NEIGHBORHOOD:
      - N(S_current) = all swaps (i,j) for i < j
   
   c. EVALUATE AND SELECT:
      - best_neighbor = null, best_move = null
      - FOR each neighbor S' from move (i,j):
         * f' = conflicts(S')
         * is_tabu = (i,j) ∈ TabuList
         * aspiration = (f' < f_best)  // Better than ever seen
         
         IF (NOT is_tabu) OR aspiration:
            IF f' < conflicts(best_neighbor):
               best_neighbor = S'
               best_move = (i,j)
   
   d. MOVE:
      - S_current = best_neighbor
      - Add best_move to TabuList
      - IF |TabuList| > tabu_tenure: Remove oldest entry
   
   e. UPDATE BEST:
      - IF conflicts(S_current) < f_best:
         S_best = S_current
         f_best = conflicts(S_current)

3. RETURN S_best
```

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect of Increasing |
|-----------|--------|-------|-------------|----------------------|
| **Board Size** | $N$ | 4-1000+ | Problem dimension | Exponentially harder |
| **Tabu Tenure** | $T$ | $\sqrt{N}$ to $N$ | Iterations a move stays forbidden | More exploration; may miss good neighbors |
| **Max Iterations** | $I_{max}$ | 100-10000 | Stopping criterion | More time to find solution |
| **Neighborhood Size** | $|N(S)|$ | $\frac{N(N-1)}{2}$ | Moves evaluated per iteration | Larger = better local choice but slower |

### Tabu Tenure Guidelines

$$T_{optimal} \approx \sqrt{N} \text{ to } N$$

| N | Recommended Tenure |
|---|-------------------|
| 8 | 3-8 |
| 20 | 5-20 |
| 100 | 10-50 |

### Key Components

| Component | Mathematical Definition | Purpose |
|-----------|------------------------|----------|
| **Tabu List** | $\{(i,j)_1, (i,j)_2, ..., (i,j)_T\}$ | Forbids recent moves |
| **Tabu Tenure** | $T = |TabuList|_{max}$ | Controls memory length |
| **Aspiration** | $f(S') < f_{best}$ | Override tabu if promising |
| **Neighborhood** | $N(S) = \{\text{all swap moves}\}$ | Defines local search space |

### VIVA Questions

**Q1: What is the purpose of the tabu list?**
> A: The tabu list prevents cycling by forbidding recently visited solutions or moves. This forces the search to explore new regions rather than oscillating between the same solutions.

**Q2: What is tabu tenure and how does it affect the search?**
> A: Tabu tenure is how long a move remains forbidden.
> - Short tenure: More local exploitation, risk of cycling
> - Long tenure: More exploration, may miss good neighbors

**Q3: Explain aspiration criteria with an example.**
> A: Aspiration criteria allow overriding tabu status under special conditions. Example: If a tabu move leads to a solution better than the best-ever found, accept it anyway because it's clearly promising.

**Q4: How is the N-Queens problem represented in this implementation?**
> A: As a permutation where index = column and value = row. Example: [2, 0, 3, 1] means Queen in column 0 is in row 2, etc. This automatically prevents row and column conflicts.

**Q5: What is the difference between short-term and long-term memory in Tabu Search?**
> A: 
> - Short-term memory: Tabu list (recent moves to avoid)
> - Long-term memory: Frequency-based diversification/intensification
>   - Diversification: Encourage moves to rarely visited regions
>   - Intensification: Encourage moves in promising regions

---

## Program 4b: Simulated Annealing for TSP

### Algorithm Overview
Simulated Annealing (SA) is inspired by the metallurgical annealing process. It accepts worse solutions with a probability that decreases over time, allowing escape from local optima.

### Mathematical Formulation

#### TSP Problem Definition

Given $n$ cities with distance matrix $D = [d_{ij}]$, find a tour visiting each city exactly once with minimum total distance.

**Objective Function:**
$$f(\pi) = \sum_{i=1}^{n-1} d_{\pi_i, \pi_{i+1}} + d_{\pi_n, \pi_1}$$

where $\pi = [\pi_1, \pi_2, ..., \pi_n]$ is a permutation of cities.

#### Metropolis Acceptance Criterion

**Energy Change:**
$$\Delta E = f(S') - f(S) = E_{new} - E_{current}$$

**Acceptance Probability:**
$$P(\text{accept}) = \begin{cases} 
1 & \text{if } \Delta E < 0 \text{ (improvement)}\\
\exp\left(-\frac{\Delta E}{T}\right) & \text{if } \Delta E \geq 0 \text{ (worsening)}
\end{cases}$$

#### Cooling Schedules

**Geometric (Most Common):**
$$T_{k+1} = \alpha \cdot T_k$$
where $\alpha \in [0.95, 0.99]$

**Linear:**
$$T_k = T_0 - k \cdot \frac{T_0 - T_{min}}{k_{max}}$$

**Logarithmic (Theoretically Optimal):**
$$T_k = \frac{T_0}{\ln(1 + k)}$$

### Detailed Algorithm

```
Algorithm: Simulated Annealing for TSP
Input: Cities C, distance matrix D, T0, α, T_min
Output: Best tour found

1. INITIALIZE:
   - S_current = random_tour(C)
   - E_current = tour_distance(S_current)
   - S_best = S_current, E_best = E_current
   - T = T0

2. WHILE T > T_min:
   
   a. GENERATE NEIGHBOR:
      - Select random i, j where i < j
      - S_new = swap(S_current, i, j)  // or 2-opt
      - E_new = tour_distance(S_new)
   
   b. CALCULATE ACCEPTANCE:
      - ΔE = E_new - E_current
      - IF ΔE < 0:
         Accept (always)
      - ELSE:
         P = exp(-ΔE / T)
         IF random() < P: Accept
   
   c. IF accepted:
      - S_current = S_new
      - E_current = E_new
      - IF E_current < E_best:
         S_best = S_current
         E_best = E_current
   
   d. COOL DOWN:
      - T = α × T

3. RETURN S_best, E_best
```

### Detailed Parameter Analysis

| Parameter | Symbol | Typical Range | Description | Effect |
|-----------|--------|---------------|-------------|--------|
| **Initial Temperature** | $T_0$ | 1000-10000 | Starting temperature | Higher = more initial exploration, accepts worse moves |
| **Cooling Rate** | $\alpha$ | 0.95-0.99 | Multiplicative decay factor | Lower = faster cooling, quicker convergence |
| **Stopping Temperature** | $T_{min}$ | $10^{-8}$ to $10^{-3}$ | Termination condition | Lower = more refinement at end |
| **Iterations per Temperature** | $L$ | 1-100 | Moves tried at each T | Higher = more exploration at each level |

### Temperature Selection Guidelines

**Initial Temperature $T_0$:**
Set such that initial acceptance rate $\approx$ 80%:
$$T_0 = -\frac{\bar{\Delta E}}{\ln(0.8)}$$

where $\bar{\Delta E}$ is the average positive energy change from random moves.

**Number of Temperature Steps:**
$$n_{steps} = \frac{\ln(T_{min}/T_0)}{\ln(\alpha)}$$

**Example:** $T_0 = 10000$, $T_{min} = 0.01$, $\alpha = 0.99$:
$$n_{steps} = \frac{\ln(0.01/10000)}{\ln(0.99)} \approx 1380 \text{ temperature levels}$$

### Neighbor Generation Methods

| Method | Description | Change in Tour Length |
|--------|-------------|----------------------|
| **2-city Swap** | Exchange positions of cities i and j | Small |
| **2-opt** | Reverse segment between i and j | Medium |
| **3-opt** | Reconnect 3 edges in different ways | Large |
| **Insert** | Remove city and insert elsewhere | Small-Medium |

### VIVA Questions

**Q1: Why does SA accept worse solutions?**
> A: Accepting worse solutions allows escape from local optima. Early in the search (high T), this happens frequently for exploration. Later (low T), it happens rarely for exploitation.

**Q2: What is the significance of the cooling schedule?**
> A: The cooling schedule controls the exploration-exploitation trade-off:
> - Fast cooling: Quicker convergence but may miss global optimum
> - Slow cooling: Better solutions but more computation time

**Q3: How is the initial temperature chosen?**
> A: Common approaches:
> - Set high enough that initial acceptance rate ≈ 80%
> - Based on expected ΔE values in the problem
> - Trial runs to determine appropriate range

**Q4: What neighbor generation method is used for TSP?**
> A: Two-city swap: Randomly select two cities and swap their positions in the tour. Other options include 2-opt (reverse a segment) or insertion moves.

**Q5: How does SA relate to the Metropolis algorithm?**
> A: SA is based on the Metropolis algorithm from statistical mechanics. The acceptance criterion exp(-ΔE/kT) comes from the Boltzmann distribution describing particle energy states at temperature T.

**Q6: What is the difference between SA and hill climbing?**
> A: Hill climbing only accepts improving moves (greedy), getting stuck in local optima. SA accepts worse moves probabilistically, enabling escape from local optima at the cost of slower convergence.

---

## Program 5: Particle Swarm Optimization (PSO)

### Algorithm Overview
PSO is a swarm intelligence algorithm inspired by bird flocking. Particles (solutions) move through the search space, influenced by their own best position and the swarm's best position.

### Mathematical Formulation

#### Velocity Update Equation

$$v_i^{t+1} = \underbrace{w \cdot v_i^t}_{\text{Inertia}} + \underbrace{c_1 \cdot r_1 \cdot (pbest_i - x_i^t)}_{\text{Cognitive Component}} + \underbrace{c_2 \cdot r_2 \cdot (gbest - x_i^t)}_{\text{Social Component}}$$

#### Position Update Equation

$$x_i^{t+1} = x_i^t + v_i^{t+1}$$

#### Component Breakdown

| Component | Formula | Physical Meaning |
|-----------|---------|------------------|
| **Inertia** | $w \cdot v_i^t$ | Momentum - tendency to continue current direction |
| **Cognitive** | $c_1 \cdot r_1 \cdot (pbest_i - x_i^t)$ | Memory - attraction to personal best |
| **Social** | $c_2 \cdot r_2 \cdot (gbest - x_i^t)$ | Cooperation - attraction to swarm's best |

### Detailed Algorithm

```
Algorithm: Particle Swarm Optimization
Input: f(x) - objective function
       bounds - [x_min, x_max] for each dimension
       n - number of particles
       max_iter - maximum iterations
       w, c1, c2 - PSO parameters
Output: gbest - best solution found

1. INITIALIZE:
   FOR each particle i = 1 TO n:
      - x_i = random position within bounds
      - v_i = random velocity (small values)
      - pbest_i = x_i
      - pbest_fitness_i = f(x_i)
   - gbest = argmin(pbest_fitness)
   - gbest_fitness = min(pbest_fitness)

2. FOR iteration = 1 TO max_iter:
   
   a. FOR each particle i:
      
      # Update velocity
      r1, r2 = random(0,1), random(0,1)
      v_i = w * v_i 
          + c1 * r1 * (pbest_i - x_i)
          + c2 * r2 * (gbest - x_i)
      
      # Clamp velocity if needed
      v_i = clamp(v_i, v_min, v_max)
      
      # Update position
      x_i = x_i + v_i
      
      # Apply bounds
      x_i = clamp(x_i, x_min, x_max)
      
      # Evaluate
      fitness = f(x_i)
      
      # Update personal best
      IF fitness < pbest_fitness_i:
         pbest_i = x_i
         pbest_fitness_i = fitness
      
      # Update global best
      IF fitness < gbest_fitness:
         gbest = x_i
         gbest_fitness = fitness

3. RETURN gbest, gbest_fitness
```

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect of Increasing |
|-----------|--------|-------|-------------|----------------------|
| **Inertia Weight** | $w$ | 0.4-0.9 | Controls momentum and exploration | More global search; slower convergence |
| **Cognitive Coefficient** | $c_1$ | 1.5-2.5 | Self-confidence; attraction to pbest | More individual exploration |
| **Social Coefficient** | $c_2$ | 1.5-2.5 | Swarm confidence; attraction to gbest | Faster convergence; may cause premature convergence |
| **Swarm Size** | $n$ | 20-100 | Number of particles | Better exploration; more evaluations |
| **Velocity Limit** | $v_{max}$ | 10-20% of range | Maximum velocity per dimension | Prevents overshooting; limits step size |

### Parameter Interaction and Tuning

#### Inertia Weight Strategies

**Fixed Value:**
$$w = 0.729 \text{ (constriction coefficient)}$$

**Linear Decreasing:**
$$w(t) = w_{max} - \frac{(w_{max} - w_{min}) \cdot t}{t_{max}}$$

Typically: $w_{max} = 0.9$, $w_{min} = 0.4$

**Nonlinear Decreasing:**
$$w(t) = w_{max} \cdot \left(\frac{w_{min}}{w_{max}}\right)^{t/t_{max}}$$

#### Constriction Factor (Alternative Formulation)

$$\chi = \frac{2}{|2 - \phi - \sqrt{\phi^2 - 4\phi}|}$$

where $\phi = c_1 + c_2 > 4$

$$v_i^{t+1} = \chi \cdot \left[v_i^t + c_1 r_1 (pbest_i - x_i) + c_2 r_2 (gbest - x_i)\right]$$

**Standard values:** $c_1 = c_2 = 2.05 \Rightarrow \chi \approx 0.729$

### Special Variants

#### Binary PSO (for Feature Selection)

**Sigmoid Transformation:**
$$S(v_{id}) = \frac{1}{1 + e^{-v_{id}}}$$

**Position Update:**
$$x_{id} = \begin{cases}
1 & \text{if } r < S(v_{id}) \\
0 & \text{otherwise}
\end{cases}$$

#### Velocity Clamping

$$v_{id} = \begin{cases}
v_{max} & \text{if } v_{id} > v_{max} \\
-v_{max} & \text{if } v_{id} < -v_{max} \\
v_{id} & \text{otherwise}
\end{cases}$$

Typically: $v_{max} = 0.1 \times (x_{max} - x_{min})$

---

### Subdivision 1: PSO from Scratch (Sphere Function Optimization)

#### Problem Description
Implement the basic PSO algorithm from scratch to minimize the Sphere function, a classic benchmark for optimization algorithms.

#### Objective Function - Sphere Function

$$f(\mathbf{x}) = \sum_{i=1}^{d} x_i^2$$

**Properties:**
- Unimodal (single global minimum)
- Convex and continuous
- Global minimum at $\mathbf{x}^* = (0, 0, ..., 0)$, $f(\mathbf{x}^*) = 0$

#### Implementation Details

| Component | Description |
|-----------|-------------|
| **Particle Class** | Stores position, velocity, personal best position, and personal best score |
| **Initialization** | Random positions within bounds, random small velocities |
| **Evaluation** | Apply sphere function to each particle's position |
| **Update Rule** | Standard PSO velocity and position update equations |
| **Boundary Handling** | Clip positions to stay within bounds |

#### Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| $w$ | 0.5 | Inertia weight |
| $c_1$ | 1.5 | Cognitive coefficient |
| $c_2$ | 1.5 | Social coefficient |
| Particles | 30 | Swarm size |
| Iterations | 100 | Maximum iterations |
| Dimensions | 3 | Search space dimensionality |
| Bounds | [-10, 10] | Search range per dimension |

---

### Subdivision 2: Traveling Salesman Problem (TSP) with PySwarms

#### Problem Description
Use PSO to find the shortest route visiting all cities exactly once and returning to the starting city.

#### Continuous-to-Discrete Encoding

Since PSO operates in continuous space but TSP requires discrete permutations, we use **argsort encoding**:

$$\text{route} = \text{argsort}(\mathbf{x})$$

**Example:**
- Particle position: $\mathbf{x} = [0.3, 0.8, 0.1, 0.5, 0.2]$
- Sorted indices: $[2, 4, 0, 3, 1]$ (visit order)

#### Objective Function

$$f(\mathbf{x}) = \sum_{i=1}^{n-1} d_{\pi_i, \pi_{i+1}} + d_{\pi_n, \pi_1}$$

where $\pi = \text{argsort}(\mathbf{x})$ and $d_{i,j}$ is the distance between cities $i$ and $j$.

#### Distance Matrix Calculation

For cities with coordinates $(x_i, y_i)$:

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

#### Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| $c_1$ | 0.5 | Cognitive coefficient |
| $c_2$ | 0.3 | Social coefficient |
| $w$ | 0.9 | Inertia weight |
| Particles | 20 | Swarm size |
| Cities | 5 | Number of cities |
| Iterations | 100 | Maximum iterations |

---

### Subdivision 3: Time Series Parameter Estimation (Airline Passengers)

#### Problem Description
Use PSO to fit a non-linear model (trend + seasonality) to the classic airline passengers dataset.

#### Model Structure

$$y(t) = \underbrace{(mt + c)}_{\text{Linear Trend}} \times \underbrace{(1 + A \sin(Bt + D))}_{\text{Seasonal Component}}$$

**Parameters to Optimize:**
| Parameter | Description |
|-----------|-------------|
| $m$ | Slope of linear trend |
| $c$ | Intercept of linear trend |
| $A$ | Amplitude of seasonality |
| $B$ | Frequency of seasonality |
| $D$ | Phase shift of seasonality |

#### Objective Function - Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

#### Data Preprocessing

**Normalization (Critical for PSO):**
$$y_{norm} = \frac{y}{y_{max}}, \quad t_{norm} = \frac{t}{t_{max}}$$

#### Parameter Bounds

| Parameter | Min | Max | Reasoning |
|-----------|-----|-----|-----------|
| $m$ | 0.0 | 1.0 | Normalized trend slope |
| $c$ | 0.0 | 0.5 | Normalized intercept |
| $A$ | 0.0 | 0.3 | Seasonality amplitude (fraction of trend) |
| $B$ | 70.0 | 85.0 | Frequency (~12 cycles in 144 months, $\approx 2\pi \times 12$) |
| $D$ | 0.0 | $2\pi$ | Phase shift |

#### PSO Configuration

| Parameter | Value |
|-----------|-------|
| $c_1, c_2$ | 0.5 |
| $w$ | 0.9 |
| Particles | 100 (increased for harder problem) |
| Iterations | 300 |

---

### Subdivision 4: Optimization using NiaPy Library

#### Problem Description
Demonstrate PSO using the NiaPy optimization library as an alternative to PySwarms.

#### NiaPy Framework Structure

```
Task → Problem → Algorithm → Run
```

| Component | Description |
|-----------|-------------|
| **Task** | Defines the optimization problem and constraints |
| **Problem** | Built-in benchmark functions (Sphere, Rastrigin, etc.) |
| **Algorithm** | PSO implementation with configurable parameters |

#### Implementation

```python
from niapy.algorithms.basic import ParticleSwarmOptimization
from niapy.task import Task
from niapy.problems import Sphere

task = Task(problem=Sphere(dimension=3), max_iters=100)
algo = ParticleSwarmOptimization(population_size=30, c1=2.0, c2=2.0, w=0.7)
best_score, best_solution = algo.run(task)
```

#### Key Differences from PySwarms

| Feature | PySwarms | NiaPy |
|---------|----------|-------|
| **API Style** | Optimizer-centric | Task-centric |
| **Objective Function** | User-defined | Built-in + custom |
| **Return Order** | (cost, position) | (score, solution) |
| **Population Parameter** | n_particles | population_size |
| **Algorithms Available** | PSO variants | 100+ algorithms |

---

### Subdivision 5: Feature Selection using Binary PSO

#### Problem Description
Use Binary PSO to select optimal features for a machine learning classifier (Logistic Regression) on the Breast Cancer dataset.

#### Binary PSO Formulation

**Sigmoid Transformation:**
$$S(v_{id}) = \frac{1}{1 + e^{-v_{id}}}$$

**Binary Position Update:**
$$x_{id} = \begin{cases}
1 & \text{if } r < S(v_{id}) \\
0 & \text{otherwise}
\end{cases}$$

where $r \sim U(0,1)$ is a random number.

#### Feature Selection Representation

- **Particle Dimension**: 30 (number of features in dataset)
- **Position Values**: Binary (0 = exclude feature, 1 = include feature)

**Example:**
$$\mathbf{x} = [1, 0, 1, 1, 0, ..., 1] \rightarrow \text{Select features at indices where } x_i = 1$$

#### Objective Function

$$f(\mathbf{x}) = 1 - \text{Accuracy}(\text{Model trained with selected features})$$

**Penalty for Empty Selection:**
If $\sum x_i = 0$, return $f(\mathbf{x}) = 1.0$ (worst case)

#### Dataset Details

| Property | Value |
|----------|-------|
| Dataset | Breast Cancer Wisconsin |
| Total Features | 30 |
| Samples | 569 |
| Classes | 2 (Malignant/Benign) |
| Train/Test Split | 70%/30% |

#### Binary PSO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $c_1$ | 0.5 | Cognitive coefficient |
| $c_2$ | 0.5 | Social coefficient |
| $w$ | 0.9 | Inertia weight |
| $k$ | 5 | Number of neighbors (local best topology) |
| $p$ | 2 | Distance metric (L2 norm) |
| Particles | 15 | Swarm size |
| Iterations | 20 | Maximum iterations |

**Note:** $k$ must be less than n_particles.

---

### Subdivision 6: Data Clustering using PSO

#### Problem Description
Use PSO to find optimal cluster centroids for K-means-style clustering on the Breast Cancer dataset.

#### Particle Encoding

Each particle represents $K$ cluster centroids in $D$-dimensional space:

$$\text{Particle} = [\underbrace{c_{1,1}, ..., c_{1,D}}_{\text{Centroid 1}}, \underbrace{c_{2,1}, ..., c_{2,D}}_{\text{Centroid 2}}, ..., \underbrace{c_{K,1}, ..., c_{K,D}}_{\text{Centroid K}}]$$

**Total Dimensions:** $K \times D$ (for 2 clusters with 30 features: 60 dimensions)

#### Objective Function - Sum of Squared Errors (SSE)

$$\text{SSE} = \sum_{i=1}^{n} \min_{j \in \{1,...,K\}} \|\mathbf{x}_i - \mathbf{c}_j\|^2$$

Each data point is assigned to the nearest centroid, and we minimize the total squared distances.

#### Data Preprocessing - Critical for PSO Clustering

**StandardScaler Normalization:**
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

This is **crucial** because:
1. Features have different scales (e.g., mean radius: 6-28 vs. mean area: 143-2501)
2. PSO bounds become manageable (approximately -3 to 3 for standard normal data)
3. Distance calculations are meaningful across all features

#### Implementation Details

| Component | Value/Description |
|-----------|-------------------|
| Clusters ($K$) | 2 |
| Features ($D$) | 30 |
| Particle Dimensions | 60 (2 × 30) |
| Bounds | [-3, 3] (scaled data range) |
| Distance Function | `pairwise_distances_argmin_min` |

#### PSO Parameters

| Parameter | Value |
|-----------|-------|
| $c_1$ | 0.5 |
| $c_2$ | 0.3 |
| $w$ | 0.9 |
| Particles | 30 |
| Iterations | 50 |

#### Comparison: PSO Clustering vs K-Means

| Aspect | PSO Clustering | K-Means |
|--------|----------------|---------|
| **Initialization** | Random within bounds | Random data points |
| **Update Rule** | PSO velocity equation | Move to cluster mean |
| **Local Minima** | Better escape through swarm | Often stuck |
| **Computation** | More expensive | Very fast |
| **Hyperparameters** | w, c1, c2, particles | K only |

---

### Applications Summary

| Subdivision | Application | Encoding | Objective |
|-------------|-------------|----------|-----------|
| 1 | Function Optimization | Continuous | Sphere function |
| 2 | TSP | Continuous → Permutation | Route distance |
| 3 | Time Series Fitting | Continuous (5 params) | MSE |
| 4 | NiaPy Demo | Continuous | Sphere function |
| 5 | Feature Selection | Binary | 1 - Accuracy |
| 6 | Clustering | Continuous centroids | SSE |

---

### VIVA Questions

**Q1: Explain the three components of the velocity update equation.**
> A: 
> - **Inertia (wv)**: Keeps particle moving in current direction
> - **Cognitive (c1·r1·(pbest-x))**: Pulls toward personal best experience
> - **Social (c2·r2·(gbest-x))**: Pulls toward swarm's best position

**Q2: What happens if c1 >> c2 or c2 >> c1?**
> A:
> - c1 >> c2: Particles trust personal experience more (exploration, slow convergence)
> - c2 >> c1: Particles follow global best (fast convergence, may get stuck in local optima)

**Q3: How is PSO applied to discrete problems like TSP?**
> A: Common approaches:
> - Continuous-to-discrete mapping (argsort of position vector)
> - Define discrete velocity operators (swap sequences)
> - Binary PSO for {0,1} decisions

**Q4: What is the role of inertia weight?**
> A: Inertia weight controls exploration vs exploitation:
> - High w (0.9): More exploration (particle maintains momentum)
> - Low w (0.4): More exploitation (particle responds to pbest/gbest)
> - Often linearly decreased from 0.9 to 0.4 during the run

**Q5: How does Binary PSO differ from continuous PSO?**
> A: In Binary PSO:
> - Velocity represents probability of bit being 1
> - Position update uses sigmoid: P(x=1) = sigmoid(v)
> - Position is 0 or 1, not continuous

**Q6: What are the advantages of PSO over Genetic Algorithms?**
> A:
> - Fewer parameters to tune
> - No crossover/mutation operators needed
> - Often faster convergence
> - Easier to implement
> - Particles share information immediately (vs. GA's generation-based)

**Q7: Why is data normalization important for PSO in time series fitting?**
> A: Normalization is crucial because:
> - Prevents numerical overflow with large values
> - Makes parameter bounds more intuitive (0-1 range)
> - Ensures all parameters have similar scales for balanced optimization
> - Improves convergence speed and stability

**Q8: Explain the argsort encoding for TSP in PSO.**
> A: Argsort encoding converts continuous particle positions to discrete permutations:
> - Each particle has continuous values (e.g., [0.3, 0.8, 0.1, 0.5])
> - argsort returns indices that would sort the array (e.g., [2, 0, 3, 1])
> - This gives a valid permutation (city visitation order)
> - Allows standard PSO operations while solving discrete problems

**Q9: What is the purpose of the penalty in feature selection objective?**
> A: If a particle selects zero features (all bits = 0):
> - The classifier cannot be trained
> - We assign maximum penalty (f = 1.0)
> - This guides particles away from empty feature subsets
> - Ensures valid solutions throughout optimization

**Q10: Why use StandardScaler before PSO clustering?**
> A: StandardScaler normalizes features to zero mean and unit variance because:
> - Different features have vastly different scales (radius: 6-28, area: 143-2501)
> - Without scaling, large-scale features dominate distance calculations
> - Bounds become manageable (approximately -3 to 3)
> - Clustering becomes balanced across all features

**Q11: How does PSO clustering differ from K-Means?**
> A: Key differences:
> - **Initialization**: PSO uses random positions; K-Means uses random data points
> - **Update**: PSO uses velocity equations; K-Means moves centroids to cluster means
> - **Exploration**: PSO maintains diversity through swarm dynamics; K-Means is purely greedy
> - **Local optima**: PSO can escape better; K-Means often stuck
> - **Speed**: K-Means is faster; PSO is more thorough

**Q12: What is the difference between PySwarms and NiaPy libraries?**
> A:
> - **PySwarms**: Specialized for PSO variants (GlobalBest, LocalBest, Binary PSO)
> - **NiaPy**: General-purpose optimization library with 100+ algorithms
> - **API**: PySwarms is optimizer-centric; NiaPy is task-centric
> - **Return order**: PySwarms returns (cost, position); NiaPy returns (score, solution)
> - **Use case**: PySwarms for PSO-focused work; NiaPy for algorithm comparison

---

## Program 6: Genetic Algorithm (GA)

### Algorithm Overview
Genetic Algorithm is an evolutionary optimization method inspired by natural selection. Solutions (chromosomes) evolve through selection, crossover, and mutation operators.

### Mathematical Formulation

#### Population Representation

A population $P$ of $N$ individuals (chromosomes):
$$P = \{C_1, C_2, ..., C_N\}$$

Each chromosome $C_i$ contains $L$ genes:
$$C_i = [g_1, g_2, ..., g_L]$$

#### Fitness Function

For the equation $a + 2b + 3c + 4d = 30$:

**Error Calculation:**
$$E(C) = |a + 2b + 3c + 4d - 30|$$

**Fitness (to maximize):**
$$F(C) = \frac{1}{1 + E(C)}$$

Note: $F(C) \rightarrow 1$ as $E(C) \rightarrow 0$ (optimal)

### Detailed Algorithm

```
Algorithm: Genetic Algorithm
Input: Population size N, max generations G,
       crossover rate Pc, mutation rate Pm
Output: Best solution found

1. INITIALIZE:
   - Generate random population P of N individuals
   - Evaluate fitness F(Ci) for all Ci ∈ P
   - best = argmax(F(Ci))

2. FOR generation = 1 TO G:
   
   a. SELECTION:
      - Select N parents using selection method
      - Parents form mating pool M
   
   b. CROSSOVER:
      FOR i = 1 TO N/2:
         - Select pair (P1, P2) from M
         - IF random() < Pc:
            (C1, C2) = crossover(P1, P2)
         - ELSE:
            (C1, C2) = (P1, P2)
         - Add C1, C2 to offspring population O
   
   c. MUTATION:
      FOR each individual Ci in O:
         FOR each gene gj in Ci:
            IF random() < Pm:
               gj = mutate(gj)
   
   d. REPLACEMENT:
      - P = O (generational) OR
      - P = select_best(P ∪ O, N) (elitist)
   
   e. EVALUATE:
      - Compute F(Ci) for all new individuals
      - Update best if improved

3. RETURN best
```

### Selection Methods with Formulas

#### Roulette Wheel Selection

**Selection Probability:**
$$P(C_i) = \frac{F(C_i)}{\sum_{j=1}^{N} F(C_j)}$$

**Cumulative Probability:**
$$Q_i = \sum_{j=1}^{i} P(C_j)$$

**Selection Process:**
1. Generate $r \sim U(0,1)$
2. Select $C_i$ where $Q_{i-1} < r \leq Q_i$

#### Tournament Selection

**Probability of Best Winning (tournament size $k$):**
$$P(\text{best wins}) = 1 - \left(1 - \frac{1}{N}\right)^k$$

**Selection Pressure:** Higher $k$ = more pressure toward fit individuals

#### Rank-Based Selection

**Linear Ranking:**
$$P(C_i) = \frac{2 - s}{N} + \frac{2(rank_i - 1)(s-1)}{N(N-1)}$$

where $s \in [1,2]$ is selection pressure and $rank_i$ is the rank of individual $i$.

### Crossover Operators with Diagrams

#### Single-Point Crossover

```
Parent 1: [A B C | D E F]     Crossover point at position 3
Parent 2: [a b c | d e f]
          ↓
Child 1:  [A B C | d e f]     First part from P1, second from P2
Child 2:  [a b c | D E F]     First part from P2, second from P1
```

**Crossover Point:** $k \sim U\{1, L-1\}$

#### Two-Point Crossover

```
Parent 1: [A | B C D | E F]
Parent 2: [a | b c d | e f]
          ↓
Child 1:  [A | b c d | E F]   Middle segment swapped
Child 2:  [a | B C D | e f]
```

#### Uniform Crossover

For each gene $j$:
$$g_j^{child} = \begin{cases}
g_j^{P1} & \text{if } r_j < 0.5 \\
g_j^{P2} & \text{otherwise}
\end{cases}$$

### Mutation Operators

#### Random Mutation

$$g_j' = \begin{cases}
\text{random}(g_{min}, g_{max}) & \text{if } r < P_m \\
g_j & \text{otherwise}
\end{cases}$$

#### Gaussian Mutation (for real-valued genes)

$$g_j' = g_j + \mathcal{N}(0, \sigma^2)$$

where $\sigma$ controls mutation step size.

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect of Increasing |
|-----------|--------|-------|-------------|----------------------|
| **Population Size** | $N$ | 20-200 | Number of individuals | Better exploration; more evaluations |
| **Crossover Rate** | $P_c$ | 0.6-0.9 | Probability of crossover | More recombination; faster mixing |
| **Mutation Rate** | $P_m$ | 0.01-0.1 | Probability per gene | More diversity; may disrupt good solutions |
| **Tournament Size** | $k$ | 2-7 | Individuals in tournament | Higher selection pressure |
| **Elitism Count** | $e$ | 1-5 | Best individuals preserved | Ensures no loss of best; may reduce diversity |
| **Generations** | $G$ | 50-500 | Number of iterations | More time to evolve |

### Parameter Tuning Guidelines

**Population Size:**
$$N \approx 10 \times L \text{ (rule of thumb)}$$

where $L$ is chromosome length.

**Mutation Rate:**
$$P_m \approx \frac{1}{L} \text{ to } \frac{1}{\sqrt{N \cdot L}}$$

**Crossover Rate:** Usually $P_c \gg P_m$ (crossover is primary search operator)

### VIVA Questions

**Q1: Explain the roulette wheel selection process.**
> A: Each individual gets a slice of a "wheel" proportional to its fitness. A random spin selects an individual. Higher fitness = larger slice = higher selection probability.
> Steps:
> 1. Calculate total fitness
> 2. Compute probability = fitness_i / total_fitness
> 3. Build cumulative probabilities
> 4. Generate random number r ∈ [0,1]
> 5. Select first individual where cumulative_prob ≥ r

**Q2: Why is crossover rate typically higher than mutation rate?**
> A: 
> - Crossover (0.6-0.9): Combines good features from parents; main exploration mechanism
> - Mutation (0.01-0.1): Prevents loss of genetic diversity; too high disrupts good solutions

**Q3: What is elitism and why is it important?**
> A: Elitism preserves the best individual(s) unchanged into the next generation. This ensures the best solution is never lost, improving convergence reliability.

**Q4: How does tournament selection work?**
> A: Randomly select k individuals from the population. The one with the highest fitness wins and becomes a parent. Repeat to get all parents. Advantage: Selection pressure controllable via tournament size k.

**Q5: What is premature convergence and how can it be prevented?**
> A: Premature convergence occurs when the population loses diversity and gets stuck in a local optimum.
> Prevention methods:
> - Increase mutation rate
> - Use larger population
> - Fitness sharing/niching
> - Restart mechanisms
> - Adaptive operators

---

## Program 7: Multi-Objective Genetic Algorithm (MOGA)

### Algorithm Overview
MOGA extends GA to handle multiple conflicting objectives simultaneously. Instead of a single optimal solution, it finds a set of Pareto-optimal solutions.

### Mathematical Formulation

#### Multi-Objective Optimization Problem

$$\min_{\mathbf{x}} \mathbf{F}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), ..., f_m(\mathbf{x})]^T$$

subject to $\mathbf{x} \in \Omega$ (feasible region)

#### Pareto Dominance Definition

Solution $\mathbf{a}$ dominates $\mathbf{b}$ (written $\mathbf{a} \prec \mathbf{b}$) if and only if:

$$\forall i \in \{1,...,m\}: f_i(\mathbf{a}) \leq f_i(\mathbf{b})$$
$$\exists j \in \{1,...,m\}: f_j(\mathbf{a}) < f_j(\mathbf{b})$$

#### Pareto Optimal Set

$$P^* = \{\mathbf{x} \in \Omega : \nexists \mathbf{x}' \in \Omega \text{ such that } \mathbf{x}' \prec \mathbf{x}\}$$

#### Pareto Front

$$PF^* = \{\mathbf{F}(\mathbf{x}) : \mathbf{x} \in P^*\}$$

### NSGA-II Algorithm

```
Algorithm: NSGA-II (Non-dominated Sorting GA II)
Input: Population size N, max generations G
Output: Pareto front approximation

1. INITIALIZE:
   - P0 = random population of size N
   - Evaluate all objectives for P0
   - Assign ranks using non-dominated sorting
   - Calculate crowding distance

2. FOR generation t = 0 TO G-1:
   
   a. CREATE OFFSPRING:
      - Qt = ∅
      - WHILE |Qt| < N:
         * Select parents using binary tournament
           (prefer: lower rank, then higher crowding distance)
         * Apply SBX crossover with probability Pc
         * Apply polynomial mutation with probability Pm
         * Add offspring to Qt
   
   b. COMBINE:
      - Rt = Pt ∪ Qt  (size 2N)
   
   c. NON-DOMINATED SORTING:
      - Classify Rt into fronts F1, F2, F3, ...
      - F1 = non-dominated solutions in Rt
      - F2 = non-dominated in Rt \ F1, etc.
   
   d. SELECT NEXT GENERATION:
      - Pt+1 = ∅, i = 1
      - WHILE |Pt+1| + |Fi| ≤ N:
         * Calculate crowding distance for Fi
         * Pt+1 = Pt+1 ∪ Fi
         * i = i + 1
      - Sort Fi by crowding distance (descending)
      - Add best (N - |Pt+1|) from Fi to Pt+1

3. RETURN F1 of final population (Pareto front)
```

### Key Formulas

#### Crowding Distance Calculation

For solution $i$ in front $F$:

$$CD_i = \sum_{m=1}^{M} \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{max} - f_m^{min}}$$

where:
- Solutions are sorted by each objective
- $f_m^{i+1}$ = objective $m$ value of next neighbor
- $f_m^{i-1}$ = objective $m$ value of previous neighbor
- Boundary solutions get $CD = \infty$

#### SBX Crossover (Simulated Binary Crossover)

For parent values $p_1, p_2$, offspring $c_1, c_2$:

$$c_1 = 0.5[(1 + \beta)p_1 + (1 - \beta)p_2]$$
$$c_2 = 0.5[(1 - \beta)p_1 + (1 + \beta)p_2]$$

where $\beta$ is sampled from:
$$P(\beta) = \begin{cases}
0.5(\eta_c + 1)\beta^{\eta_c} & \text{if } \beta \leq 1 \\
0.5(\eta_c + 1)\frac{1}{\beta^{\eta_c + 2}} & \text{otherwise}
\end{cases}$$

$\eta_c$ = distribution index (typically 15-20)

#### Polynomial Mutation

For parent value $p$ with bounds $[l, u]$:

$$c = p + (u - l) \cdot \delta$$

where:
$$\delta = \begin{cases}
(2r)^{\frac{1}{\eta_m + 1}} - 1 & \text{if } r < 0.5 \\
1 - (2(1-r))^{\frac{1}{\eta_m + 1}} & \text{otherwise}
\end{cases}$$

$\eta_m$ = distribution index (typically 20)

### Problem Example: Renewable Energy Optimization

**Decision Variables:**
- $x_1$ = Solar panel area (0-100 units)
- $x_2$ = Wind turbine count (0-50 units)
- $x_3$ = Battery storage (0-200 units)

**Objectives:**

| Objective | Formula | Goal |
|-----------|---------|------|
| Energy Output | $f_1(\mathbf{x}) = 5x_1 + 8x_2 + 2x_3$ | Maximize |
| Construction Cost | $f_2(\mathbf{x}) = 10x_1 + 50x_2 + 5x_3$ | Minimize |
| Environmental Impact | $f_3(\mathbf{x}) = 0.5x_1 + 2x_2 + 0.1x_3$ | Minimize |

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect |
|-----------|--------|-------|-------------|--------|
| **Population Size** | $N$ | 50-200 | Number of solutions | Larger = better Pareto front coverage |
| **Generations** | $G$ | 50-500 | Number of iterations | More = better convergence |
| **Crossover Rate** | $P_c$ | 0.7-0.9 | SBX application probability | Higher = more recombination |
| **Mutation Rate** | $P_m$ | 0.1-0.3 | Polynomial mutation probability | Higher = more exploration |
| **SBX Index** | $\eta_c$ | 15-20 | Controls offspring spread | Higher = children closer to parents |
| **Mutation Index** | $\eta_m$ | 20-100 | Controls mutation step | Higher = smaller mutations |

### VIVA Questions

**Q1: What is a Pareto-optimal solution?**
> A: A solution is Pareto-optimal (non-dominated) if no other solution exists that is better in all objectives. There is always a trade-off: improving one objective worsens another.

**Q2: Explain crowding distance and its purpose.**
> A: Crowding distance measures the density of solutions around a particular solution in objective space. It's calculated as the sum of distances to neighboring solutions in each objective. Purpose: Maintain diversity by preferring isolated solutions over crowded ones.

**Q3: How does NSGA-II differ from a simple weighted-sum approach?**
> A:
> - Weighted sum: Combines objectives into single value; requires weight specification; finds only one solution
> - NSGA-II: Finds entire Pareto front; no weights needed; user chooses from trade-off solutions afterward

**Q4: What is the Pareto front?**
> A: The set of all Pareto-optimal (non-dominated) solutions. It represents the trade-off surface where improving any objective requires sacrificing another.

**Q5: When would you use MOGA instead of single-objective GA?**
> A: When:
> - Multiple conflicting objectives exist
> - Trade-offs between objectives need to be understood
> - Decision-maker wants to choose from alternatives
> - No clear way to combine objectives into one

---

## Program 8: Adaptive Mutation in GA

### Algorithm Overview
Adaptive mutation dynamically adjusts mutation probability based on solution fitness, balancing exploration and exploitation automatically.

### Mathematical Formulation

#### Fixed vs Adaptive Mutation

**Fixed Mutation:**
$$P_m(C_i) = P_m \quad \forall i$$

**Adaptive Mutation (Fitness-Based):**
$$P_m(C_i) = \begin{cases}
P_m^{high} & \text{if } F(C_i) < \bar{F} \\
P_m^{low} & \text{if } F(C_i) \geq \bar{F}
\end{cases}$$

where $\bar{F} = \frac{1}{N}\sum_{j=1}^N F(C_j)$ is population average fitness.

#### Srinivas & Patnaik Adaptive Scheme

More sophisticated continuous adaptation:

$$P_m(C_i) = \begin{cases}
\frac{k_4(F_{max} - F(C_i))}{F_{max} - \bar{F}} & \text{if } F(C_i) \geq \bar{F} \\
k_3 & \text{if } F(C_i) < \bar{F}
\end{cases}$$

where:
- $k_3, k_4 \in (0, 1]$ are constants
- $F_{max}$ = maximum fitness in population
- Higher fitness → lower mutation (protection)
- Lower fitness → higher mutation (exploration)

#### Generation-Based Adaptation

$$P_m(t) = P_m^{init} \cdot \exp\left(-\frac{t}{\tau}\right) + P_m^{final}$$

or

$$P_m(t) = P_m^{max} - \frac{(P_m^{max} - P_m^{min}) \cdot t}{t_{max}}$$

### Test Function: Rastrigin

$$f(\mathbf{x}) = 10n + \sum_{i=1}^{n} \left[x_i^2 - 10\cos(2\pi x_i)\right]$$

**Properties:**
- Global minimum: $f(\mathbf{0}) = 0$
- Search domain: $x_i \in [-5.12, 5.12]$
- Number of local minima: $\approx 10^n$ (highly multimodal)
- Tests both exploration and exploitation

### Detailed Algorithm

```
Algorithm: GA with Adaptive Mutation
Input: N, G, Pc, Pm_high, Pm_low
Output: Best solution

1. INITIALIZE population P

2. FOR generation = 1 TO G:
   
   a. EVALUATE fitness F(Ci) for all Ci
   
   b. COMPUTE average fitness:
      F_avg = mean(F(C1), F(C2), ..., F(CN))
   
   c. SELECTION and CROSSOVER (standard)
   
   d. ADAPTIVE MUTATION:
      FOR each individual Ci:
         IF F(Ci) < F_avg:
            Pm = Pm_high  // Poor solution: explore more
         ELSE:
            Pm = Pm_low   // Good solution: refine
         
         Apply mutation with probability Pm
   
   e. UPDATE population

3. RETURN best solution
```

### PyGAD Implementation

```python
# Adaptive mutation configuration
mutation_type = "adaptive"
mutation_probability = [0.25, 0.01]  # [high, low]
```

**Interpretation:**
- Solutions with $F < F_{avg}$: 25% mutation rate (high exploration)
- Solutions with $F \geq F_{avg}$: 1% mutation rate (fine-tuning)

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect |
|-----------|--------|-------|-------------|--------|
| **High Mutation Rate** | $P_m^{high}$ | 0.15-0.35 | For below-average solutions | Higher = more exploration for poor solutions |
| **Low Mutation Rate** | $P_m^{low}$ | 0.005-0.05 | For above-average solutions | Lower = better preservation of good solutions |
| **Adaptation Threshold** | $\bar{F}$ | avg fitness | Fitness cutoff for adaptation | Can use median or percentile instead |

### Comparison: Fixed vs Adaptive

| Aspect | Fixed Mutation | Adaptive Mutation |
|--------|---------------|-------------------|
| **Formula** | $P_m = c$ (constant) | $P_m = f(fitness)$ |
| **Early Generations** | Same rate | High rate (more diverse) |
| **Late Generations** | Same rate | Low rate (refinement) |
| **Good Solutions** | May disrupt | Protected (low mutation) |
| **Poor Solutions** | Under-explored | Heavily mutated |
| **Parameter Tuning** | Manual selection | Self-adjusting |
| **Robustness** | Problem-dependent | More robust across problems |

### Benefits Analysis

1. **Automatic Exploration-Exploitation Balance:**
$$\text{Exploration} \propto P_m^{high} \times (\text{low-fitness count})$$
$$\text{Exploitation} \propto (1 - P_m^{low}) \times (\text{high-fitness count})$$

2. **Diversity Preservation:**
   - Poor solutions contribute new genetic material
   - Good solutions maintain successful traits

3. **Convergence Speed:**
   - Early: high diversity prevents premature convergence
   - Late: low mutation enables fine-tuning

### VIVA Questions

**Q1: What is the main idea behind adaptive mutation?**
> A: Apply high mutation to low-fitness individuals (they need exploration) and low mutation to high-fitness individuals (they're already good, just need fine-tuning). This automatically balances exploration and exploitation.

**Q2: How does PyGAD implement adaptive mutation?**
> A: PyGAD uses fitness-based selection between two mutation rates:
> - If solution fitness < population average → use higher rate
> - If solution fitness ≥ population average → use lower rate

**Q3: What are other adaptive strategies in GA?**
> A:
> - **Adaptive crossover rate**: Vary based on population diversity
> - **Self-adaptive parameters**: Encode parameters in chromosome
> - **Population-based**: Adjust based on improvement rate
> - **Fitness-proportional**: Mutation inversely proportional to fitness

**Q4: Why is the Rastrigin function used for testing?**
> A: Rastrigin has many local minima (highly multimodal), making it challenging. If an algorithm works on Rastrigin, it likely handles complex landscapes well. It tests both exploration (finding the right valley) and exploitation (finding the exact minimum).

**Q5: What are the benefits of adaptive mutation over fixed mutation?**
> A:
> - Reduces need for parameter tuning
> - Maintains diversity in early generations
> - Enables fine-tuning in later generations
> - Often produces better final solutions
> - More robust across different problems

---

## Program 9: Binary Genetic Algorithm

### Algorithm Overview
Binary GA represents solutions as binary strings (0s and 1s), requiring encoding/decoding between binary and real values.

### Mathematical Formulation

#### Binary Encoding

For a real variable $x \in [x_{min}, x_{max}]$ encoded with $n$ bits:

**Binary String:**
$$B = [b_{n-1}, b_{n-2}, ..., b_1, b_0]$$

**Decimal Conversion:**
$$D = \sum_{i=0}^{n-1} b_i \cdot 2^i$$

**Real Value Decoding:**
$$x = x_{min} + D \cdot \frac{x_{max} - x_{min}}{2^n - 1}$$

#### Precision Analysis

With $n$ bits, the precision (smallest representable change) is:
$$\epsilon = \frac{x_{max} - x_{min}}{2^n - 1}$$

**Required bits for desired precision $\epsilon$:**
$$n = \lceil \log_2\left(\frac{x_{max} - x_{min}}{\epsilon} + 1\right) \rceil$$

### Problem Definition

**Objective Function:**
$$f(x, y, z) = 2x + 3y + 10z - 3.6$$

**Goal:** Minimize $|f(x, y, z)|$

**Encoding Scheme:**
- 10 bits per variable
- Total chromosome length: 30 bits
- Variable range: $[0, 10]$

```
Chromosome Structure (30 bits):
[b29 b28 ... b20 | b19 b18 ... b10 | b9 b8 ... b0]
      x bits           y bits           z bits
```

### Example Decoding

**Given chromosome:**
```
1001010011 | 1101110110 | 1111001101
    x           y            z
```

**Step 1: Binary to Decimal**
$$x_{bin} = 1001010011_2 = 595_{10}$$
$$y_{bin} = 1101110110_2 = 886_{10}$$
$$z_{bin} = 1111001101_2 = 973_{10}$$

**Step 2: Decimal to Real**
$$x = 0 + \frac{595}{1023} \times 10 = 5.816$$
$$y = 0 + \frac{886}{1023} \times 10 = 8.661$$
$$z = 0 + \frac{973}{1023} \times 10 = 9.511$$

**Step 3: Objective Value**
$$f = 2(5.816) + 3(8.661) + 10(9.511) - 3.6 = 128.73$$

### Detailed Algorithm

```
Algorithm: Binary Genetic Algorithm
Input: Chromosome length L, population N, generations G
       Crossover rate Pc, mutation rate Pm per bit
Output: Best solution

1. INITIALIZE:
   FOR i = 1 TO N:
      Ci = random binary string of length L
      Fi = evaluate(decode(Ci))

2. FOR generation = 1 TO G:
   
   a. SELECTION:
      Select N parents using tournament/roulette
   
   b. CROSSOVER:
      FOR each pair (P1, P2):
         IF random() < Pc:
            k = random(1, L-1)  // crossover point
            C1 = P1[1:k] + P2[k+1:L]
            C2 = P2[1:k] + P1[k+1:L]
   
   c. BIT-FLIP MUTATION:
      FOR each chromosome C:
         FOR each bit bi in C:
            IF random() < Pm:
               bi = 1 - bi  // flip: 0→1 or 1→0
   
   d. DECODE AND EVALUATE:
      FOR each chromosome:
         (x, y, z) = decode(C)
         F = fitness(f(x, y, z))
   
   e. UPDATE best if improved

3. RETURN decode(best_chromosome)
```

### Genetic Operators

#### Single-Point Crossover

**Before:**
```
P1: 1001010011|1101110110 1111001101
P2: 0110101100|0010001001 0000110010
              ↑
         Crossover point k=10
```

**After:**
```
C1: 1001010011|0010001001 0000110010
C2: 0110101100|1101110110 1111001101
```

#### Bit-Flip Mutation

**Before:** `1001010011`
**Mutation at position 4:** `1001110011`

**Probability of at least one mutation in L bits:**
$$P(\text{at least one flip}) = 1 - (1 - P_m)^L$$

### Fitness Function Design

**For Minimization Problem:**

$$\text{Fitness}(C) = \frac{1}{1 + |f(x,y,z)|}$$

**Properties:**
- $f(x,y,z) = 0 \Rightarrow \text{Fitness} = 1$ (optimal)
- $|f(x,y,z)| \rightarrow \infty \Rightarrow \text{Fitness} \rightarrow 0$

### Detailed Parameter Analysis

| Parameter | Symbol | Range | Description | Effect |
|-----------|--------|-------|-------------|--------|
| **Bits per Variable** | $n$ | 8-32 | Precision of encoding | More bits = higher precision, larger search space |
| **Population Size** | $N$ | 20-200 | Number of chromosomes | Larger = better coverage |
| **Crossover Rate** | $P_c$ | 0.6-0.9 | Probability of crossover | Higher = more mixing |
| **Mutation Rate** | $P_m$ | 0.001-0.05 | Per-bit flip probability | Higher = more exploration |
| **Variable Range** | $[x_{min}, x_{max}]$ | Problem-specific | Search space bounds | Must contain optimum |

### Precision Table

| Bits (n) | Discrete Values | Precision for [0,10] |
|----------|-----------------|---------------------|
| 8 | 256 | 0.0392 |
| 10 | 1024 | 0.0098 |
| 12 | 4096 | 0.0024 |
| 16 | 65536 | 0.00015 |
| 20 | 1048576 | 0.0000095 |

### Binary vs Real-Coded GA Comparison

| Aspect | Binary GA | Real-Coded GA |
|--------|-----------|---------------|
| **Representation** | Bit string | Real numbers |
| **Crossover** | Bit-level (simple) | Value-level (SBX) |
| **Mutation** | Bit-flip | Gaussian/Polynomial |
| **Precision** | Limited by bit count | Machine precision |
| **Hamming Cliff** | Can occur | Not applicable |
| **Implementation** | Simple | More complex operators |

### VIVA Questions

**Q1: Why use binary encoding?**
> A: Benefits:
> - Simple genetic operators (crossover = cut & swap bits)
> - Natural bit-flip mutation
> - Works with any continuous variable by adjusting bit count
> - Hardware-friendly operations

**Q2: How does the number of bits affect solution precision?**
> A: More bits = higher precision. With n bits:
> - Precision = (xmax - xmin) / (2^n - 1)
> - For 10 bits, range [0, 10]: precision ≈ 0.0098

**Q3: What is the Gray code and why is it useful?**
> A: Gray code is a binary encoding where adjacent values differ by only one bit. This reduces the "Hamming cliff" problem where small real-value changes might require many bit changes in standard binary.

**Q4: What happens during single-point crossover on a 30-bit chromosome?**
> A: A random cut point (1-29) is chosen. Bits before the cut come from parent 1, bits after from parent 2 (and vice versa for the second child). If cut = 15, the x variable stays intact but y and z are mixed.

**Q5: How does bit-flip mutation work?**
> A: For each bit in the chromosome, with probability p_m (typically 0.01):
> - If bit is 0 → flip to 1
> - If bit is 1 → flip to 0
> This introduces small random changes to explore nearby solutions.

**Q6: What is the difference between binary GA and real-coded GA?**
> A:
> - **Binary GA**: Solutions as bit strings; requires encoding/decoding; simple operators
> - **Real-coded GA**: Solutions as real numbers directly; specialized operators (SBX crossover, polynomial mutation); no precision limitation

---

## Program 10: Continuous (Real-Coded) Genetic Algorithm

### Algorithm Overview
Continuous GA (also called Real-Coded GA) represents solutions directly as real numbers instead of binary strings. This eliminates the need for encoding/decoding and provides better precision for continuous optimization problems.

### Mathematical Formulation

#### Chromosome Representation

For a $d$-dimensional optimization problem:
$$\mathbf{x} = [x_1, x_2, ..., x_d] \in \mathbb{R}^d$$

Each $x_i$ is directly stored as a floating-point number.

#### Objective Function Used

$$f(x, y) = x \cdot \sin(4x) + 1.1 \cdot y \cdot \sin(2y)$$

**Properties:**
- Multimodal function (multiple local minima)
- Non-separable (variables interact)
- Search domain: $[0, 10]$ for both $x$ and $y$

### Genetic Operators for Continuous GA

#### 1. Initialization

$$x_i \sim U(x_{min}, x_{max})$$

Random uniform sampling within bounds for each dimension.

#### 2. Tournament Selection

For tournament size $k$:
1. Randomly select $k$ individuals from population
2. Return the one with best (lowest) fitness
3. Repeat to select all parents

**Selection Pressure:** Larger $k$ = stronger pressure toward best individuals

#### 3. Arithmetic Crossover

Given parents $\mathbf{p}_1$ and $\mathbf{p}_2$, with blending parameter $\alpha$:

$$\mathbf{c}_1 = \alpha \cdot \mathbf{p}_1 + (1 - \alpha) \cdot \mathbf{p}_2$$
$$\mathbf{c}_2 = \alpha \cdot \mathbf{p}_2 + (1 - \alpha) \cdot \mathbf{p}_1$$

**For $\alpha = 0.5$ (Uniform Arithmetic):**
$$\mathbf{c}_1 = \mathbf{c}_2 = \frac{\mathbf{p}_1 + \mathbf{p}_2}{2}$$

**Geometric Interpretation:** Children lie on the line segment between parents.

#### 4. Gaussian Mutation

With mutation rate $p_m$ and standard deviation $\sigma$:

$$x'_i = \begin{cases}
x_i + \mathcal{N}(0, \sigma) & \text{if } r < p_m \\
x_i & \text{otherwise}
\end{cases}$$

Where $\mathcal{N}(0, \sigma)$ is Gaussian noise with mean 0 and std $\sigma$.

**After mutation, apply bounds:**
$$x'_i = \text{clip}(x'_i, x_{min}, x_{max})$$

### Detailed Algorithm

```
Algorithm: Continuous Genetic Algorithm
Input: Population size N, generations G
       Mutation rate Pm, mutation sigma σ
       Bounds [x_min, x_max], dimensions d
Output: Best solution found

1. INITIALIZE:
   Population = N random individuals in [x_min, x_max]^d
   Evaluate fitness for all individuals
   Track global best

2. FOR generation = 1 TO G:
   
   a. EVALUATE:
      Compute cost for each individual
      Update best/avg statistics
   
   b. TOURNAMENT SELECTION:
      FOR i = 1 TO N:
         Select k random candidates
         Choose best as parent[i]
   
   c. CROSSOVER:
      FOR i = 1 TO N STEP 2:
         p1, p2 = parents[i], parents[i+1]
         c1 = α·p1 + (1-α)·p2
         c2 = α·p2 + (1-α)·p1
         Add c1, c2 to next generation
   
   d. GAUSSIAN MUTATION:
      FOR each child c:
         IF random() < Pm:
            c = c + N(0, σ)
            c = clip(c, x_min, x_max)
   
   e. Replace population with children

3. RETURN best solution, best cost
```

### Detailed Parameter Analysis

| Parameter | Symbol | Value Used | Range | Effect of Increasing |
|-----------|--------|------------|-------|----------------------|
| **Population Size** | $N$ | 100 | 20-500 | Better exploration; slower per generation |
| **Generations** | $G$ | 50 | 20-1000 | More time to converge; diminishing returns |
| **Mutation Rate** | $p_m$ | 0.1 | 0.01-0.3 | More exploration; may disrupt good solutions |
| **Mutation Sigma** | $\sigma$ | 0.5 | 0.1-2.0 | Larger steps; may jump over optima |
| **Tournament Size** | $k$ | 3 | 2-10 | Stronger selection pressure |
| **Crossover Alpha** | $\alpha$ | 0.5 | 0-1 | Controls offspring position between parents |

### Crossover Variants for Continuous GA

| Type | Formula | Characteristics |
|------|---------|----------------|
| **Arithmetic** | $c = \alpha p_1 + (1-\alpha) p_2$ | Simple, children between parents |
| **BLX-α** | $c \sim U(x_{min}' - \alpha I, x_{max}' + \alpha I)$ | Extends beyond parents |
| **SBX** | Simulated Binary Crossover | Mimics single-point crossover behavior |
| **UNDX** | Unimodal Normal Distribution | Uses 3 parents, normal distribution |

### Mutation Variants for Continuous GA

| Type | Formula | Characteristics |
|------|---------|----------------|
| **Gaussian** | $x' = x + \mathcal{N}(0, \sigma)$ | Standard, symmetric |
| **Uniform** | $x' \sim U(x - \delta, x + \delta)$ | Bounded perturbation |
| **Polynomial** | Non-linear distribution | Higher probability near parent |
| **Self-Adaptive** | $\sigma$ evolves with solution | Automatically adjusts step size |

### Comparison: Continuous GA vs Binary GA

| Aspect | Continuous GA | Binary GA |
|--------|---------------|----------|
| **Representation** | Real numbers | Bit strings |
| **Precision** | Machine precision (~15 digits) | Limited by bit count |
| **Encoding** | None needed | Binary ↔ Real required |
| **Crossover** | Arithmetic/BLX/SBX | Single-point/Two-point |
| **Mutation** | Gaussian/Polynomial | Bit-flip |
| **Hamming Cliff** | Not applicable | Can be problematic |
| **Implementation** | More complex operators | Simpler bit operations |
| **Best for** | Continuous optimization | Discrete/combinatorial |

### Visualization Components

1. **Convergence Plot**: Shows best and average cost over generations
2. **3D Landscape**: Visualizes the objective function surface
3. **Optimal Point**: Marks the found solution on the landscape

### VIVA Questions

**Q1: What is a Continuous/Real-Coded GA?**
> A: A Genetic Algorithm where solutions are represented directly as vectors of real numbers instead of binary strings. This eliminates encoding/decoding overhead and provides machine-level precision for continuous optimization problems.

**Q2: Explain Arithmetic Crossover.**
> A: Arithmetic crossover creates children as weighted averages of parents:
> - $c_1 = \alpha \cdot p_1 + (1-\alpha) \cdot p_2$
> - $c_2 = \alpha \cdot p_2 + (1-\alpha) \cdot p_1$
> - Children always lie on the line segment between parents
> - With $\alpha = 0.5$, both children are at the midpoint

**Q3: How does Gaussian Mutation work?**
> A: Gaussian mutation adds random noise sampled from a normal distribution:
> - $x' = x + \mathcal{N}(0, \sigma)$
> - $\sigma$ controls the mutation strength (step size)
> - Larger $\sigma$ = larger perturbations = more exploration
> - After mutation, values are clipped to stay within bounds

**Q4: What is Tournament Selection?**
> A: A selection method where:
> 1. Randomly pick $k$ individuals from the population
> 2. Select the one with the best fitness
> 3. Repeat until enough parents are selected
> - Tournament size $k$ controls selection pressure
> - $k=2$: Weak pressure; $k=10$: Strong pressure toward elites

**Q5: Why use Continuous GA over Binary GA?**
> A: Continuous GA is preferred when:
> - Variables are naturally continuous (not discrete)
> - High precision is needed (beyond what reasonable bit counts provide)
> - You want to avoid Hamming cliff problems
> - The problem has meaningful distance metrics in real space

**Q6: What is the effect of mutation sigma (σ)?**
> A:
> - Small $\sigma$ (0.1): Fine-tuning, local search, exploitation
> - Large $\sigma$ (2.0): Large jumps, global exploration
> - Often beneficial to start with large $\sigma$ and decrease over generations (like simulated annealing)

**Q7: What makes the objective function $f(x,y) = x\sin(4x) + 1.1y\sin(2y)$ challenging?**
> A:
> - **Multimodal**: Multiple local minima due to sine functions
> - **Non-separable**: The landscape varies across both dimensions
> - **Oscillating**: Rapid changes require careful search
> - Tests the algorithm's ability to escape local optima

**Q8: Why clip values after mutation?**
> A: Gaussian mutation can produce values outside the valid search bounds. Clipping ensures:
> - All solutions remain feasible
> - The algorithm doesn't waste time evaluating invalid solutions
> - Bounds constraints are satisfied

---

## General VIVA Questions (Applicable to All Algorithms)

### Meta-Heuristics Fundamentals

**Q: What is a meta-heuristic?**
> A: A high-level problem-independent algorithmic framework that provides strategies for developing heuristic optimization algorithms. "Meta" means "beyond" - they go beyond simple heuristics by guiding the search process.

**Q: What is the difference between exploration and exploitation?**
> A:
> - **Exploration**: Searching new, unvisited regions of the solution space
> - **Exploitation**: Focusing search around known good solutions
> - Good algorithms balance both

**Q: What is a local optimum vs global optimum?**
> A:
> - **Local optimum**: Best solution in a neighborhood (no better nearby solutions)
> - **Global optimum**: Best solution in the entire search space
> - Meta-heuristics try to escape local optima to find the global optimum

**Q: What is the No Free Lunch Theorem?**
> A: No single optimization algorithm is best for all problems. An algorithm that performs well on one class of problems will perform poorly on another. Algorithm selection should be problem-specific.

### Comparison Questions

**Q: Compare GA, PSO, and ACO.**
> A:
> | Aspect | GA | PSO | ACO |
> |--------|-----|-----|-----|
> | Inspiration | Evolution | Bird flocking | Ant foraging |
> | Solution | Chromosome | Particle position | Ant path |
> | Operators | Selection, crossover, mutation | Velocity update | Pheromone update |
> | Best for | Discrete & continuous | Continuous | Combinatorial |

**Q: When would you choose Simulated Annealing over Tabu Search?**
> A:
> - SA: When problem has continuous variables; simpler to implement; no need for move history
> - Tabu: When problem is combinatorial; when cycling is a concern; when memory can guide search effectively

**Q: What are population-based vs trajectory-based methods?**
> A:
> - **Population-based** (GA, PSO, ACO): Maintain multiple solutions; better exploration; more function evaluations
> - **Trajectory-based** (SA, Tabu): Follow single solution path; less memory; focused search

---

## Quick Reference: Algorithm Selection Guide

| Problem Type | Recommended Algorithms |
|--------------|----------------------|
| Continuous optimization | PSO, GA, SA |
| Combinatorial (TSP, scheduling) | ACO, GA, SA, Tabu |
| Multi-objective | NSGA-II, MOGA |
| Binary decisions | Binary GA, Binary PSO |
| Constrained optimization | GA with penalty, PSO with constraint handling |
| Large-scale problems | Distributed GA, Island Model |
| Dynamic problems | Adaptive algorithms, Memory-based approaches |

---

*This document covers all MHO Lab programs with detailed explanations and comprehensive VIVA preparation material.*
