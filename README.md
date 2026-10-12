# High-Performance MCCFR Poker Solver in Julia

![Julia](https://img.shields.io/badge/Language-Julia_1.10+-9558B2.svg)
![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

## The Problem & The Solution
The genesis of this project wasn't a textbook exercise, but a late-night poker game with friends. After second-guessing a marginal decision in a complex hand, I realized that human intuition is fundamentally inadequate when navigating asymmetric, incomplete-information games. I didn't just want to know if my play was "good"—I wanted to prove the mathematical truth behind the optimal action.

This curiosity led me to engineer a Game Theory Optimal (GTO) solver from scratch. Targeting the computationally massive 6-Max No-Limit Texas Hold'em (100bb) landscape, I developed a high-performance **Monte Carlo Counterfactual Regret Minimization (MCCFR)** engine in Julia. What started as a personal tool to analyze home-game spots evolved into a robust stochastic optimization framework capable of traversing massive combinatorial game trees.

## Mathematical Framework

Solving a 6-Max 100bb game tree analytically is computationally intractable due to the state-space explosion. To achieve algorithmic convergence, this engine implements two core mathematical pillars:

### 1. Dimensionality Reduction (Information Abstraction)
To prevent memory exhaustion, continuous and semi-continuous game states are clustered into discrete mathematical buckets. Hands and board textures are mapped using an isomorphic hashing function (`abstract_hand`), reducing millions of distinct node permutations into a manageable matrix without losing the statistical variance required for the Nash Equilibrium approximation.

### 2. Regret Matching Algorithm
The engine relies on minimizing the counterfactual regret over millions of stochastic iterations. For any given action $a$ in an information set $I$, the strategy is updated proportionally to the positive regret accumulated:

$$\sigma_{T+1}^{i}(I, a) = \frac{R_{+}^{T}(I, a)}{\sum_{a \in A(I)} R_{+}^{T}(I, a)}$$

*(Where $R_{+}$ is the positive regret of not having taken action $a$ in the past, driving the system naturally toward a non-exploitable state).*

## Performance & Computational Optimization

The initial prototype was heavily bottlenecked by RAM allocation and Garbage Collection (GC) overhead. The transition to Julia enabled low-level optimizations that accelerated processing speed by a factor of 7x:

* **Static Memory Allocation (`StaticArrays`):** Dynamic vectors were entirely replaced by stack-allocated static structures (`SVector`, `MVector`). This bypassed the heap allocation completely, reducing GC overhead from >15% to <1%.
* **Integer Hashing over String Allocation:** A common pitfall in game solvers is using string concatenation to track the game history (e.g., `"C3_Bucket8_H241"`). To avoid this, the engine uses purely mathematical base-10 integer hashing stored in a `UInt64`. Actions are encoded as discrete digits (`1=Fold`, `2=Call`, `3=r33`, `4=r50`, `5=r75`, `6=All-In`). When an action is taken, the history updates mathematically (`history = (history * 10) + action_id`), achieving zero-allocation node lookups and completely bypassing the memory heap.
* **JIT Compilation:** Leveraging Julia's Just-In-Time compiler allowed the deep recursive tree traversals to execute at near-C speeds (~900 iterations/second on consumer hardware).

## Scenario Configuration: Spot Isolation

Professional solvers do not compute 6 active players simultaneously at 100bb deep due to the infinite branching factor. Instead, they isolate specific "spots".

By default, the engine is configured to evaluate a **Limped Pot: Small Blind (SB) vs Big Blind (BB)**. You can easily modify this scenario inside the `train(iterations::Int)` function by toggling the `active_players` boolean vector:

```julia
# Inside the GameState initialization in train():

# Default: SB vs BB (Players 1 and 2 active)
SVector{6, Bool}(true, true, false, false, false, false)

# Scenario: Button vs Big Blind (Players 6 and 2 active)
SVector{6, Bool}(false, true, false, false, false, true)

# Scenario: 3-Way Pot (SB vs BB vs Button)
SVector{6, Bool}(true, true, false, false, false, true)
```

### Action Abstraction (Tree Pruning)
In No-Limit Texas Hold'em, the betting space is theoretically continuous. To render the game tree computable, the engine applies rigid **Action Abstraction**. The continuous bet sizing is discretized into strategically significant fractions of the pot. 

The engine natively supports the following offensive actions:
* `r33`: Raise 33% of the pot.
* `r50`: Raise 50% of the pot.
* `r75`: Raise 75% of the pot.
* `rAI`: All-In.

**Branching Factor Control:** To prevent exponential explosion during deep-stack simulations (100bb), the `get_legal_actions()` function allows strict tree pruning. By default, the engine is constrained to a maximum of 2 raises per street, and for rapid convergence, sizes can be toggled on or off (e.g., restricting the AI to only `r50` and `rAI`).

## Benchmarks & Execution Time

Because the engine avoids RAM fragmentation, it achieves extremely high throughput on consumer-grade hardware:

* **Processing Speed**: ~890 - 900 iterations/second.
* **Convergence Target**: A highly accurate Nash Equilibrium approximation for a 100bb Heads-Up spot requires roughly 100 million iterations.
* **Estimated Compute Time**: ~31.2 hours of continuous CPU execution to generate a fully solved production module.

_(Note: For testing and validation purposes, the execution block at the bottom of solver.jl is currently set to a rapid 500-iteration benchmark to instantly demonstrate the SQLite export pipeline)._

## System Architecture (Data Pipeline)

The engine does not just compute numbers in a vacuum; it is designed as a backend pipeline for mobile applications. 
1. **Simulation:** The MCCFR engine explores the pruned game tree.
2. **Compression:** Unlikely nodes (noise) are pruned mathematically. Strategy arrays are normalized and converted into discrete probabilities.
3. **Export:** The data is serialized into highly compressed JSON dictionaries and injected into a local **SQLite database** (`gto_100bb.sqlite`), enabling zero-latency read access for Android/iOS frontend interfaces.

## Quick Start

To run the benchmark and generate the SQLite database locally:

```julia
# 1. Install dependencies in the Julia REPL:
using Pkg
Pkg.add(["StaticArrays", "SQLite", "JSON3"])

# 2. Run the solver:
include("solver.jl")

