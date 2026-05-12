# Optimal Market Making and Hedging with Limit Order Book Frictions

A dynamic programming framework for optimal option hedging and market making under realistic limit order book frictions using an HJB-QVI approach.

## Overview

This project studies the problem of a trader who is short a European call option and must dynamically hedge the position using the underlying stock. Unlike the classical Black-Scholes delta-hedging framework, the model incorporates realistic market microstructure frictions such as bid-ask spreads, maker/taker fees, stochastic passive fills, temporary market impact, and inventory constraints.

The central question is:

> How should a trader optimally balance spread capture from passive liquidity provision against the need to control option hedge risk?

To answer this, the project formulates the problem as a stochastic control problem and solves it using a Hamilton-Jacobi-Bellman Quasi-Variational Inequality (HJB-QVI). The resulting policy decides when to quote passively in the limit order book and when to aggressively rebalance using market orders.

## Motivation

Classical delta hedging assumes frictionless markets, continuous trading, and execution at a single mid-price. In practice, hedging an option book is costly because traders face:

- bid-ask spreads,
- exchange fees and rebates,
- uncertain execution of limit orders,
- market impact from aggressive trades,
- inventory limits and hedge-tracking risk.

This project connects option hedging theory with the practical challenges faced by electronic market makers and derivatives traders operating in a limit order book environment.

## Methodology

### Stock Price Dynamics

The underlying stock mid-price is modeled using geometric Brownian motion:

```math
dS_t = \sigma S_t dW_t
```

The model focuses on a short-horizon setting and assumes zero drift.

### State Variables

The full state of the control problem is:

```math
(t, S_t, q_t, X_t)
```

where:

- `t` is time,
- `S_t` is the stock mid-price,
- `q_t` is inventory in the underlying stock,
- `X_t` is cash.

The trader is short a European call option and aims to maximize terminal liquidated wealth after accounting for option payoff and hedge-tracking penalty.

### Objective Function

The value function maximizes expected terminal wealth net of option payoff and cumulative hedge-tracking error:

```math
V(t,S,q,X) = \sup_{\alpha \in A} \mathbb{E}\left[X_T + q_T S_T - (S_T-K)^+ - \eta \int_t^T (q_u - \Delta(u,S_u))^2du\right]
```

The penalty term discourages deviations from the Black-Scholes delta hedge target.

### Passive Limit Orders

The trader can post bid and ask limit orders at offsets:

```math
\delta_b, \delta_a > 0
```

Passive order fills are modeled with exponential Poisson intensities:

```math
\lambda_b(\delta_b) = \Lambda_b e^{-k_b\delta_b}, \quad
\lambda_a(\delta_a) = \Lambda_a e^{-k_a\delta_a}
```

This captures the trade-off that more aggressive quotes are more likely to fill but earn less spread.

### Market Order Intervention

When hedge mismatch becomes too large, the trader can use an aggressive market order of size `xi`. The model includes spread cost, taker fees, and nonlinear temporary impact:

```math
X^+ = X - \xi S - |\xi|\left(\frac{s}{2} + \epsilon_{taker}\right) - \kappa |\xi|^\beta
```

This makes large rebalancing trades disproportionately expensive.

## HJB-QVI Formulation

The control problem combines two decisions:

1. **Continuation:** remain in the passive quoting region and optimize bid/ask quote offsets.
2. **Intervention:** use a market order to immediately rebalance inventory.

The resulting HJB-QVI balances passive market making against impulse rebalancing.

A key simplification is the cash-dimension reduction:

```math
V(t,S,q,X) = X + h(t,S,q)
```

Because cash enters the objective linearly and future control opportunities do not depend on current cash, the problem is reduced from four dimensions to three.

## Numerical Implementation

The HJB-QVI is solved using a backward dynamic programming algorithm.

Core implementation components include:

- finite-difference discretization,
- Crank-Nicolson time stepping,
- log-price transformation to stabilize the GBM diffusion term,
- fixed-point iteration for the QVI continuation/intervention update,
- grid search over bid and ask quote offsets,
- impulse optimization over market order sizes,
- offline policy storage for simulation.

The log-price transformation is used because under GBM the diffusion coefficient grows with stock price. In log space, the diffusion coefficient becomes constant, making the finite-difference stencil more stable and reusable across the grid.

## Strategy Benchmarks

The optimal HJB-QVI policy is benchmarked against two simpler strategies:

### 1. Pure Delta Hedging

Continuously rebalances aggressively toward the Black-Scholes delta target using market orders.

- Lowest hedge-tracking error.
- Highest execution cost.
- Negative expected terminal wealth in the simulations due to spread, taker fees, and market impact.

### 2. Pure Passive Market Making

Uses only passive limit orders and does not aggressively rebalance.

- Earns spread and avoids taker costs.
- Produces high terminal wealth in the stylized setting.
- Has materially worse hedge quality and larger tracking error.

### 3. Optimal HJB-QVI Strategy

Combines passive quoting with selective market-order intervention.

- Maintains positive expected terminal wealth.
- Controls hedge mismatch better than passive market making.
- Avoids the excessive transaction costs of pure delta hedging.

## Results

The strategies were evaluated for short-call books of size:

```text
N = 1, 100, 1000
```

Metrics were reported per option where appropriate. Cumulative tracking error was normalized by `N^2`.

### Cross-Book Comparison

| Book Size | Strategy | E[W_T]/opt | Std./opt | Signal/Noise | TrackErr/N^2 | Maker/opt | Taker/opt |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HJB-QVI | 3.487 | 1.072 | 3.254 | 0.2280 | 0.0259 | 0.0020 |
| 1 | Pure Delta Hedge | -0.434 | 0.165 | -2.640 | 0.0020 | 0.0000 | 0.3978 |
| 1 | Pure Passive MM | 3.488 | 1.304 | 2.674 | 0.4102 | 0.0265 | 0.0000 |
| 100 | HJB-QVI | 2.881 | 0.425 | 6.775 | 0.0160 | 0.0223 | 0.0936 |
| 100 | Pure Delta Hedge | -0.702 | 0.253 | -2.774 | 0.0020 | 0.0000 | 0.3978 |
| 100 | Pure Passive MM | 3.488 | 1.304 | 2.674 | 0.4102 | 0.0265 | 0.0000 |
| 1000 | HJB-QVI | 2.351 | 0.424 | 5.540 | 0.0095 | 0.0197 | 0.1462 |
| 1000 | Pure Delta Hedge | -1.344 | 0.471 | -2.854 | 0.0020 | 0.0000 | 0.3978 |
| 1000 | Pure Passive MM | 3.488 | 1.304 | 2.674 | 0.4102 | 0.0265 | 0.0000 |

### Large-Book Case: N = 1000

| Strategy | E[W_T]/opt | Std./opt | Signal/Noise | TrackErr/N^2 | Maker/opt | Taker/opt | MaxDD/opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| HJB-QVI | 2.351 | 0.424 | 5.540 | 0.0095 | 0.0197 | 0.1462 | 0.1881 |
| Pure Delta Hedge | -1.344 | 0.471 | -2.854 | 0.0020 | 0.0000 | 0.3978 | 1.3475 |
| Pure Passive MM | 3.488 | 1.304 | 2.674 | 0.4102 | 0.0265 | 0.0000 | 0.7959 |

For the large-book case, the HJB-QVI strategy achieved the best risk-adjusted compromise: it remained profitable, maintained low normalized tracking error, and produced the lowest mean maximum drawdown per option.

## Key Findings

- Pure delta hedging tracks the option delta most closely but is economically unattractive once execution costs are included.
- Pure passive market making earns spread but accumulates large hedge-tracking error.
- The HJB-QVI policy creates a continuation region around the delta target where passive quoting is preferred.
- When inventory mismatch becomes sufficiently large, the impulse operator activates and the strategy uses market orders to rebalance.
- The optimal policy provides a practical balance between liquidity provision, hedge risk, and transaction costs.

## Numerical Convergence

The solver used:

| Book Size | Time Steps | Stock Grid Points | Inventory Grid Points | Quote Grid Points | Typical QVI Residual |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 121 | 41 | 61 | 1e-8 |
| 100 | 200 | 121 | 41 | 61 | 1e-8 |
| 1000 | 200 | 121 | 41 | 61 | 1e-8 |

The fixed-point residual remained on the order of `1e-8`, indicating stable convergence of the continuation/intervention iteration.

