"""
black_scholes.py — Black-Scholes formulas for the hedge target.

FACTS (not assumptions — these are mathematical consequences of GBM + no-arb):
  - BS call price: C = S·Φ(d1) - K·e^{-rτ}·Φ(d2)
  - BS delta: ∂C/∂S = Φ(d1)
  - d1 = [ln(S/K) + (r + σ²/2)τ] / (σ√τ)
  - d2 = d1 - σ√τ

With our assumption r=0:
  - d1 = [ln(S/K) + σ²τ/2] / (σ√τ)
  - C = S·Φ(d1) - K·Φ(d2)
"""

import numpy as np
from scipy.stats import norm


def bs_d1(S: np.ndarray, K: float, sigma: float, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    """
    Compute d1 in the Black-Scholes formula.
    
    Args:
        S: Stock price(s), array or scalar
        K: Strike price
        sigma: Volatility
        tau: Time to maturity (T - t), must be > 0
        r: Risk-free rate (default 0)
    
    Returns:
        d1 values, same shape as broadcast(S, tau)
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    # Guard against tau=0 (handled separately at terminal time)
    tau_safe = np.maximum(tau, 1e-12)
    sqrt_tau = np.sqrt(tau_safe)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)


def bs_delta(S: np.ndarray, K: float, sigma: float, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    """
    Black-Scholes delta = Φ(d1).
    
    This is the hedge target: the number of shares needed to delta-hedge
    one short call option.
    
    At τ=0: delta = 1 if S>K, 0 if S<K (step function).
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # Terminal case: delta is the indicator 1_{S>K}
    result = np.where(tau <= 1e-12,
                      np.where(S > K, 1.0, np.where(S == K, 0.5, 0.0)),
                      norm.cdf(bs_d1(S, K, sigma, tau, r)))
    return result


def bs_call_price(S: np.ndarray, K: float, sigma: float, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    """
    Black-Scholes call option price.
    
    C = S·Φ(d1) - K·e^{-rτ}·Φ(d2)
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # Terminal case
    terminal_mask = tau <= 1e-12
    d1 = bs_d1(S, K, sigma, tau, r)
    d2 = d1 - sigma * np.sqrt(np.maximum(tau, 1e-12))

    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    # At maturity, price = max(S-K, 0)
    price = np.where(terminal_mask, np.maximum(S - K, 0.0), price)
    return price


def bs_gamma(S: np.ndarray, K: float, sigma: float, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    """
    Black-Scholes gamma = ∂²C/∂S² = φ(d1) / (S·σ·√τ).
    
    Useful for diagnostics: high gamma means delta is changing rapidly,
    so the hedging problem is harder.
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.maximum(tau, 1e-12)

    d1 = bs_d1(S, K, sigma, tau, r)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau_safe))


def bs_vega(S: np.ndarray, K: float, sigma: float, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    """Black-Scholes vega = S·φ(d1)·√τ."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.maximum(tau, 1e-12)

    d1 = bs_d1(S, K, sigma, tau, r)
    return S * norm.pdf(d1) * np.sqrt(tau_safe)
