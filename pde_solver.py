from __future__ import annotations

import numpy as np
from config import ModelParams, GridParams, build_grids
from black_scholes import bs_delta


def fill_intensity(delta: np.ndarray, Lambda: float, k: float) -> np.ndarray:
    return Lambda * np.exp(-k * delta)


def maker_rebate(model: ModelParams, size: float) -> float:
    return model.maker_rebate_per_unit * abs(size)


def market_order_cost(model: ModelParams, xi: np.ndarray | float, S: np.ndarray | float) -> np.ndarray:
    xi_arr = np.asarray(xi, dtype=float)
    S_arr = np.asarray(S, dtype=float)
    return (
        xi_arr * S_arr
        + np.abs(xi_arr) * (model.half_spread + model.eps_taker)
        + model.kappa * np.abs(xi_arr) ** model.beta
    )


def optimize_quote_offset(
    delta_grid: np.ndarray,
    Lambda: float,
    k: float,
    base_gain: np.ndarray,
    delta_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    lam = fill_intensity(delta_grid, Lambda, k)
    contribution = lam[:, None] * (delta_scale * delta_grid[:, None] + base_gain[None, :])
    best_idx = np.argmax(contribution, axis=0)
    best_contribution = contribution[best_idx, np.arange(contribution.shape[1])]
    best_delta = delta_grid[best_idx]
    best_contribution = np.maximum(best_contribution, 0.0)
    return best_contribution, best_delta


def build_generator_matrix(model: ModelParams, N_S: int, dy: float) -> np.ndarray:
    """
    Generator in y = log S for dY = -0.5 sigma^2 dt + sigma dW:
        L = 0.5 sigma^2 d_yy - 0.5 sigma^2 d_y
    """
    sig2 = model.sigma ** 2
    A = np.zeros((N_S, N_S), dtype=float)
    a2 = 0.5 * sig2 / dy**2
    a1 = -0.5 * sig2

    for j in range(1, N_S - 1):
        A[j, j - 1] += a2 + (-a1) / (2 * dy)
        A[j, j] += -2 * a2
        A[j, j + 1] += a2 + a1 / (2 * dy)

    # one-sided second-order boundary treatment
    A[0, 0] += a2 * 2 + a1 * (-3) / (2 * dy)
    A[0, 1] += a2 * (-5) + a1 * 4 / (2 * dy)
    A[0, 2] += a2 * 4 + a1 * (-1) / (2 * dy)
    A[0, 3] += a2 * (-1)

    A[-1, -1] += a2 * 2 + a1 * 3 / (2 * dy)
    A[-1, -2] += a2 * (-5) + a1 * (-4) / (2 * dy)
    A[-1, -3] += a2 * 4 + a1 * 1 / (2 * dy)
    A[-1, -4] += a2 * (-1)

    return A


def q_to_index(q: float, q_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(q_values - q)))


def compute_intervention(
    h_candidate: np.ndarray,
    S_grid: np.ndarray,
    q_idx: int,
    q_values: np.ndarray,
    model: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    best_Mh = np.full(len(S_grid), -np.inf, dtype=float)
    best_xi = np.zeros(len(S_grid), dtype=float)
    q = q_values[q_idx]

    for xi in model.market_order_grid:
        if abs(xi) < 1e-14:
            continue
        q_new = q + xi
        if q_new < q_values[0] - 1e-12 or q_new > q_values[-1] + 1e-12:
            continue

        target_idx = q_to_index(q_new, q_values)
        Mh_candidate = h_candidate[:, target_idx] - market_order_cost(model, xi, S_grid)

        improve = Mh_candidate > best_Mh
        best_Mh = np.where(improve, Mh_candidate, best_Mh)
        best_xi = np.where(improve, xi, best_xi)

    return best_Mh, best_xi


def solve_continuation_cn(
    h_next: np.ndarray,
    h_guess: np.ndarray,
    t_n: float,
    S_grid: np.ndarray,
    q_values: np.ndarray,
    model: ModelParams,
    grid: GridParams,
    A: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N_S, N_q = h_next.shape
    I = np.eye(N_S)
    lhs = I - 0.5 * dt * A
    rhs_mat = I + 0.5 * dt * A

    delta_bs = bs_delta(S_grid, model.K, model.sigma, model.T - t_n, model.r)
    target_delta = model.num_options * delta_bs
    delta_grid = grid.delta_grid
    c = model.q_step

    h_cont = np.zeros_like(h_next)
    db_store = np.zeros_like(h_next)
    da_store = np.zeros_like(h_next)

    for iq, q in enumerate(q_values):
        tracking_source = -model.eta * (q - target_delta) ** 2

        if iq + 1 < N_q:
            base_bid = (
                h_guess[:, iq + 1]
                - h_guess[:, iq]
                - c * S_grid
                + maker_rebate(model, c)
            )
            bid_contrib, db_star = optimize_quote_offset(
                delta_grid, model.Lambda_b, model.k_b, base_bid, delta_scale=c
            )
        else:
            bid_contrib = np.zeros(N_S)
            db_star = np.full(N_S, delta_grid[-1])

        if iq - 1 >= 0:
            base_ask = (
                h_guess[:, iq - 1]
                - h_guess[:, iq]
                + c * S_grid
                + maker_rebate(model, c)
            )
            ask_contrib, da_star = optimize_quote_offset(
                delta_grid, model.Lambda_a, model.k_a, base_ask, delta_scale=c
            )
        else:
            ask_contrib = np.zeros(N_S)
            da_star = np.full(N_S, delta_grid[-1])

        source = tracking_source + bid_contrib + ask_contrib
        rhs = rhs_mat @ h_next[:, iq] + dt * source
        h_cont[:, iq] = np.linalg.solve(lhs, rhs)

        db_store[:, iq] = db_star
        da_store[:, iq] = da_star

    return h_cont, db_store, da_store


def solve_hjb_qvi(model: ModelParams, grid: GridParams, verbose: bool = True):
    t_grid, S_grid, _, q_values, dt, dy = build_grids(model, grid)
    N_t = grid.N_t
    N_S = len(S_grid)
    N_q = len(q_values)

    h = np.zeros((N_t + 1, N_S, N_q), dtype=float)
    delta_b_opt = np.zeros_like(h)
    delta_a_opt = np.zeros_like(h)
    xi_opt = np.zeros_like(h)
    is_intervention = np.zeros_like(h, dtype=bool)
    qvi_residual = np.zeros(N_t + 1)

    for iq, q in enumerate(q_values):
        h[-1, :, iq] = q * S_grid - model.num_options * np.maximum(S_grid - model.K, 0.0)

    A = build_generator_matrix(model, N_S, dy)

    for n in range(N_t - 1, -1, -1):
        t_n = t_grid[n]
        h_next = h[n + 1]
        h_guess = h_next.copy()

        last_db = np.zeros((N_S, N_q))
        last_da = np.zeros((N_S, N_q))
        last_xi = np.zeros((N_S, N_q))
        last_mask = np.zeros((N_S, N_q), dtype=bool)

        for _ in range(grid.qvi_max_iter):
            h_cont, db_star, da_star = solve_continuation_cn(
                h_next, h_guess, t_n, S_grid, q_values, model, grid, A, dt
            )

            h_raw = h_cont.copy()
            xi_star_all = np.zeros((N_S, N_q))
            intervene_all = np.zeros((N_S, N_q), dtype=bool)

            for iq in range(N_q):
                Mh, xi_star = compute_intervention(h_guess, S_grid, iq, q_values, model)
                intervene = Mh > h_cont[:, iq]
                h_raw[:, iq] = np.where(intervene, Mh, h_cont[:, iq])
                xi_star_all[:, iq] = xi_star
                intervene_all[:, iq] = intervene

            # under-relaxation for large-N stability
            h_new = grid.qvi_relaxation * h_raw + (1.0 - grid.qvi_relaxation) * h_guess

            err = float(np.max(np.abs(h_new - h_guess)))
            h_guess = h_new
            last_db, last_da = db_star, da_star
            last_xi, last_mask = xi_star_all, intervene_all

            if err < grid.qvi_tol:
                qvi_residual[n] = err
                break
        else:
            qvi_residual[n] = err

        h[n] = h_guess
        delta_b_opt[n] = last_db
        delta_a_opt[n] = last_da
        xi_opt[n] = last_xi
        is_intervention[n] = last_mask

        if verbose and (n % 25 == 0 or n == N_t - 1):
            iq0 = q_to_index(0.0, q_values)
            iS0 = int(np.argmin(np.abs(S_grid - model.S0)))
            print(
                f"  N={model.num_options:>6d} | "
                f"time step {n:4d}/{N_t} | "
                f"t={t_n:.4f} | "
                f"qvi_residual={qvi_residual[n]:.3e} | "
                f"h(t,S0,q≈0)={h[n, iS0, iq0]:.6f}"
            )

    policy = {
        "delta_b": delta_b_opt,
        "delta_a": delta_a_opt,
        "xi": xi_opt,
        "is_intervention": is_intervention,
        "qvi_residual": qvi_residual,
    }
    return h, policy, t_grid, S_grid, q_values