from __future__ import annotations

import numpy as np
from config import ModelParams, SimParams
from black_scholes import bs_delta, bs_call_price
from pde_solver import fill_intensity, maker_rebate, market_order_cost


def simulate_gbm_paths(model: ModelParams, sim: SimParams):
    rng = np.random.default_rng(sim.seed)
    N_steps = int(round(model.T / sim.dt_sim))
    t_path = np.linspace(0.0, model.T, N_steps + 1)

    S_paths = np.zeros((sim.N_paths, N_steps + 1), dtype=float)
    S_paths[:, 0] = model.S0

    dt = sim.dt_sim
    Z = rng.standard_normal((sim.N_paths, N_steps))
    drift_inc = (model.mu - 0.5 * model.sigma ** 2) * dt
    diff_scale = model.sigma * np.sqrt(dt)

    for n in range(N_steps):
        S_paths[:, n + 1] = S_paths[:, n] * np.exp(drift_inc + diff_scale * Z[:, n])

    return S_paths, t_path


def _nearest_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_right = np.abs(values - right) < np.abs(values - left)
    return idx - 1 + choose_right.astype(int)


def _initial_cash(model: ModelParams) -> float:
    if not model.include_initial_premium:
        return 0.0
    return float(model.num_options * bs_call_price(model.S0, model.K, model.sigma, model.T, model.r))


def _wealth_path_from_state(
    cash: np.ndarray,
    inventory: np.ndarray,
    S_paths: np.ndarray,
    model: ModelParams,
    t_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tau = model.T - t_path[None, :]
    option_mtm = model.num_options * bs_call_price(S_paths, model.K, model.sigma, tau, model.r)
    wealth = cash + inventory * S_paths - option_mtm
    return wealth, option_mtm


def simulate_optimal_strategy(S_paths, t_path, model: ModelParams, sim: SimParams, h, policy, t_grid, S_grid, q_values):
    rng = np.random.default_rng(sim.seed + 1)
    N_paths = S_paths.shape[0]
    N_steps = S_paths.shape[1] - 1
    dt = sim.dt_sim
    q_step = model.q_step

    cash = np.zeros((N_paths, N_steps + 1), dtype=float)
    inventory = np.zeros((N_paths, N_steps + 1), dtype=float)
    tracking_error_sq = np.zeros((N_paths, N_steps + 1), dtype=float)

    maker_rebates = np.zeros((N_paths, N_steps + 1), dtype=float)
    taker_costs = np.zeros((N_paths, N_steps + 1), dtype=float)
    impact_costs = np.zeros((N_paths, N_steps + 1), dtype=float)

    n_market_orders = np.zeros(N_paths, dtype=int)
    n_limit_fills = np.zeros(N_paths, dtype=int)

    cash[:, 0] = _initial_cash(model)

    for step in range(N_steps):
        t = t_path[step]
        S = S_paths[:, step]
        tau = model.T - t
        q = inventory[:, step]

        delta_bs = bs_delta(S, model.K, model.sigma, tau, model.r)
        target_delta = model.num_options * delta_bs
        tracking_error_sq[:, step] = (q - target_delta) ** 2

        t_idx = np.clip(np.searchsorted(t_grid, t) - 1, 0, len(t_grid) - 2)
        S_idx_all = _nearest_indices(np.clip(S, S_grid[0], S_grid[-1]), S_grid)
        q_idx_all = _nearest_indices(np.clip(q, q_values[0], q_values[-1]), q_values)

        new_cash = cash[:, step].copy()
        new_q = q.copy()
        new_rebate = np.zeros(N_paths)
        new_taker = np.zeros(N_paths)
        new_impact = np.zeros(N_paths)

        u_bid = rng.random(N_paths)
        u_ask = rng.random(N_paths)

        for i in range(N_paths):
            qi = q_idx_all[i]
            S_idx = S_idx_all[i]

            if policy["is_intervention"][t_idx, S_idx, qi]:
                xi = float(policy["xi"][t_idx, S_idx, qi])
                q_new = q[i] + xi
                if abs(xi) > 1e-14 and q_values[0] - 1e-12 <= q_new <= q_values[-1] + 1e-12:
                    total_cost = market_order_cost(model, xi, S[i])
                    new_cash[i] -= total_cost
                    new_q[i] = q_new
                    new_taker[i] = abs(xi) * (model.half_spread + model.eps_taker)
                    new_impact[i] = model.kappa * abs(xi) ** model.beta
                    n_market_orders[i] += 1
                    continue

            db = float(policy["delta_b"][t_idx, S_idx, qi])
            da = float(policy["delta_a"][t_idx, S_idx, qi])

            lambda_b = fill_intensity(db, model.Lambda_b, model.k_b)
            lambda_a = fill_intensity(da, model.Lambda_a, model.k_a)

            if u_bid[i] < lambda_b * dt and new_q[i] + q_step <= q_values[-1] + 1e-12:
                rebate = maker_rebate(model, q_step)
                new_cash[i] -= q_step * (S[i] - db)
                new_cash[i] += rebate
                new_q[i] += q_step
                new_rebate[i] += rebate
                n_limit_fills[i] += 1

            if u_ask[i] < lambda_a * dt and new_q[i] - q_step >= q_values[0] - 1e-12:
                rebate = maker_rebate(model, q_step)
                new_cash[i] += q_step * (S[i] + da)
                new_cash[i] += rebate
                new_q[i] -= q_step
                new_rebate[i] += rebate
                n_limit_fills[i] += 1

        cash[:, step + 1] = new_cash
        inventory[:, step + 1] = new_q
        maker_rebates[:, step + 1] = maker_rebates[:, step] + new_rebate
        taker_costs[:, step + 1] = taker_costs[:, step] + new_taker
        impact_costs[:, step + 1] = impact_costs[:, step] + new_impact

    S_T = S_paths[:, -1]
    delta_bs_T = bs_delta(S_T, model.K, model.sigma, 0.0, model.r)
    target_delta_T = model.num_options * delta_bs_T
    tracking_error_sq[:, -1] = (inventory[:, -1] - target_delta_T) ** 2

    option_payoff = model.num_options * np.maximum(S_T - model.K, 0.0)
    terminal_wealth = cash[:, -1] + inventory[:, -1] * S_T - option_payoff
    cum_tracking_error = np.trapezoid(tracking_error_sq, t_path, axis=1)

    wealth_path, option_mtm = _wealth_path_from_state(cash, inventory, S_paths, model, t_path)

    return {
        "terminal_wealth": terminal_wealth,
        "cash": cash,
        "inventory": inventory,
        "tracking_error_sq": tracking_error_sq,
        "cum_tracking_error": cum_tracking_error,
        "option_payoff": option_payoff,
        "option_mtm_path": option_mtm,
        "wealth_path": wealth_path,
        "n_market_orders": n_market_orders,
        "n_limit_fills": n_limit_fills,
        "maker_rebates_path": maker_rebates,
        "taker_costs_path": taker_costs,
        "impact_costs_path": impact_costs,
        "S_paths": S_paths,
        "t_path": t_path,
        "initial_premium": _initial_cash(model),
    }


def simulate_pure_delta_hedge(S_paths, t_path, model: ModelParams, sim: SimParams):
    N_paths = S_paths.shape[0]
    N_steps = S_paths.shape[1] - 1

    cash = np.zeros((N_paths, N_steps + 1), dtype=float)
    inventory = np.zeros((N_paths, N_steps + 1), dtype=float)
    tracking_error_sq = np.zeros((N_paths, N_steps + 1), dtype=float)

    maker_rebates = np.zeros((N_paths, N_steps + 1), dtype=float)
    taker_costs = np.zeros((N_paths, N_steps + 1), dtype=float)
    impact_costs = np.zeros((N_paths, N_steps + 1), dtype=float)

    cash[:, 0] = _initial_cash(model)

    for step in range(N_steps):
        t = t_path[step]
        S = S_paths[:, step]
        tau = model.T - t
        q = inventory[:, step]

        delta_bs = bs_delta(S, model.K, model.sigma, tau, model.r)
        target_delta = model.num_options * delta_bs
        tracking_error_sq[:, step] = (q - target_delta) ** 2

        target_q = np.clip(np.round(target_delta / model.q_step) * model.q_step, -model.Q_max, model.Q_max)
        xi = target_q - q

        total_cost = market_order_cost(model, xi, S)
        cash[:, step + 1] = cash[:, step] - total_cost
        inventory[:, step + 1] = target_q

        taker_costs[:, step + 1] = taker_costs[:, step] + np.abs(xi) * (model.half_spread + model.eps_taker)
        impact_costs[:, step + 1] = impact_costs[:, step] + model.kappa * np.abs(xi) ** model.beta

    S_T = S_paths[:, -1]
    delta_bs_T = bs_delta(S_T, model.K, model.sigma, 0.0, model.r)
    target_delta_T = model.num_options * delta_bs_T
    tracking_error_sq[:, -1] = (inventory[:, -1] - target_delta_T) ** 2

    option_payoff = model.num_options * np.maximum(S_T - model.K, 0.0)
    terminal_wealth = cash[:, -1] + inventory[:, -1] * S_T - option_payoff
    cum_tracking_error = np.trapezoid(tracking_error_sq, t_path, axis=1)

    wealth_path, option_mtm = _wealth_path_from_state(cash, inventory, S_paths, model, t_path)

    return {
        "terminal_wealth": terminal_wealth,
        "cash": cash,
        "inventory": inventory,
        "tracking_error_sq": tracking_error_sq,
        "cum_tracking_error": cum_tracking_error,
        "option_payoff": option_payoff,
        "option_mtm_path": option_mtm,
        "wealth_path": wealth_path,
        "n_market_orders": np.sum(np.abs(np.diff(inventory, axis=1)) > 1e-12, axis=1),
        "n_limit_fills": np.zeros(N_paths, dtype=int),
        "maker_rebates_path": maker_rebates,
        "taker_costs_path": taker_costs,
        "impact_costs_path": impact_costs,
        "S_paths": S_paths,
        "t_path": t_path,
        "initial_premium": _initial_cash(model),
    }


def simulate_pure_passive(S_paths, t_path, model: ModelParams, sim: SimParams):
    rng = np.random.default_rng(sim.seed + 2)
    N_paths = S_paths.shape[0]
    N_steps = S_paths.shape[1] - 1
    dt = sim.dt_sim
    q_step = model.q_step
    gamma_eff = 0.1

    cash = np.zeros((N_paths, N_steps + 1), dtype=float)
    inventory = np.zeros((N_paths, N_steps + 1), dtype=float)
    tracking_error_sq = np.zeros((N_paths, N_steps + 1), dtype=float)

    maker_rebates = np.zeros((N_paths, N_steps + 1), dtype=float)
    taker_costs = np.zeros((N_paths, N_steps + 1), dtype=float)
    impact_costs = np.zeros((N_paths, N_steps + 1), dtype=float)
    n_limit_fills = np.zeros(N_paths, dtype=int)

    cash[:, 0] = _initial_cash(model)

    for step in range(N_steps):
        t = t_path[step]
        S = S_paths[:, step]
        tau = model.T - t
        q = inventory[:, step]

        delta_bs = bs_delta(S, model.K, model.sigma, tau, model.r)
        target_delta = model.num_options * delta_bs
        tracking_error_sq[:, step] = (q - target_delta) ** 2

        total_spread = gamma_eff * model.sigma ** 2 * tau + (2 / gamma_eff) * np.log(1 + gamma_eff / model.k_b)
        delta_half = total_spread / 2.0

        lambda_b = fill_intensity(delta_half, model.Lambda_b, model.k_b)
        lambda_a = fill_intensity(delta_half, model.Lambda_a, model.k_a)

        u_bid = rng.random(N_paths)
        u_ask = rng.random(N_paths)

        new_cash = cash[:, step].copy()
        new_q = q.copy()
        new_rebate = np.zeros(N_paths)

        bid_fill = (u_bid < lambda_b * dt) & (new_q + q_step <= model.Q_max + 1e-12)
        rb_b = maker_rebate(model, q_step)
        new_cash[bid_fill] -= q_step * (S[bid_fill] - delta_half)
        new_cash[bid_fill] += rb_b
        new_q[bid_fill] += q_step
        new_rebate[bid_fill] += rb_b
        n_limit_fills += bid_fill.astype(int)

        ask_fill = (u_ask < lambda_a * dt) & (new_q - q_step >= -model.Q_max - 1e-12)
        rb_a = maker_rebate(model, q_step)
        new_cash[ask_fill] += q_step * (S[ask_fill] + delta_half)
        new_cash[ask_fill] += rb_a
        new_q[ask_fill] -= q_step
        new_rebate[ask_fill] += rb_a
        n_limit_fills += ask_fill.astype(int)

        cash[:, step + 1] = new_cash
        inventory[:, step + 1] = new_q
        maker_rebates[:, step + 1] = maker_rebates[:, step] + new_rebate
        taker_costs[:, step + 1] = taker_costs[:, step]
        impact_costs[:, step + 1] = impact_costs[:, step]

    S_T = S_paths[:, -1]
    delta_bs_T = bs_delta(S_T, model.K, model.sigma, 0.0, model.r)
    target_delta_T = model.num_options * delta_bs_T
    tracking_error_sq[:, -1] = (inventory[:, -1] - target_delta_T) ** 2

    option_payoff = model.num_options * np.maximum(S_T - model.K, 0.0)
    terminal_wealth = cash[:, -1] + inventory[:, -1] * S_T - option_payoff
    cum_tracking_error = np.trapezoid(tracking_error_sq, t_path, axis=1)

    wealth_path, option_mtm = _wealth_path_from_state(cash, inventory, S_paths, model, t_path)

    return {
        "terminal_wealth": terminal_wealth,
        "cash": cash,
        "inventory": inventory,
        "tracking_error_sq": tracking_error_sq,
        "cum_tracking_error": cum_tracking_error,
        "option_payoff": option_payoff,
        "option_mtm_path": option_mtm,
        "wealth_path": wealth_path,
        "n_market_orders": np.zeros(N_paths, dtype=int),
        "n_limit_fills": n_limit_fills,
        "maker_rebates_path": maker_rebates,
        "taker_costs_path": taker_costs,
        "impact_costs_path": impact_costs,
        "S_paths": S_paths,
        "t_path": t_path,
        "initial_premium": _initial_cash(model),
    }