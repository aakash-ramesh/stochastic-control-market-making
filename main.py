from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

from config import ModelParams, GridParams, SimParams, save_config
from pde_solver import solve_hjb_qvi
from simulator import (
    simulate_gbm_paths,
    simulate_optimal_strategy,
    simulate_pure_delta_hedge,
    simulate_pure_passive,
)
from metrics import compute_metrics, print_metrics, save_metrics
from visualization import plot_all


def make_run_dir(base_name: str = "results") -> Path:
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = script_dir / base_name / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_notes(out_dir: Path, model: ModelParams):
    notes = f"""Notes on benchmark interpretation

1. Maker convention:
   maker_rebate_per_unit = {model.maker_rebate_per_unit}
   is treated as a positive cash rebate earned on passive fills.

2. Same market, bigger book:
   Across N-scenarios, the market environment is fixed:
       Lambda_b, Lambda_a, k_b, k_a, spread, rebate, taker fee,
       and impact parameters do not change.
   Only the short-option book size changes through N, and inventory controls scale as:
       Q_max(N)  = N * Q_max(1)
       q_step(N) = N * q_step(1)

3. Passive-MM caveat:
   Pure passive market making can look artificially strong because the model
   does not include adverse selection or permanent price impact from passive fills.
   Therefore passive-MM should be judged jointly by:
       - expected terminal wealth per option,
       - normalized cumulative tracking error,
       - drawdown,
       - inventory behavior.

4. Large-N reporting:
   Cross-N comparisons are reported using per-option wealth and normalized
   tracking error (dividing by N^2), so that results remain interpretable.
"""
    (out_dir / "benchmark_notes.txt").write_text(notes)


def main():
    print("=" * 88)
    print("  FINAL HJB-QVI RUN: SAME MARKET, BIGGER BOOK, NORMALIZED REPORTING")
    print("=" * 88)

    base_model = ModelParams(
        S0=100.0,
        sigma=0.02,
        K=100.0,
        T=1.0,
        num_options=1,
        Q_max=1.0,
        q_step=0.05,
        reference_Q_max=1.0,
        reference_q_step=0.05,
        eta=0.10,
        Lambda_b=140.0,
        Lambda_a=140.0,
        k_b=1.5,
        k_a=1.5,
        maker_rebate_per_unit=0.005,
        eps_taker=0.03,
        half_spread=0.05,
        kappa=0.02,
        beta=1.5,
        include_initial_premium=True,
    )

    grid = GridParams(
        N_t=200,
        N_S=121,
        n_std=5.0,
        delta_min=0.001,
        delta_max=2.0,
        N_delta=61,
        qvi_max_iter=200,
        qvi_tol=1e-8,
        qvi_relaxation=0.70,
    )

    sim = SimParams(
        N_paths=600,
        seed=42,
        dt_sim=base_model.T / grid.N_t,
        record_n_sample_paths=5,
    )

    run_dir = make_run_dir(base_name="results")
    print(f"\nArtifacts will be saved to: {run_dir}")

    N_scenarios = [1, 100, 1000]
    master_summary = []

    for N in N_scenarios:
        print("\n" + "=" * 88)
        print(f"  RUNNING SCENARIO: N = {N} SHORT CALL OPTION(S)")
        print("=" * 88)

        model = base_model.scaled_for_num_options(N)
        scenario_dir = run_dir / f"N_{N}"
        plots_dir = scenario_dir / "plots"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        save_config(model, grid, sim, scenario_dir)
        write_notes(scenario_dir, model)

        print(
            f"Scenario params | N={N}, "
            f"Q_max={model.Q_max}, q_step={model.q_step}, "
            f"Lambda_b={model.Lambda_b:.4f}, Lambda_a={model.Lambda_a:.4f}"
        )

        print("\n--- Solving HJB-QVI ---")
        t0 = time.time()
        h, policy, t_grid, S_grid, q_values = solve_hjb_qvi(model, grid, verbose=True)
        print(f"PDE/QVI solve complete in {time.time() - t0:.2f}s")

        print("\n--- Simulating paths ---")
        S_paths, t_path = simulate_gbm_paths(model, sim)

        results_opt = simulate_optimal_strategy(
            S_paths, t_path, model, sim, h, policy, t_grid, S_grid, q_values
        )
        results_delta = simulate_pure_delta_hedge(S_paths, t_path, model, sim)
        results_passive = simulate_pure_passive(S_paths, t_path, model, sim)

        results_opt["model"] = model
        results_delta["model"] = model
        results_passive["model"] = model

        print("\n--- Metrics ---")
        m_opt = compute_metrics(results_opt, f"Optimal (HJB-QVI), N={N}")
        m_delta = compute_metrics(results_delta, f"Pure Delta Hedge, N={N}")
        m_passive = compute_metrics(results_passive, f"Pure Passive MM, N={N}")

        for m in [m_opt, m_delta, m_passive]:
            print_metrics(m)

        save_metrics([m_opt, m_delta, m_passive], scenario_dir)

        print("\n--- Plotting ---")
        plot_all(
            h,
            policy,
            t_grid,
            S_grid,
            q_values,
            results_opt,
            results_delta,
            results_passive,
            metrics_list=[m_opt, m_delta, m_passive],
            save_dir=str(plots_dir),
            title_suffix=f" | N={N}",
        )

        master_summary.extend([
            {
                "N": N,
                "strategy": m_opt["strategy"],
                "mean_terminal_wealth_per_option": m_opt["mean_terminal_wealth_per_option"],
                "mean_cum_tracking_error_per_option_sq": m_opt["mean_cum_tracking_error_per_option_sq"],
                "reward_to_tracking_ratio": m_opt["reward_to_tracking_ratio"],
            },
            {
                "N": N,
                "strategy": m_delta["strategy"],
                "mean_terminal_wealth_per_option": m_delta["mean_terminal_wealth_per_option"],
                "mean_cum_tracking_error_per_option_sq": m_delta["mean_cum_tracking_error_per_option_sq"],
                "reward_to_tracking_ratio": m_delta["reward_to_tracking_ratio"],
            },
            {
                "N": N,
                "strategy": m_passive["strategy"],
                "mean_terminal_wealth_per_option": m_passive["mean_terminal_wealth_per_option"],
                "mean_cum_tracking_error_per_option_sq": m_passive["mean_cum_tracking_error_per_option_sq"],
                "reward_to_tracking_ratio": m_passive["reward_to_tracking_ratio"],
            },
        ])

    (run_dir / "master_summary.json").write_text(json.dumps(master_summary, indent=2))

    print("\n" + "=" * 88)
    print(f"Done. Final artifacts written to: {run_dir}")
    print("=" * 88)

    return run_dir


if __name__ == "__main__":
    main()