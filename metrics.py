from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


def compute_metrics(results: Dict[str, np.ndarray], strategy_name: str = "Strategy") -> Dict:
    model = results["model"]
    N = model.num_options

    tw = results["terminal_wealth"]
    inv = results["inventory"]
    cte = results["cum_tracking_error"]
    n_mo = results["n_market_orders"]
    n_lf = results["n_limit_fills"]
    wealth_path = results["wealth_path"]

    mean_tw = float(np.mean(tw))
    var_tw = float(np.var(tw))
    std_tw = float(np.std(tw))
    mean_cte = float(np.mean(cte))
    sharpe = float(mean_tw / std_tw) if std_tw > 0 else np.inf

    final_q = inv[:, -1]
    mean_final_q = float(np.mean(final_q))
    std_final_q = float(np.std(final_q))

    mean_n_mo = float(np.mean(n_mo))
    mean_n_lf = float(np.mean(n_lf))

    alpha = 0.05
    var_95 = float(np.percentile(tw, 100 * alpha))
    cvar_95 = float(np.mean(tw[tw <= var_95])) if np.any(tw <= var_95) else var_95

    cummax = np.maximum.accumulate(wealth_path, axis=1)
    drawdown = cummax - wealth_path
    max_dd = np.max(drawdown, axis=1)
    mean_max_dd = float(np.mean(max_dd))

    tw_p10, tw_p25, tw_p50, tw_p75, tw_p90 = [
        float(np.percentile(tw, p)) for p in [10, 25, 50, 75, 90]
    ]

    option_exercise_freq = float(np.mean(results["option_payoff"] > 0))

    total_rebate = results["maker_rebates_path"][:, -1]
    total_taker = results["taker_costs_path"][:, -1]
    total_impact = results["impact_costs_path"][:, -1]

    # Cross-N comparable metrics
    mean_tw_per_option = mean_tw / N
    std_tw_per_option = std_tw / N
    mean_cte_per_option_sq = mean_cte / (N**2)
    mean_max_dd_per_option = mean_max_dd / N
    mean_rebate_per_option = float(np.mean(total_rebate)) / N
    mean_taker_per_option = float(np.mean(total_taker)) / N
    mean_impact_per_option = float(np.mean(total_impact)) / N

    reward_to_tracking_ratio = float(mean_tw_per_option / (1.0 + mean_cte_per_option_sq))

    metrics = {
        "strategy": strategy_name,
        "N": N,
        "mean_terminal_wealth": mean_tw,
        "var_terminal_wealth": var_tw,
        "std_terminal_wealth": std_tw,
        "sharpe_ratio": sharpe,
        "mean_cum_tracking_error": mean_cte,
        "mean_terminal_wealth_per_option": mean_tw_per_option,
        "std_terminal_wealth_per_option": std_tw_per_option,
        "mean_cum_tracking_error_per_option_sq": mean_cte_per_option_sq,
        "reward_to_tracking_ratio": reward_to_tracking_ratio,
        "mean_final_inventory": mean_final_q,
        "std_final_inventory": std_final_q,
        "mean_final_inventory_per_option": mean_final_q / N,
        "std_final_inventory_per_option": std_final_q / N,
        "mean_market_orders": mean_n_mo,
        "mean_limit_fills": mean_n_lf,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "mean_max_drawdown": mean_max_dd,
        "mean_max_drawdown_per_option": mean_max_dd_per_option,
        "tw_p10": tw_p10,
        "tw_p25": tw_p25,
        "tw_p50": tw_p50,
        "tw_p75": tw_p75,
        "tw_p90": tw_p90,
        "option_exercise_freq": option_exercise_freq,
        "mean_total_maker_rebate": float(np.mean(total_rebate)),
        "mean_total_taker_cost": float(np.mean(total_taker)),
        "mean_total_impact_cost": float(np.mean(total_impact)),
        "mean_total_maker_rebate_per_option": mean_rebate_per_option,
        "mean_total_taker_cost_per_option": mean_taker_per_option,
        "mean_total_impact_cost_per_option": mean_impact_per_option,
        "initial_premium": float(results.get("initial_premium", 0.0)),
        "initial_premium_per_option": float(results.get("initial_premium", 0.0)) / N,
    }
    return metrics


def print_metrics(metrics: Dict):
    name = metrics["strategy"]
    print(f"\n{'='*76}")
    print(f"  Strategy: {name}")
    print(f"{'='*76}")
    print(f"  E[Terminal Wealth]            = {metrics['mean_terminal_wealth']:>12.6f}")
    print(f"  E[Terminal Wealth]/option     = {metrics['mean_terminal_wealth_per_option']:>12.6f}")
    print(f"  Std[Terminal Wealth]          = {metrics['std_terminal_wealth']:>12.6f}")
    print(f"  Std[Terminal Wealth]/option   = {metrics['std_terminal_wealth_per_option']:>12.6f}")
    print(f"  Sharpe Ratio                  = {metrics['sharpe_ratio']:>12.6f}")
    print(f"  E[Cum Tracking Error]         = {metrics['mean_cum_tracking_error']:>12.6f}")
    print(f"  E[Cum Tracking Error]/N^2     = {metrics['mean_cum_tracking_error_per_option_sq']:>12.6f}")
    print(f"  Reward/Tracking Ratio         = {metrics['reward_to_tracking_ratio']:>12.6f}")
    print(f"  Mean Final Inventory          = {metrics['mean_final_inventory']:>12.6f}")
    print(f"  Mean Final Inventory/option   = {metrics['mean_final_inventory_per_option']:>12.6f}")
    print(f"  Std Final Inventory           = {metrics['std_final_inventory']:>12.6f}")
    print(f"  Mean Market Orders            = {metrics['mean_market_orders']:>12.4f}")
    print(f"  Mean Limit Fills              = {metrics['mean_limit_fills']:>12.4f}")
    print(f"  VaR (95%)                     = {metrics['VaR_95']:>12.6f}")
    print(f"  CVaR (95%)                    = {metrics['CVaR_95']:>12.6f}")
    print(f"  Mean Max Drawdown             = {metrics['mean_max_drawdown']:>12.6f}")
    print(f"  Mean Max Drawdown/option      = {metrics['mean_max_drawdown_per_option']:>12.6f}")
    print(f"  Mean Maker Rebate             = {metrics['mean_total_maker_rebate']:>12.6f}")
    print(f"  Mean Maker Rebate/option      = {metrics['mean_total_maker_rebate_per_option']:>12.6f}")
    print(f"  Mean Taker Cost               = {metrics['mean_total_taker_cost']:>12.6f}")
    print(f"  Mean Taker Cost/option        = {metrics['mean_total_taker_cost_per_option']:>12.6f}")
    print(f"  Mean Impact Cost              = {metrics['mean_total_impact_cost']:>12.6f}")
    print(f"  Mean Impact Cost/option       = {metrics['mean_total_impact_cost_per_option']:>12.6f}")
    print(f"  Initial Premium               = {metrics['initial_premium']:>12.6f}")
    print(f"  Initial Premium/option        = {metrics['initial_premium_per_option']:>12.6f}")
    print(f"  Option Exercise Freq          = {metrics['option_exercise_freq']:>12.2%}")
    print(f"{'='*76}")


def comparison_table(metrics_list: List[Dict]) -> pd.DataFrame:
    rows = [
        ("E[Terminal Wealth]", "mean_terminal_wealth"),
        ("E[Terminal Wealth]/option", "mean_terminal_wealth_per_option"),
        ("Std[Terminal Wealth]/option", "std_terminal_wealth_per_option"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("E[Cum Tracking Error]/N^2", "mean_cum_tracking_error_per_option_sq"),
        ("Reward/Tracking Ratio", "reward_to_tracking_ratio"),
        ("Mean Final Inventory/option", "mean_final_inventory_per_option"),
        ("Mean Market Orders", "mean_market_orders"),
        ("Mean Limit Fills", "mean_limit_fills"),
        ("Mean Max Drawdown/option", "mean_max_drawdown_per_option"),
        ("Mean Maker Rebate/option", "mean_total_maker_rebate_per_option"),
        ("Mean Taker Cost/option", "mean_total_taker_cost_per_option"),
        ("Mean Impact Cost/option", "mean_total_impact_cost_per_option"),
    ]
    df = pd.DataFrame(
        {m["strategy"]: [m[key] for _, key in rows] for m in metrics_list},
        index=[label for label, _ in rows],
    )

    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON")
    print("=" * 100)
    print(df.round(6).to_string())
    print("=" * 100)

    return df


def save_metrics(metrics_list: List[Dict], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = comparison_table(metrics_list)
    df.to_csv(out_path / "metrics_comparison.csv")
    (out_path / "metrics_summary.json").write_text(json.dumps(metrics_list, indent=2))