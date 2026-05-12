from __future__ import annotations

import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })


def _clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.75", alpha=0.9),
    )


def plot_value_function(h, t_grid, S_grid, q_values, save_dir="plots", title_suffix=""):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    time_slices = [0, len(t_grid) // 4, len(t_grid) // 2, 3 * len(t_grid) // 4]
    panel_names = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(1, len(time_slices), figsize=(19, 4.6), constrained_layout=True)

    vmin = min(np.min(h[ti]) for ti in time_slices)
    vmax = max(np.max(h[ti]) for ti in time_slices)

    images = []
    for ax, ti, name in zip(axes, time_slices, panel_names):
        im = ax.imshow(
            h[ti].T,
            aspect="auto",
            origin="lower",
            extent=[S_grid[0], S_grid[-1], q_values[0], q_values[-1]],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)
        ax.set_xlabel(r"Stock price $S$")
        ax.set_ylabel(r"Inventory $q$")
        _panel_label(ax, rf"{name} $t={t_grid[ti]:.3f}$")
        _clean_axis(ax)

    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.90, pad=0.02)
    cbar.set_label(r"Value $h(t,S,q)$")

    fig.savefig(os.path.join(save_dir, "optimal_value_function.png"))
    plt.close(fig)


def plot_intervention_regions(policy, t_grid, S_grid, q_values, save_dir="plots", title_suffix=""):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    time_slices = [0, len(t_grid) // 4, len(t_grid) // 2, 3 * len(t_grid) // 4]
    panel_names = ["(a)", "(b)", "(c)", "(d)"]
    cmap = ListedColormap(["white", "#8b0000"])

    fig, axes = plt.subplots(1, len(time_slices), figsize=(19, 4.6), constrained_layout=True)

    for ax, ti, name in zip(axes, time_slices, panel_names):
        ax.imshow(
            policy["is_intervention"][ti].T.astype(float),
            aspect="auto",
            origin="lower",
            extent=[S_grid[0], S_grid[-1], q_values[0], q_values[-1]],
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel(r"Stock price $S$")
        ax.set_ylabel(r"Inventory $q$")
        _panel_label(ax, rf"{name} $t={t_grid[ti]:.3f}$")
        _clean_axis(ax)

    legend_handles = [
        Patch(facecolor="white", edgecolor="0.4", label="Continuation region"),
        Patch(facecolor="#8b0000", edgecolor="0.4", label="Intervention region"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.savefig(os.path.join(save_dir, "intervention_regions.png"))
    plt.close(fig)


def plot_initial_value_slice(h, S_grid, q_values, save_dir="plots", title_suffix=""):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    iS0 = int(np.argmin(np.abs(S_grid - 100.0)))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(q_values, h[0, iS0, :], linewidth=2.2, label=rf"$S \approx {S_grid[iS0]:.2f}$")
    ax.set_xlabel(r"Inventory $q$")
    ax.set_ylabel(r"$h(0,S,q)$")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "optimal_initial_value_slice.png"))
    plt.close(fig)


def plot_strategy_value_report(metrics_list: List[Dict], save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    names = [m["strategy"] for m in metrics_list]
    vals = [m["mean_terminal_wealth_per_option"] for m in metrics_list]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    bars = ax.bar(names, vals, alpha=0.90, edgecolor="0.3")
    ax.set_ylabel("Expected terminal wealth per option")
    ax.set_xlabel("Strategy")
    ax.tick_params(axis="x", rotation=12)
    _clean_axis(ax)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(os.path.join(save_dir, "strategy_value_report.png"))
    plt.close(fig)


def plot_strategy_tradeoff(metrics_list: List[Dict], save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, m in enumerate(metrics_list):
        x = m["mean_cum_tracking_error_per_option_sq"]
        y = m["mean_terminal_wealth_per_option"]
        ax.scatter(x, y, s=110, marker=markers[idx % len(markers)], label=m["strategy"])
        ax.annotate(
            m["strategy"],
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(r"Mean cumulative tracking error / $N^2$")
    ax.set_ylabel("Expected terminal wealth per option")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "strategy_tradeoff.png"))
    plt.close(fig)


def plot_qvi_residual(policy, t_grid, save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.semilogy(
        t_grid,
        np.maximum(policy["qvi_residual"], 1e-16),
        linewidth=2.0,
        label="QVI residual",
    )
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Residual")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "qvi_residual.png"))
    plt.close(fig)


def plot_sample_paths(results, strategy_name="Optimal", n_paths=4, save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    t = results["t_path"]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), sharex=True, constrained_layout=True)

    n_show = min(n_paths, results["S_paths"].shape[0])
    labels = [f"Path {i+1}" for i in range(n_show)]

    for i in range(n_show):
        axes[0].plot(t, results["S_paths"][i], linewidth=1.4, alpha=0.95, label=labels[i])
        axes[1].step(t, results["inventory"][i], where="post", linewidth=1.4, alpha=0.95)
        axes[2].plot(t, results["wealth_path"][i], linewidth=1.4, alpha=0.95)
        axes[3].plot(t, results["option_mtm_path"][i], linewidth=1.4, alpha=0.95)

    axes[0].set_ylabel(r"$S_t$")
    axes[1].set_ylabel(r"$q_t$")
    axes[2].set_ylabel(r"Wealth")
    axes[3].set_ylabel("Option MTM")
    axes[3].set_xlabel(r"Time $t$")
    axes[0].legend(loc="upper right", ncol=min(n_show, 2))

    for ax in axes:
        _clean_axis(ax)

    fname = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(os.path.join(save_dir, f"sample_paths_{fname}.png"))
    plt.close(fig)


def plot_terminal_wealth_comparison(results_list: List[Dict], names: List[str], save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for results, name in zip(results_list, names):
        tw = results["terminal_wealth"]
        ax.hist(tw, bins=45, alpha=0.35, density=True, label=name, edgecolor="none")
        ax.axvline(np.mean(tw), linestyle="--", linewidth=1.8)

    ax.set_xlabel("Terminal wealth")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "terminal_wealth_comparison.png"))
    plt.close(fig)


def plot_tracking_error(results_list: List[Dict], names: List[str], save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for results, name in zip(results_list, names):
        t = results["t_path"]
        N = results["model"].num_options
        te = results["tracking_error_sq"] / (N**2)
        mean_te = te.mean(axis=0)
        q10 = np.percentile(te, 10, axis=0)
        q90 = np.percentile(te, 90, axis=0)

        line, = ax.plot(t, mean_te, linewidth=2.0, label=name)
        ax.fill_between(t, q10, q90, alpha=0.14, color=line.get_color())

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Mean tracking error $(q-N\Delta)^2 / N^2$")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "tracking_error.png"))
    plt.close(fig)


def plot_inventory_distribution(results_list: List[Dict], names: List[str], save_dir="plots"):
    setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for results, name in zip(results_list, names):
        t = results["t_path"]
        N = results["model"].num_options
        inv = results["inventory"] / N
        mean_inv = inv.mean(axis=0)
        q10 = np.percentile(inv, 10, axis=0)
        q90 = np.percentile(inv, 90, axis=0)

        line, = ax.plot(t, mean_inv, linewidth=2.0, label=name)
        ax.fill_between(t, q10, q90, alpha=0.14, color=line.get_color())

    ax.axhline(0.0, linestyle=":", color="0.25", linewidth=1.0)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Inventory / $N$")
    ax.legend(loc="best")
    _clean_axis(ax)

    fig.savefig(os.path.join(save_dir, "inventory_distribution.png"))
    plt.close(fig)


def plot_all(
    h,
    policy,
    t_grid,
    S_grid,
    q_values,
    results_opt,
    results_delta,
    results_passive,
    metrics_list,
    save_dir="plots",
    title_suffix="",
):
    os.makedirs(save_dir, exist_ok=True)

    plot_value_function(h, t_grid, S_grid, q_values, save_dir, title_suffix=title_suffix)
    plot_intervention_regions(policy, t_grid, S_grid, q_values, save_dir, title_suffix=title_suffix)
    plot_sample_paths(results_opt, strategy_name="Optimal (HJB-QVI)", save_dir=save_dir)

    results_list = [results_opt, results_delta, results_passive]
    names = ["Optimal (HJB-QVI)", "Pure Delta Hedge", "Pure Passive MM"]

    plot_terminal_wealth_comparison(results_list, names, save_dir)
    plot_tracking_error(results_list, names, save_dir)
    plot_inventory_distribution(results_list, names, save_dir)
    plot_strategy_value_report(metrics_list, save_dir)
    plot_strategy_tradeoff(metrics_list, save_dir)
    plot_qvi_residual(policy, t_grid, save_dir)