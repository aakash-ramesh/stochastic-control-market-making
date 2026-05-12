from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np


@dataclass
class ModelParams:
    # --- Stock dynamics ---
    S0: float = 100.0
    sigma: float = 0.02
    mu: float = 0.0
    r: float = 0.0

    # --- Option / book size ---
    K: float = 100.0
    T: float = 1.0
    num_options: int = 1  # N = number of short call options

    # --- Inventory grid (actual share inventory) ---
    Q_max: float = 1.0
    q_step: float = 0.05

    # Reference controls for the 1-option case
    reference_Q_max: float | None = None
    reference_q_step: float | None = None

    # --- Tracking penalty ---
    eta: float = 0.10

    # --- Same market across all N ---
    Lambda_b: float = 140.0
    Lambda_a: float = 140.0
    k_b: float = 1.5
    k_a: float = 1.5

    # --- Costs / rebates ---
    maker_rebate_per_unit: float = 0.005
    eps_taker: float = 0.03
    half_spread: float = 0.05
    kappa: float = 0.02
    beta: float = 1.5

    # --- Economics ---
    include_initial_premium: bool = True

    def __post_init__(self):
        if self.reference_Q_max is None:
            self.reference_Q_max = self.Q_max
        if self.reference_q_step is None:
            self.reference_q_step = self.q_step

    @property
    def q_values(self) -> np.ndarray:
        n_half = int(round(self.Q_max / self.q_step))
        q = np.linspace(-self.Q_max, self.Q_max, 2 * n_half + 1)
        return np.round(q, 10)

    @property
    def num_q(self) -> int:
        return len(self.q_values)

    @property
    def market_order_grid(self) -> np.ndarray:
        return self.q_values.copy()

    def scaled_for_num_options(self, N: int) -> "ModelParams":
        """
        Same market, bigger short-call book:
          - market environment is unchanged
          - only hedge scale / inventory controls expand with N
        """
        return replace(
            self,
            num_options=N,
            Q_max=self.reference_Q_max * N,
            q_step=self.reference_q_step * N,
            reference_Q_max=self.reference_Q_max,
            reference_q_step=self.reference_q_step,
        )


@dataclass
class GridParams:
    N_t: int = 200
    N_S: int = 121
    n_std: float = 5.0
    delta_min: float = 0.001
    delta_max: float = 2.0
    N_delta: int = 61

    # QVI / fixed-point controls
    qvi_max_iter: int = 200
    qvi_tol: float = 1e-8
    qvi_relaxation: float = 0.70

    @property
    def delta_grid(self) -> np.ndarray:
        return np.linspace(self.delta_min, self.delta_max, self.N_delta)


@dataclass
class SimParams:
    N_paths: int = 600
    seed: int = 42
    dt_sim: float = 0.005
    record_n_sample_paths: int = 5


def build_grids(model: ModelParams, grid: GridParams):
    dt = model.T / grid.N_t
    t_grid = np.linspace(0.0, model.T, grid.N_t + 1)

    y0 = np.log(model.S0)
    y_range = grid.n_std * model.sigma * np.sqrt(model.T)
    y_grid = np.linspace(y0 - y_range, y0 + y_range, grid.N_S)
    dy = y_grid[1] - y_grid[0]
    S_grid = np.exp(y_grid)
    q_grid = model.q_values

    return t_grid, S_grid, y_grid, q_grid, dt, dy


def save_config(model: ModelParams, grid: GridParams, sim: SimParams, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": asdict(model),
        "grid": asdict(grid),
        "sim": asdict(sim),
    }
    (out_path / "run_config.json").write_text(json.dumps(payload, indent=2))