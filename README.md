# Updated HJB-QVI Project Code

This version fixes the main issues discussed:

1. **Crank–Nicolson diffusion step** in the PDE solver.
2. **QVI fixed-point / impulse projection** instead of applying intervention to a pre-control intermediate object once.
3. **Interim wealth marked with Black–Scholes option price** rather than intrinsic value before maturity.
4. **Optional initial premium included** for the short option.
5. **Improved boundary handling** using one-sided second-order stencils.
6. **Nonlinear temporary market-order impact** via `kappa * |xi|**beta` with no direct change to the stock state.
7. **Passive fill size `q_step`** for gradual hedging.

## Main files
- `config.py`
- `pde_solver.py`
- `simulator.py`
- `metrics.py`
- `visualization.py`
- `main.py`

## Notes
- Theory remains for one short option.
- Inventory is on a discrete grid with spacing `q_step`.
- To scale simulation to `n` sold options, multiply the option position and hedge target consistently in the simulation layer.
