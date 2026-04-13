"""Timing smoke test for one phase-3 critical-gradient objective evaluation."""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PHASE3_DIR = Path(__file__).resolve().parent
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

from desc.backend import jax
from desc.compat import rescale
from desc.examples import get
from desc.grid import QuadratureGrid
from desc.optimize.optimizer import get_combined_constraint_objectives

from critical_gradient_objective import NNCriticalGradientProxy
from qs_critical_gradient_opt import (
    OptimizationConfig,
    ProjectPaths,
    build_constraints,
    build_objective,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "data" / "neural_networks" / "cyclic_invariant_models"
MODEL_CSV = REPO_ROOT / "data" / "neural_networks" / "deephyper_exp" / "results.csv"

# Keep these aligned with the current phase3 optimization defaults.
DEFAULT_RHO = np.array(
    [0.041, 0.124, 0.206, 0.288, 0.371, 0.453, 0.535, 0.618, 0.700],
    dtype=float,
)
DEFAULT_ALPHA = np.array([0.0], dtype=float)
DEFAULT_A_OVER_LT_GRID = np.linspace(0.25, 4.25, 8, dtype=float)


def _build_proxy(
    rho: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    a_over_LT_grid: np.ndarray | None = None,
) -> NNCriticalGradientProxy:
    rho = DEFAULT_RHO if rho is None else np.asarray(rho, dtype=float)
    alpha = DEFAULT_ALPHA if alpha is None else np.asarray(alpha, dtype=float)
    a_over_LT_grid = (
        DEFAULT_A_OVER_LT_GRID
        if a_over_LT_grid is None
        else np.asarray(a_over_LT_grid, dtype=float)
    )
    eq = get("precise_QA")
    eq = rescale(eq, ("a", 1.0), verbose=False)
    proxy = NNCriticalGradientProxy(
        eq=eq,
        rho=rho,
        alpha=alpha,
        ensemble_dir=str(MODEL_DIR),
        ensemble_csv=str(MODEL_CSV),
        n_ensemble=1,
        target_flux_tube_length=75.4,
        a_over_Ln=0.9,
        a_over_LT_grid=a_over_LT_grid,
        critical_q_threshold=1.0,
        critical_log_width=0.35,
        alpha_aggregation="mean",
    )
    proxy.build(use_jit=True, verbose=0)
    return proxy


def _build_wrapped_objective(config: OptimizationConfig | None = None):
    config = OptimizationConfig() if config is None else config
    paths = ProjectPaths.from_script(str(PHASE3_DIR / "qs_critical_gradient_opt.py"))
    eq = get(config.example_name)
    eq = rescale(eq, ("a", 1.0), verbose=False)
    qs_grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)
    objective = build_objective(eq, qs_grid, config, paths)
    constraints = build_constraints(eq, config.mode_steps[0])
    options = copy.deepcopy(config.optimize_options)
    wrapped_objective, nonlinear_constraint = get_combined_constraint_objectives(
        eq=eq,
        constraints=constraints,
        objective=objective,
        things=[eq],
        opt_method=config.optimizer_name,
        method=config.optimizer_name.removeprefix("proximal-"),
        verbose=0,
        options=options,
    )
    assert nonlinear_constraint is None
    return wrapped_objective, eq


def test_phase3_objective_single_evaluation_timing():
    """Time one real phase3 objective evaluation after JIT warmup."""
    pytest.importorskip("torch")
    if not MODEL_DIR.exists():
        pytest.skip(f"Missing model directory: {MODEL_DIR}")
    if not MODEL_CSV.exists():
        pytest.skip(f"Missing model csv: {MODEL_CSV}")

    proxy = _build_proxy()
    x = proxy.things[0].pack_params(proxy.things[0].params_dict)

    start = time.perf_counter()
    result = proxy.compute_scaled_error(x, constants=proxy.constants)
    result.block_until_ready()
    first_eval = np.asarray(result)
    first_eval_s = time.perf_counter() - start

    timings = []
    steady_state = None
    for _ in range(3):
        start = time.perf_counter()
        result = proxy.compute_scaled_error(x, constants=proxy.constants)
        result.block_until_ready()
        steady_state = np.asarray(result)
        timings.append(time.perf_counter() - start)

    assert steady_state is not None
    assert first_eval.shape == (DEFAULT_RHO.size,)
    assert steady_state.shape == (DEFAULT_RHO.size,)
    assert np.isfinite(first_eval).all()
    assert np.isfinite(steady_state).all()
    np.testing.assert_allclose(steady_state, first_eval, rtol=1e-12, atol=1e-12)

    device = jax.devices()[0]
    median_eval_s = float(np.median(timings))
    print(
        "\nphase3 objective timing: "
        f"device={device.platform}, "
        f"rho={DEFAULT_RHO.size}, "
        f"alpha={DEFAULT_ALPHA.size}, "
        f"a_over_LT_grid={DEFAULT_A_OVER_LT_GRID.size}, "
        f"first_eval_s={first_eval_s:.3f}, "
        f"steady_state_eval_s={median_eval_s:.3f}, "
        f"rounds={timings!r}"
    )


def test_phase3_analysis_critical_gradient_bounds():
    """Check that the analysis-only critical-gradient estimate stays on the scan grid."""
    pytest.importorskip("torch")
    if not MODEL_DIR.exists():
        pytest.skip(f"Missing model directory: {MODEL_DIR}")
    if not MODEL_CSV.exists():
        pytest.skip(f"Missing model csv: {MODEL_CSV}")

    rho = np.array([np.sqrt(0.5)], dtype=float)
    alpha = np.array([0.0], dtype=float)
    a_over_LT_grid = np.linspace(0.5, 3.5, 4, dtype=float)
    proxy = _build_proxy(rho=rho, alpha=alpha, a_over_LT_grid=a_over_LT_grid)
    critical = proxy.compute(
        proxy.things[0].params_dict,
        constants=proxy.constants,
        return_critical_gradient=True,
    )
    critical.block_until_ready()
    critical = np.asarray(critical)

    assert critical.shape == (rho.size,)
    assert np.isfinite(critical).all()
    assert np.all(critical >= a_over_LT_grid[0] - 1e-8)
    assert np.all(critical <= a_over_LT_grid[-1] + 1e-8)


def test_phase3_wrapped_objective_gradient_timing():
    """Time the heavy wrapped scalar objective and gradient used by the optimizer."""
    pytest.importorskip("torch")
    if not MODEL_DIR.exists():
        pytest.skip(f"Missing model directory: {MODEL_DIR}")
    if not MODEL_CSV.exists():
        pytest.skip(f"Missing model csv: {MODEL_CSV}")
    if os.getenv("RUN_PHASE3_WRAPPED_OBJECTIVE_TIMING") != "1":
        pytest.skip(
            "Set RUN_PHASE3_WRAPPED_OBJECTIVE_TIMING=1 to run the wrapped-objective timing diagnostic."
        )

    print("\nphase3 wrapped timing: building reduced objective", flush=True)
    objective, eq = _build_wrapped_objective(
        OptimizationConfig(
            rho=np.array([np.sqrt(0.5)], dtype=float),
            alpha=np.array([0.0], dtype=float),
            a_over_LT_grid=np.linspace(0.5, 3.5, 4, dtype=float),
            optimizer_name="proximal-fmintr-bfgs",
        )
    )
    x = objective.x(eq)

    print("phase3 wrapped timing: scalar compile/eval", flush=True)
    start = time.perf_counter()
    scalar = objective.compute_scalar(x, constants=objective.constants)
    scalar.block_until_ready()
    scalar_compile_eval_s = time.perf_counter() - start

    print("phase3 wrapped timing: scalar steady-state", flush=True)
    start = time.perf_counter()
    scalar_steady = objective.compute_scalar(x, constants=objective.constants)
    scalar_steady.block_until_ready()
    scalar_steady_s = time.perf_counter() - start

    print("phase3 wrapped timing: grad compile/eval", flush=True)
    start = time.perf_counter()
    grad = objective.grad(x, constants=objective.constants)
    grad.block_until_ready()
    grad_compile_eval_s = time.perf_counter() - start

    print("phase3 wrapped timing: grad steady-state", flush=True)
    start = time.perf_counter()
    grad_steady = objective.grad(x, constants=objective.constants)
    grad_steady.block_until_ready()
    grad_steady_s = time.perf_counter() - start

    scalar_value = float(np.asarray(scalar))
    scalar_steady_value = float(np.asarray(scalar_steady))
    grad_value = np.asarray(grad)
    grad_steady_value = np.asarray(grad_steady)

    assert np.isfinite(scalar_value)
    assert np.isfinite(scalar_steady_value)
    assert np.isfinite(grad_value).all()
    assert np.isfinite(grad_steady_value).all()
    np.testing.assert_allclose(scalar_steady_value, scalar_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        grad_steady_value,
        grad_value,
        rtol=1e-12,
        atol=1e-12,
    )

    device = jax.devices()[0]
    print(
        "\nphase3 wrapped timing: "
        f"device={device.platform}, "
        f"method=proximal-fmintr-bfgs, "
        "surrogate=weighted_softplus_threshold_penalty, "
        f"scalar_compile_eval_s={scalar_compile_eval_s:.3f}, "
        f"scalar_steady_s={scalar_steady_s:.3f}, "
        f"grad_compile_eval_s={grad_compile_eval_s:.3f}, "
        f"grad_steady_s={grad_steady_s:.3f}, "
        f"dim_x={x.size}, "
        f"dim_f={objective.dim_f}"
    )
