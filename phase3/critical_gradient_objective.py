"""Phase3 critical-gradient objective built on top of DESC's NNITGProxy."""

from __future__ import annotations

import numpy as np

from desc.backend import jax, jnp, root_scalar
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid
from desc.objectives import NNITGProxy


class NNCriticalGradientProxy(NNITGProxy):
    """Optimize a smooth threshold-exceedance surrogate from NNITGProxy scans.

    The forward model still evaluates a fixed ``a/LT`` sweep for each cached flux-tube
    geometry. During optimization, the returned objective is a weighted softplus penalty
    on heat-flux exceedance above the chosen threshold, which is much cheaper to
    differentiate than a smooth first-crossing construction. When
    ``return_critical_gradient=True``, the class instead returns an analysis-only
    estimate of ``a/LT_crit`` based on threshold proximity across the sweep.
    """

    _static_attrs = NNITGProxy._static_attrs + [
        "_a_over_LT_grid",
        "_critical_q_threshold",
        "_critical_log_width",
        "_alpha_aggregation",
    ]

    _print_value_fmt = "NN critical-gradient penalty: "

    def __init__(
        self,
        eq,
        rho=0.5,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        alpha=None,
        npoints=96,
        nz_internal=1001,
        target_flux_tube_length=75.4,
        a_over_Ln=0.9,
        model_path=None,
        ensemble_dir=None,
        ensemble_csv=None,
        n_ensemble=10,
        solve_length_at_compute=False,
        a_over_LT_grid=None,
        a_over_LT_min=0.5,
        a_over_LT_max=10.0,
        n_a_over_LT=20,
        critical_q_threshold=1.0,
        critical_log_width=0.35,
        alpha_aggregation="mean",
        name="NN Critical Gradient Proxy",
        jac_chunk_size=None,
    ):
        if a_over_LT_grid is None:
            a_over_LT_grid = np.linspace(
                float(a_over_LT_min), float(a_over_LT_max), int(n_a_over_LT), dtype=float
            )
        a_over_LT_grid = np.atleast_1d(np.asarray(a_over_LT_grid, dtype=float))
        if a_over_LT_grid.ndim != 1 or a_over_LT_grid.size < 2:
            raise ValueError("a_over_LT_grid must be a 1D array with at least two points")
        if np.any(np.diff(a_over_LT_grid) <= 0):
            raise ValueError("a_over_LT_grid must be strictly increasing")
        if float(critical_q_threshold) <= 0.0:
            raise ValueError("critical_q_threshold must be positive")
        if float(critical_log_width) <= 0.0:
            raise ValueError("critical_log_width must be positive")
        if alpha_aggregation not in {"mean", "worst", "separate"}:
            raise ValueError("alpha_aggregation must be 'mean', 'worst', or 'separate'")

        self._a_over_LT_grid = jnp.asarray(a_over_LT_grid)
        self._critical_q_threshold = float(critical_q_threshold)
        self._critical_log_width = float(critical_log_width)
        self._alpha_aggregation = alpha_aggregation

        super().__init__(
            eq=eq,
            rho=rho,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            alpha=alpha,
            npoints=npoints,
            nz_internal=nz_internal,
            target_flux_tube_length=target_flux_tube_length,
            a_over_LT=float(a_over_LT_grid[0]),
            a_over_Ln=a_over_Ln,
            model_path=model_path,
            ensemble_dir=ensemble_dir,
            ensemble_csv=ensemble_csv,
            n_ensemble=n_ensemble,
            solve_length_at_compute=solve_length_at_compute,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        super().build(use_jit=use_jit, verbose=verbose)
        self._constants["a_over_LT_grid"] = self._a_over_LT_grid
        self._constants["critical_q_threshold"] = self._critical_q_threshold
        self._constants["critical_log_width"] = self._critical_log_width

    def _scaled_log_threshold_offset(self, q_sweep, q_threshold, log_width):
        q_sweep = jnp.clip(q_sweep, 1e-12, None)
        log_q = jnp.log(q_sweep)
        q_threshold = jnp.asarray(q_threshold, dtype=q_sweep.dtype)
        log_width = jnp.asarray(log_width, dtype=q_sweep.dtype)
        log_threshold = jnp.log(q_threshold)
        return (log_q - log_threshold) / log_width

    def _run_cnn_sweep_for_rho(
        self,
        gradpar_2d,
        signals_2d,
        theta_pest_offset,
        a_over_LT_grid,
        a_over_Ln,
        models,
        num_alpha,
    ):
        arclength_2d = compute_arclength_via_gradpar(gradpar_2d, theta_pest_offset)
        _, signals_resampled = resample_to_uniform_arclength(
            arclength_2d, signals_2d, self._npoints
        )
        signals_batch = jnp.transpose(signals_resampled, (2, 0, 1)).astype(jnp.float32)

        n_grid = int(a_over_LT_grid.size)
        tiled_signals = jnp.broadcast_to(
            signals_batch[None, :, :, :],
            (n_grid, num_alpha, signals_batch.shape[1], signals_batch.shape[2]),
        ).reshape(n_grid * num_alpha, signals_batch.shape[1], signals_batch.shape[2])
        a_over_Ln_grid = jnp.broadcast_to(
            jnp.asarray(a_over_Ln, dtype=a_over_LT_grid.dtype),
            a_over_LT_grid.shape,
        )
        scalar_grid = jnp.column_stack([a_over_LT_grid, a_over_Ln_grid]).astype(
            jnp.float32
        )
        tiled_scalars = jnp.broadcast_to(
            scalar_grid[:, None, :], (n_grid, num_alpha, 2)
        ).reshape(n_grid * num_alpha, 2)

        predictions = [f(tiled_signals, tiled_scalars) for f in models]
        stacked = jnp.stack(predictions)
        log_q = jnp.mean(stacked, axis=0)[:, 0].reshape(n_grid, num_alpha)
        q_sweep = jnp.exp(log_q)
        return q_sweep, signals_batch

    def _optimization_penalty_from_sweep(
        self, q_sweep, a_over_LT_grid, q_threshold, log_width
    ):
        scaled = self._scaled_log_threshold_offset(q_sweep, q_threshold, log_width)
        a_over_LT_grid = jnp.asarray(a_over_LT_grid, dtype=scaled.dtype)
        eps = jnp.asarray(jnp.finfo(scaled.dtype).eps, dtype=scaled.dtype)
        span = jnp.maximum(a_over_LT_grid[-1] - a_over_LT_grid[0], eps)
        early_weight = (a_over_LT_grid[-1] - a_over_LT_grid) / span
        denom = jnp.maximum(jnp.sum(early_weight), eps)
        weighted_exceedance = jax.nn.softplus(scaled) * early_weight[:, None]
        return jnp.sum(weighted_exceedance, axis=0) / denom

    def _critical_gradient_from_sweep(self, q_sweep, a_over_LT_grid, q_threshold, log_width):
        # Keep the analysis-only critical-gradient estimate out of the reverse pass.
        scaled = jax.lax.stop_gradient(
            self._scaled_log_threshold_offset(q_sweep, q_threshold, log_width)
        )
        a_over_LT_grid = jnp.asarray(a_over_LT_grid, dtype=scaled.dtype)
        proximity = jax.nn.softmax(-jnp.square(scaled), axis=0)
        return jnp.sum(proximity * a_over_LT_grid[:, None], axis=0)

    def _aggregate_alpha(self, critical_per_alpha, alpha_aggregation):
        if alpha_aggregation == "mean":
            return jnp.mean(critical_per_alpha)
        if alpha_aggregation == "worst":
            return jnp.min(critical_per_alpha)
        if alpha_aggregation == "separate":
            return critical_per_alpha
        raise ValueError(f"Unsupported alpha aggregation: {alpha_aggregation}")

    def _format_output(
        self,
        objective_all,
        critical_all,
        q_sweeps,
        all_signals,
        return_critical_gradient,
        return_q_sweep,
        return_signals,
        alpha_aggregation,
    ):
        if alpha_aggregation == "separate":
            objective_stacked = jnp.stack(objective_all)
            critical_stacked = None if critical_all is None else jnp.stack(critical_all)
        else:
            objective_stacked = jnp.atleast_1d(jnp.stack(objective_all))
            critical_stacked = (
                None
                if critical_all is None
                else jnp.atleast_1d(jnp.stack(critical_all))
            )

        value = critical_stacked if return_critical_gradient else objective_stacked

        result = [value]
        if return_q_sweep:
            result.append(jnp.stack(q_sweeps))
        if return_signals:
            result.append(jnp.stack(all_signals))
        if len(result) == 1:
            return result[0]
        return tuple(result)

    def compute(
        self,
        params,
        constants=None,
        return_critical_gradient=False,
        return_q_sweep=False,
        return_signals=False,
    ):
        if constants is None:
            constants = self.constants

        eq = self.things[0]
        rho_arr = jnp.asarray(constants["rho"])
        alpha_arr = jnp.asarray(constants["alpha"])
        toroidal_turns = constants["toroidal_turns"]
        minor_radius = constants["a"]
        a_over_LT_grid = jnp.asarray(constants["a_over_LT_grid"])
        a_over_Ln = constants["a_over_Ln"]
        q_threshold = constants["critical_q_threshold"]
        log_width = constants["critical_log_width"]
        alpha_aggregation = self._alpha_aggregation
        models = self._models

        num_rho = len(rho_arr)
        num_alpha = len(alpha_arr)
        nz = self._nz_internal

        iota_grid = constants["iota_transforms"]["grid"]
        iota_data = compute_fun(
            eq,
            ["iota", "iota_r"],
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_arr = iota_grid.compress(iota_data["iota"])
        iota_r_arr = iota_grid.compress(iota_data["iota_r"])

        gx_keys = [
            "gx_bmag",
            "gx_gbdrift",
            "gx_cvdrift",
            "gx_gbdrift0_over_shat",
            "gx_gds2",
            "gx_gds21_over_shat",
            "gx_gds22_over_shat_squared",
        ]

        if self._solve_length_at_compute:
            return self._compute_with_length_solving(
                params,
                eq,
                rho_arr,
                alpha_arr,
                iota_arr,
                iota_r_arr,
                minor_radius,
                a_over_LT_grid,
                a_over_Ln,
                q_threshold,
                log_width,
                alpha_aggregation,
                models,
                gx_keys,
                constants,
                return_critical_gradient,
                return_q_sweep,
                return_signals,
            )

        zeta_offset = jnp.linspace(-jnp.pi * toroidal_turns, jnp.pi * toroidal_turns, nz)
        rho_mesh = jnp.broadcast_to(rho_arr[None, :, None], (nz, num_rho, num_alpha))
        alpha_mesh = jnp.broadcast_to(alpha_arr[None, None, :], (nz, num_rho, num_alpha))
        zeta_mesh = jnp.broadcast_to(zeta_offset[:, None, None], (nz, num_rho, num_alpha))
        iota_mesh = jnp.broadcast_to(iota_arr[None, :, None], (nz, num_rho, num_alpha))
        iota_r_mesh = jnp.broadcast_to(iota_r_arr[None, :, None], (nz, num_rho, num_alpha))
        theta_pest_mesh = alpha_mesh + iota_mesh * zeta_mesh

        rho_flat = rho_mesh.flatten()
        theta_pest_flat = theta_pest_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()
        iota_flat = iota_mesh.flatten()
        iota_r_flat = iota_r_mesh.flatten()

        theta_desc_flat = self._map_pest_to_desc_coords(
            eq, params, rho_flat, theta_pest_flat, zeta_flat
        )
        gx_data = self._compute_gx_features_on_grid(
            eq,
            params,
            rho_flat,
            theta_desc_flat,
            zeta_flat,
            iota_flat,
            iota_r_flat,
            minor_radius,
            gx_keys,
        )

        def reshape_gx(arr):
            return arr.reshape(nz, num_rho, num_alpha)

        gx_features = {k: reshape_gx(gx_data[k]) for k in gx_keys}
        gradpar_3d = reshape_gx(gx_data["gx_gradpar"])

        objective_all = []
        critical_all = [] if return_critical_gradient else None
        q_sweeps = [] if return_q_sweep else None
        all_signals = [] if return_signals else None

        for i_rho in range(num_rho):
            gradpar_2d = gradpar_3d[:, i_rho, :]
            signals_2d = jnp.stack([gx_features[k][:, i_rho, :] for k in gx_keys])

            iota_sign = jnp.sign(iota_arr[i_rho])
            signals_2d = signals_2d.at[3].multiply(iota_sign)
            signals_2d = signals_2d.at[5].multiply(iota_sign)

            theta_pest_offset_rho = iota_arr[i_rho] * zeta_offset
            q_sweep, signals_batch = self._run_cnn_sweep_for_rho(
                gradpar_2d,
                signals_2d,
                theta_pest_offset_rho,
                a_over_LT_grid,
                a_over_Ln,
                models,
                num_alpha,
            )
            objective_per_alpha = self._optimization_penalty_from_sweep(
                q_sweep,
                a_over_LT_grid,
                q_threshold,
                log_width,
            )
            objective_all.append(self._aggregate_alpha(objective_per_alpha, alpha_aggregation))
            if return_critical_gradient:
                critical_per_alpha = self._critical_gradient_from_sweep(
                    q_sweep,
                    a_over_LT_grid,
                    q_threshold,
                    log_width,
                )
                critical_all.append(
                    self._aggregate_alpha(critical_per_alpha, alpha_aggregation)
                )
            if return_q_sweep:
                q_sweeps.append(q_sweep)
            if return_signals:
                all_signals.append(signals_batch)

        return self._format_output(
            objective_all,
            critical_all,
            q_sweeps,
            all_signals,
            return_critical_gradient,
            return_q_sweep,
            return_signals,
            alpha_aggregation,
        )

    def _compute_with_length_solving(
        self,
        params,
        eq,
        rho_arr,
        alpha_arr,
        iota_arr,
        iota_r_arr,
        minor_radius,
        a_over_LT_grid,
        a_over_Ln,
        q_threshold,
        log_width,
        alpha_aggregation,
        models,
        gx_keys,
        constants,
        return_critical_gradient,
        return_q_sweep,
        return_signals,
    ):
        num_rho = len(rho_arr)
        num_alpha = len(alpha_arr)
        nz = self._nz_internal
        target_length = constants["target_length"]
        toroidal_turns = constants["toroidal_turns"]

        objective_all = []
        critical_all = [] if return_critical_gradient else None
        q_sweeps = [] if return_q_sweep else None
        all_signals = [] if return_signals else None

        for i_rho in range(num_rho):
            rho = rho_arr[i_rho]
            iota = iota_arr[i_rho]
            iota_r = iota_r_arr[i_rho]

            x0 = toroidal_turns * jnp.abs(iota)

            def length_residual(poloidal_turns):
                theta_pest_offset = jnp.linspace(
                    -jnp.pi * poloidal_turns, jnp.pi * poloidal_turns, nz
                )
                theta_pest = theta_pest_offset
                zeta = theta_pest_offset / iota
                rho_grid = jnp.full(nz, rho)

                theta_desc = self._map_pest_to_desc_coords(
                    eq, params, rho_grid, theta_pest, zeta
                )
                grid = Grid(
                    jnp.column_stack([rho_grid, theta_desc, zeta]),
                    sort=False,
                    jitable=True,
                )
                data = compute_fun(
                    eq,
                    ["gx_gradpar"],
                    params,
                    get_transforms(["gx_gradpar"], eq, grid, jitable=True),
                    get_profiles(["gx_gradpar"], eq, grid),
                    data={
                        "a": minor_radius,
                        "iota": jnp.full(nz, iota),
                        "iota_r": jnp.full(nz, iota_r),
                        "rho": rho_grid,
                    },
                )
                length = jnp.abs(jnp.trapezoid(1.0 / data["gx_gradpar"], theta_pest))
                return length - target_length

            # The length solve is a geometry setup step, not a quantity we want to
            # differentiate through during optimization.
            poloidal_turns = jax.lax.stop_gradient(
                root_scalar(length_residual, x0, tol=1e-4, maxiter=10)
            )

            theta_pest_offset = jnp.linspace(
                -jnp.pi * poloidal_turns, jnp.pi * poloidal_turns, nz
            )
            alpha_mesh = jnp.broadcast_to(alpha_arr[None, :], (nz, num_alpha))
            theta_offset_mesh = jnp.broadcast_to(theta_pest_offset[:, None], (nz, num_alpha))
            theta_pest_mesh = alpha_mesh + theta_offset_mesh
            zeta_mesh = theta_offset_mesh / iota

            rho_flat = jnp.full(nz * num_alpha, rho)
            theta_pest_flat = theta_pest_mesh.flatten()
            zeta_flat = zeta_mesh.flatten()
            iota_flat = jnp.full(nz * num_alpha, iota)
            iota_r_flat = jnp.full(nz * num_alpha, iota_r)

            theta_desc_flat = self._map_pest_to_desc_coords(
                eq, params, rho_flat, theta_pest_flat, zeta_flat
            )
            gx_data = self._compute_gx_features_on_grid(
                eq,
                params,
                rho_flat,
                theta_desc_flat,
                zeta_flat,
                iota_flat,
                iota_r_flat,
                minor_radius,
                gx_keys,
            )

            gradpar_2d = gx_data["gx_gradpar"].reshape(nz, num_alpha)
            signals_2d = jnp.stack([gx_data[k].reshape(nz, num_alpha) for k in gx_keys])
            q_sweep, signals_batch = self._run_cnn_sweep_for_rho(
                gradpar_2d,
                signals_2d,
                theta_pest_offset,
                a_over_LT_grid,
                a_over_Ln,
                models,
                num_alpha,
            )
            objective_per_alpha = self._optimization_penalty_from_sweep(
                q_sweep,
                a_over_LT_grid,
                q_threshold,
                log_width,
            )
            objective_all.append(self._aggregate_alpha(objective_per_alpha, alpha_aggregation))
            if return_critical_gradient:
                critical_per_alpha = self._critical_gradient_from_sweep(
                    q_sweep,
                    a_over_LT_grid,
                    q_threshold,
                    log_width,
                )
                critical_all.append(
                    self._aggregate_alpha(critical_per_alpha, alpha_aggregation)
                )
            if return_q_sweep:
                q_sweeps.append(q_sweep)
            if return_signals:
                all_signals.append(signals_batch)

        return self._format_output(
            objective_all,
            critical_all,
            q_sweeps,
            all_signals,
            return_critical_gradient,
            return_q_sweep,
            return_signals,
            alpha_aggregation,
        )


# Reuse the tested DESC turbulence helpers without changing the DESC package.
from desc.compute._turbulence import (  # noqa: E402
    compute_arclength_via_gradpar,
    resample_to_uniform_arclength,
)
