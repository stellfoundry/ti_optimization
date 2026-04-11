"""Postprocessing utilities for QS + turbulence optimization runs."""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge, proton_mass

from desc.objectives import NNITGProxy
from desc.plotting import plot_comparison
from transport import (
    StellaratorPowerBalanceSolver,
    TransportConfig,
    example_density_profile,
    example_heating_source,
)


def _log(message: str) -> None:
    """Emit a UTC-timestamped log line."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class ProxyConfig:
    """Configuration for NNITGProxy-based postprocessing."""

    ensemble_dir: str
    ensemble_csv: str
    n_ensemble: int = 3
    target_flux_tube_length: float = 75.4
    a_over_LT: float = 3.0
    a_over_Ln: float = 1.0


@dataclass
class TransportEstimateConfig:
    """Configuration for the steady-state ion-temperature estimate."""

    rho_inner: float = 0.05
    rho_outer: float = 0.90
    nr: int = 16
    n_alpha: int = 8
    M_grid: int | None = None
    N_grid: int | None = None
    max_iter: int = 30
    verbose: bool = True
    edge_temperature: float = 1.0
    initial_core_temperature: float = 8.0
    initial_profile_shape: float = 0.85
    temperature_unit_eV: float = 1.0e3
    ion_mass_number: float = 2.0
    ion_charge_state: float = 1.0
    flux_scale_factor: float = 1.0
    minimum_heat_flux_scale: float = 1.0e-8

    def to_transport_config(self) -> TransportConfig:
        return TransportConfig(
            rho_inner=self.rho_inner,
            rho_outer=self.rho_outer,
            nr=self.nr,
            n_alpha=self.n_alpha,
            M_grid=self.M_grid,
            N_grid=self.N_grid,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )


class _CachedNNITGTransportModel:
    """Reuse cached NNITGProxy geometry while varying profile gradients."""

    def __init__(
        self,
        eq,
        rho_values,
        alpha_values,
        proxy_config: ProxyConfig,
        transport_config: TransportEstimateConfig,
        *,
        n_ensemble: int | None = None,
    ):
        self.eq = eq
        self.rho_values = np.atleast_1d(np.asarray(rho_values, dtype=float))
        self.alpha_values = np.atleast_1d(np.asarray(alpha_values, dtype=float))
        self.proxy_config = proxy_config
        self.verbose = bool(transport_config.verbose)
        self.n_ensemble = proxy_config.n_ensemble if n_ensemble is None else n_ensemble
        self.proxy = NNITGProxy(
            eq=eq,
            rho=self.rho_values[:1],
            alpha=self.alpha_values[:1],
            ensemble_dir=proxy_config.ensemble_dir,
            ensemble_csv=proxy_config.ensemble_csv,
            n_ensemble=self.n_ensemble,
            target_flux_tube_length=proxy_config.target_flux_tube_length,
            a_over_LT=proxy_config.a_over_LT,
            a_over_Ln=proxy_config.a_over_Ln,
        )
        self.proxy.build(use_jit=False, verbose=0)
        ref_data = eq.compute(["gx_B_reference", "gx_L_reference"])
        self.B_ref = float(np.asarray(ref_data["gx_B_reference"], dtype=float).reshape(-1)[0])
        self.L_ref = float(np.asarray(ref_data["gx_L_reference"], dtype=float).reshape(-1)[0])
        self.temperature_unit_eV = float(transport_config.temperature_unit_eV)
        self.ion_mass = float(transport_config.ion_mass_number) * proton_mass
        self.ion_charge = float(transport_config.ion_charge_state)
        self.flux_scale_factor = float(transport_config.flux_scale_factor)
        self.minimum_heat_flux_scale = float(transport_config.minimum_heat_flux_scale)
        self.models = tuple(self.proxy._models)
        self._signal_cache: dict[tuple[float, int, str], jnp.ndarray] = {}

    @staticmethod
    def _predict_for_signals(models, signals_batch, a_over_LT: float, a_over_Ln: float) -> np.ndarray:
        a_over_LT = float(np.clip(a_over_LT, 0.0, 20.0))
        a_over_Ln = float(np.clip(a_over_Ln, -20.0, 20.0))
        scalars_batch = jnp.broadcast_to(
            jnp.array([[a_over_LT, a_over_Ln]], dtype=jnp.float32),
            (signals_batch.shape[0], 2),
        )
        stacked = jnp.stack([model(signals_batch, scalars_batch) for model in models])
        log_q = jnp.mean(stacked, axis=0)
        q = np.asarray(jnp.exp(log_q[:, 0]), dtype=float)
        return np.nan_to_num(q, nan=1e12, posinf=1e12, neginf=0.0)

    def _gyrobohm_heat_flux_scale(self, Ti: float, ni: float) -> float:
        """Approximate physical heat-flux scale from GX's gyro-Bohm normalization."""
        temperature_joules = max(float(Ti), 1e-8) * self.temperature_unit_eV * elementary_charge
        density = max(float(ni), 1e-30)
        B_ref = max(abs(self.B_ref), 1e-12)
        L_ref = max(abs(self.L_ref), 1e-12)
        ion_mass = max(self.ion_mass, 1e-30)
        ion_charge = max(abs(self.ion_charge), 1e-12)

        thermal_speed = np.sqrt(2.0 * temperature_joules / ion_mass)
        cyclotron_frequency = ion_charge * elementary_charge * B_ref / ion_mass
        gyroradius = thermal_speed / max(cyclotron_frequency, 1e-30)
        gyrobohm_flux = density * temperature_joules * thermal_speed * (gyroradius / L_ref) ** 2
        return max(self.flux_scale_factor * gyrobohm_flux, self.minimum_heat_flux_scale)

    @staticmethod
    def _cache_key(rho_value: float, alpha_values: np.ndarray) -> tuple[float, int, str]:
        alpha_values = np.mod(np.asarray(alpha_values, dtype=float).reshape(-1), 2 * np.pi)
        alpha_bytes = np.round(alpha_values, decimals=12).view(np.uint8)
        alpha_hash = hashlib.blake2b(alpha_bytes, digest_size=16).hexdigest()
        return (round(float(rho_value), 12), alpha_values.size, alpha_hash)

    @staticmethod
    def _constant_surface_value(values, name: str, rho_value: float) -> float:
        values = np.asarray(values, dtype=float).reshape(-1)
        ref = float(values[0])
        if not np.allclose(values, ref, rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"{name} must be constant on each rho surface for rho={rho_value}; "
                f"got min={values.min()} max={values.max()}"
            )
        return ref

    def _signals_for_surface(self, rho_value: float, alpha_values: np.ndarray) -> jnp.ndarray:
        alpha_values = np.mod(np.asarray(alpha_values, dtype=float).reshape(-1), 2 * np.pi)
        cache_key = self._cache_key(rho_value, alpha_values)
        if cache_key not in self._signal_cache:
            if self.verbose:
                _log(
                    "[transport] cache miss:"
                    f" rho={rho_value:.6f}, n_alpha_nodes={alpha_values.size};"
                    " building NNITGProxy"
                )
            proxy = NNITGProxy(
                eq=self.eq,
                rho=np.array([float(rho_value)], dtype=float),
                alpha=alpha_values,
                ensemble_dir=self.proxy_config.ensemble_dir,
                ensemble_csv=self.proxy_config.ensemble_csv,
                n_ensemble=self.n_ensemble,
                target_flux_tube_length=self.proxy_config.target_flux_tube_length,
                a_over_LT=self.proxy_config.a_over_LT,
                a_over_Ln=self.proxy_config.a_over_Ln,
            )
            proxy.build(use_jit=False, verbose=0)
            if self.verbose:
                _log(
                    f"[transport] rho={rho_value:.6f}: computing GX geometry/signals"
                )
            _, signals_info = proxy.compute(
                self.eq.params_dict,
                return_signals=True,
                return_per_alpha=True,
            )
            self._signal_cache[cache_key] = jnp.asarray(
                signals_info["signals"][0],
                dtype=jnp.float32,
            )
            if self.verbose:
                _log(
                    "[transport] cached signals:"
                    f" rho={rho_value:.6f}, signals_shape={self._signal_cache[cache_key].shape}"
                )
        return self._signal_cache[cache_key]

    def __call__(self, rho, alpha, aLTi, aLni, Ti, ni, extra):
        del extra
        rho_arr, alpha_arr, aLTi_arr, aLni_arr, Ti_arr, ni_arr = np.broadcast_arrays(
            np.asarray(rho, dtype=float),
            np.asarray(alpha, dtype=float),
            np.asarray(aLTi, dtype=float),
            np.asarray(aLni, dtype=float),
            np.asarray(Ti, dtype=float),
            np.asarray(ni, dtype=float),
        )
        original_shape = rho_arr.shape
        rho_flat = rho_arr.reshape(-1)
        alpha_flat = alpha_arr.reshape(-1)
        aLTi_flat = aLTi_arr.reshape(-1)
        aLni_flat = aLni_arr.reshape(-1)
        Ti_flat = Ti_arr.reshape(-1)
        ni_flat = ni_arr.reshape(-1)
        q_out = np.empty_like(rho_flat, dtype=float)
        unique_rho = np.unique(rho_flat)

        if self.verbose:
            _log(
                "[transport] evaluating NN transport model:"
                f" n_nodes={rho_flat.size}, n_rho_surfaces={unique_rho.size}"
            )

        for rho_value in unique_rho:
            idx = np.where(np.isclose(rho_flat, rho_value, atol=1e-12, rtol=0.0))[0]
            cached_rho = self.rho_values[np.argmin(np.abs(self.rho_values - float(rho_value)))]
            if not np.isclose(cached_rho, rho_value, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "Transport NN adapter was called at a rho value that was not cached: "
                    f"{rho_value}"
                )
            signals = self._signals_for_surface(float(rho_value), alpha_flat[idx])
            aLTi_value = self._constant_surface_value(aLTi_flat[idx], "aLTi", rho_value)
            aLni_value = self._constant_surface_value(aLni_flat[idx], "aLni", rho_value)
            Ti_value = self._constant_surface_value(Ti_flat[idx], "Ti", rho_value)
            ni_value = self._constant_surface_value(ni_flat[idx], "ni", rho_value)
            q_normalized = self._predict_for_signals(
                self.models,
                signals,
                aLTi_value,
                aLni_value,
            )
            q_scale = self._gyrobohm_heat_flux_scale(Ti_value, ni_value)
            q_out[idx] = q_normalized * q_scale

        if self.verbose:
            _log(
                "[transport] completed NN transport model evaluation:"
                f" q_min={np.nanmin(q_out):.6e}, q_max={np.nanmax(q_out):.6e}"
            )

        q_physical = np.nan_to_num(q_out.reshape(original_shape), nan=1e12, posinf=1e12, neginf=0.0)
        if q_physical.shape == ():
            return float(q_physical)
        return q_physical


class TurbulencePostprocessor:
    """Generate standard turbulence postprocessing plots for equilibria."""

    def __init__(self, output_dir: str, proxy_config: ProxyConfig):
        self.output_dir = output_dir
        self.proxy_config = proxy_config
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _as_1d(values) -> np.ndarray:
        return np.atleast_1d(np.asarray(values, dtype=float))

    def _build_proxy(
        self,
        eq,
        rho,
        alpha,
        *,
        a_over_LT: float | None = None,
        a_over_Ln: float | None = None,
        n_ensemble: int | None = None,
    ) -> NNITGProxy:
        obj = NNITGProxy(
            eq=eq,
            rho=self._as_1d(rho),
            alpha=self._as_1d(alpha),
            ensemble_dir=self.proxy_config.ensemble_dir,
            ensemble_csv=self.proxy_config.ensemble_csv,
            n_ensemble=self.proxy_config.n_ensemble if n_ensemble is None else n_ensemble,
            target_flux_tube_length=self.proxy_config.target_flux_tube_length,
            a_over_LT=self.proxy_config.a_over_LT if a_over_LT is None else a_over_LT,
            a_over_Ln=self.proxy_config.a_over_Ln if a_over_Ln is None else a_over_Ln,
        )
        obj.build(use_jit=False, verbose=0)
        return obj

    @staticmethod
    def _transport_initial_guess(
        rho_centers: np.ndarray,
        transport_config: TransportEstimateConfig,
    ) -> np.ndarray:
        return (
            transport_config.initial_core_temperature
            * (1.0 - transport_config.initial_profile_shape * rho_centers**2)
            + transport_config.edge_temperature
        )

    @staticmethod
    def _diagnostic_label(label: str) -> str:
        cleaned = "".join(char.lower() if char.isalnum() else "_" for char in label)
        return cleaned.strip("_") or "case"

    def _save_transport_diagnostics(
        self,
        label: str,
        result: dict,
        filename: str,
    ) -> str:
        stem, _ = os.path.splitext(filename)
        diagnostic_path = os.path.join(
            self.output_dir,
            f"{stem}_{self._diagnostic_label(label)}_diagnostics.npz",
        )
        payload = {}
        for key, value in result.items():
            if isinstance(value, str):
                continue
            payload[key] = np.asarray(value)
        np.savez(diagnostic_path, **payload)
        return diagnostic_path

    def plot_flux_surface_comparison(
        self,
        equilibria: Sequence,
        labels: Sequence[str],
        filename: str = "03_flux_surface_comparison.png",
    ) -> str:
        fig, _ = plot_comparison(list(equilibria), labels=list(labels))
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def evaluate_multi_alpha(self, eq, rho, alpha_values, *, n_ensemble: int | None = None):
        obj = self._build_proxy(eq, rho, alpha_values, n_ensemble=n_ensemble)
        q, q_std = obj.compute(eq.params_dict, return_per_alpha=True, return_std=True)
        return np.asarray(q, dtype=float), np.asarray(q_std, dtype=float)

    def plot_multi_alpha_comparison(
        self,
        equilibria: Sequence,
        labels: Sequence[str],
        *,
        rho,
        alpha_values,
        n_ensemble: int | None = None,
        filename: str = "04_multi_alpha_comparison.png",
    ) -> str:
        rho = self._as_1d(rho)
        alpha_values = self._as_1d(alpha_values)
        results = [self.evaluate_multi_alpha(eq, rho, alpha_values, n_ensemble=n_ensemble) for eq in equilibria]

        if any(q.shape[0] != 1 for q, _ in results):
            raise ValueError("plot_multi_alpha_comparison expects exactly one rho value")

        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, ((q, q_std), label) in enumerate(zip(results, labels)):
            q_line = q[0]
            q_std_line = q_std[0]
            ax.errorbar(
                alpha_values,
                q_line,
                yerr=q_std_line,
                fmt="o-" if idx == 0 else "s-",
                capsize=4,
                label=label,
                color=f"C{idx}",
            )

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Q (ITG heat flux)")
        ax.set_title("Per-Field-Line Q")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def scan_temperature_gradient(
        self,
        eq,
        *,
        rho,
        alpha,
        a_over_LT_values,
        a_over_Ln: float | None = None,
        n_ensemble: int | None = None,
    ) -> np.ndarray:
        rho = self._as_1d(rho)
        alpha = self._as_1d(alpha)
        a_over_LT_values = self._as_1d(a_over_LT_values)

        obj = self._build_proxy(
            eq,
            rho,
            alpha,
            a_over_LT=float(a_over_LT_values[0]),
            a_over_Ln=self.proxy_config.a_over_Ln if a_over_Ln is None else a_over_Ln,
            n_ensemble=n_ensemble,
        )

        q_values = []
        for a_over_LT in a_over_LT_values:
            modified_constants = obj.constants.copy()
            modified_constants["a_over_LT"] = float(a_over_LT)
            q_array = obj.compute(eq.params_dict, constants=modified_constants)
            q_values.append(float(np.asarray(q_array, dtype=float).reshape(-1)[0]))
        return np.asarray(q_values, dtype=float)

    def plot_temperature_gradient_scan(
        self,
        equilibria: Sequence,
        labels: Sequence[str],
        *,
        rho,
        alpha,
        a_over_LT_values,
        a_over_Ln: float | None = None,
        n_ensemble: int | None = None,
        filename: str = "05_temperature_gradient_scan.png",
    ) -> str:
        a_over_LT_values = self._as_1d(a_over_LT_values)
        curves = [
            self.scan_temperature_gradient(
                eq,
                rho=rho,
                alpha=alpha,
                a_over_LT_values=a_over_LT_values,
                a_over_Ln=a_over_Ln,
                n_ensemble=n_ensemble,
            )
            for eq in equilibria
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        markers = ["o-", "s-", "^-", "d-"]
        for idx, (curve, label) in enumerate(zip(curves, labels)):
            marker = markers[idx % len(markers)]
            ax1.plot(a_over_LT_values, curve, marker, label=label, color=f"C{idx}")
            ax2.semilogy(a_over_LT_values, curve, marker, label=label, color=f"C{idx}")

        ax1.set_xlabel("a/LT")
        ax1.set_ylabel("Q (ITG heat flux)")
        ax1.set_title("Q vs Temperature Gradient")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("a/LT")
        ax2.set_ylabel("Q (log scale)")
        ax2.set_title("Q vs Temperature Gradient (log)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def evaluate_q_vs_rho(
        self,
        eq,
        *,
        rho_values,
        alpha,
        n_ensemble: int | None = None,
        return_std: bool = True,
    ):
        rho_values = self._as_1d(rho_values)
        alpha = self._as_1d(alpha)
        obj = self._build_proxy(eq, rho_values, alpha, n_ensemble=n_ensemble)
        if return_std:
            q, q_std = obj.compute(eq.params_dict, return_std=True)
            return np.asarray(q, dtype=float), np.asarray(q_std, dtype=float)
        q = obj.compute(eq.params_dict)
        return np.asarray(q, dtype=float), None

    def plot_q_vs_rho(
        self,
        equilibria: Sequence,
        labels: Sequence[str],
        *,
        rho_values,
        alpha,
        n_ensemble: int | None = None,
        filename: str = "06_turbulent_heat_flux_vs_rho.png",
    ) -> str:
        rho_values = self._as_1d(rho_values)
        alpha = self._as_1d(alpha)

        fig, ax = plt.subplots(figsize=(8, 5))
        markers = ["o-", "s-", "^-", "d-"]
        for idx, (eq, label) in enumerate(zip(equilibria, labels)):
            q, q_std = self.evaluate_q_vs_rho(
                eq,
                rho_values=rho_values,
                alpha=alpha,
                n_ensemble=n_ensemble,
                return_std=True,
            )
            marker = markers[idx % len(markers)]
            ax.errorbar(
                rho_values,
                q,
                yerr=q_std,
                fmt=marker,
                capsize=4,
                label=label,
                color=f"C{idx}",
            )

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("Q (ITG heat flux)")
        ax.set_title("Turbulent Heat Flux vs rho")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def solve_transport_temperature_estimate(
        self,
        eq,
        *,
        transport_config: TransportEstimateConfig | None = None,
        heating_source: Callable[[np.ndarray], np.ndarray] = example_heating_source,
        density_profile: Callable[[np.ndarray], np.ndarray] = example_density_profile,
        n_ensemble: int | None = None,
    ) -> dict[str, np.ndarray | str | bool]:
        transport_config = (
            TransportEstimateConfig() if transport_config is None else transport_config
        )
        solver_config = transport_config.to_transport_config()
        rho_faces = np.linspace(
            solver_config.rho_inner,
            solver_config.rho_outer,
            solver_config.nr + 1,
            dtype=float,
        )
        alpha_values = np.linspace(0.0, 2.0 * np.pi, solver_config.n_alpha, endpoint=False)
        gx_flux_model = _CachedNNITGTransportModel(
            eq,
            rho_faces,
            alpha_values,
            self.proxy_config,
            transport_config,
            n_ensemble=n_ensemble,
        )
        solver = StellaratorPowerBalanceSolver(
            eq=eq,
            gx_flux_model=gx_flux_model,
            heating_source=heating_source,
            density_profile=density_profile,
            config=solver_config,
        )
        ti_init = self._transport_initial_guess(solver.rho_centers, transport_config)
        result = solver.solve(
            Ti_init=ti_init,
            T_edge=transport_config.edge_temperature,
        )
        result["Ti_init"] = ti_init
        result["density"] = np.asarray(density_profile(solver.rho_centers), dtype=float)
        result["heating"] = np.asarray(heating_source(solver.rho_centers), dtype=float)
        result["gx_B_reference"] = float(gx_flux_model.B_ref)
        result["gx_L_reference"] = float(gx_flux_model.L_ref)
        result["temperature_unit_eV"] = float(transport_config.temperature_unit_eV)
        return result

    def plot_transport_temperature_estimate(
        self,
        equilibria: Sequence,
        labels: Sequence[str],
        *,
        transport_config: TransportEstimateConfig | None = None,
        heating_source: Callable[[np.ndarray], np.ndarray] = example_heating_source,
        density_profile: Callable[[np.ndarray], np.ndarray] = example_density_profile,
        n_ensemble: int | None = None,
        filename: str = "07_transport_temperature_estimate.png",
    ) -> str:
        transport_config = (
            TransportEstimateConfig() if transport_config is None else transport_config
        )
        results = [
            self.solve_transport_temperature_estimate(
                eq,
                transport_config=transport_config,
                heating_source=heating_source,
                density_profile=density_profile,
                n_ensemble=n_ensemble,
            )
            for eq in equilibria
        ]

        fig, (ax_temp, ax_flux) = plt.subplots(1, 2, figsize=(12, 5))
        markers = ["o-", "s-", "^-", "d-"]
        for idx, (result, label) in enumerate(zip(results, labels)):
            marker = markers[idx % len(markers)]
            rho_centers = np.asarray(result["rho_centers"], dtype=float)
            rho_faces = np.asarray(result["rho_faces"], dtype=float)
            ti = np.asarray(result["Ti"], dtype=float)
            q_faces = np.asarray(result["Q_faces"], dtype=float)

            ax_temp.plot(rho_centers, ti, marker, label=label, color=f"C{idx}")
            ax_flux.plot(rho_faces, q_faces, marker, label=label, color=f"C{idx}")

            status = "OK" if bool(result["success"]) else "FAIL"
            message = str(result["message"])
            residual_norm = np.linalg.norm(np.asarray(result["residual"], dtype=float))
            _log(
                f"Transport solve for {label}: {status} "
                f"(residual norm {residual_norm:.3e}, nfev {int(result['nfev'])}) "
                f"- {message}"
            )
            if not bool(result["success"]):
                flux_divergence = np.asarray(result["flux_divergence"], dtype=float)
                source = np.asarray(result["source_centers"], dtype=float)
                diagnostic_path = self._save_transport_diagnostics(label, result, filename)
                _log(
                    f"  source norm {np.linalg.norm(source):.3e}, "
                    f"flux-divergence norm {np.linalg.norm(flux_divergence):.3e}, "
                    f"max |Q| {np.max(np.abs(q_faces)):.3e}, "
                    f"B_ref {float(result['gx_B_reference']):.3e} T, "
                    f"L_ref {float(result['gx_L_reference']):.3e} m"
                )
                _log(f"  Saved diagnostics: {diagnostic_path}")

        ax_temp.set_xlabel(r"$\rho$")
        ax_temp.set_ylabel(r"$T_i$ estimate (transport-model units)")
        ax_temp.set_title("Ion Temperature Estimate")
        ax_temp.legend()
        ax_temp.grid(True, alpha=0.3)

        ax_flux.set_xlabel(r"$\rho$")
        ax_flux.set_ylabel("Transport-model Q")
        ax_flux.set_title("Flux-Surface-Averaged Heat Flux")
        ax_flux.legend()
        ax_flux.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def run_default_suite(
        self,
        initial_eq,
        optimized_eq,
        *,
        labels: Sequence[str] = ("Initial", "Optimized"),
        multi_alpha_rho=None,
        multi_alpha_values=None,
        gradient_scan_rho=None,
        gradient_scan_alpha=None,
        gradient_scan_values=None,
        gradient_scan_a_over_Ln: float | None = None,
        rho_scan_values=None,
        rho_scan_alpha=None,
        transport_config: TransportEstimateConfig | None = None,
        n_ensemble: int | None = None,
    ) -> dict[str, str]:
        multi_alpha_rho = self._as_1d(np.sqrt(0.5) if multi_alpha_rho is None else multi_alpha_rho)
        multi_alpha_values = self._as_1d(
            np.linspace(0.0, np.pi, 8) if multi_alpha_values is None else multi_alpha_values
        )
        gradient_scan_rho = self._as_1d(
            np.sqrt(0.5) if gradient_scan_rho is None else gradient_scan_rho
        )
        gradient_scan_alpha = self._as_1d(0.0 if gradient_scan_alpha is None else gradient_scan_alpha)
        gradient_scan_values = self._as_1d(
            np.linspace(0.5, 10.0, 20) if gradient_scan_values is None else gradient_scan_values
        )
        rho_scan_values = self._as_1d(
            np.linspace(0.05, 0.95, 12) if rho_scan_values is None else rho_scan_values
        )
        rho_scan_alpha = self._as_1d(0.0 if rho_scan_alpha is None else rho_scan_alpha)

        equilibria = [initial_eq, optimized_eq]
        saved_paths = {
            "03_flux_surface_comparison": self.plot_flux_surface_comparison(equilibria, labels),
            "04_multi_alpha_comparison": self.plot_multi_alpha_comparison(
                equilibria,
                labels,
                rho=multi_alpha_rho,
                alpha_values=multi_alpha_values,
                n_ensemble=n_ensemble,
            ),
            "05_temperature_gradient_scan": self.plot_temperature_gradient_scan(
                equilibria,
                labels,
                rho=gradient_scan_rho,
                alpha=gradient_scan_alpha,
                a_over_LT_values=gradient_scan_values,
                a_over_Ln=gradient_scan_a_over_Ln,
                n_ensemble=n_ensemble,
            ),
            "06_turbulent_heat_flux_vs_rho": self.plot_q_vs_rho(
                equilibria,
                labels,
                rho_values=rho_scan_values,
                alpha=rho_scan_alpha,
                n_ensemble=n_ensemble,
            ),
            "07_transport_temperature_estimate": self.plot_transport_temperature_estimate(
                equilibria,
                labels,
                transport_config=transport_config,
                n_ensemble=n_ensemble,
            ),
        }
        return saved_paths
