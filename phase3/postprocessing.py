"""Postprocessing for the Phase3 critical-gradient workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from desc.plotting import plot_comparison

from critical_gradient_objective import NNCriticalGradientProxy


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass(frozen=True)
class CriticalGradientPostprocessConfig:
    ensemble_dir: str
    ensemble_csv: str
    n_ensemble: int = 3
    target_flux_tube_length: float = 75.4
    a_over_Ln: float = 0.9
    a_over_LT_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.5, 10.0, 20, dtype=float)
    )
    critical_q_threshold: float = 1.0
    critical_log_width: float = 0.35
    alpha_aggregation: str = "mean"


class CriticalGradientPostprocessor:
    def __init__(self, output_dir: str, config: CriticalGradientPostprocessConfig):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _as_1d(values) -> np.ndarray:
        return np.atleast_1d(np.asarray(values, dtype=float))

    def _build_proxy(self, eq, rho, alpha, *, alpha_aggregation: str | None = None):
        proxy = NNCriticalGradientProxy(
            eq=eq,
            rho=self._as_1d(rho),
            alpha=self._as_1d(alpha),
            ensemble_dir=self.config.ensemble_dir,
            ensemble_csv=self.config.ensemble_csv,
            n_ensemble=self.config.n_ensemble,
            target_flux_tube_length=self.config.target_flux_tube_length,
            a_over_Ln=self.config.a_over_Ln,
            a_over_LT_grid=self.config.a_over_LT_grid,
            critical_q_threshold=self.config.critical_q_threshold,
            critical_log_width=self.config.critical_log_width,
            alpha_aggregation=self.config.alpha_aggregation
            if alpha_aggregation is None
            else alpha_aggregation,
        )
        proxy.build(use_jit=False, verbose=0)
        return proxy

    def plot_flux_surface_comparison(self, equilibria, labels, filename="03_flux_surface_comparison.png"):
        fig, _ = plot_comparison(list(equilibria), labels=list(labels))
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def evaluate_critical_gradient(self, eq, *, rho, alpha, alpha_aggregation=None):
        proxy = self._build_proxy(eq, rho, alpha, alpha_aggregation=alpha_aggregation)
        crit = proxy.compute(eq.params_dict, return_critical_gradient=True)
        return np.asarray(crit, dtype=float)

    def evaluate_critical_gradient_ensemble(
        self, eq, *, rho, alpha, alpha_aggregation=None
    ):
        proxy = self._build_proxy(eq, rho, alpha, alpha_aggregation=alpha_aggregation)
        crit, ensemble_crit = proxy.compute(
            eq.params_dict,
            return_critical_gradient=True,
            return_ensemble_critical_gradient=True,
        )
        return np.asarray(crit, dtype=float), np.asarray(ensemble_crit, dtype=float)

    def evaluate_critical_gradient_metric_uncertainty(
        self, eq, *, rho, alpha, alpha_aggregation=None
    ):
        critical, ensemble_critical = self.evaluate_critical_gradient_ensemble(
            eq, rho=rho, alpha=alpha, alpha_aggregation=alpha_aggregation
        )
        ensemble_flat = ensemble_critical.reshape(ensemble_critical.shape[0], -1)
        ensemble_metric = np.mean(ensemble_flat, axis=1)
        ddof = 1 if ensemble_metric.size > 1 else 0
        return {
            "critical_gradient": float(np.mean(np.asarray(critical, dtype=float))),
            "critical_gradient_ensemble_mean": float(np.mean(ensemble_metric)),
            "critical_gradient_ensemble_std": float(np.std(ensemble_metric, ddof=ddof)),
            "critical_gradient_ensemble_sem": float(
                np.std(ensemble_metric, ddof=ddof) / np.sqrt(max(ensemble_metric.size, 1))
            ),
            "critical_gradient_per_model": ensemble_metric.tolist(),
            "critical_gradient_per_rho": np.asarray(critical, dtype=float).tolist(),
            "critical_gradient_per_model_per_rho": np.asarray(
                ensemble_critical, dtype=float
            ).tolist(),
        }

    def evaluate_q_sweep(self, eq, *, rho, alpha):
        proxy = self._build_proxy(eq, rho, alpha, alpha_aggregation="separate")
        crit, q_sweep = proxy.compute(
            eq.params_dict,
            return_critical_gradient=True,
            return_q_sweep=True,
        )
        return np.asarray(crit, dtype=float), np.asarray(q_sweep, dtype=float)

    def plot_critical_gradient_scan_comparison(
        self,
        equilibria,
        labels,
        *,
        rho,
        alpha,
        filename="04_critical_gradient_scan.png",
    ):
        rho = self._as_1d(rho)
        if rho.size != 1:
            raise ValueError("plot_critical_gradient_scan_comparison expects exactly one rho value")
        alpha = self._as_1d(alpha)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        markers = ["o-", "s-", "^-", "d-"]

        for idx, (eq, label) in enumerate(zip(equilibria, labels)):
            critical_per_alpha, q_sweep = self.evaluate_q_sweep(eq, rho=rho, alpha=alpha)
            q_mean = q_sweep[0].mean(axis=-1)
            crit_mean = critical_per_alpha[0].mean()
            marker = markers[idx % len(markers)]

            ax1.plot(
                self.config.a_over_LT_grid,
                q_mean,
                marker,
                color=f"C{idx}",
                label=f"{label} (a/LTcrit={crit_mean:.2f})",
            )
            ax1.axvline(crit_mean, color=f"C{idx}", linestyle="--", alpha=0.8)

            ax2.semilogy(
                self.config.a_over_LT_grid,
                q_mean,
                marker,
                color=f"C{idx}",
                label=f"{label} (a/LTcrit={crit_mean:.2f})",
            )
            ax2.axvline(crit_mean, color=f"C{idx}", linestyle="--", alpha=0.8)

        for ax in (ax1, ax2):
            ax.axhline(self.config.critical_q_threshold, color="k", linestyle=":", label="Q threshold")
            ax.set_xlabel("a/LT")
            ax.grid(True, alpha=0.3)

        ax1.set_ylabel("Alpha-averaged Q")
        ax1.set_title(r"$Q(a/LT)$ with critical-gradient marker")
        ax1.legend()

        ax2.set_ylabel("Alpha-averaged Q (log)")
        ax2.set_title(r"$Q(a/LT)$ (log scale)")
        ax2.legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def plot_critical_gradient_vs_rho(
        self,
        equilibria,
        labels,
        *,
        rho_values,
        alpha,
        filename="05_critical_gradient_vs_rho.png",
    ):
        rho_values = self._as_1d(rho_values)
        alpha = self._as_1d(alpha)

        fig, ax = plt.subplots(figsize=(8, 5))
        markers = ["o-", "s-", "^-", "d-"]
        for idx, (eq, label) in enumerate(zip(equilibria, labels)):
            critical = self.evaluate_critical_gradient(eq, rho=rho_values, alpha=alpha)
            marker = markers[idx % len(markers)]
            ax.plot(rho_values, critical, marker, color=f"C{idx}", label=label)

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$a/LT_{crit}$")
        ax.set_title("Critical gradient vs rho")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        _log(f"Saved: {path}")
        return path

    def plot_heat_flux_scans_by_rho(
        self,
        equilibria,
        labels,
        *,
        rho_values,
        alpha,
        filename="06_heat_flux_scans_by_rho.png",
    ):
        rho_values = self._as_1d(rho_values)
        alpha = self._as_1d(alpha)
        if rho_values.size == 0:
            raise ValueError("plot_heat_flux_scans_by_rho expects at least one rho value")

        ncols = min(3, rho_values.size)
        nrows = int(np.ceil(rho_values.size / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.2 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()
        markers = ["o-", "s-", "^-", "d-"]

        for idx, (eq, label) in enumerate(zip(equilibria, labels)):
            _, q_sweep = self.evaluate_q_sweep(eq, rho=rho_values, alpha=alpha)
            q_mean = q_sweep.mean(axis=-1)
            marker = markers[idx % len(markers)]
            for rho_idx, _ in enumerate(rho_values):
                axes_flat[rho_idx].plot(
                    self.config.a_over_LT_grid,
                    q_mean[rho_idx],
                    marker,
                    color=f"C{idx}",
                    label=label,
                )

        for rho_idx, rho_value in enumerate(rho_values):
            ax = axes_flat[rho_idx]
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1.2, label="Q=1")
            ax.set_title(rf"$\rho={rho_value:.3f}$")
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[rho_values.size :]:
            ax.set_visible(False)

        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel(r"$a/L_{T_i}$")
        for ax in axes[:, 0]:
            if ax.get_visible():
                ax.set_ylabel("Alpha-averaged heat flux Q")

        handles, legend_labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles))
        fig.suptitle(r"Heat flux scans by $\rho$")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

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
        labels=("Initial", "Final"),
        scan_rho=np.array([np.sqrt(0.5)], dtype=float),
        scan_alpha=np.linspace(0.0, np.pi, 8, dtype=float),
        rho_scan_values=np.linspace(0.05, 0.95, 12, dtype=float),
        rho_scan_alpha=np.linspace(0.0, np.pi, 8, dtype=float),
        heat_flux_rho_values=None,
        heat_flux_alpha=None,
    ):
        equilibria = [initial_eq, optimized_eq]
        if heat_flux_rho_values is None:
            heat_flux_rho_values = rho_scan_values
        if heat_flux_alpha is None:
            heat_flux_alpha = rho_scan_alpha
        return {
            "03_flux_surface_comparison": self.plot_flux_surface_comparison(equilibria, labels),
            "04_critical_gradient_scan": self.plot_critical_gradient_scan_comparison(
                equilibria,
                labels,
                rho=scan_rho,
                alpha=scan_alpha,
            ),
            "05_critical_gradient_vs_rho": self.plot_critical_gradient_vs_rho(
                equilibria,
                labels,
                rho_values=rho_scan_values,
                alpha=rho_scan_alpha,
            ),
            "06_heat_flux_scans_by_rho": self.plot_heat_flux_scans_by_rho(
                equilibria,
                labels,
                rho_values=heat_flux_rho_values,
                alpha=heat_flux_alpha,
            ),
        }
