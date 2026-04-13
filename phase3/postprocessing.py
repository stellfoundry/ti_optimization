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

    def run_default_suite(
        self,
        initial_eq,
        optimized_eq,
        *,
        labels=("Initial", "Optimized"),
        scan_rho=np.array([np.sqrt(0.5)], dtype=float),
        scan_alpha=np.linspace(0.0, np.pi, 8, dtype=float),
        rho_scan_values=np.linspace(0.05, 0.95, 12, dtype=float),
        rho_scan_alpha=np.linspace(0.0, np.pi, 8, dtype=float),
    ):
        equilibria = [initial_eq, optimized_eq]
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
        }
