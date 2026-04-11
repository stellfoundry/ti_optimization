import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from scipy.interpolate import CubicSpline
from scipy.optimize import root

# DESC imports
from desc.grid import LinearGrid
from desc.integrals.surface_integral import surface_averages


def _log(message: str) -> None:
    """Emit a UTC-timestamped log line."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


# ----------------------------
# User-supplied GX surrogate API
# ----------------------------
# Expected signature:
#   q = gx_flux_model(rho, alpha, aLTi, aLni, Ti, ni, extra)
#
# Return:
#   local radial turbulent ion heat flux at that (rho, alpha)
#   in consistent units. ``alpha`` may be either a scalar or a 1D array.
#
# Notes:
# - If your surrogate instead returns gyroBohm-normalized flux, convert it here.
# - If it depends on additional geometric features, pass them through `extra`.
# ----------------------------


@dataclass
class TransportConfig:
    rho_inner: float = 0.05
    rho_outer: float = 0.90
    nr: int = 16                 # number of cell centers
    n_alpha: int = 8             # legacy seed count for GX model setup
    M_grid: int | None = None    # default to eq.M_grid when unset
    N_grid: int | None = None    # default to eq.N_grid when unset
    max_iter: int = 30
    verbose: bool = True


def gxnn_flux_surface_average(
    eq,
    gxnn_model,
    rho_vals,
    M=None,
    N=None,
    alpha_definition="theta_minus_iota_zeta",
    surrogate_kwargs_fn=None,
    grid=None,
    data=None,
    verbose=False,
):
    """Compute a DESC-consistent flux-surface average from a GX-style surrogate."""
    rho_vals = np.atleast_1d(np.asarray(rho_vals, dtype=float))

    if M is None:
        M = getattr(eq, "M_grid", 32)
    if N is None:
        N = getattr(eq, "N_grid", 32)

    if grid is None:
        if verbose:
            _log(
                "[transport] building DESC averaging grid:"
                f" nrho={rho_vals.size}, M={M}, N={N}"
            )
        grid = LinearGrid(
            rho=rho_vals,
            M=M,
            N=N,
            NFP=eq.NFP,
            sym=False,
            axis=False,
            endpoint=False,
        )

    if data is None:
        if verbose:
            _log("[transport] computing DESC geometry data for averaging grid")
        try:
            data = eq.compute(["rho", "sqrt(g)", "iota"], grid=grid)
            sqrt_g = np.asarray(data["sqrt(g)"])
        except Exception:
            data = eq.compute(["rho", "sqrt_g", "iota"], grid=grid)
            sqrt_g = np.asarray(data["sqrt_g"])
    elif "sqrt(g)" in data:
        sqrt_g = np.asarray(data["sqrt(g)"])
    elif "sqrt_g" in data:
        sqrt_g = np.asarray(data["sqrt_g"])
    else:
        raise KeyError("data must contain either 'sqrt(g)' or 'sqrt_g'")

    nodes = np.asarray(grid.nodes)
    rho = nodes[:, 0]
    theta = nodes[:, 1]
    zeta = nodes[:, 2]
    iota = np.asarray(data["iota"])

    if alpha_definition == "theta_minus_iota_zeta":
        alpha = theta - iota * zeta
    else:
        raise NotImplementedError(
            f"Unsupported alpha_definition={alpha_definition}"
        )
    alpha = np.mod(alpha, 2 * np.pi)

    extra_kwargs = {}
    if surrogate_kwargs_fn is not None:
        extra_kwargs = surrogate_kwargs_fn(rho, theta, zeta, data)
        if extra_kwargs is None:
            extra_kwargs = {}

    if verbose:
        _log(
            "[transport] evaluating local GX surrogate on DESC nodes:"
            f" n_nodes={rho.size}, nrho={rho_vals.size}"
        )
    q_nodes = np.asarray(gxnn_model(rho=rho, alpha=alpha, **extra_kwargs), dtype=float)
    if q_nodes.ndim == 0:
        q_nodes = np.full(rho.shape, float(q_nodes), dtype=float)
    if q_nodes.shape != rho.shape:
        raise ValueError(
            "gxnn_model must return one flux value per grid node; "
            f"expected shape {rho.shape}, got {q_nodes.shape}"
        )

    if verbose:
        _log("[transport] computing flux-surface averages")
    q_fsa_full = np.asarray(surface_averages(grid, q_nodes, sqrt_g=sqrt_g))
    q_fsa = np.asarray(grid.compress(q_fsa_full, surface_label="rho"), dtype=float)
    rho_unique = np.asarray(grid.compress(rho, surface_label="rho"), dtype=float)

    if verbose:
        _log(
            "[transport] completed flux-surface average:"
            f" q_fsa_min={np.nanmin(q_fsa):.6e}, q_fsa_max={np.nanmax(q_fsa):.6e}"
        )

    return {
        "rho_unique": rho_unique,
        "q_fsa": q_fsa,
        "q_nodes": q_nodes,
        "alpha": alpha,
        "grid": grid,
        "data": data,
    }


class StellaratorPowerBalanceSolver:
    """
    Minimal steady-state 1D transport solver for T_i(rho) on a fixed DESC equilibrium.

    Unknown:
        Ti(rho) on radial cell centers

    Equation:
        (1 / V_r) * d/drho [ V_r * Q_rho(Ti) ] = S(rho)

    where:
        V_r = dV/drho from DESC
        Q_rho(Ti) = flux-surface-averaged radial turbulent heat flux
    """

    def __init__(
        self,
        eq,
        gx_flux_model: Callable,
        heating_source: Callable[[np.ndarray], np.ndarray],
        density_profile: Callable[[np.ndarray], np.ndarray],
        config: Optional[TransportConfig] = None,
    ):
        self.eq = eq
        self.gx_flux_model = gx_flux_model
        self.heating_source = heating_source
        self.density_profile = density_profile
        self.cfg = config or TransportConfig()

        # Radial finite-volume grid
        self.rho_faces = np.linspace(self.cfg.rho_inner, self.cfg.rho_outer, self.cfg.nr + 1)
        self.rho_centers = 0.5 * (self.rho_faces[:-1] + self.rho_faces[1:])
        self.dr = np.diff(self.rho_faces)
        self._interface_state_calls = 0

        # Precompute geometry on faces and centers from DESC
        self.geom_faces = self._precompute_geometry(self.rho_faces)
        self.geom_centers = self._precompute_geometry(self.rho_centers)

    # ------------------------------------------------------------------
    # DESC geometry helpers
    # ------------------------------------------------------------------
    def _precompute_geometry(self, rho_vals: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Uses DESC to compute:
          - V_r(r): dV/drho on each surface
          - sqrt(g): Jacobian on a (rho, theta, zeta) grid for optional weighted averages
          - iota if needed
        """
        # Use a grid with explicit rho surfaces and full theta/zeta coverage.
        M_grid = getattr(self.eq, "M_grid", 32) if self.cfg.M_grid is None else self.cfg.M_grid
        N_grid = getattr(self.eq, "N_grid", 32) if self.cfg.N_grid is None else self.cfg.N_grid

        if self.cfg.verbose:
            _log(
                "[transport] precomputing geometry:"
                f" nrho={len(rho_vals)}, M={M_grid}, N={N_grid}"
            )

        grid = LinearGrid(
            rho=rho_vals,
            M=M_grid,
            N=N_grid,
            NFP=self.eq.NFP,
            sym=False,
            axis=False,
            endpoint=False,
        )

        try:
            if self.cfg.verbose:
                _log("[transport] calling eq.compute for geometry fields")
            data = self.eq.compute(
                ["rho", "V_r(r)", "sqrt(g)", "iota"],
                grid=grid,
            )
            sqrt_g = np.asarray(data["sqrt(g)"])
        except Exception:
            if self.cfg.verbose:
                _log("[transport] retrying eq.compute with sqrt_g field name")
            data = self.eq.compute(
                ["rho", "V_r(r)", "sqrt_g", "iota"],
                grid=grid,
            )
            sqrt_g = np.asarray(data["sqrt_g"])

        nodes = grid.nodes  # columns: rho, theta, zeta
        unique_rho = np.unique(nodes[:, 0])

        # Pack node-wise quantities by surface
        surf = {}
        surf["rho"] = unique_rho
        surf["grid"] = grid
        surf["data"] = data
        surf["nodes"] = nodes
        surf["sqrt_g"] = sqrt_g
        surf["iota"] = np.asarray(data["iota"])

        # V_r(r) is a flux-function quantity. Extract one value per rho.
        Vr_all = np.asarray(data["V_r(r)"])
        Vr = np.zeros_like(unique_rho)
        for i, r in enumerate(unique_rho):
            idx = np.where(np.isclose(nodes[:, 0], r))[0]
            Vr[i] = Vr_all[idx[0]]
        surf["V_r"] = Vr

        # Useful indexing by surface
        surf["surface_indices"] = [
            np.where(np.isclose(nodes[:, 0], r))[0] for r in unique_rho
        ]
        surf["surface_inverse_idx"] = np.zeros(nodes.shape[0], dtype=int)
        for i, idx in enumerate(surf["surface_indices"]):
            surf["surface_inverse_idx"][idx] = i
        if self.cfg.verbose:
            _log(
                "[transport] geometry ready:"
                f" unique_rho={unique_rho.size}, n_nodes={nodes.shape[0]}"
            )
        return surf

    # ------------------------------------------------------------------
    # Profile reconstruction
    # ------------------------------------------------------------------
    def _profile_spline(self, Ti_center: np.ndarray) -> CubicSpline:
        """
        Build a smooth spline Ti(rho). We clamp the outer boundary by using the
        last center value as a Dirichlet-like edge temperature.
        """
        rho = self.rho_centers
        return CubicSpline(rho, Ti_center, bc_type=((1, 0.0), (1, (Ti_center[-1] - Ti_center[-2]) / (rho[-1] - rho[-2]))))

    def _a_over_LTi(self, Ti_spline: CubicSpline, rho_faces: np.ndarray) -> np.ndarray:
        """
        Compute a/L_T = -a * d(ln T)/drho * drho/dr_minor

        For a first implementation, treat rho as the transport radial coordinate directly
        and use:
            a/L_T ~= - d ln T / d rho

        If you want a physical minor radius scaling later, multiply by a * d rho / d r_minor.
        """
        Ti = Ti_spline(rho_faces)
        dTi_drho = Ti_spline(rho_faces, 1)
        return -dTi_drho / np.maximum(Ti, 1e-12)

    def _a_over_Lni(self, rho_vals: np.ndarray) -> np.ndarray:
        """Compute a/L_n from the prescribed density profile."""
        rho_vals = np.asarray(rho_vals, dtype=float)
        if rho_vals.size == 0:
            return rho_vals

        step = min(1e-4, 0.25 * np.min(self.dr))
        rho_minus = np.clip(rho_vals - step, self.cfg.rho_inner, self.cfg.rho_outer)
        rho_plus = np.clip(rho_vals + step, self.cfg.rho_inner, self.cfg.rho_outer)

        n_i = np.asarray(self.density_profile(rho_vals), dtype=float)
        n_i_minus = np.asarray(self.density_profile(rho_minus), dtype=float)
        n_i_plus = np.asarray(self.density_profile(rho_plus), dtype=float)
        dn_drho = (n_i_plus - n_i_minus) / np.maximum(rho_plus - rho_minus, 1e-12)
        return -dn_drho / np.maximum(n_i, 1e-12)

    def compute_interface_fluxes(
        self,
        Ti_center: np.ndarray,
        extra_features: Optional[Callable[[float, float], Dict]] = None,
    ) -> np.ndarray:
        return self.compute_interface_state(Ti_center, extra_features=extra_features)["Q_faces"]

    def compute_interface_state(
        self,
        Ti_center: np.ndarray,
        extra_features: Optional[Callable[[float, float], Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all face-centered transport quantities used by the finite-volume solve.
        """
        Ti_spline = self._profile_spline(Ti_center)
        aLTi_faces = self._a_over_LTi(Ti_spline, self.rho_faces)
        Ti_faces = np.asarray(Ti_spline(self.rho_faces), dtype=float)
        ni_faces = np.asarray(self.density_profile(self.rho_faces), dtype=float)
        aLni_faces = self._a_over_Lni(self.rho_faces)
        surface_idx = self.geom_faces["surface_inverse_idx"]
        geom = self.geom_faces
        self._interface_state_calls += 1

        if self.cfg.verbose:
            _log(
                "[transport] compute_interface_state:"
                f" call={self._interface_state_calls}, n_faces={self.rho_faces.size},"
                f" n_nodes={geom['nodes'].shape[0]}"
            )

        def surrogate_kwargs_fn(rho, theta, zeta, data):
            kwargs = {
                "aLTi": aLTi_faces[surface_idx],
                "aLni": aLni_faces[surface_idx],
                "Ti": Ti_faces[surface_idx],
                "ni": ni_faces[surface_idx],
                "extra": {},
            }
            if extra_features is not None:
                extra = extra_features(rho, theta, zeta, data)
                kwargs["extra"] = {} if extra is None else extra
            return kwargs

        fsa_result = gxnn_flux_surface_average(
            eq=self.eq,
            gxnn_model=self.gx_flux_model,
            rho_vals=self.rho_faces,
            M=self.cfg.M_grid,
            N=self.cfg.N_grid,
            surrogate_kwargs_fn=surrogate_kwargs_fn,
            grid=geom["grid"],
            data=geom["data"],
            verbose=self.cfg.verbose,
        )
        if not np.allclose(fsa_result["rho_unique"], self.rho_faces, atol=1e-12, rtol=0.0):
            raise ValueError(
                "DESC flux-surface average returned rho surfaces that do not match "
                "the transport face grid."
            )

        if self.cfg.verbose:
            _log(
                "[transport] interface state complete:"
                f" Q_min={np.nanmin(fsa_result['q_fsa']):.6e},"
                f" Q_max={np.nanmax(fsa_result['q_fsa']):.6e}"
            )

        return {
            "Q_faces": np.asarray(fsa_result["q_fsa"], dtype=float),
            "q_nodes": np.asarray(fsa_result["q_nodes"], dtype=float),
            "alpha_nodes": np.asarray(fsa_result["alpha"], dtype=float),
            "rho_nodes": np.asarray(geom["nodes"][:, 0], dtype=float),
            "Ti_faces": Ti_faces,
            "ni_faces": ni_faces,
            "aLTi_faces": np.asarray(aLTi_faces, dtype=float),
            "aLni_faces": np.asarray(aLni_faces, dtype=float),
        }

    def _flux_divergence(self, Q_faces: np.ndarray) -> np.ndarray:
        """Compute the finite-volume divergence term at cell centers."""
        Vr_faces = self.geom_faces["V_r"]
        Vr_centers = self.geom_centers["V_r"]
        flux_divergence = np.zeros_like(self.rho_centers)
        for i in range(self.cfg.nr):
            flux_out = Vr_faces[i + 1] * Q_faces[i + 1]
            flux_in = Vr_faces[i] * Q_faces[i]
            deltaV = Vr_centers[i] * self.dr[i]
            flux_divergence[i] = (flux_out - flux_in) / max(deltaV, 1e-14)
        return flux_divergence

    def build_diagnostics(
        self,
        Ti_center: np.ndarray,
        T_edge: Optional[float] = None,
        extra_features: Optional[Callable[[float, float], Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate the full transport state at a given temperature profile."""
        Ti_center = np.asarray(Ti_center, dtype=float)
        if self.cfg.verbose:
            _log("[transport] building diagnostics")
        source_centers = np.asarray(self.heating_source(self.rho_centers), dtype=float)
        diagnostics = {
            "rho_centers": np.asarray(self.rho_centers, dtype=float),
            "rho_faces": np.asarray(self.rho_faces, dtype=float),
            "dr": np.asarray(self.dr, dtype=float),
            "Vr_faces": np.asarray(self.geom_faces["V_r"], dtype=float),
            "Vr_centers": np.asarray(self.geom_centers["V_r"], dtype=float),
            "source_centers": source_centers,
        }

        if not np.all(np.isfinite(Ti_center)) or np.any(Ti_center <= 0.0):
            diagnostics.update(
                {
                    "Ti": Ti_center,
                    "Q_faces": np.full_like(self.rho_faces, np.nan, dtype=float),
                    "q_nodes": np.full(self.geom_faces["nodes"].shape[0], np.nan, dtype=float),
                    "alpha_nodes": np.full(self.geom_faces["nodes"].shape[0], np.nan, dtype=float),
                    "rho_nodes": np.asarray(self.geom_faces["nodes"][:, 0], dtype=float),
                    "Ti_faces": np.full_like(self.rho_faces, np.nan, dtype=float),
                    "ni_faces": np.full_like(self.rho_faces, np.nan, dtype=float),
                    "aLTi_faces": np.full_like(self.rho_faces, np.nan, dtype=float),
                    "aLni_faces": np.full_like(self.rho_faces, np.nan, dtype=float),
                    "flux_divergence": np.full_like(self.rho_centers, np.nan, dtype=float),
                    "residual": self.residual(
                        Ti_center,
                        T_edge=T_edge,
                        extra_features=extra_features,
                    ),
                }
            )
            return diagnostics

        interface_state = self.compute_interface_state(Ti_center, extra_features=extra_features)
        flux_divergence = self._flux_divergence(interface_state["Q_faces"])
        residual = flux_divergence - source_centers
        residual[0] = interface_state["Q_faces"][0]
        if T_edge is not None:
            residual[-1] = Ti_center[-1] - T_edge

        diagnostics.update(interface_state)
        diagnostics["Ti"] = Ti_center
        diagnostics["flux_divergence"] = flux_divergence
        diagnostics["residual"] = residual
        return diagnostics

    # ------------------------------------------------------------------
    # Finite-volume residual
    # ------------------------------------------------------------------
    def residual(
        self,
        Ti_center: np.ndarray,
        T_edge: Optional[float] = None,
        extra_features: Optional[Callable[[float, float], Dict]] = None,
    ) -> np.ndarray:
        """
        Steady-state finite-volume residual:

            R_i = [ (V_r Q)_{i+1/2} - (V_r Q)_{i-1/2} ] / DeltaV_i - S_i

        where DeltaV_i ≈ V_r(center_i) * Delta rho_i
        """
        Ti_center = np.asarray(Ti_center, dtype=float)
        if not np.all(np.isfinite(Ti_center)) or np.any(Ti_center <= 0.0):
            penalty = np.full_like(Ti_center, 1e8, dtype=float)
            finite = np.isfinite(Ti_center)
            penalty[finite] += 1e6 * np.maximum(-Ti_center[finite], 0.0)
            return penalty

        interface_state = self.compute_interface_state(Ti_center, extra_features=extra_features)
        Q_faces = interface_state["Q_faces"]
        R = self._flux_divergence(Q_faces) - np.asarray(
            self.heating_source(self.rho_centers),
            dtype=float,
        )

        # Axis regularity:
        # enforce zero inward flux at the first face by replacing first residual
        # with Q(rho_inner) ~ 0 if rho_inner is sufficiently close to axis.
        R[0] = Q_faces[0]

        # Optional fixed-edge-temperature boundary condition
        if T_edge is not None:
            R[-1] = Ti_center[-1] - T_edge

        return R

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        Ti_init: np.ndarray,
        T_edge: Optional[float] = None,
        extra_features: Optional[Callable[[float, float], Dict]] = None,
    ):
        assert Ti_init.shape == self.rho_centers.shape

        def F(x):
            return self.residual(x, T_edge=T_edge, extra_features=extra_features)

        if self.cfg.verbose:
            _log(
                "[transport] starting scipy root solve:"
                f" nr={len(Ti_init)}, max_iter={self.cfg.max_iter}"
            )
        sol = root(
            F,
            Ti_init,
            method="hybr",
            options={"maxfev": max(self.cfg.max_iter, 1) * (len(Ti_init) + 1)},
        )

        if self.cfg.verbose:
            _log(f"Converged: {sol.success}")
            _log(f"Message: {sol.message}")
            _log(f"Residual norm: {np.linalg.norm(sol.fun)}")

        diagnostics = self.build_diagnostics(
            sol.x,
            T_edge=T_edge,
            extra_features=extra_features,
        )

        return {
            "success": sol.success,
            "message": sol.message,
            "nfev": getattr(sol, "nfev", np.nan),
            "njev": getattr(sol, "njev", np.nan),
            "rho_centers": diagnostics["rho_centers"],
            "rho_faces": diagnostics["rho_faces"],
            "Ti": diagnostics["Ti"],
            "Q_faces": diagnostics["Q_faces"],
            "q_nodes": diagnostics["q_nodes"],
            "alpha_nodes": diagnostics["alpha_nodes"],
            "rho_nodes": diagnostics["rho_nodes"],
            "Ti_faces": diagnostics["Ti_faces"],
            "ni_faces": diagnostics["ni_faces"],
            "aLTi_faces": diagnostics["aLTi_faces"],
            "aLni_faces": diagnostics["aLni_faces"],
            "Vr_faces": diagnostics["Vr_faces"],
            "Vr_centers": diagnostics["Vr_centers"],
            "dr": diagnostics["dr"],
            "source_centers": diagnostics["source_centers"],
            "flux_divergence": diagnostics["flux_divergence"],
            "residual": diagnostics["residual"],
        }


# ----------------------------------------------------------------------
# Example user hooks
# ----------------------------------------------------------------------
def example_heating_source(rho: np.ndarray) -> np.ndarray:
    """
    Analytic volumetric heating source S(rho), e.g. W / m^3.
    Gaussian-ish core heating.
    """
    S0 = 2.0e6
    return S0 * np.exp(-(rho / 0.35) ** 2)


def example_density_profile(rho: np.ndarray) -> np.ndarray:
    """
    Prescribed density profile, e.g. m^-3.
    """
    n0 = 1.5e20
    return n0 * (1 - 0.7 * rho**2)


def example_gx_flux_model(rho, alpha, aLTi, aLni, Ti, ni, extra):
    """
    Placeholder for your trained GX NN surrogate.

    This toy model just makes flux grow with gradient and vary with alpha.
    Replace completely with your real model.
    """
    alpha = np.asarray(alpha, dtype=float)
    base = 5.0e4 * np.maximum(aLTi - 1.5, 0.0) ** 2
    mod = 1.0 + 0.2 * np.cos(alpha)
    return base * mod


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
def run_example(eq):
    cfg = TransportConfig(
        rho_inner=0.05,
        rho_outer=0.90,
        nr=16,
        n_alpha=8,
        M_grid=48,
        N_grid=48,
        verbose=True,
    )

    solver = StellaratorPowerBalanceSolver(
        eq=eq,
        gx_flux_model=example_gx_flux_model,
        heating_source=example_heating_source,
        density_profile=example_density_profile,
        config=cfg,
    )

    # Initial guess for T_i(rho) in eV or keV, as long as the surrogate uses the same units
    Ti_init = 8.0 * (1 - 0.85 * solver.rho_centers**2) + 1.0

    out = solver.solve(
        Ti_init=Ti_init,
        T_edge=1.0,  # fixed edge temperature
    )
    return out
