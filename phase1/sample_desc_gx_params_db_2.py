import os
import time
from pathlib import Path

import numpy as np

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from desc import set_device

set_device("gpu")

from desc.examples import get
from desc.external.gx import GX
from desc.objectives import NNITGProxy

from expensive_eval_modules import (
    evaluate_if_needed,
    flatten_selected_params,
    fsa_grad_x_from_desc,
    load_DB_safely,
    solve_npol_for_target_length,
)

# Flux-surface / field-line location used throughout this study, and the
# flux-tube length NNITGProxy's target_flux_tube_length assumes GX also uses.
GX_PSI = 0.5
GX_ALPHA = 0.0
TARGET_FLUX_TUBE_LENGTH = 75.4


def copy_selected_params(params_dict, param_names):
    return {name: np.array(params_dict[name], copy=True) for name in param_names}



def get_boundary_mode_entries(eq):
    """Return two boundary coefficients to vary for a 2D contour study."""
    return [
        ("Rb_lmn", eq.surface.R_basis.get_idx(M=1, N=2)),
        ("Zb_lmn", eq.surface.Z_basis.get_idx(M=-1, N=2)),
    ]


def flattened_index_for_entry(param_names, sizes, entry):
    """Map (param_name, local_index) to an index in the flattened x vector."""
    name, local_index = entry

    offset = 0
    for param_name in param_names:
        if param_name == name:
            return offset + int(local_index)
        offset += sizes[param_name]

    raise ValueError(f"Entry {entry!r} is not represented by param_names={param_names}.")


def make_boundary_mode_sample_points(eq, param_names, *, dx0=1e-4, dx1=1e-4):
    """Return full x vectors that vary only two selected boundary coefficients."""
    x0, shapes, sizes = flatten_selected_params(eq, param_names)
    selected_entries = get_boundary_mode_entries(eq)
    selected_flat_indices = [
        flattened_index_for_entry(param_names, sizes, entry)
        for entry in selected_entries
    ]

    i0, i1 = selected_flat_indices

    sample_points = []

    x = x0.copy()
    sample_points.append(x)

    x = x0.copy()
    x[i0] += dx0
    sample_points.append(x)

    x = x0.copy()
    x[i0] -= dx0
    sample_points.append(x)

    x = x0.copy()
    x[i1] += dx1
    sample_points.append(x)

    x = x0.copy()
    x[i1] -= dx1
    sample_points.append(x)

    return sample_points, selected_entries, selected_flat_indices, x0


def set_selected_params_on_eq(eq, x, param_names, shapes, sizes, *, solve_eq=True, solve_kwargs=None, uid=""):
    x = np.asarray(x, dtype=float).ravel()

    expected_size = sum(sizes[name] for name in param_names)
    if x.size != expected_size:
        raise ValueError(
            f"Expected x.size={expected_size} for param_names={param_names}, "
            f"but got x.size={x.size}."
        )

    offset = 0
    for name in param_names:
        size = sizes[name]

        old_raw = getattr(eq, name)
        old_arr = np.asarray(old_raw)

        if old_arr.shape == ():
            new_value = float(x[offset])
        else:
            new_value = x[offset : offset + size].reshape(old_arr.shape)

        setattr(eq, name, new_value)

        new_arr = np.asarray(new_value)
        check = np.asarray(getattr(eq, name))

        print(f"Updated eq.{name}", flush=True)
        print(f"  old shape   = {old_arr.shape}", flush=True)
        print(f"  new shape   = {new_arr.shape}", flush=True)
        print(f"  check shape = {check.shape}", flush=True)
        print(f"  old first   = {old_arr.ravel()[:min(5, old_arr.size)]}", flush=True)
        print(f"  new first   = {new_arr.ravel()[:min(5, new_arr.size)]}", flush=True)
        print(f"  check first = {check.ravel()[:min(5, check.size)]}", flush=True)

        offset += size

    solve_info = None
    if solve_eq:
        solve_kwargs = {} if solve_kwargs is None else dict(solve_kwargs)
        solve_kwargs.setdefault("ftol", 1e-3)
        solve_kwargs.setdefault("maxiter", 300)
        solve_kwargs.setdefault("verbose", 3)
        solve_kwargs.setdefault("copy", True)

        print("Solving equilibrium after parameter update...", flush=True)
        eq, solve_info = eq.solve(**solve_kwargs)
        eq.save(f"precise_qa_{uid}.h5")
        print("Finished equilibrium solve.", flush=True)

    return eq, solve_info



def build_turbulence_objective(
    eq,
    *,
    turbulence_objective,
    gx_path,
    gx_exec,
    gx_input_file,
    launch_cmd,
    model_dir,
    model_csv,
    uid="",
):
    gx_psi = GX_PSI
    gx_alpha = GX_ALPHA

    if turbulence_objective == "gx":
        npol = solve_npol_for_target_length(
            eq, psi=gx_psi, alpha=gx_alpha, target_length=TARGET_FLUX_TUBE_LENGTH
        )
        print(
            f"Solved npol={npol:.6f} for target_flux_tube_length="
            f"{TARGET_FLUX_TUBE_LENGTH}",
            flush=True,
        )
        return GX(
            eq,
            abs_step=1e-4,
            rel_step=0,
            processes=1,
            path=str(gx_path),
            npol=npol,
            nzgrid=48,
            alpha=gx_alpha,
            psi=gx_psi,
            sigma_Bxy=-1.0,
            gx_input_file=str(gx_input_file),
            launch_cmd=launch_cmd,
            gx_gpu=None,
            timeout=600,
            tmp_dir=f"tmp_GX_{uid}",
            save_tmp=True,
            name="GX",
            **{"exec": gx_exec},
        )

    if turbulence_objective == "nnitgproxy":
        return NNITGProxy(
            eq=eq,
            rho=np.array([np.sqrt(gx_psi)]),
            alpha=np.array([gx_alpha]),
            weight=1.0,
            ensemble_dir=str(model_dir),
            ensemble_csv=str(model_csv),
            n_ensemble=100,
            target_flux_tube_length=TARGET_FLUX_TUBE_LENGTH,
            a_over_LT=3.0,
            # Matches gx.in's fprim = [0.9, 0.0] (ion density gradient).
            a_over_Ln=0.9,
        )

    raise ValueError(
        f"Unknown turbulence_objective={turbulence_objective!r}. "
        "Use 'gx' or 'nnitgproxy'."
    )


def evaluate_at_params(
    x,
    *,
    param_names,
    turbulence_objective,
    gx_path,
    gx_exec,
    gx_input_file,
    launch_cmd,
    model_dir,
    model_csv,
):
    eq = get("precise_qa")

    x0, shapes, sizes = flatten_selected_params(eq, param_names)
    x = np.asarray(x, dtype=float).ravel()

    # Unique identifier based on the perturbation of the two adjusted values.
    selected_entries = get_boundary_mode_entries(eq)
    selected_flat_indices = [
        flattened_index_for_entry(param_names, sizes, entry)
        for entry in selected_entries
    ]
    d0 = float(x[selected_flat_indices[0]] - x0[selected_flat_indices[0]])
    d1 = float(x[selected_flat_indices[1]] - x0[selected_flat_indices[1]])
    uid = f"dx0_{d0:.3e}_dx1_{d1:.3e}"

    base_params = copy_selected_params(eq.params_dict, param_names)

    print(f"Nominal x0 = {x0}", flush=True)
    print(f"Input x    = {x}", flush=True)

    eq, solve_info = set_selected_params_on_eq(
        eq,
        x,
        param_names,
        shapes,
        sizes,
        solve_eq=True,
        solve_kwargs={"ftol": 1e-3, "maxiter": 300, "verbose": 3, "copy": True},
        uid=uid,
    )
    updated_params = copy_selected_params(eq.params_dict, param_names)

    obj = build_turbulence_objective(
        eq,
        turbulence_objective=turbulence_objective,
        gx_path=gx_path,
        gx_exec=gx_exec,
        gx_input_file=gx_input_file,
        launch_cmd=launch_cmd,
        model_dir=model_dir,
        model_csv=model_csv,
        uid=uid,
    )

    obj.build()

    t0 = time.time()
    f = obj.compute(eq.params_dict)
    elapsed = time.time() - t0

    value = float(np.ravel(f)[0])
    metadata = {
        "elapsed_sec": elapsed,
        "success": True,
        "turbulence_objective": turbulence_objective,
        "param_names": param_names,
        "x": x.copy(),
        "nominal_x0": x0.copy(),
        "base_params": base_params,
        "updated_params": updated_params,
        "solve_info": solve_info,
    }

    if turbulence_objective == "nnitgproxy":
        # NNITGProxy ~= Q_GX * <|grad(rho)|>, i.e. its raw output is in GX's
        # convention multiplied by the <grad(x)> factor. To recover Q_GX for
        # an apples-to-apples comparison, divide by that factor (not
        # multiply (ask Churchill)
        raw_value = value
        grad_x_factor = fsa_grad_x_from_desc(eq, s=GX_PSI)
        value = raw_value / grad_x_factor
        metadata["nnitgproxy_raw_value"] = raw_value
        metadata["grad_x_factor"] = grad_x_factor
        print(
            f"Applied grad_x correction: raw={raw_value:.6e} "
            f"/ factor={grad_x_factor:.6e} -> corrected={value:.6e}",
            flush=True,
        )

    calc_status = 0

    print(
        f"Evaluated {turbulence_objective} f = {value:.16e} "
        f"in {elapsed:.2f} sec",
        flush=True,
    )
    return value, metadata, calc_status


def main():
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent

    gx_path = Path(os.environ.get("TI_OPT_GX_DIR", repo_root / "gx")).resolve()
    gx_exec = os.environ.get("TI_OPT_GX_EXEC", "gx")
    gx_input_file = Path(
        os.environ.get("TI_OPT_GX_INPUT_FILE", script_dir / "gx.in")
    ).resolve()
    gx_launch = Path(
        os.environ.get("TI_OPT_GX_LAUNCH", script_dir / "gx-launch.sh")
    ).resolve()
    launch_cmd = [str(gx_launch)] if gx_launch.is_file() else None

    data_dir = Path(
        os.environ.get(
            "TI_OPT_NN_DATA_DIR",
            "/global/cfs/cdirs/m4505/rchurchi/ti_optimization/data/neural_networks",
        )
    ).resolve()
    model_dir = data_dir / "cyclic_invariant_models"
    model_csv = data_dir / "deephyper_exp" / "results.csv"

    # Switch here: "gx" for full GX, "nnitgproxy" for NN surrogate.
    # turbulence_objective = "nnitgproxy"
    turbulence_objective = "gx"

    database_name = (
        f"/pscratch/sd/j/jmlarson/"
        f"desc_{turbulence_objective}_boundary2_samples.npy"
    )

    # Absolute-value parameterization:
    # x contains the actual values to write into eq.params_dict, in flattened order.
    # We use the full Rb_lmn/Zb_lmn flattened vector, but sample_points below
    # vary only two selected boundary-mode coefficients.
    param_names = ["Rb_lmn", "Zb_lmn"]

    eq_tmp = get("precise_qa")
    x0_tmp, shapes_tmp, sizes_tmp = flatten_selected_params(eq_tmp, param_names)

    selected_entries = get_boundary_mode_entries(eq_tmp)
    selected_flat_indices = [
        flattened_index_for_entry(param_names, sizes_tmp, entry)
        for entry in selected_entries
    ]

    x_selected0 = x0_tmp[selected_flat_indices]

    n_grid = 5

    half_width0 = 2e-4
    half_width1 = 2e-4

    grid0 = x_selected0[0] + np.linspace(-half_width0, half_width0, n_grid)
    grid1 = x_selected0[1] + np.linspace(-half_width1, half_width1, n_grid)

    sample_points = []

    # for value0 in grid0:
    #     for value1 in grid1:
    #         x = x0_tmp.copy()
    #         x[selected_flat_indices[0]] = value0
    #         x[selected_flat_indices[1]] = value1
    #         sample_points.append(x)

    # Zero-perturbation case:
    x = x0_tmp.copy()
    sample_points.append(x)

    # Top-right corner (both parameters at their largest perturbation):
    x = x0_tmp.copy()
    x[selected_flat_indices[0]] = x_selected0[0] + half_width0
    x[selected_flat_indices[1]] = x_selected0[1] + half_width1
    sample_points.append(x)

    print(f"Generated {len(sample_points)} grid samples.", flush=True)

    print(f"database_name        = {database_name}", flush=True)
    print(f"turbulence_objective = {turbulence_objective}", flush=True)
    print(f"param_names          = {param_names}", flush=True)
    # print(f"selected_entries     = {selected_entries}", flush=True)
    # print(f"selected_flat_indices = {selected_flat_indices}", flush=True)
    # print(
    #     f"selected nominal values = {nominal_x0[selected_flat_indices]}",
    #     flush=True,
    # )
    # print(f"gx_path              = {gx_path}", flush=True)
    print(f"gx_exec              = {gx_exec}", flush=True)
    print(f"gx_input_file        = {gx_input_file}", flush=True)
    print(f"gx_launch            = {gx_launch}", flush=True)
    print(f"launch_cmd           = {launch_cmd}", flush=True)
    print(f"model_dir            = {model_dir}", flush=True)
    print(f"model_csv            = {model_csv}", flush=True)

    def expensive_eval(x):
        try:
            return evaluate_at_params(
                x,
                param_names=param_names,
                turbulence_objective=turbulence_objective,
                gx_path=gx_path,
                gx_exec=gx_exec,
                gx_input_file=gx_input_file,
                launch_cmd=launch_cmd,
                model_dir=model_dir,
                model_csv=model_csv,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            value = np.nan
            metadata = {
                "elapsed_sec": np.nan,
                "success": False,
                "error": repr(e),
                "turbulence_objective": turbulence_objective,
                "param_names": param_names,
                "x": np.asarray(x, dtype=float).ravel().copy(),
            }
            calc_status = 1
            print(f"Evaluation failed: {repr(e)}", flush=True)
            return value, metadata, calc_status

    for k, x in enumerate(sample_points):
        x = np.asarray(x, dtype=float).ravel()

        print("=" * 80, flush=True)
        print(f"Sample {k} of {len(sample_points) - 1}", flush=True)
        print(f"x selected values = {x[selected_flat_indices]}", flush=True)

        entry, did_eval = evaluate_if_needed(database_name, x, expensive_eval)

        print(f"did_eval    = {did_eval}", flush=True)
        print(f"calc_status = {entry['calc_status']}", flush=True)
        print(f"values      = {entry['values']}", flush=True)
        print(f"metadata keys = {list(entry['metadata'].keys())}", flush=True)

    DB = load_DB_safely(database_name)
    print("=" * 80, flush=True)
    print(f"Database now has {len(DB)} entries.", flush=True)


if __name__ == "__main__":
    main()
