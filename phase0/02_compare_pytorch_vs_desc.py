#!/usr/bin/env python
"""Compare PyTorch vs DESC (JAX) CNN inference on pre-computed GX data.

Loads the flux-tube tensor and GX simulation results, runs both PyTorch
CyclicInvariantNet and DESC's JIT-compiled forward pass on the test set,
and compares R^2 scores and per-sample predictions.
"""

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import csv
import os
import pickle
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from desc.compute._turbulence import (
    _convert_pytorch_weights,
    _make_jit_forward,
)

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")
ML_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "data_for_ML")
TENSOR_PATH = os.path.join(
    ML_DATA_DIR, "20241210-01-assembleFluxTubeTensor_allConfigs_filtered.pkl"
)
GX_FIXED_PATH = os.path.join(
    ML_DATA_DIR,
    "20241210-01-GX_results_for_fixed_gradients_allConfigs_filtered.pkl",
)
GX_RANDOM_PATH = os.path.join(
    ML_DATA_DIR,
    "20241210-01-GX_results_for_random_gradients_allConfigs_filtered.pkl",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ---------------------------------------------------------------------------
# Data loading (reproduces Cyclic_net.data_load for dataset_type=5, pre_method=2)
# ---------------------------------------------------------------------------
def load_test_data():
    """Load tensor + GX results and return the test split.

    Returns
    -------
    test_signals : np.ndarray, shape (n_test, 96, 7)
    test_targets : np.ndarray, shape (n_test,)   log(Q) clipped at -2
    test_tprims : np.ndarray, shape (n_test,)
    test_fprims : np.ndarray, shape (n_test,)
    """
    import torch
    from torch.utils.data import TensorDataset, random_split

    with open(TENSOR_PATH, "rb") as f:
        in_data1 = pickle.load(f)
    with open(GX_FIXED_PATH, "rb") as f:
        out_fixed = pickle.load(f)
    with open(GX_RANDOM_PATH, "rb") as f:
        out_random = pickle.load(f)

    # Reproduce data_load dataset_type=5 logic:
    # Negate fixed-gradient tprims (trick to filter test set later)
    out_fixed["tprims"] = -1 * out_fixed["tprims"]

    # Duplicate tensor for both subsets, concatenate outputs
    tensor = np.concatenate((in_data1["tensor"], in_data1["tensor"]))
    Q_all = np.concatenate(
        (out_fixed["Q_avgs_without_FSA_grad_x"],
         out_random["Q_avgs_without_FSA_grad_x"])
    )
    tprims = np.concatenate((out_fixed["tprims"], out_random["tprims"]))
    fprims = np.concatenate((out_fixed["fprims"], out_random["fprims"]))

    # Filter positive Q
    mask = Q_all > 0
    n_dropped = len(Q_all) - mask.sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} samples with Q <= 0")
    tensor = tensor[mask]
    Q_all = Q_all[mask]
    tprims = tprims[mask]
    fprims = fprims[mask]

    # pre_method=2: log(Q) with clipping at -2
    log_Q = np.log(Q_all)
    log_Q[log_Q < -2] = -2

    # 80/10/10 split (must match training seed)
    torch.manual_seed(42)
    device = torch.device("cpu")
    X = torch.FloatTensor(tensor).to(device)
    y = torch.FloatTensor(log_Q).to(device)
    tp = torch.FloatTensor(tprims).to(device)
    fp = torch.FloatTensor(fprims).to(device)
    dataset = TensorDataset(X, y, tp, fp)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Keep only random-gradient samples in test set (tprims >= 0)
    kept = [i for i in test_dataset.indices if dataset[i][2] >= 0]
    test_sub = torch.utils.data.Subset(dataset, kept)

    n_test = len(test_sub)
    signals = np.empty((n_test, 96, 7), dtype=np.float32)
    targets = np.empty(n_test, dtype=np.float32)
    t_arr = np.empty(n_test, dtype=np.float32)
    f_arr = np.empty(n_test, dtype=np.float32)
    for j in range(n_test):
        s = test_sub[j]
        signals[j] = s[0].numpy()
        targets[j] = s[1].item()
        t_arr[j] = s[2].item()
        f_arr[j] = s[3].item()

    return signals, targets, t_arr, f_arr


# ---------------------------------------------------------------------------
# Hyperparameter + model loading
# ---------------------------------------------------------------------------
def load_hyperparameters(csv_path, job_ids):
    """Read hyperparameters for given job IDs from DeepHyper CSV."""
    hp_dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            jid = row["m:task_id"]
            if jid not in job_ids:
                continue
            hp_dict[jid] = {
                "use_batch_norm": row.get("p:use_batch_norm", "True").lower()
                in ("true", "1"),
                "dataset": 5,
            }
            for i in range(1, 6):
                hp_dict[jid][f"kernel_size{i}"] = int(
                    row.get(f"p:kernel_size{i}", 3)
                )
                hp_dict[jid][f"conv_channels{i}"] = int(
                    row.get(f"p:conv_channels{i}", 16)
                )
            for i in range(1, 3):
                hp_dict[jid][f"fc_dim{i}"] = int(row.get(f"p:fc_dim{i}", 16))
    return hp_dict


def get_top_model_ids(csv_path, top_k=10, pre_method=2):
    """Return top-k model task IDs sorted by objective (descending)."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["objective"] != "F"]
    if rows and "p:pre_method" in rows[0]:
        rows = [r for r in rows if int(r["p:pre_method"]) == pre_method]
    for r in rows:
        r["objective"] = float(r["objective"])
    rows.sort(key=lambda r: r["objective"], reverse=True)
    return [r["m:task_id"] for r in rows[:top_k]]


# ---------------------------------------------------------------------------
# Inline CyclicInvariantNet
# ---------------------------------------------------------------------------
def _build_pytorch_model(hp):
    """Build a CyclicInvariantNet from hyperparameters. Requires torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CyclicInvariantNet(nn.Module):
        def __init__(self, hp):
            super(CyclicInvariantNet, self).__init__()
            self.use_batch_norm = hp["use_batch_norm"]
            self.dataset = hp["dataset"]  ## dataset type

            # Convolutional layers for the 96-dimensional coordinate
            self.conv_layers = nn.ModuleList()
            self.conv_batch_norms = nn.ModuleList() if self.use_batch_norm else None
            self.pool_layers = nn.ModuleList()

            conv_channels = [7] + [hp[f"conv_channels{i+1}"] for i in range(5)]
            kernel_sizes = [hp[f"kernel_size{i+1}"] for i in range(5)]

            for i in range(len(conv_channels) - 1):
                self.conv_layers.append(
                    nn.Conv1d(
                        conv_channels[i],
                        conv_channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        padding="same",
                        padding_mode="circular",
                    )
                )
                if self.use_batch_norm:
                    self.conv_batch_norms.append(nn.BatchNorm1d(conv_channels[i + 1]))

                self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            # Global average pooling
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

            # Dense layers
            self.fc_layers = nn.ModuleList()
            self.fc_batch_norms = nn.ModuleList() if self.use_batch_norm else None

            extra = 0
            if (self.dataset == 4) or (self.dataset == 5) or (self.dataset == 6):
                extra = 2
            fc_dims = [conv_channels[-1] + extra] + [hp[f"fc_dim{i+1}"] for i in range(2)]
            for i in range(len(fc_dims) - 1):
                self.fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))
                if self.use_batch_norm:
                    self.fc_batch_norms.append(nn.BatchNorm1d(fc_dims[i + 1]))

            self.output_layer = nn.Linear(fc_dims[-1], 1)

        def forward(self, x, tprims=None, fprims=None):
            # x shape: (batch_size, 96, 7)
            # Transpose to (batch_size, 7, 96) for conv1d
            x = x.transpose(1, 2)

            # Apply convolutional layers
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x)
                if self.use_batch_norm:
                    x = self.conv_batch_norms[i](x)
                x = F.relu(x)
                x = self.pool_layers[i](x)

            # Global average pooling
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            if (self.dataset == 4) or (self.dataset == 5) or (self.dataset == 6):
                x = torch.cat((x, tprims.unsqueeze(1), fprims.unsqueeze(1)), dim=1)

            # Apply dense layers
            for i, fc_layer in enumerate(self.fc_layers):
                x = fc_layer(x)
                if self.use_batch_norm:
                    x = self.fc_batch_norms[i](x)
                x = F.relu(x)

            # Output layer
            x = self.output_layer(x)
            return x

    return CyclicInvariantNet(hp)


# ---------------------------------------------------------------------------
# Inference routines
# ---------------------------------------------------------------------------
def run_pytorch_inference(signals, tprims, fprims, n_ensemble, batch_size=64):
    """Run PyTorch ensemble. Returns (mean_log_Q, std_log_Q) or (None, None)."""
    try:
        import torch
    except ImportError:
        print("  PyTorch not available, skipping")
        return None, None

    job_ids = get_top_model_ids(MODEL_CSV, top_k=n_ensemble, pre_method=2)
    hp_dict = load_hyperparameters(MODEL_CSV, set(job_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for jid in job_ids:
        path = os.path.join(MODEL_DIR, f"model_{jid}.pth")
        if not os.path.exists(path) or jid not in hp_dict:
            continue
        sd = torch.load(path, map_location=device, weights_only=False)
        model = _build_pytorch_model(hp_dict[jid])
        model.load_state_dict(sd, strict=True)
        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        print("  No PyTorch models loaded")
        return None, None

    print(f"  Loaded {len(models)} PyTorch models")
    X = torch.FloatTensor(signals).to(device)
    tp = torch.FloatTensor(tprims).to(device)
    fp = torch.FloatTensor(fprims).to(device)

    all_preds = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb, tb, fb = X[start:end], tp[start:end], fp[start:end]
            batch_preds = [m(xb, tb, fb).squeeze(-1) for m in models]
            all_preds.append(torch.stack(batch_preds).cpu().numpy())

    elapsed = time.time() - t0
    preds = np.concatenate(all_preds, axis=1)  # (n_models, n_samples)
    mean_log_Q = preds.mean(axis=0)
    std_log_Q = preds.std(axis=0)
    print(f"  Inference: {elapsed:.2f}s")
    return mean_log_Q, std_log_Q


def run_desc_inference(signals, tprims, fprims, n_ensemble, batch_size=64):
    """Run DESC JIT ensemble. Returns (mean_log_Q, std_log_Q)."""
    import torch

    job_ids = get_top_model_ids(MODEL_CSV, top_k=n_ensemble, pre_method=2)

    models = []
    for jid in job_ids:
        path = os.path.join(MODEL_DIR, f"model_{jid}.pth")
        if not os.path.exists(path):
            continue
        sd = torch.load(path, map_location="cpu", weights_only=False)
        weights = _convert_pytorch_weights(sd)
        models.append(_make_jit_forward(weights))

    # Warm up JIT
    dummy_s = jnp.zeros((1, 7, 96), dtype=jnp.float32)
    dummy_sc = jnp.zeros((1, 2), dtype=jnp.float32)
    for f in models:
        f(dummy_s, dummy_sc).block_until_ready()
    print(f"  Loaded {len(models)} DESC JIT models")

    # DESC expects (N, 7, 96) -- transpose from (N, 96, 7)
    signals_desc = np.transpose(signals, (0, 2, 1))
    scalars = np.column_stack([tprims, fprims])
    signals_jax = jnp.array(signals_desc, dtype=jnp.float32)
    scalars_jax = jnp.array(scalars, dtype=jnp.float32)

    all_mean = []
    all_std = []
    t0 = time.time()
    for start in range(0, len(signals), batch_size):
        end = min(start + batch_size, len(signals))
        sb = signals_jax[start:end]
        scb = scalars_jax[start:end]
        preds = [f(sb, scb) for f in models]
        stacked = jnp.stack(preds)  # (n_models, batch, 1)
        all_mean.append(np.asarray(jnp.mean(stacked, axis=0)).flatten())
        all_std.append(np.asarray(jnp.std(stacked, axis=0)).flatten())

    elapsed = time.time() - t0
    mean_log_Q = np.concatenate(all_mean)
    std_log_Q = np.concatenate(all_std)
    print(f"  Inference: {elapsed:.2f}s")
    return mean_log_Q, std_log_Q


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def r2_score(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(actual_log, pred_log, pred_std_log, r2, title, path):
    """Scatter plot: predicted vs actual Q on log-log scale."""
    actual_Q = np.exp(actual_log)
    pred_Q = np.exp(pred_log)
    lower = np.exp(pred_log - pred_std_log)
    upper = np.exp(pred_log + pred_std_log)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.vlines(actual_Q, lower, upper, alpha=0.4, linewidth=0.5, color="gray")
    ax.scatter(actual_Q, pred_Q, s=1.5, alpha=0.5, color="r", zorder=10)
    ax.plot([1e-1, 1e3], [1e-1, 1e3], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-1, 1e3])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("True Q from GX")
    ax.set_ylabel("Predicted Q")
    ax.set_title(title)
    ax.text(0.04, 0.92, f"R^2 = {r2:.4f}", transform=ax.transAxes)
    ax.text(0.04, 0.86, f"N = {len(actual_Q):,}", transform=ax.transAxes)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_pytorch_vs_desc(pt_Q, desc_Q, r2, path):
    """Scatter plot: PyTorch Q vs DESC Q."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(desc_Q, pt_Q, s=1.5, alpha=0.5, color="r", zorder=10)
    q_range = [1e-1, 1e3]
    ax.plot(q_range, q_range, "k--", lw=1, label="Perfect Agreement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(q_range)
    ax.set_ylim(q_range)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("DESC Predicted Q")
    ax.set_ylabel("PyTorch Predicted Q")
    ax.set_title("PyTorch vs DESC Ensemble")

    ax.text(0.04, 0.92, f"R^2 = {r2:.4f}", transform=ax.transAxes)
    ax.text(0.04, 0.86, f"N = {len(desc_Q):,}", transform=ax.transAxes)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PyTorch vs DESC CNN Comparison")
    print("=" * 60)

    for p in [MODEL_DIR, TENSOR_PATH, GX_FIXED_PATH, GX_RANDOM_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: Not found: {p}")
            return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_ensemble = 10

    # --- Load data ---
    print("\n1. Loading test data (dataset_type=5, pre_method=2)...")
    signals, targets, tprims, fprims = load_test_data()
    print(f"  Test set: {len(signals)} samples")
    print(f"  Signals shape: {signals.shape}")
    print(f"  Target (log Q) range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"  a/LT range: [{tprims.min():.2f}, {tprims.max():.2f}]")
    print(f"  a/Ln range: [{fprims.min():.2f}, {fprims.max():.2f}]")

    # --- DESC inference ---
    print(f"\n2. Running DESC ensemble ({n_ensemble} models)...")
    desc_log_Q, desc_std = run_desc_inference(
        signals, tprims, fprims, n_ensemble
    )
    desc_r2 = r2_score(targets, desc_log_Q)
    print(f"  DESC R^2 vs GX: {desc_r2:.6f}")

    # --- PyTorch inference ---
    print(f"\n3. Running PyTorch ensemble ({n_ensemble} models)...")
    pt_log_Q, pt_std = run_pytorch_inference(
        signals, tprims, fprims, n_ensemble
    )

    has_pytorch = pt_log_Q is not None
    if has_pytorch:
        pt_r2 = r2_score(targets, pt_log_Q)
        print(f"  PyTorch R^2 vs GX: {pt_r2:.6f}")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Test samples:       {len(signals):,}")
    print(f"  Ensemble size:      {n_ensemble}")
    print(f"  DESC R^2 vs GX:     {desc_r2:.6f}")

    if has_pytorch:
        comparison_r2 = r2_score(np.exp(pt_log_Q), np.exp(desc_log_Q))
        diff_log = pt_log_Q - desc_log_Q
        print(f"  PyTorch R^2 vs GX:  {pt_r2:.6f}")
        print(f"  PyTorch vs DESC R^2: {comparison_r2:.6f}")
        print(f"  Max |diff| (log Q): {np.max(np.abs(diff_log)):.6e}")
        print(f"  Mean |diff| (log Q): {np.mean(np.abs(diff_log)):.6e}")

    # --- Plots ---
    print(f"\n4. Generating plots...")

    plot_pred_vs_actual(
        targets, desc_log_Q, desc_std, desc_r2,
        "DESC Ensemble vs GX",
        os.path.join(OUTPUT_DIR, "02_desc_vs_actual.png"),
    )

    if has_pytorch:
        plot_pred_vs_actual(
            targets, pt_log_Q, pt_std, pt_r2,
            "PyTorch Ensemble vs GX",
            os.path.join(OUTPUT_DIR, "02_pytorch_vs_actual.png"),
        )
        plot_pytorch_vs_desc(
            np.exp(pt_log_Q), np.exp(desc_log_Q), comparison_r2,
            os.path.join(OUTPUT_DIR, "02_pytorch_vs_desc.png"),
        )

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
