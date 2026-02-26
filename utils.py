#!/usr/bin/env python3
"""Helpers for downloading NNITGProxy model assets."""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Union

URL_WEIGHTS = "https://drive.google.com/uc?id=1H4RrddZp8iGizTcqISttpWX54eYQa4Bo"
URL_RESULTS = "https://drive.google.com/uc?id=1v3BAQpytXqnrXyuwCTlUo6u1mXjEyHFY"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_model_dir() -> Path:
    return _repo_root() / "data" / "neural_networks" / "cyclic_invariant_models"


def _download_archive(url: str, archive_path: Path) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise ModuleNotFoundError(
            "gdown is required to download model assets. Install with: pip install gdown"
        ) from exc

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(archive_path), quiet=False)
    if not archive_path.exists():
        raise FileNotFoundError(f"Download failed for {url}")
    return archive_path


def _extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar:
        tar.extractall(path=extract_dir)
    return extract_dir


def _find_model_dir(extract_dir: Path) -> Path:
    candidates = [
        extract_dir / "cyclic_invariant_models",
        extract_dir / "neural_networks" / "cyclic_invariant_models",
        extract_dir / "data" / "neural_networks" / "cyclic_invariant_models",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = sorted(p for p in extract_dir.rglob("cyclic_invariant_models") if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"Could not find cyclic_invariant_models in extracted archive at {extract_dir}"
        )
    return matches[0]


def _find_results_csv(extract_dir: Path) -> Path:
    candidates = [
        extract_dir / "deephyper_exp" / "results.csv",
        extract_dir / "neural_networks" / "deephyper_exp" / "results.csv",
        extract_dir / "data" / "neural_networks" / "deephyper_exp" / "results.csv",
        extract_dir / "results.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(
        p for p in extract_dir.rglob("results.csv") if p.is_file() and p.name == "results.csv"
    )
    if not matches:
        raise FileNotFoundError(f"Could not find results.csv in extracted archive at {extract_dir}")
    return matches[0]


def download_model_weights(
    file_path: Optional[Union[str, Path]] = None, force: bool = False
) -> None:
    """Download NN model assets to expected DESC objective locations.

    Expected layout:
      <repo>/data/neural_networks/cyclic_invariant_models
      <repo>/data/neural_networks/deephyper_exp/results.csv

    Parameters
    ----------
    file_path:
        Target path for the cyclic_invariant_models directory. If omitted, defaults to:
        <repo>/data/neural_networks/cyclic_invariant_models
    force:
        If True, re-download and replace any existing assets.
    """

    model_dir = Path(file_path).expanduser().resolve() if file_path else _default_model_dir()
    data_dir = model_dir.parent
    model_csv = data_dir / "deephyper_exp" / "results.csv"

    needs_weights = force or not model_dir.is_dir()
    needs_csv = force or not model_csv.is_file()

    if not needs_weights and not needs_csv:
        print(f"Model assets already exist under: {data_dir}")
        return

    print(f"Ensuring model assets under: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nnitg_weights_") as tmp:
        tmp_root = Path(tmp)

        if needs_weights:
            print("Downloading cyclic_invariant_models archive...")
            weights_archive = _download_archive(URL_WEIGHTS, tmp_root / "weights.tar.gz")
            weights_extract = _extract_archive(weights_archive, tmp_root / "weights_extract")
            src_model_dir = _find_model_dir(weights_extract)
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.copytree(src_model_dir, model_dir)
            print(f"Installed model directory: {model_dir}")

        if needs_csv:
            print("Downloading deephyper_exp/results.csv archive...")
            results_archive = _download_archive(URL_RESULTS, tmp_root / "results.tar.gz")
            results_extract = _extract_archive(results_archive, tmp_root / "results_extract")
            src_results_csv = _find_results_csv(results_extract)
            shutil.copy2(src_results_csv, model_csv)
            print(f"Installed model CSV: {model_csv}")

    print("Download and extraction completed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NNITGProxy model assets into <repo>/data/neural_networks."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Target cyclic_invariant_models directory. "
            "Default: <repo>/data/neural_networks/cyclic_invariant_models"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and replace existing files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_model_weights(file_path=args.model_dir, force=args.force)
