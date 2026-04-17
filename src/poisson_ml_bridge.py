#!/usr/bin/env python3
"""Bridge module to run external ML Poisson inference in-process.

This keeps ML logic outside `main.py` while allowing zero-copy interaction with
SYCL shared-memory buffers exposed as NumPy arrays by `sycl_pic`.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_PREDICT_SCRIPT = Path(__file__).resolve().parent / "predict_poisson_ml.py"
DEFAULT_CKPT_DIR = Path("/home/student/Dhruvil_Om_BMP_2026/PoissonSolver/outputs_poisson_v2/final_checkpoints")
DEFAULT_CONFIG = Path("/home/student/Dhruvil_Om_BMP_2026/PoissonSolver/outputs_poisson_v2/.hydra/config.yaml")


class ExternalPoissonMLSolver:
    """Load external predictor once and solve rho->phi into provided buffers."""

    def __init__(
        self,
        predict_script: Path | str = DEFAULT_PREDICT_SCRIPT,
        ckpt_dir: Path | str = DEFAULT_CKPT_DIR,
        config_path: Path | str = DEFAULT_CONFIG,
        device: str | None = None,
    ):
        self.predict_script = Path(predict_script)
        self.ckpt_dir = str(Path(ckpt_dir))
        self.config_path = str(Path(config_path))
        self.device = device
        self.debug = os.environ.get("ML_POISSON_DEBUG", "0") == "1"
        self.debug_every = int(os.environ.get("ML_POISSON_DEBUG_EVERY", "100"))
        self._calls = 0

        if not self.predict_script.exists():
            raise FileNotFoundError(f"Predictor script not found: {self.predict_script}")

        self._predict_module = self._load_module(self.predict_script)

    @staticmethod
    def _load_module(script_path: Path):
        spec = importlib.util.spec_from_file_location("external_predict_poisson", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def solve_inplace(
        self,
        rho_flat: np.ndarray,
        phi_flat: np.ndarray,
        grid_x: int,
        grid_y: int,
        iteration: int | None = None,
    ) -> float:
        """Predict phi from rho and write in-place to the provided flat buffer."""
        t0 = time.perf_counter()
        rho_2d = np.asarray(rho_flat).reshape(grid_y, grid_x)
        phi_2d = np.asarray(phi_flat).reshape(grid_y, grid_x)

        if hasattr(self._predict_module, "predict_into_arrays"):
            self._predict_module.predict_into_arrays(
                rho_2d,
                phi_2d,
                ckpt_dir=self.ckpt_dir,
                config_path=self.config_path,
                device=self.device,
            )
        elif hasattr(self._predict_module, "predict_from_array"):
            phi_pred = self._predict_module.predict_from_array(
                rho_2d,
                ckpt_dir=self.ckpt_dir,
                config_path=self.config_path,
                device=self.device,
            )
            phi_2d[:, :] = phi_pred
        else:
            raise RuntimeError(
                "External predictor does not expose predict_into_arrays/predict_from_array"
            )
        elapsed = time.perf_counter() - t0
        self._calls += 1

        if self.debug:
            do_log = self._calls <= 3
            if iteration is not None and self.debug_every > 0 and iteration % self.debug_every == 0:
                do_log = True
            if do_log:
                sample = (
                    float(phi_2d[0, 0]),
                    float(phi_2d[grid_y // 2, grid_x // 2]),
                    float(phi_2d[-1, -1]),
                )
                print(
                    f"[ML-Poisson] iter={iteration} call={self._calls} "
                    f"infer_time={elapsed:.6f}s phi_samples={sample}"
                )

        return elapsed
