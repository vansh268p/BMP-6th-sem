# predict_poisson_ml.py
# ─────────────────────────────────────────────────────────────────────────────
# Single-sample Poisson solver inference for SYCL-PIC integration.
#
# The ML model operates on a 256×256 grid (matching training data).
# The SYCL simulation uses a 257×257 grid.  We crop rho to 256×256 for
# inference and pad phi back to 257×257 (boundaries = 0).
#
# CRITICAL: Normalization constants MUST match utils.py training normalization!
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import tempfile
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils.checkpoint import load_checkpoint
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch


# ── Normalization constants — MUST match utils.py / training exactly! ────────
# Training data rho is in SI units (C/m³, ~1e-8 scale).
# The SYCL simulation provides rho in sim units: rho_sim = rho_SI / eps0 (~1e3).
# We convert sim→SI by multiplying by EPSILON before dividing by RHO_NORM.
RHO_NORM = 1.23223e-08    # charge density normalization (std of training rho)
PHI_NORM = 9.21588e-01    # potential normalization (std of training phi)
EPSILON  = 8.854e-12      # vacuum permittivity (F/m) — sim→SI conversion

GRID_SIZE = 257            # model AND simulation grid are both 257×257


# ── Model wrapper — matches training architecture ───────────────────────────
class MdlsSymWrapper(Arch):
    """
    Multi-basis DeepONet wrapper for Poisson solver.

    phi(x,y) = Σ_k  FNO_k(ρ)(x,y) * FC_k(x,y)   when num_basis > 1
    phi(x,y) = FNO(ρ)(x,y) * FC(x,y)              when num_basis = 1

    Includes Dirichlet boundary condition enforcement (grounded walls).
    """

    def __init__(
        self,
        input_keys=[Key("rho"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=None,
        branch_net=None,
        num_basis=1,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.num_basis = num_basis

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {
                rho: dict_tensor[rho].view(xy_input_shape[0], -1, 1)
                for rho in ["x", "y"]
            },
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        fc_out  = self.trunk_net(xy)                         # (B, N, num_basis)
        fno_out = self.branch_net(dict_tensor["rho_prime"])  # (B, num_basis, H, W)

        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )  # (B, num_basis, H, W)

        if self.num_basis > 1:
            out = (fc_out * fno_out).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            out = fc_out * fno_out  # (B, 1, H, W)

        # ── Enforce Dirichlet BC (grounded walls) ────────────────────────
        # 1. Remove scalar drift (null space of Laplacian)
        boundary_mean = (
            out[:, :, 0, :].mean(dim=-1) + out[:, :, -1, :].mean(dim=-1) +
            out[:, :, :, 0].mean(dim=-1) + out[:, :, :, -1].mean(dim=-1)
        ) / 4.0
        out = out - boundary_mean.view(-1, 1, 1, 1)

        # 2. Hard clamp edges to zero (matching PIC grounded walls)
        out[:, :, 0, :] = 0.0
        out[:, :, -1, :] = 0.0
        out[:, :, :, 0] = 0.0
        out[:, :, :, -1] = 0.0

        return self.split_output(out, self.output_key_dict, dim=1)


def load_rho_from_textfile(filepath: str, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Load a space-separated text file of doubles."""
    rho = np.loadtxt(filepath)
    if rho.shape != (grid_size, grid_size):
        raise ValueError(
            f"Expected rho shape ({grid_size}, {grid_size}), got {rho.shape}"
        )
    return rho


def rho_to_temp_hdf5(rho: np.ndarray, tmp_dir: str) -> str:
    """
    Convert a raw 2D rho array into a single-sample HDF5 file with the same
    layout the training dataset uses: keys 'rho' and 'potential', each with
    shape (1, 1, G, G). The potential is filled with zeros (placeholder).
    """
    tmp_path = os.path.join(tmp_dir, "single_input.hdf5")
    rho_4d = rho[np.newaxis, np.newaxis, :, :].astype(np.float32)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("rho", data=rho_4d)
        f.create_dataset("potential", data=np.zeros_like(rho_4d))
    return tmp_path


def build_coordinate_grids(device: torch.device):
    """Build the (x, y) meshgrid tensors for the 257×257 grid."""
    lin = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    x_t = torch.from_numpy(xx).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    y_t = torch.from_numpy(yy).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    return x_t, y_t


def build_model(cfg, device: torch.device):
    """Instantiate branch (FNO) + trunk (FC) + wrapper, matching training."""
    num_basis = int(cfg.model.get("num_basis", cfg.model.fno.out_channels))

    model_branch = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    ).to(device)

    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
        num_basis=num_basis,
    ).to(device)

    return model, model_branch, model_trunk


class PoissonPredictor:
    """Load the trained model once and run repeated single-sample inference.

    Handles the 257→256 crop / 256→257 pad automatically so callers can
    pass simulation-sized arrays without worrying about the model grid.
    """

    def __init__(self, ckpt_dir: str, config_path: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(config_path)
        self.model, self.model_branch, self.model_trunk = build_model(self.cfg, self.device)
        load_checkpoint(
            ckpt_dir,
            models=[self.model_branch, self.model_trunk],
            device=self.device,
        )
        self.model.eval()
        self.x_coord, self.y_coord = build_coordinate_grids(self.device)

    def predict(self, rho_raw: np.ndarray) -> np.ndarray:
        """Predict phi from a 257×257 rho array.

        The model was trained on 257×257 grids — same as the simulation.
        No grid resizing needed.
        """
        if rho_raw.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(
                f"Expected rho shape ({GRID_SIZE},{GRID_SIZE}), got {rho_raw.shape}"
            )

        # Convert simulation rho to training data convention, then normalise.
        # Simulation rho ∝ (n_e - n_i)/ε₀  (electrons contribute POSITIVELY)
        # Training rho = e*(n_i - n_e)      (ions contribute POSITIVELY)
        # These have OPPOSITE signs, hence we multiply by -EPSILON to flip.
        rho_tensor = (
            torch.from_numpy(np.asarray(rho_raw, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            * (-EPSILON)
            / RHO_NORM
        ).to(self.device)

        with torch.no_grad():
            out = self.model({"rho_prime": rho_tensor, "x": self.x_coord, "y": self.y_coord})

        phi = out["phi"][0, 0].detach().cpu().numpy().astype(np.float64) * PHI_NORM
        return phi

    def predict_into(self, rho_raw: np.ndarray, phi_out: np.ndarray) -> None:
        """Run inference and write into a caller-provided output buffer in-place."""
        phi_out[:, :] = self.predict(rho_raw)


_PREDICTOR_CACHE: dict[tuple[str, str, str], PoissonPredictor] = {}


def get_cached_predictor(
    ckpt_dir: str,
    config_path: str,
    device: str | None = None,
) -> PoissonPredictor:
    """Return a cached predictor instance for fast repeated calls."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (os.path.abspath(ckpt_dir), os.path.abspath(config_path), resolved_device)
    predictor = _PREDICTOR_CACHE.get(cache_key)
    if predictor is None:
        predictor_device = torch.device(resolved_device)
        predictor = PoissonPredictor(
            ckpt_dir=ckpt_dir,
            config_path=config_path,
            device=predictor_device,
        )
        _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def predict_from_array(
    rho_raw: np.ndarray,
    ckpt_dir: str,
    config_path: str,
    device: str | None = None,
) -> np.ndarray:
    """Convenience API: in-memory rho -> in-memory phi."""
    predictor = get_cached_predictor(ckpt_dir=ckpt_dir, config_path=config_path, device=device)
    return predictor.predict(rho_raw)


def predict_into_arrays(
    rho_raw: np.ndarray,
    phi_out: np.ndarray,
    ckpt_dir: str,
    config_path: str,
    device: str | None = None,
) -> None:
    """Convenience API: in-memory rho -> write into caller-owned phi buffer."""
    predictor = get_cached_predictor(ckpt_dir=ckpt_dir, config_path=config_path, device=device)
    predictor.predict_into(rho_raw, phi_out)


def save_potential_textfile(phi: np.ndarray, filepath: str):
    """Save the predicted potential as a space-separated text file."""
    np.savetxt(filepath, phi, fmt="%.6e")
    print(f"Predicted potential saved to: {filepath}")


def visualize(rho: np.ndarray, phi_pred: np.ndarray,
              phi_expected, save_path: str):
    """
    Generate the output image.
    If phi_expected is provided → 4-panel: rho | pred | expected | error
    Otherwise → 2-panel: rho | pred
    """
    has_expected = phi_expected is not None

    ncols = 4 if has_expected else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 6))

    # Panel 1: Input charge density
    im0 = axes[0].imshow(rho, cmap="RdBu_r")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Input Charge Density (ρ)")

    # Panel 2: Predicted potential
    im1 = axes[1].imshow(phi_pred, cmap="viridis")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Predicted Potential (φ)")

    if has_expected:
        vmin, vmax = phi_expected.min(), phi_expected.max()
        axes[1].images[0].set_clim(vmin, vmax)

        # Panel 3: Expected potential
        im2 = axes[2].imshow(phi_expected, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title("Expected Potential (φ)")

        # Panel 4: Absolute error
        diff = np.abs(phi_pred - phi_expected)
        rel_err = diff.mean() / (np.abs(phi_expected).mean() + 1e-12)
        im3 = axes[3].imshow(diff, cmap="hot")
        plt.colorbar(im3, ax=axes[3])
        axes[3].set_title(f"|Error|  —  Mean rel err: {rel_err:.4%}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualisation saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run single-sample Poisson solver inference on a 2D rho text file."
    )
    parser.add_argument(
        "--rho_file", type=str, default="rho_sample",
        help="Path to the input rho text file (space-separated doubles)."
    )
    parser.add_argument(
        "--expected_file", type=str, default=None,
        help="Optional path to the expected potential text file for comparison."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./outputs_poisson_latest3/checkpoints",
        help="Path to the directory containing trained model checkpoints."
    )
    parser.add_argument(
        "--config", type=str, default="./conf/config_deeponet.yaml",
        help="Path to the Hydra YAML config used during training."
    )
    parser.add_argument(
        "--output_potential", type=str, default="predicted_potential.txt",
        help="Path for the output predicted-potential text file."
    )
    parser.add_argument(
        "--output_image", type=str, default="prediction_result.png",
        help="Path for the output visualisation PNG."
    )
    parser.add_argument(
        "--grid_size", type=int, default=None,
        help="Grid size of input (auto-detected from rho_file if not set)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"RHO_NORM = {RHO_NORM:.6e}, PHI_NORM = {PHI_NORM:.6e}")

    # ── Load model config ────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    # ── Read the input rho text file ─────────────────────────────────────────
    print(f"Reading rho from: {args.rho_file}")
    rho_raw = np.loadtxt(args.rho_file)
    grid_size = args.grid_size or rho_raw.shape[0]
    if rho_raw.shape != (grid_size, grid_size):
        raise ValueError(f"Expected ({grid_size}, {grid_size}), got {rho_raw.shape}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"rho range: [{rho_raw.min():.6e}, {rho_raw.max():.6e}]")

    # ── Normalise rho and prepare model input tensor ─────────────────────
    rho_tensor = (
        torch.from_numpy(rho_raw[:GRID_SIZE, :GRID_SIZE].astype(np.float32))
        .unsqueeze(0).unsqueeze(0)           # (1, 1, 257, 257)
        / RHO_NORM
    ).to(device)
    print(f"Normalized rho range: [{rho_tensor.min().item():.4f}, {rho_tensor.max().item():.4f}]")

    x_coord, y_coord = build_coordinate_grids(device)

    # ── Build and load model ─────────────────────────────────────────────
    model, model_branch, model_trunk = build_model(cfg, device)

    print(f"Loading checkpoint from: {args.ckpt_dir}")
    load_checkpoint(
        args.ckpt_dir,
        models=[model_branch, model_trunk],
        device=device,
    )
    model.eval()
    print("Checkpoint loaded successfully.")

    # ── Inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(
            {"rho_prime": rho_tensor, "x": x_coord, "y": y_coord}
        )

    # ── Denormalise the prediction back to physical units ────────────────────
    phi_pred = out["phi"][0, 0].cpu().numpy() * PHI_NORM
    print(f"Predicted potential range: [{phi_pred.min():.6e}, {phi_pred.max():.6e}]")

    # ── Save predicted potential as a 2D text file ───────────────────────────
    save_potential_textfile(phi_pred, args.output_potential)

    # ── Load expected potential if provided ──────────────────────────────────
    phi_expected = None
    if args.expected_file is not None:
        print(f"Reading expected potential from: {args.expected_file}")
        phi_expected = np.loadtxt(args.expected_file)

    # ── Generate visualisation ───────────────────────────────────────────────
    visualize(rho_raw, phi_pred, phi_expected, args.output_image)

    print("Done.")


if __name__ == "__main__":
    main()
