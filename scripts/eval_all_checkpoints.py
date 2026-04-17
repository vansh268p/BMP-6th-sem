#!/usr/bin/env python3
"""Evaluate ALL checkpoint directories against validation data.
Reports MAE, relative error, and max error for each checkpoint.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import h5py
from omegaconf import OmegaConf
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils.checkpoint import load_checkpoint
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch

# Data normalization (from utils.py — same for all runs)
RHO_NORM = 1.23223e-08
PHI_NORM = 9.21588e-01

# ── Model wrapper (same as predict_poisson_ml.py) ──────────────────────────
class MdlsSymWrapper(Arch):
    def __init__(self, input_keys=[Key("rho"), Key("x"), Key("y")],
                 output_keys=[Key("phi")], trunk_net=None, branch_net=None, num_basis=1):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.num_basis = num_basis

    def forward(self, dict_tensor):
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {rho: dict_tensor[rho].view(xy_input_shape[0], -1, 1) for rho in ["x", "y"]},
            ["x", "y"], detach_dict=self.detach_key_dict, dim=-1,
        )
        fc_out = self.trunk_net(xy)
        fno_out = self.branch_net(dict_tensor["rho_prime"])
        fc_out = fc_out.view(xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1])
        if self.num_basis > 1:
            out = (fc_out * fno_out).sum(dim=1, keepdim=True)
        else:
            out = fc_out * fno_out
        # Dirichlet BC enforcement
        boundary_mean = (
            out[:, :, 0, :].mean(dim=-1) + out[:, :, -1, :].mean(dim=-1) +
            out[:, :, :, 0].mean(dim=-1) + out[:, :, :, -1].mean(dim=-1)
        ) / 4.0
        out = out - boundary_mean.view(-1, 1, 1, 1)
        out[:, :, 0, :] = 0.0; out[:, :, -1, :] = 0.0
        out[:, :, :, 0] = 0.0; out[:, :, :, -1] = 0.0
        return self.split_output(out, self.output_key_dict, dim=1)


def evaluate_checkpoint(ckpt_dir, config_path, val_rho, val_phi, device, num_samples=20):
    """Load checkpoint, predict on validation samples, return error metrics."""
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        return None, f"Config load failed: {e}"

    try:
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
            trunk_net=model_trunk, branch_net=model_branch, num_basis=num_basis,
        ).to(device)
        load_checkpoint(ckpt_dir, models=[model_branch, model_trunk], device=device)
        model.eval()
    except Exception as e:
        return None, f"Model load failed: {e}"

    # Build coordinate grids matching validation data size
    grid_size = val_rho.shape[-1]
    lin = np.linspace(0, 1, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    x_t = torch.from_numpy(xx).view(1, 1, grid_size, grid_size).to(device)
    y_t = torch.from_numpy(yy).view(1, 1, grid_size, grid_size).to(device)

    # Evaluate on samples
    n = min(num_samples, val_rho.shape[0])
    maes, rel_errs, max_errs = [], [], []

    with torch.no_grad():
        for i in range(n):
            rho_np = val_rho[i]  # (1, H, W)
            phi_gt = val_phi[i, 0]  # (H, W)

            rho_t = torch.from_numpy(rho_np.astype(np.float32)).unsqueeze(0).to(device) / RHO_NORM
            out = model({"rho_prime": rho_t, "x": x_t, "y": y_t})
            phi_pred = out["phi"][0, 0].cpu().numpy() * PHI_NORM

            mae = np.abs(phi_pred - phi_gt).mean()
            denom = np.abs(phi_gt).mean()
            rel_err = mae / max(denom, 1e-12)
            max_err = np.abs(phi_pred - phi_gt).max()

            maes.append(mae)
            rel_errs.append(rel_err)
            max_errs.append(max_err)

    metrics = {
        "mae": np.mean(maes),
        "rel_err": np.mean(rel_errs),
        "max_err": np.mean(max_errs),
        "num_basis": num_basis,
        "epochs": "N/A",
    }
    # Try to get epoch count from checkpoint filenames
    try:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("FNO")]
        if ckpt_files:
            epochs = max(int(f.split(".")[-2]) for f in ckpt_files)
            metrics["epochs"] = epochs
    except:
        pass

    return metrics, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load validation data
    val_path = "/home/student/Dhruvil_Om_BMP_2026/PoissonSolver/validation.hdf5"
    print(f"Loading validation data from {val_path}")
    with h5py.File(val_path, "r") as f:
        val_rho = np.array(f["rho"][:20])   # (20, 1, 257, 257)
        val_phi = np.array(f["potential"][:20])

    print(f"val_rho shape={val_rho.shape}, val_phi shape={val_phi.shape}")

    # All checkpoint directories
    base = "/home/student/Dhruvil_Om_BMP_2026/PoissonSolver"
    ckpt_dirs = []
    for root, dirs, files in os.walk(base):
        if "checkpoints" in dirs:
            ckpt_path = os.path.join(root, "checkpoints")
            config_path = os.path.join(root, ".hydra", "config.yaml")
            if os.path.isfile(config_path):
                # Check if there are actual model files
                mdlus = [f for f in os.listdir(ckpt_path) if f.endswith(".mdlus")]
                if mdlus:
                    ckpt_dirs.append((root, ckpt_path, config_path))

    print(f"\nFound {len(ckpt_dirs)} checkpoint directories\n")
    print(f"{'Directory':<65} {'Basis':>5} {'Epoch':>5} {'MAE':>12} {'RelErr%':>10} {'MaxErr':>12} {'Status'}")
    print("=" * 140)

    results = []
    for run_dir, ckpt_path, config_path in sorted(ckpt_dirs):
        short_name = run_dir.replace(base + "/", "")
        metrics, error = evaluate_checkpoint(ckpt_path, config_path, val_rho, val_phi, device)
        if error:
            print(f"{short_name:<65} {'—':>5} {'—':>5} {'—':>12} {'—':>10} {'—':>12} {error}")
        else:
            print(f"{short_name:<65} {metrics['num_basis']:>5} {str(metrics['epochs']):>5} "
                  f"{metrics['mae']:>12.6e} {metrics['rel_err']*100:>9.4f}% {metrics['max_err']:>12.6e}")
            results.append((short_name, metrics))

    # Print ranking
    if results:
        results.sort(key=lambda x: x[1]["rel_err"])
        print("\n" + "=" * 80)
        print("RANKING (by relative error, lower is better):")
        print("=" * 80)
        for rank, (name, m) in enumerate(results, 1):
            print(f"  #{rank}  {m['rel_err']*100:.4f}%  MAE={m['mae']:.4e}  basis={m['num_basis']}  epochs={m['epochs']}  {name}")


if __name__ == "__main__":
    main()
