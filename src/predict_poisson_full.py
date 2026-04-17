# predict_poisson_full.py
# ─────────────────────────────────────────────────────────────────────────────
# Full validation prediction: run the trained DeepONet model on every sample
# in a validation HDF5 dataset and produce per-sample images, per-cell error
# heatmaps, per-input CSV metrics, and a summary report.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from physicsnemo.utils.checkpoint import load_checkpoint


GRID_SIZE = 256


class MdlsSymWrapper(Arch):
    """Wrapper model matching the training architecture."""

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
        fc_out = self.trunk_net(xy)
        fno_out = self.branch_net(dict_tensor["rho_prime"])

        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        if self.num_basis > 1:
            out = (fc_out * fno_out).sum(dim=1, keepdim=True)
        else:
            out = fc_out * fno_out

        # Strictly Enforce Physical Dirichlet Boundary Condition
        boundary_mean = (
            out[:, :, 0, :].mean(dim=-1) + out[:, :, -1, :].mean(dim=-1) +
            out[:, :, :, 0].mean(dim=-1) + out[:, :, :, -1].mean(dim=-1)
        ) / 4.0
        out = out - boundary_mean.view(-1, 1, 1, 1)

        out[:, :, 0, :] = 0.0
        out[:, :, -1, :] = 0.0
        out[:, :, :, 0] = 0.0
        out[:, :, :, -1] = 0.0

        return self.split_output(out, self.output_key_dict, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prediction on full validation.hdf5 and save analysis artifacts."
    )
    parser.add_argument(
        "--validation_hdf5",
        type=str,
        default="./validation.hdf5",
        help="Path to validation HDF5 dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./conf/config_deeponet.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./outputs_poisson_latest3/checkpoints",
        help="Checkpoint directory to load model weights from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions_simulations1",
        help="Directory to store prediction images and error summaries.",
    )
    parser.add_argument(
        "--rho_norm",
        type=float,
        default=1.23223e-08,
        help="Rho normalization constant used in training.",
    )
    parser.add_argument(
        "--phi_norm",
        type=float,
        default=9.21588e-01,
        help="Phi normalization constant used in training.",
    )
    return parser.parse_args()


def build_model(cfg, device: torch.device):
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

    num_basis = int(cfg.model.get("num_basis", cfg.model.fno.out_channels))
    model = MdlsSymWrapper(
        input_keys=[Key("rho_prime"), Key("x"), Key("y")],
        output_keys=[Key("phi")],
        trunk_net=model_trunk,
        branch_net=model_branch,
        num_basis=num_basis,
    ).to(device)

    return model, model_branch, model_trunk


def build_coordinate_grids(device: torch.device):
    lin = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    x_t = torch.from_numpy(xx).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    y_t = torch.from_numpy(yy).view(1, 1, GRID_SIZE, GRID_SIZE).to(device)
    return x_t, y_t


def to_2d(sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3 and sample.shape[0] == 1:
        return sample[0]
    raise ValueError(f"Unexpected sample shape: {sample.shape}")


def save_sample_figure(
    rho_grid: np.ndarray,
    phi_pred: np.ndarray,
    phi_true: np.ndarray,
    phi_expected: np.ndarray,
    error_abs: np.ndarray,
    sample_idx: int,
    save_path: str,
):
    fig, axes = plt.subplots(1, 5, figsize=(32, 6))

    im0 = axes[0].imshow(rho_grid, cmap="RdBu_r")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Input rho")

    vmin = min(phi_pred.min(), phi_true.min(), phi_expected.min())
    vmax = max(phi_pred.max(), phi_true.max(), phi_expected.max())

    im1 = axes[1].imshow(phi_pred, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Predicted phi")

    im2 = axes[2].imshow(phi_true, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=axes[2])
    axes[2].set_title("True phi")

    im3 = axes[3].imshow(phi_expected, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im3, ax=axes[3])
    axes[3].set_title("Expected phi")

    im4 = axes[4].imshow(error_abs, cmap="hot")
    plt.colorbar(im4, ax=axes[4])
    axes[4].set_title("Absolute error")

    fig.suptitle(f"Validation sample {sample_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_per_input_csv(rows: List[dict], csv_path: str):
    fieldnames = [
        "sample_index",
        "mean_error",
        "mean_abs_error",
        "rmse",
        "max_abs_error",
        "ape",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    validation_hdf5 = os.path.abspath(args.validation_hdf5)
    config_path = os.path.abspath(args.config)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    output_dir = os.path.abspath(args.output_dir)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cfg = OmegaConf.load(config_path)
    model, model_branch, model_trunk = build_model(cfg, device)

    print(f"Loading checkpoint from: {ckpt_dir}")
    load_checkpoint(
        ckpt_dir,
        models=[model_branch, model_trunk],
        device=device,
    )
    model.eval()

    x_coord, y_coord = build_coordinate_grids(device)

    with h5py.File(validation_hdf5, "r") as h5f:
        if "rho" not in h5f or "potential" not in h5f:
            raise KeyError("validation.hdf5 must contain 'rho' and 'potential' datasets.")

        n_samples = len(h5f["rho"])
        if n_samples == 0:
            raise ValueError("validation.hdf5 contains no samples.")

        sum_error = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        sum_abs_error = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        per_input_rows = []

        for idx in range(n_samples):
            rho_grid = to_2d(np.array(h5f["rho"][idx]))[:GRID_SIZE, :GRID_SIZE].astype(np.float64)
            phi_true = to_2d(np.array(h5f["potential"][idx]))[:GRID_SIZE, :GRID_SIZE].astype(np.float64)

            rho_tensor = (
                torch.from_numpy((rho_grid / args.rho_norm).astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                out = model(
                    {"rho_prime": rho_tensor, "x": x_coord, "y": y_coord}
                )

            phi_pred = out["phi"][0, 0].detach().cpu().numpy().astype(np.float64) * args.phi_norm

            phi_expected = phi_true.copy()
            error = phi_pred - phi_expected
            error_abs = np.abs(error)

            sum_error += error
            sum_abs_error += error_abs

            mean_error = float(np.mean(error))
            mean_abs_error = float(np.mean(error_abs))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            max_abs_error = float(np.max(error_abs))

            true_denom = np.where(phi_expected == 0, 1.0, phi_expected)
            ape = float(np.mean(error_abs / np.abs(true_denom)) * 100)

            per_input_rows.append(
                {
                    "sample_index": idx,
                    "mean_error": f"{mean_error:.10e}",
                    "mean_abs_error": f"{mean_abs_error:.10e}",
                    "rmse": f"{rmse:.10e}",
                    "max_abs_error": f"{max_abs_error:.10e}",
                    "ape": f"{ape:.10e}",
                }
            )

            image_path = os.path.join(images_dir, f"sample_{idx:04d}.png")
            save_sample_figure(
                rho_grid=rho_grid,
                phi_pred=phi_pred,
                phi_true=phi_true,
                phi_expected=phi_expected,
                error_abs=error_abs,
                sample_idx=idx,
                save_path=image_path,
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == n_samples:
                print(f"Processed {idx + 1}/{n_samples} samples")

    per_cell_mean_error = sum_error / n_samples
    per_cell_mean_abs_error = sum_abs_error / n_samples

    np.save(os.path.join(output_dir, "per_cell_mean_error.npy"), per_cell_mean_error)
    np.save(os.path.join(output_dir, "per_cell_mean_abs_error.npy"), per_cell_mean_abs_error)
    np.savetxt(
        os.path.join(output_dir, "per_cell_mean_error.csv"),
        per_cell_mean_error,
        delimiter=",",
        fmt="%.10e",
    )
    np.savetxt(
        os.path.join(output_dir, "per_cell_mean_abs_error.csv"),
        per_cell_mean_abs_error,
        delimiter=",",
        fmt="%.10e",
    )

    write_per_input_csv(
        per_input_rows,
        os.path.join(output_dir, "per_input_error_metrics.csv"),
    )

    sorted_rows = sorted(
        per_input_rows,
        key=lambda r: float(r["mean_abs_error"]),
        reverse=True,
    )
    write_per_input_csv(
        sorted_rows,
        os.path.join(output_dir, "bottleneck_inputs_sorted_by_mae.csv"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(per_cell_mean_error, cmap="coolwarm")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Per-cell mean signed error")

    im1 = axes[1].imshow(per_cell_mean_abs_error, cmap="hot")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Per-cell mean absolute error")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_cell_error_heatmaps.png"), dpi=150)
    plt.close(fig)

    overall_mean_error = float(np.mean(per_cell_mean_error))
    overall_mean_abs_error = float(np.mean(per_cell_mean_abs_error))

    # Calculate overall APE
    df_ape = [float(row["ape"]) for row in per_input_rows]
    overall_ape = float(np.mean(df_ape))

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Full validation prediction summary\n")
        f.write("================================\n")
        f.write(f"Validation dataset: {validation_hdf5}\n")
        f.write(f"Checkpoint directory: {ckpt_dir}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Rho norm: {args.rho_norm:.10e}\n")
        f.write(f"Phi norm: {args.phi_norm:.10e}\n")
        f.write("Expected phi source: validation potential (same as true phi)\n")
        f.write(f"Overall mean signed error: {overall_mean_error:.10e}\n")
        f.write(f"Overall mean absolute error: {overall_mean_abs_error:.10e}\n")
        f.write(f"Overall mean absolute percentage error (APE): {overall_ape:.10e}%\n")

    print("Done. Outputs saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
