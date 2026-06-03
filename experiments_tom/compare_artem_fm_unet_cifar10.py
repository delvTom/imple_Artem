from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.metrics import calculate_mse, calculate_r2_score
from local_diffusion.models.pca_locality import WeightedStreamingSoftmax
from local_diffusion.utils.wiener import (
    compute_wiener_filter,
    load_wiener_filter,
    save_wiener_filter,
)


def parse_t_values(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def load_torchcfm_unet(checkpoint_path: Path, device: torch.device, state_key: str):
    try:
        from torchcfm.models.unet.unet import UNetModelWrapper
    except ImportError as exc:
        raise ImportError(
            "torchcfm is required to load Alexander Tong's CIFAR-10 FM UNet. "
            "Install the requirements from that repository, then rerun this script."
        ) from exc

    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if state_key in checkpoint:
        state = checkpoint[state_key]
    else:
        state = checkpoint

    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_or_compute_wiener(dataset, device: torch.device, wiener_path: Path):
    if (wiener_path / "U.pt").exists():
        print("Loading existing Wiener SVD...")
        return load_wiener_filter(wiener_path, device=device)

    print("Computing covariance and SVD...")
    S, mean = compute_wiener_filter(
        dataloader=dataset.dataloader,
        device=device,
        resolution=dataset.resolution,
        n_channels=dataset.in_channels,
    )
    U, LA, Vh = torch.linalg.svd(S)
    save_wiener_filter(U, LA, Vh, mean, wiener_path)
    return U, LA, Vh, mean


def build_fm_mask(U, LA, Vh, t: float, threshold: float, eps: float = 1e-6):
    sigma2 = ((1.0 - t) / t) ** 2
    shrink = LA / (LA + sigma2)
    LLt = U @ torch.diag(shrink) @ Vh

    denom = torch.diagonal(LLt).unsqueeze(1)
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
    normalized = LLt / denom

    if threshold > 0:
        mask = (normalized.abs() >= threshold).float()
    else:
        mask = normalized

    return mask


@torch.no_grad()
def artem_fm_predict_x1(
    xt: torch.Tensor,
    dataset,
    mask: torch.Tensor,
    t: float,
    *,
    temperature: float,
) -> torch.Tensor:
    xt_flat = xt.flatten(start_dim=1)
    t_tensor = torch.tensor(t, device=xt.device, dtype=xt.dtype)
    noise_var = (1.0 - t_tensor) ** 2

    first_moment = WeightedStreamingSoftmax(device=xt.device, dtype=xt.dtype)

    for batch in tqdm(dataset.dataloader, desc="Artem xi", leave=False):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        x1_candidates = images.to(xt.device, dtype=xt.dtype).flatten(start_dim=1)

        delta = (xt_flat.unsqueeze(1) - t_tensor * x1_candidates.unsqueeze(0)) ** 2
        masked_dist = torch.einsum("bkn,nm->bkm", delta, mask)
        logits = -masked_dist / (2.0 * noise_var * temperature)
        first_moment.add(x1_candidates, logits)

    pred = first_moment.get_average()
    if pred is None:
        raise RuntimeError("No xi samples were used for Artem prediction.")

    return pred.view_as(xt)


def collect_x1_batch(dataset, num_samples: int, device: torch.device) -> torch.Tensor:
    chunks = []
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(images)
        if sum(chunk.shape[0] for chunk in chunks) >= num_samples:
            break
    return torch.cat(chunks, dim=0)[:num_samples].to(device)


def plot_metrics(results: list[dict], out_path: Path) -> None:
    t_values = [row["t"] for row in results]
    mse_velocity = [row["mse_velocity"] for row in results]
    mse_x1 = [row["mse_x1"] for row in results]
    r2_velocity = [row["r2_velocity"] for row in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    axes[0].plot(t_values, mse_velocity, marker="o")
    axes[0].set_title("Velocity MSE")
    axes[0].set_xlabel("t")
    axes[0].set_yscale("log")

    axes[1].plot(t_values, mse_x1, marker="o")
    axes[1].set_title("x1-hat MSE")
    axes[1].set_xlabel("t")
    axes[1].set_yscale("log")

    axes[2].plot(t_values, r2_velocity, marker="o")
    axes[2].set_title("Velocity R2")
    axes[2].set_xlabel("t")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/fm_unet/cifar10/fm_cifar10_weights_step_400000.pt"),
    )
    parser.add_argument("--state-key", default="ema_model")
    parser.add_argument("--num-xt", type=int, default=256)
    parser.add_argument("--num-xi", type=int, default=5000)
    parser.add_argument("--batch-xt", type=int, default=16)
    parser.add_argument("--xi-batch-size", type=int, default=256)
    parser.add_argument(
        "--t-values",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments_tom/results/artem_vs_fm_unet_cifar10"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    t_values = parse_t_values(args.t_values)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(
        "configs/wiener/cifar10.yaml",
        overrides=[
            "metrics.wandb.enabled=false",
            f"dataset.batch_size={args.xi_batch_size}",
            f"dataset.subset_size={args.num_xi}",
        ],
    )
    xi_dataset = build_dataset(cfg.dataset)

    xt_cfg = load_config(
        "configs/wiener/cifar10.yaml",
        overrides=[
            "metrics.wandb.enabled=false",
            f"dataset.batch_size={args.batch_xt}",
            f"dataset.subset_size={max(args.num_xt, args.batch_xt)}",
        ],
    )
    xt_dataset = build_dataset(xt_cfg.dataset)

    U, LA, Vh, mean = load_or_compute_wiener(
        xi_dataset,
        device=device,
        wiener_path=Path("data/models/wiener/cifar10_32"),
    )
    del mean

    fm_unet = load_torchcfm_unet(args.checkpoint, device, args.state_key)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x1_all = collect_x1_batch(xt_dataset, args.num_xt, device)
    x0_all = torch.randn(x1_all.shape, device=device, generator=generator)

    results = []
    for t in t_values:
        print(f"\n=== t={t:.2f} ===")
        mask = build_fm_mask(U, LA, Vh, t, threshold=args.threshold)

        velocity_mse_values = []
        x1_mse_values = []
        velocity_r2_values = []
        x1_r2_values = []

        for start in range(0, args.num_xt, args.batch_xt):
            end = min(start + args.batch_xt, args.num_xt)
            x1 = x1_all[start:end]
            x0 = x0_all[start:end]
            xt = (1.0 - t) * x0 + t * x1

            pred_x1_artem = artem_fm_predict_x1(
                xt,
                xi_dataset,
                mask,
                t,
                temperature=args.temperature,
            )
            v_artem = (pred_x1_artem - xt) / (1.0 - t)

            t_batch = torch.full((xt.shape[0],), t, device=device, dtype=xt.dtype)
            v_unet = fm_unet(t_batch, xt)
            pred_x1_unet = xt + (1.0 - t) * v_unet

            velocity_mse_values.append(calculate_mse(v_artem, v_unet))
            x1_mse_values.append(calculate_mse(pred_x1_artem, pred_x1_unet))
            velocity_r2_values.append(calculate_r2_score(v_artem, v_unet))
            x1_r2_values.append(calculate_r2_score(pred_x1_artem, pred_x1_unet))

        row = {
            "t": t,
            "mse_velocity": sum(velocity_mse_values) / len(velocity_mse_values),
            "mse_x1": sum(x1_mse_values) / len(x1_mse_values),
            "r2_velocity": sum(velocity_r2_values) / len(velocity_r2_values),
            "r2_x1": sum(x1_r2_values) / len(x1_r2_values),
            "num_xt": args.num_xt,
            "num_xi": args.num_xi,
            "threshold": args.threshold,
            "temperature": args.temperature,
        }
        print(row)
        results.append(row)

    result_path = args.out_dir / "metrics.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_metrics(results, args.out_dir / "metrics.png")
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
