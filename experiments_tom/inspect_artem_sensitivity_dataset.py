from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.models.baseline_unet import BaselineUNet
from local_diffusion.utils.wiener import load_wiener_filter


def parse_timesteps(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def get_q_positions(dataset_name: str, height: int, width: int) -> list[tuple[int, int]]:
    if dataset_name == "celeba_hq":
        return [
            (int(0.41 * height), int(0.36 * width)),
            (int(0.41 * height), int(0.64 * width)),
            (int(0.53 * height), int(0.50 * width)),
            (int(0.28 * height), int(0.50 * width)),
            (int(0.67 * height), int(0.50 * width)),
        ]
    if dataset_name == "afhq":
        return [
            (int(0.34 * height), int(0.35 * width)),
            (int(0.34 * height), int(0.65 * width)),
            (int(0.53 * height), int(0.50 * width)),
            (int(0.23 * height), int(0.50 * width)),
            (int(0.50 * height), int(0.25 * width)),
        ]

    offset = max(5, height // 5)
    return [
        (height // 2, width // 2),
        (height // 2 - offset, width // 2),
        (height // 2 + offset, width // 2),
        (height // 2, width // 2 - offset),
        (height // 2, width // 2 + offset),
    ]


def collect_images(dataset, num_samples: int, device: torch.device) -> torch.Tensor:
    chunks = []
    total = 0
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(images)
        total += images.shape[0]
        if total >= num_samples:
            break
    return torch.cat(chunks, dim=0)[:num_samples].to(device)


def make_noisy_samples(scheduler, x0_clean, timestep: int, generator):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(x0_clean.device)
    beta_prod_t = 1.0 - alpha_prod_t
    noise = torch.randn(x0_clean.shape, device=x0_clean.device, generator=generator)
    return torch.sqrt(alpha_prod_t) * x0_clean + torch.sqrt(beta_prod_t) * noise


def wiener_column_sensitivity(U, LA, Vh, scheduler, timestep, qy, qx, channels, height, width):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(U.device)
    beta_prod_t = 1.0 - alpha_prod_t
    shrink = alpha_prod_t * LA / (beta_prod_t + alpha_prod_t * LA)

    fields = []
    for out_channel in range(channels):
        q_idx = out_channel * height * width + qy * width + qx
        column = U @ (shrink * Vh[:, q_idx])
        denom = column[q_idx]
        if denom.abs() < 1e-6:
            denom = torch.ones_like(denom)
        field = (column / denom).abs().reshape(channels, height, width)
        fields.append(field)

    sensitivity = torch.stack(fields, dim=0).sum(dim=(0, 1)).detach().cpu()
    return sensitivity / (sensitivity.max() + 1e-8)


def artem_column_mask(U, LA, Vh, scheduler, timestep, q_idx, threshold):
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(U.device)
    beta_prod_t = 1.0 - alpha_prod_t
    shrink = alpha_prod_t * LA / (beta_prod_t + alpha_prod_t * LA)

    column = U @ (shrink * Vh[:, q_idx])
    denom = column[q_idx]
    if denom.abs() < 1e-6:
        denom = torch.ones_like(denom)
    normalized = column / denom
    if threshold > 0:
        return (normalized.abs() >= threshold).to(normalized.dtype)
    return normalized


def artem_scalar_grad(
    xt,
    xi_flat,
    mask_col,
    scheduler,
    timestep,
    q_idx,
    *,
    temperature,
):
    xt_flat = xt.detach().clone().flatten(start_dim=1).requires_grad_(True)

    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(xt.device, dtype=xt.dtype)
    beta_prod_t = 1.0 - alpha_prod_t
    sqrt_alpha = torch.sqrt(alpha_prod_t)

    delta = (xt_flat - sqrt_alpha * xi_flat) ** 2
    masked_dist = delta @ mask_col.to(xt.device, dtype=xt.dtype)
    logits = -masked_dist / (2.0 * beta_prod_t * temperature)
    weights = torch.softmax(logits, dim=0)
    pred_scalar = torch.sum(weights * xi_flat[:, q_idx])

    grad = torch.autograd.grad(pred_scalar, xt_flat)[0]
    return grad[0].abs().detach().cpu()


def artem_sensitivity_for_q(
    xt,
    xi_flat,
    U,
    LA,
    Vh,
    scheduler,
    timestep,
    qy,
    qx,
    channels,
    height,
    width,
    *,
    threshold,
    temperature,
):
    grads = []
    for out_channel in range(channels):
        q_idx = out_channel * height * width + qy * width + qx
        mask_col = artem_column_mask(U, LA, Vh, scheduler, timestep, q_idx, threshold)
        grad = artem_scalar_grad(
            xt,
            xi_flat,
            mask_col,
            scheduler,
            timestep,
            q_idx,
            temperature=temperature,
        )
        grads.append(grad.reshape(channels, height, width))

    sensitivity = torch.stack(grads, dim=0).sum(dim=(0, 1))
    return sensitivity


def plot_heatmap_grid(fields_by_q_t, q_positions, timesteps, out_path, label):
    n_rows = len(q_positions)
    n_cols = len(timesteps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols, 2.85 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, (qy, qx) in enumerate(q_positions):
        for c, timestep in enumerate(timesteps):
            ax = axes[r, c]
            ax.imshow(fields_by_q_t[(r, c)], cmap="turbo", vmin=0, vmax=1, interpolation="bilinear")
            ax.scatter([qx], [qy], c="red", s=24)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step={timestep}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{label} q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_wiener_artem_comparison(wiener_fields, artem_fields, q_positions, timesteps, out_path):
    n_rows = 2 * len(q_positions)
    n_cols = len(timesteps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols, 2.55 * n_rows))
    if n_cols == 1:
        axes = axes[:, None]

    for r, (qy, qx) in enumerate(q_positions):
        for c, timestep in enumerate(timesteps):
            wiener_ax = axes[2 * r, c]
            artem_ax = axes[2 * r + 1, c]

            wiener_ax.imshow(wiener_fields[(r, c)], cmap="turbo", vmin=0, vmax=1, interpolation="bilinear")
            artem_ax.imshow(artem_fields[(r, c)], cmap="turbo", vmin=0, vmax=1, interpolation="bilinear")

            for ax in (wiener_ax, artem_ax):
                ax.scatter([qx], [qy], c="red", s=24)
                ax.set_xticks([])
                ax.set_yticks([])

            if r == 0:
                wiener_ax.set_title(f"step={timestep}", fontsize=10)
            if c == 0:
                wiener_ax.set_ylabel(f"Wiener q=({qy},{qx})", fontsize=9)
                artem_ax.set_ylabel(f"Artem q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def parse_datasets(value: str) -> list[str]:
    available = ["mnist", "fashion_mnist", "cifar10", "celeba_hq", "afhq"]
    if value == "all":
        return available
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {available}")
    return datasets


def run_dataset(dataset_name: str, args, default_thresholds: dict[str, float]):
    print(f"\n=== {dataset_name} ===")
    threshold = args.threshold
    if threshold is None:
        threshold = default_thresholds.get(dataset_name, 0.02)

    cfg = load_config(
        f"configs/wiener/{dataset_name}.yaml",
        overrides=[
            "metrics.wandb.enabled=false",
            f"dataset.batch_size={args.dataset_batch_size}",
        ],
    )

    device = torch.device(cfg.experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print("device:", device)

    dataset = build_dataset(cfg.dataset)
    scheduler_ref = BaselineUNet(
        resolution=dataset.resolution,
        device=str(device),
        num_steps=1000,
        model_path=cfg.metrics.baseline_path,
        dataset_name=cfg.dataset.name,
        in_channels=dataset.in_channels,
        out_channels=dataset.in_channels,
    )

    height = width = dataset.resolution
    channels = dataset.in_channels
    timesteps = [int(x) for x in args.timesteps.split(",") if x.strip()]
    q_positions = get_q_positions(dataset_name, height, width)

    out_dir = args.out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    U, LA, Vh, mean = load_wiener_filter(
        Path("data/models/wiener") / f"{dataset_name}_{dataset.resolution}",
        device=device,
    )
    del mean

    x0_clean = collect_images(dataset, args.num_xt, device)
    xi = collect_images(dataset, args.num_xi, device)
    xi_flat = xi.flatten(start_dim=1).detach()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    artem_fields = {}
    wiener_fields = {}
    for c, timestep in enumerate(timesteps):
        xt_all = make_noisy_samples(scheduler_ref.scheduler, x0_clean, timestep, generator)

        for r, (qy, qx) in enumerate(q_positions):
            accumulator = torch.zeros(height, width)
            for idx in tqdm(
                range(args.num_xt),
                desc=f"{dataset_name} Artem grad step={timestep} q=({qy},{qx})",
                leave=False,
            ):
                field = artem_sensitivity_for_q(
                    xt_all[idx : idx + 1],
                    xi_flat,
                    U,
                    LA,
                    Vh,
                    scheduler_ref.scheduler,
                    timestep,
                    qy,
                    qx,
                    channels,
                    height,
                    width,
                    threshold=threshold,
                    temperature=args.temperature,
                )
                accumulator += field

            field_mean = accumulator / args.num_xt
            artem_fields[(r, c)] = field_mean / (field_mean.max() + 1e-8)
            wiener_fields[(r, c)] = wiener_column_sensitivity(
                U,
                LA,
                Vh,
                scheduler_ref.scheduler,
                timestep,
                qy,
                qx,
                channels,
                height,
                width,
            )

    plot_heatmap_grid(
        artem_fields,
        q_positions,
        timesteps,
        out_dir / "artem_estimator_sensitivity_heatmap.png",
        label="Artem",
    )
    plot_wiener_artem_comparison(
        wiener_fields,
        artem_fields,
        q_positions,
        timesteps,
        out_dir / "wiener_vs_artem_sensitivity_heatmap.png",
    )

    print(f"Saved outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="afhq")
    parser.add_argument("--num-xt", type=int, default=16)
    parser.add_argument("--num-xi", type=int, default=512)
    parser.add_argument("--timesteps", default="950,750,500,250,50")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-batch-size", type=int, default=256)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments_tom/figures/artem_sensitivity"),
    )
    args = parser.parse_args()

    default_thresholds = {
        "mnist": 0.005,
        "fashion_mnist": 0.005,
        "cifar10": 0.02,
        "celeba_hq": 0.02,
        "afhq": 0.02,
    }

    for dataset_name in parse_datasets(args.dataset):
        run_dataset(dataset_name, args, default_thresholds)


if __name__ == "__main__":
    main()
