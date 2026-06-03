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


DATASETS = ["mnist", "fashion_mnist", "cifar10", "celeba_hq", "afhq"]


def parse_timesteps(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_datasets(value: str) -> list[str]:
    if value == "all":
        return DATASETS
    datasets = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {DATASETS}")
    return datasets


def get_q_positions(dataset_name: str, height: int, width: int) -> list[tuple[int, int]]:
    if dataset_name == "celeba_hq":
        return [
            (int(0.41 * height), int(0.36 * width)),  # left eye
            (int(0.41 * height), int(0.64 * width)),  # right eye
            (int(0.53 * height), int(0.50 * width)),  # nose
            (int(0.28 * height), int(0.50 * width)),  # forehead
            (int(0.67 * height), int(0.50 * width)),  # mouth / chin
        ]
    if dataset_name == "afhq":
        return [
            (int(0.34 * height), int(0.35 * width)),  # left eye
            (int(0.34 * height), int(0.65 * width)),  # right eye
            (int(0.53 * height), int(0.50 * width)),  # nose / muzzle
            (int(0.23 * height), int(0.50 * width)),  # forehead
            (int(0.50 * height), int(0.25 * width)),  # cheek / whisker area
        ]

    offset = max(5, height // 5)
    return [
        (height // 2, width // 2),
        (height // 2 - offset, width // 2),
        (height // 2 + offset, width // 2),
        (height // 2, width // 2 - offset),
        (height // 2, width // 2 + offset),
    ]


def wiener_path_for(dataset_name: str, resolution: int) -> Path:
    return Path("data/models/wiener") / f"{dataset_name}_{resolution}"


def plot_sensitivity_heatmap_grid(fields_by_q_t, q_positions, timesteps, out_path, *, title_prefix=""):
    n_rows = len(q_positions)
    n_cols = len(timesteps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.9 * n_cols, 2.85 * n_rows),
    )

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, (qy, qx) in enumerate(q_positions):
        for c, timestep in enumerate(timesteps):
            ax = axes[r, c]
            field_q = fields_by_q_t[(r, c)]

            ax.imshow(field_q, cmap="turbo", vmin=0, vmax=1, interpolation="bilinear")
            ax.scatter([qx], [qy], c="red", s=24)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"{title_prefix}step={timestep}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_wiener_unet_comparison_grid(wiener_fields, unet_fields, q_positions, timesteps, out_path):
    n_rows = 2 * len(q_positions)
    n_cols = len(timesteps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.9 * n_cols, 2.55 * n_rows),
    )

    if n_cols == 1:
        axes = axes[:, None]

    for r, (qy, qx) in enumerate(q_positions):
        for c, timestep in enumerate(timesteps):
            wiener_ax = axes[2 * r, c]
            unet_ax = axes[2 * r + 1, c]

            wiener_ax.imshow(
                wiener_fields[(r, c)],
                cmap="turbo",
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )
            unet_ax.imshow(
                unet_fields[(r, c)],
                cmap="turbo",
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )

            for ax in (wiener_ax, unet_ax):
                ax.scatter([qx], [qy], c="red", s=24)
                ax.set_xticks([])
                ax.set_yticks([])

            if r == 0:
                wiener_ax.set_title(f"step={timestep}", fontsize=10)
            if c == 0:
                wiener_ax.set_ylabel(f"Wiener q=({qy},{qx})", fontsize=9)
                unet_ax.set_ylabel(f"UNet q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


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


def make_noisy_samples(model, x0_clean, timestep: int, generator):
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep].to(x0_clean.device)
    beta_prod_t = 1.0 - alpha_prod_t
    noise = torch.randn(x0_clean.shape, device=x0_clean.device, generator=generator)
    return torch.sqrt(alpha_prod_t) * x0_clean + torch.sqrt(beta_prod_t) * noise


def sensitivity_for_q(model, xt, timestep: int, qy: int, qx: int) -> torch.Tensor:
    xt = xt.detach().clone().requires_grad_(True)
    t = torch.tensor(timestep, device=xt.device, dtype=torch.long)

    pred_x0 = model.denoise(xt, t)

    grads = []
    for channel in range(pred_x0.shape[1]):
        scalar = pred_x0[0, channel, qy, qx]
        grad = torch.autograd.grad(
            scalar,
            xt,
            retain_graph=channel < pred_x0.shape[1] - 1,
        )[0]
        grads.append(grad[0].abs())

    sensitivity = torch.stack(grads, dim=0).sum(dim=0)
    sensitivity = sensitivity.sum(dim=0)
    return sensitivity.detach().cpu()


def wiener_sensitivity_for_q(
    U,
    LA,
    Vh,
    model,
    timestep: int,
    qy: int,
    qx: int,
    channels: int,
    height: int,
    width: int,
):
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep].to(U.device)
    beta_prod_t = 1.0 - alpha_prod_t
    shrink = alpha_prod_t * LA / (beta_prod_t + alpha_prod_t * LA)

    q_rows = torch.tensor(
        [ch * height * width + qy * width + qx for ch in range(channels)],
        device=U.device,
    )

    rows = (U.index_select(0, q_rows) * shrink.unsqueeze(0)) @ Vh
    denom = rows[torch.arange(channels, device=U.device), q_rows].unsqueeze(1)
    denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)

    normalized = rows / denom
    sensitivity = normalized.abs().reshape(channels, channels, height, width).sum(dim=(0, 1))
    sensitivity = sensitivity.detach().cpu()
    return sensitivity / (sensitivity.max() + 1e-8)


def run_dataset(dataset_name: str, args) -> None:
    print(f"\n=== {dataset_name} ===")
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
    if cfg.metrics.baseline_path is None:
        raise ValueError(f"{dataset_name} config must define metrics.baseline_path.")

    model = BaselineUNet(
        resolution=dataset.resolution,
        device=str(device),
        num_steps=1000,
        model_path=cfg.metrics.baseline_path,
        dataset_name=cfg.dataset.name,
        in_channels=dataset.in_channels,
        out_channels=dataset.in_channels,
    )
    model.eval()

    height = width = dataset.resolution
    channels = dataset.in_channels
    q_positions = get_q_positions(dataset_name, height, width)
    timesteps = parse_timesteps(args.timesteps)

    out_dir = args.out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    x0_clean = collect_images(dataset, args.num_xt, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    unet_fields_by_q_t = {}
    for c, timestep in enumerate(timesteps):
        xt_all = make_noisy_samples(model, x0_clean, timestep, generator)

        for r, (qy, qx) in enumerate(q_positions):
            accumulator = torch.zeros(height, width)

            for idx in tqdm(
                range(args.num_xt),
                desc=f"{dataset_name} grad step={timestep} q=({qy},{qx})",
                leave=False,
            ):
                field = sensitivity_for_q(
                    model,
                    xt_all[idx : idx + 1],
                    timestep,
                    qy,
                    qx,
                )
                accumulator += field

            field_mean = accumulator / args.num_xt
            field_mean = field_mean / (field_mean.max() + 1e-8)
            unet_fields_by_q_t[(r, c)] = field_mean

    U, LA, Vh, mean = load_wiener_filter(wiener_path_for(dataset_name, dataset.resolution), device=device)
    del mean

    wiener_fields_by_q_t = {}
    for c, timestep in enumerate(timesteps):
        for r, (qy, qx) in enumerate(q_positions):
            wiener_fields_by_q_t[(r, c)] = wiener_sensitivity_for_q(
                U,
                LA,
                Vh,
                model,
                timestep,
                qy,
                qx,
                channels,
                height,
                width,
            )

    plot_sensitivity_heatmap_grid(
        fields_by_q_t=unet_fields_by_q_t,
        q_positions=q_positions,
        timesteps=timesteps,
        out_path=out_dir / "unet_x0_denoiser_sensitivity_heatmap.png",
    )
    plot_wiener_unet_comparison_grid(
        wiener_fields=wiener_fields_by_q_t,
        unet_fields=unet_fields_by_q_t,
        q_positions=q_positions,
        timesteps=timesteps,
        out_path=out_dir / "wiener_vs_unet_sensitivity_heatmap.png",
    )

    print(f"Saved outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--num-xt", type=int, default=128)
    parser.add_argument("--timesteps", default="950,750,500,250,50")
    parser.add_argument("--dataset-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments_tom/figures/unet_sensitivity_all_datasets"),
    )
    args = parser.parse_args()

    for dataset_name in parse_datasets(args.datasets):
        run_dataset(dataset_name, args)


if __name__ == "__main__":
    main()
