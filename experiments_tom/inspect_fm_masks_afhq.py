from pathlib import Path
import os

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.utils.wiener import (
    compute_wiener_filter,
    load_wiener_filter,
    save_wiener_filter,
)


def image_to_display(img):
    img = img[0].detach().cpu()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0)


def plot_xt_trajectory_grid(x0_batch, x1_batch, q_positions, t_values, out_path):
    n_rows = x1_batch.shape[0]
    n_cols = len(t_values) + 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.35 * n_rows),
    )

    if n_rows == 1:
        axes = axes[None, :]

    for r in range(n_rows):
        qy, qx = q_positions[r]
        panels = [("x0 noise", x0_batch[r : r + 1])]
        panels += [
            (f"t={t:.2f}", (1.0 - t) * x0_batch[r : r + 1] + t * x1_batch[r : r + 1])
            for t in t_values
        ]
        panels += [("x1 data", x1_batch[r : r + 1])]

        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]

            ax.imshow(image_to_display(img))
            ax.scatter([qx], [qy], c="red", s=18)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"img {r}, q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_field_on_xt_grid(
    x0_batch,
    x1_batch,
    fields_by_img_t,
    q_positions,
    t_values,
    out_path,
    *,
    cmap="viridis",
    alpha=0.85,
    mask_zeros=False,
):
    n_rows = x1_batch.shape[0]
    n_cols = len(t_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.45 * n_rows),
    )

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r in range(n_rows):
        qy, qx = q_positions[r]

        for c, t in enumerate(t_values):
            ax = axes[r, c]
            xt = (1.0 - t) * x0_batch[r : r + 1] + t * x1_batch[r : r + 1]
            field_q = fields_by_img_t[(r, c)]
            if mask_zeros:
                field_q = field_q.clone()
                field_q[field_q <= 0] = torch.nan

            ax.imshow(image_to_display(xt))
            ax.imshow(field_q, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
            ax.scatter([qx], [qy], c="red", s=18)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"t={t:.2f}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"img {r}, q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_sensitivity_heatmap_grid(
    fields_by_img_t,
    q_positions,
    t_values,
    out_path,
    *,
    cmap="turbo",
):
    n_rows = len(q_positions)
    n_cols = len(t_values)

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
        for c, t in enumerate(t_values):
            ax = axes[r, c]
            field_q = fields_by_img_t[(r, c)]

            ax.imshow(field_q, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
            ax.scatter([qx], [qy], c="red", s=24)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"t={t:.2f}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"img {r}, q=({qy},{qx})", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def build_fm_fields_for_q(U, LA, Vh, t, qy, qx, channels, height, width, threshold=0.02, eps=1e-6):
    sigma2 = ((1.0 - t) / t) ** 2
    shrink = LA / (LA + sigma2)

    q_rows = torch.tensor(
        [ch * height * width + qy * width + qx for ch in range(channels)],
        device=U.device,
    )

    rows = (U.index_select(0, q_rows) * shrink.unsqueeze(0)) @ Vh
    denom = rows[torch.arange(channels, device=U.device), q_rows].unsqueeze(1)
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)

    normalized = rows / denom
    sensitivity = normalized.abs().reshape(channels, channels, height, width).amax(dim=(0, 1))
    sensitivity = sensitivity.detach().cpu()
    mask = (sensitivity >= threshold).float()
    sensitivity = sensitivity / (sensitivity.max() + 1e-8)
    return mask, sensitivity


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = load_config(
        "configs/wiener/afhq.yaml",
        overrides=[
            "metrics.wandb.enabled=false",
            "dataset.batch_size=256",
        ],
    )

    dataset = build_dataset(cfg.dataset)
    wiener_path = Path("data/models/wiener/afhq_64")

    if (wiener_path / "U.pt").exists():
        print("Loading existing Wiener SVD...")
        U, LA, Vh, mean = load_wiener_filter(wiener_path, device=device)
    else:
        print("Computing covariance and SVD...")
        S, mean = compute_wiener_filter(
            dataloader=dataset.dataloader,
            device=device,
            resolution=dataset.resolution,
            n_channels=dataset.in_channels,
        )
        U, LA, Vh = torch.linalg.svd(S)
        save_wiener_filter(U, LA, Vh, mean, wiener_path)

    del mean

    H = W = dataset.resolution
    C = dataset.in_channels
    n_images = 5

    torch.manual_seed(42)
    generator = torch.Generator(device=device).manual_seed(42)

    first_batch = next(iter(dataset.dataloader))
    images = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    x1_batch = images[:n_images].to(device)
    x0_batch = torch.randn(x1_batch.shape, device=device, generator=generator)

    q_positions = [
        (int(0.34 * H), int(0.35 * W)),  # left eye
        (int(0.34 * H), int(0.65 * W)),  # right eye
        (int(0.53 * H), int(0.50 * W)),  # nose / muzzle
        (int(0.23 * H), int(0.50 * W)),  # forehead
        (int(0.50 * H), int(0.25 * W)),  # cheek / whisker area
    ]

    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    out_dir = Path("experiments_tom/figures/fm_masks_afhq")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_xt_trajectory_grid(
        x0_batch=x0_batch,
        x1_batch=x1_batch,
        q_positions=q_positions,
        t_values=t_values,
        out_path=out_dir / "fm_xt_trajectory_grid.png",
    )

    masks_by_img_t = {}
    sens_by_img_t = {}

    for c, t in enumerate(t_values):
        for r, (qy, qx) in enumerate(q_positions):
            mask_q, sens_q = build_fm_fields_for_q(U, LA, Vh, t, qy, qx, C, H, W)

            masks_by_img_t[(r, c)] = mask_q
            sens_by_img_t[(r, c)] = sens_q

            print(
                f"t={t:.2f} | img={r} | q=({qy},{qx}) | "
                f"active spatial pixels={int(mask_q.sum().item())}"
            )

    plot_sensitivity_heatmap_grid(
        fields_by_img_t=sens_by_img_t,
        q_positions=q_positions,
        t_values=t_values,
        out_path=out_dir / "fm_sensitivity_heatmap.png",
        cmap="turbo",
    )

    plot_field_on_xt_grid(
        x0_batch=x0_batch,
        x1_batch=x1_batch,
        fields_by_img_t=masks_by_img_t,
        q_positions=q_positions,
        t_values=t_values,
        out_path=out_dir / "fm_binary_mask_on_xt.png",
        cmap="viridis",
        alpha=0.85,
        mask_zeros=True,
    )


if __name__ == "__main__":
    main()
