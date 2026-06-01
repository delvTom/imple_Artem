from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.utils.wiener import compute_wiener_filter, save_wiener_filter, load_wiener_filter




def plot_mask_grid(x_display, masks_by_q_t, q_positions, t_values, out_path):
    n_rows = len(q_positions)
    n_cols = len(t_values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    H, W = x_display.shape

    for r, (qy, qx) in enumerate(q_positions):
        for c, t in enumerate(t_values):
            ax = axes[r, c]

            mask = masks_by_q_t[(r, c)]

            ax.imshow(x_display, cmap="gray")
            ax.imshow(mask, cmap="viridis", alpha=0.55)

            ax.scatter([qx], [qy], c="red", s=20)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"t={t:.2f}")
            if c == 0:
                ax.set_ylabel(f"q=({qy},{qx})")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_sensitivity_on_xt_grid(x0, x1, masks_by_q_t, q_positions, t_values, out_path):
    n_rows = len(q_positions)
    n_cols = len(t_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for c, t in enumerate(t_values):
        xt = (1.0 - t) * x0 + t * x1
        x_bg = xt[0, 0].detach().cpu()

        for r, (qy, qx) in enumerate(q_positions):
            ax = axes[r, c]
            mask_q = masks_by_q_t[(r, c)]

            ax.imshow(x_bg, cmap="gray", vmin=-1, vmax=1)
            ax.imshow(mask_q, cmap="viridis", alpha=0.55)
            ax.scatter([qx], [qy], c="red", s=18)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"t={t:.2f}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"q=({qy},{qx})", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)




def plot_xt_trajectory(x0, x1, t_values, out_path):
    n_cols = len(t_values) + 2

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(2.5 * n_cols, 2.5),
    )

    panels = [("x0 noise", x0)]
    panels += [(f"t={t:.2f}", (1.0 - t) * x0 + t * x1) for t in t_values]
    panels += [("x1 data", x1)]

    for ax, (title, img) in zip(axes, panels):
        img = img[0, 0].detach().cpu()

        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_field_on_xt_grid(
    x0,
    x1,
    fields_by_q_t,
    q_positions,
    t_values,
    out_path,
    *,
    cmap="turbo",
    alpha=0.85,
    title_prefix="",
):
    n_rows = len(q_positions)
    n_cols = len(t_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )

    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for c, t in enumerate(t_values):
        xt = (1.0 - t) * x0 + t * x1
        x_bg = xt[0, 0].detach().cpu()

        for r, (qy, qx) in enumerate(q_positions):
            ax = axes[r, c]
            field_q = fields_by_q_t[(r, c)]

            ax.imshow(x_bg, cmap="gray", vmin=-1, vmax=1)
            ax.imshow(field_q, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
            ax.scatter([qx], [qy], c="red", s=18)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"{title_prefix}t={t:.2f}", fontsize=12)
            if c == 0:
                ax.set_ylabel(f"q=({qy},{qx})", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def build_fm_mask(U, LA, Vh, t, threshold=0.02, eps=1e-6):
    sigma2 = ((1.0 - t) / t) ** 2

    shrink = LA / (LA + sigma2)
    LLt = U @ torch.diag(shrink) @ Vh

    denom = torch.diagonal(LLt).unsqueeze(1)
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)

    normalized = LLt / denom
    mask = (normalized.abs() >= threshold).float()
    return mask, normalized


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = load_config("configs/wiener/mnist.yaml", overrides=[
        "metrics.wandb.enabled=false",
        "dataset.batch_size=512",
    ])

    dataset = build_dataset(cfg.dataset)

    wiener_path = Path("data/models/wiener/mnist_28")

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

    H = W = dataset.resolution
    C = dataset.in_channels

    # Reproductibilité
    torch.manual_seed(42)
    generator = torch.Generator(device=device).manual_seed(42)

    # Récupérer une vraie image x1 depuis MNIST
    first_batch = next(iter(dataset.dataloader))
    images = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    x1 = images[0:1].to(device)  # [1, C, H, W]

    # Générer un bruit x0
    x0 = torch.randn(x1.shape, device=device, generator=generator)

    # Plusieurs pixels q à inspecter
    q_positions = [
        (H // 2, W // 2),      # centre
        (H // 2 - 5, W // 2),  # haut du chiffre
        (H // 2 + 5, W // 2),  # bas du chiffre
        (H // 2, W // 2 - 5),  # gauche
        (H // 2, W // 2 + 5),  # droite
    ]

    t_values = [0.1, 0.4, 0.7, 0.9]


    out_dir = Path("experiments_tom/figures/fm_masks_mnist")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_xt_trajectory(
    x0=x0,
    x1=x1,
    t_values=t_values,
    out_path=out_dir / "fm_xt_trajectory.png",
    )



    # Construire une grande figure : lignes = pixels q, colonnes = t
    masks_by_q_t = {}
    sens_by_q_t = {}

    active_counts = {}

    for c, t in enumerate(t_values):
        mask, normalized = build_fm_mask(U, LA, Vh, t)

        for r, (qy, qx) in enumerate(q_positions):
            q = qy * W + qx

            mask_q = mask[q].reshape(C, H, W)[0].detach().cpu() 
            sens_q = normalized[q].abs().reshape(C, H, W)[0].detach().cpu()
            sens_q = sens_q / (sens_q.max() + 1e-8)


            masks_by_q_t[(r, c)] = mask_q
            sens_by_q_t[(r, c)] = sens_q
            active_counts[(r, c)] = int(mask_q.sum().item())
            

            print(
                f"t={t:.2f} | q=({qy},{qx}) | active pixels={int(mask_q.sum().item())}"
            )
    
  

    plot_field_on_xt_grid(
    x0=x0,
    x1=x1,
    fields_by_q_t=sens_by_q_t,
    q_positions=q_positions,
    t_values=t_values,
    out_path=out_dir / "fm_sensitivity_continuous_on_xt.png",
    cmap="viridis",
    alpha=0.65,
    )

    plot_field_on_xt_grid(
    x0=x0,
    x1=x1,
    fields_by_q_t=masks_by_q_t,
    q_positions=q_positions,
    t_values=t_values,
    out_path=out_dir / "fm_binary_mask_on_xt.png",
    cmap="viridis",
    alpha=0.85,
    )

   


if __name__ == "__main__":
    main()