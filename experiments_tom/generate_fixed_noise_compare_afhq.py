from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.metrics import calculate_mse, calculate_r2_score
from local_diffusion.models import create_model
from local_diffusion.models.baseline_unet import BaselineUNet


@torch.no_grad()
def sample_from_fixed_latents(model, initial_latents, *, return_intermediates=False):
    latents = initial_latents.clone()
    trajectory_xt = []
    trajectory_x0 = []
    timesteps = []
    last_pred_x0 = None

    for timestep in tqdm(model.scheduler.timesteps, desc=f"Sampling {model.__class__.__name__}"):
        pred_x0 = model.denoise(latents, timestep)
        predicted_noise = model.compute_noise_from_x0(latents, pred_x0, timestep)
        step_output = model.scheduler.step(
            model_output=predicted_noise,
            timestep=timestep,
            sample=latents,
        )

        if return_intermediates:
            trajectory_xt.append(latents.detach().cpu())
            trajectory_x0.append(pred_x0.detach().cpu())
            timesteps.append(int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep))

        last_pred_x0 = pred_x0
        latents = step_output.prev_sample

    if last_pred_x0 is None:
        raise RuntimeError("Sampling loop did not execute.")

    return {
        "images": last_pred_x0.detach().cpu(),
        "trajectory_xt": trajectory_xt,
        "trajectory_x0": trajectory_x0,
        "timesteps": timesteps,
    }


def save_final_comparison(dataset, initial_latents, artem_images, unet_images, out_path):
    panels = []
    for idx in range(initial_latents.shape[0]):
        panels.append(dataset.postprocess(initial_latents[idx : idx + 1]).cpu()[0])
        panels.append(dataset.postprocess(artem_images[idx : idx + 1]).cpu()[0])
        panels.append(dataset.postprocess(unet_images[idx : idx + 1]).cpu()[0])

    grid = make_grid(panels, nrow=3, padding=2, normalize=False)
    save_image(grid, out_path)


def save_trajectory_comparison(dataset, artem_result, unet_result, out_dir, max_steps=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    timesteps = artem_result["timesteps"]
    if not timesteps:
        return

    step_indices = torch.linspace(0, len(timesteps) - 1, steps=min(max_steps, len(timesteps))).long().tolist()

    for sample_idx in range(artem_result["images"].shape[0]):
        panels = []
        titles = []
        for step_idx in step_indices:
            panels.append(dataset.postprocess(artem_result["trajectory_x0"][step_idx][sample_idx : sample_idx + 1]).cpu()[0])
            panels.append(dataset.postprocess(unet_result["trajectory_x0"][step_idx][sample_idx : sample_idx + 1]).cpu()[0])
            titles.append(f"t={timesteps[step_idx]}")

        grid = make_grid(panels, nrow=2, padding=2, normalize=False)
        save_image(grid, out_dir / f"sample_{sample_idx:02d}_x0_pred_trajectory.png")

        fig, ax = plt.subplots(figsize=(3, 0.4 * len(titles)))
        ax.axis("off")
        ax.text(
            0,
            1,
            "\n".join(f"{title}: Artem | UNet" for title in titles),
            va="top",
            fontsize=9,
        )
        fig.savefig(out_dir / f"sample_{sample_idx:02d}_legend.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pca_locality/afhq.yaml")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments_tom/results/fixed_noise_compare_afhq"),
    )
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        overrides=[
            "metrics.wandb.enabled=false",
            "dataset.batch_size=256",
            f"sampling.num_inference_steps={args.num_steps}",
            f"sampling.num_samples={args.num_samples}",
            f"sampling.batch_size={args.batch_size}",
            f"experiment.seed={args.seed}",
        ],
    )

    device = torch.device(cfg.experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print("device:", device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg.dataset)

    model_params = cfg.model.params
    artem_model = create_model(
        cfg.model.name,
        dataset=dataset,
        device=str(device),
        num_steps=args.num_steps,
        params=model_params,
    )
    artem_model.train(dataset)

    baseline_path = cfg.metrics.baseline_path
    if baseline_path is None:
        raise ValueError("Config must provide metrics.baseline_path for UNet comparison.")

    unet_model = BaselineUNet(
        resolution=dataset.resolution,
        device=str(device),
        num_steps=args.num_steps,
        model_path=baseline_path,
        dataset_name=cfg.dataset.name,
        in_channels=dataset.in_channels,
        out_channels=dataset.in_channels,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.num_samples, dataset.in_channels, dataset.resolution, dataset.resolution)
    initial_latents = torch.randn(shape, generator=generator, device=device)
    initial_latents = initial_latents * artem_model.scheduler.init_noise_sigma

    artem_result = sample_from_fixed_latents(artem_model, initial_latents, return_intermediates=True)
    unet_result = sample_from_fixed_latents(unet_model, initial_latents, return_intermediates=True)

    artem_images = artem_result["images"]
    unet_images = unet_result["images"]

    save_final_comparison(
        dataset,
        initial_latents.detach().cpu(),
        artem_images,
        unet_images,
        args.out_dir / "fixed_noise_final_comparison.png",
    )
    save_trajectory_comparison(
        dataset,
        artem_result,
        unet_result,
        args.out_dir / "trajectories",
    )

    metrics = {
        "mse_final_x0": calculate_mse(artem_images, unet_images),
        "r2_final_x0": calculate_r2_score(artem_images, unet_images),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "config": args.config,
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)
    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
