import argparse
import csv
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import HRDownscaleDataset, prepare_div2k_patch_paths
from model import Downscaler


CROP_SIZE = 512
SCALE = 2
PATCHES_PER_IMAGE = 8
INITIAL_LR = 0.001
MIN_LR = 0.00003
LR_LOSS_WEIGHT = 1.0
HR_LOSS_WEIGHT = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RGB neural downscaler on DIV2K.")
    parser.add_argument("--n-residual-blocks", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint (.pth state dict) to load before training.")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Epochs between validation runs and checkpoint saves.")
    return parser.parse_args()


def mse(pred, target):
    return (pred - target).square().mean()


def create_dataloaders(batch_size):
    train_groups = prepare_div2k_patch_paths(split="train")
    val_groups = prepare_div2k_patch_paths(split="validation")

    train_dataset = HRDownscaleDataset(
        train_groups,
        crop_size=CROP_SIZE,
        scale=SCALE,
        train=True,
        patches_per_image=PATCHES_PER_IMAGE,
    )
    val_dataset = HRDownscaleDataset(
        val_groups,
        crop_size=CROP_SIZE,
        scale=SCALE,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"DIV2K train images={len(train_groups)} validation images={len(val_groups)}")
    print(f"train patches per epoch={len(train_dataset)} validation patches={len(val_dataset)}")
    return train_dataloader, val_dataloader


def compute_losses(model, hr_images, lr_targets):
    pred_lr, pred_hr = model(hr_images)
    lr_loss = mse(pred_lr, lr_targets)
    hr_loss = mse(pred_hr, hr_images)
    loss = LR_LOSS_WEIGHT * lr_loss + HR_LOSS_WEIGHT * hr_loss
    return loss, lr_loss, hr_loss


def mean_metrics(losses, lr_losses, hr_losses, images_per_second):
    return {
        "loss": np.mean(losses),
        "lr_loss": np.mean(lr_losses),
        "hr_loss": np.mean(hr_losses),
        "images_per_second": images_per_second,
    }


def evaluate(model, dataloader, device):
    losses = []
    lr_losses = []
    hr_losses = []
    images_per_second = 0.0

    model.eval()
    with torch.no_grad():
        progress = tqdm(dataloader, desc="validation", leave=False)
        for hr_images, lr_targets in progress:
            batch_start_time = time.perf_counter()
            hr_images = hr_images.to(device)
            lr_targets = lr_targets.to(device)

            loss, lr_loss, hr_loss = compute_losses(model, hr_images, lr_targets)

            losses.append(loss.item())
            lr_losses.append(lr_loss.item())
            hr_losses.append(hr_loss.item())
            batch_time = time.perf_counter() - batch_start_time
            images_per_second = hr_images.shape[0] / max(batch_time, 1e-9)
            progress.set_postfix(
                loss=f"{np.mean(losses):.6f}",
                lr=f"{np.mean(lr_losses):.6f}",
                hr=f"{np.mean(hr_losses):.6f}",
                img_s=f"{images_per_second:.1f}",
            )

    return mean_metrics(losses, lr_losses, hr_losses, images_per_second)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    losses = []
    lr_losses = []
    hr_losses = []
    images_per_second = 0.0

    model.train()
    batch_progress = tqdm(dataloader, desc=f"train epoch {epoch}", leave=False)
    for hr_images, lr_targets in batch_progress:
        batch_start_time = time.perf_counter()
        hr_images = hr_images.to(device)
        lr_targets = lr_targets.to(device)

        optimizer.zero_grad()
        loss, lr_loss, hr_loss = compute_losses(model, hr_images, lr_targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lr_losses.append(lr_loss.item())
        hr_losses.append(hr_loss.item())
        batch_time = time.perf_counter() - batch_start_time
        images_per_second = hr_images.shape[0] / max(batch_time, 1e-9)
        batch_progress.set_postfix(
            loss=f"{np.mean(losses):.6f}",
            lr=f"{np.mean(lr_losses):.6f}",
            hr=f"{np.mean(hr_losses):.6f}",
            img_s=f"{images_per_second:.1f}",
        )

    return mean_metrics(losses, lr_losses, hr_losses, images_per_second)


def register_checkpoint(model, epoch, val_metrics):
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    raw_model = unwrap_compiled_model(model)
    torch.save(raw_model.state_dict(), checkpoint_dir / f"checkpoint-{epoch}.pth")

    metrics_path = Path("checkpoints.csv")
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "loss", "lr_loss", "hr_loss", "timestamp"])
        writer.writerow([
            epoch,
            val_metrics["loss"],
            val_metrics["lr_loss"],
            val_metrics["hr_loss"],
            datetime.datetime.now().isoformat(timespec="seconds"),
        ])


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


def main():
    args = parse_args()

    print(f"using args: {args}")
    print(f"preparing dataset")
    train_dataloader, val_dataloader = create_dataloaders(args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    model = Downscaler(c_in=3, n_residual_blocks=args.n_residual_blocks).to(device)

    print(f"loading checkpoint")
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"loaded checkpoint from {args.checkpoint}")

    print(f"compiling model")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=MIN_LR,
    )

    epoch_progress = tqdm(range(args.epochs), desc="epochs")
    for epoch in epoch_progress:
        train_metrics = train_one_epoch(model, train_dataloader, optimizer, device, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_progress.set_postfix(
            loss=f"{train_metrics['loss']:.6f}",
            lr_loss=f"{train_metrics['lr_loss']:.6f}",
            hr_loss=f"{train_metrics['hr_loss']:.6f}",
            img_s=f"{train_metrics['images_per_second']:.1f}",
            lr=f"{current_lr:.6g}",
        )
        tqdm.write(
            f"epoch {epoch}: loss={train_metrics['loss']:.6f} "
            f"lr_loss={train_metrics['lr_loss']:.6f} "
            f"hr_loss={train_metrics['hr_loss']:.6f} "
            f"img/s={train_metrics['images_per_second']:.1f} lr={current_lr:.6g} "
        )

        if epoch == 0 or (epoch + 1) % args.checkpoint_interval == 0:
            val_metrics = evaluate(model, val_dataloader, device)
            tqdm.write(
                f"validation epoch {epoch}: loss={val_metrics['loss']:.6f} "
                f"lr_loss={val_metrics['lr_loss']:.6f} "
                f"hr_loss={val_metrics['hr_loss']:.6f} "
                f"img/s={val_metrics['images_per_second']:.1f} "
            )
            scheduler.step(val_metrics["loss"])
            register_checkpoint(model, epoch, val_metrics)

    torch.save(unwrap_compiled_model(model).state_dict(), "model.pth")


if __name__ == "__main__":
    main()
