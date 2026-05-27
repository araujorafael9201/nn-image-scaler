import csv
import datetime
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


class HRDownscaleDataset(Dataset):
    def __init__(self, image_paths, crop_size=512, scale=2, train=False, patches_per_image=1):
        self.image_paths = list(image_paths)
        self.crop_size = crop_size
        self.lr_size = crop_size // scale
        self.train = train
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()
        self.resize_lr = transforms.Resize(
            (self.lr_size, self.lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def _load_image(self, item):
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        return Image.open(item).convert("RGB")

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx % len(self.image_paths)])

        if self.train:
            top, left, height, width = transforms.RandomCrop.get_params(
                image,
                output_size=(self.crop_size, self.crop_size),
            )
            X = transforms.functional.crop(image, top, left, height, width)
        else:
            X = transforms.functional.center_crop(image, (self.crop_size, self.crop_size))

        y = self.resize_lr(X)
        X = self.to_tensor(X)
        y = self.to_tensor(y)

        return X, y


DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "validation": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
DIV2K_FOLDERS = {
    "train": "DIV2K_train_HR",
    "validation": "DIV2K_valid_HR",
}


def prepare_div2k_hr_paths(root="data/div2k", split="train"):
    root = Path(root)
    folder = root / DIV2K_FOLDERS[split]
    if not folder.exists():
        download_and_extract_archive(
            DIV2K_URLS[split],
            download_root=str(root),
            filename=f"{DIV2K_FOLDERS[split]}.zip",
        )
    return sorted(folder.glob("*.png"))


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int = 16, c_out: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        return self.relu(self.block(X) + self.skip(X))


class Downscaler(nn.Module):
    def __init__(self, c_in: int = 3, n_residual_blocks: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(c_in=16, c_out=16) for _ in range(n_residual_blocks)
        ])
        self.lr_head = nn.Conv2d(16, c_in, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c_in, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, c_in, kernel_size=3, padding=1),
        )

    def forward(self, X):
        encoded = self.encoder(X)
        features_lr = self.res_blocks(encoded)
        pred_lr = self.lr_head(features_lr)
        pred_hr = self.decoder(pred_lr)
        return pred_lr, pred_hr


device = None


lr_loss_weight = 1.0
hr_loss_weight = 1.0


def mse(pred, target):
    return (pred - target).square().mean()


initial_lr = 0.001
validation_interval = 5


def evaluate(model, dataloader):
    losses = []
    lr_losses = []
    hr_losses = []

    model.eval()
    with torch.no_grad():
        progress = tqdm(dataloader, desc="validation", leave=False)
        images_per_second = 0.0
        for X, y in progress:
            batch_start_time = time.perf_counter()
            X, y = X.to(device), y.to(device)

            pred_lr, pred_hr = model(X)
            lr_loss = mse(pred_lr, y)
            hr_loss = mse(pred_hr, X)
            loss = lr_loss_weight * lr_loss + hr_loss_weight * hr_loss

            losses.append(loss.item())
            lr_losses.append(lr_loss.item())
            hr_losses.append(hr_loss.item())
            batch_time = time.perf_counter() - batch_start_time
            images_per_second = X.shape[0] / max(batch_time, 1e-9)
            progress.set_postfix(
                loss=f"{np.mean(losses):.6f}",
                lr=f"{np.mean(lr_losses):.6f}",
                hr=f"{np.mean(hr_losses):.6f}",
                img_s=f"{images_per_second:.1f}",
            )

    return {
        "loss": np.mean(losses),
        "lr_loss": np.mean(lr_losses),
        "hr_loss": np.mean(hr_losses),
        "images_per_second": images_per_second,
    }


def main():
    global device

    crop_size = 512
    scale = 2
    batch_size = 48
    patches_per_image = 8

    train_hr_paths = prepare_div2k_hr_paths(split="train")
    val_hr_paths = prepare_div2k_hr_paths(split="validation")

    train_dataset = HRDownscaleDataset(
        train_hr_paths,
        crop_size=crop_size,
        scale=scale,
        train=True,
        patches_per_image=patches_per_image,
    )
    val_dataset = HRDownscaleDataset(
        val_hr_paths,
        crop_size=crop_size,
        scale=scale,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    print(f"DIV2K train images={len(train_hr_paths)} validation images={len(val_hr_paths)}")
    print(f"train patches per epoch={len(train_dataset)} validation patches={len(val_dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    model = Downscaler(c_in=3, n_residual_blocks=10).to(device)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00003,
    )

    epoch_progress = tqdm(range(50), desc="epochs")
    for epoch in epoch_progress:
        losses = []
        lr_losses = []
        hr_losses = []
        images_per_second = 0.0

        model.train()
        batch_progress = tqdm(train_dataloader, desc=f"train epoch {epoch}", leave=False)
        for X, y in batch_progress:
            batch_start_time = time.perf_counter()
            X, y = X.to(device), y.to(device)

            optim.zero_grad()

            pred_lr, pred_hr = model(X)
            lr_loss = mse(pred_lr, y)
            hr_loss = mse(pred_hr, X)

            loss = lr_loss_weight * lr_loss + hr_loss_weight * hr_loss

            loss.backward()
            optim.step()

            losses.append(loss.item())
            lr_losses.append(lr_loss.item())
            hr_losses.append(hr_loss.item())
            batch_time = time.perf_counter() - batch_start_time
            images_per_second = X.shape[0] / max(batch_time, 1e-9)
            batch_progress.set_postfix(
                loss=f"{np.mean(losses):.6f}",
                lr=f"{np.mean(lr_losses):.6f}",
                hr=f"{np.mean(hr_losses):.6f}",
                img_s=f"{images_per_second:.1f}",
            )

        loss = np.mean(losses)
        lr_loss = np.mean(lr_losses)
        hr_loss = np.mean(hr_losses)

        current_lr = optim.param_groups[0]["lr"]
        epoch_progress.set_postfix(
            loss=f"{loss:.6f}",
            lr_loss=f"{lr_loss:.6f}",
            hr_loss=f"{hr_loss:.6f}",
            img_s=f"{images_per_second:.1f}",
            lr=f"{current_lr:.6g}",
        )
        tqdm.write(
            f"epoch {epoch}: loss={loss:.6f} lr_loss={lr_loss:.6f} "
            f"hr_loss={hr_loss:.6f} img/s={images_per_second:.1f} lr={current_lr:.6g} "
        )

        if epoch == 0 or (epoch + 1) % validation_interval == 0:
            val_metrics = evaluate(model, val_dataloader)
            tqdm.write(
                f"validation epoch {epoch}: loss={val_metrics['loss']:.6f} "
                f"lr_loss={val_metrics['lr_loss']:.6f} "
                f"hr_loss={val_metrics['hr_loss']:.6f} "
                f"img/s={val_metrics['images_per_second']:.1f} "
            )
            scheduler.step(val_metrics["loss"])

            register_checkpoint(model, epoch, val_metrics)

    raw_model = getattr(model, "_orig_mod", model)
    torch.save(raw_model.state_dict(), "model.pth")


def register_checkpoint(model, epoch, val_metrics):
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)
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


if __name__ == "__main__":
    main()
