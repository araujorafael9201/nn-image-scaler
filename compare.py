import argparse
import re
from pathlib import Path

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


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
    def __init__(self, c_in: int = 1, n_residual_blocks: int = 10):
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


def rgb_to_yuv(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat((y, u, v), dim=1)


def yuv_to_rgb(yuv):
    y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat((r, g, b), dim=1)


def build_urban100_hr_indices(raw_dataset, scale="2", image_key="image"):
    pattern = re.compile(r"img_(\d+)_SRF_(\d+)_(.+)\.png$")
    hr_indices = []

    for idx in range(len(raw_dataset)):
        image = raw_dataset[idx][image_key]
        filename = getattr(image, "filename", "") or ""
        match = pattern.search(filename)
        if match is None:
            continue

        image_id, image_scale, method = match.groups()
        if image_scale == scale and method == "HR":
            hr_indices.append((int(image_id), idx))

    return [idx for _, idx in sorted(hr_indices)]


def load_model(model_path, device):
    model = Downscaler(c_in=1, n_residual_blocks=10).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        model = checkpoint.to(device)
        model.eval()
        return model

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_hr_images(seed, max_samples):
    raw_dataset = datasets.load_dataset("Voxel51/Urban100", split="train")
    hr_indices = build_urban100_hr_indices(raw_dataset)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(hr_indices), generator=generator).tolist()
    selected = [hr_indices[i] for i in order[:max_samples]]

    print(f"{len(raw_dataset)} raw rows -> {len(hr_indices)} HR images")
    print(f"comparison samples: {len(selected)}")

    return [raw_dataset[idx]["image"].convert("RGB") for idx in selected]


def tensor_to_pil(image):
    image = image.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0)
    image = (image * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(image)


def pil_to_tensor(image):
    return transforms.ToTensor()(image).unsqueeze(0)


def downscale_size(image):
    width, height = image.size
    return (width // 2, height // 2)


def resize_pil(image, size, resample):
    return image.resize(size, resample=resample)


def neural_downscale(model, image, device):
    X = pil_to_tensor(image).to(device)
    X_yuv = rgb_to_yuv(X)
    pred_lr_luma, _ = model(X_yuv[:, 0:1])
    uv_lr = F.interpolate(
        X_yuv[:, 1:3],
        size=pred_lr_luma.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    pred_lr = yuv_to_rgb(torch.cat((pred_lr_luma, uv_lr), dim=1))
    return tensor_to_pil(pred_lr[0])


def save_image_grid(images, labels, output_path):
    label_height = 32
    padding = 16
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)
    total_height = max(img.height for img in images) + label_height
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label in zip(images, labels):
        draw.text((x, 8), label, fill="black")
        canvas.paste(img, (x, label_height))
        x += img.width + padding

    canvas.save(output_path)


def save_comparisons(model, hr_images, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("nearest", Image.Resampling.NEAREST),
        ("bilinear", Image.Resampling.BILINEAR),
        ("bicubic", Image.Resampling.BICUBIC),
        ("lanczos", Image.Resampling.LANCZOS),
    ]

    with torch.no_grad():
        for sample_id, hr_image in enumerate(hr_images):
            lr_size = downscale_size(hr_image)
            labels = ["source_HR"]
            images = [hr_image]

            for label, resample in methods:
                labels.append(label)
                images.append(resize_pil(hr_image, lr_size, resample))

            labels.append("neural_Y_bicubic_UV")
            images.append(neural_downscale(model, hr_image, device))

            save_image_grid(images, labels, output_dir / f"sample_{sample_id:03d}_comparison.png")

    return len(hr_images)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare native HR downsampling methods.")
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_outputs"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, device)
    hr_images = load_hr_images(args.seed, args.max_samples)
    saved = save_comparisons(model, hr_images, args.output_dir, device)
    print(f"Saved {saved} samples to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
