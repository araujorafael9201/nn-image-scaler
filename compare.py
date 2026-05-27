import argparse
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms


DIV2K_FOLDERS = {
    "train": "DIV2K_train_HR",
    "validation": "DIV2K_valid_HR",
}


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
    def __init__(self, c_in: int = 3, n_residual_blocks: int = 10):
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
    model = Downscaler(c_in=3, n_residual_blocks=10).to(device)
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


def center_crop_pil(image, crop_size):
    if crop_size is None or crop_size <= 0:
        return image

    width, height = image.size
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def load_urban100_hr_images(seed, max_samples, crop_size):
    import datasets

    raw_dataset = datasets.load_dataset("Voxel51/Urban100", split="train")
    hr_indices = build_urban100_hr_indices(raw_dataset)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(hr_indices), generator=generator).tolist()
    selected = [hr_indices[i] for i in order[:max_samples]]

    print(f"Urban100: {len(raw_dataset)} raw rows -> {len(hr_indices)} HR images")
    print(f"comparison samples: {len(selected)}")

    return [
        center_crop_pil(raw_dataset[idx]["image"].convert("RGB"), crop_size)
        for idx in selected
    ]


def load_div2k_hr_images(root, split, seed, max_samples, crop_size):
    folder = Path(root) / DIV2K_FOLDERS[split]
    image_paths = sorted(folder.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(
            f"No DIV2K PNGs found in {folder}. Expected files like "
            f"data/div2k/{DIV2K_FOLDERS[split]}/0801.png."
        )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(image_paths), generator=generator).tolist()
    selected = [image_paths[i] for i in order[:max_samples]]

    print(f"DIV2K {split}: {len(image_paths)} HR images in {folder}")
    print(f"comparison samples: {len(selected)}")

    images = []
    for path in selected:
        image = Image.open(path).convert("RGB")
        image = center_crop_pil(image, crop_size)
        print(f"{path.name}: using {image.width}x{image.height}")
        images.append(image)
    return images


def load_hr_images(args):
    if args.dataset == "urban100":
        return load_urban100_hr_images(args.seed, args.max_samples, args.crop_size)
    return load_div2k_hr_images(
        args.div2k_root,
        args.div2k_split,
        args.seed,
        args.max_samples,
        args.crop_size,
    )


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
    pred_lr, _ = model(X)
    return tensor_to_pil(pred_lr[0])


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_downscale(fn, device):
    synchronize_if_needed(device)
    start_time = time.perf_counter()
    image = fn()
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    return image, elapsed


def format_elapsed(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def save_image_grid(images, labels, output_path):
    label_height = 46
    padding = 16
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)
    total_height = max(img.height for img in images) + label_height
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label in zip(images, labels):
        draw.multiline_text((x, 6), label, fill="black", spacing=2)
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
            labels = [f"source_HR\n{hr_image.width}x{hr_image.height}"]
            images = [hr_image]

            for label, resample in methods:
                downscaled, elapsed = timed_downscale(
                    lambda resample=resample: resize_pil(hr_image, lr_size, resample),
                    device,
                )
                labels.append(f"{label}\n{format_elapsed(elapsed)}")
                images.append(downscaled)

            downscaled, elapsed = timed_downscale(
                lambda: neural_downscale(model, hr_image, device),
                device,
            )
            labels.append(f"neural_RGB\n{format_elapsed(elapsed)}")
            images.append(downscaled)

            save_image_grid(images, labels, output_dir / f"sample_{sample_id:03d}_comparison.png")

    return len(hr_images)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare native HR downsampling methods.")
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_outputs"))
    parser.add_argument("--dataset", choices=("div2k", "urban100"), default="div2k")
    parser.add_argument("--div2k-root", type=Path, default=Path("data/div2k"))
    parser.add_argument("--div2k-split", choices=tuple(DIV2K_FOLDERS), default="validation")
    parser.add_argument("--crop-size", type=int, default=0, help="Center crop size before downscaling. Default 0 uses full images.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"using device {device}")
    model = load_model(args.model, device)
    hr_images = load_hr_images(args)
    saved = save_comparisons(model, hr_images, args.output_dir, device)
    print(f"Saved {saved} samples to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
