import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


class HRDownscaleDataset(Dataset):
    def __init__(self, image_paths, crop_size=512, scale=2, train=False, patches_per_image=1):
        self.crop_size = crop_size
        self.lr_size = crop_size // scale
        self.train = train
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()
        self.resize_lr = transforms.Resize(
            (self.lr_size, self.lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        first = next(iter(image_paths), None)
        if isinstance(first, list):
            self.image_groups = image_paths
            self.pre_cropped = True
        else:
            self.image_groups = [[p] for p in image_paths]
            self.pre_cropped = False

    def __len__(self):
        return len(self.image_groups) * self.patches_per_image

    def __getitem__(self, idx):
        group = self.image_groups[idx % len(self.image_groups)]
        source = random.choice(group)
        image = source if isinstance(source, Image.Image) else Image.open(source).convert("RGB")

        if not self.pre_cropped:
            if self.train:
                top, left, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(self.crop_size, self.crop_size)
                )
                image = transforms.functional.crop(image, top, left, h, w)
            else:
                image = transforms.functional.center_crop(image, (self.crop_size, self.crop_size))

        lr_image = self.resize_lr(image)
        return self.to_tensor(image), self.to_tensor(lr_image)


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


def prepare_div2k_patch_paths(root="data/div2k", split="train", crop_size=512, n_patches=3):
    src_paths = prepare_div2k_hr_paths(root, split)
    cache_dir = Path(root) / f"patches_{crop_size}" / DIV2K_FOLDERS[split]
    cache_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for src in src_paths:
        group = []
        img = None
        for i in range(n_patches):
            dst = cache_dir / f"{src.stem}_p{i}.png"
            if not dst.exists():
                if img is None:
                    img = Image.open(src).convert("RGB")
                top, left, h, w = transforms.RandomCrop.get_params(img, (crop_size, crop_size))
                transforms.functional.crop(img, top, left, h, w).save(dst)
            group.append(dst)
        groups.append(group)
    return groups
