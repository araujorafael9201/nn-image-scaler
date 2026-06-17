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
        self.images = [
            img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            for img in image_paths
        ]

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        image = self.images[idx % len(self.images)]

        if self.train:
            top, left, height, width = transforms.RandomCrop.get_params(
                image,
                output_size=(self.crop_size, self.crop_size),
            )
            hr_image = transforms.functional.crop(image, top, left, height, width)
        else:
            hr_image = transforms.functional.center_crop(image, (self.crop_size, self.crop_size))

        lr_image = self.resize_lr(hr_image)
        return self.to_tensor(hr_image), self.to_tensor(lr_image)


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
