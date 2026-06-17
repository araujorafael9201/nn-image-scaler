import torch.nn as nn


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
    def __init__(self, c_in: int = 3, n_residual_blocks: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(c_in=32, c_out=32) for _ in range(n_residual_blocks)
        ])
        self.lr_head = nn.Conv2d(32, c_in, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c_in, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, c_in, kernel_size=3, padding=1),
        )

    def forward(self, X):
        encoded = self.encoder(X)
        features_lr = self.res_blocks(encoded)
        pred_lr = self.lr_head(features_lr)
        pred_hr = self.decoder(pred_lr)
        return pred_lr, pred_hr
