from typing import Type
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstructed_x = self.decoder(z)

        return reconstructed_x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            encoder_class: Type[Encoder] = Encoder,
            decoder_class: Type[Decoder] = Decoder,
            criterion_class: Type[torch.optim.Optimizer] = nn.MSELoss,
    ) -> None:
        super().__init__()

        self.enc = encoder_class()
        self.dec = decoder_class()
        self.criterion = criterion_class()

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
