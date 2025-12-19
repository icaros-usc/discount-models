"""MLP GAN for MNIST."""
import torch
from torch import nn


class MLPGenerator(nn.Module):
    """MLP generator network for MNIST GAN."""

    def __init__(self, nz):
        super().__init__()

        # Size of the latent space (number of dimensions).
        self.nz = nz

        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return self.main(x).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    """MLP discriminator network for MNIST GAN."""

    def __init__(self):
        super().__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)
