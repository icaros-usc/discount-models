import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from .autoencoder import AEDecoder, VAEDecoder


def main(file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = AEDecoder().to(device)
    generator.load_state_dict(torch.load(file, map_location=device))
    z = torch.randn((25, 4, 10, 10))
    imgs = (generator(z) + 1.0) / 2.0  # pylint: disable = not-callable

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    img_grid = make_grid(imgs, nrow=5, padding=0)
    img_grid = np.transpose(img_grid.cpu().numpy(), (1, 2, 0))
    ax.imshow(img_grid)
    fig.savefig("samples.png")


if __name__ == "__main__":
    fire.Fire(main)
