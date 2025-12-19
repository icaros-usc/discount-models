"""Makes centroids for Fashion MNIST archive.

Usage:
    uv run -m src.cvt.fashion_mnist_centroids N_CENTROIDS
"""
import random

import fire
import numpy as np
import torch
from torchvision import datasets, transforms


def main(n_centroids: int = 1000):
    # Set random seed for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Load the Fashion MNIST test set.
    transform = transforms.ToTensor()
    mnist_test = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )

    # Select n_centroids random indices.
    indices = rng.choice(
        len(mnist_test),
        size=n_centroids,
        replace=False,
    )

    # Extract images and labels.
    images = []
    labels = []

    for idx in indices:
        img, label = mnist_test[idx]
        img = img.view(-1).numpy()  # Flatten to (784,)
        images.append(img)
        labels.append(label)

    # Convert to numpy arrays.
    images_array = np.stack(images)  # Shape: (n_centroids, 784)
    labels_array = np.array(labels)  # Shape: (n_centroids,)

    print(f"Image stats: Min {np.min(images_array)}, "
          f"Max {np.max(images_array)}")

    # Save to .npy files
    np.save(f'fashion_{n_centroids}_centroids.npy', images_array)
    np.save(f'fashion_{n_centroids}_labels.npy', labels_array)

    print(f"Saved images to 'fashion_{n_centroids}_centroids.npy' with shape",
          images_array.shape)
    print(f"Saved labels to 'fashion_{n_centroids}_labels.npy' with shape",
          labels_array.shape)


if __name__ == "__main__":
    fire.Fire(main)
