"""Creates centroids for LHQ by passing through CLIP.

Usage:
    # 1. Download the LHQ dataset; put it in the root of this repo as
    #    `./lhq_256_root/lhq`
    # 2. Run the following from root of this repo:
    uv run -m src.cvt.lhq_centroids N_CENTROIDS

Examples:
    uv run -m src.cvt.lhq_centroids --n_centroids 10000 --mode clip --samples all --directory lhq_256_root
"""
import random

import clip  # install with: pip install git+https://github.com/openai/CLIP.git
import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    BICUBIC = transforms.InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# ----- Config -----
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    n_centroids: int = 1000,
    mode: str = "clip",
    samples: str = "all",
    directory: str = "./lhq_256_root",
):
    if mode == "clip":
        # ----- Load CLIP -----
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        # ----- Load dataset -----
        dataset = datasets.ImageFolder(directory, transform=preprocess)

        if samples == "all":
            # Sample random indices
            indices = random.sample(range(len(dataset)), n_centroids)
        else:
            # Find the label index corresponding to `samples`.
            class_idx = dataset.class_to_idx[samples]

            # Get all indices where the label is `samples`.
            indices = [
                i for i, (_, label) in enumerate(dataset.samples)
                if label == class_idx
            ]

            # Sample from only given indices
            indices = random.sample(indices, n_centroids)

        subset = Subset(dataset, indices)

        # Retrieve file paths manually from subset
        file_paths = [dataset.samples[i][0] for i in indices]
        labels = [dataset.samples[i][1] for i in indices]

        # Prepare DataLoader
        dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

        # ----- Encode images -----
        all_embeddings = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Encoding images with CLIP"):
                images = images.to(DEVICE)
                embeddings = model.encode_image(images)
                embeddings = F.normalize(embeddings, dim=-1)
                embeddings = embeddings.detach().cpu().numpy()
                all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)

    elif mode == "image":

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        res = 128

        preprocess = transforms.Compose([
            transforms.Resize(res, interpolation=BICUBIC),
            transforms.CenterCrop(res),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            #  Normalize((0.48145466, 0.4578275, 0.40821073),
            #            (0.26862954, 0.26130258, 0.27577711)),
        ])

        # ----- Load dataset -----
        dataset = datasets.ImageFolder(directory, transform=preprocess)

        # Sample random indices
        indices = random.sample(range(len(dataset)), n_centroids)
        subset = Subset(dataset, indices)

        # Retrieve file paths manually from subset
        file_paths = [dataset.samples[i][0] for i in indices]
        labels = [dataset.samples[i][1] for i in indices]

        # Prepare DataLoader
        dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

        # ----- Encode images -----
        all_embeddings = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Converting to numpy"):
                print("Image shape:", images.shape)
                all_embeddings.append(images.detach().cpu().numpy().reshape(
                    len(images), -1))

        all_embeddings = np.vstack(all_embeddings)
        print("Final shape:", all_embeddings.shape)

    else:
        raise NotImplementedError()

    # ----- Save results -----

    if samples != "all":
        embeddings_file = f"lhq_{samples}_{mode}_embeddings.npy"
        labels_file = f"lhq_{samples}_{mode}_labels.npy"
        paths_file = f"lhq_{samples}_{mode}_paths.txt"
    else:
        embeddings_file = f"lhq_{mode}_embeddings.npy"
        labels_file = f"lhq_{mode}_labels.npy"
        paths_file = f"lhq_{mode}_paths.txt"

    np.save(embeddings_file, all_embeddings)
    np.save(labels_file, np.array(labels))

    with open(paths_file, 'w') as f:
        f.writelines([path + "\n" for path in file_paths])

    print(
        f"Saved:\n - Embeddings: {embeddings_file}\n - Labels: {labels_file}\n - Paths: {paths_file}"
    )


if __name__ == "__main__":
    fire.Fire(main)
