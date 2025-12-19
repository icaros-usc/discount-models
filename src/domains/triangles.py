"""Provides the Triangles domain.

Adapted from:
https://github.com/google/brain-tokyo-workshop/blob/master/es-clip/painter.py
https://github.com/google/evojax/blob/main/examples/notebooks/AbstractPainting01.ipynb
"""

import math
from functools import partial
from pathlib import Path

import jax
import numpy as np
import torch
import torch.nn.functional as F
from jax import lax
from jax import numpy as jnp
from omegaconf import DictConfig
from PIL import Image
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

from src.domains.domain_base import DomainBase, EvaluateTorchMixin
from src.domains.mnist.dcgan import DCDiscriminator
from src.domains.mnist.mlpgan import MLPDiscriminator


def quantitize(a, quant):
    return lax.cond(quant > 0, lambda v: jnp.rint(v * quant) / quant, lambda v: v, a)


def put_triangle_fn(i_triangle, packed_args):
    canvas, xyv, xy0, xy1, xy2, r, g, b, a, quant = packed_args
    n_batch, h, w = canvas.shape[:3]

    xy0_triangle = lax.dynamic_slice_in_dim(xy0, i_triangle, 1, axis=1)
    xy1_triangle = lax.dynamic_slice_in_dim(xy1, i_triangle, 1, axis=1)
    xy2_triangle = lax.dynamic_slice_in_dim(xy2, i_triangle, 1, axis=1)

    cross_0 = jnp.cross((xyv - xy0_triangle), (xy1_triangle - xy0_triangle))
    cross_1 = jnp.cross((xyv - xy1_triangle), (xy2_triangle - xy1_triangle))
    cross_2 = jnp.cross((xyv - xy2_triangle), (xy0_triangle - xy2_triangle))

    in_triangle_p = jnp.logical_and(
        jnp.logical_and(cross_0 >= 0, cross_1 >= 0), cross_2 >= 0
    )
    in_triangle_n = jnp.logical_and(
        jnp.logical_and(cross_0 <= 0, cross_1 <= 0), cross_2 <= 0
    )
    in_triangle = jnp.logical_or(in_triangle_p, in_triangle_n).reshape((n_batch, h, w))

    region_mask_triangle = jnp.repeat(
        jnp.expand_dims(in_triangle, axis=-1), repeats=3, axis=-1
    ).astype(jnp.float32)

    r_triangle = lax.dynamic_index_in_dim(r, i_triangle, axis=-1, keepdims=False)
    g_triangle = lax.dynamic_index_in_dim(g, i_triangle, axis=-1, keepdims=False)
    b_triangle = lax.dynamic_index_in_dim(b, i_triangle, axis=-1, keepdims=False)
    a_triangle = lax.dynamic_index_in_dim(a, i_triangle, axis=-1, keepdims=False)
    r_triangle = quantitize(r_triangle, quant)
    g_triangle = quantitize(g_triangle, quant)
    b_triangle = quantitize(b_triangle, quant)
    a_triangle = quantitize(a_triangle, quant)

    color_plane = jnp.expand_dims(
        jnp.stack([r_triangle, g_triangle, b_triangle], axis=-1), axis=[1, 2]
    )
    a_mask_triangle = jnp.expand_dims(a_triangle, axis=[1, 2, 3])
    next_canvas = (
        (canvas * (1.0 - region_mask_triangle))
        + (canvas * region_mask_triangle * (1.0 - a_mask_triangle))
        + (color_plane * region_mask_triangle * a_mask_triangle)
    )
    next_canvas = quantitize(next_canvas, quant)
    next_canvas = jnp.clip(a=next_canvas, a_min=0.0, a_max=1.0)
    canvas = next_canvas

    packed_args = canvas, xyv, xy0, xy1, xy2, r, g, b, a, quant
    return packed_args


@partial(jax.jit, static_argnums=0)
def render_rgb(static_params, key, params):
    h, w, alpha_scale, n_triangle, background, quant = static_params

    n_batch = params.shape[0]
    n_feature_per_triangle = 10

    params = params.reshape((n_batch, n_triangle, n_feature_per_triangle))

    params = (params - params.min(axis=1, keepdims=True)) / (
        params.max(axis=1, keepdims=True) - params.min(axis=1, keepdims=True)
    )

    x0, y0, x1, y1, x2, y2, r, g, b, a = [
        params[:, :, i_feature] for i_feature in range(n_feature_per_triangle)
    ]

    x0, x1, x2 = x0 * (h - 1), x1 * (h - 1), x2 * (h - 1)
    y0, y1, y2 = y0 * (w - 1), y1 * (w - 1), y2 * (w - 1)
    xy0 = jnp.stack([x0, y0], axis=-1).astype(jnp.int32)
    xy1 = jnp.stack([x1, y1], axis=-1).astype(jnp.int32)
    xy2 = jnp.stack([x2, y2], axis=-1).astype(jnp.int32)
    #  r, g, b, a = r, g, b, a * alpha_scale
    a = a * alpha_scale

    xv, yv = jnp.meshgrid(jnp.arange(0, h), jnp.arange(0, w), indexing="ij")
    xyv = jnp.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    xyv = jnp.repeat(jnp.expand_dims(xyv, axis=0), repeats=n_batch, axis=0)

    if background == "noise":
        key, subkey = jax.random.split(key)
        canvas = jax.random.uniform(
            key=subkey, shape=(n_batch, h, w, 3), dtype=jnp.float32
        )
    elif background == "black":
        canvas = jnp.zeros(shape=(n_batch, h, w, 3), dtype=jnp.float32)
    elif background == "white":
        canvas = jnp.ones(shape=(n_batch, h, w, 3), dtype=jnp.float32)

    # pylint: disable-next = used-before-assignment
    packed_args = canvas, xyv, xy0, xy1, xy2, r, g, b, a, quant
    packed_args = lax.fori_loop(0, n_triangle, put_triangle_fn, packed_args)
    canvas, *_ = packed_args

    return canvas


def put_triangle_grayscale_fn(i_triangle, packed_args):
    canvas, xyv, xy0, xy1, xy2, gray, a, quant = packed_args
    n_batch, h, w = canvas.shape

    xy0_triangle = lax.dynamic_slice_in_dim(xy0, i_triangle, 1, axis=1)
    xy1_triangle = lax.dynamic_slice_in_dim(xy1, i_triangle, 1, axis=1)
    xy2_triangle = lax.dynamic_slice_in_dim(xy2, i_triangle, 1, axis=1)

    cross_0 = jnp.cross((xyv - xy0_triangle), (xy1_triangle - xy0_triangle))
    cross_1 = jnp.cross((xyv - xy1_triangle), (xy2_triangle - xy1_triangle))
    cross_2 = jnp.cross((xyv - xy2_triangle), (xy0_triangle - xy2_triangle))

    in_triangle_p = jnp.logical_and(
        jnp.logical_and(cross_0 >= 0, cross_1 >= 0), cross_2 >= 0
    )
    in_triangle_n = jnp.logical_and(
        jnp.logical_and(cross_0 <= 0, cross_1 <= 0), cross_2 <= 0
    )
    in_triangle = jnp.logical_or(in_triangle_p, in_triangle_n).reshape((n_batch, h, w))

    region_mask_triangle = in_triangle

    gray_triangle = lax.dynamic_index_in_dim(gray, i_triangle, axis=-1, keepdims=False)
    a_triangle = lax.dynamic_index_in_dim(a, i_triangle, axis=-1, keepdims=False)
    gray_triangle = quantitize(gray_triangle, quant)
    a_triangle = quantitize(a_triangle, quant)

    color_plane = jnp.expand_dims(jnp.stack([gray_triangle], axis=-1), axis=1)
    a_mask_triangle = jnp.expand_dims(a_triangle, axis=[1, 2])
    next_canvas = (
        (canvas * (1.0 - region_mask_triangle))
        + (canvas * region_mask_triangle * (1.0 - a_mask_triangle))
        + (color_plane * region_mask_triangle * a_mask_triangle)
    )
    next_canvas = quantitize(next_canvas, quant)
    next_canvas = jnp.clip(a=next_canvas, a_min=0.0, a_max=1.0)
    canvas = next_canvas

    packed_args = canvas, xyv, xy0, xy1, xy2, gray, a, quant
    return packed_args


@partial(jax.jit, static_argnums=0)
def render_grayscale(static_params, key, params):
    h, w, alpha_scale, n_triangle, background, quant = static_params

    n_batch = params.shape[0]
    n_feature_per_triangle = 8

    params = params.reshape((n_batch, n_triangle, n_feature_per_triangle))

    # Normalize each feature.
    params = (params - params.min(axis=1, keepdims=True)) / (
        params.max(axis=1, keepdims=True) - params.min(axis=1, keepdims=True)
    )

    x0, y0, x1, y1, x2, y2, gray, a = [
        params[:, :, i_feature] for i_feature in range(n_feature_per_triangle)
    ]

    x0, x1, x2 = x0 * (h - 1), x1 * (h - 1), x2 * (h - 1)
    y0, y1, y2 = y0 * (w - 1), y1 * (w - 1), y2 * (w - 1)
    xy0 = jnp.stack([x0, y0], axis=-1).astype(jnp.int32)
    xy1 = jnp.stack([x1, y1], axis=-1).astype(jnp.int32)
    xy2 = jnp.stack([x2, y2], axis=-1).astype(jnp.int32)
    #  gray, a = gray, a * alpha_scale
    a = a * alpha_scale

    xv, yv = jnp.meshgrid(jnp.arange(0, h), jnp.arange(0, w), indexing="ij")
    xyv = jnp.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)
    xyv = jnp.repeat(jnp.expand_dims(xyv, axis=0), repeats=n_batch, axis=0)

    if background == "noise":
        key, subkey = jax.random.split(key)
        canvas = jax.random.uniform(
            key=subkey, shape=(n_batch, h, w), dtype=jnp.float32
        )
    elif background == "black":
        canvas = jnp.zeros(shape=(n_batch, h, w), dtype=jnp.float32)
    elif background == "white":
        canvas = jnp.ones(shape=(n_batch, h, w), dtype=jnp.float32)

    # pylint: disable-next = used-before-assignment
    packed_args = canvas, xyv, xy0, xy1, xy2, gray, a, quant
    packed_args = lax.fori_loop(0, n_triangle, put_triangle_grayscale_fn, packed_args)
    canvas, *_ = packed_args

    return canvas


class Triangles(EvaluateTorchMixin, DomainBase):
    """Arm repertoire domain."""

    def __init__(  # pylint: disable = super-init-not-called
        self,
        config: DictConfig,
        seed: int,
        device: torch.device,
    ):
        """Initializes from a single config."""
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.device = device

        if self.config.mode == "rgba":
            self.render = render_rgb
        elif self.config.mode == "grayscale":
            self.render = render_grayscale
        else:
            raise ValueError

        self.jax_key = jax.random.PRNGKey(self.rng.integers(10000))

        # Set up different objectives.
        mnist_root_dir = Path(__file__).parent / "mnist" / "mnist_weights"
        if self.config.objective == "ones":
            pass  # Don't do anything.
        elif self.config.objective == "mnist_mlp_discriminator":
            self.discriminator = MLPDiscriminator().to(device)
            self.discriminator.load_state_dict(
                torch.load(
                    str(mnist_root_dir / "mnist_mlp_discriminator.pth"),
                    map_location=device,
                )
            )
            self.discriminator.eval()
        elif self.config.objective == "mnist_dc_discriminator":
            self.discriminator = DCDiscriminator().to(device)
            self.discriminator.load_state_dict(
                torch.load(
                    str(mnist_root_dir / "netD_epoch_99.pth"),
                    map_location=self.device,
                )
            )
            self.discriminator.eval()
        elif self.config.objective == "centroid_mse":
            self.centroid_kd_tree = KDTree(np.load(self.config.centroids))
        elif self.config.objective == "centroid_cosine_similarity":
            self.centroid_nn = NearestNeighbors(
                n_neighbors=1, metric="cosine", n_jobs=16
            )
            self.centroid_nn.fit(np.load(self.config.centroids))
        else:
            raise ValueError(f"Unknown objective {self.config.objective}")

        if self.config.clip_embed:
            import clip

            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

            # CLIP expects a PIL image; since we only have numpy arrays and
            # they're going to be 224x224, we can just make our own
            # preprocessing here.
            self.clip_preprocess = transforms.Compose(
                [
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

    def initial_solution(self):
        return self.rng.random(self.config.solution_dim)

    def evaluate(self, solutions, grad=False):
        """Renders the triangles as images."""
        if grad:
            raise NotImplementedError

        self.jax_key, subkey = jax.random.split(self.jax_key)

        # Render images.
        static_params = (
            self.config.h,
            self.config.w,
            self.config.alpha_scale,
            self.config.n_triangle,
            self.config.background,
            self.config.quant,
        )
        jax_params = jnp.array(solutions)
        jax_params = jnp.repeat(jax_params, self.config.rollouts, axis=0)
        # pylint: disable-next = not-callable
        canvas = self.render(static_params, subkey, jax_params)

        # Compute measures.
        if self.config.clip_embed:
            # Measures are CLIP embeddings of the images.
            images = torch.tensor(
                np.asarray(canvas.transpose(0, 3, 2, 1)), device=self.device
            )
            preprocessed_images = self.clip_preprocess(images)
            measures = F.normalize(
                self.clip_model.encode_image(preprocessed_images), dim=-1
            )
            measures = measures.detach().cpu().numpy()
        else:
            # Measures are the (flattened) images.
            if self.config.mode == "rgba":
                # Rearrange to match PyTorch format.
                canvas = canvas.transpose(0, 3, 2, 1)
            measures = np.asarray(canvas).reshape(
                len(solutions) * self.config.rollouts, -1
            )

        # For debugging. We can also upscale during rendering, if needed.
        #  image = render_mnist_batch_pil(measures)
        #  image.save("mnist_grid.png")

        # Compute objectives.
        if self.config.objective == "ones":
            objectives = np.ones(len(solutions), dtype=np.float32)
        elif self.config.objective == "mnist_mlp_discriminator":
            if self.config.rollouts > 1:
                raise NotImplementedError(
                    "mnist_mlp_discriminator only works with 1 rollout."
                )
            with torch.no_grad():
                imgs = torch.tensor(
                    measures,
                    device=self.device,
                ).reshape(-1, 28, 28)
                # Normalize from [0, 1] to [-1, 1].
                unnormalized_imgs = (imgs - 0.5) * 2
                torch_objs = self.discriminator(unnormalized_imgs).squeeze()
            objectives = torch_objs.detach().cpu().numpy()
        elif self.config.objective == "mnist_dc_discriminator":
            if self.config.rollouts > 1:
                raise NotImplementedError(
                    "mnist_mlp_discriminator only works with 1 rollout."
                )
            with torch.no_grad():
                imgs = torch.tensor(
                    measures,
                    device=self.device,
                ).reshape(-1, 1, 28, 28)
                # Normalize from [0, 1] to [-1, 1].
                unnormalized_imgs = (imgs - 0.5) * 2
                torch_objs = self.discriminator(unnormalized_imgs).squeeze()
            objectives = torch_objs.detach().cpu().numpy()
        elif self.config.objective == "centroid_mse":
            if self.config.rollouts > 1:
                raise NotImplementedError("centroid_mse only works with 1 rollout.")

            l2_dist, _ = self.centroid_kd_tree.query(measures, workers=16)

            # Convert L2 distance to MSE.
            mse = np.square(l2_dist) / self.config.measure_dim

            # Normalize to [0, 1].
            objectives = 1.0 - mse
        elif self.config.objective == "centroid_cosine_similarity":
            if self.config.rollouts > 1:
                raise NotImplementedError(
                    "centroid_cosine_similarity only works with 1 rollout."
                )

            cosine_distances, _ = self.centroid_nn.kneighbors(
                measures, n_neighbors=1, return_distance=True
            )

            # Convert from cosine _distance_ to cosine _similarity_.
            cosine_similarities = 1.0 - cosine_distances

            # (n, 1) -> (n,)
            cosine_similarities = cosine_similarities.squeeze(-1)

            # Normalize to [0, 1].
            objectives = (cosine_similarities + 1.0) / 2.0
        else:
            raise ValueError(f"Unknown objective {self.config.objective}")

        if self.config.clip_embed:
            return objectives, measures, {"image": np.asarray(canvas)}
        else:
            return objectives, measures, {}


## VISUALIZATION FUNCTIONS ##
# These were mostly generated by ChatGPT.


def render_mnist_batch_pil(images, cols=10, padding=2, bg_color=255):
    """Renders a batch of MNIST images into a single image using PIL.

    Parameters:
    - images (np.ndarray): Array of shape (batch_size, 28, 28)
    - cols (int): Number of columns in the grid
    - padding (int): Pixels of padding between images
    - bg_color (int): Background color (0-255, white=255)

    Returns:
    - PIL.Image.Image: Composite image
    """
    if len(images.shape) != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Input must be of shape (batch_size, 28, 28)")

    images = (images * 255 if images.max() <= 1.0 else images).astype(np.uint8)
    batch_size = images.shape[0]
    rows = (batch_size + cols - 1) // cols

    grid_width = cols * 28 + (cols - 1) * padding
    grid_height = rows * 28 + (rows - 1) * padding

    grid_img = Image.new("L", (grid_width, grid_height), color=bg_color)

    for idx, img_array in enumerate(images):
        row, col = divmod(idx, cols)
        top = row * (28 + padding)
        left = col * (28 + padding)
        img = Image.fromarray(img_array)
        grid_img.paste(img, (left, top))
    return grid_img


def render_two_mnist_batches_side_by_side(
    batch1, batch2, cols=10, padding=2, bg_color=255
):
    """Renders two batches of MNIST images side by side per row using PIL.

    Parameters:
    - batch1 (np.ndarray): First batch of shape (batch_size, 28, 28)
    - batch2 (np.ndarray): Second batch of shape (batch_size, 28, 28)
    - cols (int): Number of pairs per row
    - padding (int): Pixels of padding between images
    - bg_color (int): Background color (0-255, white=255)

    Returns:
    - PIL.Image.Image: Composite image
    """
    if batch1.shape != batch2.shape or batch1.shape[1:] != (28, 28):
        raise ValueError("Both batches must be of shape (batch_size, 28, 28)")

    # Normalize if needed
    if batch1.max() <= 1.0:
        batch1 = (batch1 * 255).astype(np.uint8)
    if batch2.max() <= 1.0:
        batch2 = (batch2 * 255).astype(np.uint8)

    batch_size = batch1.shape[0]
    rows = (batch_size + cols - 1) // cols

    pair_width = 2 * 28 + padding  # Each row has two images side by side
    grid_width = cols * pair_width + (cols - 1) * padding
    grid_height = rows * 28 + (rows - 1) * padding

    grid_img = Image.new("L", (grid_width, grid_height), color=bg_color)

    for idx in range(batch_size):
        row, col = divmod(idx, cols)
        top = row * (28 + padding)
        left = col * (pair_width + padding)

        img1 = Image.fromarray(batch1[idx])
        img2 = Image.fromarray(batch2[idx])

        grid_img.paste(img1, (left, top))
        grid_img.paste(img2, (left + 28 + padding, top))

    return grid_img


def render_two_mnist_batches_side_by_side_upscale(
    batch1, batch2, cols=5, padding=2, bg_color=255, res=None
):
    """Renders two batches of MNIST images side by side per row using PIL.

    Upscales the second batch to match the first batch.

    Parameters:
    - batch1 (np.ndarray): First batch of shape (batch_size, res, res)
    - batch2 (np.ndarray): Second batch of shape (batch_size, res, res)
    - cols (int): Number of pairs per row
    - padding (int): Pixels of padding between images
    - bg_color (int): Background color (0-255, white=255)
    - res (int): Resolution at which to plot images.

    Returns:
    - PIL.Image.Image: Composite image
    """
    # Normalize if needed
    if batch1.max() <= 1.0:
        batch1 = (batch1 * 255).astype(np.uint8)
    if batch2.max() <= 1.0:
        batch2 = (batch2 * 255).astype(np.uint8)

    batch_size = batch1.shape[0]
    rows = (batch_size + cols - 1) // cols

    pair_width = 2 * res + padding  # Each row has two images side by side
    grid_width = cols * pair_width + (cols - 1) * padding
    grid_height = rows * res + (rows - 1) * padding

    grid_img = Image.new("L", (grid_width, grid_height), color=bg_color)

    for idx in range(batch_size):
        row, col = divmod(idx, cols)
        top = row * (res + padding)
        left = col * (pair_width + padding)

        img1 = Image.fromarray(batch1[idx])
        img2 = Image.fromarray(batch2[idx]).resize((res, res), resample=Image.NEAREST)

        grid_img.paste(img1, (left, top))
        grid_img.paste(img2, (left + res + padding, top))

    return grid_img


def render_color_image_grid(
    images_np, images_per_row=10, padding=2, bg_color=(255, 255, 255)
):
    """Create a PIL Image grid from a batch of float [H, W, 3] NumPy images in [0, 1] range.

    Args:
        images_np (list or np.ndarray): Batch of images, each [H, W, 3] in [0, 1] range.
        images_per_row (int): Number of images per row in the grid.
        padding (int): Space (in pixels) between images.
        bg_color (tuple): Background color (RGB) for the grid.

    Returns:
        PIL.Image: The composed grid image.
    """
    # Convert to list of PIL images
    images = []
    for img in images_np:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img).convert("RGB")
        images.append(pil_img)

    # Dimensions
    img_w, img_h = images[0].size
    n_images = len(images)
    n_cols = images_per_row
    n_rows = math.ceil(n_images / n_cols)

    # Create blank canvas
    grid_w = n_cols * img_w + (n_cols - 1) * padding
    grid_h = n_rows * img_h + (n_rows - 1) * padding
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    # Paste images into grid
    for idx, img in enumerate(images):
        row, col = divmod(idx, n_cols)
        x = col * (img_w + padding)
        y = row * (img_h + padding)
        grid_img.paste(img, (x, y))
    return grid_img


def render_two_color_image_batches_side_by_side(
    batch1_np,
    batch2_pil,
    images_per_row=10,
    padding=2,
    bg_color=(255, 255, 255),
    res=None,
):
    """Create a PIL Image grid from two batches of float [H, W, 3] NumPy images in [0, 1] range.

    Each pair (batch1[i], batch2[i]) is shown side-by-side.

    Args:
        batch1_np (list or np.ndarray): First batch of images, each [H, W, 3] in [0, 1].
        batch2_pil (list of PIL images): Second batch.
        images_per_row (int): Number of image pairs per row.
        padding (int): Space (in pixels) between images.
        bg_color (tuple): Background color (RGB) for the grid.
        res (int): Image resolution.

    Returns:
        PIL.Image: The composed grid image.
    """
    if len(batch1_np) != len(batch2_pil):
        raise ValueError("Both batches must have the same number of images.")

    images = []
    for img1, pil_img2 in zip(batch1_np, batch2_pil, strict=True):
        img1 = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
        pil_img1 = Image.fromarray(img1).convert("RGB")
        pil_img2 = pil_img2.resize((res, res))
        images.append((pil_img1, pil_img2))

    img_w, img_h = images[0][0].size
    pair_w = 2 * img_w + padding
    n_pairs = len(images)
    n_cols = images_per_row
    n_rows = math.ceil(n_pairs / n_cols)

    grid_w = n_cols * pair_w + (n_cols - 1) * padding
    grid_h = n_rows * img_h + (n_rows - 1) * padding
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    for idx, (img1, img2) in enumerate(images):
        row, col = divmod(idx, n_cols)
        x = col * (pair_w + padding)
        y = row * (img_h + padding)

        grid_img.paste(img1, (x, y))
        grid_img.paste(img2, (x + img_w + padding, y))

    return grid_img


def afhq_pil_image(solutions, indices, res):
    jax_key = jax.random.PRNGKey(42)
    static_params = (
        res,  # height
        res,  # width
        0.5,  # alpha_scale
        50,  # n_triangle
        "white",  # background
        0,  # quant
    )
    jax_params = jnp.array(solutions)
    images = render_rgb(static_params, jax_key, jax_params)

    with open("./src/cvt/centroids/afhq_clip_paths.txt", encoding="utf-8") as file:
        # Leave out final blank line.
        filenames = file.read().split("\n")[:-1]
        assert len(filenames) == 1000
    archive_images = [Image.open(filenames[idx]) for idx in indices]

    pil_image = render_two_color_image_batches_side_by_side(
        np.asarray(images),
        archive_images,
        res=res,
        padding=4,
    )

    return pil_image
