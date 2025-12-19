"""Latent space illumination (LSI) domain with StyleGAN2 and CLIP.

Adapted from:
- https://colab.research.google.com/github/ouhenio/StyleGAN3-CLIP-notebook/blob/main/StyleGAN3%2BCLIP.ipynb
- https://github.com/icaros-usc/cma_mae/blob/main/experiments/lsi_clip_2/lsi.py
- https://docs.pyribs.org/en/latest/tutorials/tom_cruise_dqd.html

This file assumes it is being run from the root directory of the repo after
following the instructions for setting up StyleGAN 2.
"""

import logging
import pickle
import sys

import clip
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

from .domain_base import DomainBase, EvaluateTorchMixin

#  sys.path.append("./stylegan2-ada-pytorch")
sys.path.append("./stylegan3")

log = logging.getLogger(__name__)


class CLIP:
    """Manages a CLIP model."""

    def __init__(self, clip_model_name="ViT-B/32", device="cpu"):
        self.device = device
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.transform = transforms.CenterCrop(224)

    @torch.no_grad()
    def embed_text(self, prompt):
        return F.normalize(
            self.model.encode_text(clip.tokenize(prompt).to(self.device)).float(),
            dim=-1,
        )

    def embed_cutout(self, image):
        return F.normalize(self.model.encode_image(self.normalize(image)), dim=-1)

    def embed_image(self, image):
        n = image.shape[0]
        centered_img = self.transform(image)
        embeds = self.embed_cutout(centered_img)
        embeds = rearrange(embeds, "(cc n) c -> cc n c", n=n)
        return embeds


class Generator:
    """Manages a StyleGAN3 model."""

    def __init__(self, model_filename, device="cpu"):
        self.device = device
        with open(model_filename, "rb") as fp:
            self.model = pickle.load(fp)["G_ema"].to(device)
            self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.init_stats()
        self.latent_shape = (-1, 512)

    def init_stats(self):
        zs = torch.randn([10000, self.model.mapping.z_dim], device=self.device)
        ws = self.model.mapping(zs, None)
        self.w_stds = ws.std(0)
        qs = ((ws - self.model.mapping.w_avg) / self.w_stds).reshape(10000, -1)
        self.q_norm = torch.norm(qs, dim=1).mean() * 0.35

    def gen_random_ws(self, num_latents):
        zs = torch.randn([num_latents, self.model.mapping.z_dim], device=self.device)
        ws = self.model.mapping(zs, None)
        return ws


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def cos_sim_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().mul(2)


def prompts_dist_loss(x, targets, loss):
    """Computes the given losses over multiple target prompts."""
    if len(targets) == 1:
        # Keeps consistent results vs previous method for single objective guidance
        return loss(x, targets[0])

    # Each loss is shape (1, len(x)).
    distances = [loss(x, target) for target in targets]

    # Shape: (1, len(x), n_targets) after stack.
    # Shape: (1, len(x)) after sum.
    loss = torch.stack(distances, dim=-1).sum(dim=-1)

    return loss


def transform_obj(objs):
    """Remaps the CLIP objective.

    Objective is remapped so that it is maximizing the range [0, 1] (formerly [0, 100])
    -- recall that the loss is cosine similarity which is in the range [-1, 1].
    """
    return (10.0 - objs * 5.0) * 0.1  # 10.0


class Classifier:
    def __init__(
        self,
        gen_model,
        class_model,
        prompt,
        space="w_plus",  # GAN space in which we operate. Options are "w" or "w_plus".
    ):
        self.device = gen_model.device
        self.gen_model = gen_model
        self.class_model = class_model

        self.init_objective(prompt)
        #  self.init_objective(f'A photo of the face of {prompt}.')

        # Prompts for the measures. Modify these to search for different
        # ranges of images.
        self.measures = []
        #  self.add_measure(f'A photo of {prompt} as a small child.',
        #                   f'A photo of {prompt} as an elderly person.')
        #  self.add_measure(f'A photo of {prompt} with long hair.',
        #                   f'A photo of {prompt} with short hair.')

        self.space = space

    def init_objective(self, text_prompt):
        texts = [frase.strip() for frase in text_prompt.split("|") if frase]
        self.obj_targets = [self.class_model.embed_text(text) for text in texts]

    def add_measure(self, positive_text, negative_text):
        texts = [frase.strip() for frase in positive_text.split("|") if frase]
        negative_targets = [self.class_model.embed_text(text) for text in texts]

        texts = [frase.strip() for frase in negative_text.split("|") if frase]
        positive_targets = [self.class_model.embed_text(text) for text in texts]

        self.measures.append((negative_targets, positive_targets))

    def find_good_start_latent(self, batch_size=16, num_batches=32):
        with torch.inference_mode():
            qs = []
            losses = []
            G = self.gen_model.model  # pylint: disable=invalid-name
            w_stds = self.gen_model.w_stds
            for _ in range(num_batches):
                q = (
                    G.mapping(
                        torch.randn([batch_size, G.mapping.z_dim], device=self.device),
                        None,
                        truncation_psi=0.7,
                    )
                    - G.mapping.w_avg
                ) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = self.class_model.embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(
                    embeds, self.obj_targets, spherical_dist_loss
                ).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)

            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0)

        if self.space == "w":
            # q should be shape (1, n_layers, 512) -- usually (1, 18, 512), and
            # all the entries along the n_layers dim will be repeated.
            return q[0][0]
        elif self.space == "w_plus":
            return q.flatten()
        else:
            raise NotImplementedError

    def generate_image(self, latent_code):
        if latent_code.ndim == 1:
            latent_code = latent_code[None]
        ws, _ = self.transform_to_w(latent_code)
        images = self.gen_model.model.synthesis(ws, noise_mode="const")
        return images

    def transform_to_w(self, latent_codes: np.ndarray):
        qs = []
        ws = []
        # TODO: Speed up by batching? Unsure if doing so will mess with gradient
        # computation.
        for cur_code in latent_codes:
            if self.space == "w":
                # Repeat along the num_layers dim so that the code can be passed
                # into the synthesis network.
                q = torch.tensor(
                    # Note: cur_code is still a numpy array, and np.repeat
                    # operates differently from torch.repeat.
                    #
                    # Note: Hard-coded for StyleGAN3 256x256, which has 16
                    # layers. May be nice to use
                    # self.gen_model.model.synthesis.num_layers in the future,
                    # but that doesn't necessarily match, e.g., it can have 14
                    # while there are actually 16 expected.
                    cur_code[None].repeat(16, axis=0),
                    device=self.device,
                    requires_grad=True,
                )
            elif self.space == "w_plus":
                # Reshape into (num_layers, 512).
                q = torch.tensor(
                    cur_code.reshape(self.gen_model.latent_shape),
                    device=self.device,
                    requires_grad=True,
                )
            else:
                raise NotImplementedError

            qs.append(q)

            w = q * self.gen_model.w_stds + self.gen_model.model.mapping.w_avg
            ws.append(w)

        ws = torch.stack(ws, dim=0)
        return ws, qs

    def compute_objective_loss(self, embeds, qs, dim=None):
        # The mean averages the loss over the input prompts in `obj_targets`.
        loss = prompts_dist_loss(embeds, self.obj_targets, spherical_dist_loss).mean(0)

        diff = torch.max(torch.norm(qs, dim=dim), self.gen_model.q_norm)
        reg_loss = (diff - self.gen_model.q_norm).pow(2)
        loss = loss + 0.2 * reg_loss

        return loss

    def compute_objective(self, sols):
        ws, qs = self.transform_to_w(sols)

        images = self.gen_model.model.synthesis(ws, noise_mode="const")
        embeds = self.class_model.embed_image(images.add(1).div(2))

        loss = self.compute_objective_loss(embeds, qs[0])
        loss.backward()

        value = loss.cpu().detach().numpy()
        jacobian = -qs[0].grad.cpu().detach().numpy()
        return (
            transform_obj(value),
            jacobian.flatten(),
        )

    def compute_measure(self, index, sols):
        """Computes a *single* measure and its gradient."""
        ws, qs = self.transform_to_w(sols)

        images = self.gen_model.model.synthesis(ws, noise_mode="const")
        embeds = self.class_model.embed_image(images.add(1).div(2))

        measure_targets = self.measures[index]
        pos_loss = prompts_dist_loss(embeds, measure_targets[0], cos_sim_loss).mean(0)
        neg_loss = prompts_dist_loss(embeds, measure_targets[1], cos_sim_loss).mean(0)
        loss = pos_loss - neg_loss
        loss.backward()

        value = loss.cpu().detach().numpy()
        jacobian = qs[0].grad.cpu().detach().numpy()
        return (
            value,
            jacobian.flatten(),
        )

    def compute_measures(self, sols):
        """Computes *all* measures and their gradients."""
        values = []
        jacobian = []
        for i in range(len(self.measures)):
            value, jac = self.compute_measure(i, sols)
            values.append(value)
            jacobian.append(jac)

        return (
            np.stack(values, axis=0),
            np.stack(jacobian, axis=0),
        )

    def compute_all_no_grad(self, sols):
        """Computes the objective and measure without gradients."""
        with torch.inference_mode():
            ws, qs = self.transform_to_w(sols)
            qs = torch.stack(qs, dim=0)

            images = self.gen_model.model.synthesis(ws, noise_mode="const")
            embeds = self.class_model.embed_image(images.add(1).div(2))

            loss = self.compute_objective_loss(embeds, qs, dim=(1, 2))
            objective_batch = loss.cpu().detach().numpy()

            measures_batch = []
            for measure_targets in self.measures:
                pos_loss = prompts_dist_loss(
                    embeds,
                    measure_targets[0],
                    cos_sim_loss,
                ).mean(0)
                neg_loss = prompts_dist_loss(
                    embeds, measure_targets[1], cos_sim_loss
                ).mean(0)
                loss = pos_loss - neg_loss
                value = loss.cpu().detach().numpy()
                measures_batch.append(value)

        return (transform_obj(objective_batch), np.stack(measures_batch, axis=0).T)

    def compute_objective_no_grad(self, sols: np.ndarray):
        """Computes the objective without gradients."""
        with torch.inference_mode():
            ws, qs = self.transform_to_w(sols)
            qs = torch.stack(qs, dim=0)

            images = self.gen_model.model.synthesis(ws, noise_mode="const")
            embeds = self.class_model.embed_image(images.add(1).div(2))

            loss = self.compute_objective_loss(embeds, qs, dim=(1, 2))
            objective_batch = loss.detach().cpu().numpy()

        # embeds has shape [1, batch_size, 512], so remove first dim.
        return transform_obj(objective_batch), embeds.detach().cpu().numpy()[0]


class LSIFace(EvaluateTorchMixin, DomainBase):
    """LSI domain for faces."""

    def __init__(self, config: DictConfig, seed: int, device: torch.device):
        super().__init__(config, seed, device)

        self.clip_model = CLIP(self.config.clip_model_name, device=device)
        self.gen_model = Generator(self.config.stylegan_filename, device=device)
        self.classifier = Classifier(
            self.gen_model,
            self.clip_model,
            prompt=self.config.obj_prompt,
            space=self.config.space,
        )

        #  self.initial_sol = self.classifier.find_good_start_latent().detach().cpu().numpy()
        #  objs, _ = self.classifier.compute_objective_no_grad(
        #      self.initial_sol[None])
        #  log.info(f"Initial Solution Objective: {objs[0]}")

        # Set up centroid distance objective.
        if self.config.add_centroid_dist:
            self.centroid_nn = NearestNeighbors(
                n_neighbors=1, metric="cosine", n_jobs=16
            )
            self.centroid_nn.fit(np.load(self.config.centroids))

    def initial_solution(self):
        #  return self.initial_sol
        return self.classifier.find_good_start_latent().detach().cpu().numpy()

    def evaluate(self, solutions, grad=False):
        """Accepts latent vectors as input."""
        if grad:
            raise NotImplementedError
            #  if len(solutions) > 1:
            #      raise NotImplementedError(
            #          "Can only compute gradient for one solution")
            #  objective, jacobian_obj = self.classifier.compute_objective(
            #      solutions)
            #  measures, jacobian_meas = self.classifier.compute_measures(
            #      solutions)
            #  return (
            #      objective,
            #      # Seems to come out with shape [measure_dim, 1], so reshape to
            #      # [1, measure_dim].
            #      measures.T,
            #      {
            #          "objective_grads": jacobian_obj[None],
            #          "measure_grads": jacobian_meas[None],
            #      },
            #  )
        else:
            objectives, embeds = self.classifier.compute_objective_no_grad(solutions)

            if self.config.add_centroid_dist:
                cosine_distances, _ = self.centroid_nn.kneighbors(
                    embeds, n_neighbors=1, return_distance=True
                )

                # Convert from cosine _distance_ to cosine _similarity_.
                cosine_similarities = 1.0 - cosine_distances

                # (n, 1) -> (n,)
                cosine_similarities = cosine_similarities.squeeze(-1)

                # Normalize to [0, 1].
                centroid_objectives = (cosine_similarities + 1.0) / 2.0

                # Add the two objectives together, weighting them equally.
                objectives = (objectives + centroid_objectives) / 2.0

            return objectives, embeds, {}


def plot_image_pairs(
    torch_images,
    pil_images,
    pairs_per_row=5,
    separator_width=4,
    separator_color=(0, 0, 0),
):
    assert len(torch_images) == len(pil_images), (
        "Both lists must have the same number of images."
    )

    img_width, img_height = 256, 256
    num_pairs = len(torch_images)
    num_rows = (num_pairs + pairs_per_row - 1) // pairs_per_row

    # Total columns = 2 images per pair + (1 separator between each image and between each pair)
    total_cols = pairs_per_row * 2 + (pairs_per_row + 1)
    total_rows = num_rows + 1  # includes top and bottom row separator

    total_width = total_cols * separator_width + pairs_per_row * 2 * img_width
    total_height = total_rows * separator_width + num_rows * img_height

    # Create a blank canvas and draw separators
    grid_img = Image.new("RGB", (total_width, total_height), color=separator_color)

    for idx, (torch_img, pil_img) in enumerate(
        zip(torch_images, pil_images, strict=True)
    ):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row

        # Calculate x offsets with separators
        x_img1 = separator_width + pair_col * (2 * img_width + 3 * separator_width)
        x_img2 = x_img1 + img_width + separator_width
        y_offset = separator_width + row * (img_height + separator_width)

        img_tensor = torch_img.detach().cpu()
        img_tensor = img_tensor * 255  # Assume images are [0, 1].
        img_tensor = img_tensor.clamp(0, 255).byte()
        torch_pil = transforms.functional.to_pil_image(img_tensor)

        # Resize PIL image if needed
        pil_img_resized = pil_img.resize((img_width, img_height))

        # Paste both images
        grid_img.paste(torch_pil, (x_img1, y_offset))
        grid_img.paste(pil_img_resized, (x_img2, y_offset))

    return grid_img


def plot_image_pairs_no_outside_edges(
    torch_images,
    pil_images,
    pairs_per_row=5,
    separator_width=4,
    separator_color=(0, 0, 0),
):
    assert len(torch_images) == len(pil_images), (
        "Both lists must have the same number of images."
    )

    img_width, img_height = 256, 256
    num_pairs = len(torch_images)
    num_rows = (num_pairs + pairs_per_row - 1) // pairs_per_row

    # Total columns = 2 images per pair + (1 separator between image1 and image2 per pair) + (1 separator between pairs except last)
    num_separators_x = pairs_per_row * 1 + (
        pairs_per_row - 1
    )  # 1 between each pair and between images
    total_width = pairs_per_row * 2 * img_width + num_separators_x * separator_width

    # Total rows = num_rows of images + (num_rows - 1) separators between rows
    num_separators_y = num_rows - 1
    total_height = num_rows * img_height + num_separators_y * separator_width

    # Create a blank canvas
    grid_img = Image.new("RGB", (total_width, total_height), color=separator_color)

    for idx, (torch_img, pil_img) in enumerate(
        zip(torch_images, pil_images, strict=True)
    ):
        row = idx // pairs_per_row
        pair_col = idx % pairs_per_row

        # Compute horizontal offset
        num_prev_pairs = pair_col
        x_offset = (
            pair_col * 2 * img_width
            + num_prev_pairs * separator_width  # separator between pairs
            + pair_col * separator_width  # separator between images in pair
        )
        x_img1 = x_offset
        x_img2 = x_img1 + img_width + separator_width

        # Compute vertical offset
        y_offset = row * img_height + row * separator_width

        # Convert torch tensor to PIL image
        img_tensor = torch_img.detach().cpu()
        img_tensor = img_tensor * 255  # Assume images are [0, 1].
        img_tensor = img_tensor.clamp(0, 255).byte()
        torch_pil = transforms.functional.to_pil_image(img_tensor)

        pil_img_resized = pil_img.resize((img_width, img_height))

        # Paste both images
        grid_img.paste(torch_pil, (x_img1, y_offset))
        grid_img.paste(pil_img_resized, (x_img2, y_offset))

    return grid_img


def human_landscape_face(solutions, classifier, indices):
    with torch.no_grad():
        human_images = classifier.generate_image(solutions).add(1).div(2)

    with open("./src/cvt/centroids/lhq_clip_paths_10k.txt", encoding="utf-8") as file:
        # Exclude final blank line.
        filenames = file.read().split("\n")[:-1]
    landscape_images = [Image.open(filenames[idx]) for idx in indices]

    return plot_image_pairs(
        human_images, landscape_images, pairs_per_row=5, separator_width=5
    )
