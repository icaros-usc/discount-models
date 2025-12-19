"""Latent space illumination (LSI) domain with StyleGAN2 and CLIP.

Adapted from:
https://docs.pyribs.org/en/latest/tutorials/tom_cruise_dqd.html

This file assumes it is being run from the root directory of the repo after
following the instructions for setting up StyleGAN 2.
"""
import pickle
import sys

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from omegaconf import DictConfig

from .domain_base import DomainBase, EvaluateTorchMixin

#  sys.path.append("./stylegan2-ada-pytorch")
sys.path.append("./stylegan3")


def norm1(prompt):
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()


class CLIP:
    """Manages a CLIP model."""

    def __init__(self, clip_model_name="ViT-B/32", device='cpu'):
        self.device = device
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        self.transform = transforms.CenterCrop(224)

    @torch.no_grad()
    def embed_text(self, prompt):
        return norm1(
            self.model.encode_text(clip.tokenize(prompt).to(
                self.device)).float())

    def embed_cutout(self, image):
        return norm1(self.model.encode_image(self.normalize(image)))

    def embed_image(self, image):
        n = image.shape[0]
        centered_img = self.transform(image)
        embeds = self.embed_cutout(centered_img)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds


class Generator:
    """Manages a StyleGAN2 model."""

    def __init__(self, device='cpu'):
        self.device = device
        model_filename = './stylegan2-ffhq-256x256.pkl'
        with open(model_filename, 'rb') as fp:
            self.model = pickle.load(fp)['G_ema'].to(device)
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
        zs = torch.randn([num_latents, self.model.mapping.z_dim],
                         device=self.device)
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
    if len(
            targets
    ) == 1:  # Keeps consistent results vs previous method for single objective guidance
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    loss = torch.stack(distances, dim=-1).sum(dim=-1)
    return loss


def transform_obj(objs):
    """Remaps the CLIP objective so that it is maximizing the range [0, 1]
    (formerly [0, 100])."""
    return (10.0 - objs * 5.0) * 0.1  # 10.0


class Classifier:

    def __init__(self, gen_model, class_model, prompt):
        self.device = gen_model.device
        self.gen_model = gen_model
        self.class_model = class_model

        self.init_objective(f'A photo of the face of {prompt}.')

        # Prompts for the measures. Modify these to search for different
        # ranges of images.
        self.measures = []
        self.add_measure(f'A photo of {prompt} as a small child.',
                         f'A photo of {prompt} as an elderly person.')
        self.add_measure(f'A photo of {prompt} with long hair.',
                         f'A photo of {prompt} with short hair.')

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
            G = self.gen_model.model
            w_stds = self.gen_model.w_stds
            for _ in range(num_batches):
                q = (G.mapping(torch.randn([batch_size, G.mapping.z_dim],
                                           device=self.device),
                               None,
                               truncation_psi=0.7) - G.mapping.w_avg) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = self.class_model.embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(embeds, self.obj_targets,
                                         spherical_dist_loss).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)

            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0)

        return q.flatten()

    def generate_image(self, latent_code):
        ws, _ = self.transform_to_w([latent_code])
        images = self.gen_model.model.synthesis(ws, noise_mode='const')
        return images

    def transform_to_w(self, latent_codes):
        qs = []
        ws = []
        for cur_code in latent_codes:
            q = torch.tensor(
                cur_code.reshape(self.gen_model.latent_shape),
                device=self.device,
                requires_grad=True,
            )
            qs.append(q)
            w = q * self.gen_model.w_stds + self.gen_model.model.mapping.w_avg
            ws.append(w)

        ws = torch.stack(ws, dim=0)
        return ws, qs

    def compute_objective_loss(self, embeds, qs, dim=None):
        # The mean averages the loss over the input prompts in `obj_targets`.
        loss = prompts_dist_loss(embeds, self.obj_targets,
                                 spherical_dist_loss).mean(0)

        diff = torch.max(torch.norm(qs, dim=dim), self.gen_model.q_norm)
        reg_loss = (diff - self.gen_model.q_norm).pow(2)
        loss = loss + 0.2 * reg_loss

        return loss

    def compute_objective(self, sols):
        ws, qs = self.transform_to_w(sols)

        images = self.gen_model.model.synthesis(ws, noise_mode='const')
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

        images = self.gen_model.model.synthesis(ws, noise_mode='const')
        embeds = self.class_model.embed_image(images.add(1).div(2))

        measure_targets = self.measures[index]
        pos_loss = prompts_dist_loss(embeds, measure_targets[0],
                                     cos_sim_loss).mean(0)
        neg_loss = prompts_dist_loss(embeds, measure_targets[1],
                                     cos_sim_loss).mean(0)
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

            images = self.gen_model.model.synthesis(ws, noise_mode='const')
            embeds = self.class_model.embed_image(images.add(1).div(2))

            loss = self.compute_objective_loss(embeds, qs, dim=(1, 2))
            objective_batch = loss.cpu().detach().numpy()

            measures_batch = []
            for i in range(len(self.measures)):
                measure_targets = self.measures[i]
                pos_loss = prompts_dist_loss(
                    embeds,
                    measure_targets[0],
                    cos_sim_loss,
                ).mean(0)
                neg_loss = prompts_dist_loss(embeds, measure_targets[1],
                                             cos_sim_loss).mean(0)
                loss = pos_loss - neg_loss
                value = loss.cpu().detach().numpy()
                measures_batch.append(value)

        return (transform_obj(objective_batch), np.stack(measures_batch,
                                                         axis=0).T)


class LSIClip(EvaluateTorchMixin, DomainBase):
    """MNIST domain with boldness and lightness as the measures."""

    def __init__(self, config: DictConfig, seed: int, device: torch.device):
        super().__init__(config, seed, device)

        self.clip_model = CLIP(device=device)
        self.gen_model = Generator(device=device)
        self.classifier = Classifier(self.gen_model,
                                     self.clip_model,
                                     prompt=self.config.prompt)

        self.initial_sol = self.classifier.find_good_start_latent().cpu(
        ).detach().numpy()

    def initial_solution(self):
        return self.initial_sol

    def evaluate(self, solutions, grad=False):
        """Accepts latent vectors as input."""
        if grad:
            if len(solutions) > 1:
                raise NotImplementedError(
                    "Can only compute gradient for one solution")
            objective, jacobian_obj = self.classifier.compute_objective(
                solutions)
            measures, jacobian_meas = self.classifier.compute_measures(
                solutions)
            return (
                objective,
                # Seems to come out with shape [measure_dim, 1], so reshape to
                # [1, measure_dim].
                measures.T,
                {
                    "objective_grads": jacobian_obj[None],
                    "measure_grads": jacobian_meas[None],
                },
            )
        else:
            objectives, measures = self.classifier.compute_all_no_grad(
                solutions)
            return objectives, measures, {}
