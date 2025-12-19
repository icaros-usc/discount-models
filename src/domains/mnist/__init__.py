"""MNIST domains.

For the definition of `networks`, see `get_networks`

Adapted from this tutorial:
https://docs.pyribs.org/en/stable/tutorials/lsi_mnist.html
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from ..domain_base import DomainBase, EvaluateNumpyMixin

## Utilities ##

MNIST_NZ = 128


def get_networks(
    model: str,
    root_dir: str = None,
    device=None,
):
    """Retrieves a pre-trained LeNet-5 classifier and generative model.

    Args:
        model: Type of generative model to retrieve.
            Options: ["mlpgan", "dcgan", "ae", "vae"]
        root_dir: Directory where weights are located. Defaults to the
            `mnist_weights` directory where this file itself is located.
        device: Defaults to CUDA if available and CPU otherwise. Pass in to
            override this behavior.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if root_dir is None:
        root_dir = Path(__file__).parent / "mnist_weights"
    else:
        root_dir = Path(root_dir)

    # Load the weights of each network from its file.
    if model == "mlpgan":
        from .mlpgan import MLPDiscriminator, MLPGenerator

        generator = MLPGenerator(nz=MNIST_NZ).to(device)
        generator.load_state_dict(
            torch.load(
                str(root_dir / "mnist_mlp_generator.pth"),
                map_location=device,
            ))
        generator.eval()

        discriminator = MLPDiscriminator().to(device)
        discriminator.load_state_dict(
            torch.load(
                str(root_dir / "mnist_mlp_discriminator.pth"),
                map_location=device,
            ))
        discriminator.eval()

        other_models = {"discriminator": discriminator}

    elif model == "dcgan":
        from .dcgan import DCDiscriminator, DCGenerator

        generator = DCGenerator().to(device)
        generator.load_state_dict(
            torch.load(
                str(root_dir / "netG_epoch_99.pth"),
                map_location=device,
            ))
        generator.eval()

        discriminator = DCDiscriminator().to(device)
        discriminator.load_state_dict(
            torch.load(
                str(root_dir / "netD_epoch_99.pth"),
                map_location=device,
            ))
        discriminator.eval()

        other_models = {"discriminator": discriminator}

    elif model == "ae":
        from .autoencoder import AEDecoder, AEEncoder

        # Technically a decoder rather than a generator, but it serves the
        # purpose of a generator nevertheless.
        generator = AEDecoder().to(device)
        generator.load_state_dict(
            torch.load(
                str(root_dir / "ae_decoder.pth"),
                map_location=device,
            ))
        generator.eval()

        encoder = AEEncoder().to(device)
        encoder.load_state_dict(
            torch.load(
                str(root_dir / "ae_encoder.pth"),
                map_location=device,
            ))
        encoder.eval()

        other_models = {"encoder": encoder}

    elif model == "vae":
        import yaml

        import src.diffusion2
        from src.diffusion2.src.models.vae import VAE, VAEDecoderWrapper

        main_dir = Path(src.diffusion2.__file__).parent
        with (main_dir / "config" / "mnist_class_cond.yaml").open("r") as file:
            config = yaml.safe_load(file)
            dataset_config = config['dataset_params']
            autoencoder_config = config['autoencoder_params']
            train_config = config['train_params']

        model = VAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_config).to(device)
        model.load_state_dict(
            torch.load(
                main_dir / train_config['task_name'] /
                train_config['vae_autoencoder_ckpt_name'],
                map_location=device,
            ))
        model.eval()

        generator = VAEDecoderWrapper(model)
        other_models = {"vae": model}

    else:
        raise ValueError(f"Unknown gan {model}")

    from .lenet5 import LeNet5
    c_state_dict = torch.load(
        str(root_dir / "mnist_classifier.pth"),
        map_location=device,
    )
    lenet5 = LeNet5().to(device)
    lenet5.load_state_dict(c_state_dict)
    lenet5.eval()

    return generator, lenet5, other_models


## Domain classes ##


class MnistBoldLight(EvaluateNumpyMixin, DomainBase):
    """MNIST domain with boldness and lightness as the measures."""

    def __init__(self, config: DictConfig, seed: int, device: torch.device):
        super().__init__(config, seed, device)
        (
            self.generator,
            self.lenet5,
            self.other_models,
        ) = get_networks(config.model, config.root_dir, device=device)

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    @staticmethod
    def boldness_lightness(imgs):
        """Computes the boldness and lightness of the images.

        Boldness is the number of white pixels; lightness is the mean value
        of the white pixels.

        We consider pixels with values larger than or equal to 0.5 to be
        "white".

        Args:
            imgs: (batch_size, channels, width, height) torch array of images.
        """
        boldness = torch.count_nonzero(imgs >= 0.5, axis=(1, 2, 3))

        # Add 1 to boldness to avoid division by zero.
        lightness = torch.sum(
            (imgs * (imgs >= 0.5)), axis=(1, 2, 3)) / (boldness + 1.0)

        return torch.stack((boldness, lightness), axis=1)

    def _evaluate_images_internal(self, generated_imgs, grad=False):
        # Normalize the images from [-1,1] to [0,1].
        normalized_imgs = (generated_imgs + 1.0) / 2.0

        # We optimize the score of the digit being 8. Other digits may also
        # be used.
        all_objectives = torch.exp(
            # pylint: disable-next = not-callable
            self.lenet5(normalized_imgs)[:, self.config.digit])

        # Each measures entry is [boldness, lightness].
        all_measures = self.boldness_lightness(normalized_imgs)

        return all_objectives, all_measures, {"mnist_img": normalized_imgs}

    def evaluate_images_torch(self, generated_imgs, grad=False):
        """Accepts unnormalized images as input.

        Note: Latent space regularization is not available in this method.
        """
        with torch.no_grad():
            return self._evaluate_images_internal(generated_imgs, grad=grad)

    def evaluate_torch(self, solutions, grad=False):
        """Accepts latent vectors as input."""
        with torch.no_grad():
            # Shape: len(sols) x 1 x 28 x 28
            # pylint: disable-next = not-callable
            generated_imgs = self.generator(solutions)

            all_objectives, all_measures, info = self._evaluate_images_internal(
                generated_imgs)

            # Add a loss based on the norm of the latent vector -- anything beyond
            # cutoff_scale * sqrt(n) is penalized (where n is 1D dimensionality of
            # solutions).
            if self.config.get("reg_loss"):
                # Normalize with respect to L2 norm -- the avg L2 norm of a
                # standard Gaussian is sqrt(n).
                avg_l2_norm = np.sqrt(np.prod(solutions.shape[1:]))
                cutoff = torch.full(
                    (len(solutions),),
                    self.config.cutoff_scale * avg_l2_norm,
                    dtype=torch.float32,
                    device=self.device,
                )
                # pylint: disable-next = not-callable
                reg_loss = (torch.maximum(torch.linalg.norm(solutions, axis=1),
                                          cutoff) - cutoff) / cutoff
                # Subtract regularizer loss.
                all_objectives -= reg_loss

            return all_objectives, all_measures, info


class MnistDigits(EvaluateNumpyMixin, DomainBase):
    """MNIST domain with digits as the measures.

    The objective is a regularization loss that forces the solutions to have
    smaller magnitude.

    The measures are a list of digits, specified via `self.config.digits`.
    """

    def __init__(self, config: DictConfig, seed: int, device: torch.device):
        super().__init__(config, seed, device)
        (
            self.generator,
            self.lenet5,
            self.other_models,
        ) = get_networks(config.model, config.root_dir, device=device)

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    # TODO: The objective currently depends on the latent vectors, so we cannot
    # make a model that only treats the images. Come up with a new objective and
    # then implement evaluate_images_torch for this domain.
    def evaluate_torch(self, solutions, grad=False):
        """Accepts latent vectors as input."""
        with torch.no_grad():
            # Shape: len(sols) x 1 x 28 x 28
            # pylint: disable-next = not-callable
            generated_imgs = self.generator(solutions)

            n_imgs = len(generated_imgs)

            # Normalize the images from [-1,1] to [0,1].
            normalized_imgs = (generated_imgs + 1.0) / 2.0

            # Base objective is 1.0.
            all_objectives = torch.ones((n_imgs,),
                                        dtype=torch.float32,
                                        device=self.device)

            # Normalize with respect to L2 norm -- the avg L2 norm of a
            # standard Gaussian is sqrt(n).
            avg_l2_norm = np.sqrt(self.generator.nz)
            cutoff = torch.full(
                (n_imgs,),
                self.config.cutoff_scale * avg_l2_norm,
                dtype=torch.float32,
                device=self.device,
            )
            # pylint: disable-next = not-callable
            reg_loss = (torch.maximum(torch.linalg.norm(solutions, axis=1),
                                      cutoff) - cutoff) / cutoff
            # Subtract regularizer loss.
            all_objectives -= reg_loss

            # Measures are classifier scores for the digits.
            all_measures = torch.exp(
                # pylint: disable-next = not-callable
                self.lenet5(normalized_imgs)[:, self.config.digits])
            return all_objectives, all_measures, {"mnist_img": normalized_imgs}
