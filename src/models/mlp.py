"""Provides MLP models."""

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.discount_model_base import DiscountModelBase

log = logging.getLogger(__name__)


class MLP(nn.Module):
    """MLP archive.

    Feedforward network with identical activations on every layer. There is no
    activation on the last layer.

    Some methods return self so that one can do archive = MLPArchive().method()

    Args:
        layer_specs: List of tuples of (in_shape, out_shape, bias (optional)) for linear
            layers.
        activation: Activation layer class, e.g. nn.Tanh
        normalize: Whether to normalize the inputs. Pass "zero_one" to normalize to [0,
            1] or "negative_one_one" to normalize to [-1, 1]. Or pass False to indicate
            no normalization.
        norm_low: If normalize is True, this is the lower bound of the inputs for
            normalizing.
        norm_high: If normalize is True, this is the upper bound of the inputs for
            normalizing.
    """

    def __init__(
        self,
        layer_specs,
        activation,
        normalize: str | bool = False,
        norm_low: list[int] | None = None,
        norm_high: list[int] | None = None,
    ) -> None:
        super().__init__()

        layers = []
        for i, shape in enumerate(layer_specs):
            layers.append(
                nn.Linear(
                    shape[0], shape[1], bias=shape[2] if len(shape) == 3 else True
                )
            )
            if i != len(layer_specs) - 1:
                layers.append(activation())

        self.model = nn.Sequential(*layers)

        self.normalize = normalize
        self.norm_low = norm_low
        self.norm_high = norm_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to [-1, 1].
        if self.normalize:
            self.norm_low = torch.as_tensor(
                self.norm_low, device=x.device
            ).requires_grad_(False)
            self.norm_high = torch.as_tensor(
                self.norm_high, device=x.device
            ).requires_grad_(False)

            if self.normalize == "negative_one_one":
                x = 2 * (x - self.norm_low) / (self.norm_high - self.norm_low) - 1
            elif self.normalize == "zero_one":
                x = (x - self.norm_low) / (self.norm_high - self.norm_low)
            else:
                raise ValueError("Unknown normalization method.")

        return self.model(x)

    def initialize(self, func, bias_func=nn.init.zeros_):
        """Initializes weights for Linear layers with func.

        Both funcs usually comes from nn.init -- pass func="pytorch_default" to
        use the default pytorch initialization everywhere.
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if func == "pytorch_default":
                    m.reset_parameters()
                else:
                    func(m.weight)
                    if m.bias is not None:
                        bias_func(m.bias)

        self.apply(init_weights)

        return self

    def serialize(self):
        """Returns 1D array with all parameters in the model."""
        return nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        nn.utils.vector_to_parameters(torch.from_numpy(array), self.parameters())
        return self

    def gradient(self):
        """Returns 1D array with gradient of all parameters in the model."""
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )


class MLPDiscountModel(DiscountModelBase):
    def __init__(self, cfg: DictConfig, seed: int, device: torch.device) -> None:
        super().__init__(cfg, seed, device)

        self.model = hydra.utils.instantiate(self.cfg.model.args)
        self.model.initialize(**self.cfg.init.weights)
        self.model.to(device)

        self.optimizer = hydra.utils.instantiate(
            self.cfg.optimizer.args,
            params=self.model.parameters(),
        )

    def num_params(self) -> int:
        return self.count_params(self.model)

    def training_loop(
        self, measures: torch.Tensor, targets: torch.Tensor
    ) -> list[float]:
        """Regresses the discount model to match the given targets at the given measures.

        Returns the losses from the training epochs.
        """
        dataset = TensorDataset(measures, targets)
        dataloader = DataLoader(
            dataset,
            self.cfg.train.batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss(reduction="mean")

        all_epoch_loss = []

        for _ in range(1, self.cfg.train.epochs + 1):
            epoch_loss = 0.0

            for b_measures, b_targets in dataloader:
                cur = self.model(b_measures).squeeze(dim=1)

                self.optimizer.zero_grad()
                loss = criterion(cur, b_targets)
                loss.backward()
                self.optimizer.step()

                # Multiply so that we track the total loss even if batch size
                # varies.
                epoch_loss += loss.item() * len(b_measures)

            # Divide by total elements in dataset.
            epoch_loss /= len(dataloader.dataset)
            all_epoch_loss.append(epoch_loss)

            if epoch_loss <= self.cfg.train.cutoff_loss:
                break

        return all_epoch_loss

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
