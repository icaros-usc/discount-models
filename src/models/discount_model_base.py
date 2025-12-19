"""Provides DiscountModelBase."""

import logging
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


class DiscountModelBase(ABC):
    """Base class for discount models.

    Each method should try to use settings from the config as much as possible,
    instead of relying on method parameters. This helps keep the API consistent.
    """

    def __init__(
        self,
        cfg: DictConfig,
        seed: int,
        device: torch.device,
    ) -> None:
        """Initializes from a single config."""
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = device

    @staticmethod
    def count_params(model: nn.Module) -> int:
        """Utility for counting parameters in a torch model."""
        return sum(p.numel() for p in model.parameters())

    def num_params(self) -> int | dict[str, int]:
        """Counts number of parameters in this model.

        Returns:
            Number of params, or dict mapping from names of components to number
            of params for each component.
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Set the model into eval mode (like in PyTorch).

        Default is to switch all nn.Module attrs to eval mode.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.eval()

    def train(self) -> None:
        """Set the model into train mode (like in PyTorch).

        Default is to switch all nn.Module attrs to train mode.
        """
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.train()

    def training_loop(self, measures: torch.Tensor, targets: torch.Tensor) -> Any:  # noqa: ANN401
        """Regresses the discount model to match the given targets at the given measures.

        Args:
            measures: (batch_size, measure_dim) array of measure values.
            targets: (batch_size,) array of target values for the discount function.

        Returns:
            Any data associated with training.
        """
        raise NotImplementedError

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Retrieves discount values from the model for the given inputs (i.e., measures).

        Note that this method does NOT put the model in eval mode or use
        no_grad.

        Args:
            inputs: Inputs to the model, typically of (batch_size, input_dim).

        Returns:
            A (len(inputs),) array of discount values.
        """
        raise NotImplementedError

    def chunked_inference(
        self,
        inputs: np.ndarray | torch.Tensor,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Passes in the given inputs to the model in chunks.

        This method also puts the model in eval mode and uses no_grad when
        running the inference.
        """
        if verbose:
            log.info("Chunked inference")

        if batch_size is None:
            batch_size = len(inputs)

        dataloader = DataLoader(
            dataset=TensorDataset(
                torch.tensor(inputs, dtype=torch.float32, device=self.device)
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        self.eval()
        discounts = []
        for (b_inputs,) in tqdm.tqdm(dataloader) if verbose else dataloader:
            with torch.no_grad():
                b_discounts = self.inference(b_inputs)
            discounts.append(b_discounts)
        self.train()

        # Concatenate all the chunks together.
        discounts = torch.cat(discounts, dim=0)

        # Turn (X, 1) into (X,).
        if discounts.ndim == 2:
            discounts = discounts.squeeze(dim=1)

        return discounts

    def save(self, directory: str | Path) -> None:
        """Saves the model in the given directory.

        Default is to save `self.model` in `directory / model.pth`.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        # pylint: disable-next = no-member
        torch.save(self.model.state_dict(), directory / "model.pth")

    def load(self, directory: str | Path) -> "DiscountModelBase":
        """Loads the model from the given directory.

        Default is to load `self.model` from `directory / model.pth`.
        """
        weights = torch.load(Path(directory) / "model.pth", map_location=self.device)
        # pylint: disable-next = no-member
        self.model.load_state_dict(weights)
        return self
