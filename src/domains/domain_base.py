"""Provides DomainBase and some useful utilities."""
from abc import ABC
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig


class DomainBase(ABC):
    """Base class for domains.

    Each method should try to use settings from the config as much as possible,
    instead of relying on method parameters. This helps keep the API consistent.

    The evaluate methods return a tuple of (objective, measures, info) where the
    objective and measures are arrays and info is a dict.

    If `grad` is passed into the evaluate methods, then the gradients are
    returned via the `info` dictionary as `objective_grads` and `measure_grads`
    keys. Note that not all domains support gradient computation.
    """

    def __init__(
        self,
        config: DictConfig,
        seed: int,
        device: torch.device,
    ):
        """Initializes from a single config."""
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.device = device

    def initial_solution(self) -> np.ndarray:
        """Returns an initial solution."""
        raise NotImplementedError

    def evaluate(
        self,
        solutions: np.ndarray,
        grad=False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Evaluates the batch of solutions."""
        raise NotImplementedError

    def evaluate_torch(
        self,
        solutions: torch.Tensor,
        grad=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Evaluation done in PyTorch.

        Unfortunately, this evaluation is not meant to be differentiated
        through. Instead, it is meant to be identical to `evaluate` but differs
        in that it takes PyTorch tensors as input. For example, children classes
        may use `torch.no_grad()` in their implementation of this method.
        """
        raise NotImplementedError


class EvaluateTorchMixin:
    """Provides an evaluate_torch function that calls the existing evaluate."""

    def evaluate_torch(self, solutions, grad=False):
        numpy_sols = solutions.detach().cpu().numpy()

        all_objectives, all_measures, info = self.evaluate(numpy_sols,
                                                           grad=grad)

        return (
            torch.tensor(all_objectives,
                         dtype=solutions.dtype,
                         device=solutions.device),
            torch.tensor(all_measures,
                         dtype=solutions.dtype,
                         device=solutions.device),
            {
                k:
                    torch.tensor(v,
                                 dtype=solutions.dtype,
                                 device=solutions.device)
                for k, v in info.items()
            },
        )


class EvaluateNumpyMixin:
    """Provides an evaluate function that calls the existing evaluate_torch.

    Note that the provided method casts to float32 since that is the most common
    dtype used with PyTorch. This may not be valid for all cases though.
    """

    def evaluate(self, solutions, grad=False):
        tensor_sols = torch.tensor(
            solutions,
            dtype=torch.float32,
            device=self.device,
        )

        all_objectives, all_measures, info = self.evaluate_torch(tensor_sols,
                                                                 grad=grad)

        return (
            all_objectives.detach().cpu().numpy(),
            all_measures.detach().cpu().numpy(),
            {
                k: v.detach().cpu().numpy() for k, v in info.items()
            },
        )
