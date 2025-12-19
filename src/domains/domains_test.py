"""Tests for domain functions."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from src.domains.arm import Arm
from src.domains.lin_proj import FlatLinProj, Sphere


@pytest.mark.parametrize("domain", [Sphere, FlatLinProj, Arm])
@pytest.mark.parametrize("n_solutions", [1, 2, 5])
def test_evaluate_torch_matches_evaluate(domain, n_solutions):
    rng = np.random.default_rng(42)
    device = torch.device("cpu")
    domain_module = domain(
        OmegaConf.create(
            {
                # Minimal config.
                "solution_dim": 100,
                "measure_dim": 2,
            }
        ),
        42,
        device,
    )

    solutions = rng.standard_normal(size=(n_solutions, 100))
    objectives, measures, info = domain_module.evaluate(solutions, grad=True)

    torch_objectives, torch_measures, torch_info = domain_module.evaluate_torch(
        torch.tensor(solutions, device=device), grad=True
    )

    assert_allclose(objectives, torch_objectives)
    assert_allclose(measures, torch_measures)

    assert info.keys() == torch_info.keys()
    for k in info:
        assert_allclose(info[k], torch_info[k])
