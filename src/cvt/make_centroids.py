"""Creates and saves centroids for the CVTArchive.

Usage:
    uv run -m src.cvt.make_centroids \
        domain=flat_multi_100d \
        +centroid_cells=10000 \
        +centroid_file=src/cvt/centroids/flat_multi_100d.npy

    uv run -m src.cvt.make_centroids \
        domain=sphere_20_100d \
        +centroid_cells=10000 \
        +centroid_file=src/cvt/centroids/sphere_20_100d.npy

    uv run -m src.cvt.make_centroids \
        domain=sphere_50_100d \
        +centroid_cells=10000 \
        +centroid_file=src/cvt/centroids/sphere_50_100d.npy
"""

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from ribs.archives import k_means_centroids

from src.utils.hydra_utils import define_resolvers
from src.utils.logging import setup_logdir_from_hydra

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """Runs experiment."""
    define_resolvers()

    logdir = setup_logdir_from_hydra()
    log.info(f"Logging directory: {logdir.logdir}")

    centroids, _ = k_means_centroids(
        centroids=cfg.centroid_cells,
        ranges=list(
            zip(
                cfg.domain.config.measure_low,
                cfg.domain.config.measure_high,
                strict=True,
            )
        ),
        samples=100_000,
        dtype=np.float64,
        seed=42,
    )

    log.info(f"Centroid File: {cfg.centroid_file}")
    np.save(cfg.centroid_file, centroids)


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
