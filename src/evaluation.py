"""Tools for evaluation."""

import logging

import hydra
import numpy as np
from matplotlib.axes import Axes
from omegaconf import DictConfig
from ribs.archives import ArchiveBase, CVTArchive, GridArchive
from ribs.visualize import grid_archive_heatmap

from src.models.discount_model_base import DiscountModelBase

log = logging.getLogger(__name__)


def compute_centers(archive: GridArchive | CVTArchive) -> np.ndarray:
    """Computes the centers of the archive's cells.

    Returns an array of size (archive.cells, measure_dim) containing all the centers of
    the cells in the archive.
    """
    if isinstance(archive, GridArchive):
        centers = [(b[:-1] + b[1:]) / 2.0 for b in archive.boundaries]
        measure_grid = np.meshgrid(*centers)
        measure_coords = np.stack([x.ravel() for x in measure_grid], axis=1)
        return measure_coords
    elif isinstance(archive, CVTArchive):
        return archive.centroids
    else:
        raise NotImplementedError


def make_discount_archive(
    discount_model: DiscountModelBase, cfg: DictConfig
) -> ArchiveBase:
    """Creates an archive that stores the value of the discount model at each cell.

    Args:
        discount_model: The model to plot.
        cfg: Base discount config.

    Returns:
        An archive filled with discount values.
    """
    # Note that this archive has the exact same resolution as the result archive. In the
    # future, it may be useful to plot the discount function at a higher resolution by
    # using an archive that has more cells than the result archive.
    discount_archive = hydra.utils.instantiate(
        cfg.algo.result_archive.args, solution_dim=0
    )

    measure_coords = compute_centers(discount_archive)
    discounts = discount_model.chunked_inference(measure_coords).detach().cpu().numpy()
    discount_archive.add(
        np.empty((len(measure_coords), 0)),
        discounts,
        measure_coords,
    )
    return discount_archive


def plot_discount_archive(
    discount_archive: GridArchive, ax: Axes, domain_cfg: dict
) -> None:
    """Heatmap showing discount values."""
    ax.set_title("Discount Model")
    grid_archive_heatmap(
        discount_archive,
        ax=ax,
        rasterized=True,
        # Note: This assumes the min threshold / discount is same as the min
        # objective.
        vmin=domain_cfg["obj_low"],
        vmax=domain_cfg["obj_high"],
    )
