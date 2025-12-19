"""Visualizes results from running MNIST."""
import itertools

import numpy as np
import torch
from torchvision.utils import make_grid


def show_whole_archive(
    ax,
    archive,
    generator,
    device,
    ncols,
    nrows,
):
    """Plots a figure with images across the entire archive.

    Args:
        ncols: Number of columns of images (e.g. along the "Boldness" axis).
        nrows: Number of rows of images (e.g. along the "Lightness" axis).
    """

    # List of images.
    imgs = []

    # Convert archive to a df with solutions available.
    df = archive.data(fields=["objective", "measures", "solution"],
                      return_type="pandas")

    # Compute the min and max measures where we want to display images. Right
    # now we just use archive bounds; we can also just select the rectangle
    # that contains all the solutions (i.e., the min and max measures for which
    # we found solutions).
    measure_bounds = [
        (archive.lower_bounds[0], archive.upper_bounds[0]),
        (archive.lower_bounds[1], archive.upper_bounds[1]),
    ]
    delta_measure_0 = archive.interval_size[0] / ncols
    delta_measure_1 = archive.interval_size[1] / nrows

    for col, row in itertools.product(reversed(range(nrows)), range(ncols)):
        # Compute bounds of a box in measure space.
        measure_0_low = measure_bounds[0][0] + delta_measure_0 * row
        measure_0_high = measure_bounds[0][0] + delta_measure_0 * (row + 1)
        measure_1_low = measure_bounds[1][0] + delta_measure_1 * col
        measure_1_high = measure_bounds[1][0] + delta_measure_1 * (col + 1)

        # Query for a solution with measures within this box.
        query_string = (
            f"{measure_0_low} <= measures_0 & measures_0 <= {measure_0_high} & "
            f"{measure_1_low} <= measures_1 & measures_1 <= {measure_1_high}")
        df_box = df.query(query_string)

        if not df_box.empty:
            # Select the solution with highest objective in the box.
            max_obj_idx = df_box['objective'].argmax()
            sol = (
                df_box.loc[:,
                           "solution_0":f"solution_{archive.solution_dim - 1}"].
                iloc[max_obj_idx])

            # Convert the latent vector solution to an image.
            with torch.no_grad():
                img = generator(
                    torch.tensor(
                        np.asarray(sol).reshape(1, archive.solution_dim),
                        dtype=torch.float32,
                        device=device,
                    )).cpu()
            # Normalize images to [0,1].
            normalized = (img.squeeze()[None] + 1.0) / 2.0

            imgs.append(normalized)
        else:
            # Fill remaining tiles with white squares.
            imgs.append(torch.ones((1, 28, 28)))

    make_img_grid(ax, imgs, nrows, ncols, measure_bounds)


def make_img_grid(ax, imgs, nrows, ncols, measure_bounds):

    def create_archive_tick_labels(measure_range, num_ticks):
        delta = (measure_range[1] - measure_range[0]) / num_ticks
        ticklabels = [
            round(delta * p + measure_range[0], 3) for p in range(num_ticks + 1)
        ]
        return ticklabels

    img_grid = make_grid(imgs, nrow=ncols, padding=0)
    img_grid = np.transpose(img_grid.cpu().numpy(), (1, 2, 0))
    ax.imshow(img_grid)

    num_x_ticks = ncols
    x_ticklabels = create_archive_tick_labels(measure_bounds[0], num_x_ticks)
    x_tick_range = img_grid.shape[1]
    x_ticks = np.arange(0, x_tick_range + 1e-9, step=x_tick_range / num_x_ticks)
    ax.set_xticks(x_ticks, x_ticklabels)

    num_y_ticks = nrows
    y_ticklabels = create_archive_tick_labels(measure_bounds[1], num_y_ticks)
    y_ticklabels.reverse()
    y_tick_range = img_grid.shape[0]
    y_ticks = np.arange(0, y_tick_range + 1e-9, step=y_tick_range / num_y_ticks)
    ax.set_yticks(y_ticks, y_ticklabels)
