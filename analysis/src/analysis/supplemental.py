"""Generates additional figures and videos.

Usage:
    python -m src.analysis.supplemental COMMAND
"""
import functools
import pickle as pkl
import matplotlib
import shutil
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from logdir import LogDir
from loguru import logger
from slugify import slugify
from src.analysis.figures import load_manifest
from src.mpl_styles.utils import mpl_style_file

import ribs.visualize
from ribs.archives import GridArchive


def visualize_env_archives(mode, root_dir, viz_data, env, cur_subfig):
    """Plots visualizations for one environment in viz_data.

    cur_subfig may be either a subfigure or a regular figure.

    Pass custom_gen to plot generations that are not the last generation. In
    this case, custom_gen must be a dict mapping from algorithm name to the
    generation for that algorithm.
    """
    ncols = len(viz_data[env]["Algorithms"])
    ax = cur_subfig.subplots(
        1,
        ncols,
        gridspec_kw={"wspace": {
            "heatmap": 0.05,
            "histogram": 0.3,
        }[mode]},
    )

    for col, (cur_ax, algo) in enumerate(zip(ax, viz_data[env]["Algorithms"])):
        cur_ax.set_title(algo)

        logdir = LogDir("Foobar",
                        custom_dir=root_dir /
                        viz_data[env]["Algorithms"][algo]["logdir"])
        try:
            data = np.load(logdir.file("archive.npz"))
        except FileNotFoundError:
            data = np.load(logdir.file("scheduler/archive.npz"))

        xrange, yrange = {
            "2D LP (Sphere)": [(-256, 256), (-256, 256)],
            "2D LP (Rastrigin)": [(-256, 256), (-256, 256)],
            "2D LP (Flat)": [(-256, 256), (-256, 256)],
            "Arm Repertoire": [(-100, 100), (-100, 100)],
        }[env]

        archive = GridArchive(
            solution_dim=0,
            dims=(100, 100),
            ranges=(xrange, yrange),
        )
        archive.add(np.empty((len(data["solution"]), 0)), data["objective"],
                    data["measures"])

        if mode == "heatmap":
            # See heatmap.py for these settings.
            ribs.visualize.grid_archive_heatmap(
                archive=archive,
                ax=cur_ax,
                #  df=data,
                # Purple.
                #  cmap=[[126 / 255, 87 / 255, 194 / 255]],
                # Gray.
                #  cmap=[[180 / 255, 180 / 255, 180 / 255]],
                vmin=0,
                vmax=1,
                rasterized=True,
                aspect="equal",
                #  cbar=None,
            )
            cur_ax.set_xticks(
                [archive.lower_bounds[0], archive.upper_bounds[0]])
            cur_ax.set_yticks(
                [archive.lower_bounds[1], archive.upper_bounds[1]])

            if col != 0:
                # No yticks beyond first plot.
                cur_ax.set_yticks([])

            if col == 0:
                if "ylabel" in viz_data[env]:
                    # Set y label on left plot.
                    #  cur_ax.set_ylabel(viz_data[env]["ylabel"])
                    pass

                # Set title on left plot.
                cur_ax.set_ylabel(env)
            if col == ncols // 2:
                if "xlabel" in viz_data[env]:
                    # Set x label on middle plot.
                    cur_ax.set_xlabel(viz_data[env]["xlabel"])

            if col == ncols - 1:
                # Add colorbar when on last plot in column.

                # Remove all current colorbars.
                for a in cur_subfig.axes:
                    if a.get_label() == "<colorbar>":
                        a.remove()

                # Retrieve the heatmap mesh.
                artist = None
                for child in cur_ax.get_children():
                    if isinstance(child, matplotlib.collections.QuadMesh):
                        artist = child

                # Add axes for the colorbar. Solutions for this are complicated:
                # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                ratio = cur_ax.get_position().width / 0.129638671875
                # The axis width is expressed as a fraction of the figure width.
                # We first took the width when there were 5 figures, which was
                # 0.129638671875, and now we express width as a ratio of axis
                # width to that width.
                cax = cur_subfig.add_axes([
                    cur_ax.get_position().x1 + 0.01 * ratio,
                    cur_ax.get_position().y0,
                    0.01 * ratio,
                    cur_ax.get_position().height,
                ])

                # Create colorbar.
                cur_subfig.colorbar(artist, cax=cax)
        elif mode == "histogram":
            min_score, max_score = 0, 1

            # We cut off the histogram at the min score because the min score is
            # known, but we increase the max score a bit to show solutions which
            # exceed the reward threshold.
            bin_counts, bins, patches = cur_ax.hist(  # pylint: disable = unused-variable
                archive.as_pandas().objective_batch(),
                range=(min_score, max_score + 400),
                bins=100,
            )

            # Rough estimate of max items in a bin.
            cur_ax.set_ylim(top=150)

            # Force ticks, as there are no ticks when the plot is empty.
            cur_ax.set_yticks([0, 50, 100, 150])

            # Alternative - logarithmic scale (harder to understand).
            #  cur_ax.set_yscale("log")
            #  cur_ax.set_ylim(top=1000)

            # Set axis grid to show up behind the histogram.
            # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
            cur_ax.set_axisbelow(True)

            # Set up grid lines on y-axis. Style copied from simple.mplstyle.
            cur_ax.grid(color="0.9", linestyle="-", linewidth=0.3, axis="y")

            # Color the histogram with viridis.
            # https://stackoverflow.com/questions/51347451/how-to-fill-histogram-with-color-gradient-where-a-fixed-point-represents-the-mid
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            maxi = np.abs(bin_centers).max()
            norm = plt.Normalize(-maxi, maxi)
            cm = plt.cm.get_cmap("viridis")
            for c, p in zip(bin_centers, patches):
                # Also rasterize so we do not see "white lines" between bars.
                p.set(facecolor=cm(norm(c)), rasterized=True)

            # Hide spines.
            for pos in ['right', 'top', 'left']:
                cur_ax.spines[pos].set_visible(False)

            if col == ncols // 2:
                # Set x label on middle plot.
                cur_ax.set_xlabel("Objective")

        # Adjust position of plots -- somehow, we need to adjust it on every
        # iteration rather than just adjusting it once at the end.

        # `left` and `right` are tuned for the 5 column case -- they come out to
        # 0.065 and 0.95 for 5 columns.
        #  left = 0.065 * 5 / ncols
        #  right = 1 - 0.05 * 5 / ncols

        left = 0.11
        right = 0.98
        if mode == "heatmap":
            cur_subfig.subplots_adjust(left=left, right=right)
            #  cur_subfig.tight_layout()
        elif mode == "histogram":
            cur_subfig.subplots_adjust(left=left,
                                       bottom=0.25,
                                       right=right,
                                       top=0.7)

    # Add suptitle at center of plots.
    #  center_x = (ax[0].get_position().x0 + ax[-1].get_position().x1) / 2
    #  cur_subfig.suptitle(env, x=center_x, y=0.95)


# Figure sizes for one row of archive visualizations (e.g. heatmaps).
ARCHIVE_FIG_WIDTH = 1.2
ARCHIVE_FIG_HEIGHT = 1.4


def visualize_archives(manifest: str,
                       output: str = None,
                       mode: str = "heatmap",
                       video: bool = False,
                       sans: bool = False):
    """Generates archive visualizations for appendix.

    Requires a manifest which looks like this:

        Archive Visualization:
          Environment 1:
            heatmap: True/False  # Whether to plot heatmap in this environment
                                 # (not all environments support heatmaps).
            xlabel: "XXX"  # Label for x-axis.
            ylabel: "XXX"  # Label for y-axis.
            # Algorithms are assumed to be the same across all environments.
            Algorithms:
              Name 1:
                logdir: XXXX
              Name 2:
                logdir: XXXX
              ...
          Environment 2:
            heatmap: ...
            xlabel: ...
            ylabel: ...
            Algorithms:
              ...
          ...

    Note this manifest can be included in the same document as the one for
    figures.py and for `agent_videos`.

    Args:
        manifest: Path to manifest file.
        output: Output directory for figures. Within this directory, we save a
            separate figure for each env.
        mode: Either "heatmap" or "histogram".
        video: Whether this function is being called as part of video
            generation. This induces a few special settings.
        sans: Pass this in to use Sans Serif fonts.
    """
    assert mode in ["heatmap", "histogram"], \
        f"Invalid mode {mode}"

    outdir = Path(output)
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir()

    logger.info("Loading manifest")
    viz_data, root_dir = load_manifest(manifest, "Archive Visualization")
    if mode == "heatmap":
        # Only keep environments that support heatmap.
        viz_data = {
            key: val
            for key, val in viz_data.items()
            if val.get("heatmap", False)
        }

    logger.info("Plotting visualizations")
    with mpl_style_file("archive_visualization_sans.mplstyle"
                        if sans else "archive_visualization.mplstyle") as f:
        with plt.style.context(f):
            if video:
                logger.info("Using video mode")

            for env in viz_data:
                n_algos = len(viz_data[env]["Algorithms"])
                fig = plt.figure(figsize=(ARCHIVE_FIG_WIDTH * n_algos,
                                          ARCHIVE_FIG_HEIGHT))

                visualize_env_archives(
                    mode,
                    root_dir,
                    viz_data,
                    env,
                    fig,
                )

                # The colorbar is giving trouble, so tight_layout does not work,
                # and bbox_inches="tight" does not work either as it seems to
                # cut things off despite this:
                # https://stackoverflow.com/questions/35393863/matplotlib-error-figure-includes-axes-that-are-not-compatible-with-tight-layou
                # Hence, we manually rearrange everything e.g. with
                # subplots_adjust.
                fig.savefig(outdir /
                            f"{slugify(env)}{'_sans' if sans else ''}.pdf",
                            dpi=600 if video else 300)
                plt.close(fig)

    logger.info("Done")


heatmap_figure = functools.partial(visualize_archives,
                                   mode="heatmap",
                                   video=False)
histogram_figure = functools.partial(visualize_archives,
                                     mode="histogram",
                                     video=False)

if __name__ == "__main__":
    fire.Fire({
        "heatmap_figure": heatmap_figure,
        # Histogram doesn't work because there are no objective values.
        "histogram_figure": histogram_figure,
    })
