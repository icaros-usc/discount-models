"""Plots metrics from one or more experiments.

Usage:
    python -m src.analysis.plot_metrics LOGDIR1 LOGDIR2 ...
Example:
    python -m src.analysis.plot_metrics \
        logs/2021-01-18_13-39-09_map-elites-damage-ant
"""

import argparse
import itertools
import pickle as pkl
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from logdir import LogDir

from src.mpl_styles import QualitativeMap
from src.mpl_styles.utils import mpl_style_file
from src.utils.metric_logger import MetricLogger

# Matplotlib markers.
MARKERS = itertools.cycle("oD^vX+")


@dataclass
class Experiment:
    """Data that can be used to plot an experiment."""

    name: str  # Used for labelling in plots.
    cutoff: int  # Cutoff generation, e.g. "only plot up to gen 200."
    metrics: MetricLogger
    objs: List[float]  # Objective values in the archive at the cutoff gen.


def plot_metrics(  # pylint: disable = too-many-locals
    experiments: List[Experiment],
    command: str,
    x_eval: bool,
    auc: bool,
    colormap: "matplotlib colormap" = QualitativeMap,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plots the metrics in a matplotlib figure and returns the figure.

    Also creates a table of metrics and places those in a DataFrame. The table
    is also returned.

    Args:
        command: String with the command used to invoke this script.
    """
    all_names = set()
    for e in experiments:
        all_names |= set(e.metrics.names)

    # Create the figure and layout the plots.
    num_plots = len(all_names)
    fig_plots = num_plots + 4
    cols = 4
    rows = fig_plots // cols + int(bool(fig_plots % cols))
    fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    ax = ax.ravel()

    # Clear unused plots.
    for i in range(fig_plots, len(ax)):
        ax[i].set_axis_off()

    name_to_ax = {}
    for i, name in enumerate(sorted(all_names)):
        name_to_ax[name] = ax[i]
        ax[i].set_title(name)
        ax[i].set_xlabel("Evaluations" if x_eval else "Generations")

    ccdf = ax[num_plots]
    ccdf.set_title("Archive Performance CCDF")
    ccdf.set_xlabel("Performance")
    ccdf.set_ylabel("Num Entries")
    ccdf.grid(True)

    ccdf_norm = ax[num_plots + 1]
    ccdf_norm.set_title("Archive Performance CCDF (Normalized)")
    ccdf_norm.set_xlabel("Performance")
    ccdf_norm.set_ylabel("Proportion of Entries")
    ccdf_norm.set_yticks(np.arange(0, 1.01, 0.1))
    ccdf_norm.grid(True)

    if experiments[0].objs is not None:
        dist = ax[num_plots + 2]
        dist.set_title("Distribution of Performance")
        dist.set_xlabel("Performance")
        dist.set_ylabel("Density")

        hist = ax[num_plots + 3]
        hist.set_title("Histogram of Performance")
        hist.set_xlabel("Performance")
        hist.set_ylabel("Number of Entries")
        concat_objs = np.concatenate([e.objs for e in experiments])
        concat_labels = sum(([e.name] * len(e.objs) for e in experiments), start=[])
        sns.histplot(
            data=pd.DataFrame(
                {
                    "obj": concat_objs,
                    "Experiment": concat_labels,
                }
            ),
            x="obj",
            hue="Experiment",
            ax=hist,
            fill=True,
            element="step",
        )

    table_data = {
        "Name": [],
        "QD Score": [],
        "QD AUC": [],
        "Coverage": [],
        "Size AUC": [],
        "Best": [],
        "Mean": [],
        "Corrected QD Score": [],
        "Corrected Coverage": [],
    }
    if not auc:
        table_data.pop("QD AUC")
        table_data.pop("Size AUC")
    default = {"y": [pd.NA]}

    for e_idx, (e, marker) in enumerate(zip(experiments, MARKERS)):
        data = e.metrics.get_plot_data()
        total_evals = np.asarray(data["Evaluations"]["y"], dtype=int)

        # Cut off extra data - keep in mind some data start with an extra 0 at
        # the beginning.
        for d in data:
            cutoff = e.cutoff + 1 if data[d]["x"][0] == 0 else e.cutoff
            data[d]["y"] = data[d]["y"][:cutoff]
            data[d]["x"] = data[d]["x"][:cutoff]
            if x_eval:  # Switch x-axis to be evals.
                data[d]["x"] = total_evals[data[d]["x"]]

        # Assumes same number of evals per generation.
        evals_per_gen = total_evals[-1] / (len(total_evals) - 1)
        qd_scores = data.get("Actual QD Score", default)["y"]
        coverages = data.get("Coverage", default)["y"]

        # Collect data for the table.
        table_data["Name"].append(e.name)
        table_data["QD Score"].append(qd_scores[-1])
        if auc:
            table_data["QD AUC"].append(sum(qd_scores) * evals_per_gen)
        table_data["Coverage"].append(coverages[-1])
        if auc:
            table_data["Size AUC"].append(sum(archive_sizes) * evals_per_gen)
        table_data["Best"].append(data.get("Best Objective", default)["y"][-1])
        table_data["Mean"].append(data.get("Mean Objective", default)["y"][-1])
        table_data["Corrected QD Score"].append(data.get("Corrected QD Score", default)["y"][-1])
        table_data["Corrected Coverage"].append(data.get("Corrected Coverage", default)["y"][-1])

        # Plot all metrics.
        color = colormap(e_idx)
        for name in data:
            vals = data[name]["y"]
            if len(experiments) == 1:
                # Add info in title if plotting just one experiment.
                name_to_ax[name].set_title(
                    (
                        f"{name}\n"
                        f"Final: {vals[-1]:.3f}\n"
                        f"Max: {np.max(vals):.3f} Min: {np.min(vals):.3f}\n"
                        f"Mean: {np.mean(vals):.3f} Median: {np.median(vals):.3f}"
                    ),
                    pad=12.0,
                )
            name_to_ax[name].plot(
                data[name]["x"],
                vals,
                label=e.name,
                marker=marker,
                markevery=0.25,
                color=color,
            )

        if e.objs is not None:
            # CCDF of performance in the archive.
            ccdf.hist(
                e.objs,
                50,
                histtype="step",
                density=False,
                cumulative=-1,
                label=e.name,
                color=color,
            )

            # Also a CCDF but normalized.
            ccdf_norm.hist(
                e.objs,
                50,
                histtype="step",
                density=True,
                cumulative=-1,
                label=e.name,
                color=color,
            )

            sns.kdeplot(ax=dist, x=e.objs, fill=True, label=e.name, color=color)

    # Add legends.
    for i in range(0, num_plots + 3):
        ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))

    # Write command in the figure title.
    title = textwrap.fill(
        "Comparison of "
        + ", ".join(f"({i}) {e.name}" for i, e in enumerate(experiments, 1)),
        width=160,
    )
    fig.suptitle(
        f"{title}\n\n== COMMAND ==\n\n{command}",
        fontfamily="monospace",
        horizontalalignment="left",
        x=0.05,
    )

    # Leave space for the suptitle.
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig, pd.DataFrame(table_data)


def get_cmd_str(width: int = 160) -> str:
    """Output the command used to invoke this script, wrapped to the given
    width."""
    # Wrap the command.
    lines = textwrap.wrap(
        "python -m src.analysis.plot_metrics " + " ".join(sys.argv[1:]),
        width=width,
        break_on_hyphens=False,
    )

    # Add backslashes so the command can be interpreted properly.
    cmd = " \\\n    ".join(lines)

    return cmd


def main(
    logdirs: List[str],
    user_gen: int = None,
    x_eval: bool = False,
    user_eval: int = None,
    auc: bool = False,
    sparse: bool = False,
    output: str = None,
):
    """Plots visualization of metrics in the given logdir(s).

    If there is one logdir, metrics will be read in from the logdir and written
    to metrics.pdf in the logdir (by default). Otherwise, metrics will be read
    in from all logdirs and written to a single file called metrics.pdf (by
    default).

    See command line flags for info on args.
    """
    if x_eval:
        user_gen = None

    experiments = []

    for logdir_name in logdirs:
        print(f"Loading from {logdir_name}")
        logdir = LogDir("Experiment", custom_dir=logdir_name)

        metrics = MetricLogger.from_json(logdir.file("metrics_sparse.json" if sparse else "metrics.json"))
        print("-> Loaded metrics")

        max_gens = metrics.total_itrs
        total_evals = metrics.get_single("Evaluations")["y"]
        if x_eval and user_eval is not None:
            cutoff = np.searchsorted(total_evals, user_eval, side="right") - 1
        elif not x_eval and user_gen is not None:
            cutoff = min(user_gen, max_gens)
        else:
            cutoff = max_gens
        print(f"-> Cutoff generation: {cutoff}")

        # TODO: Below code only loads last generation.

        # Load the result archive, or if that is not available, load the
        # archive.
        paths = list((logdir.logdir / "archive").glob("result_archive_*.pkl"))
        if len(paths) == 0:
            paths = list((logdir.logdir / "archive").glob("archive_*.pkl"))

        if len(paths) == 0:
            # This must be a run that does not have an archive at all.
            objs = None
        else:
            archive_path = paths[0]
            with archive_path.open("rb") as file:
                data = pkl.load(file)
                objs = data["objective"]
            print("-> Loaded objs from FINAL generation")

        experiments.append(Experiment(logdir.logdir.name, cutoff, metrics, objs))

    with mpl_style_file("simple.mplstyle") as f:
        with plt.style.context(f):
            print("Generating plots")
            fig, table_df = plot_metrics(experiments, get_cmd_str(), x_eval, auc)

            if output is None:
                metrics_file = (
                    logdir.file("metrics.pdf") if len(logdirs) == 1 else "metrics.pdf"
                )
            else:
                metrics_file = output
            fig.savefig(metrics_file)

            table_str = table_df.to_latex(
                index=False,
                column_format="l" + "r" * (len(table_df.columns) - 1),
                float_format="{:.2f}".format,
            )

            print(f"\n==== Table ====\n\n{table_str}")
            print(f"Saved plots to {metrics_file}")


def parse_flags():
    """Parses flags from command line."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "logdirs",
        nargs="+",
        help="Logging directories from which to plot metrics",
        metavar="LOGDIR",
    )
    parser.add_argument(
        "-g",
        "--user-gen",
        type=int,
        default=None,
        help="Maximum generation to plot. Ignored if x-eval is passed.",
    )
    parser.add_argument(
        "-x",
        "--x-eval",
        action="store_true",
        default=False,
        help="Plot evaluations on the x-axis instead of generations.",
    )
    parser.add_argument(
        "-e",
        "--user-eval",
        type=int,
        default=None,
        help="Maximum number of evaluations to plot if --x-eval is True",
    )
    parser.add_argument(
        "-s",
        "--sparse",
        action="store_true",
        default=False,
        help="Load sparse metrics (i.e., metrics_sparse.json).",
    )
    parser.add_argument(
        "-a",
        "--auc",
        action="store_true",
        default=False,
        help="Output AUC (Area Under Curve) in the table.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="OUTPUT.pdf",
        help=(
            "Path to save plots. Defaults to 'logdir / metrics.pdf' if there "
            "is one logdir, and just 'metrics.pdf' if there is more than one "
            "logdir"
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_flags()
    main(
        ARGS.logdirs,
        ARGS.user_gen,
        ARGS.x_eval,
        ARGS.user_eval,
        ARGS.auc,
        ARGS.sparse,
        ARGS.output,
    )
