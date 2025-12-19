r"""Plots figures and tables for the paper.

## Overview

The experiments output logging directories which are large and difficult to
manage. This script first gathers the relevant data from these directories into
one file, `figure_data.json`. `figure_data.json` is then passed around in order
to make the figures.

## Generating `figure_data.json` from logging directories

(If you already have `figure_data.json`, skip this section)

After running your experiments, arrange your logging directories as follows:

    logs/  # Any name is okay; you could even put it in `.` but that is messy.
      - manifest.yaml
      - logging-directory-1
      - logging-directory-2
      - ...

`manifest.yaml` lists the directories that were generated from your experiments.
It must be located in the same directory as all the logging directories, and it
must have the following format:

    Paper:  # Top-level object.
      Environment 1:
        old_min_obj: -8765.4321...
        min_obj: -1234.5678...
        max_obj: 3210.5678...
        archive_size: 1024
        algorithms:
            Algorithm 1:
              - dir: logging-directory-1...
                seed: 1
              - dir: logging-directory-2...
                seed: 3  # Make sure this matches the seed for the experiment.
              ...
            Algorithm 2:
              - exclude  # Causes this algorithm to be excluded.
              - dir: logging-directory-3...
                seed: 1
              ...
            Algorithm 3:
              - no_old_min_obj  # Causes old_min_obj to be ignored for this
                                # algorithm.
              - dir: logging-directory-4...
                seed: 2
              ...
            ...
      ...

The fields are as follows:
- `old_min_obj` and `min_obj` are used for the QD score calculations --
  `old_min_obj` is the min that was used for calculating the QD score during the
  experiment, and after the experiment, we recalculate the QD score with the
  `min_obj`. This is necessary since the final QD score offset is based on the
  lowest objective value that was ever inserted into the archive (see the
  `find_min` function below), and we do not know this value during the
  experiments.
- `max_obj` is the maximum objective in the environment
- `archive_size` is the number of cells in the archive grid

You can leave `min_obj` blank for now. We'll generate it in the next step.

Once you have this manifest, run the following commands (replace
`logs/manifest.yaml` with the path to your manifest). Run all Python commands in
the Singularity container associated with this project, e.g. run `make shell`
to start a shell in the container and run the commands in that shell.

    # Collect min objectives for each environment with the following command,
    # and manually add these under the min_obj field in the manifest.
    python -m src.analysis.figures find_min logs/manifest.yaml

    # Generate `figure_data.json`
    python -m src.analysis.figures collect logs/manifest.yaml

For reference, figure_data.json looks like this:

    {
        "Env 1": {
            # List of entries for the algorithm, where each entry contains data
            # from a logging directory.
            "Algo 1": [
                {
                    # Number of evaluations completed after each iteration. Used
                    # on x-axis.
                    "Evaluations": [...],

                    # Number of iterations completed. Usually just counts from 0
                    # like [0, 1, 2, ...]
                    "Iterations": [...],

                    # Metrics with a series of values from each generation. Some
                    # lists have length `gens + 1` because they include a 0 at
                    # the start, and others only have length `gens`.
                    "series_metrics": {
                        "QD Score": [...],

                        # QD Score divided by max QD score. Only used in
                        # statistical analysis.
                        "Normalized QD Score": [...],

                        "Archive Coverage": [...],

                        "Best Performance": [...],
                    }

                    # Metrics that only have a final value.
                    "point_metrics": {
                        # Total runtime in hours.
                        "Runtime (Hours)": XXX,
                    },
                },
                ...
            ],
            ...
        },
        ...
    }

## Generating figures

Run these commands to generate all figures associated with the paper (replace
`figure_data.json` with the path to your figure_data). The default values are
set such that these commands generate the versions shown in the paper. Run all
Python commands in the Singularity container associated with this project, e.g.
run `make shell` to start a shell in the container and run the commands in that
shell.

    # For the comparison figure.
    python -m src.analysis.figures comparison figure_data.json

    # For the higher-res version of the comparison figure.
    python -m src.analysis.figures comparison_high_res figure_data.json

    # To generate the latex source for the tables in the paper.
    python -m src.analysis.figures table figure_data.json

    # To generate statistical test results.
    python -m src.analysis.figures tests figure_data.json

If including the Latex files output by these commands, make sure to also put
these commands in your paper:

    \usepackage{booktabs}
    \usepackage{multirow}
    \usepackage{array}
    \newcolumntype{L}[1]
        {>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
    \newcolumntype{C}[1]
        {>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
    \newcolumntype{R}[1]
        {>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

## Help

Run the following for more help:

    python -m src.analysis.figures COMMAND --help
"""

import itertools
import json
import shutil
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import seaborn as sns
import slugify
from logdir import LogDir
from loguru import logger
from ruamel.yaml import YAML
from statsmodels.graphics.factorplots import interaction_plot

from src.mpl_styles import QUALITATIVE_COLORS
from src.mpl_styles.utils import mpl_style_file
from src.utils.metric_logger import MetricLogger

# Metrics which we record in figure_data.json but do not plot.
METRIC_BLACKLIST = []

# Reordered version of Seaborn colorblind palette. Use np.asarray so we can rearrange
# with indices.
COLORBLIND_REORDERED = list(
    np.asarray(sns.color_palette("colorblind"))[[0, 1, 8, 2, 4, 3, 5, 6, 7, 9]]
)
# Unsure if these are colorblind-friendly, but they are needed since we don't have
# enough colors in some plots.
COLORBLIND_REORDERED.extend(
    [
        [0.5, 0.5, 0.5],  # gray
        [1.0, 1.0, 1.0],  # black
    ]
)

COLORBLIND_DUPLICATES = np.array(sns.color_palette("colorblind"))[[0, 0, 1, 1]]


def load_manifest(manifest: str, entry: str = "Paper"):
    """Loads the data and root directory of the manifest."""
    manifest = Path(manifest)
    data = YAML().load(manifest)[entry]
    root_dir = manifest.parent
    return data, root_dir


def exclude_from_manifest(paper_data, env, algo):
    """Whether to exclude the given algo."""
    return "exclude" in paper_data[env]["algorithms"][algo]


def verify_manifest(
    paper_data, root_dir: Path, reps: int, key: str, check_seed: bool = True
):
    """Checks logging directories for correctness."""
    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            results = paper_data[env]["algorithms"][algo]
            if results[0] == "no_old_min_obj":
                results = results[1:]
            name = f"(Env: {env} Algo: {algo})"

            if reps is not None:
                # Check that the reps are correct.
                assert len(results) == reps, f"{name} {len(results)} dirs, needs {reps}"

            # Check logging dirs are unique.
            logdirs = [d["dir"] for d in results]
            assert len(logdirs) == len(set(logdirs)), (
                f"{name} {len(logdirs)} dirs listed, {len(set(logdirs))} unique"
            )

            if check_seed:
                # Check seeds are unique.
                seeds = [d["seed"] for d in results]
                if len(seeds) != len(set(seeds)):
                    logger.warning(
                        f"{name} {len(seeds)} seeds listed, {len(set(seeds))} unique"
                    )

                # Check seeds match.
                for d in results:
                    logdir = LogDir("tmp", custom_dir=root_dir / d["dir"])
                    actual_seed = int(logdir.pfile("seed").open("r").read())
                    assert actual_seed == d["seed"], (
                        f"{name} Seed for {logdir} ({d['seed']}) does not "
                        f"match actual seed ({actual_seed})"
                    )


def collect(
    manifest: str,
    key: str = "Paper",
    reps: int = 10,
    output: str = "figure_data.json",
    check_seed: bool = True,
    robust: bool = False,
    mode: str = "discrete",
):
    """Collects data from logging directories and aggregates into a single JSON.

    Args:
        manifest: Path to YAML file holding paths to logging directories.
        key: The key in the manifest for loading the data.
        reps: Number of times each experiment should be repeated.
        output: Path to save results.
        check_seed: Whether to check seeds.
        robust: Collect robustness metrics output by analysis/robustness.py.
    """
    logger.info("Loading manifest")
    paper_data, root_dir = load_manifest(manifest, key)

    #  logger.info("Verifying logdirs")
    #  verify_manifest(paper_data, root_dir, reps, key, check_seed)

    # Mapping from the name in the raw metrics to the name in the output.
    if mode == "discrete":
        metric_names = {
            "QD Score": "QD Score",
            "Archive Coverage": "Coverage",
            #  "Final Discount Loss": "Discount Loss",
        }
    elif mode == "distortion":
        metric_names = {
            "Unique Cells": "Unique Cells",
        }
    elif mode == "time":
        metric_names = {
            "Total Time: All": "Time",
        }
    elif mode == "model":
        metric_names = {
            "Mean Feature Error": "Mean Feature Error",
        }
    else:
        raise ValueError

    figure_data = {}

    logger.info("Loading Plot Data")
    for env in paper_data:
        figure_data[env] = OrderedDict()
        env_data = paper_data[env]

        for algo in env_data["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            figure_data[env][algo] = []

            entries = env_data["algorithms"][algo]
            no_old_min_obj = entries[0] == "no_old_min_obj"
            if no_old_min_obj:
                entries = entries[1:]

            for entry in entries:
                figure_data[env][algo].append(cur_data := {})
                logdir = LogDir("Experiment", custom_dir=root_dir / entry["dir"])
                if mode in ["discrete", "distortion", "time"]:
                    metrics = MetricLogger.from_json(logdir.file("metrics.json"))
                elif mode == "model":
                    metrics = MetricLogger.from_json(logdir.file("metrics_sparse.json"))
                else:
                    raise ValueError

                cur_data["Evaluations"] = metrics.get_single("Evaluations")["y"]
                cur_data["Iterations"] = (
                    np.arange(len(cur_data["Evaluations"])) * metrics.x_scale
                ).tolist()

                cur_data["series_metrics"] = {}
                for actual_name, figure_name in metric_names.items():
                    if actual_name not in metrics.keys:
                        continue
                    data = metrics.get_single(actual_name)
                    cur_data["series_metrics"][figure_name] = data["y"]

                cur_data["point_metrics"] = {}

    logger.info("Saving to {}", output)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(figure_data, file)
    logger.info("Done")


def legend_info(names: Iterable, palette: dict, markers: dict):
    """Creates legend handles and labels for the given palette and markers.

    Yes, this is kind of a hack.
    """
    _, ax = plt.subplots(1, 1)
    for name in names:
        # We just need the legend handles, so the plot itself is bogus.
        ax.plot(
            [0],
            [0],
            label=name,
            color=palette[name],
            marker=markers[name],
            markeredgewidth="0.75",
            markeredgecolor="white",
        )
    return ax.get_legend_handles_labels()


def load_figure_data(figure_data: str):
    with open(figure_data, "r") as file:
        return json.load(file)


def metric_from_entry(entry, metric):
    """Retrieves the metric from either series_metrics or point_metrics.

    entry is a dict in the list associated with figure_data[env][algo]
    """
    return (
        entry["series_metrics"][metric][-1]
        if metric in entry["series_metrics"]
        else entry["point_metrics"][metric]
    )


PLOT_INFO = {
    # Main paper.
    ("QD Score", "2D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "2D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "10D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "10D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "2D LP (Rastrigin)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "2D LP (Rastrigin)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "10D LP (Rastrigin)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "10D LP (Rastrigin)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "2D LP (Flat)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "2D LP (Flat)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "10D LP (Flat)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "10D LP (Flat)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "Arm Repertoire"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("Coverage", "Arm Repertoire"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "TA (MNIST)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1000],
        "yticks": [0, 500, 1000],
    },
    ("Coverage", "TA (MNIST)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "TA (F-MNIST)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1000],
        "yticks": [0, 500, 1000],
    },
    ("Coverage", "TA (F-MNIST)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("QD Score", "LSI (Hiker)"): {
        "xlim": [0, 10000],
        "ylim": [-400, 400],
        "yticks": [-400, 0, 400],
    },
    ("Coverage", "LSI (Hiker)"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.10],
        "yticks": [0, 0.05, 0.10],
    },
    ###########################################k
    ("QD Score", "Arm 10D"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("QD Score", "Arm 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("QD Score", "Arm 100D (Restart = 500)"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("QD Score", "Sphere LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 8000],
        "yticks": [0, 4000, 8000],
    },
    ("QD Score", "Flat LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 10000],
        "yticks": [0, 5000, 10000],
    },
    ("QD Score", "Flat Multi-Dim LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 10000],
        "yticks": [0, 5000, 10000],
    },
    ("QD Score", "MNIST Bold Light"): {
        "xlim": [0, 10000],
        "ylim": [0, 4000],
        "yticks": [0, 2000, 4000],
    },
    ("Coverage", "Arm 10D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "Arm 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "Arm 100D (Restart = 500)"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "Sphere LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "Flat LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "Flat Multi-Dim LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1],
    },
    ("Coverage", "MNIST Bold Light"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.4],
    },
    ("Discount Loss", "Arm 10D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.05],
    },
    ("Discount Loss", "Arm 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.05],
    },
    ("Discount Loss", "Sphere LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.05],
    },
    ("Discount Loss", "Flat LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.05],
    },
    ("Discount Loss", "MNIST Bold Light"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.05],
    },
    ("Mean Feature Error", "Arm 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.4],
        "yticks": [0, 0.2, 0.4],
    },
    ("Mean Feature Error", "Sphere LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.4],
        "yticks": [0, 0.2, 0.4],
    },
    ("Mean Feature Error", "Flat LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.4],
        "yticks": [0, 0.2, 0.4],
    },
    ("Mean Feature Error", "Flat Multi-Dim LP 100D"): {
        "xlim": [0, 10000],
        "ylim": [0, 1.0],
        "yticks": [0, 0.5, 1.0],
    },
    ("Mean Feature Error", "MNIST Bold Light"): {
        "xlim": [0, 10000],
        "ylim": [0, 0.6],
        "yticks": [0, 0.3, 0.6],
    },
    ("Unique Cells", "2D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 400],
        "yticks": [0, 200, 400],
    },
    ("Unique Cells", "10D LP (Sphere)"): {
        "xlim": [0, 10000],
        "ylim": [0, 120],
        "yticks": [0, 60, 120],
    },
    ("Unique Cells", "CMA-MAE"): {
        "xlim": [0, 10000],
        "ylim": [0, 400],
        "yticks": [0, 200, 400],
    },
}


# Note: It is also possible to plot with respect to Evaluations by switching out
# Iterations with Evaluations in the code below.
def comparison(
    figure_data: str = "./figure_data.json",
    output: str = "comparison",
    palette_name: str = "colorblind_reordered",
    height: float = 1.9,
    plot_every: int = 25,
    sans: bool = False,
    iteration: int = -1,
    plot_solo: bool = True,
    show_legend: bool = True,
    compare_env: bool = False,
    show_col_titles: bool = True,
    groups: list[list[str]] = (None,),
):
    """Plots the figure comparing metrics of all algorithms.

    Args:
        figure_data: Path to JSON file with figure data.
        outputs: Output directory for saving the figures.
        palette: Either a Seaborn color palette, "qualitative_colors" for
            QUALITATIVE_COLORS, or "colorblind_reordered" for
            COLORBLIND_REORDERED.
        height: Height in inches of each plot.
        plot_every: How frequently to plot points, e.g. plot every 100th point.
        sans: Pass this in to use Sans Serif fonts.
        iteration: Pass this to cut off the plots at some iteration. Defaults to
            the last iteration.
        groups: List of list of environments for grouping the results (so that
            they are plotted in separate documents).
    """
    logger.info("Creating output directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    figure_data = load_figure_data(figure_data)

    for group_i, group in enumerate(groups):
        if group is not None:
            cur_figure_data = {env: figure_data[env] for env in group}
            extra_name = f"{group_i}"
        else:
            cur_figure_data = figure_data
            extra_name = ""

        if plot_solo:
            conds = [True, False]
        else:
            conds = [False]

        for qd_score_only in conds:
            plot_data = {
                "Environment": [],
                "Algorithm": [],
                "Metric": [],
                "Iterations": [],
                "Score": [],
            }

            logger.info("Loading Plot Data")
            if compare_env:
                all_algos = list(cur_figure_data)
            else:
                all_algos = OrderedDict()  # Set of all algorithms, ordered by listing.
            for env in cur_figure_data:
                cur_data = {
                    "Environment": [],
                    "Algorithm": [],
                    "Metric": [],
                    "Iterations": [],
                    "Score": [],
                }
                for algo in cur_figure_data[env]:
                    if not compare_env:
                        all_algos[algo] = None
                    for entry in cur_figure_data[env][algo]:
                        # Has a length of generations + 1, since we add an extra 0
                        # at the start.
                        evals = np.asarray(entry["Iterations"])

                        entry_metrics = entry["series_metrics"]

                        # Reverse so that algorithms are ordered properly in terms
                        # of layers -- we need to reverse here so that the reverse
                        # later on works.
                        for metric in reversed(entry_metrics):
                            if qd_score_only:
                                if metric != "QD Score":
                                    continue
                            else:
                                if metric in METRIC_BLACKLIST:
                                    continue

                            # Plot fewer data points to reduce file size.

                            # Metrics may have length of generations or generations
                            # + 1, as only some metrics (like archive size) add a 0
                            # at the start.
                            metric_data = entry_metrics[metric]
                            raw_len = len(metric_data)
                            not_use_zero = int(len(metric_data) != len(evals))
                            gens = len(evals) - 1
                            # Start at 0 or 1 and end at gens.
                            x_data = np.arange(not_use_zero, gens + 1)

                            idx = list(range(0, raw_len, plot_every))
                            if idx[-1] != raw_len - 1:
                                # Make sure last index is included.
                                idx.append(raw_len - 1)

                            indexed_x_data = x_data[idx]
                            indexed_evals = evals[indexed_x_data]
                            indexed_metric_data = np.asarray(entry_metrics[metric])[idx]
                            data_len = len(indexed_evals)

                            cur_data["Environment"].append(np.full(data_len, env))
                            cur_data["Algorithm"].append(np.full(data_len, algo))
                            cur_data["Metric"].append(np.full(data_len, metric))
                            cur_data["Iterations"].append(indexed_evals)
                            cur_data["Score"].append(indexed_metric_data)

                # Reverse so that algorithms are ordered properly in terms of
                # layers.
                for d in plot_data:
                    plot_data[d].append(np.concatenate(cur_data[d])[::-1])

            # Flatten everything so that Seaborn understands it.
            for d in plot_data:
                plot_data[d] = np.concatenate(plot_data[d])

            logger.info("Generating Plot")
            with (
                mpl_style_file(
                    "simple_sans.mplstyle" if sans else "simple.mplstyle"
                ) as f,
                plt.style.context(f),
            ):
                if palette_name == "qualitative_colors":
                    colors = QUALITATIVE_COLORS
                elif palette_name == "colorblind_reordered":
                    # Rearrange the color-blind template.
                    colors = COLORBLIND_REORDERED
                elif palette_name == "colorblind_duplicates":
                    colors = COLORBLIND_DUPLICATES
                else:
                    colors = sns.color_palette(palette_name)

                palette = dict(zip(all_algos, colors, strict=False))
                markers = dict(
                    zip(all_algos, itertools.cycle("oD^vXPps"), strict=False)
                )

                if compare_env:
                    grid = sns.relplot(
                        data=plot_data,
                        x="Iterations",
                        y="Score",
                        hue="Environment",
                        style="Environment",
                        row="Metric",
                        col="Algorithm",
                        kind="line",
                        errorbar="se",
                        markers=markers,
                        markevery=(0.5, 10.0),
                        dashes=False,
                        # Slightly taller when only plotting coverage.
                        height=1.2 * height if qd_score_only else height,
                        #  aspect=1.35 if coverage_only else 1.61803,  # Golden ratio.
                        aspect=1.35 if qd_score_only else 1.5,
                        facet_kws={"sharey": False},
                        palette=palette,
                        legend=False,
                        linewidth=1.25,
                    )
                else:
                    grid = sns.relplot(
                        data=plot_data,
                        x="Iterations",
                        y="Score",
                        hue="Algorithm",
                        style="Algorithm",
                        row="Metric",
                        col="Environment",
                        kind="line",
                        errorbar="se",
                        markers=markers,
                        markevery=(0.5, 10.0),
                        dashes=False,
                        # Slightly taller when only plotting coverage.
                        height=1.2 * height if qd_score_only else height,
                        #  aspect=1.35 if coverage_only else 1.61803,  # Golden ratio.
                        aspect=1.35 if qd_score_only else 1.5,
                        facet_kws={"sharey": False},
                        palette=palette,
                        legend=False,
                        linewidth=1.25,
                    )

                # Set titles to be the env name.
                if show_col_titles:
                    grid.set_titles("{col_name}")
                else:
                    grid.set_titles("")

                # Turn off titles below top row (no need to repeat).
                for ax in grid.axes[1:].ravel():
                    ax.set_title("")

                # Set the labels along the left column to be the name of the
                # metric.
                if compare_env:
                    left_col = next(iter(cur_figure_data[next(iter(cur_figure_data))]))
                else:
                    left_col = next(iter(cur_figure_data))

                for (row_val, col_val), ax in grid.axes_dict.items():
                    ax.set_axisbelow(True)
                    if col_val == left_col:
                        ax.set_ylabel(row_val, labelpad=10.0)
                    else:
                        ax.set_ylabel("")

                    if (row_val, col_val) in PLOT_INFO:
                        d = PLOT_INFO[(row_val, col_val)]
                        if "xlim" in d:
                            # Note: It seems this does not work; all axes have
                            # to share the same x limits under seaborn?
                            ax.set_xlim(d["xlim"])
                        if "xticks" in d:
                            ax.set_xticks(d["xticks"])
                        if "ylim" in d:
                            ax.set_ylim(d["ylim"])
                        if "yticks" in d:
                            ax.set_yticks(d["yticks"])
                            if row_val == "Coverage":
                                ax.set_yticklabels(
                                    [f"{int(100 * tick)}%" for tick in d["yticks"]]
                                )

                # Add legend and resize figure to fit it.
                fig_width, fig_height = grid.fig.get_size_inches()
                if show_legend:
                    grid.fig.legend(
                        *legend_info(all_algos, palette, markers),
                        # Default -- center it.
                        bbox_to_anchor=[0.5, 1.0],
                        # Use to shift it over a bit to the right.
                        #  bbox_to_anchor=[0.58, 1.0],
                        loc="upper center",
                        # Change // 1 to // 2 etc. for more rows.
                        ncol=(len(palette) + 1) // 2,
                    )
                    # Adjust this based on number of rows above.
                    #  legend_height = 0.30  # For one row.
                    legend_height = 0.60  # For two rows (default).
                    #  legend_height = 0.50  # For two rows (shorter).
                else:
                    legend_height = 0.0
                grid.fig.set_size_inches(fig_width, fig_height + legend_height)

                # Save the figure.
                grid.fig.tight_layout(
                    rect=(0, 0, 1, fig_height / (fig_height + legend_height)), pad=0.4
                )
                name = "comparison-qd-score" if qd_score_only else "comparison"
                for extension in ["pdf", "png", "svg"]:
                    filename = (
                        output
                        / f"{name}{'-sans' if sans else ''}{extra_name}.{extension}"
                    )
                    logger.info("Saving {}", filename)
                    grid.fig.savefig(filename, dpi=300)

    logger.info("Done")


def comparison_high_res(
    figure_data: str = "./figure_data.json",
    output: str = "comparison_high_res",
):
    """Generates the larger version of the figure for the supplemental material.

    Simply calls comparison with the appropriate args.
    """
    return comparison(figure_data, output, height=4, plot_every=5)


# Header lines for table files.
TABLE_HEADER = r"""
% THIS FILE IS AUTO-GENERATED. DO NOT MODIFY THIS FILE DIRECTLY.

"""


def table(
    figure_data: str = "figure_data.json",
    transpose: bool = True,
    output: str = "results_table.tex",
    show_std: bool = True,
):
    """Creates Latex tables showing final values of metrics.

    Make sure to include the "booktabs" and "array" package in your Latex
    document.

    With transpose=False, a table is generated for each environment. Each table
    has the algorithms as rows and the metrics as columns.

    With transpose=True, a table is generated for each metric. Each table has
    the algorithms as rows and the environments as columns.

    Args:
        figure_data: Path to JSON file with figure data.
        transpose: See above.
        output: Path to save Latex table.
    """
    figure_data = load_figure_data(figure_data)

    # Safe to assume all envs have same metrics.
    first_env = list(figure_data)[0]
    first_algo = list(figure_data[first_env])[0]
    first_entry = figure_data[first_env][first_algo][0]
    metric_names = list(first_entry["series_metrics"]) + list(
        first_entry["point_metrics"]
    )
    for name in METRIC_BLACKLIST:
        if name in metric_names:
            metric_names.remove(name)
    logger.info("Metric names: {}", metric_names)

    table_data = {}
    logger.info("Gathering table data")
    for env in figure_data:
        table_data[env] = pd.DataFrame(
            index=list(figure_data[env]), columns=metric_names, dtype=str
        )
        for algo in figure_data[env]:
            for metric in metric_names:
                if metric in first_entry["series_metrics"]:
                    final_metric_vals = np.array(
                        [
                            entry["series_metrics"][metric][-1]
                            for entry in figure_data[env][algo]
                        ]
                    )
                else:
                    final_metric_vals = np.array(
                        [
                            entry["point_metrics"][metric]
                            for entry in figure_data[env][algo]
                        ]
                    )

                mean = final_metric_vals.mean()
                se = scipy.stats.sem(final_metric_vals)

                if metric == "Coverage":
                    if show_std:
                        metric_str = f"{mean * 100:,.2f}"
                        metric_str += f" \\textpm {se * 100:.2f}\\%"
                    else:
                        metric_str = f"{mean * 100:,.2f}\\%"
                elif metric == "Mean Feat Err":
                    metric_str = f"{mean:,.5f}"
                    if show_std:
                        metric_str += f" \\textpm {se:.5f}"
                else:
                    metric_str = f"{mean:,.2f}"
                    if show_std:
                        metric_str += f" \\textpm {se:.2f}"

                table_data[env][metric][algo] = metric_str

    if transpose:
        # "Invert" table_data.
        table_data = {
            metric: pd.DataFrame({env: df[metric] for env, df in table_data.items()})
            for metric in metric_names
        }

    logger.info("Writing to {}", output)
    with open(output, "w") as file:
        file.write(TABLE_HEADER)
        for name, df in table_data.items():
            if name == "QD Score AUC":
                caption = name + " (multiple of $10^{12}$)"
            elif name == "QD Score":
                caption = name + " (multiple of $10^{6}$)"
            else:
                caption = name

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(name) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                df.to_latex(
                    column_format="l" + " R{0.9in}" * len(df.columns),
                    escape=False,
                )
            )
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")

    logger.info("Done")


def single_table(
    figure_data: str = "figure_data.json",
    output: str = "results_single_table.tex",
    output_format: str = "latex",
    iteration: int = -1,
    mode: str = "discrete",
    show_std: bool = True,
    table_groups: list[list[str]] = (None,),
):
    """Creates a single Latex table for the paper.

    Make sure to include the "booktabs" and "array" package in your Latex
    document.

    Args:
        figure_data: Path to JSON file with figure data.
        output: Path to save Latex table.
        iteration: The iteration at which to take the metrics (for series
            metrics). Defaults to the last iteration.
    """
    figure_data = load_figure_data(figure_data)

    # Safe to assume all envs have same metrics.
    first_env = list(figure_data)[0]
    first_algo = list(figure_data[first_env])[0]
    first_entry = figure_data[first_env][first_algo][0]

    # Mapping from metric name in the table (key) to the name in
    # figure_data.json (value), e.g., we could abbreviate "QD": "QD Score".
    if mode == "discrete":
        metric_names = {
            "QD Score": "QD Score",
            "Coverage": "Coverage",
        }
    elif mode == "time":
        metric_names = {
            "Time": "Time",
        }
    elif mode == "model":
        metric_names = {
            "Mean Feat Err": "Mean Feature Error",
        }

    logger.info("Gathering table data")

    envs = list(figure_data)

    # Collect all algorithms.
    algos = []
    for env in figure_data:
        for algo in figure_data[env]:
            if algo not in algos:
                algos.append(algo)

    table_df = pd.DataFrame(
        index=algos,
        columns=pd.MultiIndex.from_product([envs, metric_names.keys()]),
        dtype=str,
    )

    for env in figure_data:
        for metric, raw_name in metric_names.items():
            # The best metric for these is the minimum.
            # Use the abbreviated name that appears in the table.
            min_metrics = ["Cross-Entropy", "Mean Feat Err", "Time"]

            # Second entry of best stores all the algos with the best metric.
            if metric in min_metrics:
                best = np.inf, []
            else:
                best = -np.inf, []

            # If the metric doesn't start with zero, we need to offset the
            # iteration by 1 to get the true final value.
            not_use_zero = len(first_entry["Iterations"]) != len(
                first_entry["series_metrics"][raw_name]
            )
            this_iter = (
                (iteration - 1) if (not_use_zero and iteration != -1) else iteration
            )

            for algo in figure_data[env]:
                if raw_name in first_entry["series_metrics"]:
                    final_metric_vals = np.array(
                        [
                            entry["series_metrics"][raw_name][this_iter]
                            for entry in figure_data[env][algo]
                        ]
                    )
                else:
                    final_metric_vals = np.array(
                        [
                            entry["point_metrics"][raw_name]
                            for entry in figure_data[env][algo]
                        ]
                    )

                #  if metric == "QD":
                #      final_metric_vals /= 1e6

                mean = final_metric_vals.mean()
                se = scipy.stats.sem(final_metric_vals)

                if output_format == "latex":
                    pm = "\\textpm"
                elif output_format == "markdown":
                    pm = "±"

                # Use the abbreviated name that appears in the table.
                if metric == "Coverage":
                    if show_std:
                        metric_str = f"{mean * 100:,.2f}"
                        metric_str += f" {pm} {se * 100:.2f}\\%"
                    else:
                        metric_str = f"{mean * 100:,.2f}\\%"
                elif metric == "Mean Feat Err":
                    metric_str = f"{mean:,.5f}"
                    if show_std:
                        metric_str += f" {pm} {se:.5f}"
                else:
                    metric_str = f"{mean:,.2f}"
                    if show_std:
                        metric_str += f" {pm} {se:.2f}"

                table_df[env, metric][algo] = metric_str

                if metric in min_metrics:
                    if mean < best[0]:
                        best = (mean, [algo])
                    elif np.isclose(mean, best[0]):
                        # If metrics are equal, append to the list of best algos.
                        best[1].append(algo)
                else:
                    if mean > best[0]:
                        best = (mean, [algo])
                    elif np.isclose(mean, best[0]):
                        # If metrics are equal, append to the list of best algos.
                        best[1].append(algo)

            # Highlight the best metrics by bolding them.
            for b in best[1]:
                if output_format == "latex":
                    table_df[env, metric][b] = "{\\bf" + table_df[env, metric][b] + "}"
                elif output_format == "markdown":
                    table_df[env, metric][b] = "**" + table_df[env, metric][b] + "**"
                else:
                    raise NotImplementedError()

    logger.info("Writing to {}", output)
    with open(output, "w") as file:
        if output_format == "latex":
            file.write("\\begin{table*}[t]\n")
            for group in table_groups:
                if group is None:
                    df = table_df
                else:
                    df = table_df[group]

                latex_str = df.to_latex(
                    column_format="l" + "r" * len(df.columns), escape=False
                )
                # yapf: disable
                latex_str = latex_str.replace("\\multicolumn{2}{r}", "\\multicolumn{2}{c}") \
                                     .replace("NaN", "---")
                # yapf: enable

                file.write(latex_str)
            file.write("\\end{table*}\n")
            file.write("\n")
        elif output_format == "markdown":
            for group in table_groups:
                if group is None:
                    df = table_df
                else:
                    df = table_df[group]
                    file.write(f"### {', '.join(group)}\n\n")

                markdown_str = df.to_markdown().replace(" nan ", " --  ")
                file.write(markdown_str)
                file.write("\n\n")
        else:
            raise NotImplementedError()

    logger.info("Done")


def calc_simple_main_effects(figure_data, anova_res, metric):
    """Calculates simple main effects in each environment.

    Reference:
    http://www.cee.uma.pt/ron/Discovering%20Statistics%20Using%20SPSS,%20Second%20Edition%20CD-ROM/Calculating%20Simple%20Effects.pdf
    """
    df_residual = anova_res["DF"][3]
    ms_residual = anova_res["MS"][3]

    data = {
        "Environment": ["Residual"],
        "SS": [anova_res["SS"][3]],
        "DF": [df_residual],
        "MS": [ms_residual],
        "F": [np.nan],
        "p-unc": [np.nan],
        "significant": [False],
    }

    for env in figure_data:
        data["Environment"].append(env)

        algos, metric_vals = [], []
        for algo in figure_data[env]:
            entry_metrics = [
                metric_from_entry(entry, metric) for entry in figure_data[env][algo]
            ]

            algos.extend([algo] * len(entry_metrics))
            metric_vals.extend(entry_metrics)

        one_way = pingouin.anova(
            dv=metric,
            between=["Algorithm"],
            data=pd.DataFrame(
                {
                    "Algorithm": algos,
                    metric: metric_vals,
                }
            ),
            detailed=True,
        )

        f_val = one_way["MS"][0] / ms_residual
        p_unc = scipy.stats.f(one_way["DF"][0], df_residual).sf(f_val)
        sig = p_unc < 0.05

        data["SS"].append(one_way["SS"][0])
        data["DF"].append(one_way["DF"][0])
        data["MS"].append(one_way["MS"][0])
        data["F"].append(f_val)
        data["p-unc"].append(p_unc)
        data["significant"].append(sig)

    return pd.DataFrame(data)


def run_pairwise_ttests(figure_data, metric):
    """Runs pairwise t-tests for the hypotheses in the paper."""
    metric_vals = {
        env: {
            algo: [metric_from_entry(entry, metric) for entry in figure_data[env][algo]]
            for algo in figure_data[env]
        }
        for env in figure_data
    }

    def compare_to(main_algo, other_algos, bonf_n=None):
        results = {}
        for env in figure_data:
            df = pd.concat(
                [
                    pingouin.ttest(
                        metric_vals[env][main_algo],
                        metric_vals[env][algo],
                        paired=False,
                        alternative="two-sided",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in other_algos
                ],
                ignore_index=True,
            )

            # Some hypotheses require overriding bonf_n.
            bonf_n = len(df["p-val"]) if bonf_n is None else bonf_n
            # Adapted from pingouin multicomp implementation:
            # https://github.com/raphaelvallat/pingouin/blob/c66b6853cfcbe1d6d9702c87c09050594b4cacb4/pingouin/multicomp.py#L122
            df["p-val"] = np.clip(df["p-val"] * bonf_n, None, 1)
            df["significant"] = np.less(
                df["p-val"],
                0.05,  # alpha
            )
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "Algorithm 1": [main_algo] * len(other_algos),
                            "Algorithm 2": other_algos,
                        }
                    ),
                    df,
                ],
                axis=1,
            )
            results[env] = df
        return results

    coverage_vals = {
        env: {
            algo: [
                metric_from_entry(entry, "Coverage") for entry in figure_data[env][algo]
            ]
            for algo in figure_data[env]
        }
        for env in figure_data
    }

    # Only works on Coverage.
    def noninferiority(main_algo, ref_algo, margin, bonf_n=None):
        results = {}
        for env in figure_data:
            df = pd.concat(
                [
                    pingouin.ttest(
                        coverage_vals[env][main_algo],
                        np.array(coverage_vals[env][algo]) - margin,
                        paired=False,
                        alternative="greater",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in [ref_algo]
                ],
                ignore_index=True,
            )

            # Some hypotheses require overriding bonf_n.
            bonf_n = len(df["p-val"]) if bonf_n is None else bonf_n
            # Adapted from pingouin multicomp implementation:
            # https://github.com/raphaelvallat/pingouin/blob/c66b6853cfcbe1d6d9702c87c09050594b4cacb4/pingouin/multicomp.py#L122
            df["p-val"] = np.clip(df["p-val"] * bonf_n, None, 1)
            df["significant"] = np.less(
                df["p-val"],
                0.05,  # alpha
            )
            df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "Algorithm 1": [main_algo],
                            "Algorithm 2": [ref_algo],
                        }
                    ),
                    df,
                ],
                axis=1,
            )
            results[env] = df
        return results

    return {}
    #  if metric == "Normalized QD Score":
    #      # Examples.
    #      margin = 100000.0
    #      return {
    #          "H2 PGA-ME":
    #              compare_to("PGA-ME", ["LM-MA-MAE", "sep-CMA-MAE", "OpenAI-MAE"],
    #                         6),
    #          "H2 1":
    #              noninferiority("sep-CMA-MAE", "PGA-ME", margin, 3),
    #      }
    #  elif metric == "Normalized QD Score AUC":
    #      return {}
    #  elif metric == "QD Score":
    #      return {}
    #  elif metric == "Runtime (Hours)":
    #      return {}
    #  else:
    #      raise NotImplementedError(f"No hypotheses for {metric}")


def run_pairwise(figure_data, metric, alpha=0.05, test="tukey"):
    metric_vals = {
        env: {
            algo: [metric_from_entry(entry, metric) for entry in figure_data[env][algo]]
            for algo in figure_data[env]
        }
        for env in figure_data
    }

    # Raw results from running the pairwise comparisons. This DF looks something
    # like:
    #
    #      A           B      p_tukey/pval
    # sep-CMA-MAE   CMA-MAE     0.05
    raw_pairwise = {}
    for env in figure_data:
        df = pd.DataFrame(
            {
                "Algorithm": [
                    algo
                    for algo in metric_vals[env]
                    for metric in metric_vals[env][algo]
                ],
                "Metric": [
                    metric
                    for algo in metric_vals[env]
                    for metric in metric_vals[env][algo]
                ],
            }
        )
        if test == "tukey":
            raw_pairwise[env] = pingouin.pairwise_tukey(
                df, dv="Metric", between="Algorithm"
            )
        elif test == "games-howell":
            raw_pairwise[env] = pingouin.pairwise_gameshowell(
                df, dv="Metric", between="Algorithm"
            )

    # Processed to look like:
    #
    #               sep-CMA-MAE      CMA-MAE    ...
    # sep-CMA-MAE       N/A             >
    #   CMA-MAE          <
    envs = list(figure_data)
    algos = list(figure_data[next(iter(figure_data))])
    res = pd.DataFrame(
        index=algos,
        columns=pd.MultiIndex.from_product([envs, algos]),
        dtype=object,
    )

    for env in raw_pairwise:
        # Algorithms can't compare with themselves.
        for algo in algos:
            res.loc[algo, (env, algo)] = "N/A"

        # Use symbols to mark other comparisons.
        # pylint: disable = unused-variable
        for index, row in raw_pairwise[env].iterrows():
            if test == "tukey":
                p = row.loc["p-tukey"]
            elif test == "games-howell":
                p = row.loc["pval"]
            a, b = row.loc["A"], row.loc["B"]
            if p > alpha:
                # No significant difference.
                res.loc[a, (env, b)] = "~"
                res.loc[b, (env, a)] = "~"
            else:
                # Significant result.
                if row.loc["T"] < 0:
                    res.loc[a, (env, b)] = "<"
                    res.loc[b, (env, a)] = ">"
                else:
                    res.loc[a, (env, b)] = ">"
                    res.loc[b, (env, a)] = "<"

    processed_pairwise = res
    return raw_pairwise, processed_pairwise


def tests_for_metric(
    figure_data, root_dir: Path, metric: str, table_groups: list[list[str]]
):
    """Saves tests for the metric into a subdirectory of root_dir.

    Returns results from various tests.
    """
    output = root_dir / slugify.slugify(metric)
    output.mkdir()

    data = {
        "Environment": [],
        "Algorithm": [],
        metric: [],
    }
    grouped_scores = []
    grouped_scores_by_env = {}

    logger.info("Loading {}", metric)
    n_envs = len(figure_data)
    n_algos = 0
    for env in figure_data:
        grouped_scores_by_env[env] = []

        # Not all envs have same number of algos, so take max.
        n_algos = max(n_algos, len(figure_data[env]))

        for algo in figure_data[env]:
            entry_scores = [
                metric_from_entry(entry, metric) for entry in figure_data[env][algo]
            ]

            data["Environment"].extend([env] * len(entry_scores))
            data["Algorithm"].extend([algo] * len(entry_scores))
            data[metric].extend(entry_scores)

            grouped_scores.append(entry_scores)
            grouped_scores_by_env[env].append(entry_scores)
    df = pd.DataFrame(data)
    df.to_csv(output / "data.csv", index=False)

    logger.info("Drawing displot for normality")
    sns.displot(
        data=df,
        x=metric,
        col="Algorithm",
        row="Environment",
        facet_kws={"sharex": False, "sharey": False},
        kind="kde",
    )
    plt.savefig(output / "displot.pdf")

    logger.info("Drawing qqplot for normality")
    fig, axs = plt.subplots(
        nrows=n_envs,
        ncols=n_algos,
        figsize=(4 * n_algos, 4 * n_envs),
    )
    for i_env, env in enumerate(figure_data):
        for i_algo, algo in enumerate(figure_data[env]):
            ax = axs[i_env, i_algo]
            entry_scores = [
                metric_from_entry(entry, metric) for entry in figure_data[env][algo]
            ]
            pingouin.qqplot(entry_scores, dist="norm", ax=ax)
            normality = pingouin.normality(entry_scores, method="shapiro")
            ax.set_title(
                f"{env} | {algo}\n"
                f"Normal: {normality['normal'][0]}, p={normality['pval'][0]}"
            )
    fig.tight_layout()
    fig.savefig(output / "qqplot.pdf")

    logger.info("Drawing interaction plots")
    interaction_plot(
        df["Environment"],
        df["Algorithm"],
        df[metric],
        colors=COLORBLIND_REORDERED[: len(set(df["Algorithm"]))],
    )
    plt.savefig(output / "interaction_plot_1.png")
    interaction_plot(
        df["Algorithm"],
        df["Environment"],
        df[metric],
        colors=COLORBLIND_REORDERED[: len(set(df["Environment"]))],
    )
    plt.savefig(output / "interaction_plot_2.png")

    logger.info("Running ANOVA")
    anova_res = pingouin.anova(
        dv=metric,
        between=["Environment", "Algorithm"],
        data=df,
    )
    anova_res["significant"] = anova_res["p-unc"] < 0.05

    logger.info("Running One-Way ANOVAs")
    one_anova_res = {
        env: pingouin.anova(
            data=df[df["Environment"] == env], dv=metric, between="Algorithm"
        )
        for env in figure_data
    }
    one_anova_str = "\n\n".join(
        f"### {env}\n{one_anova_res[env].to_markdown()}" for env in one_anova_res
    )

    logger.info("Running One-Way Welch ANOVAs")
    welch_anova_res = {
        env: pingouin.welch_anova(
            data=df[df["Environment"] == env], dv=metric, between="Algorithm"
        )
        for env in figure_data
    }
    welch_anova_str = "\n\n".join(
        f"### {env}\n{welch_anova_res[env].to_markdown()}" for env in welch_anova_res
    )

    logger.info("Running simple main effects")
    simple_main = calc_simple_main_effects(figure_data, anova_res, metric)

    logger.info("Running pairwise t-tests")
    ttests = run_pairwise_ttests(figure_data, metric)
    ttest_str_parts = {
        hypothesis: "\n\n".join(
            f"""\
#### {env}

{d.to_markdown()}
"""
            for env, d in results.items()
        )
        for hypothesis, results in ttests.items()
    }
    ttests_str = "\n\n".join(
        f"""\
### {hypothesis}

{str_part}
"""
        for hypothesis, str_part in ttest_str_parts.items()
    )

    logger.info("Running Tukey tests")
    raw_tukey, processed_tukey = run_pairwise(figure_data, metric, test="tukey")

    logger.info("Running Games-Howell tests")
    raw_gameshowell, processed_gameshowell = run_pairwise(
        figure_data, metric, test="games-howell"
    )

    logger.info("Checking homoscedasticity (equal variances)")
    var_test = pingouin.homoscedasticity(
        grouped_scores,
        method="levene",
        alpha=0.05,
    )

    logger.info("Homoscedasticity in each environment")
    var_test_by_env = {
        env: pingouin.homoscedasticity(
            grouped_scores_by_env[env], method="levene", alpha=0.05
        )
        for env in grouped_scores_by_env
    }
    var_test_by_env_str = "\n\n".join(
        f"### {env}\n{var_test_by_env[env].to_markdown()}" for env in var_test_by_env
    )

    markdown_file = output / "README.md"
    logger.info("Writing results to {}", markdown_file)
    with open(markdown_file, "w") as file:
        file.write(f"""\
# Statistical Tests for {metric}

Significance tests done by checking if `p < 0.05`

## ANOVA

{anova_res.to_markdown()}

See [here](https://pingouin-stats.org/generated/pingouin.anova.html) for more
info on the pingouin ANOVA method.

**Table Columns:**

- `Source`: Factor names
- `SS`: Sums of squares
- `DF`: Degrees of freedom
- `MS`: Mean squares
- `F`: F-values
- `p-unc`: uncorrected p-values
- `np2`: Partial eta-square effect sizes

### Analyses

- Environment: $F({anova_res["DF"][0]}, {anova_res["DF"][3]}) = {anova_res["F"][0]:.2f}$
- Algorithm: $F({anova_res["DF"][1]}, {anova_res["DF"][3]}) = {anova_res["F"][1]:.2f}$
- Environment * Algorithm: $F({anova_res["DF"][2]}, {anova_res["DF"][3]}) = {anova_res["F"][2]:.2f}$

## One-Way ANOVA in each environment

{one_anova_str}

## One-Way Welch ANOVA in each environment

{welch_anova_str}

## Normality

See [displot.pdf](./displot.pdf) and [qqplot.pdf](./qqplot.pdf)

## Homoscedasticity (Equal Variances)

{var_test.to_markdown()}

{var_test_by_env_str}

## Interaction Plots

![Interaction Plot 1](./interaction_plot_1.png)

![Interaction Plot 2](./interaction_plot_2.png)

## Simple Main Effects

{simple_main.to_markdown()}

## Pairwise t-tests

(p-values Bonferroni corrected within each environment / simple main effect;
alpha is still 0.05)

{ttests_str}

## Tukey

{processed_tukey.to_markdown()}

## Games-Howell

{processed_gameshowell.to_markdown()}
""")

    write_tests_as_latex(
        metric,
        anova_res,
        simple_main,
        ttests,
        processed_tukey,
        processed_gameshowell,
        output,
        table_groups,
    )

    logger.info("Done")


def format_p_val(p, include_p=False, math=False):
    """Formats p-values for Latex."""
    if p < 0.001:
        p_str = f"{'p ' if include_p else ''}< 0.001"
    elif p >= 1.0:
        p_str = f"{'p = ' if include_p else ''}1"
    else:
        p_str = f"{'p = ' if include_p else ''}{p:.3f}"

    # Put significant p-values in bold.
    if p < 0.05:
        p_str = ("\\mathbf{" if math else "\\textbf{") + p_str + "}"

    return p_str


def write_tests_as_latex(
    metric,
    anova_res,
    simple_main,
    ttests,
    processed_tukey,
    processed_gameshowell,
    output: Path,
    table_groups,
):
    """Write the tests to latex files in the output directory."""
    logger.info("Writing tests as latex")

    # Write ANOVA results as list.
    with open(output / "anova.tex", "w") as file:
        caption = f"Simple main effects for {CAPTION_NAME[metric]}"
        label = f"{slugify.slugify(metric)}-main"
        file.write(TABLE_HEADER)
        file.write("\\begin{itemize}\n")
        file.write(
            "\\item Interaction effect: "
            f"$F({anova_res['DF'][2]}, {anova_res['DF'][3]}) = "
            f"{anova_res['F'][2]:.2f}, "
            f"{format_p_val(anova_res['p-unc'][2], True, True)}$\n"
        )
        file.write("\\item Simple main effects:\n")
        file.write("  \\begin{itemize}\n")

        err_deg = simple_main["DF"][0]
        only_main = simple_main.loc[1:]
        for env, df, f, p in zip(
            only_main["Environment"],
            only_main["DF"],
            only_main["F"],
            only_main["p-unc"],
        ):
            file.write(
                f"  \\item {env}: "
                f"$F({df}, {err_deg}) = {f:.2f}, {format_p_val(p, True, True)}$"
                "\n"
            )

        file.write("  \\end{itemize}\n")
        file.write("\\end{itemize}\n")
        file.write("\n")

    # Write ttests as tables.
    table_i = 0
    for hypothesis, hyp_results in ttests.items():
        with open(output / f"ttests-{table_i}.tex", "w") as file:
            file.write(TABLE_HEADER)

            first_env_df = next(iter(hyp_results.values()))
            pval_df = pd.DataFrame(
                # Index is the algorithms.
                index=pd.MultiIndex.from_frame(
                    first_env_df[["Algorithm 1", "Algorithm 2"]]
                ),
                # Columns are environments.
                columns=list(hyp_results),
                dtype=str,
            )

            for env, env_df in hyp_results.items():
                for algo1, algo2, p in zip(
                    env_df["Algorithm 1"], env_df["Algorithm 2"], env_df["p-val"]
                ):
                    pval_df[env][algo1, algo2] = format_p_val(p)

            caption = f"{hypothesis}"
            label = f"{slugify.slugify(metric)}-ttests-{table_i}"
            table_i += 1

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(label) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                pval_df.to_latex(
                    column_format=" L{1.2 in}" * 2 + " R{0.9in}" * len(pval_df.columns),
                    escape=False,
                    multirow=True,
                )
            )
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")

    # Write tukey as a Latex table.
    with open(output / "tukey.tex", "w") as file:
        file.write(TABLE_HEADER)
        for group in table_groups:
            sub_df = processed_tukey if group is None else processed_tukey[group]

            latex_str = sub_df.to_latex(
                column_format="l" + "r" * len(sub_df.columns),
                escape=False,  # Escape all the special < characters.
                multirow=True,
            )
            # yapf: disable
            latex_str = latex_str.replace("<", "$<$") \
                                 .replace(">", "$>$") \
                                 .replace("~", "$-$") \
                                 .replace("N/A", "$\\varnothing$") \
                                 .replace("NaN", "$\\varnothing$")
            # yapf: enable
            file.write(latex_str + "\n")

    # Write Games-Howell as a Latex table.
    with open(output / "games-howell.tex", "w") as file:
        file.write(TABLE_HEADER)
        for group in table_groups:
            sub_df = (
                processed_gameshowell if group is None else processed_gameshowell[group]
            )

            latex_str = sub_df.to_latex(
                column_format="l" + "r" * len(sub_df.columns),
                escape=False,  # Escape all the special < characters.
                multirow=True,
            )
            # yapf: disable
            latex_str = latex_str.replace("<", "$<$") \
                                 .replace(">", "$>$") \
                                 .replace("~", "$-$") \
                                 .replace("N/A", "$\\varnothing$") \
                                 .replace("NaN", "$\\varnothing$")
            # yapf: enable
            file.write(latex_str + "\n")


# Name in table captions.
CAPTION_NAME = {
    "QD Score": "QD Score",
    "Coverage": "Coverage",
}


def tests(
    figure_data: str = "figure_data.json",
    output: str = "stats_tests",
    table_groups: list[list[str]] = (None,),
):
    """Outputs information about statistical tests.

    We use Normalized scores since the ANOVA needs values to have the same
    scale.

    Args:
        figure_data: Path to JSON file with figure data.
        output: Directory to save (extended) outputs.
    """
    logger.info("Creating logging directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    logger.info("Loading figure data")
    figure_data = load_figure_data(figure_data)

    with mpl_style_file("simple.mplstyle") as f:
        with plt.style.context(f):
            for metric in [
                "QD Score",
                "Coverage",
            ]:
                logger.info("===== {} =====", metric)
                tests_for_metric(figure_data, output, metric, table_groups)


def ablation_plot(
    figure_data: str = "./figure_data.json",
    output: str = "ablation_plot",
    palette_name: str = "colorblind_reordered",
    height: float = 1.9,
    sans: bool = False,
    plot_solo: bool = False,
    groups: list[list[str]] = (None,),
):
    """Plots results for ablation study.

    Args:
        figure_data: Path to JSON file with figure data.
        outputs: Output directory for saving the figures.
        palette: Either a Seaborn color palette, "qualitative_colors" for
            QUALITATIVE_COLORS, or "colorblind_reordered" for
            COLORBLIND_REORDERED.
        height: Height in inches of each plot.
        plot_every: How frequently to plot points, e.g. plot every 100th point.
        sans: Pass this in to use Sans Serif fonts.
    """
    logger.info("Creating output directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    figure_data = load_figure_data(figure_data)

    for group_i, group in enumerate(groups):
        if group is not None:
            cur_figure_data = {env: figure_data[env] for env in group}
            extra_name = f"{group_i}"
        else:
            cur_figure_data = figure_data
            extra_name = ""

        if plot_solo:
            conds = [True, False]
        else:
            conds = [False]

        for qd_score_only in conds:
            first_algo = next(iter(cur_figure_data[next(iter(cur_figure_data))]))
            ablation_name = "$" + first_algo.split(" = ")[0][1:] + "$"

            plot_data = {
                "Environment": [],
                ablation_name: [],
                "Metric": [],
                "Score": [],
            }

            logger.info("Loading Plot Data")
            all_algos = OrderedDict()  # Set of all algorithms, ordered by listing.
            for env in cur_figure_data:
                cur_data = {
                    "Environment": [],
                    ablation_name: [],
                    "Metric": [],
                    "Score": [],
                }
                for algo in cur_figure_data[env]:
                    all_algos[algo] = None
                    for entry in cur_figure_data[env][algo]:
                        entry_metrics = entry["series_metrics"]

                        for metric in entry_metrics:
                            if qd_score_only:
                                if metric != "Coverage":
                                    continue
                            else:
                                if metric in METRIC_BLACKLIST:
                                    continue

                            cur_data["Environment"].append([env])
                            cur_data[ablation_name].append(
                                [
                                    # example: retrieve 0.01 from "$h = 0.01$"
                                    float(algo.split(" = ")[1][:-1])
                                ]
                            )
                            cur_data["Metric"].append([metric])
                            cur_data["Score"].append([entry_metrics[metric][-1]])

                # Reverse so that algorithms are ordered properly in terms of
                # layers.
                for d in plot_data:
                    plot_data[d].append(np.concatenate(cur_data[d]))

            # Flatten everything so that Seaborn understands it.
            for d in plot_data:
                plot_data[d] = np.concatenate(plot_data[d])

            logger.info("Generating Plot")
            with mpl_style_file(
                "simple_sans.mplstyle" if sans else "simple.mplstyle"
            ) as f:
                with plt.style.context(f):
                    if palette_name == "qualitative_colors":
                        colors = QUALITATIVE_COLORS
                    elif palette_name == "colorblind_reordered":
                        # Rearrange the color-blind template.
                        colors = COLORBLIND_REORDERED
                    else:
                        colors = sns.color_palette(palette_name)

                    # palette = dict(zip(all_algos, colors))
                    # markers = dict(zip(all_algos, itertools.cycle("oD^vXPps")))

                    # This takes a while since it has to generate the bootstrapped
                    # confidence intervals.
                    grid = sns.relplot(
                        data=plot_data,
                        x=ablation_name,
                        y="Score",
                        row="Metric",
                        col="Environment",
                        kind="line",
                        errorbar="se",
                        # Note that seaborn `markers` does not help here.
                        # Instead, we need to use the `marker` (note the lack of
                        # `s`) to specify that we want to mark the points.
                        # https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
                        marker="o",
                        markevery=1,
                        # Similar to `marker`, we can force the color here.
                        color=COLORBLIND_REORDERED[0],
                        dashes=False,
                        # Slightly taller when only plotting coverage.
                        height=1.2 * height if qd_score_only else height,
                        aspect=1.35 if qd_score_only else 1.61803,  # Golden ratio.
                        # aspect=1.35 if coverage_only else 1.5,
                        facet_kws={"sharey": False},
                        # palette=palette,
                        legend=False,
                        linewidth=1.25,
                    )

                    # Set titles to be the env name.
                    grid.set_titles("{col_name}")

                    # Turn off titles below top row (no need to repeat).
                    for ax in grid.axes[1:].ravel():
                        ax.set_title("")

                    # Set the labels along the left column to be the name of the
                    # metric.
                    left_col = list(cur_figure_data)[0]
                    for (row_val, col_val), ax in grid.axes_dict.items():
                        ax.set_axisbelow(True)
                        if col_val == left_col:
                            ax.set_ylabel(row_val, labelpad=10.0)
                        else:
                            ax.set_ylabel("")

                        if ablation_name == "$\\alpha$":
                            ax.set_xscale("symlog", linthresh=0.001)
                            ax.set_xlim([0, 1])
                            ax.set_xticks([0.0, 0.001, 0.01, 0.1, 1.0])
                        elif ablation_name == "$n_{empty}$":
                            ax.set_xscale("symlog", linthresh=10)
                            ax.set_xlim([0, 1000])
                            ax.set_xticks([0, 10, 100, 1000])

                        if (row_val, col_val) in PLOT_INFO:
                            d = PLOT_INFO[(row_val, col_val)]
                            if "ylim" in d:
                                ax.set_ylim(d["ylim"])
                            if "yticks" in d:
                                ax.set_yticks(d["yticks"])
                                if row_val == "Coverage":
                                    ax.set_yticklabels(
                                        [f"{int(100 * tick)}%" for tick in d["yticks"]]
                                    )

                    # Add legend and resize figure to fit it.
                    # grid.fig.legend(
                    #     *legend_info(all_algos, palette, markers),
                    #     bbox_to_anchor=[0.5, 1.0],
                    #     loc="upper center",
                    #     # Change // 1 to // 2 etc. for more rows.
                    #     ncol=(len(palette) + 1) // 4,
                    # )
                    # fig_width, fig_height = grid.fig.get_size_inches()
                    # legend_height = 0
                    # grid.fig.set_size_inches(fig_width, fig_height + legend_height)

                    # Save the figure.
                    grid.fig.tight_layout(rect=(0, 0, 1, 1))
                    # grid.fig.tight_layout(rect=(0, 0, 1, fig_height /
                    #                             (fig_height + legend_height)))
                    name = "ablation-qd-score" if qd_score_only else "ablation"
                    for extension in ["pdf", "png", "svg"]:
                        filename = (
                            output
                            / f"{name}{'-sans' if sans else ''}{extra_name}.{extension}"
                        )
                        logger.info("Saving {}", filename)
                        grid.fig.savefig(filename, dpi=300)

    logger.info("Done")


if __name__ == "__main__":
    fire.Fire()
