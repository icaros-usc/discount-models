# Analysis

Code for creating figures and such.

## Installation

Note that this code runs with dependencies that are separated from those in the
main repo, which helps make it easier to move this directory around between
projects.

```bash
uv sync
```

## Running Analysis

First run the experiments described in the README at the root of this
repository. Then, create a `manifest.yaml` file following the instructions at
the top of `src/analysis/figures.py`. Finally, the following scripts can be used
to analyze and plot the various results. We have included examples of calling
the commands, assuming the results and manifest files are located in a directory
called `../results/paper`.

```bash
bash scripts/analysis.sh ../results/paper/manifest.yaml
bash scripts/heatmaps.sh ../results/paper/manifest.yaml
```
