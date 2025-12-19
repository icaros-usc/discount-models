#!/bin/bash
# Runs analysis for the main results.
#
# Usage:
# 1. Create `manifest.yaml` in the result directory.
# 2. Activate the analysis environment.
# 3. bash scripts/analysis.sh MANIFEST_FILE
#
# Example:
#     (after creating and editing ../results/paper/manifest.yaml)
#     conda activate ./env
#     bash scripts/analysis.sh ../results/paper/manifest.yaml
#     ls ../results/paper/manifest/

MANIFEST="$1"
DIR="${MANIFEST%.*}"
FIGURE_DATA="$DIR/figure_data.json"
COMPARISON_OUTPUT="$DIR/comparison/"
TABLE_OUTPUT="$DIR/results_single_table.tex"
STATS_OUTPUT="$DIR/stats_tests/"

mkdir -p $DIR
cp table.tex $DIR

# Creates the manifest file.
uv run -m src.analysis.figures collect "$MANIFEST" --output "$FIGURE_DATA"

# Full plots, in a single row in a single file.
uv run -m src.analysis.figures comparison --figure_data "$FIGURE_DATA" --output "$COMPARISON_OUTPUT"

# Splits the plots across a couple files so they can fit in the paper; used in
# the appendix.
# NOTE: This arg is --groups, not --table-groups
uv run -m src.analysis.figures comparison \
  --figure_data "$FIGURE_DATA" \
  --output "$DIR/comparison_split" \
  --plot-solo False \
  --groups "[['2D LP (Sphere)', '10D LP (Sphere)', '20D LP (Sphere)', '50D LP (Sphere)'], ['2D LP (Rastrigin)', '10D LP (Rastrigin)', '2D LP (Flat)', '10D LP (Flat)'], ['Arm Repertoire', 'TA (MNIST)', 'TA (F-MNIST)', 'LSI (Hiker)']]"

# Splits the results across multiple tables so that they fit better; used in the
# appendix.
uv run -m src.analysis.figures single_table \
  --figure_data "$FIGURE_DATA" \
  --iteration=10000 \
  --output "$DIR/results_split_table.tex" \
  --show_std True \
  --table-groups "[['2D LP (Sphere)', '10D LP (Sphere)', '20D LP (Sphere)', '50D LP (Sphere)'], ['2D LP (Rastrigin)', '10D LP (Rastrigin)', '2D LP (Flat)', '10D LP (Flat)'], ['Arm Repertoire', 'TA (MNIST)', 'TA (F-MNIST)', 'LSI (Hiker)']]"

# All results in a single table, but without showing std.
# uv run -m src.analysis.figures single_table \
#   --figure_data "$FIGURE_DATA" \
#   --iteration=10000 \
#   --output "$TABLE_OUTPUT" \
#   --show_std False
# (cd $DIR && pdflatex "table.tex")

# Old table command.
# uv run -m src.analysis.figures table \
#   --figure_data "$FIGURE_DATA" \
#   --output "$DIR/results_table.tex" \
#   --transpose False

# Splits the results across a couple tables and only shows the mean; used in the
# main paper.
uv run -m src.analysis.figures single_table \
  --figure_data "$FIGURE_DATA" \
  --iteration=10000 \
  --output "$DIR/results_table_main_paper.tex" \
  --show_std False \
  --table-groups "[['2D LP (Sphere)', '10D LP (Sphere)', '20D LP (Sphere)', '50D LP (Sphere)'], ['2D LP (Rastrigin)', '10D LP (Rastrigin)', '2D LP (Flat)', '10D LP (Flat)'], ['Arm Repertoire', 'TA (MNIST)', 'TA (F-MNIST)', 'LSI (Hiker)']]"

# Results in Markdown format; each domain gets its own row. Can be used to
# generate results for a README file.
uv run -m src.analysis.figures single_table \
  --figure_data "$FIGURE_DATA" \
  --iteration=10000 \
  --output "$DIR/results_by_domain.md" \
  --output_format "markdown" \
  --show_std True \
  --table-groups "[['2D LP (Sphere)'], ['10D LP (Sphere)'], ['20D LP (Sphere)'],
  ['50D LP (Sphere)'], ['2D LP (Rastrigin)'], ['10D LP (Rastrigin)'], ['2D LP (Flat)'], ['10D LP (Flat)'], ['Arm Repertoire'], ['TA (MNIST)'], ['TA (F-MNIST)'], ['LSI (Hiker)']]"

# Statistical tests.
uv run -m src.analysis.figures tests \
  --figure_data "$FIGURE_DATA" \
  --output "$STATS_OUTPUT" \
  --table-groups "[['2D LP (Sphere)', '10D LP (Sphere)', '20D LP (Sphere)', '50D LP (Sphere)'], ['2D LP (Rastrigin)', '10D LP (Rastrigin)', '2D LP (Flat)', '10D LP (Flat)'], ['Arm Repertoire', 'TA (MNIST)', 'TA (F-MNIST)', 'LSI (Hiker)']]"
