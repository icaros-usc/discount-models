#!/bin/bash
DOMAIN="$1"
NAME="cma_mae_${DOMAIN}"

# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun hydra/launcher=joblib \
  hydra.launcher.n_jobs=40 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_lr=\${algo.archive.args.learning_rate}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=cma_mae \
  algo/archive=grid_100_mae \
  algo/result_archive=grid_100 \
  itrs=10000 \
  seed=$(uv run scripts/seeds.py 20)
