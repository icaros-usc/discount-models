#!/bin/bash
DOMAIN="$1"
NAME="map_elites_line_${DOMAIN}"

# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun hydra/launcher=joblib \
  hydra.launcher.n_jobs=40 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=map_elites_line \
  algo/archive=cvt_multi_100d \
  itrs=10000 \
  seed=$(uv run scripts/seeds.py 20)
