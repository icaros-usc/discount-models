#!/bin/bash
DOMAIN="$1"
NAME="cma_mae_${DOMAIN}"

# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun hydra/launcher=joblib \
  hydra.launcher.n_jobs=1 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_restart=\${algo._emitter_dict.e1.type.args.restart_rule}_lr=\${algo.archive.args.learning_rate}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=cma_mae \
  algo/archive=cvt_mnist_mae \
  algo/result_archive=cvt_mnist \
  itrs=10000 \
  seed=$(uv run scripts/seeds.py 5)
