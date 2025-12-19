#!/bin/bash
DOMAIN="lsi_face"
NAME="cma_mae_${DOMAIN}"

# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun hydra/launcher=joblib \
  hydra.launcher.n_jobs=1 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_lr=\${algo.archive.args.learning_rate}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=cma_mae \
  algo/archive=cvt_lhq_10k_mae \
  algo/result_archive=cvt_lhq_10k \
  algo.archive.args.learning_rate=0.01 \
  algo._emitter_dict.e1.num=1 \
  itrs=10000 \
  domain.config.obj_prompt="A photo of the face of a hiker." \
  domain.config.space=w \
  domain.config.solution_dim=512 \
  domain.config.add_centroid_dist=True \
  seed=$(uv run scripts/seeds.py 5)
