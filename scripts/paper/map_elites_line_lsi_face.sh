#!/bin/bash
DOMAIN="lsi_face"
NAME="map_elites_line_${DOMAIN}"

# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun hydra/launcher=joblib \
  hydra.launcher.n_jobs=1 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=map_elites_line \
  algo/archive=cvt_lhq_10k \
  itrs=10000 \
  domain.config.obj_prompt="A photo of the face of a hiker." \
  domain.config.space=w \
  domain.config.solution_dim=512 \
  domain.config.add_centroid_dist=True \
  seed=$(uv run scripts/seeds.py 5)
