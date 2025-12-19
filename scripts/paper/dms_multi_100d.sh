#!/bin/bash
DOMAIN="$1"
MODE="$2"
NAME="dms_${DOMAIN}_${MODE}"

case "$MODE" in
  # Default learning rate specified in model_based.yaml
  # Default restart rule specified in es_imp.yaml
  # Default n_empty specified in discount_model/mlp.yaml
  default)
    # Run without any extra args.
    EXTRA_ARGS=""
    ;;
  lr)
    # Run `lr` first to get all the results.
    EXTRA_ARGS="algo.archive.args.learning_rate=0.0,0.001,0.01,0.1,1.0"
    ;;
  restart_basic)
    # Choose either restart_basic or restart_100 depending on which is default
    # in the given domain (choose the one that is NOT used in the domain).
    EXTRA_ARGS="algo._emitter_dict.e1.type.args.restart_rule=basic"
    ;;
  restart_100)
    EXTRA_ARGS="algo._emitter_dict.e1.type.args.restart_rule=100"
    ;;
  n_empty)
    # 100 is already default for all domains and included in the `lr` run.
    EXTRA_ARGS="algo.archive.args.empty_points=0,10,1000"
    ;;
  *)
    echo "Unknown mode"
    exit 1
    ;;
esac


# TQDM is disabled to hopefully help speed up the jobs by preventing them from
# competing for stdout.
TQDM_DISABLE=1 uv run -m src.qd --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=20 \
  hydra.job.name="$NAME" \
  hydra.sweep.dir="'./logs/$NAME/\${now:%Y-%m-%d_%H-%M-%S}'" \
  hydra.sweep.subdir="'\${hydra:job.num}__\${hydra:runtime.choices.domain}_lr=\${algo.archive.args.learning_rate}_restart=\${algo._emitter_dict.e1.type.args.restart_rule}_empty=\${algo.archive.args.empty_points}_seed=\${seed}'" \
  domain="${DOMAIN}" \
  algo=dms \
  algo/archive=discount \
  algo/result_archive=cvt_multi_100d \
  itrs=10000 \
  $EXTRA_ARGS \
  seed=$(uv run scripts/seeds.py 20)
