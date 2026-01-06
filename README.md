# Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces

This repository is the implementation of
[Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](https://arxiv.org/abs/2601.01082)
by Bryon Tjanaka, Henry Chen, Matthew C. Fontaine, and Stefanos Nikolaidis.

## Installation

```bash
uv sync

# For StyleGAN3.
git clone https://github.com/NVlabs/stylegan3.git
curl -L 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-ffhqu-256x256.pkl' -o 'stylegan3-t-ffhqu-256x256.pkl'
```

If running in the LSI (Hiker) domain, download the LHQ256 dataset from
[here](https://github.com/universome/alis/blob/master/lhq.md). Place the dataset
in the `lhq_256_root/` directory, such that the directory structure looks like
this:

```
lhq_256_root
└── lhq_256
    ├── 0000000.png
    ├── 0000001.png
    ├── 0000002.png
    ├── 0000003.png
    ...
```

### Generating Centroids

We have already generated centroids and stored them in this repo under
`src/cvt/centroids/`. However, new centroids can be generated as follows. Place
any generated files under `src/cvt/centroids/`, or modify the `centroids` path
in the configs in `config/algo/archive/`.

1. 10D, 20D, and 50D LP domains:

   ```bash
   # 10D
   uv run -m src.cvt.make_centroids \
       domain=flat_multi_100d \
       +centroid_cells=10000 \
       +centroid_file=src/cvt/centroids/flat_multi_100d.npy

   # 20D
   uv run -m src.cvt.make_centroids \
       domain=sphere_20_100d \
       +centroid_cells=10000 \
       +centroid_file=src/cvt/centroids/sphere_20_100d.npy

   # 50D
   uv run -m src.cvt.make_centroids \
       domain=sphere_50_100d \
       +centroid_cells=10000 \
       +centroid_file=src/cvt/centroids/sphere_50_100d.npy
   ```

1. TA (MNIST):
   ```bash
   uv run -m src.cvt.mnist_centroids
   ```
1. TA (F-MNIST):
   ```bash
   uv run -m src.cvt.fashion_mnist_centroids
   ```
1. LSI (Hiker): Make sure to download the LHQ256 dataset as described above.
   Then:
   ```bash
   uv run -m src.cvt.lhq_centroids
   ```

## Experiments

The following commands will run DMS and all baselines for the paper in all
domains. Note that we use the following names for the domains in the paper:

| Name in Paper      | Name in Code              |
| :----------------- | :------------------------ |
| 2D LP (Sphere)     | `sphere_100d`             |
| 10D LP (Sphere)    | `sphere_multi_100d`       |
| 20D LP (Sphere)    | `sphere_20_100d`          |
| 50D LP (Sphere)    | `sphere_50_100d`          |
| 2D LP (Rastrigin)  | `rastrigin_100d`          |
| 10D LP (Rastrigin) | `rastrigin_multi_100d`    |
| 2D LP (Flat)       | `flat_100d`               |
| 10D LP (Flat)      | `flat_multi_100d`         |
| Arm Repertoire     | `arm_100d`                |
| TA (MNIST)         | `triangles_mnist`         |
| TA (F-MNIST)       | `triangles_fashion_mnist` |
| LSI (Hiker)        | `lsi_face`                |

```bash
# DMS -- for the benchmark domains, we include commands to run all ablations in
# the paper, i.e., learning rate, restart rule, and number of empty points.
bash scripts/paper/dms_100d.sh sphere_100d lr
bash scripts/paper/dms_100d.sh sphere_100d restart_100
bash scripts/paper/dms_100d.sh sphere_100d n_empty
bash scripts/paper/dms_multi_100d.sh sphere_multi_100d lr
bash scripts/paper/dms_multi_100d.sh sphere_multi_100d restart_basic
bash scripts/paper/dms_multi_100d.sh sphere_multi_100d n_empty
bash scripts/paper/dms_20_100d.sh sphere_20_100d default
bash scripts/paper/dms_50_100d.sh sphere_50_100d default
bash scripts/paper/dms_100d.sh rastrigin_100d lr
bash scripts/paper/dms_100d.sh rastrigin_100d restart_100
bash scripts/paper/dms_100d.sh rastrigin_100d n_empty
bash scripts/paper/dms_multi_100d.sh rastrigin_multi_100d lr
bash scripts/paper/dms_multi_100d.sh rastrigin_multi_100d restart_basic
bash scripts/paper/dms_multi_100d.sh rastrigin_multi_100d n_empty
bash scripts/paper/dms_100d.sh flat_100d lr
bash scripts/paper/dms_100d.sh flat_100d restart_100
bash scripts/paper/dms_100d.sh flat_100d n_empty
bash scripts/paper/dms_multi_100d.sh flat_multi_100d lr
bash scripts/paper/dms_multi_100d.sh flat_multi_100d restart_basic
bash scripts/paper/dms_multi_100d.sh flat_multi_100d n_empty
bash scripts/paper/dms_100d.sh arm_100d lr
bash scripts/paper/dms_100d.sh arm_100d restart_100
bash scripts/paper/dms_100d.sh arm_100d n_empty
bash scripts/paper/dms_triangles_mnist.sh triangles_mnist
bash scripts/paper/dms_triangles_mnist.sh triangles_fashion_mnist
bash scripts/paper/dms_lsi_face.sh

# CMA-MAE
bash scripts/paper/cma_mae_100d.sh sphere_100d
bash scripts/paper/cma_mae_multi_100d.sh sphere_multi_100d
bash scripts/paper/cma_mae_20_100d.sh sphere_20_100d
bash scripts/paper/cma_mae_50_100d.sh sphere_50_100d
bash scripts/paper/cma_mae_100d.sh rastrigin_100d
bash scripts/paper/cma_mae_multi_100d.sh rastrigin_multi_100d
bash scripts/paper/cma_mae_100d.sh flat_100d
bash scripts/paper/cma_mae_multi_100d.sh flat_multi_100d
bash scripts/paper/cma_mae_100d.sh flat_100d
bash scripts/paper/cma_mae_triangles_mnist.sh triangles_mnist
bash scripts/paper/cma_mae_triangles_mnist.sh triangles_fashion_mnist
bash scripts/paper/cma_mae_lsi_face.sh

# DDS
bash scripts/paper/dds_100d.sh sphere_100d
bash scripts/paper/dds_multi_100d.sh sphere_multi_100d
bash scripts/paper/dds_20_100d.sh sphere_20_100d
bash scripts/paper/dds_50_100d.sh sphere_50_100d
bash scripts/paper/dds_100d.sh rastrigin_100d
bash scripts/paper/dds_multi_100d.sh rastrigin_multi_100d
bash scripts/paper/dds_100d.sh flat_100d
bash scripts/paper/dds_multi_100d.sh flat_multi_100d
bash scripts/paper/dds_100d.sh arm_100d

# MAP-Elites (line)
bash scripts/paper/map_elites_line_100d.sh sphere_100d
bash scripts/paper/map_elites_line_multi_100d.sh sphere_multi_100d
bash scripts/paper/map_elites_line_20_100d.sh sphere_20_100d
bash scripts/paper/map_elites_line_50_100d.sh sphere_50_100d
bash scripts/paper/map_elites_line_100d.sh rastrigin_100d
bash scripts/paper/map_elites_line_multi_100d.sh rastrigin_multi_100d
bash scripts/paper/map_elites_line_100d.sh flat_100d
bash scripts/paper/map_elites_line_multi_100d.sh flat_multi_100d
bash scripts/paper/map_elites_line_100d.sh arm_100d
bash scripts/paper/map_elites_line_triangles_mnist.sh triangles_mnist
bash scripts/paper/map_elites_line_triangles_mnist.sh triangles_fashion_mnist
bash scripts/paper/map_elites_line_lsi_face.sh

# MAP-Elites
bash scripts/paper/map_elites_100d.sh sphere_100d
bash scripts/paper/map_elites_multi_100d.sh sphere_multi_100d
bash scripts/paper/map_elites_20_100d.sh sphere_20_100d
bash scripts/paper/map_elites_50_100d.sh sphere_50_100d
bash scripts/paper/map_elites_100d.sh rastrigin_100d
bash scripts/paper/map_elites_multi_100d.sh rastrigin_multi_100d
bash scripts/paper/map_elites_100d.sh flat_100d
bash scripts/paper/map_elites_multi_100d.sh flat_multi_100d
bash scripts/paper/map_elites_100d.sh arm_100d
bash scripts/paper/map_elites_triangles_mnist.sh triangles_mnist
bash scripts/paper/map_elites_triangles_mnist.sh triangles_fashion_mnist
bash scripts/paper/map_elites_lsi_face.sh
```

## Analysis

Running the experiments above will produce logging directories under the `logs/`
directory, e.g., `logs/dms_arm_100d_lr/`. If desired, move the results into a
new `results/` directory, and in that directory, create a `manifest.yaml`
listing out the directories -- refer to `analysis/src/analysis/figures.py` for
the format of this manifest.

Follow the instructions in `analysis/README.md` for generating plots with the
given manifest. Note that the analysis code uses different dependencies from the
main repo.

## Citation

```bibtex
@misc{tjanaka2026discountmodelsearchquality,
      title={Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces},
      author={Bryon Tjanaka and Henry Chen and Matthew C. Fontaine and Stefanos Nikolaidis},
      year={2026},
      eprint={2601.01082},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.01082},
}
```
