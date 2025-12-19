"""QD algorithms."""

import functools
import logging
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from hydra.core.hydra_config import HydraConfig
from logdir import LogDir
from omegaconf import DictConfig
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap
from tqdm.contrib.logging import logging_redirect_tqdm

from src.domains.domain_base import DomainBase
from src.evaluation import compute_centers, make_discount_archive, plot_discount_archive
from src.utils.code_timer import CodeTimer
from src.utils.hydra_utils import define_resolvers
from src.utils.logging import setup_logdir_from_hydra
from src.utils.metric_logger import MetricLogger
from src.visualize import visualize_discount_points_2

# Use when debugging warnings.
#  import src.utils.warn_traceback  # pylint: disable = unused-import

log = logging.getLogger(__name__)


def build_qd_algo(
    cfg: DictConfig,
    domain_module: DomainBase,
    device: torch.device,
) -> Scheduler:
    """Creates a scheduler based on the algorithm configuration.

    Returns:
        A scheduler for running the algorithm.
    """
    # Create result archive.
    result_archive = None
    if cfg.algo.get("result_archive"):
        result_archive = hydra.utils.instantiate(
            cfg.algo.result_archive.args, seed=cfg.seed
        )

    if cfg.algo.get("discount_model"):
        # Create discount archive with discount model.
        discount_model = hydra.utils.instantiate(
            cfg.algo.discount_model,
            seed=None if cfg.seed is None else cfg.seed + 420,
            device=device,
            _recursive_=False,
        )
        archive = hydra.utils.instantiate(
            cfg.algo.archive.args,
            seed=cfg.seed,
            discount_model=discount_model,
            device=device,
            result_archive=result_archive,
        )
    else:
        # Create regular archive.
        archive = hydra.utils.instantiate(cfg.algo.archive.args, seed=cfg.seed)

    # Usually, emitters take in the archive. However, it may sometimes be necessary to
    # take in the result_archive, such as in DDS.
    archive_for_emitter = (
        result_archive if cfg.algo.get("pass_result_archive_to_emitters") else archive
    )

    # Create emitters. Each emitter needs a different seed so that they do not
    # all do the same thing, hence we use a SeedSequence to generate seeds.
    seed_sequence = np.random.SeedSequence(cfg.seed)
    emitters = []
    for e in cfg.algo.emitters:
        emitters += [
            hydra.utils.instantiate(
                e.type.args,
                archive=archive_for_emitter,
                x0=domain_module.initial_solution(),
                seed=s,
            )
            for s in seed_sequence.spawn(e.num)
        ]

    # Create scheduler.
    scheduler = hydra.utils.instantiate(
        cfg.algo.scheduler.args,
        archive=archive,
        emitters=emitters,
        result_archive=result_archive,
    )

    log.info(
        f"Created {scheduler.__class__.__name__} for "
        f"{HydraConfig.get().runtime.choices.algo}"
    )

    return scheduler


def make_plots_full(
    scheduler: Scheduler,
    cfg: DictConfig,
    domain_module: DomainBase,
    logdir: LogDir,
    name: str | int,
    filetype: str = "png",
    discount_train_info: dict | None = None,
) -> None:
    """General plotting code."""
    filename = f"{name:06}" if isinstance(name, int) else name

    # Plot the archives for domains with 2D measure spaces. We currently just support
    # GridArchive.
    if cfg.domain.config.measure_dim == 2:
        # This is a rough condition that captures when histogram discount functions are
        # being used.
        histogram_discount = isinstance(scheduler.archive, GridArchive) and (
            scheduler.archive is not scheduler.result_archive
        )

        if not isinstance(scheduler.result_archive, GridArchive):
            raise ValueError("Only plots GridArchive for now.")
        elif cfg.algo.get("discount_model"):
            # In this case, we plot the result archive, discount model, and discount
            # model with points.
            ncols = 3
            figwidth = 5 * (ncols - 1) + 6
        elif histogram_discount:
            # In this case, we plot the result archive and histogram discount function.
            ncols = 2
            figwidth = 5 * ncols
        else:
            # Finally, here we just plot the result archive on its own.
            ncols = 1
            figwidth = 5 * ncols

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(figwidth, 4))
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        axs[0].set_title("Result Archive")
        grid_archive_heatmap(
            scheduler.result_archive,
            ax=axs[0],
            vmin=cfg.domain.config.obj_low,
            vmax=cfg.domain.config.obj_high,
            rasterized=True,
        )

        if cfg.algo.get("discount_model"):
            discount_archive = make_discount_archive(
                scheduler.archive.discount_model, cfg
            )
            plot_discount_archive(discount_archive, axs[1], domain_module.config)
            visualize_discount_points_2(
                discount_train_info, discount_archive, axs[2], domain_module.config
            )
        elif histogram_discount:
            # Make archive that shows the true discount function.
            true_discount_archive = GridArchive(
                solution_dim=0,
                dims=cfg.algo.archive.args.dims,
                ranges=cfg.algo.archive.args.ranges,
            )
            cell_centers = compute_centers(true_discount_archive)

            # Default discount of threshold_min.
            if "threshold_min" in cfg.algo.archive.args:
                true_discount_archive.add(
                    np.empty((len(cell_centers), 0)),
                    np.full(len(cell_centers), cfg.algo.archive.args.threshold_min),
                    cell_centers,
                )

            # Add in thresholds from archive.
            data = scheduler.archive.data()
            true_discount_archive.add(
                np.empty((len(data["solution"]), 0)),
                data["threshold"],
                data["measures"],
            )

            axs[1].set_title("Discount Function")
            grid_archive_heatmap(
                true_discount_archive,
                ax=axs[1],
                vmin=cfg.domain.config.obj_low,
                vmax=cfg.domain.config.obj_high,
                rasterized=True,
            )
        else:
            # Nothing to do here.
            pass

        for ax in axs:
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(logdir.pfile(f"heatmaps_{filename}.{filetype}"), dpi=300)
        plt.close(fig)

    # Special plotting for other domains.

    if (
        HydraConfig.get().runtime.choices.domain == "triangles_mnist"
        and len(scheduler.result_archive) > 0
    ):
        imgs = scheduler.result_archive.sample_elites(
            min(len(scheduler.result_archive), 100),
            replace=False,
        )["measures"]
        from src.domains.triangles import render_mnist_batch_pil

        pil_image = render_mnist_batch_pil(imgs.reshape(-1, 28, 28))
        pil_image.save(logdir.pfile(f"mnist_samples_{filename}.png"))

    if (
        HydraConfig.get().runtime.choices.domain
        in ["triangles_afhq", "triangles_afhq_l2"]
        and len(scheduler.result_archive) > 0
    ):
        from src.domains.triangles import afhq_pil_image

        sample = scheduler.result_archive.sample_elites(
            min(len(scheduler.result_archive), 100),
            replace=False,
        )
        pil_image = afhq_pil_image(sample["solution"], sample["index"], 128)
        pil_image.save(logdir.pfile(f"afhq_samples_{filename}.png"))

    if (
        HydraConfig.get().runtime.choices.domain == "lsi_face"
        and len(scheduler.result_archive) > 0
    ):
        from src.domains.lsi_face import human_landscape_face

        sample = scheduler.result_archive.sample_elites(
            min(len(scheduler.result_archive), 40),
            replace=False,
        )
        pil_image = human_landscape_face(
            sample["solution"], domain_module.classifier, sample["index"]
        )
        pil_image.save(logdir.pfile(f"face_samples_{filename}.png"))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs experiment."""
    define_resolvers()

    logdir = setup_logdir_from_hydra()
    log.info(f"Logging directory: {logdir.logdir}")

    ## COMPONENT INITIALIZATION ##

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_module = hydra.utils.instantiate(
        cfg.domain,
        seed=None if cfg.seed is None else cfg.seed + 42,
        device=device,
    )
    scheduler = build_qd_algo(cfg, domain_module, device)

    if cfg.algo.get("discount_model"):
        discount_train_info = scheduler.archive.init_discount_model()
        log.info(f"Losses from Discount Init: {discount_train_info['losses']}")
    else:
        discount_train_info = None

    make_plots = functools.partial(
        make_plots_full, scheduler, cfg, domain_module, logdir
    )
    make_plots(0, discount_train_info=discount_train_info)

    # Set up timer and metrics.
    timer = CodeTimer(
        [
            "All",  # Everything in the iteration (except metrics, which are negligible).
            "Algorithm",  # Algorithm only -- excludes logging and saving, includes evals.
            "Internal",  # Algorithm time excluding evals.
            "Domain Evaluation",  # Evaluation in the domain for main solutions.
            "Discount Model Training",  # Time spent training the discount model.
        ]
    )
    metric_list = [
        ("Evaluations", True, 0),
        ("QD Score", True, 0.0),
        ("Archive Coverage", True, 0.0),
        ("Objective Max", False),
        ("Restarts", True, 0),
        ("Unique Cells", False),
        ("Solutions Per Cell", False),
        *(
            [
                ("Final Discount Loss", False),
                ("Num Empty", False),
                ("Discount Epochs", False),
            ]
            if cfg.algo.get("discount_model")
            else []
        ),
        *timer.metric_list(),
    ]
    metrics = MetricLogger(metric_list)

    ## EXECUTION LOOP ##

    with logging_redirect_tqdm():
        for itr in tqdm.trange(1, cfg.itrs + 1):
            timer.start(["Algorithm", "All"])

            ## DQD ASK-TELL ##

            if cfg.algo.get("dqd"):
                solutions = scheduler.ask_dqd()
                objectives, measures, info = domain_module.evaluate(
                    solutions, grad=True
                )
                jacobian = np.concatenate(
                    (
                        info["objective_grads"][:, None, :],
                        info["measure_grads"],
                    ),
                    axis=1,
                )

                fields = {}
                if cfg.domain.config.get("is_mnist"):
                    fields["mnist_img"] = info["mnist_img"]

                scheduler.tell_dqd(objectives, measures, jacobian, **fields)

            ## PRIMARY ASK-TELL ##

            solutions = scheduler.ask()

            timer.start("Domain Evaluation")
            objectives, measures, info = domain_module.evaluate(solutions)

            # Handle any special cases for evaluations.
            fields = {}
            if cfg.domain.config.get("is_mnist"):
                fields["mnist_img"] = info["mnist_img"]

            timer.end("Domain Evaluation")

            scheduler.tell(objectives, measures, **fields)

            timer.start("Discount Model Training")
            if (
                cfg.algo.get("discount_model")
                and itr % scheduler.archive.train_freq == 0
            ):
                discount_train_info = scheduler.archive.train_discount_model()
            else:
                discount_train_info = None
            timer.end("Discount Model Training")

            timer.end("Algorithm")

            # Include plotting in the timing, in case it takes a while.
            if itr % cfg.log_freq == 0 or itr == cfg.itrs:
                make_plots(itr, discount_train_info=discount_train_info)

            timer.end("All")

            ## METRICS AND EVALUATION ##

            # Update time.
            timer.itr_time["Internal"] = (
                timer.itr_time["Algorithm"] - timer.itr_time["Domain Evaluation"]
            )
            timer.calc_totals()
            timer_dict = timer.metrics_dict()
            timer.clear()

            # Update metrics.
            metrics.start_itr()
            stats = scheduler.result_archive.stats
            total_restarts = sum(
                e.restarts for e in scheduler.emitters if hasattr(e, "restarts")
            )
            indices = scheduler.result_archive.index_of(measures)
            unique_indices = len(set(indices))
            metrics_dict = {
                "Evaluations": metrics.get_last("Evaluations") + len(solutions),
                # Convert stats to Python objects since they are 0-D np arrays.
                "QD Score": stats.qd_score.item(),
                "Archive Coverage": stats.coverage.item(),
                "Objective Max": stats.obj_max.item(),
                "Restarts": total_restarts,
                "Unique Cells": unique_indices,
                "Solutions Per Cell": len(indices) / unique_indices,
            }
            if cfg.algo.get("discount_model"):
                metrics_dict.update(
                    {
                        "Final Discount Loss": discount_train_info["losses"][-1],
                        "Num Empty": discount_train_info["n_empty"],
                        "Discount Epochs": discount_train_info["epochs"],
                    }
                )
            metrics_dict.update(timer_dict)
            metrics.add_dict(metrics_dict)
            metrics.end_itr()

            if itr % cfg.log_freq == 0 or itr == cfg.itrs:
                log.info(
                    f"{itr:5d} | "
                    f"Cov: {stats.coverage * 100:.3f}%  "
                    f"Size: {stats.num_elites:5d}  "
                    f"QD: {stats.qd_score:.3f}"
                )

    # Plot final archives as a PDF (not a PNG like during the run).
    make_plots("final", "pdf", discount_train_info=discount_train_info)

    # Save and plot metrics.
    metrics.to_json(logdir.file("metrics.json"))
    metrics.plot_graphic(logdir.file("metrics_final.svg"))

    # Save the final archive as numpy.
    np.savez_compressed(logdir.pfile("archive.npz"), **scheduler.result_archive.data())

    # Save scheduler.
    with logdir.pfile("scheduler.pkl", touch=True).open("wb") as file:
        pickle.dump(scheduler, file)

    log.info("Summary:")
    for name, val in metrics.summary().items():
        log.info(f"- {name}:\t{val}")
    log.info(f"Logging directory: {logdir.logdir}")
    log.info("Done")


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
