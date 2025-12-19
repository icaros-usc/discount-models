"""Misc visualization tools."""

import numpy as np
from matplotlib import pyplot as plt

from src.evaluation import plot_discount_archive


def visualize(solutions, link_lengths, objectives, ax, context):
    lim = 1.05 * np.sum(link_lengths)  # Add a bit of a border.
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    for i, solution in enumerate(solutions):
        ax.set_title(f"Objective: {objectives[i]}")
        pos = np.array([0, 0])  # Starting position of the next joint.
        cum_thetas = np.cumsum(solution)
        for link_length, cum_theta in zip(link_lengths, cum_thetas, strict=True):
            # Calculate the end of this link.
            next_pos = pos + link_length * np.array(
                [np.cos(cum_theta), np.sin(cum_theta)]
            )
            ax.plot(
                [pos[0], next_pos[0]],
                [pos[1], next_pos[1]],
                "-ko",
                ms=0.5,
                linewidth=0.5,
            )
            pos = next_pos

        # Add points for the start and end positions and conditioned positions
        ax.plot(context[0], context[1], "cx", ms=10)
        ax.plot(0, 0, "ro", ms=2)
        final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"
        ax.plot(pos[0], pos[1], "go", ms=2, label=final_label)

    plt.savefig("./visualizations")


def visualize_single(solution, link_lengths, objectives, ax, context):
    lim = 1.05 * np.sum(link_lengths)  # Add a bit of a border.
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_title(f"Objective: {objectives[0]}")
    pos = np.array([0, 0])  # Starting position of the next joint.
    cum_thetas = np.cumsum(solution)
    for link_length, cum_theta in zip(link_lengths, cum_thetas, strict=True):
        # Calculate the end of this link.
        next_pos = pos + link_length * np.array([np.cos(cum_theta), np.sin(cum_theta)])
        ax.plot(
            [pos[0], next_pos[0]], [pos[1], next_pos[1]], "-ko", ms=0.5, linewidth=0.5
        )
        pos = next_pos

    # Add points for the start and end positions and conditioned positions
    ax.plot(context[0], context[1], "cx", ms=10)
    ax.plot(0, 0, "ro", ms=2)
    final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"
    ax.plot(pos[0], pos[1], "go", ms=2, label=final_label)


def visualize_discount_points(discount_train_info, discount_archive, ax, domain_cfg):
    if discount_train_info is None:
        new_measures = np.empty((0, 2))
        empty_measures = np.empty((0, 2))
        non_empty_measures = np.empty((0, 2))
        same_measures = np.empty((0, 2))
    else:
        new_measures = discount_train_info["new_measures"]
        empty_measures = discount_train_info["empty_measures"]
        non_empty_measures = discount_train_info["non_empty_measures"]
        same_measures = discount_train_info["same_measures"]

    plot_discount_archive(discount_archive, ax, domain_cfg)

    ax.scatter(
        new_measures[:, 0],
        new_measures[:, 1],
        s=50,
        c="cornflowerblue",
        marker=".",
        label="Emitter Samples",
        alpha=0.7,
    )
    ax.scatter(
        empty_measures[:, 0],
        empty_measures[:, 1],
        s=40,
        c="yellow",
        marker="^",
        label="Empty Points",
        alpha=0.7,
    )
    ax.scatter(
        non_empty_measures[:, 0],
        non_empty_measures[:, 1],
        s=36,
        c="red",
        marker="^",
        label="Non Empty Points",
        alpha=0.7,
    )
    ax.scatter(
        same_measures[:, 0],
        same_measures[:, 1],
        s=40,
        c="lime",
        marker="x",
        label="Same",
        alpha=0.7,
    )

    ax.legend(loc="lower left", bbox_to_anchor=(1.25, 0.0), borderaxespad=0.0)
    ax.set_title("Discount Points")


def visualize_discount_points_2(discount_train_info, discount_archive, ax, domain_cfg):
    """Unlike above, does not plot non-empty or same points."""
    if discount_train_info is None:
        new_measures = np.empty((0, 2))
        empty_measures = np.empty((0, 2))
    else:
        new_measures = discount_train_info["new_measures"]
        empty_measures = discount_train_info["empty_measures"]

    plot_discount_archive(discount_archive, ax, domain_cfg)

    ax.scatter(
        new_measures[:, 0],
        new_measures[:, 1],
        s=10,
        c="cornflowerblue",
        marker=".",
        label="Emitter Samples",
        alpha=0.7,
    )
    ax.scatter(
        empty_measures[:, 0],
        empty_measures[:, 1],
        s=8,
        c="yellow",
        marker="^",
        label="Empty Points",
        alpha=0.7,
    )

    ax.legend(loc="lower left", bbox_to_anchor=(1.5, 0.0), borderaxespad=0.0)
    ax.set_title("Discount Points")
