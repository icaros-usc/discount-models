"""Provides linear projection domains.

Adapted from:
https://github.com/icaros-usc/cma_mae/blob/main/experiments/lin_proj/lin_proj.py
"""
import numpy as np

from src.domains.domain_base import DomainBase, EvaluateTorchMixin


class Sphere(EvaluateTorchMixin, DomainBase):
    """Sphere linear projection domain."""

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    def evaluate(self, solutions, grad=False):
        """Sphere function evaluation and measures for a batch of solutions.

        Args:
            solutions (np.ndarray): (batch_size, dim) batch of solutions.
        Returns:
            objectives (np.ndarray): (batch_size,) batch of objectives.
            measures (np.ndarray): (batch_size, 2) batch of measures.
            info (dict): If `grad` is passed in, it contains:
                - objective_grads (np.ndarray): (batch_size, solution_dim) batch
                  of objective gradients.
                - measure_grads (np.ndarray): (batch_size, 2, solution_dim)
                  batch of measure gradients.
        """
        dim = solutions.shape[1]

        # Shift the Sphere function so that the optimal value is at x_i = 2.048.
        sphere_shift = 5.12 * 0.4

        # Normalize the objective to the range [0, 1] where 1 is optimal.
        best_obj = 0.0
        worst_obj = (-5.12 - sphere_shift)**2 * dim
        raw_obj = np.sum(np.square(solutions - sphere_shift), axis=1)
        objectives = (raw_obj - worst_obj) / (best_obj - worst_obj)  # * XXX
        # Note: We can multiply `objectives` by X to scale it to [0, X].

        # Calculate measures.
        clipped = solutions.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]

        splits = np.array_split(clipped, self.config.measure_dim, axis=1)
        measures = np.sum(splits, axis=-1).T

        if grad:
            # Compute gradient of the objective.
            objective_grads = -2 * (solutions - sphere_shift)

            # Compute gradient of the measures.
            derivatives = np.ones(solutions.shape)
            derivatives[clip_mask] = -5.12 / np.square(solutions[clip_mask])

            masks = np.zeros((self.config.measure_dim, solutions.shape[1]))
            start = 0
            for i, block in enumerate(splits):
                block_dim = block.shape[1]
                masks[i, start:start + block_dim] = 1.0
                start += block_dim

            # Repeat along the measure dimension. Shape should now be
            # (batch_size, measure_dim, solution_dim).
            derivatives = np.repeat(derivatives[:, None, :],
                                    self.config.measure_dim,
                                    axis=1)
            measure_grads = derivatives * masks

            return (
                objectives,
                measures,
                {
                    "objective_grads": objective_grads,
                    "measure_grads": measure_grads,
                },
            )

        else:
            return objectives, measures, {}


class Rastrigin(EvaluateTorchMixin, DomainBase):
    """Rastrigin linear projection domain."""

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    def evaluate(self, solutions, grad=False):
        """Rastrigin function evaluation and measures for a batch of solutions.

        Args:
            solutions (np.ndarray): (batch_size, dim) batch of solutions.
        Returns:
            objectives (np.ndarray): (batch_size,) batch of objectives.
            measures (np.ndarray): (batch_size, 2) batch of measures.
            info (dict): If `grad` is passed in, it contains:
                - objective_grads (np.ndarray): (batch_size, solution_dim) batch
                  of objective gradients.
                - measure_grads (np.ndarray): (batch_size, 2, solution_dim)
                  batch of measure gradients.
        """
        A = 10.0  # pylint: disable = invalid-name
        dim = solutions.shape[1]

        # Shift the Rastrigin function so that the optimal value is at x_i = 2.048.
        target_shift = 5.12 * 0.4

        best_obj = np.zeros(len(solutions))
        displacement = -5.12 * np.ones(solutions.shape) - target_shift
        sum_terms = np.square(displacement) - A * np.cos(
            2 * np.pi * displacement)
        worst_obj = 10 * dim + np.sum(sum_terms, axis=1)

        displacement = solutions - target_shift
        sum_terms = np.square(displacement) - A * np.cos(
            2 * np.pi * displacement)
        raw_obj = 10 * dim + np.sum(sum_terms, axis=1)

        # Normalize the objective to the range [0, 1] where 1 is optimal.
        # Approximate 0 by the bottom-left corner.
        objectives = (raw_obj - worst_obj) / (best_obj - worst_obj)  # * XXX
        # Note: We can multiply `objectives` by X to scale it to [0, X].

        # Calculate measures.
        clipped = solutions.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]

        splits = np.array_split(clipped, self.config.measure_dim, axis=1)
        measures = np.sum(splits, axis=-1).T

        if grad:
            # Compute gradient of the objective.
            objective_grads = -(2 * displacement + 2 * np.pi * A *
                                np.sin(2 * np.pi * displacement))

            # Compute gradient of the measures.
            derivatives = np.ones(solutions.shape)
            derivatives[clip_mask] = -5.12 / np.square(solutions[clip_mask])

            masks = np.zeros((self.config.measure_dim, solutions.shape[1]))
            start = 0
            for i, block in enumerate(splits):
                block_dim = block.shape[1]
                masks[i, start:start + block_dim] = 1.0
                start += block_dim

            # Repeat along the measure dimension. Shape should now be
            # (batch_size, measure_dim, solution_dim).
            derivatives = np.repeat(derivatives[:, None, :],
                                    self.config.measure_dim,
                                    axis=1)
            measure_grads = derivatives * masks

            return (
                objectives,
                measures,
                {
                    "objective_grads": objective_grads,
                    "measure_grads": measure_grads,
                },
            )

        else:
            return objectives, measures, {}


class FlatLinProj(EvaluateTorchMixin, DomainBase):
    """Flat linear projection domain that can handle multiple dimensions.

    The objective is 1.0 everywhere.

    Described in Density Descent Search paper.
    """

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    def evaluate(self, solutions, grad=False):
        objectives = np.ones(len(solutions), dtype=solutions.dtype)

        clipped = solutions.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]

        splits = np.array_split(clipped, self.config.measure_dim, axis=1)
        measures = np.sum(splits, axis=-1).T

        if grad:
            # Since the objective is 1.0, the gradient is always 0.
            objective_grads = np.zeros(solutions.shape)

            # Compute gradient of the measures.
            derivatives = np.ones(solutions.shape)
            derivatives[clip_mask] = -5.12 / np.square(solutions[clip_mask])

            masks = np.zeros((self.config.measure_dim, solutions.shape[1]))
            start = 0
            for i, block in enumerate(splits):
                block_dim = block.shape[1]
                masks[i, start:start + block_dim] = 1.0
                start += block_dim

            # Repeat along the measure dimension. Shape should now be
            # (batch_size, measure_dim, solution_dim).
            derivatives = np.repeat(derivatives[:, None, :],
                                    self.config.measure_dim,
                                    axis=1)
            measure_grads = derivatives * masks

            return (
                objectives,
                measures,
                {
                    "objective_grads": objective_grads,
                    "measure_grads": measure_grads,
                },
            )

        else:
            return objectives, measures, {}
