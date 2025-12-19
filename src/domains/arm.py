"""Provides arm domain."""
import numpy as np

from src.domains.domain_base import DomainBase, EvaluateTorchMixin


class Arm(EvaluateTorchMixin, DomainBase):
    """Arm repertoire domain."""

    def initial_solution(self):
        return np.zeros(self.config.solution_dim)

    def evaluate(self, solutions, grad=False):
        """
        The length of each arm link is assumed to be 1.0.

        Args:
            solutions: A (batch_size, solution_dim) array where each
                row contains the joint angles for the arm.
        Returns:
            objectives: (batch_size,) array of objectives.
            measures: (batch_size, 2) array of measures.
            info (dict): If `grad` is passed in, it contains:
                - objective_grads (np.ndarray): (batch_size, solution_dim) batch
                  of objective gradients.
                - measure_grads (np.ndarray): (batch_size, 2, solution_dim)
                  batch of measure gradients.
        """

        # Assume link lengths are all 1.0 -- the code is written to be more
        # general and support variable link lengths.
        link_lengths = np.ones(solutions.shape[1])

        n_dim = link_lengths.shape[0]
        objectives = -np.var(solutions, axis=1)

        # Remap the objective from [-1, 0] to [0, 1]
        objectives = (objectives + 1.0) # * 100.0

        # theta_1, theta_1 + theta_2, ...
        cum_theta = np.cumsum(solutions, axis=1)
        # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
        x_pos = link_lengths[None] * np.cos(cum_theta)
        # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
        y_pos = link_lengths[None] * np.sin(cum_theta)

        measures = np.concatenate(
            (
                np.sum(x_pos, axis=1, keepdims=True),
                np.sum(y_pos, axis=1, keepdims=True),
            ),
            axis=1,
        )

        if grad:
            objective_grads = None
            measure_grads = None

            means = np.mean(solutions, axis=1)
            means = np.expand_dims(means, axis=1)

            base = n_dim * np.ones(n_dim)
            objective_grads = -2 * (solutions - means) / base

            sum_0 = np.zeros(len(solutions))
            sum_1 = np.zeros(len(solutions))

            measure_grads = np.zeros((len(solutions), 2, n_dim))
            for i in range(n_dim - 1, -1, -1):
                sum_0 += -link_lengths[i] * np.sin(cum_theta[:, i])
                sum_1 += link_lengths[i] * np.cos(cum_theta[:, i])

                measure_grads[:, 0, i] = sum_0
                measure_grads[:, 1, i] = sum_1

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
