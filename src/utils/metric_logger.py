"""Provides a utility for logging metrics across iterations."""

import json
import logging
from collections import OrderedDict
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.utils.text_plot import text_plot


class MetricLogger:
    """Tracks various pieces of scalar metrics across generations.

    Args:
        metric_list: A list of metric descriptions. Each description is a tuple of
            (name, use_zero, initial_value) where use_zero tells whether the metric
            starts on iteration 0 (some values, such as max objective, do not make sense
            if they start on iteration 0, while others, like archive size, do make
            sense). If use_zero is provided, an initial_value for the metric must be
            provided indicating its value on iteration 0. Commonly, this initial value
            will be 0.
        x_scale: Amount by which to scale the x-axis. By default, we simply count 0, 1,
            2, ... However, some metrics may only be recorded every couple iterations,
            e.g., 0, 50, 100, ... -- in this case x_scale would be 50.

    Usage:
        metrics = MetricLogger()
        for itr in range(iterations):
            metrics.start_itr()
            metrics.add("metric 1", value)
            metrics.add("metric 2", value)
            metrics.end_itr()
            metrics.display_text()
    """

    def __init__(
        self,
        metric_list: list[tuple[str, bool]],
        x_scale: int | None = None,
    ) -> None:
        self._active_itr = False
        self._total_itrs = 0
        self._added_this_itr = None
        self._x_scale = 1 if x_scale is None else x_scale

        self._metrics = OrderedDict()
        for entry in metric_list:
            metric_name = entry[0]
            use_zero = entry[1]
            self._metrics[metric_name] = {
                # entry[2] is the initial_value.
                "data": [self._to_python_type(entry[2])] if use_zero else [],
                "use_zero": use_zero,
            }

    @property
    def keys(self):
        """Dict keys with the names of metrics in this logger."""
        return self._metrics.keys()

    @property
    def names(self) -> list[str]:
        """List of names of metrics in this logger."""
        return list(self._metrics.keys())

    @property
    def total_itrs(self) -> int:
        """Total number of iterations completed so far."""
        return self._total_itrs

    @property
    def x_scale(self) -> int:
        """Amount by which to scale the x-axis."""
        return self._x_scale

    def to_json(self, jsonfile: str) -> None:
        """Saves the logger's info in JSON format.

        Args:
            jsonfile: Name of the file to save to.
        """
        with open(jsonfile, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "active_itr": self._active_itr,
                    "total_itrs": self._total_itrs,
                    "added_this_itr": list(self._added_this_itr),
                    "metrics": self._metrics,
                    "x_scale": self.x_scale,
                },
                file,
                indent=2,
            )

    @staticmethod
    def from_json(jsonfile: str) -> "MetricLogger":
        """Constructs a logger from the data in the JSON file.

        Args:
            jsonfile: Name of the file to load from.
        """
        with open(jsonfile, encoding="utf-8") as file:
            data = json.load(file)

        # pylint: disable = protected-access
        metrics = MetricLogger(metric_list=[], x_scale=data.get("x_scale"))
        metrics._active_itr = data["active_itr"]
        metrics._total_itrs = data["total_itrs"]
        metrics._added_this_itr = set(data["added_this_itr"])
        metrics._metrics = OrderedDict(data["metrics"])

        return metrics

    def start_itr(self):
        """Starts the iteration."""
        if self._active_itr:
            raise RuntimeError("Already in the middle of an iteration.")
        self._active_itr = True
        self._added_this_itr = set()

    def end_itr(self) -> None:
        """Ends the iteration.

        Raises:
            RuntimeError: This method was called without calling start_itr().
            RuntimeError: Not all metrics were added before calling this method.
        """
        if not self._active_itr:
            raise RuntimeError("Iteration has not been started. Call start_itr().")
        self._active_itr = False
        self._total_itrs += 1

        # Check whether all metrics were added.
        remaining_metrics = set(self._metrics.keys()) - self._added_this_itr
        if len(remaining_metrics) > 0:
            raise RuntimeError(
                f"The following metrics were not added this itr: {remaining_metrics}"
            )

    @staticmethod
    def _to_python_type(value):
        """Converts the value to a Python type. Necessary because JSON only accepts Python dtypes."""
        if isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        return value

    def add(
        self, name: str, value: float | int, logger: logging.Logger | None = None
    ) -> None:
        """Adds the given metric.

        Args:
            name: The name of the metric. This must be one of the metrics provided in
                the constructor.
            value: the scalar value to log.
            logger: If not None, this logger will be used to log the metric to the
                console immediately.

        Raises:
            RuntimeError: The metric name is not recognized.
            RuntimeError: The metric has already been added this itr.
        """
        if not self._active_itr:
            raise RuntimeError("Iteration has not been started. Call start_itr().")
        if name not in self._metrics:
            raise RuntimeError(f"Unknown metric '{name}'")
        if name in self._added_this_itr:
            raise RuntimeError(f"Metric '{name}' already added this itr")

        value = self._to_python_type(value)

        self._metrics[name]["data"].append(value)
        self._added_this_itr.add(name)
        if logger is not None:
            logger.info("%s: %s", name, str(value))

    def add_post(
        self, name: str, values: Sequence[float] | Sequence[int], use_zero: bool
    ) -> None:
        """Add a new metric that was not given at initialization.

        This method cannot be called in the middle of an iteration.

        Raises:
            RuntimeError: Iteration is currently active.
            ValueError: values is of the wrong length.
        """
        if self._active_itr:
            raise RuntimeError("Call end_itr() before calling this method.")

        expected_length = self._total_itrs + int(use_zero)
        if len(values) != expected_length:
            raise ValueError(
                f"values should be length {expected_length} but is length {len(values)}"
            )

        self._metrics[name] = {
            "data": list(values),
            "use_zero": use_zero,
        }

    def remove(self, name: str) -> None:
        """Removes a metric if it exists."""
        self._metrics.pop(name, None)

    def add_dict(self, metrics: dict) -> None:
        """Shortcut for adding multiple metrics for an iteration.

        Args:
            metrics: A dictionary mapping from metric names to the values.
        """
        for name, value in metrics.items():
            self.add(name, value)

    def get_plot_data(self) -> dict:
        """Returns the data in a form suitable for plotting metrics vs. itrs.

        Specifically, the data looks like this::

            {
                name: {
                    "x": [0, 1, 2, ...] # 0 may be excluded.
                    "y": [...] # metric values.
                }
                ... # More metrics
            }

        Note this method will only work when not in an active iteration, as data
        may be updated during an iteration.

        Returns:
            See above.

        Raises:
            RuntimeError: Iteration is currently active.
        """
        if self._active_itr:
            raise RuntimeError("Call end_itr() before calling this method.")
        data = {}
        x_with_zero = (np.arange(self._total_itrs + 1) * self.x_scale).tolist()
        x_no_zero = (np.arange(1, self._total_itrs + 1) * self.x_scale).tolist()
        for name in self._metrics:
            data[name] = {
                "x": (x_with_zero if self._metrics[name]["use_zero"] else x_no_zero),
                "y": self._metrics[name]["data"],
            }
        return data

    def get_single(self, name: str) -> dict:
        """Returns the data for plotting one metric.

        Args:
            name: Name of the metric to retrieve.

        Returns:
            Dict with plot data for the given metric. Equivalent to one of the
            entries in get_plot_data().

        Raises:
            IndexError: name is not a valid metric.
        """
        if name not in self._metrics:
            raise IndexError(f"'{name}' is not a known metric")
        return {
            "x": (
                np.arange(
                    int(not self._metrics[name]["use_zero"]), self._total_itrs + 1
                )
                * self.x_scale
            ).tolist(),
            "y": self._metrics[name]["data"],
        }

    def get_last(self, name: str, default=None):
        """Returns the last value of a metric.

        If the metric has no values, `default` is returned instead.
        """
        vals = self._metrics[name]["data"]
        if len(vals) == 0:
            return default
        return vals[-1]

    def summary(self) -> dict:
        """Return the last metrics."""
        return {name: array["data"][-1] for name, array in self._metrics.items()}

    def get_plot_text(self, plot_width: int = 80, plot_height: int = 20) -> str:
        """Generates string with plots of all the data.

        Args:
            plot_width: Width of each plot in characters.
            plot_height: Height of each plot in characters.

        Returns:
            A multi-line string with all the plots joined together.
        """
        data = self.get_plot_data()
        output = []
        for name, array in data.items():
            output.extend(
                [
                    f"=== {name} (Last val: {array['y'][-1]}) ===",
                    text_plot(array["x"], array["y"], plot_width, plot_height),
                ]
            )
        return "\n".join(output)

    def display_text(self, plot_width: int = 80, plot_height: int = 20) -> None:
        """Print out all the plots."""
        print(self.get_plot_text(plot_width, plot_height))

    def plot_graphic(self, file: str, ncols: int = 4) -> None:
        """Plots to a file with matplotlib."""
        data = self.get_plot_data()
        nrows = int(np.ceil(len(data) / ncols))
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 4, nrows * 3),
        )
        axs = axs.ravel()

        # The exact number of axes may not match the data available, so strict=False.
        for ax, (name, d) in zip(axs, data.items(), strict=False):
            ax.set_title(name)
            ax.plot(d["x"], d["y"])

        fig.tight_layout()
        fig.savefig(file, dpi=300)
        plt.close(fig)
