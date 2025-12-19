"""Minimal utility for timing code."""

import time


class CodeTimer:
    """Minimal utility for timing code.

    Skimps heavily on error checking.
    """

    def __init__(self, names):
        self.totals = dict.fromkeys(names, 0.0)
        self.itr_time = {}
        self.starts = {}

    def start(self, names, time_type="wall"):
        """Sets the start time for all the given names."""
        if time_type == "wall":
            t = time.time()
        elif time_type == "process":
            t = time.process_time()
        else:
            raise ValueError("Unknown time_type")

        if isinstance(names, str):
            names = [names]

        for n in names:
            self.starts[n] = t

    def end(self, names, time_type="wall"):
        if time_type == "wall":
            t = time.time()
        elif time_type == "process":
            t = time.process_time()
        else:
            raise ValueError("Unknown time_type")

        if isinstance(names, str):
            names = [names]

        for n in names:
            self.itr_time[n] = t - self.starts[n]

    def calc_totals(self):
        """Accumulates total time.

        Before calling this, all itr times must be set.
        """
        for n in self.totals:
            self.totals[n] = self.totals[n] + self.itr_time[n]

    def clear(self):
        self.itr_time.clear()

    def metric_list(self):
        """Exports metric list for MetricLogger."""
        metrics = []
        for n in self.totals:
            metrics.append((f"Time: {n}", False))
            metrics.append((f"Total Time: {n}", True, 0.0))
        return metrics

    def metrics_dict(self):
        """Exports metrics for MetricLogger."""
        d = {}
        for n in self.totals:
            d[f"Time: {n}"] = self.itr_time[n]
            d[f"Total Time: {n}"] = self.totals[n]
        return d
