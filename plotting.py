"""Logging + reward plots. Fully implemented -- nothing to do here.

Writes runs/<name>/log.csv and refreshes runs/<name>/progress.png periodically
so you can watch learning happen while training runs.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def moving_average(x, window):
    if len(x) < window or window < 2:
        return np.asarray(x, dtype=float)
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="valid")


class Logger:
    """Accumulates per-episode stats, mirrors them to CSV, and plots on demand."""

    FIELDS = ["episode", "step", "reward", "score", "length", "epsilon", "loss"]

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.rows = []
        self.csv_path = os.path.join(run_dir, "log.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log_episode(self, episode, step, reward, score, length, epsilon, loss):
        row = [episode, step, reward, score, length, epsilon, loss]
        self.rows.append(row)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _col(self, name):
        i = self.FIELDS.index(name)
        return [r[i] for r in self.rows]

    def plot(self, window=50):
        if len(self.rows) < 2:
            return
        steps = self._col("step")
        panels = [
            ("reward", "Episode return", "tab:blue"),
            ("score", "Pipes passed", "tab:green"),
            ("length", "Episode length (frames)", "tab:orange"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"DQN on Flappy Bird -- {os.path.basename(self.run_dir)}", fontsize=13)

        for ax, (key, title, color) in zip(axes.flat, panels):
            y = self._col(key)
            ax.plot(steps, y, alpha=0.22, color=color, linewidth=0.8, label="per episode")
            ma = moving_average(y, window)
            if len(ma) > 1:
                ax.plot(steps[-len(ma):], ma, color=color, linewidth=2.0,
                        label=f"moving avg ({window} ep)")
            ax.set_title(title)
            ax.set_xlabel("environment steps")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            if key in ("score", "length") and max(y) > 100:
                ax.set_yscale("symlog")  # scores explode once it starts working

        ax = axes.flat[3]
        ax.plot(steps, self._col("epsilon"), color="tab:red", label="epsilon")
        ax.set_ylabel("epsilon", color="tab:red")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        losses = [l for l in self._col("loss") if l is not None and l == l]
        if losses:
            ax2.plot(steps[-len(losses):], losses, color="tab:purple", alpha=0.7, label="loss")
            ax2.set_ylabel("TD loss", color="tab:purple")
            ax2.set_yscale("log")
        ax.set_title("Exploration & TD loss")

        fig.tight_layout()
        out = os.path.join(self.run_dir, "progress.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out
