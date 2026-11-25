"""Statistical information of a paired accuracy difference, as a function of trials per task.

The paired per-task difference d_i = mean_k(B_i) - mean_k(A_i), with k trials per task on each side, has

    var(d_i) = tau2 + sigma2 / k

where tau2 is the variance of the TRUE per-task differences (effect heterogeneity across tasks) and sigma2 is the
per-trial Bernoulli variance within a task, summed over both sides (mean_i p_i(1 - p_i) for A plus the same for B).
The information about the mean difference after k trials of each of n tasks is

    I(k) = n / (tau2 + sigma2 / k)

so more trials per task stop paying once sigma2 / k is small next to tau2: on val200 (tau2 = 0.0081, sigma2 =
0.249) I(24) / I(2) is 7.2, not 12, and I(24) / I(8) is 2.1, not 3. The information fraction of a look at k_j
trials is I(k_j) / I(K); it is the "time" a group-sequential boundary is a function of, not k_j / K.

Observed information at a look is 1 / SE^2 of the paired mean difference (`paired.Paired.se`).

Pure and stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["Information", "VAL200"]


@dataclass(frozen=True)
class Information:
    """Planning values of the variance components of a per-task paired difference (see the module docstring)."""
    tau2: float        # variance of the true per-task differences (effect heterogeneity)
    sigma2: float      # per-trial within-task variance, summed over both sides

    def __post_init__(self) -> None:
        if self.tau2 < 0 or self.sigma2 <= 0:
            raise ValueError(f"need tau2 >= 0 and sigma2 > 0, got {self.tau2}, {self.sigma2}")

    def variance(self, k: int, k_ref: int | None = None) -> float:
        """var(d_i) with k candidate trials per task and k_ref reference trials (None: k, same units)."""
        if k_ref is None:
            return self.tau2 + self.sigma2 / k
        return self.tau2 + self.sigma2 / 2 / k + self.sigma2 / 2 / k_ref

    def info(self, n_tasks: int, k: int, k_ref: int | None = None) -> float:
        """I(k) = n / var(d_i): the inverse variance of the paired mean difference."""
        return n_tasks / self.variance(k, k_ref)

    def fractions(self, trial_looks: Sequence[int], k_ref: int | None = None) -> tuple[float, ...]:
        """I(k_j) / I(K) for looks at k_j trials per task, K = the last look."""
        last = self.variance(trial_looks[-1], k_ref)
        return tuple(last / self.variance(k, k_ref) for k in trial_looks)


#: Estimates from the val200 runs (defects-val200-qwen3.5-4b-64k, stack-r2: orig vs stderr, 200 tasks x 8 trials):
#: sigma2 = 2 x mean_i s_i (8 - s_i) / (8 * 7) = 0.248; tau2 = var(d_i) - mean within-task part = 0.0081.
VAL200 = Information(tau2=0.0081, sigma2=0.249)
