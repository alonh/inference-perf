"""Trajectory kernels: positive-definite similarities over tool-call trajectories.

These operate on a turn's tool-call *structure* rather than its text, and are the
building blocks for the paper-grounded trajectory metrics (Raj et al. Eq. 1/3 and the
cross-condition MMD). Two views of a trajectory:

  - composition ("which tools, how often")  -> action histogram  -> Jensen-Shannon kernel
  - ordering     ("in what sequence")        -> tool-name sequence -> Global Alignment Kernel

Both kernels are normalized to (0, 1] with k(x, x) = 1, so they double as pairwise
similarities in a U-statistic.
"""

import math
from collections import Counter
from typing import Dict, Sequence


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence (base-2, in [0,1]) between two categorical dists."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            if ak > 0.0 and m[k] > 0.0:
                s += ak * math.log2(ak / m[k])
        return s

    jsd = 0.5 * kl(p) + 0.5 * kl(q)
    # Clamp tiny negatives from float error; JSD in [0,1] for base-2.
    return max(0.0, min(1.0, jsd))


def action_histogram(name_seq: Sequence[str]) -> Dict[str, float]:
    """Per-trajectory action histogram h_tau(a) = fraction of steps invoking action a."""
    if not name_seq:
        return {}
    c = Counter(name_seq)
    t = len(name_seq)
    return {a: n / t for a, n in c.items()}


def js_kernel(hist_a: Dict[str, float], hist_b: Dict[str, float], gamma: float = 1.0) -> float:
    """k_JS(p,q) = exp(-gamma * JSD(p||q)). Composition similarity in (0,1], k(x,x)=1.

    Two empty histories (no tool calls) are treated as identical (kernel 1.0)."""
    if not hist_a and not hist_b:
        return 1.0
    return math.exp(-gamma * js_divergence(hist_a, hist_b))


def global_alignment_kernel(a: Sequence[str], b: Sequence[str], sigma: float = 1.0) -> float:
    """Global Alignment Kernel over two tool-NAME sequences, normalized to [0,1].

    The paper uses GAK as a positive-definite analogue of edit distance for ordering
    (directly kernelizing Levenshtein is not guaranteed PD). We use the standard
    Cuturi-style DP with a local similarity kappa(x,y) = exp(-cost/sigma), cost = 0 for a
    match and 1 for a mismatch, then normalize k(a,b)/sqrt(k(a,a)*k(b,b)) so the diagonal
    is 1 and the value lies in [0,1].
    """

    def raw(x: Sequence[str], y: Sequence[str]) -> float:
        n, m = len(x), len(y)
        if n == 0 and m == 0:
            return 1.0
        if n == 0 or m == 0:
            return 0.0
        # kappa in (0,1]; soft-match local kernel.
        def kappa(i: int, j: int) -> float:
            return math.exp(-(0.0 if x[i] == y[j] else 1.0) / sigma)

        # DP over alignment paths (diagonal + two gaps), summing path weights.
        prev = [0.0] * (m + 1)
        prev[0] = 1.0
        for i in range(1, n + 1):
            cur = [0.0] * (m + 1)
            for j in range(1, m + 1):
                cur[j] = kappa(i - 1, j - 1) * (prev[j] + prev[j - 1] + cur[j - 1])
            prev = cur
        return prev[m]

    kab = raw(a, b)
    kaa = raw(a, a)
    kbb = raw(b, b)
    denom = math.sqrt(kaa * kbb)
    if denom == 0.0:
        return 1.0 if kab == 0.0 and not a and not b else 0.0
    return max(0.0, min(1.0, kab / denom))
