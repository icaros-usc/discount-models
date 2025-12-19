"""Generates N unique integer seeds in the range [0, 10000).

Usage:
    python scripts/seeds.py N

    # Or, with uv:
    uv run scripts/seeds.py N
"""

import sys

import numpy as np

if __name__ == "__main__":
    n = int(sys.argv[1])
    rng = np.random.default_rng()
    nums = rng.choice(np.arange(10000), size=n, replace=False)
    print(",".join(map(str, nums)))
