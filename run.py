"""Reproduce every result: python run.py

Thread limits are set before NumPy and Torch are imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.pipeline import main

if __name__ == "__main__":
    main()
