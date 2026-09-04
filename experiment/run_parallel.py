"""
run_parallel.py — Parallel Runner cho Uplift Modeling Baselines trên Criteo
===========================================================================
Sử dụng:
    python experiment/run_parallel.py --model all --parallel 2
    python experiment/run_parallel.py --models tarnet cfrnet dragonnet --parallel 3
    python experiment/run_parallel.py --models tarnet efin --seeds 1 2 3 4 5 --parallel 2
    python experiment/run_parallel.py --config experiment/config.yaml --parallel 2
"""

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment.main import main

if __name__ == "__main__":
    # Mặc định --parallel 2 nếu user không chỉ định
    if "--parallel" not in sys.argv and "--n_jobs" not in sys.argv:
        sys.argv.extend(["--parallel", "2"])
    main()
