"""Run all manuscript experiments from a single command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ROOT_DIR / "experiments" / "experiment_1_numerical_methods" / "run_experiment.py",
    ROOT_DIR / "experiments" / "experiment_2_physical_properties" / "run_experiment.py",
    ROOT_DIR / "experiments" / "experiment_3_prognostic_evolution" / "run_experiment.py",
]
SUPPLEMENTARY = ROOT_DIR / "experiments" / "supplementary_bubble_mountain" / "run_experiment.py"


def main() -> None:
    """Execute the paper experiments and optionally the supplementary model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-supplementary", action="store_true",
                        help="also run the slower bubble and mountain-wave demonstrations")
    arguments = parser.parse_args()
    scripts = EXPERIMENTS + ([SUPPLEMENTARY] if arguments.include_supplementary else [])
    for script in scripts:
        print(f"\nRunning {script.relative_to(ROOT_DIR)}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT_DIR, check=True)
    print("\nAll requested experiments completed successfully.")


if __name__ == "__main__":
    main()
