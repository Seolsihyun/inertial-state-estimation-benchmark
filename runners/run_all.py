from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runners.run_filter import parse_args as parse_filter_args, run_filter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all internal filters on one dataset.")
    parser.add_argument("--filters", nargs="+", default=["ekf", "ukf", "pf", "eskf", "inekf"])
    parser.add_argument("--config", default="config/euroc.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    results = []
    for name in args.filters:
        filter_args = parse_filter_args([
            "--filter", name,
            "--config", args.config,
            "--max-steps", str(args.max_steps),
            *( ["--output-root", args.output_root] if args.output_root else [] ),
            *( ["--particles", str(args.particles)] if args.particles is not None else [] ),
            *( ["--no-plots"] if args.no_plots else [] ),
        ])
        results.append(run_filter(filter_args))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
