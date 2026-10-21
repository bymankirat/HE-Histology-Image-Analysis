"""Run the whole pipeline.

    python run_all.py              # every step
    python run_all.py --from s03   # start part way through, using what is already in data/

Each step writes its results to data/derived/ and prints the numbers it found, so a run can
be read from the log alone.
"""
import argparse
import importlib
import sys
import time

STEPS = ["s01_images", "s02_validate", "s03_staining", "s04_real", "s05_report"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0], help="first step to run")
    ap.add_argument("--only", help="run a single step")
    args = ap.parse_args()

    def resolve(name):
        matches = [s for s in STEPS if s == name or s.startswith(name)]
        if not matches:
            raise SystemExit(f"unknown step {name!r}; choose from {', '.join(STEPS)}")
        return matches[0]

    steps = [resolve(args.only)] if args.only else STEPS[STEPS.index(resolve(args.start)):]
    t0 = time.time()
    for name in steps:
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        importlib.import_module(f"pipeline.{name}").main()
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
