"""Run the test suite without pytest, for environments where it is not installed.

`python -m pytest -q` is the normal way in; this runner exists so the same tests can be
executed anywhere Python and the pipeline's own dependencies are available.
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ["tests.test_stains", "tests.test_nuclei", "tests.test_synthetic"]


def main() -> int:
    passed, failed = 0, []
    t0 = time.time()
    for name in MODULES:
        mod = importlib.import_module(name)
        for fn_name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, fn_name)
            try:
                fn()
                passed += 1
                print(f"  ok   {name.split('.')[-1]}.{fn_name}")
            except Exception:
                failed.append(f"{name}.{fn_name}")
                print(f"  FAIL {name.split('.')[-1]}.{fn_name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed in {time.time() - t0:.1f}s")
    for f in failed:
        print("  failed:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
