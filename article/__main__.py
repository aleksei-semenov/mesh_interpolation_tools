"""Command-line runner for article experiments.

    python -m article list            # show available experiments
    python -m article run e01         # run experiments whose name starts with "e01"
    python -m article run e01 e03     # several
    python -m article run all         # everything
"""

import importlib
import pkgutil
import sys
import time

from article import experiments


def discover():
    """All modules in article/experiments/ named eNN_*.py, sorted."""
    return sorted(
        m.name for m in pkgutil.iter_modules(experiments.__path__)
        if m.name[:1] == "e" and m.name[1:3].isdigit()
    )


def describe(name):
    mod = importlib.import_module(f"article.experiments.{name}")
    doc = (mod.__doc__ or "").strip().splitlines()
    return doc[0] if doc else ""


def run(name):
    print(f"[{name}]")
    mod = importlib.import_module(f"article.experiments.{name}")
    t0 = time.perf_counter()
    mod.main()
    print(f"  done in {time.perf_counter() - t0:.1f} s")


def cli(argv):
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    names = discover()
    if argv[0] == "list":
        for n in names:
            print(f"{n:40s} {describe(n)}")
        return 0
    if argv[0] == "run":
        wanted = argv[1:] or ["all"]
        selected = names if "all" in wanted else [
            n for n in names if any(n.startswith(w) for w in wanted)
        ]
        if not selected:
            print(f"No experiment matches {wanted}. Try `python -m article list`.")
            return 1
        for n in selected:
            run(n)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":  # required on Windows: the library uses multiprocessing
    sys.exit(cli(sys.argv[1:]))
