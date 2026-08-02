"""Measure the same dense batches through Mojo and upstream hilbertcurve."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from hilbertcurve import HilbertCurve  # noqa: E402


def upstream_class():
    path = importlib.metadata.distribution("hilbertcurve").locate_file("hilbertcurve/hilbertcurve.py")
    spec = importlib.util.spec_from_file_location("upstream_hilbertcurve_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HilbertCurve


UpstreamHilbertCurve = upstream_class()


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    rng = np.random.default_rng(7)
    cases = [("points_from_distances (200k, p=10 n=3)", 10, 3),
             ("distances_from_points (200k, p=10 n=3)", 10, 3),
             ("points_from_distances (100k, p=15 n=4)", 15, 4)]
    print("| case | mojo-hilbertcurve | hilbertcurve | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, p, n in cases:
        distances = rng.integers(0, 1 << min(p * n, 62), size=200_000 if "200k" in name else 100_000,
                                 dtype=np.int64)
        ours = HilbertCurve(p, n)
        theirs = UpstreamHilbertCurve(p, n)
        if name.startswith("points"):
            our_call = lambda: ours.points_from_distances(distances)
            their_call = lambda: theirs.points_from_distances(distances)
        else:
            points = np.asarray(theirs.points_from_distances(distances), dtype=np.int64)
            our_call = lambda: ours.distances_from_points(points)
            their_call = lambda: theirs.distances_from_points(points)
        our_call()
        a, b = timeit(our_call), timeit(their_call)
        ratio = b / a
        result = "faster" if a < b else "slower"
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {ratio:.2f}x | {result} |")


if __name__ == "__main__":
    main()
