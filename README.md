# mojo-hilbertcurve

`mojo-hilbertcurve` is a Mojo-accelerated, API-compatible port of the
[`hilbertcurve`](https://pypi.org/project/hilbertcurve/) Python package. It
maps an integer distance along an n-dimensional Hilbert curve to a point, and
maps points back to distances.

The covered public API is `hilbertcurve.hilbertcurve.HilbertCurve`: its
constructor, `point_from_distance`, `points_from_distances`,
`distance_from_point`, `distances_from_points`, bounds attributes, and
`match_type` behavior. Valid dense batches with `p * n <= 63` use Mojo
`int64` kernels. Wider grids retain the upstream arbitrary-precision result
through an exact pure-Python fallback; they are correct but not accelerated.
`n_procs` is accepted and exposed for source compatibility, but does not
control the native worker pool. Large fixed-width batches run independent rows
through a bounded Mojo worker pool; small batches remain serial to avoid launch
overhead.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi` sets `PYTHONPATH=python`, so this runs directly from the checkout. The
wrapper also rebuilds `dist/libmojo-hilbertcurve.so` automatically if its Mojo
source is newer.

## Usage

```python
from hilbertcurve import HilbertCurve

curve = HilbertCurve(p=3, n=2)
assert curve.point_from_distance(12) == [1, 3]
assert curve.distance_from_point([1, 3]) == 12

distances = [0, 1, 2, 3, 12]
points = curve.points_from_distances(distances)
assert curve.distances_from_points(points) == distances
```

## Performance

Measured with `pixi run bench` on 2026-08-02, Linux 6.8, Intel Xeon E5-2697 v4
at 2.30 GHz (72 logical CPUs), against `hilbertcurve` 2.0.5. Times are the
best of three; the benchmark task uses a machine-wide `flock`.

| case | mojo-hilbertcurve | hilbertcurve | ratio | result |
| --- | ---: | ---: | ---: | --- |
| points_from_distances (200k, p=10 n=3) | 77.1 ms | 1718.4 ms | 22.29x | faster |
| distances_from_points (200k, p=10 n=3) | 26.6 ms | 3609.4 ms | 135.76x | faster |
| points_from_distances (100k, p=15 n=4) | 46.7 ms | 1518.4 ms | 32.55x | faster |

## How it works

The Python package validates NumPy integer batches with vectorized range checks
and passes contiguous `numpy.int64` buffers directly to the C ABI. Other valid
iterables retain the upstream-compatible validation path. The point buffer is
row-major (`count × n`), and the distance buffer is a contiguous
one-dimensional `int64` array. Mojo performs the transpose, Gray-code, and
excess-work transforms in place; no allocation or ownership crosses the FFI
boundary. The inverse transform uses a private contiguous working copy so a
caller-owned point array is never mutated.

Mojo exports cannot expose parametric pointer types in this toolchain, so the
C ABI takes `Int` buffer addresses and reconstructs mutable pointers inside
the shared library. This keeps the Python-owned NumPy allocation alive for the
entire call and avoids per-point FFI overhead.

There is no GPU path. The transforms are branch-heavy integer bit permutations
with repeated small-buffer accesses, so their effective arithmetic intensity is
too low to repay host-device transfer and launch costs.

## Development

```bash
pixi run build && pixi run test && pixi run bench
```

The test suite checks exhaustive small curves, randomized high-bit curves,
constructor and validation behavior, output-type matching, and the
arbitrary-precision fallback against the installed upstream package.

## License

MIT
