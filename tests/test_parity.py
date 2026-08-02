"""Parity checks against the published hilbertcurve package."""

from __future__ import annotations

import importlib.metadata
import importlib.util

import numpy as np
import pytest

from hilbertcurve import HilbertCurve


def upstream_curve():
    path = importlib.metadata.distribution("hilbertcurve").locate_file("hilbertcurve/hilbertcurve.py")
    spec = importlib.util.spec_from_file_location("upstream_hilbertcurve", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HilbertCurve


UpstreamHilbertCurve = upstream_curve()


@pytest.mark.parametrize("p,n", [(1, 1), (1, 4), (2, 2), (3, 2), (4, 3), (8, 4), (20, 3), (31, 2)])
def test_points_and_distances_match_upstream(p, n):
    theirs = UpstreamHilbertCurve(p, n)
    ours = HilbertCurve(p, n)
    max_h = (1 << (p * n)) - 1
    if p * n <= 12:
        distances = list(range(max_h + 1))
    else:
        rng = np.random.default_rng(p * 100 + n)
        distances = [0, max_h] + [int(v) for v in rng.integers(0, max_h + 1, size=100)]
    expected_points = theirs.points_from_distances(distances)
    actual_points = ours.points_from_distances(distances)
    assert actual_points == expected_points
    assert ours.distances_from_points(actual_points) == distances
    assert ours.distances_from_points(expected_points) == theirs.distances_from_points(expected_points)
    for distance, point in zip(distances[:20], expected_points[:20]):
        assert ours.point_from_distance(distance) == theirs.point_from_distance(distance)
        assert ours.distance_from_point(point) == theirs.distance_from_point(point)


def test_int64_boundary_and_mojo_shared_library():
    ours = HilbertCurve(63, 1)
    distances = [0, 1, (1 << 62) + 17, (1 << 63) - 1]
    assert ours.points_from_distances(distances) == [[distance] for distance in distances]
    assert ours.distances_from_points([[distance] for distance in distances]) == distances
    from hilbertcurve._lib import lib
    assert hasattr(lib(), "mhc_points_from_distances")
    assert hasattr(lib(), "mhc_distances_from_points")


def test_c_abi_rejects_null_nonempty_buffers_and_accepts_empty_batches():
    from hilbertcurve._lib import lib

    native = lib()
    assert native.mhc_points_from_distances(0, 0, 1, 1, 1) == 0
    assert native.mhc_distances_from_points(0, 0, 1, 1, 1) == 0
    assert native.mhc_points_from_distances(0, 0, 0, 1, 1) == 1
    assert native.mhc_distances_from_points(0, 0, 0, 1, 1) == 1

    curve = HilbertCurve(3, 2)
    distances = np.array([], dtype=np.int64)
    points = np.empty((0, 2), dtype=np.int64)
    result_points = curve.points_from_distances(distances, match_type=True)
    result_distances = curve.distances_from_points(points, match_type=True)
    assert result_points.shape == (0, 2)
    assert result_distances.shape == (0,)


@pytest.mark.parametrize("count", [4095, 4096])
def test_large_batch_parallel_threshold_matches_upstream(count):
    p, n = 10, 3
    rng = np.random.default_rng(count)
    distances = [int(value) for value in rng.integers(0, 1 << (p * n), size=count)]
    ours = HilbertCurve(p, n)
    theirs = UpstreamHilbertCurve(p, n)
    points = theirs.points_from_distances(distances)
    assert ours.points_from_distances(distances) == points
    assert ours.distances_from_points(points) == distances


def test_match_type_and_n_procs_contract():
    distances = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    ours = HilbertCurve(2, 2, n_procs=1)
    theirs = UpstreamHilbertCurve(2, 2)
    actual_points = ours.points_from_distances(distances, match_type=True)
    expected_points = theirs.points_from_distances(distances, match_type=True)
    assert isinstance(actual_points, np.ndarray)
    assert actual_points.dtype == distances.dtype
    assert np.array_equal(actual_points, expected_points)
    assert np.array_equal(ours.distances_from_points(actual_points, match_type=True), distances)
    assert HilbertCurve(2, 2, n_procs=-1).n_procs > 0


def test_numpy_batches_use_the_mojo_path_without_mutating_input():
    rng = np.random.default_rng(42)
    distances = rng.integers(0, 1 << 30, size=4097, dtype=np.int64)
    ours = HilbertCurve(10, 3)
    theirs = UpstreamHilbertCurve(10, 3)
    expected_points = np.asarray(theirs.points_from_distances(distances), dtype=np.int64)
    actual_points = ours.points_from_distances(distances, match_type=True)
    assert np.array_equal(actual_points, expected_points)
    points_before = actual_points.copy()
    assert np.array_equal(ours.distances_from_points(actual_points, match_type=True), distances)
    assert np.array_equal(actual_points, points_before)


def test_numpy_noncontiguous_and_non_int64_inputs_are_exact():
    curve = HilbertCurve(10, 3)
    distances = np.arange(0, 200, dtype=np.uint32)[::2]
    expected = [curve.point_from_distance(int(distance)) for distance in distances]
    actual = curve.points_from_distances(distances, match_type=True)
    assert actual.dtype == distances.dtype
    assert np.array_equal(actual, np.asarray(expected, dtype=np.uint32))

    points = actual[:, ::-1][:, ::-1]
    before = points.copy()
    assert np.array_equal(curve.distances_from_points(points, match_type=True), distances)
    assert np.array_equal(points, before)


@pytest.mark.parametrize(
    "method,args,error",
    [
        ("points_from_distances", ([-1],), ValueError),
        ("points_from_distances", ([16],), ValueError),
        ("points_from_distances", ([1.5],), TypeError),
        ("distances_from_points", ([[0]],), ValueError),
        ("distances_from_points", ([[4, 0]],), ValueError),
        ("distances_from_points", ([[-1, 0]],), ValueError),
        ("distances_from_points", ([[0.5, 0]],), TypeError),
    ],
)
def test_batch_validation_matches_upstream(method, args, error):
    ours = HilbertCurve(2, 2)
    theirs = UpstreamHilbertCurve(2, 2)
    with pytest.raises(error):
        getattr(theirs, method)(*args)
    with pytest.raises(error):
        getattr(ours, method)(*args)


def test_arbitrary_precision_fallback_matches_upstream():
    p, n = 40, 2
    distances = [0, 1, (1 << 79) - 1, (1 << 60) + 1234567]
    ours = HilbertCurve(p, n)
    theirs = UpstreamHilbertCurve(p, n)
    assert ours.points_from_distances(distances) == theirs.points_from_distances(distances)
    points = theirs.points_from_distances(distances)
    assert ours.distances_from_points(points) == theirs.distances_from_points(points)


def test_constructor_and_repr_match_upstream():
    for args in [(0, 2), (2, 0), (2.5, 2), (2, 2, -2)]:
        with pytest.raises((TypeError, ValueError)):
            UpstreamHilbertCurve(*args)
        with pytest.raises((TypeError, ValueError)):
            HilbertCurve(*args)
    assert repr(HilbertCurve(3, 2)) == repr(UpstreamHilbertCurve(3, 2))
