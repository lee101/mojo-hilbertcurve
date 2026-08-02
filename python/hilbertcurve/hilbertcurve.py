"""A drop-in HilbertCurve whose dense int64 batches run in Mojo."""

from __future__ import annotations

import multiprocessing
from typing import Iterable, List, Union

import numpy as np

from ._lib import lib


def _binary_repr(num: int, width: int) -> str:
    return format(num, "b").zfill(width)


def _point_from_distance(distance: int, p: int, n: int) -> List[int]:
    x = [int(_binary_repr(distance, p * n)[i::n], 2) for i in range(n)]
    z = 2 << (p - 1)
    t = x[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        x[i] ^= x[i - 1]
    x[0] ^= t
    q = 2
    while q != z:
        mask = q - 1
        for i in range(n - 1, -1, -1):
            if x[i] & q:
                x[0] ^= mask
            else:
                t = (x[0] ^ x[i]) & mask
                x[0] ^= t
                x[i] ^= t
        q <<= 1
    return x


def _distance_from_point(point: Iterable[int], p: int, n: int) -> int:
    x = [int(el) for el in point]
    m = 1 << (p - 1)
    q = m
    while q > 1:
        mask = q - 1
        for i in range(n):
            if x[i] & q:
                x[0] ^= mask
            else:
                t = (x[0] ^ x[i]) & mask
                x[0] ^= t
                x[i] ^= t
        q >>= 1
    for i in range(1, n):
        x[i] ^= x[i - 1]
    t = 0
    q = m
    while q > 1:
        if x[n - 1] & q:
            t ^= q - 1
        q >>= 1
    for i in range(n):
        x[i] ^= t
    return int("".join(_binary_repr(x[i], p)[j] for j in range(p) for i in range(n)), 2)


class HilbertCurve:
    """Map distances to points and back using the upstream ``hilbertcurve`` API.

    Mojo accelerates validated batches where ``p * n <= 63``.  Larger grids
    retain upstream's arbitrary-precision behavior through the reference path.
    """

    def __init__(
        self,
        p: Union[int, float],
        n: Union[int, float],
        n_procs: int = 0,
    ) -> None:
        if (p % 1) != 0:
            raise TypeError("p is not an integer and can not be converted")
        if (n % 1) != 0:
            raise TypeError("n is not an integer and can not be converted")
        if (n_procs % 1) != 0:
            raise TypeError("n_procs is not an integer and can not be converted")
        self.p = int(p)
        self.n = int(n)
        if self.p <= 0:
            raise ValueError(f"p must be > 0 (got p={p} as input)")
        if self.n <= 0:
            raise ValueError(f"n must be > 0 (got n={n} as input)")
        self.min_h = 0
        self.max_h = 2 ** (self.p * self.n) - 1
        self.min_x = 0
        self.max_x = 2 ** self.p - 1
        n_procs = int(n_procs)
        if n_procs == -1:
            self.n_procs = multiprocessing.cpu_count()
        elif n_procs == 0:
            self.n_procs = 0
        elif n_procs > 0:
            self.n_procs = n_procs
        else:
            raise ValueError(f"n_procs must be >= -1 (got n_procs={n_procs} as input)")

    @property
    def _mojo_usable(self) -> bool:
        return self.p * self.n <= 63

    def _hilbert_integer_to_transpose(self, h: int) -> List[int]:
        return [int(_binary_repr(h, self.p * self.n)[i::self.n], 2) for i in range(self.n)]

    def _transpose_to_hilbert_integer(self, x: Iterable[int]) -> int:
        bits = [_binary_repr(x[i], self.p) for i in range(self.n)]
        return int("".join(y[i] for i in range(self.p) for y in bits), 2)

    def point_from_distance(self, distance: int) -> Iterable[int]:
        distance = int(distance)
        if self._mojo_usable and 0 <= distance <= self.max_h:
            src = np.asarray([distance], dtype=np.int64)
            dst = np.empty((1, self.n), dtype=np.int64)
            if not lib().mhc_points_from_distances(
                src.ctypes.data, dst.ctypes.data, 1, self.p, self.n
            ):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            return dst[0].tolist()
        return _point_from_distance(distance, self.p, self.n)

    def points_from_distances(
        self, distances: Iterable[int], match_type: bool = False
    ) -> Iterable[Iterable[int]]:
        original = distances
        if (
            self._mojo_usable
            and isinstance(original, np.ndarray)
            and original.ndim == 1
            and np.issubdtype(original.dtype, np.integer)
        ):
            if np.any(original > self.max_h):
                raise ValueError("all values in distances must be <= 2**(p*n)-1")
            if np.issubdtype(original.dtype, np.signedinteger) and np.any(original < self.min_h):
                raise ValueError("all values in distances must be >= 0")
            src = original if original.dtype == np.int64 and original.flags.c_contiguous else np.ascontiguousarray(
                original, dtype=np.int64
            )
            dst = np.empty((len(src), self.n), dtype=np.int64)
            if not lib().mhc_points_from_distances(src.ctypes.data, dst.ctypes.data, len(src), self.p, self.n):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            if match_type:
                return dst.astype(original.dtype, copy=False)
            return dst.tolist()
        values = list(distances)
        for ii, dist in enumerate(values):
            if (dist % 1) != 0:
                raise TypeError("all values in distances must be int or floats that are convertible to int "
                                f"but found distances[{ii}]={dist}")
            if dist > self.max_h:
                raise ValueError("all values in distances must be <= 2**(p*n)-1="
                                 f"{self.max_h} but found distances[{ii}]={dist} ")
            if dist < self.min_h:
                raise ValueError("all values in distances must be >= "
                                 f"{self.min_h} but found distances[{ii}]={dist}")
        ints = [int(v) for v in values]
        if self._mojo_usable:
            src = np.asarray(ints, dtype=np.int64)
            dst = np.empty((len(src), self.n), dtype=np.int64)
            if not lib().mhc_points_from_distances(src.ctypes.data, dst.ctypes.data, len(src), self.p, self.n):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            points = dst.tolist()
        else:
            points = [_point_from_distance(d, self.p, self.n) for d in ints]
        if match_type:
            if isinstance(original, np.ndarray):
                return np.asarray(points, dtype=original.dtype)
            target_type = type(original)
            return target_type([target_type(vec) for vec in points])
        return points

    def distance_from_point(self, point: Iterable[int]) -> int:
        values = [int(el) for el in point]
        if self._mojo_usable and len(values) == self.n and all(0 <= x <= self.max_x for x in values):
            src = np.asarray(values, dtype=np.int64).reshape(1, self.n)
            dst = np.empty(1, dtype=np.int64)
            if not lib().mhc_distances_from_points(
                src.ctypes.data, dst.ctypes.data, 1, self.p, self.n
            ):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            return int(dst[0])
        return _distance_from_point(values, self.p, self.n)

    def distances_from_points(
        self, points: Iterable[Iterable[int]], match_type: bool = False
    ) -> Iterable[int]:
        original = points
        if (
            self._mojo_usable
            and isinstance(original, np.ndarray)
            and original.ndim == 2
            and original.shape[1] == self.n
            and np.issubdtype(original.dtype, np.integer)
        ):
            if np.any(original > self.max_x):
                raise ValueError("all coordinate values in all vectors in points must be <= 2**p-1")
            if np.issubdtype(original.dtype, np.signedinteger) and np.any(original < self.min_x):
                raise ValueError("all coordinate values in all vectors in points must be > 0")
            src = np.array(original, dtype=np.int64, order="C", copy=True)
            dst = np.empty(len(src), dtype=np.int64)
            if not lib().mhc_distances_from_points(src.ctypes.data, dst.ctypes.data, len(src), self.p, self.n):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            if match_type:
                return dst.astype(original.dtype, copy=False)
            return dst.tolist()
        rows = list(points)
        for ii, point in enumerate(rows):
            if len(point) != self.n:
                raise ValueError(f"all vectors in points must have length n={self.n} "
                                 f"but found points[{ii}]={point}")
            if any(elx > self.max_x for elx in point):
                raise ValueError("all coordinate values in all vectors in points must be <= 2**p-1="
                                 f"{self.max_x} but found points[{ii}]={point}")
            if any(elx < self.min_x for elx in point):
                raise ValueError("all coordinate values in all vectors in points must be > "
                                 f"{self.min_x} but found points[{ii}]={point}")
            if any((elx % 1) != 0 for elx in point):
                raise TypeError("all coordinate values in all vectors in points must be int or floats "
                                f"that are convertible to int but found points[{ii}]={point}")
        ints = [[int(x) for x in row] for row in rows]
        if self._mojo_usable:
            src = np.asarray(ints, dtype=np.int64).reshape(len(ints), self.n)
            dst = np.empty(len(src), dtype=np.int64)
            if not lib().mhc_distances_from_points(src.ctypes.data, dst.ctypes.data, len(src), self.p, self.n):
                raise RuntimeError("Mojo Hilbert kernel rejected a valid int64 batch")
            distances = dst.tolist()
        else:
            distances = [_distance_from_point(point, self.p, self.n) for point in ints]
        if match_type:
            if isinstance(original, np.ndarray):
                return np.asarray(distances, dtype=original.dtype)
            return type(original)(distances)
        return distances

    def __str__(self):
        return f"HilbertCruve(p={self.p}, n={self.n}, n_procs={self.n_procs})"

    def __repr__(self):
        return self.__str__()
