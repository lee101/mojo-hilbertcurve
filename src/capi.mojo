"""C ABI for batched, fixed-width Hilbert curve conversion.

The Python facade owns every buffer.  `Int` addresses keep these exports
non-parametric; `Int` is the signed 64-bit lane used by NumPy int64 arrays.
"""

from std.algorithm.functional import parallelize

comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 4096


def ptr(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def point_from_distance(distance: Int, points: IPtr, offset: Int, p: Int, n: Int):
    var axis = 0
    while axis < n:
        var coordinate = 0
        var bit = 0
        while bit < p:
            var source_bit = p * n - 1 - (bit * n + axis)
            coordinate |= ((distance >> source_bit) & 1) << (p - 1 - bit)
            bit += 1
        points.store(offset + axis, coordinate)
        axis += 1

    var m = 1 << (p - 1)
    var t = points.load(offset + n - 1) >> 1
    axis = n - 1
    while axis > 0:
        points.store(offset + axis, points.load(offset + axis) ^ points.load(offset + axis - 1))
        axis -= 1
    points.store(offset, points.load(offset) ^ t)

    var q = 2
    while q <= m:
        var mask = q - 1
        axis = n - 1
        while axis >= 0:
            if (points.load(offset + axis) & q) != 0:
                points.store(offset, points.load(offset) ^ mask)
            else:
                t = (points.load(offset) ^ points.load(offset + axis)) & mask
                points.store(offset, points.load(offset) ^ t)
                points.store(offset + axis, points.load(offset + axis) ^ t)
            axis -= 1
        if q == m:
            break
        q <<= 1


def distance_from_point(points: IPtr, offset: Int, p: Int, n: Int) -> Int:
    var m = 1 << (p - 1)
    var q = m
    var axis: Int
    var t = 0
    while q > 1:
        var mask = q - 1
        axis = 0
        while axis < n:
            if (points.load(offset + axis) & q) != 0:
                points.store(offset, points.load(offset) ^ mask)
            else:
                t = (points.load(offset) ^ points.load(offset + axis)) & mask
                points.store(offset, points.load(offset) ^ t)
                points.store(offset + axis, points.load(offset + axis) ^ t)
            axis += 1
        q >>= 1

    axis = 1
    while axis < n:
        points.store(offset + axis, points.load(offset + axis) ^ points.load(offset + axis - 1))
        axis += 1
    t = 0
    q = m
    while q > 1:
        if (points.load(offset + n - 1) & q) != 0:
            t ^= q - 1
        q >>= 1
    axis = 0
    while axis < n:
        points.store(offset + axis, points.load(offset + axis) ^ t)
        axis += 1

    var distance = 0
    var bit = 0
    while bit < p:
        axis = 0
        while axis < n:
            distance = (distance << 1) | ((points.load(offset + axis) >> (p - 1 - bit)) & 1)
            axis += 1
        bit += 1
    return distance


def usable(count: Int, p: Int, n: Int) -> Bool:
    # Do not multiply untrusted C ABI arguments: overflowing ``p * n`` could
    # accidentally admit an invalid kernel invocation.
    return count >= 0 and p > 0 and n > 0 and p <= 63 / n


def points_batch(distances: IPtr, points: IPtr, count: Int, p: Int, n: Int):
    if count < PARALLEL_THRESHOLD:
        var row = 0
        while row < count:
            point_from_distance(distances.load(row), points, row * n, p, n)
            row += 1
        return

    @parameter
    def work(row: Int):
        point_from_distance(distances.load(row), points, row * n, p, n)

    parallelize[work](count, 8)


def distances_batch(points: IPtr, distances: IPtr, count: Int, p: Int, n: Int):
    if count < PARALLEL_THRESHOLD:
        var row = 0
        while row < count:
            distances.store(row, distance_from_point(points, row * n, p, n))
            row += 1
        return

    @parameter
    def work(row: Int):
        distances.store(row, distance_from_point(points, row * n, p, n))

    parallelize[work](count, 8)


@export("mhc_points_from_distances")
def mhc_points_from_distances(distances_addr: Int, points_addr: Int, count: Int, p: Int, n: Int) abi("C") -> Int:
    if not usable(count, p, n):
        return 0
    if count > 0 and (distances_addr == 0 or points_addr == 0):
        return 0
    var distances = ptr(distances_addr)
    var points = ptr(points_addr)
    points_batch(distances, points, count, p, n)
    return 1


@export("mhc_distances_from_points")
def mhc_distances_from_points(points_addr: Int, distances_addr: Int, count: Int, p: Int, n: Int) abi("C") -> Int:
    if not usable(count, p, n):
        return 0
    if count > 0 and (points_addr == 0 or distances_addr == 0):
        return 0
    var points = ptr(points_addr)
    var distances = ptr(distances_addr)
    distances_batch(points, distances, count, p, n)
    return 1
