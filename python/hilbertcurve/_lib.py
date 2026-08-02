"""Load the fixed-width Mojo Hilbert kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_HILBERTCURVE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-hilbertcurve.so"
)
I64 = ctypes.c_int64


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_HILBERTCURVE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_HILBERTCURVE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJO_HILBERTCURVE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name in ("mhc_points_from_distances", "mhc_distances_from_points"):
            fn = getattr(_library, name)
            fn.argtypes = [I64, I64, I64, I64, I64]
            fn.restype = I64
    return _library
