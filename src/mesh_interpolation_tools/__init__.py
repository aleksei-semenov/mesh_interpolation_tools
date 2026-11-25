"""Conservative interpolation between triangular meshes, structured grids and HEALPix."""

from importlib.metadata import PackageNotFoundError, version as _version

from .tri2reg import tri2reg
from .tri2arb import tri2arb
from .tri2tri import conservative_interp_tri2tri

try:
    __version__ = _version("mesh-interpolation-tools")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "tri2reg",
    "tri2arb",
    "conservative_interp_tri2tri",
    "conservative_interp_spherical_tri2healpix",
    "__version__",
]


def __getattr__(name):
    # HEALPix support needs the optional `healpy` dependency, so import it lazily:
    # the 2D tools keep working on systems where healpy is not installed.
    if name == "conservative_interp_spherical_tri2healpix":
        try:
            from .tri2healpix import conservative_interp_spherical_tri2healpix
        except ImportError as exc:
            raise ImportError(
                "HEALPix interpolation requires healpy. Install it with "
                "`pip install mesh-interpolation-tools[healpix]`."
            ) from exc
        return conservative_interp_spherical_tri2healpix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
