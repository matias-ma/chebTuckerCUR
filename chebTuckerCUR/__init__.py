"""chebtucker — Tucker-qTensCUR approximation for multivariate functions.

Public API
----------
build_chebtucker(f, opts, d=None) -> ChebTucker
    Fit a Tucker decomposition via the Tucker-qTensCUR algorithm.

ChebTucker
    Evaluation object:  approx(x)  or  approx(x_batch).

TuckerOptions
    Configuration dataclass (frozen).  Key fields::

        max_rank       : int    = 8
        tucker_update  : str    = "C"   # "C" | "CU"
        domain         : array  = None    # defaults to [-1,1]^d
        cheb_tol       : float  = 1e-8
        n_starts       : int    = 256
"""

from .build import build_chebtucker
from .options import TuckerOptions
from .tucker import ChebTucker

__all__ = [
    "build_chebtucker",
    "ChebTucker",
    "TuckerOptions",
]
