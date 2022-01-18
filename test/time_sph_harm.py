"""This test the equality and execution speed of different implementations to
generate spherical harmonics.

Exemplary execution:
======================
node "C18TTLT"
======================

======================
_TIMEIT_REPEAT = 5
_TIMEIT_NUMBER = 500
_N_MAX = 8
======================

======================
_KIND = 'complex'
======================

sph.sph_harm
time:                          0.33s

spaudiopy.sh_matrix
time:                          0.39s
time factor:                   1.16 ... WORSE
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph.sph_harm_large (fallback to sph.sph_harm)
time:                          0.33s
time factor:                   1.00 ... EVEN
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph.sph_harm_large (force_large)
time:                         13.73s
time factor:                  41.28 ... WORSE
result sum:   5.287666796125258e-15 ... OKAY
result max:   8.568339948907339e-16 ... OKAY

======================
_KIND = 'complex_GumDur'
======================

sph.sph_harm
time:                          0.35s

sph_harm_2
time:                          0.35s
time factor:                   1.02 ... EVEN
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph_harm_3
time:                          0.35s
time factor:                   1.01 ... EVEN
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph_harm_4
time:                          0.38s
time factor:                   1.09 ... WORSE
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph.sph_harm_large (force_large)
time:                         13.69s
time factor:                  39.31 ... WORSE
result sum:     3.1285120834878e-15 ... OKAY
result max:   8.568339948907339e-16 ... OKAY

sph.sph_harm COMPLEX (mismatch expected)
time:                          0.33s
time factor:                   0.95 ... BETTER
result sum:      15.318139446419302 ... MISMATCH
result max:       1.355973823707794 ... MISMATCH

======================
_KIND = 'real'
======================

sph.sph_harm
time:                          0.36s

spaudiopy.sh_matrix
time:                          0.54s
time factor:                   1.51 ... WORSE
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph.sph_harm_large (force_large)
time:                         13.88s
time factor:                  38.46 ... WORSE
result sum:   5.789664227699895e-15 ... OKAY
result max:   9.992007221626409e-16 ... OKAY

======================
_KIND = 'real_Zotter'
======================

sph.sph_harm
time:                          0.37s

sph_harm_2
time:                          0.37s
time factor:                   1.00 ... EVEN
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph_harm_3
time:                          0.39s
time factor:                   1.05 ... WORSE
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT

sph_harm_4
time:                          0.40s
time factor:                   1.08 ... WORSE
result sum:  1.7488523968906112e-15 ... OKAY
result max:  1.1102230246251565e-16 ... OKAY

sph.sph_harm_large (force_large)
time:                         13.71s
time factor:                  37.08 ... WORSE
result sum:   6.429683438309033e-15 ... OKAY
result max:   9.992007221626409e-16 ... OKAY

sph.sph_harm COMPLEX (mismatch expected)
time:                          0.38s
time factor:                   1.03 ... WORSE
result sum:       7.676718890119865 ... MISMATCH
result max:        1.77470334799578 ... MISMATCH

sph.sph_harm COMPLEX (mismatch expected)
time:                          0.36s
time factor:                   0.97 ... BETTER
result sum:      14.423863129737137 ... MISMATCH
result max:      1.4644711794479288 ... MISMATCH
"""

import platform

import numpy as _np
from scipy import special as scy

from sound_field_analysis import sph
from sound_field_analysis.utils import time_it


def sh_matrix(N, azi, colat, SH_type="complex", weights=None):
    """
    From https://github.com/chris-hld/spaudiopy/blob/master/spaudiopy/sph.py

    Notes
    -----
    Imports are not done inside the function in order to not compromise
    execution time (this has a minor impact). Therefore, the `scyspecial`
    and `np` calls have been adapted to the convention used in the other
    functions.
    """
    azi = _np.asarray(azi)
    colat = _np.asarray(colat)
    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if weights is None:
        weights = _np.ones(Q)
    if SH_type == "complex":
        Ymn = _np.zeros([Q, (N + 1) ** 2], dtype=_np.complex_)
    elif SH_type == "real":
        Ymn = _np.zeros([Q, (N + 1) ** 2], dtype=_np.float_)
    else:
        raise ValueError("SH_type unknown.")

    idx = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            if SH_type == "complex":
                Ymn[:, idx] = weights * scy.sph_harm(m, n, azi, colat)
            elif SH_type == "real":
                if m == 0:
                    Ymn[:, idx] = weights * _np.real(scy.sph_harm(0, n, azi, colat))
                if m < 0:
                    Ymn[:, idx] = (
                        weights
                        * _np.sqrt(2)
                        * (-1) ** abs(m)
                        * _np.imag(scy.sph_harm(abs(m), n, azi, colat))
                    )
                if m > 0:
                    Ymn[:, idx] = (
                        weights
                        * _np.sqrt(2)
                        * (-1) ** abs(m)
                        * _np.real(scy.sph_harm(abs(m), n, azi, colat))
                    )

            idx += 1
    return Ymn


def sph_harm_all_func(func, nMax, az, co, kind, force_large=False):
    m, n = sph.mnArrays(nMax)
    mA, azA = _np.meshgrid(m, az)
    nA, coA = _np.meshgrid(n, co)
    if force_large:
        return func(mA, nA, azA, coA, kind=kind, force_large=force_large)
    else:
        return func(mA, nA, azA, coA, kind=kind)


def sph_harm_2(m, n, az, co, kind="complex"):
    kind = kind.lower()
    if "complex" in kind:
        Y = _np.asarray(scy.sph_harm(m, n, az, co))
        if kind in ["complex_gumdur", "complex_sfs"]:
            # apply Condon-Shortley phase also for positive m
            _np.multiply(Y, _np.float_power(-1.0, m, where=m > 0), out=Y, where=m > 0)
        return Y

    else:  # "real"
        Y = _np.asarray(scy.sph_harm(abs(m), n, az, co))
        _np.multiply(Y, _np.sqrt(2), out=Y, where=m > 0)
        _np.multiply(Y.imag, _np.sqrt(2), out=Y, where=m < 0, dtype=Y.dtype)
        if kind in ["real_zotter", "real_akt", "real_sfs"]:
            _np.multiply(Y, -1, out=Y, where=m < 0)  # negate for negative m
        return _np.float_power(-1.0, m) * Y.real


def sph_harm_3(m, n, az, co, kind="complex"):
    kind = kind.lower()
    if "complex" in kind:
        Y = _np.asarray(scy.sph_harm(m, n, az, co))
        if kind in ["complex_gumdur", "complex_sfs"]:
            # apply Condon-Shortley phase also for positive m
            Y[m > 0] *= _np.float_power(-1.0, m[m > 0])
        return Y

    else:  # "real"
        Y = _np.asarray(scy.sph_harm(abs(m), n, az, co))
        Y[m > 0] *= _np.sqrt(2)
        Y[m < 0] = Y.imag[m < 0] * _np.sqrt(2)
        if kind in ["real_zotter", "real_akt", "real_sfs"]:
            Y[m < 0] *= -1  # negate for negative m
        return _np.float_power(-1.0, m) * Y.real


def sph_harm_4(m, n, az, co, kind="complex"):
    kind = kind.lower()
    if "complex" in kind:
        Y = _np.asarray(scy.sph_harm(m, n, 0, co)) * _np.exp(1j * m * az)
        if kind in ["complex_gumdur", "complex_sfs"]:
            # apply Condon-Shortley phase also for positive m
            _np.multiply(Y, _np.float_power(-1.0, m), out=Y, where=m > 0)
        return Y
    else:  # "real"
        Y = _np.asarray(scy.sph_harm(abs(m), n, 0, co))
        _np.multiply(Y, _np.sqrt(2) * _np.cos(m * az), out=Y, where=m > 0)
        _np.multiply(Y, _np.sqrt(2) * _np.sin(abs(m) * az), out=Y, where=m < 0)
        if kind in ["real_zotter", "real_akt", "real_sfs"]:
            _np.multiply(Y, -1, out=Y, where=m < 0)  # negate for negative m
        return _np.float_power(-1.0, m) * Y.real


# set parameters
_TIMEIT_REPEAT = 5
_TIMEIT_NUMBER = 500
_AZ = _np.linspace(start=0, stop=2 * _np.pi, num=50)
_CO = _np.random.uniform(low=0, high=_np.pi, size=_AZ.size)
_N_MAX = 8

print("======================")
print(f'node "{platform.node()}"')
print("======================\n")
print("======================")
print(f"{_TIMEIT_REPEAT = }")
print(f"{_TIMEIT_NUMBER = }")
print(f"{_N_MAX = }")
print("======================\n")

_KIND = "complex"
print("======================")
print(f"{_KIND = }")
print("======================\n")

ref = time_it(
    description="sph.sph_harm",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
time_it(
    description="spaudiopy.sh_matrix",
    stmt="sh_matrix(N=_N_MAX, azi=_AZ, colat=_CO, SH_type=_KIND, weights=None)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slower
time_it(
    description="sph.sph_harm_large (fallback to sph.sph_harm)",
    stmt="sph_harm_all_func(sph.sph_harm_large, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph.sph_harm_large (force_large)",
    stmt="sph_harm_all_func(sph.sph_harm_large, _N_MAX, _AZ, _CO, _KIND, True)",
    setup="",
    _globals=locals(),
    repeat=1,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slowest

_KIND = "complex_GumDur"
print("======================")
print(f"{_KIND = }")
print("======================\n")
ref = time_it(
    description="sph.sph_harm",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
time_it(
    description="sph_harm_2",
    stmt="sph_harm_all_func(sph_harm_2, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph_harm_3",
    stmt="sph_harm_all_func(sph_harm_3, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph_harm_4",
    stmt="sph_harm_all_func(sph_harm_4, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph.sph_harm_large (force_large)",
    stmt="sph_harm_all_func(sph.sph_harm_large, _N_MAX, _AZ, _CO, _KIND, True)",
    setup="",
    _globals=locals(),
    repeat=1,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slowest
time_it(
    description="sph.sph_harm COMPLEX (mismatch expected)",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, 'complex')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # for timing reference (mismatch in case of 'complex_GumDur' kind is expected)


_KIND = "real"
print("======================")
print(f"{_KIND = }")
print("======================\n")

ref = time_it(
    description="sph.sph_harm",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
time_it(
    description="spaudiopy.sh_matrix",
    stmt="sh_matrix(N=_N_MAX, azi=_AZ, colat=_CO, SH_type=_KIND, weights=None)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slower
time_it(
    description="sph.sph_harm_large (force_large)",
    stmt="sph_harm_all_func(sph.sph_harm_large, _N_MAX, _AZ, _CO, _KIND, True)",
    setup="",
    _globals=locals(),
    repeat=1,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slowest

_KIND = "real_Zotter"
print("======================")
print(f"{_KIND = }")
print("======================\n")
ref = time_it(
    description="sph.sph_harm",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
time_it(
    description="sph_harm_2",
    stmt="sph_harm_all_func(sph_harm_2, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph_harm_3",
    stmt="sph_harm_all_func(sph_harm_3, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph_harm_4",
    stmt="sph_harm_all_func(sph_harm_4, _N_MAX, _AZ, _CO, _KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)
time_it(
    description="sph.sph_harm_large (force_large)",
    stmt="sph_harm_all_func(sph.sph_harm_large, _N_MAX, _AZ, _CO, _KIND, True)",
    setup="",
    _globals=locals(),
    repeat=1,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # slowest
time_it(
    description="sph.sph_harm REAL (mismatch expected)",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, 'real')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # for timing reference (mismatch in case of 'real_Zotter' kind is expected)
time_it(
    description="sph.sph_harm COMPLEX (mismatch expected)",
    stmt="sph_harm_all_func(sph.sph_harm, _N_MAX, _AZ, _CO, 'complex')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
)  # for timing reference (mismatch in case of 'real_Zotter' kind is expected)
