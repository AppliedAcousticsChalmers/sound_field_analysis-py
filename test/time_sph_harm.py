"""This test the equality and execution speed of different implementations to
generate spherical harmonics.
"""

import platform
import sys
import timeit
from time import sleep

import numpy as _np
from scipy import special as scy

from sound_field_analysis.sph import mnArrays


def _timeit(description, stmt, setup, _globals, repeat, number, reference=None):
    print(description)

    # do timing and get results
    result = timeit.Timer(stmt=stmt, setup=setup, globals=_globals) \
        .repeat(repeat=repeat, number=number)
    result = (min(list(zip(*result))[0]), result[0][1])
    # (time, computation result)

    # print conclusion
    print(f"time: {result[0]:-29.2f}s")
    sleep(0.05)  # to get correct output order

    if reference:
        t = result[0] / reference[0]
        file = sys.stdout
        if abs(t - 1) < 2e-2:
            grade = "EVEN"
        elif t < 1:
            grade = "BETTER"
        else:
            grade = "WORSE"
            file = sys.stderr
        print(f"time factor: {t:-22.2f} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

        # flip computation result, if matrices do not match
        if reference[1].shape != result[1].shape:
            reference = (reference[0], reference[1].T)
        if reference[1].shape != result[1].shape:
            print(f'result: {"":22} {"DIMENSION MISMATCH"}', file=sys.stderr)
            sleep(0.05)  # to get correct output order
            print()
            return result

        r = _np.abs(_np.sum(_np.subtract(result[1], reference[1])))
        file = sys.stdout
        if r == 0:
            grade = "PERFECT"
        elif r < 1e-10:
            grade = "OKAY"
        else:
            grade = "MISMATCH"
            file = sys.stderr
        print(f"result sum:  {r:-22} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

        r = _np.abs(_np.subtract(result[1], reference[1])).max()
        file = sys.stdout
        if r == 0:
            grade = "PERFECT"
        elif r < 1e-10:
            grade = "OKAY"
        else:
            grade = "MISMATCH"
            file = sys.stderr
        print(f"result max:  {r:-22} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

    print()
    return result


def sh_matrix(N, azi, colat, SH_type='complex', weights=None):
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
    if SH_type == 'complex':
        Ymn = _np.zeros([Q, (N + 1) ** 2], dtype=_np.complex_)
    elif SH_type == 'real':
        Ymn = _np.zeros([Q, (N + 1) ** 2], dtype=_np.float_)
    else:
        raise ValueError('SH_type unknown.')

    idx = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            if SH_type == 'complex':
                Ymn[:, idx] = weights * scy.sph_harm(m, n, azi, colat)
            elif SH_type == 'real':
                if m == 0:
                    Ymn[:, idx] = weights * _np.real(
                        scy.sph_harm(0, n, azi, colat))
                if m < 0:
                    Ymn[:, idx] = weights * _np.sqrt(2) * (-1) ** abs(m) * \
                                  _np.imag(
                                      scy.sph_harm(abs(m), n, azi, colat))
                if m > 0:
                    Ymn[:, idx] = weights * _np.sqrt(2) * (-1) ** abs(m) * \
                                  _np.real(
                                      scy.sph_harm(abs(m), n, azi, colat))

            idx += 1
    return Ymn


def sph_harm_all_func(func, _N_MAX, az, co, kind='complex'):
    m, n = mnArrays(_N_MAX)
    mA, azA = _np.meshgrid(m, az)
    nA, coA = _np.meshgrid(n, co)
    return func(mA, nA, azA, coA, kind=kind)


def sph_harm_1(m, n, az, co, kind='complex'):
    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        Y[_np.where(m > 0)] = (_np.float_power(-1.0, m)[_np.where(m > 0)]
                               * _np.sqrt(2) * _np.real(Y[_np.where(m > 0)]))
        Y[_np.where(m == 0)] = _np.real(Y[_np.where(m == 0)])
        Y[_np.where(m < 0)] = _np.sqrt(2) * _np.imag(Y[_np.where(m < 0)])
        return _np.real(Y)


def sph_harm_2(m, n, az, co, kind='complex'):
    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        Y[_np.where(m > 0)] = (_np.float_power(-1.0, m)[_np.where(m > 0)]
                               * _np.sqrt(2) * _np.real(Y[_np.where(m > 0)]))
        Y[_np.where(m < 0)] = _np.sqrt(2) * _np.imag(Y[_np.where(m < 0)])
        return _np.real(Y)


def sph_harm_3(m, n, az, co, kind='complex'):
    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        Y[m > 0] = (_np.float_power(-1.0, m)[m > 0] * _np.sqrt(2)
                    * _np.real(Y[m > 0]))
        Y[m < 0] = _np.sqrt(2) * _np.imag(Y[m < 0])
        return _np.real(Y)


def sph_harm_4(m, n, az, co, kind='complex'):
    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        mg0 = m > 0
        ml0 = m < 0
        Y[mg0] = _np.float_power(-1.0, m)[mg0] * _np.sqrt(2) * _np.real(Y[mg0])
        Y[ml0] = _np.sqrt(2) * _np.imag(Y[ml0])
        return _np.real(Y)


def sph_harm_5(m, n, az, co, kind='complex'):
    Y = scy.sph_harm(m, n, az, co)
    if kind == 'complex':
        return Y
    else:  # kind == 'real'
        mg0 = m > 0
        me0 = m == 0
        ml0 = m < 0
        Y_real = _np.zeros(Y.shape, dtype=_np.float_)
        Y_real[mg0] = (_np.float_power(-1.0, m)[mg0] * _np.sqrt(2)
                       * _np.real(Y[mg0]))
        Y_real[me0] = _np.real(Y[me0])
        Y_real[ml0] = _np.sqrt(2) * _np.imag(Y[ml0])
        return Y_real


# modify timer template to also receive evaluation results from `timeit` function call
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        result = {stmt}
    _t1 = _timer()
    return _t1 - _t0, result  # time, computation result
"""

# set parameters
_TIMEIT_REPEAT = 5
_TIMEIT_NUMBER = 5000
(_N_MAX, _AZ, _CO, _KIND) = (8, 0.1, 0.1, 'real')

print("======================")
print(f"_TIMEIT_REPEAT = {_TIMEIT_REPEAT}")
print(f"_TIMEIT_NUMBER = {_TIMEIT_NUMBER}")
print(f"_N_MAX = {_N_MAX}")
print(f"_KIND = {_KIND}")
print("======================")
print(f'node "{platform.node()}"')
print("======================\n")

ref = _timeit(
    description="sph_harm_1",
    stmt="sph_harm_all_func(sph_harm_1, _N_MAX, _AZ, _CO, kind=_KIND)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)

_timeit(
    description="spaudiopy.sh_matrix",
    stmt="sh_matrix(N=_N_MAX, azi=_AZ, colat=_CO, SH_type=_KIND, weights=None)",
    setup="",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)  # slowest, not sure if mismatch is due to a bug or a different convention

_timeit(
    description="pyshtools",
    stmt="""\
result = spharm(lmax=_N_MAX, theta=_CO, phi=_AZ, kind=_KIND, degrees=False,
                normalization='ortho', csphase=1, packed=False)
result = SHCilmToVector(result)[_np.newaxis, :]""",
    setup="""\
from pyshtools.expand import spharm
from pyshtools.shio import SHCilmToVector""",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)  # fastest, but does not result in similar coefficient order yet
"""
Notes
-----
pyshtools has a lot of additional dependencies that are not required for the
purpose of generating the spherical harmonic basis functions. The following
dependencies can be added to the environment.yml for a minimal setup.

channels:
  - defaults
  - conda-forge
dependencies:
  - pyshtools [--no-deps]  # to not get tons of additional dependencies
  - openblas  # required dependency for pyshtools
"""

_timeit(
    description="sph_harm_2",
    stmt="sph_harm_all_func(sph_harm_2, _N_MAX, _AZ, _CO, kind=_KIND)",
    setup="",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
_timeit(
    description="sph_harm_3",
    stmt="sph_harm_all_func(sph_harm_3, _N_MAX, _AZ, _CO, kind=_KIND)",
    setup="",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
_timeit(
    description="sph_harm_4",
    stmt="sph_harm_all_func(sph_harm_4, _N_MAX, _AZ, _CO, kind=_KIND)",
    setup="",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)  # fastest (apart from pyshtools(
_timeit(
    description="sph_harm_5",
    stmt="sph_harm_all_func(sph_harm_5, _N_MAX, _AZ, _CO, kind=_KIND)",
    setup="",
    _globals=locals(),
    reference=ref,
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)
_timeit(
    description="sph_harm COMPLEX",
    stmt="sph_harm_all_func(sph_harm_1, _N_MAX, _AZ, _CO, kind='complex')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
)  # for timing reference (mismatch in case of 'real' kind is expected)

# ======================
# _TIMEIT_REPEAT = 5
# _TIMEIT_NUMBER = 5000
# _N_MAX = 8
# _KIND = real
# ======================
# node "C18TTLT"
# ======================
#
# sph_harm_1
# time:                          0.88s
#
# spaudiopy.sh_matrix
# time:                          3.98s
# time factor:                   4.54 ... WORSE
# result sum:      0.7027539767877192 ... MISMATCH
# result max:     0.12727637177922763 ... MISMATCH
#
# pyshtools
# time:                          0.52s
# time factor:                   0.60 ... BETTER
# result sum:       0.702753976787717 ... MISMATCH
# result max:      0.9629001108439836 ... MISMATCH
#
# sph_harm_2
# time:                          0.63s
# time factor:                   0.72 ... BETTER
# result sum:                     0.0 ... PERFECT
# result max:                     0.0 ... PERFECT
#
# sph_harm_3
# time:                          0.57s
# time factor:                   0.65 ... BETTER
# result sum:                     0.0 ... PERFECT
# result max:                     0.0 ... PERFECT
#
# sph_harm_4
# time:                          0.55s
# time factor:                   0.62 ... BETTER
# result sum:                     0.0 ... PERFECT
# result max:                     0.0 ... PERFECT
#
# sph_harm_5
# time:                          0.56s
# time factor:                   0.64 ... BETTER
# result sum:                     0.0 ... PERFECT
# result max:                     0.0 ... PERFECT
#
# sph_harm COMPLEX
# time:                          0.40s
