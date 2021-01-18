"""This test the equality and execution speed of different implementations to
spatially decompose a sound field.

Exemplary execution:
======================
_TIMEIT_REPEAT = 10
_TIMEIT_NUMBER = 60
_FILE = ../examples/data/CR1_VSA_110RS_L_struct.mat
_ORDER_MAX = 8
_NFFT = 8192
======================
node "C18TTLT"
======================

spatFT weighted complex
time:                          0.73s
result dtype:            complex128 ... MATCH

spatFT pinv complex
time:                          0.81s
time factor:                   1.10 ... WORSE
result sum:      2.3582693394716214 ... MISMATCH
result max:      0.2972262194517645 ... MISMATCH
result dtype:            complex128 ... MATCH

spatFT weighted real
time:                          0.67s
time factor:                   0.92 ... BETTER
result sum:      16.248586482874316 ... MISMATCH
result max:      14.309612031760384 ... MISMATCH
result dtype:            complex128 ... MATCH

spatFT_RT complex
time:                          0.61s
time factor:                   0.84 ... BETTER
result sum:                     0.0 ... PERFECT
result max:                     0.0 ... PERFECT
result dtype:            complex128 ... MATCH

spatFT_RT real
time:                          0.60s
time factor:                   0.82 ... BETTER
result sum:      16.248586482874316 ... MISMATCH
result max:      14.309612031760384 ... MISMATCH
result dtype:            complex128 ... MATCH
"""

import platform

import numpy as np

from sound_field_analysis import io
from sound_field_analysis.process import FFT, spatFT, spatFT_RT
from sound_field_analysis.sph import sph_harm_all
from sound_field_analysis.utils import time_it


def _spatFT(ir, order_max, kind):
    return spatFT(
        FFT(ir.signal.signal, fs=ir.signal.fs, calculate_freqs=False),
        position_grid=ir.grid,
        order_max=order_max,
        kind=kind,
    )


def _spatFT_RT(ir, bases):
    return spatFT_RT(
        FFT(ir.signal.signal, fs=ir.signal.fs, calculate_freqs=False),
        spherical_harmonic_weighted=bases,
    )


# set parameters
_TIMEIT_REPEAT = 10
_TIMEIT_NUMBER = 60
(_FILE, _ORDER_MAX, _NFFT) = ("../examples/data/CR1_VSA_110RS_L_struct.mat", 8, 8192)

print("======================")
print(f"_TIMEIT_REPEAT = {_TIMEIT_REPEAT}")
print(f"_TIMEIT_NUMBER = {_TIMEIT_NUMBER}")
print(f"_FILE = {_FILE}")
print(f"_ORDER_MAX = {_ORDER_MAX}")
print(f"_NFFT = {_NFFT}")
print("======================")
print(f'node "{platform.node()}"')
print("======================\n")

# load MIRO data
ir = io.read_miro_struct(_FILE)

# truncate impulse response to desired length
ir = io.ArraySignal(
    signal=io.TimeSignal(
        signal=ir.signal.signal[:, :_NFFT], fs=ir.signal.fs, delay=ir.signal.delay
    ),
    grid=ir.grid,
    center_signal=ir.center_signal,
    configuration=ir.configuration,
    temperature=ir.temperature,
)

# generate version of the data set without quadrature weights, which will
# require the computation of a pseudo-inverse matrix in `spatFT()`
ir_pinv = io.ArraySignal(
    signal=ir.signal,
    grid=io.SphericalGrid(
        azimuth=ir.grid.azimuth,
        colatitude=ir.grid.colatitude,
        radius=ir.grid.radius,
        weight=None,
    ),
)

# generate complex and real spherical harmonic base functions (for `spatFT_RT()`)
bases_c = np.conj(
    sph_harm_all(_ORDER_MAX, ir.grid.azimuth, ir.grid.colatitude, kind="complex")
).T * (4 * np.pi * ir.grid.weight)
bases_r = np.conj(
    sph_harm_all(_ORDER_MAX, ir.grid.azimuth, ir.grid.colatitude, kind="real")
).T * (4 * np.pi * ir.grid.weight)

ref = time_it(
    description="spatFT weighted complex",
    stmt="_spatFT(ir, _ORDER_MAX, 'complex')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    check_dtype="complex128",
)

time_it(
    description="spatFT pinv complex",
    stmt="_spatFT(ir_pinv, _ORDER_MAX, 'complex')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
    check_dtype="complex128",
)  # slower (expected due to pseudo-inverse matrix operation being more
# expensive), data mismatch (expected due to minor differences according to
# the weighting method)

time_it(
    description="spatFT weighted real",
    stmt="_spatFT(ir, _ORDER_MAX, 'real')",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
    check_dtype="complex128",
)  # slightly faster, data mismatch (expected due to the different SH
# convention)

time_it(
    description="spatFT_RT complex",
    stmt="_spatFT_RT(ir_pinv, bases_c)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
    check_dtype="complex128",
)  # fastest (expected due to skipping generation of the SH base functions)

time_it(
    description="spatFT_RT real",
    stmt="_spatFT_RT(ir_pinv, bases_r)",
    setup="",
    _globals=locals(),
    repeat=_TIMEIT_REPEAT,
    number=_TIMEIT_NUMBER,
    reference=ref,
    check_dtype="complex128",
)  # fastest (expected due to skipping generation of the SH base functions),
# data mismatch (expected due to the different SH convention)
