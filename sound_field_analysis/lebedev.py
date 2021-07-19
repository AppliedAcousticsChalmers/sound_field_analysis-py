"""
Module to generate Lebedev grids and quadrature weights for degrees 6, 14, 26,
38, 50, 74, 86, 110, 146, 170, 194:

`genGrid(n)`
    Generate Lebedev grid geometry of degree `n`.

Adapted from Richard P. Mullers Python version,
https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py
C version: Dmitri Laikov
F77 version: Christoph van Wuellen, http://www.ccl.net

Users of this code are asked to include reference [3]_ in their publications,
and in the user- and programmers-manuals describing their codes.

References
----------
.. [3] Lebedev, V. I. (1977). “Spherical quadrature formulas exact to orders
25-29,” Sib. Math. J., 18, 99–107. doi:10.1007/BF00966954
"""
from collections import namedtuple

import numpy as _np


# fmt: off
def _genOh_a00(v):
    """(0, 0, a) etc. (6 points)"""
    a = 1.0
    return [(a, 0, 0, v), (-a, 0, 0, v), (0, a, 0, v),
            (0, -a, 0, v), (0, 0, a, v), (0, 0, -a, v)]


def _genOh_aa0(v):
    """(0, a, a) etc, a=1/sqrt(2) (12 points)"""
    a = _np.sqrt(0.5)
    return [(0, a, a, v), (0, -a, a, v), (0, a, -a, v), (0, -a, -a, v),
            (a, 0, a, v), (-a, 0, a, v), (a, 0, -a, v), (-a, 0, -a, v),
            (a, a, 0, v), (-a, a, 0, v), (a, -a, 0, v), (-a, -a, 0, v)]


def _genOh_aaa(v):
    """(a, a, a) etc, a=1/sqrt(3) (8 points)"""
    a = _np.sqrt(1. / 3.)
    return [(a, a, +a, v), (-a, a, +a, v), (a, -a, +a, v), (-a, -a, +a, v),
            (a, a, -a, v), (-a, a, -a, v), (a, -a, -a, v), (-a, -a, -a, v)]


def _genOh_aab(v, a):
    """(a, a, b) etc, b=sqrt(1-2 a^2), a input (24 points)"""
    b = _np.sqrt(1.0 - 2.0 * a * a)
    return [(a, a, +b, v), (-a, a, +b, v), (a, -a, +b, v), (-a, -a, +b, v),
            (a, a, -b, v), (-a, a, -b, v), (a, -a, -b, v), (-a, -a, -b, v),
            (a, b, +a, v), (-a, b, +a, v), (a, -b, +a, v), (-a, -b, +a, v),
            (a, b, -a, v), (-a, b, -a, v), (a, -b, -a, v), (-a, -b, -a, v),
            (b, a, +a, v), (-b, a, +a, v), (b, -a, +a, v), (-b, -a, +a, v),
            (b, a, -a, v), (-b, a, -a, v), (b, -a, -a, v), (-b, -a, -a, v)]


def _genOh_ab0(v, a):
    """(a, b, 0) etc, b=sqrt(1-a^2), a input (24 points)"""
    b = _np.sqrt(1.0 - a * a)
    return [(a, b, 0, v), (-a, b, 0, v), (a, -b, 0, v), (-a, -b, 0, v),
            (b, a, 0, v), (-b, a, 0, v), (b, -a, 0, v), (-b, -a, 0, v),
            (a, 0, b, v), (-a, 0, b, v), (a, 0, -b, v), (-a, 0, -b, v),
            (b, 0, a, v), (-b, 0, a, v), (b, 0, -a, v), (-b, 0, -a, v),
            (0, a, b, v), (0, -a, b, v), (0, a, -b, v), (0, -a, -b, v),
            (0, b, a, v), (0, -b, a, v), (0, b, -a, v), (0, -b, -a, v)]


def _genOh_abc(v, a, b):
    """(a, b, c) etc, c=sqrt(1-a^2-b^2), a, b input  (48 points)"""
    c = _np.sqrt(1.0 - a * a - b * b)
    return [(a, b, +c, v), (-a, b, +c, v), (a, -b, +c, v), (-a, -b, +c, v),
            (a, b, -c, v), (-a, b, -c, v), (a, -b, -c, v), (-a, -b, -c, v),
            (a, c, +b, v), (-a, c, +b, v), (a, -c, +b, v), (-a, -c, +b, v),
            (a, c, -b, v), (-a, c, -b, v), (a, -c, -b, v), (-a, -c, -b, v),
            (b, a, +c, v), (-b, a, +c, v), (b, -a, +c, v), (-b, -a, +c, v),
            (b, a, -c, v), (-b, a, -c, v), (b, -a, -c, v), (-b, -a, -c, v),
            (b, c, +a, v), (-b, c, +a, v), (b, -c, +a, v), (-b, -c, +a, v),
            (b, c, -a, v), (-b, c, -a, v), (b, -c, -a, v), (-b, -c, -a, v),
            (c, a, +b, v), (-c, a, +b, v), (c, -a, +b, v), (-c, -a, +b, v),
            (c, a, -b, v), (-c, a, -b, v), (c, -a, -b, v), (-c, -a, -b, v),
            (c, b, +a, v), (-c, b, +a, v), (c, -b, +a, v), (-c, -b, +a, v),
            (c, b, -a, v), (-c, b, -a, v), (c, -b, -a, v), (-c, -b, -a, v)]
# fmt: on


def _leb6():
    return _genOh_a00(0.1666666666666667)


def _leb14():
    return _genOh_a00(0.06666666666666667) + _genOh_aaa(0.07500000000000000)


def _leb26():
    return (
        _genOh_a00(0.04761904761904762)
        + _genOh_aa0(0.03809523809523810)
        + _genOh_aaa(0.03214285714285714)
    )


def _leb38():
    return (
        _genOh_a00(0.009523809523809524)
        + _genOh_aaa(0.3214285714285714e-1)
        + _genOh_ab0(0.2857142857142857e-1, 0.4597008433809831e0)
    )


def _leb50():
    return (
        _genOh_a00(0.1269841269841270e-1)
        + _genOh_aa0(0.2257495590828924e-1)
        + _genOh_aaa(0.2109375000000000e-1)
        + _genOh_aab(0.2017333553791887e-1, 0.3015113445777636e0)
    )


def _leb74():
    return (
        _genOh_a00(0.5130671797338464e-3)
        + _genOh_aa0(0.1660406956574204e-1)
        + _genOh_aaa(-0.2958603896103896e-1)
        + _genOh_aab(0.2657620708215946e-1, 0.4803844614152614e0)
        + _genOh_ab0(0.1652217099371571e-1, 0.3207726489807764e0)
    )


def _leb86():
    return (
        _genOh_a00(0.1154401154401154e-1)
        + _genOh_aaa(0.1194390908585628e-1)
        + _genOh_aab(0.1111055571060340e-1, 0.3696028464541502e0)
        + _genOh_aab(0.1187650129453714e-1, 0.6943540066026664e0)
        + _genOh_ab0(0.1181230374690448e-1, 0.3742430390903412e0)
    )


def _leb110():
    return (
        _genOh_a00(0.3828270494937162e-2)
        + _genOh_aaa(0.9793737512487512e-2)
        + _genOh_aab(0.8211737283191111e-2, 0.1851156353447362e0)
        + _genOh_aab(0.9942814891178103e-2, 0.6904210483822922e0)
        + _genOh_aab(0.9595471336070963e-2, 0.3956894730559419e0)
        + _genOh_ab0(0.9694996361663028e-2, 0.4783690288121502e0)
    )


def _leb146():
    return (
        _genOh_a00(0.5996313688621381e-3)
        + _genOh_aa0(0.7372999718620756e-2)
        + _genOh_aaa(0.7210515360144488e-2)
        + _genOh_aab(0.7116355493117555e-2, 0.6764410400114264e0)
        + _genOh_aab(0.6753829486314477e-2, 0.4174961227965453e0)
        + _genOh_aab(0.7574394159054034e-2, 0.1574676672039082e0)
        + _genOh_abc(0.6991087353303262e-2, 0.1403553811713183e0, 0.4493328323269557e0)
    )


def _leb170():
    return (
        _genOh_a00(0.5544842902037365e-2)
        + _genOh_aa0(0.6071332770670752e-2)
        + _genOh_aaa(0.6383674773515093e-2)
        + _genOh_aab(0.5183387587747790e-2, 0.2551252621114134e0)
        + _genOh_aab(0.6317929009813725e-2, 0.6743601460362766e0)
        + _genOh_aab(0.6201670006589077e-2, 0.4318910696719410e0)
        + _genOh_ab0(0.5477143385137348e-2, 0.2613931360335988e0)
        + _genOh_abc(0.5968383987681156e-2, 0.4990453161796037e0, 0.1446630744325115e0)
    )


def _leb194():
    return (
        _genOh_a00(0.1782340447244611e-2)
        + _genOh_aa0(0.5716905949977102e-2)
        + _genOh_aaa(0.5573383178848738e-2)
        + _genOh_aab(0.5608704082587997e-2, 0.6712973442695226e0)
        + _genOh_aab(0.5158237711805383e-2, 0.2892465627575439e0)
        + _genOh_aab(0.5518771467273614e-2, 0.4446933178717437e0)
        + _genOh_aab(0.4106777028169394e-2, 0.1299335447650067e0)
        + _genOh_ab0(0.5051846064614808e-2, 0.3457702197611283e0)
        + _genOh_abc(0.5530248916233094e-2, 0.1590417105383530e0, 0.8360360154824589e0)
    )


LebFunc = {
    6: _leb6,
    14: _leb14,
    26: _leb26,
    38: _leb38,
    50: _leb50,
    74: _leb74,
    86: _leb86,
    110: _leb110,
    146: _leb146,
    170: _leb170,
    194: _leb194,
}


def genGrid(n):
    """Generate Lebedev grid geometry of degree `n`.

    Parameters
    ----------
    n : int{6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194}
       Lebedev degree

    Returns
    -------
    lebGrid : named tuple
       Named tuple to store `x`, `y`, `z` cartesian coordinates and quadrature
       weights `w`

    Raises
    ------
    ValueError
       in case no grid could be generated for given degree
    """
    try:
        leb = _np.array(LebFunc[n]())  # try retrieving grid first
    except KeyError:
        raise ValueError(f"No grid available for degree {n}")

    lebGrid = namedtuple("lebGrid", "x y z w")
    lebGrid.x = leb[:, 0]
    lebGrid.y = leb[:, 1]
    lebGrid.z = leb[:, 2]
    lebGrid.w = leb[:, 3]
    return lebGrid
