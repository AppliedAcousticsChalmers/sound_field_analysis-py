"""Generate Lebedev grid and coefficients
This module only exposes the function  `lebGrid = lebedev.genGrid(degree)`.

lebGrid is a named tuple containing the coordinates .x, .y, .z and the weights .w
Possible degrees: 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194

Adapted from Richard P. Mullers Python version,
https://github.com/gabrielelanaro/pyquante/blob/master/Data/lebedev_write.py
C version: Dmitri Laikov
F77 version: Christoph van Wuellen, http://www.ccl.net

Users of this code are asked to include reference [1] in their publications,
and in the user- and programmers-manuals describing their codes.

[1] V.I. Lebedev, and D.N. Laikov
'A quadrature formula for the sphere of the 131st algebraic order of accuracy'
Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
"""

import numpy as _np


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


def _leb6():
    return _genOh_a00(0.1666666666666667)


def _leb14():
    return _genOh_a00(0.06666666666666667)\
        + _genOh_aaa(0.07500000000000000)


def _leb26():
    return _genOh_a00(0.04761904761904762)\
        + _genOh_aa0(0.03809523809523810) \
        + _genOh_aaa(0.03214285714285714)


def _leb38():
    return _genOh_a00(0.009523809523809524)\
        + _genOh_aaa(0.3214285714285714E-1) \
        + _genOh_ab0(0.2857142857142857E-1, 0.4597008433809831E+0)


def _leb50():
    return _genOh_a00(0.1269841269841270E-1)\
        + _genOh_aa0(0.2257495590828924E-1) \
        + _genOh_aaa(0.2109375000000000E-1) \
        + _genOh_aab(0.2017333553791887E-1, 0.3015113445777636E+0)


def _leb74():
    return _genOh_a00(0.5130671797338464E-3)\
        + _genOh_aa0(0.1660406956574204E-1) \
        + _genOh_aaa(-0.2958603896103896E-1) \
        + _genOh_aab(0.2657620708215946E-1, 0.4803844614152614E+0) \
        + _genOh_ab0(0.1652217099371571E-1, 0.3207726489807764E+0)


def _leb86():
    return _genOh_a00(0.1154401154401154E-1) \
        + _genOh_aaa(0.1194390908585628E-1) \
        + _genOh_aab(0.1111055571060340E-1, 0.3696028464541502E+0) \
        + _genOh_aab(0.1187650129453714E-1, 0.6943540066026664E+0) \
        + _genOh_ab0(0.1181230374690448E-1, 0.3742430390903412E+0)


def _leb110():
    return _genOh_a00(0.3828270494937162E-2) \
        + _genOh_aaa(0.9793737512487512E-2) \
        + _genOh_aab(0.8211737283191111E-2, 0.1851156353447362E+0) \
        + _genOh_aab(0.9942814891178103E-2, 0.6904210483822922E+0) \
        + _genOh_aab(0.9595471336070963E-2, 0.3956894730559419E+0) \
        + _genOh_ab0(0.9694996361663028E-2, 0.4783690288121502E+0)


def _leb146():
    return _genOh_a00(0.5996313688621381E-3) \
        + _genOh_aa0(0.7372999718620756E-2) \
        + _genOh_aaa(0.7210515360144488E-2) \
        + _genOh_aab(0.7116355493117555E-2, 0.6764410400114264E+0) \
        + _genOh_aab(0.6753829486314477E-2, 0.4174961227965453E+0) \
        + _genOh_aab(0.7574394159054034E-2, 0.1574676672039082E+0) \
        + _genOh_abc(0.6991087353303262E-2, 0.1403553811713183E+0, 0.4493328323269557E+0)


def _leb170():
    return _genOh_a00(0.5544842902037365E-2) \
        + _genOh_aa0(0.6071332770670752E-2) \
        + _genOh_aaa(0.6383674773515093E-2) \
        + _genOh_aab(0.5183387587747790E-2, 0.2551252621114134E+0) \
        + _genOh_aab(0.6317929009813725E-2, 0.6743601460362766E+0) \
        + _genOh_aab(0.6201670006589077E-2, 0.4318910696719410E+0) \
        + _genOh_ab0(0.5477143385137348E-2, 0.2613931360335988E+0) \
        + _genOh_abc(0.5968383987681156E-2, 0.4990453161796037E+0, 0.1446630744325115E+0)


def _leb194():
    return _genOh_a00(0.1782340447244611E-2) \
        + _genOh_aa0(0.5716905949977102E-2) \
        + _genOh_aaa(0.5573383178848738E-2) \
        + _genOh_aab(0.5608704082587997E-2, 0.6712973442695226E+0) \
        + _genOh_aab(0.5158237711805383E-2, 0.2892465627575439E+0) \
        + _genOh_aab(0.5518771467273614E-2, 0.4446933178717437E+0) \
        + _genOh_aab(0.4106777028169394E-2, 0.1299335447650067E+0) \
        + _genOh_ab0(0.5051846064614808E-2, 0.3457702197611283E+0) \
        + _genOh_abc(0.5530248916233094E-2, 0.1590417105383530E+0, 0.8360360154824589E+0)


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
    194: _leb194
}


def genGrid(n):
    """Returns Lebedev coefficients of n'th degree

    Parameters
    ----------
    n : int{6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194}
       Lebedev degree

    Returns
    -------
    lebGrid : named tuple
       lebGrid is a named tuple containing .x, .y, .z and .w

    Raises
    ------
    ValueError
       in case no grid could be generated for given degree
    """
    try:
        leb = _np.array(LebFunc[n]())  # try retrieving grid first

        from collections import namedtuple
        lebGrid = namedtuple('lebGrid', 'x y z w')
        lebGrid.x = leb[:, 0]
        lebGrid.y = leb[:, 1]
        lebGrid.z = leb[:, 2]
        lebGrid.w = leb[:, 3]
        return lebGrid
    except KeyError:
        raise ValueError('No grid available for degree', n)
