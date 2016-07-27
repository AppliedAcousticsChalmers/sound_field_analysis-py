"""Plotting functions
Helps visualizing spherical microphone data.

Generally, you probably want to first extract the amplitude information in spherical coordinates:
>> plot.makeMTX(Pnm, dn, Nviz=3, krIndex=1, oversize=1)
And then visualize that:
>> visualize3D(vizMTX, style='shape')

Other valid styles are 'sphere' and 'flat'.
"""
import numpy as _np
from collections import namedtuple

from plotly.offline import plot as pltoff
import plotly.graph_objs as go

from .process import pdc

pi = _np.pi


def showTrace(trace, colorize=True):
    """ Wrapper around plotlys offline .plot() function

    Parameters
    ----------
    trace : plotly_trace
       Plotly generated trace to be displayed offline
    colorize : Bool, optional
       Toggles bw / colored plot [Default: True]
    """
    data = [trace]

    # if colorize:
    #    data[0].autocolorscale = False
    #    data[0].surfacecolor = [0, 0.5, 1]
    pltoff(data)


def makeMTX(Pnm, dn, Nviz=3, krIndex=1, oversize=1):
    """mtxData = makeMTX(Nviz=3, Pnm, dn, krIndex)

    Parameters
    ----------
    Pnm       Spatial Fourier Coefficients (from S/T/C)
    dn        Modal Radial Filters (from M/F)
    N         Order of the spatial fourier transform     [default = 3]
    krIndex   Index of kr Vector                         [default = 1]
    oversize  Integer Factor to increase the resolution. Set oversize = 1
              (default) to use the mtxData matrix for visual3D(), map3D().
    #Returns
    -------
    mtxData   3D-matrix-data in 1[deg] steps

    Notes
    -----
    The file generates a SOFiA mtxData Matrix of 181x360 pixels for the
    visualisation with visualize3D() in 1[deg] Steps (65160 plane waves).
    The HD version generally allows to raise the resolution (oversize > 1).
    (visual3D(), map3D() admit 1[deg] data only, oversize = 1)
    """
    if oversize < 1:
        raise ValueError('Oversize parameter must be >= 1')

    # Generate angles for sphere with 1[deg] spacing
    angles = _np.mgrid[0:360, 0:181].T.reshape((-1, 2)) * _np.pi / 180

    # Compute plane wave decomposition for all angles at given kr
    Y = pdc(Nviz, angles, Pnm[:, krIndex], dn[:, krIndex])

    return Y.reshape((181, -1))  # Return pwd data as [181, 360] matrix


def normalizeMTX(MTX):
    """ Normalizes a matrix to [-1 ... 1]

    Parameters
    ----------
    MTX : array_like
       Matrix to be normalized

    Returns
    -------
    MTX : array_liked
       Normalized Matrix
    """
    MTX -= MTX.min()
    return MTX / MTX.max()


def genSphCoords():
    """ Generates cartesian (x,y,z) and spherical (theta, phi) coordinates of a sphere
    Returns
    -------
    coords : named tuple
        holds cartesian (x,y,z) and spherical (theta, phi) coordinates
    """
    coords = namedtuple('coords', ['x', 'y', 'z', 'theta', 'phi'])
    theta = _np.linspace(0, 2 * pi, 360)
    phi = _np.linspace(0, pi, 181)
    coords.x = _np.outer(_np.cos(theta), _np.sin(phi))
    coords.y = _np.outer(_np.sin(theta), _np.sin(phi))
    coords.z = _np.outer(_np.ones(360), _np.cos(phi))

    coords.theta, coords.phi = _np.meshgrid(_np.linspace(0, _np.pi, 181), _np.linspace(0, 2 * _np.pi, 360))
    return coords


def sph2cartMTX(vizMTX):
    """ Converts the spherical vizMTX data to named tuple contaibubg .xs/.ys/.zs

    Parameters
    ----------
    vizMTX : array_like
       [180 x 360] matrix that hold amplitude information over phi and theta

    Returns
    -------
    V : named_tuple
       Contains .xs, .ys, .zs cartesian coordinates
    """
    rs = vizMTX.reshape((181, -1)).T

    coords = genSphCoords()
    V = namedtuple('V', ['xs', 'ys', 'zs'])
    V.xs = rs * _np.sin(coords.theta) * _np.cos(coords.phi)
    V.ys = rs * _np.sin(coords.theta) * _np.sin(coords.phi)
    V.zs = rs * _np.cos(coords.theta)
    return V


def genShape(vizMTX):
    """ Returns trace of shape with intensity as radial extension

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization

    Returns
    -------
    T : plotly_trace
       Trace of desired shape

    TODO: fix camera position
    """
    V = sph2cartMTX(vizMTX)

    trace = go.Surface(
        x=_np.abs(V.xs),
        y=_np.abs(V.ys),
        z=_np.abs(V.zs),
        surfacecolor=_np.abs(vizMTX.reshape((181, -1))).T,
        colorscale='Viridis',
        showscale=False,
        hoverinfo='none'
    )
    return trace


def genSphere(vizMTX):
    """ Returns trace of sphere with intensity as surface color

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization

    Returns
    -------
    T : plotly_trace
       Trace of desired sphere
    """
    coords = genSphCoords()

    trace = go.Surface(
        x=coords.x,
        y=coords.y,
        z=coords.z,
        surfacecolor=_np.abs(vizMTX.reshape((181, -1))).T,
        colorscale='Viridis',
        showscale=False,
        hoverinfo='none'
    )
    return trace


def genFlat(vizMTX):
    """ Returns trace of flat surface with intensity as surface elevation and color

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization

    Returns
    -------
    T : plotly_trace
       Trace of desired surface

    TODO: Fix orientation and axis limits
    """

    trace = go.Surface(
        x=_np.r_[0:360],
        y=_np.r_[0:180],
        z=_np.abs(vizMTX).T,
        surfacecolor=_np.abs(vizMTX.reshape((181, -1))).T,
        colorscale='Viridis',
        showscale=False,
        hoverinfo='none'
    )
    return trace


def genVisual(vizMTX, style='shape', normalize=True):
    """ Returns desired trace after cleaning the data

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization
    style : string{'shape', 'sphere', 'flat'}, optional
       Style of visualization. [Default: 'Shape']
    normalize : Bool, optional
       Toggle normalization of data to [-1 ... 1] [Default: True]

    Returns
    -------
    T : plotly_trace
       Trace of desired visualization
    """
    vizMTX = _np.abs(vizMTX)  # Can we be sure to only need the abs?
    if normalize:
        vizMTX = normalizeMTX(vizMTX)

    if style == 'shape':
        return genShape(vizMTX)
    elif style == 'sphere':
        return genSphere(vizMTX)
    elif style == 'flat':
        return genFlat(vizMTX)
    else:
        raise ValueError('Provided style "' + style + '" not available. Try sphere, shape or flat.')


def visualize3D(vizMTX, style='shape', colorize=True):
    """Visualize matrix data, such as from makeMTX(Pnm, dn)

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization
    style : string{'shape', 'sphere', 'flat'}, optional
       Style of visualization. [Default: 'shape']
    normalize : Bool, optional
       Toggle normalization of data to [-1 ... 1] [Default: True]

    # TODO: Colorization, contour plot
    -----
    """

    showTrace(genVisual(vizMTX, style=style, normalize=True))
