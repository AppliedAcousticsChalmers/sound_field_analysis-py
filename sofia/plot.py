"""Plotting functions
- makeMTX: Generate 3D-matrix data
- visualize3D: Draw matrix data in 3D
"""
import numpy as _np
from collections import namedtuple

from plotly.offline import plot as pltoff
import plotly.graph_objs as go

from .process import pdc
from .sph import sph2cart

pi = _np.pi


def show(trace, colorize=True):
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
    return vizMTX


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
        return genSphere(vizMTX)
    else:
        raise ValueError('Provided style "' + style + '" not available. Try sphere, shape or flat.')


def visualize3D(vizMTX, style='sphere', colorize=True, offset=0., scale=1., **kargs):
    """Visualize matrix data, such as from makeMTX(Pnm, dn)

    Parameters
    ----------
    vizMTX     SOFiA 3D-matrix-data [1[deg] steps]
    style       'sphere',   surface colors indicate the intensity (default)
                'flat',     surface colors indicate the intensity (TODO)
                'scatter',  extension indicates the intensity
                'shape',    extension indicates the intensity
    offset      linear offset for shape (Default: 0)
    scale       scaling factor for shape (Default: 1)

    Notes
    -----
    TODO: Implement flat style, fix color position in sphere and shape, make colormap selectable, move grid generation into function
    """

    # Prepare data: reshape to [65160 x 1], take abs, normalize
    vizMTX = _np.abs(vizMTX.reshape((65160)))
    vizMTX = normalizeMTX(vizMTX)

    if style not in ('sphere', 'flat', 'shape', 'scatter'):
        raise ValueError('Provided style "' + style + '" not available. Try sphere, flat, shape or scatter.')

    # Create scene
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')

    # Create view with camera on target
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.set_range(x=[-0.1, 0.1])

    # Create correct visual object from mtxData
    visObj = genVisual(vizMTX, style=style, colorize=colorize, offset=offset, scale=scale)

    # Add visual object and show canvas
    view.add(visObj)
    canvas.show()

    return canvas


def plotGrid(rows, cols, vizMTX, bgcolor='white', style='shape', colorize=False, normalize=True):
    canvas = scene.SceneCanvas(keys='interactive', bgcolor=bgcolor)

    vizMTX = _np.atleast_3d(vizMTX)
    N = vizMTX.shape[0]
    if rows * cols != N:
        raise ValueError('rows (' + str(rows) + ') * cols (' + str(cols) + ') must be number of objects (' + str(N) + ').')

    # Top-level grid that holds subfigures
    grid = canvas.central_widget.add_grid()

    # Add ViewBoxes to the grid
    for row in range(0, rows):
        for col in range(0, cols):
            temp = grid.add_view(row=row, col=col, border_color=(0.5, 0.5, 0.5, 1), camera='turntable')
            temp.add(genVisual(vizMTX[col + cols * row], style=style, colorize=colorize, normalize=normalize))

    canvas.show()
    return canvas


def generateAngles():
    """Returns a [65160 x 1] grid of all radiant angles in 1 deg steps"""
    return _np.mgrid[0:360, 0:181].T.reshape((-1, 2)) * _np.pi / 180
