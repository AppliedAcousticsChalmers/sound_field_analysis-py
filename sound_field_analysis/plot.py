"""Plotting functions
Helps visualizing spherical microphone data.

Generally, you probably want to first extract the amplitude information in spherical coordinates:
>> plot.makeMTX(Pnm, dn, Nviz=3, krIndex=1, oversize=1)
And then visualize that:
>> plot3D(vizMTX, style='shape')

Other valid styles are 'sphere' and 'flat'.
"""
import numpy as _np
from collections import namedtuple

from plotly.offline import plot as pltoff
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly import tools

from .process import PWDecomp
from .utils import env_info, progress_bar

pi = _np.pi


def showTrace(trace, layout=None, colorize=True):
    """ Wrapper around plotlys offline .plot() function

    Parameters
    ----------
    trace : plotly_trace
       Plotly generated trace to be displayed offline
    colorize : Bool, optional
       Toggles bw / colored plot [Default: True]

    Returns
    -------
    fig : plotly_fig_handle
       JSON representation of generated figure
    """
    if not layout:
        layout = go.Layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                aspectmode='cube'
            )
        )
    # Wrap trace in array if needed
    if not isinstance(trace, list):
        trace = [trace]

    fig = go.Figure(
        data=trace,
        layout=layout
    )

    # if colorize:
    #    data[0].autocolorscale = False
    #    data[0].surfacecolor = [0, 0.5, 1]
    if env_info() == 'jupyter_notebook':
        iplot(fig)
    else:
        pltoff(fig)

    return fig


def makeMTX(Pnm, dn, krIndex=1, Nviz=3, oversize=1):
    """mtxData = makeMTX(Nviz=3, Pnm, dn, krIndex)

    Parameters
    ----------
    Pnm : array_like
       Spatial Fourier Coefficients (e.g. from S/T/C)
    dn : array_like
       Modal Radial Filters (e.g. from M/F)
    krIndex : int
       Index of kr to be computed [Default: 1]
    Nviz : int, optional
       Order of the spatial fourier transform [Default: 3]
    oversize : int, optional
       Integer Factor to increase the resolution. [Default: 1]

    Returns
    -------
    mtxData : array_like
       3D-matrix-data in 1[deg] steps

    Note
    ----
    The file generates a Matrix of 181x360 pixels for the
    visualisation with visualize3D() in 1[deg] Steps (65160 plane waves).
    The HD version generally allows to raise the resolution (oversize > 1).
    (visual3D(), map3D() admit 1[deg] data only, oversize = 1)
    """
    if oversize < 1:
        raise ValueError('Oversize parameter must be >= 1')

    # Generate angles for sphere with 1[deg] spacing
    angles = _np.mgrid[0:360, 0:181].T.reshape((-1, 2)) * _np.pi / 180

    # Compute plane wave decomposition for all angles at given kr
    Y = PWDecomp(Nviz, angles, Pnm[:, krIndex], dn[:, krIndex])

    return Y.reshape((181, -1))  # Return pwd data as [181, 360] matrix


def makeFullMTX(Pnm, dn, kr, Nviz=3):
    """ Generates visualization matrix for a set of spatial fourier coefficients over all kr
    Parameters
    ----------
    Pnm : array_like
       Spatial Fourier Coefficients (e.g. from S/T/C)
    dn : array_like
       Modal Radial Filters (e.g. from M/F)
    kr : array_like
       kr-vector
       ::
          Can also be a matrix [krm; krs] for rigid sphere configurations:
          [1,:] => krm referring to the microphone radius
          [2,:] => krs referring to the sphere radius (scatterer)
    Nviz : int, optional
       Order of the spatial fourier transform [Default: 3]

    Returns
    -------
    vizMtx : array_like
       Computed visualization matrix over all kr
    """
    N = kr.size
    vizMtx = [None] * N
    for k in range(0, N):
        progress_bar(k, N, 'Visual matrix generation')
        vizMtx[k] = makeMTX(Pnm, dn, k, Nviz)
    return vizMtx


def normalizeMTX(MTX, logScale=False):
    """ Normalizes a matrix to [0 ... 1]

    Parameters
    ----------
    MTX : array_like
       Matrix to be normalized
    logScale : bool
       Toggle conversion logScale [Default: False]

    Returns
    -------
    MTX : array_liked
       Normalized Matrix
    """
    MTX -= MTX.min()
    MTX /= MTX.max()

    if logScale:
        MTX += 0.00001
        MTX = _np.log10(_np.abs(MTX))
        MTX += 5
        MTX /= 5.000004343
        # MTX = 20 * _np.log10(_np.abs(MTX))
    return MTX


def genSphCoords():
    """ Generates cartesian (x,y,z) and spherical (theta, phi) coordinates of a sphere
    Returns
    -------
    coords : named tuple
        holds cartesian (x,y,z) and spherical (theta, phi) coordinates
    """
    coords = namedtuple('coords', ['x', 'y', 'z', 'az', 'el'])
    az = _np.linspace(0, 2 * pi, 360)
    el = _np.linspace(0, pi, 181)
    coords.x = _np.outer(_np.cos(az), _np.sin(el))
    coords.y = _np.outer(_np.sin(az), _np.sin(el))
    coords.z = _np.outer(_np.ones(360), _np.cos(el))

    coords.el, coords.az = _np.meshgrid(_np.linspace(0, _np.pi, 181),
                                        _np.linspace(0, 2 * _np.pi, 360))
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
    rs = _np.abs(vizMTX.reshape((181, -1)).T)

    coords = genSphCoords()
    V = namedtuple('V', ['xs', 'ys', 'zs'])
    V.xs = rs * _np.sin(coords.el) * _np.cos(coords.az)
    V.ys = rs * _np.sin(coords.el) * _np.sin(coords.az)
    V.zs = rs * _np.cos(coords.el)
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

    TODO
    ----
    Fix camera position
    """
    V = sph2cartMTX(vizMTX)

    trace = go.Surface(
        x=V.xs,
        y=V.ys,
        z=V.zs,
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

    TODO
    ----
    Fix orientation and axis limits
    """

    trace = go.Surface(
        x=_np.r_[0:360],
        y=_np.r_[0:181],
        z=_np.abs(vizMTX),
        surfacecolor=_np.abs(vizMTX.reshape((181, -1))),
        colorscale='Viridis',
        showscale=False,
        hoverinfo='none'
    )
    return trace


def genVisual(vizMTX, style='shape', normalize=True, logScale=False):
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
        vizMTX = normalizeMTX(vizMTX, logScale=logScale)

    if style == 'shape':
        return genShape(vizMTX)
    elif style == 'sphere':
        return genSphere(vizMTX)
    elif style == 'flat':
        return genFlat(vizMTX)
    else:
        raise ValueError('Provided style "' + style + '" not available. Try sphere, shape or flat.')


def plot2D(data, title=None, type=None, fs=44100):
    """Visualize 2D data using plotly.

    Parameters
    ----------
    data : array_like
       Data to be plotted, separated along the first dimension (rows).
    title : string
       Add title to be displayed on plot
    type : string{None, 'time', 'linFFT', 'logFFT'}
       Type of data to be displayed. [Default: None]
    fs : int
       Sampling rate in Hz. [Default: 44100]
    """

    data = _np.atleast_2d(data)
    N = data.shape[0]

    # X vector: samples or time
    x = _np.arange(data.shape[1] - 1, dtype=_np.float_)

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Samples'
        ),
        yaxis=dict(
            title='Amplitude'
        )
    )

    if type == 'time':
        x /= fs
        layout.xaxis.title = 'Time [s]'
    elif type == 'linFFT':
        x = _np.fft.rfftfreq(x.shape[0] * 2 - 1, 1 / fs)
        layout.yaxis.title = 'Amplitude [dB]'
        layout.xaxis.title = 'Frequency [Hz]'
    elif type == 'logFFT':
        x = _np.fft.rfftfreq(x.shape[0] * 2 - 1, 1 / fs)
        layout.yaxis.title = 'Amplitude [dB]'
        layout.xaxis.title = 'Frequency [Hz]'
        layout.xaxis.type = 'log'

    traces = [None] * N

    for k in range(0, N):
        traces[k] = go.Scatter(
            x=x,
            y=data[k]
        )

    showTrace(traces, layout=layout)


def plot3D(vizMTX, style='shape', layout=None, colorize=True, logScale=False):
    """Visualize matrix data, such as from makeMTX(Pnm, dn)

    Parameters
    ----------
    vizMTX : array_like
       Matrix holding spherical data for visualization
    style : string{'shape', 'sphere', 'flat'}, optional
       Style of visualization. [Default: 'shape']
    normalize : Bool, optional
       Toggle normalization of data to [-1 ... 1] [Default: True]

    TODO
    ----
    Colorization, contour plot
    """

    if style == 'flat':
        layout = go.Layout(
            scene=dict(
                xaxis=dict(range=[0, 360]),
                yaxis=dict(range=[0, 181]),
                aspectmode='manual',
                aspectratio=dict(x=3.6, y=1.81, z=1)
            )
        )

    showTrace(genVisual(vizMTX, style=style, normalize=True, logScale=logScale), layout=layout)


def frqToKr(fTarget, fVec):
    """Returns the kr bin closest  to the target frequency

    Parameters
    ----------
    fTarget : float
       Target frequency
    fVec : array_like
       Array containing the available frequencys

    Returns
    -------
    krTarget : int
       kr bin closest to target frequency
    """

    return (_np.abs(fVec - fTarget)).argmin()


def plot3Dgrid(rows, cols, vizMTX, style, normalize=True):
    fig = tools.make_subplots(rows=rows, cols=cols,
                              specs=[[{'is_3d': True}, {'is_3d': True}],
                                     [{'is_3d': True}, {'is_3d': True}]])

    for IDX in range(0, len(vizMTX)):
        fig.append_trace(genVisual(vizMTX[IDX], style=style, normalize=normalize), IDX % rows + 1, IDX // cols + 1)

    if env_info() == 'jupyter_notebook':
        iplot(fig)
    else:
        pltoff(fig)

    return fig
