"""
Plotting functions
Helps visualizing spherical microphone data.
"""
from collections import namedtuple

import numpy as _np
import plotly.graph_objs as go
from plotly import offline as plotly_off, subplots

from .process import plane_wave_decomp
from .utils import current_time, env_info, progress_bar


def showTrace(trace, layout=None, title=None):
    """ Wrapper around Plotly's offline .plot() function

    Parameters
    ----------
    trace : plotly_trace
        Plotly generated trace to be displayed offline
    layout : plotly.graph_objs.Layout, optional
        Layout of plot to be displayed offline
    title : str, optional
        Title of plot to be displayed offline
    # colorize : bool, optional
    #     Toggles bw / colored plot [Default: True]

    Returns
    -------
    fig : plotly_fig_handle
        JSON representation of generated figure
    """
    if layout is None:
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

    if title is not None:
        fig.layout.update(title=title)
        filename = f'{title}.html'
    else:
        try:
            filename = f'{fig.layout.title}.html'
        except TypeError:
            filename = f'{current_time()}.html'

    # if colorize:
    #    data[0].autocolorscale = False
    #    data[0].surfacecolor = [0, 0.5, 1]
    if env_info() == 'jupyter_notebook':
        plotly_off.init_notebook_mode()
        plotly_off.iplot(fig)
    else:
        plotly_off.plot(fig, filename=filename)

    return fig


def makeMTX(spat_coeffs, radial_filter, kr_IDX, viz_order=None, stepsize_deg=1):
    """Returns a plane wave decomposition over a full sphere

    Parameters
    ----------
    spat_coeffs : array_like
        Spatial fourier coefficients
    radial_filter : array_like
        Modal radial filters
    kr_IDX : int
        Index of kr to be computed
    viz_order : int, optional
        Order of the spatial fourier transform [Default: Highest available]
    stepsize_deg : float, optional
        Integer Factor to increase the resolution. [Default: 1]

    Returns
    -------
    mtxData : array_like
        Plane wave decomposition (frequency domain)

    Note
    ----
    The file generates a Matrix of 181x360 pixels for the
    visualisation with visualize3D() in 1[deg] Steps (65160 plane waves).
    """

    if not viz_order:
        viz_order = _np.int(_np.ceil(_np.sqrt(spat_coeffs.shape[0]) - 1))

    angles = _np.mgrid[0:360:stepsize_deg, 0:181:stepsize_deg].reshape((2, -1)) * _np.pi / 180
    Y = plane_wave_decomp(viz_order, angles, spat_coeffs[:, kr_IDX], radial_filter[:, kr_IDX])

    return Y.reshape((360, -1)).T  # Return pwd data as [181, 360] matrix


def makeFullMTX(Pnm, dn, kr, viz_order=None):
    """ Generates visualization matrix for a set of spatial fourier coefficients over all kr
    Parameters
    ----------
    Pnm : array_like
        Spatial Fourier Coefficients (e.g. from S/T/C)
    dn : array_like
        Modal Radial Filters (e.g. from M/F)
    kr : array_like
        kr-vector
    viz_order : int, optional
        Order of the spatial fourier tplane_wave_decompransform [Default: Highest available]

    Returns
    -------
    vizMtx : array_like
        Computed visualization matrix over all kr
    """
    kr = _np.asarray(kr)
    if not viz_order:
        viz_order = _np.int(_np.ceil(_np.sqrt(Pnm.shape[0]) - 1))

    N = kr.size
    vizMtx = [None] * N
    for k in range(0, N):
        progress_bar(k, N, 'Visual matrix generation')
        vizMtx[k] = makeMTX(Pnm, dn, k, viz_order)
    return vizMtx


def normalizeMTX(MTX, logScale=False):
    """ Normalizes a matrix to [0 ... 1]

    Parameters
    ----------
    MTX : array_like
        Matrix to be normalized
    logScale : bool, optional
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
    az = _np.linspace(0, 2 * _np.pi, 360)
    el = _np.linspace(0, _np.pi, 181)
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
    normalize : bool, optional
        Toggle normalization of data to [-1 ... 1] [Default: True]
    logScale : bool, optional
        Toggle conversion logScale [Default: False]

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
        raise ValueError(f'Provided style "{style}" not available. Try sphere, shape or flat.')


def layout_2D(viz_type=None, title=None):
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Samples'
        ),
        yaxis=dict(
            title='Amplitude'
        )
    )

    if viz_type == 'TIME':
        layout.title = 'Time domain plot'
        layout.xaxis.title = 'Time in s'
    elif viz_type == 'ETC':
        layout.title = 'Time domain plot (ETC)'
        layout.yaxis.title = 'Amplitude in dB'
        layout.xaxis.title = 'Time in s'
    elif viz_type == 'LINFFT':
        layout.title = 'Frequency domain plot (linear)'
        layout.yaxis.title = 'Amplitude in dB'
        layout.xaxis.title = 'Frequency in Hz'
    elif viz_type == 'LOGFFT':
        layout.title = 'Frequency domain plot (logarithmic)'
        layout.yaxis.title = 'Amplitude in dB'
        layout.xaxis.title = 'Frequency in Hz'
        layout.xaxis.type = 'log'

    return layout


def prepare_2D_x(L, viz_type, fs):
    # X vector: samples or time
    x = _np.arange(L - 1, dtype=_np.float_)

    if viz_type in ['TIME', 'ETC']:
        x /= fs
    elif viz_type in ['LINFFT', 'LOGFFT']:
        x = _np.fft.rfftfreq(x.shape[0] * 2 - 1, 1 / fs)

    return x


def prepare_2D_traces(data, viz_type, fs, line_names):
    data = _np.atleast_2d(data)
    N, L = data.shape

    x = prepare_2D_x(L, viz_type, fs)

    traces = [None] * N

    for k in range(0, N):
        y = data[k]
        traces[k] = go.Scatter(x=x, y=y if viz_type == 'TIME' else 20 * _np.log10(_np.abs(y)))
        try:
            traces[k].name = line_names[k]
        except (TypeError, IndentationError):
            pass

    return traces


def plot2D(data, title=None, viz_type=None, fs=44100, line_names=None):
    """Visualize 2D data using plotly.

    Parameters
    ----------
    data : array_like
        Data to be plotted, separated along the first dimension (rows)
    title : str, optional
        Add title to be displayed on plot
    viz_type : str{None, 'Time', 'ETC', 'LinFFT', 'LogFFT'}, optional
        Type of data to be displayed [Default: None]
    fs : int, optional
        Sampling rate in Hz [Default: 44100]
    line_names : list of str, optional
        Add legend to be displayed on plot, with one entry for each data row [Default: None]
    """
    viz_type = viz_type.strip().upper()  # remove whitespaces and make upper case

    layout = layout_2D(viz_type=viz_type, title=title)
    # noinspection PyTypeChecker
    traces = prepare_2D_traces(data=data, viz_type=viz_type, fs=fs, line_names=line_names)

    showTrace(traces, layout=layout, title=title)


def plot3D(vizMTX, style='shape', layout=None, normalize=True, logScale=False):
    """Visualize matrix data, such as from makeMTX(Pnm, dn)

    Parameters
    ----------
    vizMTX : array_like
        Matrix holding spherical data for visualization
    layout : plotly.graph_objs.Layout, optional
        Layout of plot to be displayed offline
    style : string{'shape', 'sphere', 'flat'}, optional
        Style of visualization. [Default: 'shape']
    normalize : bool, optional
        Toggle normalization of data to [-1 ... 1] [Default: True]
    logScale : bool, optional
        Toggle conversion logScale [Default: False]

    # TODO
    # ----
    # Colorization, contour plot
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

    showTrace(genVisual(vizMTX, style=style, normalize=normalize, logScale=logScale), layout=layout)


def plot3Dgrid(rows, cols, viz_data, style, normalize=True, title=None):
    if len(viz_data) > rows * cols:
        raise ValueError('Number of plot data is more than the specified rows and columns.')
    fig = subplots.make_subplots(rows, cols, specs=[[{'is_3d': True}] * cols] * rows, print_grid=False)

    if style == 'flat':
        layout_3D = dict(
            xaxis=dict(range=[0, 360]),
            yaxis=dict(range=[0, 181]),
            aspectmode='manual',
            aspectratio=dict(x=3.6, y=1.81, z=1)
        )
    else:
        layout_3D = dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='cube'
        )

    rows, cols = _np.mgrid[1:rows + 1, 1: cols + 1]
    rows = rows.flatten()
    cols = cols.flatten()
    for IDX in range(0, len(viz_data)):
        cur_row = int(rows[IDX])
        cur_col = int(cols[IDX])
        fig.add_trace(genVisual(viz_data[IDX], style=style, normalize=normalize), cur_row, cur_col)
        fig.layout[f'scene{IDX + 1:d}'].update(layout_3D)

    if title is not None:
        fig.layout.update(title=title)
        filename = f'{title}.html'
    else:
        filename = f'{current_time()}.html'

    if env_info() == 'jupyter_notebook':
        plotly_off.iplot(fig)
    else:
        plotly_off.plot(fig, filename=filename)
