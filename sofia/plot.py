"""Plotting functions
- makeMTX: Generate 3D-matrix data
- visualize3D: Draw matrix data in 3D
"""
import numpy as _np
from vispy import scene, color
from .process import pdc
from .sph import sph2cart


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


def normalizeMTX(vizMTX):
    vizMTX -= vizMTX.min()
    return vizMTX / vizMTX.max()


def genShape(vizMTX, offset=0, scale=1.0, colorize=False):
    thetas, phis = _np.meshgrid(_np.linspace(0, _np.pi, 181), _np.linspace(0, 2 * _np.pi, 360))
    rs = offset + scale * vizMTX.reshape((181, -1)).T
    xs = rs * _np.sin(thetas) * _np.cos(phis)
    ys = rs * _np.sin(thetas) * _np.sin(phis)
    zs = rs * _np.cos(thetas)
    if colorize:
        cm = color.get_colormap('viridis')
        colors = cm[vizMTX]
        return scene.visuals.GridMesh(xs, ys, zs, colors=colors.rgba.reshape((181, -1, 4)))
    else:
        return scene.visuals.GridMesh(xs, ys, zs)


def genSphere(vizMTX, colorize=False):
    #  visObj = scene.visuals.Sphere(radius=1, rows=362, cols=91, method='latitude', face_colors=colors)
    thetas, phis = _np.meshgrid(_np.linspace(0, _np.pi, 181), _np.linspace(0, 2 * _np.pi, 360))
    rs = 1
    xs = rs * _np.sin(thetas) * _np.cos(phis)
    ys = rs * _np.sin(thetas) * _np.sin(phis)
    zs = rs * _np.cos(thetas)
    if colorize:
        cm = color.get_colormap('viridis')
        colors = cm[vizMTX]
        return scene.visuals.GridMesh(xs, ys, zs, colors=colors.rgba.reshape((181, -1, 4)))
    else:
        return scene.visuals.GridMesh(xs, ys, zs)


def genScatter(vizMTX, colorize=False):
    vizMTX = vizMTX.reshape([-1, 1])
    # Recreate angles
    angles = _np.array(generateAngles())

    sphCoords = _np.concatenate((angles, _np.atleast_2d(vizMTX)), axis=1)
    xyzCoords = _np.array(sph2cart(*sphCoords.T))
    scatterObj = scene.visuals.Markers()

    if colorize:
        cm = color.get_colormap('viridis')
        colors = cm[_np.squeeze(vizMTX)]
        scatterObj.set_data(xyzCoords.T, size=10, edge_color=None, face_color=colors)
    else:
        scatterObj.set_data(xyzCoords.T, size=10, edge_color=None, face_color='black')

    return scatterObj


def genVisual(vizMTX, style='shape', colorize=False, offset=0, scale=1.0, normalize=True):
    if normalize:
        vizMTX = normalizeMTX(vizMTX)

    if style == 'shape':
        return genShape(vizMTX, colorize=colorize, offset=offset, scale=scale)
    elif style == 'sphere':
        return genSphere(vizMTX, colorize=colorize)
    elif style == 'scatter':
        return genScatter(vizMTX, colorize=colorize)
    else:
        raise ValueError('Provided style "' + style + '" not available. Try sphere, shape or scatter.')


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
