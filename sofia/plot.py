"""Plotting functions
- makeMTX: Generate 3D-matrix-data
- visualize3D: Plot 3D data
"""
import numpy as _np
from vispy import scene, color
from .process import pdc
from .sph import sph2cart

def makeMTX(Pnm, dn, Nviz=3, krIndex=1, oversize=1):
    """mtxData = makeMTX(Nviz=3, Pnm, dn, krIndex)
    ------------------------------------------------------------------------
    mtxData   3D-matrix-data in 1[deg] steps
    ------------------------------------------------------------------------
    Pnm       Spatial Fourier Coefficients (from S/T/C)
    dn        Modal Radial Filters (from M/F)
    N         Order of the spatial fourier transform     [default = 3]
    krIndex   Index of kr Vector                         [default = 1]
    oversize  Integer Factor to increase the resolution. Set oversize = 1
              (default) to use the mtxData matrix for visual3D(), map3D().

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


def visualize3D(vizMTX, style='sphere', **kargs):
    """Visualize matrix data, such as from makeMTX(Pnm, dn)
    vizMTX     SOFiA 3D-matrix-data [1[deg] steps]
    style       'sphere',   surface colors indicate the intensity (default)
                'flat',     surface colors indicate the intensity
                'shape',    extension indicates the intensity
                'cshape',   extension+colors indicate the intensity
    colormap    select colormap (not yet implemented)
    """

    # Prepare data: reshape to [65160 x 1], take abs, normalize
    vizMTX = _np.abs(vizMTX.reshape((65160, -1)))
    vizMTX -= vizMTX.min()
    vizMTX /= vizMTX.max()

    # Generate colors
    cm = color.get_colormap('viridis')
    colors = cm.map(vizMTX)

    # Recreate angles
    angles = _np.array(generateAngles())

    # TODO: other styles, proper sphere
    if style == 'sphere':
        sphCoords = _np.concatenate((angles, _np.ones((angles.shape[0], 1))), axis=1)
    elif style == 'shape' or style == 'cshape':
        sphCoords = _np.concatenate((angles, vizMTX), axis=1)
    else:
        raise ValueError('Provided style "' + style + '" not available. Try sphere, flat, shape or cshape.')

    xyzCoords = _np.array(sph2cart(*sphCoords.T))

    # Create scatter object from mtxData
    scatter = scene.visuals.Markers()
    if style == 'cshape':
        scatter.set_data(xyzCoords.T, size=10, face_color=colors, edge_color=None)
    else:
        scatter.set_data(xyzCoords.T, size=10, face_color='black', edge_color=None)

    # Create scene
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(800, 600), show=True)

    # Create view with camera on target
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    view.camera.set_range(x=[-3, 3])

    # Add scattered points to view
    view.add(scatter)


def generateAngles():
    """Returns a [65160 x 1] grid of all radiant angles in 1 deg steps"""
    return _np.mgrid[0:360, 0:181].T.reshape((-1, 2)) * _np.pi / 180
