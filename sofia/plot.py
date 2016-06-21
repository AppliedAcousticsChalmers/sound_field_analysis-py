"""Plotting functions
- makeMTX: Generate 3D-matrix-data
- visualize3D: Plot 3D data
"""
import numpy as _np
from vispy import app, visuals, scene
from .process import pdc


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
    pass
