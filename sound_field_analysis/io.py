"""
Input-Output functions"""

from collections import namedtuple

import numpy as _np
import scipy.io as sio
import scipy.io.wavfile

from . import utils


class ArrayConfiguration(namedtuple('ArrayConfiguration', 'array_radius array_type transducer_type scatter_radius '
                                                          'dual_radius')):
    """Named tuple ArrayConfiguration"""
    __slots__ = ()

    def __new__(cls, array_radius, array_type, transducer_type, scatter_radius=None, dual_radius=None):
        """
        Parameters
        ----------
        array_radius : float or array_like
           Radius of array
        array_type : {'open', 'rigid'}
           Type array
        transducer_type: {'omni', 'cardioid'}
           Type of transducer,
        scatter_radius : float, optional
           Radius of scatterer, required for `array_type` == 'rigid'. (Default: equal to array_radius)
        dual_radius : float, optional
           Radius of second array, required for `array_type` == 'dual'
        """
        if array_type not in {'open', 'rigid', 'dual'}:
            raise ValueError('Sphere configuration has to be either open, rigid, or dual.')
        if transducer_type not in {'omni', 'cardioid'}:
            raise ValueError('Transducer type has to be either omni or cardioid.')
        if array_type == 'rigid' and scatter_radius is None:
            scatter_radius = array_radius
        if array_type == 'dual' and dual_radius is None:
            raise ValueError('For a dual array configuration, dual_radius must be provided.')
        if array_type == 'dual' and transducer_type == 'cardioid':
            raise ValueError('For a dual array configuration, cardioid transducers are not supported.')

        # noinspection PyArgumentList
        self = super(ArrayConfiguration, cls).__new__(cls, array_radius, array_type, transducer_type, scatter_radius,
                                                      dual_radius)
        return self

    def __repr__(self):
        return 'ArrayConfiguration(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in
            zip(['array_radius', 'array_type', 'transducer_type', 'scatter_radius', 'dual_radius'], self)) + ')'


class TimeSignal(namedtuple('TimeSignal', 'signal fs delay')):
    """Named tuple TimeSignal"""
    __slots__ = ()

    def __new__(cls, signal, fs, delay=None):
        """
        Parameters
        ----------
        signal : array_like
           Array of signals of shape [nSignals x nSamples]
        fs : int or array_like
           Sampling frequency
        delay : float or array_like, optional
           [Default: None]
        """
        signal = _np.atleast_2d(signal)
        no_of_signals = signal.shape[1]
        fs = _np.asarray(fs)
        delay = _np.asarray(delay)

        if (fs.size != 1) and (fs.size != no_of_signals):
            raise ValueError('fs can either be a scalar or an array with one element per signal.')
        if (delay.size != 1) and (delay.size != no_of_signals):
            raise ValueError('delay can either be a scalar or an array with one element per signal.')

        # noinspection PyArgumentList
        self = super(TimeSignal, cls).__new__(cls, signal, fs, delay)
        return self

    def __repr__(self):
        return 'TimeSignal(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['signal', 'fs', 'delay'], self)) + ')'


class SphericalGrid(namedtuple('SphericalGrid', 'azimuth colatitude radius weight')):
    """Named tuple SphericalGrid"""
    __slots__ = ()

    def __new__(cls, azimuth, colatitude, radius=None, weight=None):
        """
        Parameters
        ----------
        azimuth, colatitude : array_like
           Grid sampling point directions in radians
        radius, weight : float or array_like, optional
            Grid sampling point distances and weights
        """
        azimuth = _np.asarray(azimuth)
        colatitude = _np.asarray(colatitude)
        if radius is not None:
            radius = _np.asarray(radius)
        if weight is not None:
            weight = _np.asarray(weight)
        if azimuth.size != colatitude.size:
            raise ValueError('Azimuth and colatitude have to contain the same number of elements.')
        if (radius is not None) and (radius.size != 1) and (radius.size != azimuth.size):
            raise ValueError('Radius can either be a scalar or an array of same size as azimuth/colatitude.')
        if (weight is not None) and (weight.size != 1) and (weight.size != azimuth.size):
            raise ValueError('Weight can either be a scalar or an array of same size as azimuth/colatitude.')

        # noinspection PyArgumentList
        self = super(SphericalGrid, cls).__new__(cls, azimuth, colatitude, radius, weight)
        return self

    def __repr__(self):
        return 'SphericalGrid(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['azimuth', 'colatitude', 'radius', 'weight'], self)) + ')'


class ArraySignal(namedtuple('ArraySignal', 'signal grid center_signal configuration temperature')):
    """Named tuple ArraySignal"""
    __slots__ = ()

    def __new__(cls, signal, grid, center_signal=None, configuration=None, temperature=None):
        """
        Parameters
        ----------
        signal : TimeSignal
           Array Time domain signals and sampling frequency
        grid : SphericalGrid
           Measurement grid of time domain signals
        center_signal : TimeSignal
           Center measurement time domain signal and sampling frequency
        configuration : ArrayConfiguration
           Information on array configuration
        temperature : array_like, optional
           Temperature in room or at each sampling position
        """
        signal = TimeSignal(*signal)
        grid = SphericalGrid(*grid)
        if configuration is not None:
            configuration = ArrayConfiguration(*configuration)

        # noinspection PyArgumentList
        self = super(ArraySignal, cls).__new__(cls, signal, grid, center_signal, configuration, temperature)
        return self

    def __repr__(self):
        return 'ArraySignal(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['signal', 'grid', 'center_signal', 'configuration', 'temperature'], self)) + ')'


class HrirSignal(namedtuple('HrirSignal', 'l r grid center_signal')):
    """Named tuple HrirSignal"""
    __slots__ = ()

    def __new__(cls, l, r, grid, center_signal=None):
        """
        Parameters
        ----------
        l : TimeSignal
           Left ear time domain signals and sampling frequency
        r : TimeSignal
           Right ear time domain signals and sampling frequency
        grid : SphericalGrid
           Measurement grid of time domain signals
        center_signal : TimeSignal
           Center measurement time domain signal and sampling frequency
        """
        l = TimeSignal(*l)
        r = TimeSignal(*r)

        grid = SphericalGrid(*grid)
        if center_signal is not None:
            center_signal = TimeSignal(*center_signal)

        # noinspection PyArgumentList
        self = super(HrirSignal, cls).__new__(cls, l, r, grid, center_signal)
        return self

    def __repr__(self):
        return 'HrirSignal(\n' + ',\n'.join(
            '    {0} = {1}'.format(name, repr(data).replace('\n', '\n      '))
            for name, data in zip(['l', 'r', 'grid', 'center_signal'], self)) + ')'


def read_miro_struct(file_name, channel='irChOne', transducer_type='omni', scatter_radius=None,
                     get_center_signal=False):
    """ Reads miro matlab files.

    Parameters
    ----------
    file_name : filepath
       Path to file that has been exported as a struct
    channel : string, optional
       Channel that holds required signals. [Default: 'irChOne']
    transducer_type : {omni, cardoid}, optional
       Sets the type of transducer used in the recording. [Default: 'omni']
    scatter_radius : float, option
       Radius of the scatterer. [Default: None]
    get_center_signal : bool, optional
        If center signal should be loaded. [Default: False]

    Returns
    -------
    array_signal : ArraySignal
       Tuple containing a TimeSignal `signal`, SphericalGrid `grid`, TimeSignal 'center_signal',
       ArrayConfiguration `configuration` and the air temperature

    Notes
    -----
    This function expects a slightly modified miro file in that it expects a field `colatitude` instead of
    `elevation`. This is for avoiding confusion as may miro file contain colatitude data in the elevation field.

    To import center signal measurements the matlab method miro_to_struct has to be extended. Center measurements are
    included in every measurement provided at http://audiogroup.web.th-koeln.de/.

    """
    current_data = sio.loadmat(file_name)

    time_signal = TimeSignal(signal=_np.squeeze(current_data[channel]).T,
                             fs=_np.squeeze(current_data['fs']))

    center_signal = None
    if get_center_signal:
        try:
            center_signal = TimeSignal(signal=_np.squeeze(current_data['irCenter']).T,
                                       fs=_np.squeeze(current_data['fs']))
        except KeyError:
            print('WARNING: Center signal not included in miro struct, use extended miro_to_struct.m!')
            center_signal = None

    mic_grid = SphericalGrid(azimuth=_np.squeeze(current_data['azimuth']),
                             colatitude=_np.squeeze(current_data['colatitude']),
                             radius=_np.squeeze(current_data['radius']),
                             weight=_np.squeeze(current_data['quadWeight']))

    if (mic_grid.colatitude < 0).any():
        print("WARNING: The 'colatitude' data contains negative values, which is an indication that it is actually "
              "elevation")

    if _np.squeeze(current_data['scatterer']):
        sphere_config = 'rigid'
    else:
        sphere_config = 'open'
    array_config = ArrayConfiguration(mic_grid.radius, sphere_config, transducer_type, scatter_radius)

    return ArraySignal(time_signal, mic_grid, center_signal, array_config, _np.squeeze(current_data['avgAirTemp']))


def read_SOFA_file(file_name):
    """ Reads Head Related Impulse Responses or Array impuse responses (DRIRs) stored as Spatially Oriented Format for Acoustics (SOFA) files files,
        and convert them to Array Signal or HRIR Signal class

    Parameters
    ----------
    file_name : filepath
       Path to SOFA file

    Returns
    -------
    sofa_signal : ArraySignal
       Tuple containing a TimeSignal `signal`, SphericalGrid `grid`, TimeSignal 'center_signal',
       ArrayConfiguration `configuration` and the air temperature

   sofa_signal : HRIRSignal
       Tuple containing the TimeSignals 'l' for the left, and 'r' for the right ear, SphericalGrid `grid`, TimeSignal 'center_signal'

    Notes
    -----
    * Depends python package pysofaconventions:
        https://github.com/andresperezlopez/pysofaconventions
    * Up to now, importing 'SimpleFreeFieldHRIR' and 'SingleRoomDRIR' are provided only.

    """
    # check if package 'pysofaconventions' is available
    try:
        import pysofaconventions as sofa
    except ImportError:
        print('Could not found pysofaconventions. Could not load SOFA file')
        return None

    def print_sofa_infos(SOFA_convention):
        print(f'\n --> samplerate: {SOFA_convention.getSamplingRate()[0]:.0f} Hz' \
              f', receivers: {SOFA_convention.ncfile.file.dimensions["R"].size}' \
              f', emitters: {SOFA_convention.ncfile.file.dimensions["E"].size}' \
              f', measurements: {SOFA_convention.ncfile.file.dimensions["M"].size}' \
              f', samples: {SOFA_convention.ncfile.file.dimensions["N"].size}' \
              f', format: {SOFA_convention.getDataIR().dtype}' \
              f'\n --> SOFA_convention: {SOFA_convention.getGlobalAttributeValue("SOFAConventions")}' \
              f', version: {SOFA_convention.getGlobalAttributeValue("SOFAConventionsVersion")}')
        try:
            print(f' --> listener: {SOFA_convention.getGlobalAttributeValue("ListenerDescription")}')
        except sofa.SOFAError:
            pass
        try:
            print(f' --> author: {SOFA_convention.getGlobalAttributeValue("Author")}')
        except sofa.SOFAError:
            pass

    def load_convention(SOFA_file):
        convention = SOFA_file.getGlobalAttributeValue('SOFAConventions')
        if convention == 'SimpleFreeFieldHRIR':
            return sofa.SOFASimpleFreeFieldHRIR(SOFA_file.ncfile.filename, "r")
        elif convention == 'SingleRoomDRIR':
            return sofa.SOFASingleRoomDRIR(SOFA_file.ncfile.filename, "r")
        else:
            raise ValueError(f'Unknown or unimplemented SOFA convention!')

    # load SOFA file
    SOFA_file = sofa.SOFAFile(file_name, 'r')

    # load SOFA convention
    SOFA_convention = load_convention(SOFA_file)

    # check validity of sofa_file and sofa_convention
    if not SOFA_file.isValid():
        raise ValueError('Invalid SOFA file.')
    elif not SOFA_convention.isValid():
        raise ValueError('Invalid SOFA convention.')
    else:
        # print SOFA file infos
        print(f'\n open {file_name}')
        print_sofa_infos(SOFA_convention)

        # store SOFA data as named tupel
        if SOFA_file.getGlobalAttributeValue('SOFAConventions') == 'SimpleFreeFieldHRIR':
            left_ear = TimeSignal(signal=_np.squeeze(SOFA_file.getDataIR()[:, 0, :]),
                                  fs=_np.squeeze(int(SOFA_file.getSamplingRate()[0])))
            right_ear = TimeSignal(signal=_np.squeeze(SOFA_file.getDataIR()[:, 1, :]),
                                   fs=_np.squeeze(int(SOFA_file.getSamplingRate()[0])))

            # given spherical coordinates azimuth: [degree], elevation: [degree], radius: [meters]
            pos_grid_deg = SOFA_file.getSourcePositionValues()
            pos_grid_deg = pos_grid_deg.T.filled(0).copy()  # transform into regular `numpy.ndarray`

            # transform spherical grid to radiants, and elevation to colatitude
            pos_grid_deg[1, :] = 90 - pos_grid_deg[1, :]
            pos_grid_rad = utils.deg2rad(pos_grid_deg[0:2])

            hrir_gird = SphericalGrid(azimuth=_np.squeeze(pos_grid_rad[0]),
                                      colatitude=_np.squeeze(pos_grid_rad[1]),
                                      radius=_np.squeeze(pos_grid_deg[2]))  # store original radius

            return HrirSignal(l=left_ear, r=right_ear, grid=hrir_gird)

        elif SOFA_file.getGlobalAttributeValue('SOFAConventions') == 'SingleRoomDRIR':
            time_signal = TimeSignal(signal=_np.squeeze(SOFA_file.getDataIR()),
                                     fs=_np.squeeze(int(SOFA_file.getSamplingRate()[0])))

            # given cartesian coordinates x: [meters], y: [meters], z: [meters]
            pos_grid_cart = SOFA_file.getReceiverPositionValues()[:, :, 0]
            pos_grid_cart = pos_grid_cart.T.filled(0)  # transform into regular `numpy.ndarray`

            # transform cartesian grid to spherical coordinates in radiants
            pos_grid_sph = utils.cart2sph(pos_grid_cart, is_deg=False)

            mic_grid = SphericalGrid(azimuth=pos_grid_sph[0],
                                     colatitude=pos_grid_sph[1],
                                     radius=pos_grid_sph[2])

            # assume rigid sphere and omnidirectional transducers according to SOFA 1.0, AES69-2015
            array_config = ArrayConfiguration(array_radius=pos_grid_sph[2].mean(),
                                              array_type='rigid',
                                              transducer_type='omni')

            return ArraySignal(signal=time_signal, grid=mic_grid, configuration=array_config)
        else:
            import sys
            print('WARNING: Could not load SOFA file.', file=sys.stderr)


def empty_time_signal(no_of_signals, signal_length):
    """Returns an empty np rec array that has the proper data structure

    Parameters
    ----------
    no_of_signals : int
       Number of signals to be stored in the recarray
    signal_length : int
       Length of the signals to be stored in the recarray

    Returns
    -------
    time_data : recarray
       Structured array  with following fields:
    ::
       .signal           [Channels X Samples]
       .fs               Sampling frequency in [Hz]
       .azimuth          Azimuth of sampling points
       .colatitude       Colatitude of sampling points
       .radius           Array radius in [m]
       .grid_weights     Weights of quadrature
       .air_temperature  Average temperature in [C]
    """
    return _np.rec.array(_np.zeros(no_of_signals,
                                   dtype=[('signal', str(signal_length) + 'f8'),
                                          ('fs', 'f8'),
                                          ('azimuth', 'f8'),
                                          ('colatitude', 'f8'),
                                          ('radius', 'f8'),
                                          ('grid_weights', 'f8'),
                                          ('air_temperature', 'f8')]))


def load_array_signal(filename):
    """Convenience function to load ArraySignal saved into np data structures

    Parameters
    ----------
    filename : string
       File to load

    Returns
    -------
    Y : ArraySignal
       See io.ArraySignal
    """
    return ArraySignal(*_np.load(filename))


def read_wavefile(filename):
    """ Reads in WAV files and returns data [Nsig x Nsamples] and fs
    Parameters
    ----------
    filename : string
       Filename of wave file to be read

    Returns
    -------
    data : array_like
       Data of dim [Nsig x Nsamples]
    fs : int
       Sampling frequency of read data
    """
    fs, data = sio.wavfile.read(filename)
    return data.T, fs


def write_SSR_IRs(filename, time_data_l, time_data_r, wavformat="float"):
    """Takes two time signals and writes out the horizontal plane as HRIRs for the SoundScapeRenderer.
    Ideally, both hold 360 IRs but smaller sets are tried to be scaled up using repeat.

    Parameters
    ----------
    filename : string
       filename to write to
    time_data_l, time_data_r : io.ArraySignal
       ArraySignals for left/right ear
    wavformat : string
       wav file format to write. Either "float" or "int16"
    """
    import sys
    # equator_IDX_left = utils.nearest_to_value_logical_IDX(time_data_l.grid.colatitude, _np.pi / 2)
    # equator_IDX_right = utils.nearest_to_value_logical_IDX(time_data_r.grid.colatitude, _np.pi / 2)

    # IRs_left = time_data_l.signal.signal[equator_IDX_left]
    # IRs_right = time_data_r.signal.signal[equator_IDX_right]
    IRs_left = time_data_l.signal.signal
    IRs_right = time_data_r.signal.signal

    IRs_to_write = utils.interleave_channels(IRs_left, IRs_right, style="SSR")
    # data_to_write = utils.simple_resample(IRs_to_write, original_fs=time_data_l.signal.fs, target_fs=44100)
    data_to_write = IRs_to_write

    # get absolute max value
    max_val = _np.max(_np.abs([time_data_l.signal.signal, time_data_r.signal.signal]))
    if max_val > 1.0:
        if wavformat == "int16":
            raise ValueError("At least one value exceeds 1.0, exporting to 'int16' will lead to clipping. "
                             "Choose waveformat 'float' instead or normalize data.")
        print("WARNING: At least one value exceeds 1.0!", file=sys.stderr)

    if wavformat == "float":
        sio.wavfile.write(filename, int(time_data_l.signal.fs), data_to_write.astype(_np.float32).T)
    elif wavformat == "int16":
        sio.wavfile.write(filename, int(time_data_l.signal.fs), (data_to_write * 32767).astype(_np.int16).T)
    else:
        raise TypeError("Format " + wavformat + "not known. Should be either 'float' or 'int16'.")
