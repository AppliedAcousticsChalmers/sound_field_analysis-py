"""
Module containing various input and output functions:

`TimeSignal`
    Named tuple to store time domain data and related metadata.
`SphericalGrid`
    Named tuple to store spherical sampling grid geometry.
`ArrayConfiguration`
    Named tuple to store microphone array characteristics.
`ArraySignal`
    Named tuple to store microphone array data in terms of `TimeSignal`,
    `SphericalGrid` and `ArrayConfiguration`
`HrirSignal`
    Named tuple to store Head Related Impulse Response grid data in terms of
    `TimeSignal` for either ear and `SphericalGrid`
`read_miro_struct`
    Read Head Related Impulse Responses (HRIRs) or Array / Directional
    Impulse Responses (DRIRs) stored as MIRO Matlab files and convert them to
    `ArraySignal`.
`read_SOFA_file`
    Read Head Related Impulse Responses (HRIRs) or Array / Directional
    Impulse Responses (DRIRs) stored as Spatially Oriented Format for Acoustics
    (SOFA) files and convert them to `ArraySignal` or `HrirSignal`.
`empty_time_signal`
    Returns an empty np rec array that has the proper data structure.
`load_array_signal`
    Convenience function to load ArraySignal saved into np data structures.
`read_wavefile`
    Reads in WAV files and returns data [Nsig x Nsamples] and fs.
`write_SSR_IRs`
    Takes two time signals and writes out the horizontal plane as HRIRs for
    the SoundScapeRenderer. Ideally, both hold 360 IRs but smaller sets are
    tried to be scaled up using repeat.
"""
import sys
from collections import namedtuple

import numpy as _np
import pysofaconventions as sofa
import scipy.io as sio
import scipy.io.wavfile

from . import utils


class TimeSignal(namedtuple("TimeSignal", "signal fs delay")):
    """Named tuple to store time domain data and related metadata."""

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
            raise ValueError(
                "Sampling frequency can either be a scalar or an array with one element per signal."
            )
        if (delay.size != 1) and (delay.size != no_of_signals):
            raise ValueError(
                "Delay can either be a scalar or an array with one element per signal."
            )

        # noinspection PyArgumentList
        self = super(TimeSignal, cls).__new__(cls, signal, fs, delay)
        return self

    def __repr__(self):
        return utils.get_named_tuple__repr__(self)


class SphericalGrid(namedtuple("SphericalGrid", "azimuth colatitude radius weight")):
    """Named tuple to store spherical sampling grid geometry."""

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
            if radius.size == 0:
                radius = None
        if weight is not None:
            weight = _np.asarray(weight)
            if weight.size == 0:
                weight = None
        if azimuth.size != colatitude.size:
            raise ValueError(
                "Azimuth and colatitude have to contain the same number of elements."
            )
        if radius is not None and radius.size not in (1, azimuth.size):
            raise ValueError(
                "Radius can either be a scalar or an array of same size as "
                "azimuth/colatitude."
            )
        if weight is not None and weight.size not in (1, azimuth.size):
            raise ValueError(
                "Weight can either be a scalar or an array of same size as "
                "azimuth/colatitude."
            )

        # noinspection PyArgumentList
        self = super(SphericalGrid, cls).__new__(
            cls, azimuth, colatitude, radius, weight
        )
        return self

    def __repr__(self):
        return utils.get_named_tuple__repr__(self)


class ArrayConfiguration(
    namedtuple(
        "ArrayConfiguration",
        "array_radius array_type transducer_type scatter_radius dual_radius",
    )
):
    """Named tuple to store microphone array characteristics."""

    __slots__ = ()

    def __new__(
        cls,
        array_radius,
        array_type,
        transducer_type,
        scatter_radius=None,
        dual_radius=None,
    ):
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
            Radius of scatterer, required for `array_type` == 'rigid'.
            [Default: equal to array_radius]
        dual_radius : float, optional
            Radius of second array, required for `array_type` == 'dual'
        """
        if array_type not in {"open", "rigid", "dual"}:
            raise ValueError(
                "Sphere configuration has to be either open, rigid, or dual."
            )
        if transducer_type not in {"omni", "cardioid"}:
            raise ValueError("Transducer type has to be either omni or cardioid.")
        if array_type == "rigid" and scatter_radius is None:
            scatter_radius = array_radius
        if array_type == "dual" and dual_radius is None:
            raise ValueError(
                "For a dual array configuration, dual_radius must be provided."
            )
        if array_type == "dual" and transducer_type == "cardioid":
            raise ValueError(
                "For a dual array configuration, cardioid transducers are not supported."
            )

        # noinspection PyArgumentList
        self = super(ArrayConfiguration, cls).__new__(
            cls, array_radius, array_type, transducer_type, scatter_radius, dual_radius
        )
        return self

    def __repr__(self):
        return utils.get_named_tuple__repr__(self)


class ArraySignal(
    namedtuple("ArraySignal", "signal grid center_signal configuration temperature")
):
    """
    Named tuple to store microphone array data in terms of `TimeSignal`,
    `SphericalGrid` and `ArrayConfiguration`.
    """

    __slots__ = ()

    def __new__(
        cls, signal, grid, center_signal=None, configuration=None, temperature=None
    ):
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
        self = super(ArraySignal, cls).__new__(
            cls, signal, grid, center_signal, configuration, temperature
        )
        return self

    def __repr__(self):
        return utils.get_named_tuple__repr__(self)


class HrirSignal(namedtuple("HrirSignal", "l r grid center_signal")):
    """
    Named tuple to store Head Related Impulse Response grid data in terms of
    `TimeSignal` for either ear and `SphericalGrid`.
    """

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
        return utils.get_named_tuple__repr__(self)


def read_miro_struct(
    file_name,
    channel="irChOne",
    transducer_type="omni",
    scatter_radius=None,
    get_center_signal=False,
):
    """Read Head Related Impulse Responses (HRIRs) or Array / Directional
    Impulse Responses (DRIRs) stored as MIRO Matlab files and convert them to
    `ArraySignal`.

    Parameters
    ----------
    file_name : filepath
        Path to file that has been exported as a struct
    channel : string, optional
        Channel that holds required signals. [Default: 'irChOne']
    transducer_type : {omni, cardioid}, optional
        Sets the type of transducer used in the recording. [Default: 'omni']
    scatter_radius : float, option
        Radius of the scatterer. [Default: None]
    get_center_signal : bool, optional
        If center signal should be loaded. [Default: False]

    Returns
    -------
    array_signal : ArraySignal
        Tuple containing a TimeSignal `signal`, SphericalGrid `grid`,
        TimeSignal 'center_signal', ArrayConfiguration `configuration` and
        the air temperature

    Notes
    -----
    This function expects a slightly modified miro file in that it expects a
    field `colatitude` instead of `elevation`. This is for avoiding confusion as
    may miro file contain colatitude data in the elevation field.

    To import center signal measurements the matlab method miro_to_struct has to
    be extended. Center measurements are included in every measurement
    provided at https://audiogroup.web.th-koeln.de/.
    """
    current_data = sio.loadmat(file_name)

    time_signal = TimeSignal(
        signal=_np.squeeze(current_data[channel]).T, fs=_np.squeeze(current_data["fs"])
    )

    center_signal = None
    if get_center_signal:
        try:
            center_signal = TimeSignal(
                signal=_np.squeeze(current_data["irCenter"]).T,
                fs=_np.squeeze(current_data["fs"]),
            )
        except KeyError:
            print(
                "WARNING: Center signal not included in miro struct, use "
                "extended miro_to_struct.m!",
                file=sys.stderr,
            )
            center_signal = None

    mic_grid = SphericalGrid(
        azimuth=_np.squeeze(current_data["azimuth"]),
        colatitude=_np.squeeze(current_data["colatitude"]),
        radius=_np.squeeze(current_data["radius"]),
        weight=_np.squeeze(current_data["quadWeight"])
        if "quadWeight" in current_data
        else None,
    )

    if (mic_grid.colatitude < 0).any():
        print(
            'WARNING: The "colatitude" data contains negative values, which '
            "is an indication that it is actually elevation",
            file=sys.stderr,
        )

    if _np.squeeze(current_data["scatterer"]):
        sphere_config = "rigid"
    else:
        sphere_config = "open"
    array_config = ArrayConfiguration(
        mic_grid.radius, sphere_config, transducer_type, scatter_radius
    )

    return ArraySignal(
        time_signal,
        mic_grid,
        center_signal,
        array_config,
        _np.squeeze(current_data["avgAirTemp"]),
    )


# noinspection PyPep8Naming
def read_SOFA_file(file_name):
    """Read Head Related Impulse Responses (HRIRs) or Array / Directional
    Impulse Response (DRIRs) stored as Spatially Oriented Format for Acoustics
    (SOFA) files and convert them to`ArraySignal` or `HrirSignal`.

    Parameters
    ----------
    file_name : filepath
        Path to SOFA file

    Returns
    -------
    ArraySignal or HRIRSignal
        Names tuples containing a the loaded file contents

    Raises
    ------
    NotImplementedError
        In case SOFA conventions other then "SimpleFreeFieldHRIR" or
        "SingleRoomDRIR" should be loaded
    ValueError
        In case source / receiver grid given in units not according to the SOFA
        convention
    ValueError
        In case impulse response data is incomplete
    """

    def _print_sofa_infos(convention):
        log_str = (
            f" --> samplerate: {convention.getSamplingRate()[0]:.0f} Hz"
            f', receivers: {convention.ncfile.file.dimensions["R"].size}'
            f', emitters: {convention.ncfile.file.dimensions["E"].size}'
            f', measurements: {convention.ncfile.file.dimensions["M"].size}'
            f', samples: {convention.ncfile.file.dimensions["N"].size}'
            f", format: {convention.getDataIR().dtype}"
            f'\n --> convention: {convention.getGlobalAttributeValue("SOFAConventions")}'
            f', version: {convention.getGlobalAttributeValue("SOFAConventionsVersion")}'
        )
        try:
            log_str = f'{log_str}\n --> listener: {convention.getGlobalAttributeValue("ListenerDescription")}'
        except sofa.SOFAError:
            pass
        try:
            log_str = f'{log_str}\n --> author: {convention.getGlobalAttributeValue("Author")}'
        except sofa.SOFAError:
            pass
        print(log_str)

    def _load_convention(_file):
        convention = _file.getGlobalAttributeValue("SOFAConventions")
        if convention == "SimpleFreeFieldHRIR":
            return sofa.SOFASimpleFreeFieldHRIR(_file.ncfile.filename, "r")
        elif convention == "SingleRoomDRIR":
            return sofa.SOFASingleRoomDRIR(_file.ncfile.filename, "r")
        else:
            raise NotImplementedError(
                f'Loading SOFA convention "{convention}" is not implemented yet.'
            )

    def _check_irs(irs):
        if isinstance(irs, _np.ma.MaskedArray):
            # check that all values exist
            if _np.ma.count_masked(irs):
                raise ValueError(f"incomplete IR data at positions {irs.mask}.")
            # transform into regular `numpy.ndarray`
            irs = irs.filled(0)
        return irs.copy()

    # load SOFA file and check validity
    file = sofa.SOFAFile(file_name, "r")
    if not file.isValid():
        raise ValueError("Invalid SOFA file.")
    # load specific convention and check validity
    file = _load_convention(file)
    if not file.isValid():
        raise ValueError(
            f"Invalid SOFA file according to "
            f"\"{file.getGlobalAttributeValue('SOFAConventions')}\" convention."
        )

    # print SOFA file infos
    print(f'\nopen SOFA file "{file_name}"')
    _print_sofa_infos(file)

    # store SOFA data as named tuple
    if isinstance(file, sofa.SOFAConventions.SOFASimpleFreeFieldHRIR):
        hrir_l = TimeSignal(
            signal=_check_irs(file.getDataIR()[:, 0]), fs=int(file.getSamplingRate()[0])
        )
        hrir_r = TimeSignal(
            signal=_check_irs(file.getDataIR()[:, 1]), fs=int(file.getSamplingRate()[0])
        )

        # transform grid into azimuth, colatitude, radius in radians
        grid_acr_rad = utils.SOFA_grid2acr(
            grid_values=file.getSourcePositionValues(),
            grid_info=file.getSourcePositionInfo(),
        )

        hrir_grid = SphericalGrid(
            azimuth=grid_acr_rad[0], colatitude=grid_acr_rad[1], radius=grid_acr_rad[2]
        )
        return HrirSignal(l=hrir_l, r=hrir_r, grid=hrir_grid)

    else:  # isinstance(file, sofa.SOFAConventions.SOFASingleRoomDRIR):
        drir_signal = TimeSignal(
            signal=_check_irs(_np.squeeze(file.getDataIR())),
            fs=int(file.getSamplingRate()[0]),
        )

        # transform grid into azimuth, colatitude, radius in radians
        grid_acr_rad = utils.SOFA_grid2acr(
            grid_values=file.getReceiverPositionValues()[:, :, 0],
            grid_info=file.getReceiverPositionInfo(),
        )

        # assume rigid sphere and omnidirectional transducers according to
        # SOFA 1.0, AES69-2015
        drir_configuration = ArrayConfiguration(
            array_radius=grid_acr_rad[2].mean(),
            array_type="rigid",
            transducer_type="omni",
        )
        drir_grid = SphericalGrid(
            azimuth=grid_acr_rad[0], colatitude=grid_acr_rad[1], radius=grid_acr_rad[2]
        )
        return ArraySignal(
            signal=drir_signal, grid=drir_grid, configuration=drir_configuration
        )


def empty_time_signal(no_of_signals, signal_length):
    """Returns an empty np rec array that has the proper data structure.

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
    return _np.rec.array(
        _np.zeros(
            no_of_signals,
            dtype=[
                ("signal", f"{signal_length}f8"),
                ("fs", "f8"),
                ("azimuth", "f8"),
                ("colatitude", "f8"),
                ("radius", "f8"),
                ("grid_weights", "f8"),
                ("air_temperature", "f8"),
            ],
        )
    )


def load_array_signal(filename):
    """Convenience function to load ArraySignal saved into np data structures.

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
    """Reads in WAV files and returns data [Nsig x Nsamples] and fs.

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


# noinspection PyPep8Naming
def write_SSR_IRs(filename, time_data_l, time_data_r, wavformat="float32"):
    """Takes two time signals and writes out the horizontal plane as HRIRs for
    the SoundScapeRenderer. Ideally, both hold 360 IRs but smaller sets are
    tried to be scaled up using repeat.

    Parameters
    ----------
    filename : string
        filename to write to
    time_data_l, time_data_r : io.ArraySignal
        ArraySignals for left/right ear
    wavformat : {float32, int32, int16}, optional
        wav file format to write [Default: float32]

    Raises
    ------
    ValueError
        in case unknown wavformat is provided
    ValueError
        in case integer format should be exported and amplitude exceeds 1.0
    """
    # make lower case and remove spaces
    wavformat = wavformat.lower().strip()

    # equator_IDX_left = utils.nearest_to_value_logical_IDX(
    #     time_data_l.grid.colatitude, _np.pi / 2
    # )
    # equator_IDX_right = utils.nearest_to_value_logical_IDX(
    #     time_data_r.grid.colatitude, _np.pi / 2
    # )

    # irs_left = time_data_l.signal.signal[equator_IDX_left]
    # irs_right = time_data_r.signal.signal[equator_IDX_right]
    irs_left = time_data_l.signal.signal
    irs_right = time_data_r.signal.signal

    irs_to_write = utils.interleave_channels(
        left_channel=irs_left, right_channel=irs_right, style="SSR"
    )
    # data_to_write = utils.simple_resample(
    #     irs_to_write, original_fs=time_data_l.signal.fs, target_fs=44100
    # )
    data_to_write = irs_to_write

    # get absolute max value
    max_val = _np.abs([time_data_l.signal.signal, time_data_r.signal.signal]).max()
    if max_val > 1.0:
        if "int" in wavformat:
            raise ValueError(
                "At least one amplitude value exceeds 1.0, exporting to an "
                "integer format will lead to clipping. Choose wavformat "
                '"float32" instead or normalize data!'
            )
        print("WARNING: At least one amplitude value exceeds 1.0!", file=sys.stderr)

    if wavformat in ["float32", "float"]:
        sio.wavfile.write(
            filename=filename,
            rate=int(time_data_l.signal.fs),
            data=data_to_write.T.astype(_np.float32),
        )
    elif wavformat in ["int32", "int16"]:
        sio.wavfile.write(
            filename=filename,
            rate=int(time_data_l.signal.fs),
            data=(data_to_write.T * _np.iinfo(wavformat).max).astype(wavformat),
        )
    else:
        raise ValueError(
            f'Format "{wavformat}" unknown, should be either "float32", "int32" or "int16".'
        )
