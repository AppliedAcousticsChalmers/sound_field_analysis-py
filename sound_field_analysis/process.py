"""
Module containing functions to act on the Spatial Fourier Coefficients.

`BEMA`
    Perform Bandwidth Extension for Microphone Arrays (BEMA) spatial
    anti-aliasing.
`FFT`
    Perform real-valued Fast Fourier Transform (FFT).
`iFFT`
    Perform inverse real-valued (Fast) Fourier Transform (iFFT).
`spatFT`
    Perform spatial Fourier transform.
`spatFT_RT`
    Perform spatial Fourier transform for real-time applications
    (otherwise use `spatFT()` for more more convenience and flexibility).
`spatFT_LSF`
    Perform spatial Fourier transform by least square fit to provided data.
`iSpatFT`
    Perform inverse spatial Fourier transform.
`plane_wave_decomp`
    Perform plane wave decomposition.
`rfi`
    Perform radial filter improvement (RFI)
`sfe`
    Perform sound field extrapolation (SFE) - WORK IN PROGRESS.
`wdr`
    Perform Wigner-D rotation (WDR) - NOT YET IMPLEMENTED.
`convolve`
    Convolve two arrays A & B row-wise where one or both can be
    one-dimensional for SIMO/SISO convolution.
"""
import sys

import numpy as _np
from scipy.linalg import lstsq
from scipy.signal import butter, fftconvolve, sosfreqz

from .io import SphericalGrid, TimeSignal
from .sph import besselj, hankel1, sph_harm_all


def BEMA(Pnm, center_sig, dn, transition, avg_band_width=1, fade=True, max_order=None):
    """Perform Bandwidth Extension for Microphone Arrays (BEMA) spatial
    anti-aliasing, according to [4]_.

    Parameters
    ----------
    Pnm : array_like
        Sound field SH spatial Fourier coefficients
    center_sig : array_like
        Center microphone in shape [0, NFFT]
    dn : array_like
        Radial filters for the current array configuration
    transition : int
        Highest stable bin, approx: transition = (NFFT/fs) * (N*c)/(2*pi*r)
    avg_band_width : int, optional
        Averaging Bandwidth in oct [Default: 1]
    fade : bool, optional
        Fade over if True, else hard cut [Default: True]
    max_order : int, optional
        Maximum transform order [Default: highest available order]

    Returns
    -------
    Pnm : array_like
        BEMA optimized sound field SH coefficients

    References
    ----------
    .. [4] Bernschütz, B. (2012). “Bandwidth Extension for Microphone Arrays,”
    AES Conv. 133, Audio Engineering Society, San Francisco, USA, 1–10.
    """

    if not max_order:
        max_order = int(_np.sqrt(Pnm.shape[0] - 1))  # SH order

    transition = int(_np.floor(transition))

    # computing spatio-temporal Image Imn
    Imn = _np.zeros((Pnm.shape[0], 1), dtype=complex)
    # first bin for averaging
    start_avg_bin = int(_np.floor(transition / (_np.power(2, avg_band_width))))

    modeCnt = 0
    avgPower = 0
    for n in range(0, max_order + 1):  # sh orders
        for _ in range(0, 2 * n + 1):  # modes
            for inBin in range(start_avg_bin - 1, transition):  # bins
                synthBin = Pnm[modeCnt, inBin] * dn[n, inBin]
                Imn[modeCnt, 0] += synthBin
                avgPower += _np.abs(synthBin) ** 2
            modeCnt += 1

    # energy normalization (Eq. 9)
    energySum = _np.sum(_np.power(_np.abs(Imn), 2))
    normFactor = avgPower / (transition - start_avg_bin + 1)
    Imn *= _np.sqrt(normFactor / energySum)

    # normalize center signal to its own average level in the source band
    # (Eq. 13)
    center_sig[_np.where(center_sig == 0)] = 1e-12
    sq_avg = _np.sqrt(
        _np.mean(_np.power(_np.abs(center_sig[:, (start_avg_bin - 1) : transition]), 2))
    )
    center_sig = _np.multiply(center_sig, (1 / sq_avg))

    Pnm_synth = _np.zeros(Pnm.shape, dtype=complex)
    modeCnt = 0
    for n in range(0, max_order + 1):
        for _ in range(0, 2 * n + 1):
            for inBin in range(start_avg_bin - 1, Pnm_synth.shape[1]):
                Pnm_synth[modeCnt, inBin] = (
                    Imn[modeCnt, 0] * (1 / dn[n, inBin]) * center_sig[0, inBin]
                )
            modeCnt += 1

    # Phase correction (Eq. 16)
    phaseOffset = _np.angle(Pnm[0, transition] / Pnm_synth[0, transition])
    Pnm_synth *= _np.exp(1j * phaseOffset)

    # Merge original and synthetic SH coefficients. (Eq. 14)
    Pnm = _np.concatenate(
        (Pnm[:, 0 : (transition - 1)], Pnm_synth[:, (transition - 1) :]), axis=1
    )

    # Fade in of synthetic SH coefficients. (Eq. 17)
    if fade:
        Pnm_fade0 = Pnm[:, (start_avg_bin - 1) : (transition - 1)]
        Pnm_fadeS = Pnm_synth[:, (start_avg_bin - 1) : (transition - 1)]

        fadeUp = _np.linspace(0, 1, Pnm_fade0.shape[1])
        fadeDown = _np.flip(fadeUp)

        Pnm_fade0 *= fadeDown
        Pnm_fadeS *= fadeUp
        Pnm[:, start_avg_bin - 1 : transition - 1] = Pnm_fade0 + Pnm_fadeS

    return Pnm


def FFT(
    time_signals,
    fs=None,
    NFFT=None,
    oversampling=1,
    first_sample=0,
    last_sample=None,
    calculate_freqs=True,
):
    """Perform real-valued Fast Fourier Transform (FFT).

    Parameters
    ----------
    time_signals : TimeSignal/tuple/object
        Time-domain signals to be transformed.
    fs : int, optional
        Sampling frequency - only optional no frequency vector should be
        calculated or if a TimeSignal or tuple/array containing fs is passed
    NFFT : int, optional
        Number of frequency bins. Resulting array will have size NFFT//2+1
        [Default: Next power of 2]
    oversampling : int, optional
        Oversample the incoming signal to increase frequency resolution
        [Default: 1]
    first_sample : int, optional
        First time domain sample to be included [Default: 0]
    last_sample : int, optional
        Last time domain sample to be included [Default: -1]
    calculate_freqs : bool, optional
        Calculate frequency scale if True, else return only spectrum
        [Default: True]

    Returns
    -------
    fftData : array_like
        One-sided frequency domain spectrum
    f : array_like, optional
        Vector of frequency bins of one-sided spectrum, in case of
        calculate_freqs

    Notes
    -----
    An oversampling*NFFT point Fourier Transform is applied to the time domain
    data, where NFFT is the next power of two of the number of samples.
    Time-windowing can be used by providing a first_sample and last_sample
    index.
    """
    if isinstance(time_signals, TimeSignal):
        signals = time_signals.signal
        fs = time_signals.fs
    elif fs is None and calculate_freqs:
        raise ValueError(
            "No valid signal found. Either pass an io.TimeSignal, a "
            "tuple/array containing the signal and the sampling frequency "
            "or use the fs argument."
        )
    else:
        signals = time_signals

    signals = _np.atleast_2d(signals)

    if last_sample is None:
        # assign lastSample to length of signals if not provided
        last_sample = signals.shape[1]

    if oversampling < 1:
        raise ValueError("oversampling must be >= 1.")

    if last_sample < first_sample or last_sample > signals.shape[1]:
        raise ValueError("lastSample must be between firstSample and nSamples.")

    if first_sample < 0 or first_sample > last_sample:
        raise ValueError("firstSample must be between 0 and lastSample.")

    total_samples = last_sample - first_sample
    signals = signals[:, first_sample:last_sample]

    if not NFFT:
        NFFT = int(2 ** _np.ceil(_np.log2(total_samples)))

    fftData = _np.fft.rfft(signals, NFFT * oversampling, 1)
    if not calculate_freqs:
        return fftData

    f = _np.fft.rfftfreq(NFFT * oversampling, d=1 / fs)
    return fftData, f


def iFFT(Y, output_length=None, window=False):
    """Perform inverse real-valued (Fast) Fourier Transform (iFFT).

    Parameters
    ----------
    Y : array_like
        Frequency domain data [Nsignals x Nbins]
    output_length : int, optional
        Length of returned time-domain signal (Default: 2 x len(Y) + 1)
    window : str, optional
        Window applied to the resulting time-domain signal

    Returns
    -------
    y : array_like
        Reconstructed time-domain signal
    """
    Y = _np.atleast_2d(Y)
    y = _np.fft.irfft(Y, n=output_length)

    if window:
        no_of_samples = y.shape[-1]

        window = window.lower()
        if window == "hann":
            window_array = _np.hanning(no_of_samples)
        elif window == "hamming":
            window_array = _np.hamming(no_of_samples)
        elif window == "blackman":
            window_array = _np.blackman(no_of_samples)
        elif window == "kaiser":
            window_array = _np.kaiser(no_of_samples, 3)
        else:
            raise ValueError(
                "Selected window must be one of hann, hamming, blackman or kaiser."
            )

        y *= window_array

    return y


def spatFT(
    data, position_grid, order_max=10, kind="complex", spherical_harmonic_bases=None
):
    """Perform spatial Fourier transform.

    Parameters
    ----------
    data : array_like
        Data to be transformed, with signals in rows and frequency bins in
        columns
    position_grid : array_like or io.SphericalGrid
        Azimuths/Colatitudes/Gridweights of spatial sampling points
    order_max : int, optional
        Maximum transform order [Default: 10]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]

    Returns
    -------
    Pnm : array_like
        Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in
        columns

    Notes
    -----
    In case no weights in spatial sampling grid are given, the pseudo inverse
    of the SH bases is computed according to Eq. 3.34 in [5]_.

    References
    ----------
    .. [5] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
    (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg, 2nd ed.,
    196 pages. doi:10.1007/978-3-319-99561-8
    """
    data = _np.atleast_2d(data)
    position_grid = SphericalGrid(*position_grid)

    # Re-generate spherical harmonic bases if they were not provided or their
    # order is too small
    if (
        spherical_harmonic_bases is None
        or spherical_harmonic_bases.shape[0] < data.shape[0]
        or spherical_harmonic_bases.shape[1] < (order_max + 1) ** 2
    ):
        spherical_harmonic_bases = sph_harm_all(
            order_max, position_grid.azimuth, position_grid.colatitude, kind=kind
        )

    if position_grid.weight is None:
        # calculate pseudo inverse in case no spatial sampling point weights
        # are given
        spherical_harmonic_weighted = _np.linalg.pinv(spherical_harmonic_bases)
    else:
        # apply spatial sampling point weights in case they are given
        spherical_harmonic_weighted = _np.conj(spherical_harmonic_bases).T * (
            4 * _np.pi * position_grid.weight
        )

    return spatFT_RT(data, spherical_harmonic_weighted)


def spatFT_RT(data, spherical_harmonic_weighted):
    """Perform spatial Fourier transform for real-time applications
    (otherwise use `spatFT()` for more more convenience and flexibility).

    Parameters
    ----------
    data : array_like
        Data to be transformed, with signals in rows and frequency bins in
        columns
    spherical_harmonic_weighted : array_like
        Spherical harmonic base coefficients (already weighted by spatial
        sampling grid)

    Returns
    -------
    Pnm : array_like
        Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in
        columns
    """
    return _np.dot(spherical_harmonic_weighted, data)


def spatFT_LSF(
    data, position_grid, order_max=10, kind="complex", spherical_harmonic_bases=None
):
    """Perform spatial Fourier transform by least square fit to provided data.

    Parameters
    ----------
    data : array_like, complex
        Data to be fitted to
    position_grid : array_like, or io.SphericalGrid
        Azimuth / colatitude data locations
    order_max: int, optional
        Maximum transform order [Default: 10]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]

    Returns
    -------
    coefficients: array_like, float
        Fitted spherical harmonic coefficients (indexing: n**2 + n + m + 1)
    """
    # Re-generate spherical harmonic bases if they were not provided or their order is too small
    if (
        spherical_harmonic_bases is None
        or spherical_harmonic_bases.shape[0] < data.shape[0]
        or spherical_harmonic_bases.shape[1] < (order_max + 1) ** 2
    ):
        position_grid = SphericalGrid(*position_grid)
        spherical_harmonic_bases = sph_harm_all(
            order_max, position_grid.azimuth, position_grid.colatitude, kind=kind
        )

    return lstsq(spherical_harmonic_bases, data)[0]


def iSpatFT(
    spherical_coefficients,
    position_grid,
    order_max=None,
    kind="complex",
    spherical_harmonic_bases=None,
):
    """Perform inverse spatial Fourier transform.

    Parameters
    ----------
    spherical_coefficients : array_like
        Spatial Fourier coefficients with columns representing frequency bins
    position_grid : array_like or io.SphericalGrid
        Azimuth/Colatitude angles of spherical coefficients
    order_max : int, optional
        Maximum transform order [Default: highest available order]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]

    Returns
    -------
    P : array_like
        Sound pressures with frequency bins in columns and angles in rows

    TODO
    ----
    Check `spherical_coefficients` and `spherical_harmonic_bases` length
    correspond with `order_max`
    """
    position_grid = SphericalGrid(*position_grid)
    spherical_coefficients = _np.atleast_2d(spherical_coefficients)
    number_of_coefficients = spherical_coefficients.shape[0]

    # TODO: Check `spherical_coefficients` and `spherical_harmonic_bases` length
    #  correspond with `order_max`
    if order_max is None:
        order_max = int(_np.sqrt(number_of_coefficients) - 1)

    # Re-generate spherical harmonic bases if they were not provided or their
    # order is too small
    if (
        spherical_harmonic_bases is None
        or spherical_harmonic_bases.shape[1] < number_of_coefficients
        or spherical_harmonic_bases.shape[1] != position_grid.azimuths.size
    ):
        spherical_harmonic_bases = sph_harm_all(
            order_max, position_grid.azimuth, position_grid.colatitude, kind=kind
        )

    return _np.dot(spherical_harmonic_bases, spherical_coefficients)


def plane_wave_decomp(
    order, wave_direction, field_coeffs, radial_filter, weights=None, kind="complex"
):
    """Perform plane wave decomposition.

    Parameters
    ----------
    order : int
        Decomposition order
    wave_direction : array_like
        Direction of plane wave as [azimuth, colatitude] pair.
        io.SphericalGrid is used internally
    field_coeffs : array_like
        Spatial fourier coefficients
    radial_filter : array_like
        Radial filters
    weights : array_like, optional
        Weighting function. Either scalar, one per directions or of dimension
        (nKR_bins x  nDirections). [Default: None]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']

    Returns
    -------
    Y : matrix of floats
        Matrix of the decomposed wave field with kr bins in rows
    """
    wave_direction = SphericalGrid(*wave_direction)
    number_of_angles = wave_direction.azimuth.size
    field_coeffs = _np.atleast_2d(field_coeffs)
    radial_filter = _np.atleast_2d(radial_filter)

    if field_coeffs.shape[0] == 1:
        field_coeffs = field_coeffs.T
    if radial_filter.shape[0] == 1:
        radial_filter = radial_filter.T

    NMDeliveredSize, FFTBlocklengthPnm = field_coeffs.shape
    Ndn, FFTBlocklengthdn = radial_filter.shape

    if FFTBlocklengthdn != FFTBlocklengthPnm:
        raise ValueError(
            "FFT Blocksizes of field coefficients (Pnm) and "
            "radial filter (dn) are not consistent."
        )

    max_order = int(_np.floor(_np.sqrt(NMDeliveredSize) - 1))
    if order > max_order:
        raise ValueError(
            f"The provided coefficients deliver a maximum order of {max_order} "
            f"but order {order} was requested."
        )

    gaincorrection = 4 * _np.pi / ((order + 1) ** 2)

    if weights is not None:
        weights = _np.asarray(weights)
        if weights.ndim == 1:
            number_of_weights = weights.size
            if (
                number_of_weights != FFTBlocklengthPnm
                and number_of_weights != number_of_angles
                and number_of_weights != 1
            ):
                raise ValueError(
                    "Weights is not a scalar nor consistent with shape of the "
                    "field coefficients (Pnm)."
                )
            if number_of_weights == number_of_angles:
                weights = _np.broadcast_to(
                    weights, (FFTBlocklengthPnm, number_of_angles)
                ).T
        else:
            if weights.shape != (number_of_angles, FFTBlocklengthPnm):
                raise ValueError(
                    "Weights is not a scalar nor consistent with shape of field "
                    "coefficients (Pnm) and radial filter (dn)."
                )
    else:
        weights = 1

    sph_harms = sph_harm_all(
        order, wave_direction.azimuth, wave_direction.colatitude, kind=kind
    )
    filtered_coeffs = field_coeffs * _np.repeat(
        radial_filter, _np.r_[: order + 1] * 2 + 1, axis=0
    )
    OutputArray = _np.dot(sph_harms, filtered_coeffs)
    OutputArray = _np.multiply(OutputArray, weights)
    return OutputArray * gaincorrection


def rfi(dn, kernelSize=512, highPass=0.0):
    """Perform radial filter improvement (RFI), according to [6]_.

    Parameters
    ----------
    dn : array_like
        Analytical frequency domain radial filters
        (e.g. `gen.radial_filter_fullspec()`)
    kernelSize : int, optional
        Target filter kernel size [Default: 512]
    highPass : float, optional
        Highpass Filter from 0.0 (off) to 1.0 (maximum kr) [Default: 0.0]

    Returns
    -------
    dn : array_like
        Improved radial filters
    kernelSize : int
        Filter kernel size (total)
    latency : float
        Approximate signal latency due to the filters

    Notes
    -----
    This function improves the FIR radial filters from
    `gen.radial_filter_fullspec()`. The filters are made causal and are
    windowed in time domain. The DC components are estimated. The R/F/I module
    should always be inserted to the filter path when treating measured data
    even if no use is made of the included kernel downscaling or highpass
    filters.

    Do NOT use R/F/I for single open sphere filters (e.g. simulations).

    IMPORTANT
        Remember to choose a kernel size being large enough to cover all filter
        latencies and response slopes. Otherwise undesired cyclic convolution
        artifacts may appear in the output signal.

    HIGHPASS
        If HPF is on (0<highPass<=1) the radial filters and HPF share the
        available taps and the latency keeps constant. Be careful when using
        very small kernel sizes because since there might be too few taps.
        Observe the filters by plotting their spectra and impulse responses!
        > Be very careful if NFFT/max(kr) < 25
        > Do not use R/F/I if NFFT/max(kr) < 15

    References
    ----------
    .. [6] Bernschütz, B., Pörschmann, C., Spors, S., and Weinzierl, S. (2011).
    “SOFiA Sound Field Analysis Toolbox,” Int. Conf. Spat. Audio, Verband
    Deutscher Tonmeister e.V., Detmold, Germany, 7–15.

    TODO
    ----
    Implement `kernelsize` extension
    """
    dn = _np.atleast_2d(dn)
    sourceKernelSize = (dn.shape[-1] - 1) * 2

    if kernelSize > sourceKernelSize:
        # TODO: Implement `kernelsize` extension
        raise ValueError(
            "Kernelsize greater than radial filters. Extension of "
            "kernelsize not yet implemented."
        )

    # in case highpass should be applied, half both individual kernel sizes
    endKernelSize = kernelSize
    if highPass:
        kernelSize //= 2

    # DC-component estimation
    dn_diff = _np.abs(dn[:, 1] / dn[:, 2])
    oddOrders = range(1, dn.shape[0], 2)
    dn[oddOrders, 0] = (
        -1j
        * dn[oddOrders, 1]
        * 2
        * (sourceKernelSize / kernelSize)
        * dn_diff[oddOrders]
    )

    # transform into time domain
    dn_ir = iFFT(dn)

    # make filters causal by circular shift
    dn_ir = _np.roll(dn_ir, dn_ir.shape[-1] // 2, axis=-1)

    # downsize kernel
    latency = kernelSize / 2
    mid = dn_ir.shape[-1] / 2
    dn_ir = dn_ir[:, int(mid - latency) : int(mid + latency)]

    # apply Hanning / cosine window
    # 0.5 + 0.5 * _np.cos(2 * _np.pi * (_np.arange(0, kernelSize)
    # - ((kernelSize - 1) / 2)) / (kernelSize - 1))
    dn_ir *= _np.hanning(kernelSize)

    # transform into one-sided spectrum
    dn = FFT(dn_ir, NFFT=endKernelSize, calculate_freqs=False)

    # calculate high pass (need to be zero phase since radial filters are
    # already linear phase)
    if 0 < highPass <= 1:
        # by designing an IIR filter and taking the magnitude response
        # calculate IIR filter coefficients
        hp_sos = butter(8, highPass, btype="highpass", output="sos", analog=False)
        # calculate "equivalent" zero phase FIR filter one-sided spectrum
        _, HP = sosfreqz(hp_sos, worN=kernelSize // 2 + 1, whole=False)
        HP[_np.isnan(HP)] = 0  # prevent NaNs
        # make filter zero phase by circular shift
        hp = _np.roll(iFFT(HP), kernelSize // 2, axis=-1)
        latency += kernelSize / 2
        # append zeros in time domain
        HP = FFT(hp, NFFT=endKernelSize, calculate_freqs=False)

        # # by designing an FIR filter with FIRLS
        # from scipy.signal import firls
        #
        # STOP_BAND_ATTENUATION = -30
        # # calculate FIR linear phase filter coefficients
        # bands = [0, highPass / 2, highPass, 1]
        # # magnitudes as linear
        # desired = 10 ** (
        #     _np.array([STOP_BAND_ATTENUATION, STOP_BAND_ATTENUATION, 0, 0]) / 20
        # )
        # hp = firls(kernelSize - 1, bands=bands, desired=desired)
        # # make filter zero phase by circular shift
        # hp = _np.roll(hp, hp.shape[-1] // 2, axis=-1)
        # # transform into one-sided spectrum
        # HP = FFT(hp, calculate_freqs=False)

        # # by designing an FIR filter with FIRWIN
        # from scipy.signal import firwin
        #
        # # calculate FIR linear phase filter coefficients
        # hp = firwin(kernelSize - 1, highPass, pass_zero=False, window="hamming")
        # # make filter zero phase by circular shift
        # hp = _np.roll(hp, hp.shape[-1] // 2, axis=-1)
        # # transform into one-sided spectrum
        # HP = FFT(hp, calculate_freqs=False)

        # apply high pass
        dn *= HP

    return dn, kernelSize, latency


def sfe(Pnm_kra, kra, krb, problem="interior"):
    """Perform sound field extrapolation (SFE) - WORK IN PROGRESS.

    Parameters
    ----------
    Pnm_kra : array_like
        Spatial Fourier coefficients (e.g. from `spatFT()`)
    kra,krb : array_like
        k * ra/rb vector
    problem : string{'interior', 'exterior'}
        Select between interior and exterior problem [Default: interior]

    Returns
    -------
    Pnm_kra : array_like
        Extrapolated spatial Fourier coefficients

    TODO
    ----
    Verify sound field extrapolation
    """
    if kra.shape[1] != Pnm_kra.shape[1] or kra.shape[1] != krb.shape[1]:
        raise ValueError("FFTData: Complex Input Data expected.")
    problem = problem.lower()
    if problem not in {"interior", "exterior"}:
        raise ValueError("Invalid problem: Choose either interior or exterior.")

    FCoeff = Pnm_kra.shape[0]
    N = int(_np.floor(_np.sqrt(FCoeff) - 1))

    nvector = _np.zeros(FCoeff)
    IDX = 1
    for n in range(0, N + 1):
        for _ in range(-n, n + 1):
            nvector[IDX] = n
            IDX += 1

    nvector = _np.tile(nvector, (1, Pnm_kra.shape[1]))
    kra = _np.tile(kra, (FCoeff, 1))
    krb = _np.tile(krb, (FCoeff, 1))

    if problem == "interior":
        jn_kra = _np.sqrt(_np.pi / (2 * kra)) * besselj(nvector + 5, kra)
        jn_krb = _np.sqrt(_np.pi / (2 * krb)) * besselj(nvector + 5, krb)
        exp = jn_krb / jn_kra

        if _np.any(_np.abs(exp) > 1e2):  # 40dB
            print(
                "WARNING: Extrapolation might be unstable for one or more "
                "frequencies/orders!",
                file=sys.stderr,
            )

    else:  # problem == 'exterior':
        hn_kra = _np.sqrt(_np.pi / (2 * kra)) * hankel1(nvector + 0.5, kra)
        hn_krb = _np.sqrt(_np.pi / (2 * krb)) * hankel1(nvector + 0.5, krb)
        exp = hn_krb / hn_kra

    # TODO: Verify sound field extrapolation
    return Pnm_kra * exp.T


# noinspection PyUnusedLocal
def wdr(Pnm, xAngle, yAngle, zAngle):
    """Perform Wigner-D rotation (WDR) - NOT YET IMPLEMENTED.

    Parameters
    ----------
    Pnm : array_like
        Spatial Fourier coefficients
    xAngle, yAngle, zAngle : float
        Rotation angle around the x/y/z-Axis

    Returns
    -------
    PnmRot: array_like
        Rotated spatial Fourier coefficients

    TODO
    ----
    Implement Wigner-D rotations
    """
    # TODO: Implement Wigner-D rotations
    print(
        "!WARNING. Wigner-D Rotation is not yet implemented. Continuing with "
        "un-rotated coefficients!",
        file=sys.stderr,
    )
    return Pnm


def convolve(A, B, FFT=None):
    """Convolve two arrays A & B row-wise where one or both can be
    one-dimensional for SIMO/SISO convolution.

    Parameters
    ----------
    A, B: array_like
        Data to perform the convolution on of shape [Nsignals x NSamples]
    FFT: bool, optional
        Selects whether time or frequency domain convolution is applied.
        [Default: On if Nsamples > 500 for both]

    Returns
    -------
    out: array
        Array containing row-wise, linear convolution of A and B
    """
    A = _np.atleast_2d(A)
    B = _np.atleast_2d(B)

    N_sigA, L_sigA = A.shape
    N_sigB, L_sigB = B.shape
    FFT = FFT is None and (L_sigA > 500 and L_sigB > 500)

    if (N_sigA != N_sigB) and not (N_sigA == 1 or N_sigB == 1):
        raise ValueError(
            "Number of rows must either match or at least one "
            "must be one-dimensional."
        )

    if N_sigA == 1 and N_sigB != 1:
        A = _np.broadcast_to(A, (N_sigB, L_sigA))
    elif N_sigA != 1 and N_sigB == 1:
        B = _np.broadcast_to(B, (N_sigA, L_sigB))

    out = []

    for IDX, cur_row in enumerate(A):
        out.append(
            fftconvolve(cur_row, B[IDX]) if FFT else _np.convolve(cur_row, B[IDX])
        )

    return _np.array(out)
