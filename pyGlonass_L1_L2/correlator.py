# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
pySoftGNSS: A Python-Based GNSS Software Receiver.

pySoftGNSS is a Python implementation of a post-processing GNSS software
receiver. Its overall receiver architecture is inspired by the open-source
MATLAB SoftGNSS project and follows the conventional processing chain of
signal acquisition, tracking, navigation-message decoding, and position
computation.

The Python source code, class organization, data interfaces, and integration
of the SIMD- and GPU-accelerated correlators were developed specifically for
pySoftGNSS. SoftGNSS is acknowledged as the architectural reference, while
pySoftGNSS is maintained as a separate Python implementation.

Author:
Yafeng Li
School of Automation
Beijing Information Science and Technology University

August 2026

Copyright (C) 2026 Yafeng Li.
-------------------------------------------------------------------------------

correlator.py - Module Description
----------------------------------
Generate, sample, and correlate GLONASS L1/L2 C/A codes.

"""

import ctypes

import numpy as np

#%% CPU SIMD correlator for channel-serial tracking
class CorrSIMDSerialBPSK:
    """Call the channel-serial AVX2 BPSK tracking correlator."""

    def __init__(self):
        """Load the SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDSerialBPSK.dll"
        self._simdLibrary = ctypes.CDLL(dllPath)

        int16Vector = np.ctypeslib.ndpointer(
            dtype=np.int16, ndim=1, flags="C_CONTIGUOUS")
        int32Vector = np.ctypeslib.ndpointer(
            dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

        self._simdLibrary.corrEngine.argtypes = [
            ctypes.c_int,     # settings.fileType
            ctypes.c_int,     # rawSignal.size
            ctypes.c_int,     # settings.rShiftBits
            ctypes.c_double,  # settings.dllCorrelatorSpacing
            int16Vector,      # rawSignal
            int32Vector,      # caCode
            ctypes.c_int,     # caCode.size
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, caCode, remCarrPhase,
                   carrPhaseStep, remCodePhase, codePhaseStep):
        """Correlate one channel and one code period.

        Args
        ----
            settings    - object
                        Receiver settings containing ``fileType``,
                        ``rShiftBits`` and ``dllCorrelatorSpacing``.
            rawSignal   - numpy.ndarray
                        Input int16 IF samples. Complex samples are stored as
                        interleaved I/Q values.
            caCode      - numpy.ndarray
                        Guarded int32 code with layout
                        ``[last, one period, first]``.
            remCarrPhase - float
                         Residual carrier phase in radians.
            carrPhaseStep - float
                          Carrier phase increment in radians per sample.
            remCodePhase - float
                         Residual code phase in chips.
            codePhaseStep - float
                          Code phase increment in chips per sample.
        Returns
        -------
            correValues - numpy.ndarray
                        Correlations ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            caCode,
            caCode.size,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
        )
        if not corrPointer:
            raise RuntimeError("SIMD correlator initialization failed")

        # The C++ output is static and is overwritten by the next call.
        return np.ctypeslib.as_array(corrPointer, shape=(6,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the C++ engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-serial tracking
class CorrGPUSerialBPSK:
    """Call the channel-serial CUDA BPSK tracking correlator."""

    def __init__(self):
        """Load the CUDA DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrGPUSerialBPSK.dll"
        self._gpuLibrary = ctypes.CDLL(dllPath)

        int16Vector = np.ctypeslib.ndpointer(
            dtype=np.int16, ndim=1, flags="C_CONTIGUOUS")
        int8Vector = np.ctypeslib.ndpointer(
            dtype=np.int8, ndim=1, flags="C_CONTIGUOUS")

        self._gpuLibrary.corrEngine.argtypes = [
            ctypes.c_int,     # settings.fileType
            ctypes.c_int,     # rawSignal.size
            ctypes.c_double,  # settings.dllCorrelatorSpacing
            int16Vector,      # rawSignal
            int8Vector,       # caCode
            ctypes.c_int,     # caCode.size
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # Frequency channel K
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, caCode, remCarrPhase,
                   carrPhaseStep, remCodePhase, codePhaseStep, K):
        """Correlate one channel and one code period on the GPU.

        Args
        ----
            settings    - object
                        Receiver settings containing ``fileType`` and
                        ``dllCorrelatorSpacing``.
            rawSignal   - numpy.ndarray
                        Input int16 IF samples. Complex samples are stored as
                        interleaved I/Q values.
            caCode      - numpy.ndarray
                        Guarded int8 code with layout
                        ``[last, one period, first]``.
            remCarrPhase - float
                         Residual carrier phase in radians.
            carrPhaseStep - float
                          Carrier phase increment in radians per sample.
            remCodePhase - float
                         Residual code phase in chips.
            codePhaseStep - float
                          Code phase increment in chips per sample.
            K           - int
                        Current GLONASS frequency channel used for code
                        caching.

        Returns
        -------
            correValues - numpy.ndarray
                        Correlations ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            caCode,
            caCode.size,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            int(K),
        )
        if not corrPointer:
            raise RuntimeError("GPU serial correlator failed")
        return np.ctypeslib.as_array(corrPointer, shape=(6,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% CPU SIMD correlator for channel-parallel tracking
class CorrSIMDParallelBPSK:
    """Call the channel-parallel AVX2 BPSK tracking correlator."""

    def __init__(self):
        """Load the parallel SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDParallelBPSK.dll"
        self._simdLibrary = ctypes.CDLL(dllPath)

        int16Vector = np.ctypeslib.ndpointer(
            dtype=np.int16, ndim=1, flags="C_CONTIGUOUS")
        int32Vector = np.ctypeslib.ndpointer(
            dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        int32Table = np.ctypeslib.ndpointer(
            dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
        float64Vector = np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

        self._simdLibrary.corrEngine.argtypes = [
            ctypes.c_int,    # settings.fileType
            ctypes.c_int,    # rawSignal.size
            ctypes.c_int,    # settings.rShiftBits
            ctypes.c_double, # settings.dllCorrelatorSpacing
            int16Vector,     # rawSignal
            int32Table,      # caCodeTable
            ctypes.c_int,    # codeLen
            ctypes.c_int,    # channelCnt
            float64Vector,   # remCarrPhase
            float64Vector,   # carrPhaseStep
            float64Vector,   # remCodePhase
            float64Vector,   # codePhaseStep
            int32Vector,     # startIdx
            int32Vector,     # chSampSize
            ctypes.c_int,    # isDataRead
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)
        # The clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, caCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep,
                   startIdx, chSampSize, isDataRead):
        """Correlate all active channels against one shared IF block.

        Args
        ----
            settings    - object
                        Receiver settings containing ``fileType``,
                        ``rShiftBits`` and ``dllCorrelatorSpacing``.
            rawSignal   - numpy.ndarray
                        Shared input int16 IF signal.
            caCodeTable - numpy.ndarray
                        C-contiguous int32 guarded-code table with one row
                        per channel.
            remCarrPhase - numpy.ndarray
                         Initial carrier phases in radians.
            carrPhaseStep - numpy.ndarray
                          Carrier phase increments in radians per sample.
            remCodePhase - numpy.ndarray
                         Initial code phases in chips.
            codePhaseStep - numpy.ndarray
                          Code phase increments in chips per sample.
            startIdx    - numpy.ndarray
                        Zero-based logical sample offsets in the shared
                        signal block.
            chSampSize  - numpy.ndarray
                        Code-period sample count for each channel.
            isDataRead  - int
                        Shared-data refresh flag.

        Returns
        -------
            correValues - numpy.ndarray
                        Correlation matrix with shape ``(6, channelCnt)``;
                        rows are ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        codeLen = caCodeTable.shape[1]

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            caCodeTable,
            codeLen,
            channelCnt,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            startIdx,
            chSampSize,
            isDataRead,
        )
        if not corrPointer:
            raise RuntimeError("SIMD parallel correlator failed")

        corrValues = np.ctypeslib.as_array(corrPointer, shape=(6 * channelCnt,))
        return corrValues.reshape(channelCnt, 6).T.copy()

    def close(self):
        """Release the reusable buffers allocated by the SIMD engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-parallel tracking
class CorrGPUParallelBPSK:
    """Call the channel-parallel CUDA BPSK tracking correlator."""

    def __init__(self):
        """Load the CUDA DLL and declare its ctypes interface.

        Returns
        -------
            None
        """
        dllPath = "../native_Correlators/corrGPUParallelBPSK.dll"
        self._gpuLibrary = ctypes.CDLL(dllPath)

        int16Vector = np.ctypeslib.ndpointer(
            dtype=np.int16, ndim=1, flags="C_CONTIGUOUS")
        int8Table = np.ctypeslib.ndpointer(
            dtype=np.int8, ndim=2, flags="C_CONTIGUOUS")
        int32Vector = np.ctypeslib.ndpointer(
            dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        float64Vector = np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

        self._gpuLibrary.corrEngine.argtypes = [
            ctypes.c_int,     # settings.fileType
            ctypes.c_int,     # rawSignal.size
            ctypes.c_double,  # settings.dllCorrelatorSpacing
            int16Vector,      # rawSignal
            int8Table,        # caCodeTable
            ctypes.c_int,     # codeLen
            ctypes.c_int,     # channelCnt
            float64Vector,    # remCarrPhase
            float64Vector,    # carrPhaseStep
            float64Vector,    # remCodePhase
            float64Vector,    # codePhaseStep
            int32Vector,      # startIdx
            int32Vector,      # chSampSize
            ctypes.c_int,     # isDataRead
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)
        # The clearup function
        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, caCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep,
                   startIdx, chSampSize, isDataRead):
        """Correlate all active channels against one shared IF block.

        Args
        ----
            settings    - object
                        Receiver settings containing ``fileType`` and
                        ``dllCorrelatorSpacing``.
            rawSignal   - numpy.ndarray
                        Shared input int16 IF signal.
            caCodeTable - numpy.ndarray
                        C-contiguous int8 guarded-code table with one row per
                        channel.
            remCarrPhase - numpy.ndarray
                         Residual carrier phases in radians.
            carrPhaseStep - numpy.ndarray
                          Carrier phase increments in radians per sample.
            remCodePhase - numpy.ndarray
                         Residual code phases in chips.
            codePhaseStep - numpy.ndarray
                          Code phase increments in chips per sample.
            startIdx    - numpy.ndarray
                        Zero-based logical sample offsets in ``rawSignal``.
            chSampSize  - numpy.ndarray
                        Logical sample count for each channel.
            isDataRead  - int
                        ``1`` uploads new data; ``0`` reuses the data.

        Returns
        -------
            correValues - numpy.ndarray
                        Correlation matrix with shape ``(6, channelCnt)``;
                        rows are ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        codeLen = caCodeTable.shape[1]
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            caCodeTable,
            codeLen,
            channelCnt,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            startIdx,
            chSampSize,
            isDataRead,
        )
        if not corrPointer:
            raise RuntimeError("GPU parallel correlator failed")

        corrValues = np.ctypeslib.as_array(corrPointer, shape=(6 * channelCnt,))
        return corrValues.reshape(channelCnt, 6).T.copy()

    def close(self):
        """Release buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% Python (Numpy) correlator for channel-serial tracking
def corrPySerialBPSK(settings, rawSignal,
         codeChips, remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep):
    """Compute serial early, prompt, and late BPSK correlations.

    Args
    ----
        settings    - object
                    Receiver settings containing ``fileType`` and
                    ``dllCorrelatorSpacing``.
        rawSignal   - numpy.ndarray
                    Input IF signal. Complex samples are stored as
                    interleaved I/Q values.
        codeChips   - numpy.ndarray
                    Guarded code with layout
                    ``[last, one period, first]``.
        remCarrPhase - float
                     Residual carrier phase in radians.
        carrPhaseStep - float
                      Carrier phase increment in radians per sample.
        remCodePhase - float
                     Residual code phase in chips.
        codePhaseStep - float
                      Code phase increment in chips per sample.

    Returns
    -------
        correValues - tuple of float
                    Correlations ``(I_E, Q_E, I_P, Q_P, I_L, Q_L)``.
    """
    # Define early-late offset (in chips)
    earlyLateSpc = settings.dllCorrelatorSpacing
    # For complex data
    if settings.fileType == 2:
        rawSignal = rawSignal[::2] + rawSignal[1::2] * 1j

    blksize = rawSignal.size
    # --- Generate local code replica -----------------------------------------
    # Time index for each sampling point;
    sampleInd = np.arange(blksize)

    codeInd = sampleInd * codePhaseStep
    # Define index into early code vector
    tcode1 = np.ceil((remCodePhase - earlyLateSpc) + codeInd).astype(np.int32)
    earlyCode = codeChips.take(tcode1)

    # Define index into late code vector
    tcode2 = np.ceil((remCodePhase + earlyLateSpc) + codeInd).astype(np.int32)
    lateCode = codeChips.take(tcode2)

    # Define index into prompt code vector
    tcode = remCodePhase + codeInd
    promptCode = codeChips.take(np.ceil(tcode).astype(np.int32))

    # --- Generate local carrier ----------------------------------------------
    # Get the argument to sin/cos functions
    trigarg = sampleInd * carrPhaseStep + remCarrPhase
    # Compute the signal used to mix the collected data to baseband
    carrsig = np.exp(-1j*trigarg)


    # --- Do correlation ------------------------------------------------------
    # First mix to baseband
    basebandSignal = carrsig * rawSignal

    iBasebandSignal = basebandSignal.real
    qBasebandSignal = basebandSignal.imag
    # Now get early, late, and prompt values for each
    I_E = iBasebandSignal.dot(earlyCode)
    Q_E = qBasebandSignal.dot(earlyCode)
    I_P = iBasebandSignal.dot(promptCode)
    Q_P = qBasebandSignal.dot(promptCode)
    I_L = iBasebandSignal.dot(lateCode)
    Q_L = qBasebandSignal.dot(lateCode)

    return I_E, Q_E, I_P, Q_P, I_L, Q_L

#%% Python (Numpy) correlator for channel-parallel tracking
def corrPyParallelBPSK(settings, rawSignal, caCodeTable, remCarrPhase,
                           carrPhaseStep, remCodePhase, codePhaseStep,
                           startIdx, chSampSize, isDataRead):
    """Correlate all active tracking channels against one shared IF block.

    Args
    ----
        settings    - object
                    Receiver settings containing ``fileType`` and
                    ``dllCorrelatorSpacing``.
        rawSignal   - numpy.ndarray
                    Shared input IF signal.
        caCodeTable - numpy.ndarray
                    Guarded C/A-code table with one row per channel.
        remCarrPhase - numpy.ndarray
                     Initial carrier phases in radians.
        carrPhaseStep - numpy.ndarray
                      Carrier phase increments in radians per sample.
        remCodePhase - numpy.ndarray
                     Initial code phases in chips.
        codePhaseStep - numpy.ndarray
                      Code phase increments in chips per sample.
        startIdx    - numpy.ndarray
                    Zero-based logical sample offsets in the shared signal
                    block.
        chSampSize  - numpy.ndarray
                    Code-period sample count for each channel.
        isDataRead  - int
                    Shared-data refresh flag: ``1`` for new data and ``0``
                    to reuse the current block.

    Returns
    -------
        correValues - numpy.ndarray
                    Correlation matrix with shape ``(6, channelCnt)``;
                    rows are ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
    """
    # For complex data, split the interleaved buffer [I0 Q0 I1 Q1 ...] into
    # I and Q views of the shared input block.
    if settings.fileType == 2:
        rawSignalI = rawSignal[::2]
        rawSignalQ = rawSignal[1::2]

    # Number of active tracking channels in this millisecond.
    channelCnt = startIdx.size

    # Output order per channel: [IE QE IP QP IL QL]^T.
    correValues = np.zeros((6, channelCnt), dtype=np.float64)

    # Define early-late offset (in chips).
    earlyLateSpc = settings.dllCorrelatorSpacing

    for channelNr in range(channelCnt):
        # Find the size of the current code period in whole samples for the
        # current channel, and the start index of this channel block within the
        # 1-second rawSignal buffer.
        blksize = chSampSize[channelNr]
        codeStartIdx = startIdx[channelNr]

        # Get a vector with the local code for the current channel. The table
        # already contains the guard chips required by ceil(tcode).
        caCode = caCodeTable[channelNr, :]

        # Extract the current signal block to be processed by this channel.
        signalIndex = slice(codeStartIdx, codeStartIdx + blksize)
        if settings.fileType == 1:
            rawSignalBlock = rawSignal[signalIndex]
        else:
            rawSignalBlockI = rawSignalI[signalIndex]
            rawSignalBlockQ = rawSignalQ[signalIndex]
            rawSignalBlock = rawSignalBlockI + 1j * rawSignalBlockQ

        # --- Set up all the code phase tracking information ------------------
        sampleIndex = np.arange(blksize)
        codePhase = (remCodePhase[channelNr] +
                     codePhaseStep[channelNr] * sampleIndex)

        # Define index into early code vector.
        tcode2 = np.ceil(codePhase - earlyLateSpc).astype(np.int32)
        earlyCode = caCode[tcode2]

        # Define index into late code vector.
        tcode2 = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCode = caCode[tcode2]

        # Define index into prompt code vector. Python uses zero-based indices,
        # so MATLAB's trailing +1 index offset is not required.
        tcode2 = np.ceil(codePhase).astype(np.int32)
        promptCode = caCode[tcode2]

        # --- Generate the carrier frequency to mix the signal to baseband ----
        # carrPhaseStep is already the carrier phase step in radians per
        # sample, so the local carrier phase is formed directly with sample
        # indices instead of a time vector in seconds.
        trigarg = (carrPhaseStep[channelNr] * sampleIndex +
                   remCarrPhase[channelNr])
        carrsig = np.exp(-1j * trigarg)

        # --- Do correlation to generate the six standard accumulated values---
        # First mix to baseband.
        basebandSignal = carrsig * rawSignalBlock
        iBasebandSignal = basebandSignal.real
        qBasebandSignal = basebandSignal.imag

        # Now get early, late, and prompt values for each.
        I_E = np.sum(earlyCode * iBasebandSignal)
        Q_E = np.sum(earlyCode * qBasebandSignal)
        I_P = np.sum(promptCode * iBasebandSignal)
        Q_P = np.sum(promptCode * qBasebandSignal)
        I_L = np.sum(lateCode * iBasebandSignal)
        Q_L = np.sum(lateCode * qBasebandSignal)

        # Output order matches the SIMD / GPU correlator interface used by
        # trkChannelsParallel.
        correValues[:, channelNr] = (I_E, Q_E, I_P, Q_P, I_L, Q_L)

    return correValues

#%% GLONASS L1/L2 C/A code -----------------------------------------------
def generateCAcode(settings, K=0):
    """Generate one period of the common GLONASS L1/L2 C/A code.

    The same 511-chip ranging code is used by every GLONASS FDMA frequency
    channel. The channel number K is accepted to preserve the common
    pySoftGNSS code-generator interface.

    Args
    ----
        settings    - object
                    Receiver settings containing codeLength.
        K           - int
                    GLONASS frequency channel number; not used by the code
                    generator.

    Returns
    -------
        caCode      - numpy.ndarray
                    One bipolar GLONASS C/A-code period. Binary zero is
                    represented by +1 and binary one by -1.
    """
    #--- Generate the common GLONASS code ---------------------------------
    codeLength = settings.codeLength

    # Load the nine-stage shift register.
    reg = -np.ones(9, dtype=np.int8)
    code = np.empty(codeLength, dtype=np.int8)

    # Python uses zero-based indices, so MATLAB register stages 7, 5 and 9
    # are addressed below as 6, 4 and 8.
    for index in range(codeLength):
        # Output from the seventh register stage.
        code[index] = reg[6]
        # Feedback is the modulo-2 sum of stages 5 and 9; multiplication is
        # equivalent to XOR for the bipolar representation.
        save1 = reg[4] * reg[8]
        reg[1:] = reg[:-1]
        reg[0] = save1

    return code


#%% GLONASS L1/L2 C/A-code sampling ----------------------------------------
def codeSampling(settings, K, sampleLen):
    """Digitize the GLONASS C/A code at the receiver sampling frequency.

    Args
    ----
        settings    - object
                    Receiver settings containing the sampling frequency,
                    code frequency and code length.
        K           - int
                    GLONASS frequency channel number; accepted for the common
                    receiver interface.
        sampleLen   - int
                    Number of output samples.

    Returns
    -------
        caCodesTable - numpy.ndarray
                     Sampled common GLONASS C/A-code sequence.
    """
    # Generate the one-chip-per-element code.
    caCode = generateCAcode(settings, K)

    # Make zero-based indices at code-rate/sampling-rate increments and wrap
    # samples beyond the 511-chip code period.
    codeValueIndex = np.mod(
        np.floor(np.arange(sampleLen, dtype=np.float64) *
                 settings.codeFreqBasis/settings.samplingFreq),
        settings.codeLength).astype(np.int32)

    return caCode[codeValueIndex]
