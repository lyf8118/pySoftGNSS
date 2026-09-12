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
Generate, sample, and correlate Galileo E5aI/E5aQ codes.

"""

import ctypes
from functools import lru_cache

import numpy as np


#%% CPU SIMD correlator for channel-serial tracking
class CorrSIMDSerialQPSK:
    """Call the channel-serial AVX2 QPSK tracking correlator."""

    def __init__(self):
        """Load the SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDSerialQPSK.dll"
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
            int32Vector,      # E5aCodeTable
            ctypes.c_int,     # E5aCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, E5aCodeTable, remCarrPhase,
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
            E5aCodeTable - numpy.ndarray
                        Concatenated guarded int32 E5aI/E5aQ codes:
                        ``[E5aI(last, period, first),
                        E5aQ(last, period, first)]``.
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
                        E5aI correlations followed by E5aQ correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E5aCodeTable,
            E5aCodeTable.size // 2,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
        )
        if not corrPointer:
            raise RuntimeError("SIMD correlator initialization failed")

        # The C++ output is static and is overwritten by the next call.
        return np.ctypeslib.as_array(corrPointer, shape=(12,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the C++ engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-serial tracking
class CorrGPUSerialQPSK:
    """Call the channel-serial CUDA QPSK tracking correlator."""

    def __init__(self):
        """Load the CUDA DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrGPUSerialQPSK.dll"
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
            int8Vector,       # E5aCodeTable
            ctypes.c_int,     # E5aCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, E5aCodeTable, remCarrPhase,
                   carrPhaseStep, remCodePhase, codePhaseStep, PRN):
        """Correlate one channel and one code period on the GPU.

        Args
        ----
            settings    - object
                        Receiver settings containing ``fileType`` and
                        ``dllCorrelatorSpacing``.
            rawSignal   - numpy.ndarray
                        Input int16 IF samples. Complex samples are stored as
                        interleaved I/Q values.
            E5aCodeTable - numpy.ndarray
                        Concatenated guarded int8 E5aI/E5aQ codes:
                        ``[E5aI(last, period, first),
                        E5aQ(last, period, first)]``.
            remCarrPhase - float
                         Residual carrier phase in radians.
            carrPhaseStep - float
                          Carrier phase increment in radians per sample.
            remCodePhase - float
                         Residual code phase in chips.
            codePhaseStep - float
                          Code phase increment in chips per sample.
            PRN         - int
                        Current satellite PRN used for code caching.

        Returns
        -------
            correValues - numpy.ndarray
                        E5aI correlations followed by E5aQ correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E5aCodeTable,
            E5aCodeTable.size // 2,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            int(PRN),
        )
        if not corrPointer:
            raise RuntimeError("GPU serial correlator failed")
        return np.ctypeslib.as_array(corrPointer, shape=(12,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% CPU SIMD correlator for channel-parallel tracking
class CorrSIMDParallelQPSK:
    """Call the channel-parallel AVX2 QPSK tracking correlator."""

    def __init__(self):
        """Load the parallel SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDParallelQPSK.dll"
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
            int32Table,      # E5aCodeTable
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

    def corrEngine(self, settings, rawSignal, E5aCodeTable,
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
            E5aCodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded E5aI/E5aQ code branches for every channel.
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
                        Correlation matrix with shape ``(12, channelCnt)``.
                        The first six rows are E5aI and the final six rows are
                        E5aQ; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        codeLen = E5aCodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E5aCodeTable,
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

        corrValues = np.ctypeslib.as_array(corrPointer, shape=(12 * channelCnt,))
        return corrValues.reshape(channelCnt, 12).T.copy()

    def close(self):
        """Release the reusable buffers allocated by the SIMD engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-parallel tracking
class CorrGPUParallelQPSK:
    """Call the channel-parallel CUDA QPSK tracking correlator."""

    def __init__(self):
        """Load the parallel CUDA DLL and declare its ctypes interface.

        Returns
        -------
            None
        """
        dllPath = "../native_Correlators/corrGPUParallelQPSK.dll"
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
            int8Table,        # E5aCodeTable
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

    def corrEngine(self, settings, rawSignal, E5aCodeTable,
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
            E5aCodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded E5aI/E5aQ code branches for every channel.
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
                        Correlation matrix with shape ``(12, channelCnt)``.
                        The first six rows are E5aI and the final six rows are
                        E5aQ; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        codeLen = E5aCodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E5aCodeTable,
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

        corrValues = np.ctypeslib.as_array(corrPointer, shape=(12 * channelCnt,))
        return corrValues.reshape(channelCnt, 12).T.copy()

    def close(self):
        """Release buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% Python (Numpy) correlator for channel-serial tracking
def corrPySerialQPSK(settings, rawSignal, E5aCodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one E5a channel with the local E5aI and E5aQ codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one E5a code period. Complex samples are
                      represented by interleaved I and Q values.
        E5aCodeTable - numpy.ndarray
                      Guarded local-code vector containing the E5aI data and
                      E5aQ pilot codes in that order.
        remCarrPhase - float
                      Residual carrier phase in radians.
        carrPhaseStep - float
                       Carrier phase increment in radians per sample.
        remCodePhase - float
                      Residual primary-code phase in chips.
        codePhaseStep - float
                       Code-phase increment in chips per IF sample.
    Returns
    -------
        correValues   - ndarray
                      Twelve accumulated I/Q correlations. Each group of six
                      is ordered as ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for the
                      E5aI data and E5aQ pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = E5aCodeTable.size // 2
    E5aCodeD = E5aCodeTable[:codeLen]
    E5aCodeP = E5aCodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the E5aI data and E5aQ pilot outputs.
    correValues = np.zeros(12, dtype=np.float64)

    # For complex data, form one complex vector from interleaved I/Q samples.
    if settings.fileType == 2:
        rawSignal = rawSignal[::2] + 1j * rawSignal[1::2]

    blksize = rawSignal.size

    # --- Generate local code replicas -------------------------------------
    # Time index for each sampling point.
    sampleInd = np.arange(blksize)
    codeInd = sampleInd * codePhaseStep
    # Python uses zero-based indices, so MATLAB's trailing +1 local-code
    # offset is not required below.

    # Define index into early code vectors.
    tcode = remCodePhase - earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    earlyCodeD = E5aCodeD[tcode2]
    earlyCodeP = E5aCodeP[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = E5aCodeD[tcode2]
    lateCodeP = E5aCodeP[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = E5aCodeD[tcode2]
    promptCodeP = E5aCodeP[tcode2]

    # --- Generate the carrier frequency to mix the signal to baseband ------
    # Get the argument to sin/cos functions.
    trigarg = sampleInd * carrPhaseStep + remCarrPhase
    # Compute the signal used to mix the collected data to baseband.
    carrsig = np.exp(-1j * trigarg)

    # --- Do correlation to generate the standard accumulated values --------
    # First mix to baseband.
    basebandSignal = carrsig * rawSignal
    iBasebandSignal = basebandSignal.real
    qBasebandSignal = basebandSignal.imag

    # Get Early, Prompt, and Late accumulated values for the data channel.
    correValues[:6] = (earlyCodeD.dot(iBasebandSignal),
                       earlyCodeD.dot(qBasebandSignal),
                       promptCodeD.dot(iBasebandSignal),
                       promptCodeD.dot(qBasebandSignal),
                       lateCodeD.dot(iBasebandSignal),
                       lateCodeD.dot(qBasebandSignal))

    # Get Early, Prompt, and Late accumulated values for the pilot channel.
    correValues[6:12] = (earlyCodeP.dot(iBasebandSignal),
                         earlyCodeP.dot(qBasebandSignal),
                         promptCodeP.dot(iBasebandSignal),
                         promptCodeP.dot(qBasebandSignal),
                         lateCodeP.dot(iBasebandSignal),
                         lateCodeP.dot(qBasebandSignal))
    return correValues


#%% Python (Numpy) correlator for channel-parallel tracking
def corrPyParallelQPSK(settings, rawSignal, E5aCodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active E5a channels against one shared IF block.

    This function has the same calling interface as the SIMD and GPU
    correlators. Each output column is ordered as
    ``[I_E, Q_E, I_P, Q_P, I_L, Q_L, pilot_I_E, pilot_Q_E,
    pilot_I_P, pilot_Q_P, pilot_I_L, pilot_Q_L]``.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      Shared IF-signal block. Complex samples are stored as
                      interleaved I and Q values.
        E5aCodeTable - numpy.ndarray
                      One guarded E5aI/E5aQ local-code row for each active
                      channel.
        remCarrPhase - ndarray
                      Residual carrier phase for each channel, in radians.
        carrPhaseStep - ndarray
                       Carrier phase increment for each channel, in radians
                       per sample.
        remCodePhase - ndarray
                      Residual code phase for each channel.
        codePhaseStep - ndarray
                       Code-phase increment for each channel.
        startIdx     - ndarray
                      Start sample of each channel within ``rawSignal``.
        chSampSize   - ndarray
                      Number of samples in each channel's code period.
        isDataRead   - int
                      Shared-data refresh flag retained for interface parity.
    Returns
    -------
        correValues   - ndarray
                      A ``12 x channelCnt`` matrix. Each column contains
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for E5aI and E5aQ.
    """
    # For complex data, split the interleaved buffer [I0 Q0 I1 Q1 ...] into
    # I and Q views of the shared input block.
    if settings.fileType == 2:
        rawSignalI = rawSignal[::2]
        rawSignalQ = rawSignal[1::2]

    # Number of active tracking channels in this tracking epoch.
    channelCnt = startIdx.size
    # Output order per channel contains data and pilot values. Each group of
    # six is [I_E, Q_E, I_P, Q_P, I_L, Q_L].
    correValues = np.zeros((12, channelCnt), dtype=np.float64)
    # Define early-late offset in primary-code chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    for channelNr in range(channelCnt):
        # Find the size of the current code period and the start index of this
        # channel block within the shared rawSignal buffer.
        blksize = chSampSize[channelNr]
        codeStartIdx = startIdx[channelNr]

        # Get the two local-code branches for the current channel. The table
        # already contains the guard values required by ceil(tcode).
        E5aCode = E5aCodeTable[channelNr, :]
        codeLen = E5aCode.size // 2
        E5aCodeD = E5aCode[:codeLen]
        E5aCodeP = E5aCode[codeLen:]

        # Extract the current real or complex signal block.
        signalIndex = slice(codeStartIdx, codeStartIdx + blksize)
        if settings.fileType == 1:
            rawSignalBlock = rawSignal[signalIndex]
        else:
            rawSignalBlockI = rawSignalI[signalIndex]
            rawSignalBlockQ = rawSignalQ[signalIndex]
            rawSignalBlock = rawSignalBlockI + 1j * rawSignalBlockQ

        # --- Set up all the code-phase tracking information ----------------
        sampleIndex = np.arange(blksize)
        codePhase = (remCodePhase[channelNr]
                     + codePhaseStep[channelNr] * sampleIndex)

        # Define index into Early code vectors.
        tcode = np.ceil(codePhase - earlyLateSpc).astype(np.int32)
        earlyCodeD = E5aCodeD[tcode]
        earlyCodeP = E5aCodeP[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = E5aCodeD[tcode]
        lateCodeP = E5aCodeP[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = E5aCodeD[tcode]
        promptCodeP = E5aCodeP[tcode]

        # --- Generate the carrier frequency to mix the signal to baseband ---
        # carrPhaseStep is already the carrier phase step in radians per
        # sample, so form the local carrier directly from sample indices.
        trigarg = (carrPhaseStep[channelNr] * sampleIndex
                   + remCarrPhase[channelNr])
        carrsig = np.exp(-1j * trigarg)

        # --- Do correlation to generate the standard accumulated values -----
        # First mix to baseband.
        basebandSignal = carrsig * rawSignalBlock
        iBasebandSignal = basebandSignal.real
        qBasebandSignal = basebandSignal.imag

        # Get Early, Prompt, and Late values for the data channel.
        correValues[:6, channelNr] = (
            earlyCodeD.dot(iBasebandSignal),
            earlyCodeD.dot(qBasebandSignal),
            promptCodeD.dot(iBasebandSignal),
            promptCodeD.dot(qBasebandSignal),
            lateCodeD.dot(iBasebandSignal),
            lateCodeD.dot(qBasebandSignal))

        # Get Early, Prompt, and Late values for the pilot channel.
        correValues[6:12, channelNr] = (
            earlyCodeP.dot(iBasebandSignal),
            earlyCodeP.dot(qBasebandSignal),
            promptCodeP.dot(iBasebandSignal),
            promptCodeP.dot(qBasebandSignal),
            lateCodeP.dot(iBasebandSignal),
            lateCodeP.dot(qBasebandSignal))
    return correValues


#%% Galileo E5aI-code generation -------------------------------------------
@lru_cache(maxsize=100)
def generateDataCode(PRN, flag=1):
    """Generate a Galileo E5aI primary or tiered code.

    Args
    ----
        PRN         - int
                     PRN number of the sequence.
        flag        - int
                     1: primary code; 2: tiered code.

    Returns
    -------
        E5aI        - ndarray
                     E5aI primary/tiered code sequence in bipolar format.
    """
    # Start values for Register 1 are all ones.
    Register1 = np.ones(14, dtype=np.int8)

    # Initial-state table for Register 2 from the Galileo OS SIS ICD.
    e5ai_init = tuple(int(value, 8) for value in (
        "30305 14234 27213 20577 23312 33463 15614 12537 01527 30236 "
        "27344 07272 36377 17046 06434 15405 24252 11631 24776 00630 "
        "11560 17272 27445 31702 13012 14401 34727 22627 30623 27256 "
        "01520 14211 31465 22164 33516 02737 21316 35425 35633 24655 "
        "14054 27027 06604 31455 34465 25273 20763 31721 17312 13277"
    ).split())

    # Feedback-tap coefficients for Registers 1 and 2. Take the 14
    # significant bits from higher to lower register stages.
    taps1_coef = np.fromiter(
        (int(bit) for bit in f"{int('40503', 8):015b}"[:14]),
        dtype=np.int8)
    taps2_coef = np.fromiter(
        (int(bit) for bit in f"{int('50661', 8):015b}"[:14]),
        dtype=np.int8)

    # Start values for Register 2. The fixed-width binary conversion keeps
    # a possible leading zero that MATLAB dec2bin otherwise omits.
    PRN = int(PRN)
    # Python uses zero-based table indices; MATLAB selects row PRN.
    Register2 = np.fromiter(
        (int(bit) for bit in f"{e5ai_init[PRN-1]:014b}"), dtype=np.int8)

    # Generate the E5aI primary code.
    Pri_E5aI = np.empty(10230, dtype=np.int8)
    for ind in range(Pri_E5aI.size):
        RegOut1 = Register1*taps1_coef
        RegOut2 = Register2*taps2_coef
        Pri_E5aI[ind] = ((1-2*RegOut1[0])*(1-2*RegOut2[0]))
        # Exclusive-OR feedback. NumPy XOR on the binary taps is equivalent
        # to the bipolar product and reconversion used in MATLAB.
        feedback1 = np.bitwise_xor.reduce(RegOut1)
        feedback2 = np.bitwise_xor.reduce(RegOut2)
        Register1[:-1] = Register1[1:]
        Register2[:-1] = Register2[1:]
        Register1[-1] = feedback1
        Register2[-1] = feedback2

    #----- E5aI primary or tiered code generation -------------------------
    if flag == 1:
        return Pri_E5aI

    # Secondary code 842E9.
    SecondaryCode = np.array([
        -1,1,1,1, 1,-1,1,1, 1,1,-1,1, -1,-1,-1,1, -1,1,1,-1],
        dtype=np.int8)
    return np.tile(Pri_E5aI, SecondaryCode.size)*np.repeat(
        SecondaryCode, Pri_E5aI.size)


#%% Galileo E5aQ-code generation -------------------------------------------
@lru_cache(maxsize=100)
def generatePilotCode(PRN, flag=1):
    """Generate a Galileo E5aQ primary code.

    Args
    ----
        PRN         - int
                     PRN number of the sequence.
        flag        - int
                     1: primary code. The tracking chain uses this mode.

    Returns
    -------
        E5aQ        - ndarray
                     E5aQ primary code sequence in bipolar format.
    """
    # Start values for Register 1 are all ones.
    Register1 = np.ones(14, dtype=np.int8)

    # Initial-state table for Register 2 from the Galileo OS SIS ICD.
    e5aq_init = tuple(int(value, 8) for value in (
        "25652 05142 24723 31751 27366 24660 33655 27450 07626 01705 "
        "12717 32122 16075 16644 37556 02477 02265 06430 25046 12735 "
        "04262 11230 00037 06137 04312 20606 11162 22252 30533 24614 "
        "07767 32705 05052 27553 03711 02041 34775 05274 37356 16205 "
        "36270 06600 26773 17375 35267 36255 12044 26442 21621 25411"
    ).split())

    # Feedback-tap coefficients for Registers 1 and 2. Take the 14
    # significant bits from higher to lower register stages.
    taps1_coef = np.fromiter(
        (int(bit) for bit in f"{int('40503', 8):015b}"[:14]),
        dtype=np.int8)
    taps2_coef = np.fromiter(
        (int(bit) for bit in f"{int('50661', 8):015b}"[:14]),
        dtype=np.int8)

    # Start values for Register 2. The fixed-width binary conversion keeps
    # a possible leading zero that MATLAB dec2bin otherwise omits.
    PRN = int(PRN)
    # Python uses zero-based table indices; MATLAB selects row PRN.
    Register2 = np.fromiter(
        (int(bit) for bit in f"{e5aq_init[PRN-1]:014b}"), dtype=np.int8)

    # Generate the E5aQ primary code.
    Pri_E5aQ = np.empty(10230, dtype=np.int8)
    for ind in range(Pri_E5aQ.size):
        RegOut1 = Register1*taps1_coef
        RegOut2 = Register2*taps2_coef
        Pri_E5aQ[ind] = ((1-2*RegOut1[0])*(1-2*RegOut2[0]))
        # Exclusive-OR feedback. NumPy XOR on the binary taps is equivalent
        # to the bipolar product and reconversion used in MATLAB.
        feedback1 = np.bitwise_xor.reduce(RegOut1)
        feedback2 = np.bitwise_xor.reduce(RegOut2)
        Register1[:-1] = Register1[1:]
        Register2[:-1] = Register2[1:]
        Register1[-1] = feedback1
        Register2[-1] = feedback2

    #----- E5aQ primary or tiered code generation -------------------------
    if flag == 1:
        return Pri_E5aQ
    raise ValueError("E5aQ tiered-code generation is not used by this receiver")


#%% Galileo E5aI/E5aQ-code sampling ---------------------------------------
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one Galileo E5a primary-code component.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.
        sampleLen   - int
                    Number of output samples.
        component   - str
                    ``"data"`` for E5aI or ``"pilot"`` for E5aQ.

    Returns
    -------
        codeSamples - numpy.ndarray
                    Sampled E5a primary code for the specified PRN.
    """
    #--- Generate the selected E5a primary code ---------------------------
    if component == "data":
        code = generateDataCode(int(PRN))
    elif component == "pilot":
        code = generatePilotCode(int(PRN))
    else:
        raise ValueError('component must be "data" or "pilot"')

    #=== Digitizing ========================================================
    #--- Make the index array used to read E5a-code values ----------------
    codeValueIndex = np.ceil(
        np.arange(1, sampleLen + 1) * code.size /
        settings.samplesPerCode).astype(np.int32) - 1

    #--- Correct the last index due to numerical rounding -----------------
    codeValueIndex[-1] = code.size - 1

    #--- Make the digitized version of the E5a code -----------------------
    return code[codeValueIndex]
