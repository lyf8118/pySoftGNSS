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
Generate, sample, and correlate Galileo E6B/E6C codes.

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
            int32Vector,      # E6CodeTable
            ctypes.c_int,     # E6CodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, E6CodeTable, remCarrPhase,
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
            E6CodeTable - numpy.ndarray
                        Concatenated guarded int32 E6B/E6C codes:
                        ``[E6B(last, period, first),
                        E6C(last, period, first)]``.
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
                        E6B correlations followed by E6C correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E6CodeTable,
            E6CodeTable.size // 2,
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
            int8Vector,       # E6CodeTable
            ctypes.c_int,     # E6CodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, E6CodeTable, remCarrPhase,
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
            E6CodeTable - numpy.ndarray
                        Concatenated guarded int8 E6B/E6C codes:
                        ``[E6B(last, period, first),
                        E6C(last, period, first)]``.
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
                        E6B correlations followed by E6C correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E6CodeTable,
            E6CodeTable.size // 2,
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
            int32Table,      # E6CodeTable
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

    def corrEngine(self, settings, rawSignal, E6CodeTable,
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
            E6CodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded E6B/E6C code branches for every channel.
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
                        The first six rows are E6B and the final six rows are
                        E6C; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        codeLen = E6CodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E6CodeTable,
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
            int8Table,        # E6CodeTable
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

    def corrEngine(self, settings, rawSignal, E6CodeTable,
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
            E6CodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded E6B/E6C code branches for every channel.
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
                        The first six rows are E6B and the final six rows are
                        E6C; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        codeLen = E6CodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            E6CodeTable,
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
def corrPySerialQPSK(settings, rawSignal, E6CodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one E6 channel with the local E6B and E6C codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one E6 code period. Complex samples are
                      represented by interleaved I and Q values.
        E6CodeTable - numpy.ndarray
                      Guarded local-code vector containing the E6B data and
                      E6C pilot codes in that order.
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
                      E6B data and E6C pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = E6CodeTable.size // 2
    E6CodeD = E6CodeTable[:codeLen]
    E6CodeP = E6CodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the E6B data and E6C pilot outputs.
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
    earlyCodeD = E6CodeD[tcode2]
    earlyCodeP = E6CodeP[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = E6CodeD[tcode2]
    lateCodeP = E6CodeP[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = E6CodeD[tcode2]
    promptCodeP = E6CodeP[tcode2]

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
def corrPyParallelQPSK(settings, rawSignal, E6CodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active E6 channels against one shared IF block.

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
        E6CodeTable - numpy.ndarray
                      One guarded E6B/E6C local-code row for each active
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
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for E6B and E6C.
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
        E6Code = E6CodeTable[channelNr, :]
        codeLen = E6Code.size // 2
        E6CodeD = E6Code[:codeLen]
        E6CodeP = E6Code[codeLen:]

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
        earlyCodeD = E6CodeD[tcode]
        earlyCodeP = E6CodeP[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = E6CodeD[tcode]
        lateCodeP = E6CodeP[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = E6CodeD[tcode]
        promptCodeP = E6CodeP[tcode]

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


#%% Galileo E6B-code generation --------------------------------------------
@lru_cache(maxsize=50)
def generateDataCode(PRN, flag=1):
    """Generate a Galileo E6B primary code.

    Args
    ----
        PRN         - int
                     PRN number of the sequence.
        flag        - int
                     Kept for the common generator interface; E6B is primary.

    Returns
    -------
        E6Bcode     - ndarray
                     E6B primary-code sequence in bipolar format.
    """
    import re

    if not hasattr(generateDataCode, "E6B_code"):
        # Read the hexadecimal memory-code table once.
        with open("./result_Cache/E6B_codes.dat", encoding="utf-8") as fid:
            codeText = fid.read()
        codeText = codeText[codeText.index("[")+1:codeText.rindex("]")]
        E6B_code = []
        for codeRow in codeText.split(";"):
            hexParts = re.findall(r"'([0-9A-F]+)'", codeRow)
            if hexParts:
                E6B_code.append("".join(hexParts))
        generateDataCode.E6B_code = tuple(E6B_code)

    # Convert the requested code from hexadecimal to bipolar chips.
    # Python uses zero-based rows; MATLAB selects the PRN-th code row.
    code = generateDataCode.E6B_code[int(PRN)-1]
    m = np.fromiter(
        ((int(value, 16) >> shift) & 1
         for value in code for shift in (3,2,1,0)), dtype=np.int8
        ).reshape(-1, 4)
    m1 = m.reshape(-1)
    E6Bcode = 1-2*m1[:5115]
    return E6Bcode


#%% Galileo E6C-code generation --------------------------------------------
@lru_cache(maxsize=50)
def generatePilotCode(PRN, flag=1):
    """Generate a Galileo E6C primary code.

    Args
    ----
        PRN         - int
                     PRN number of the sequence.
        flag        - int
                     1: primary code. The tracking chain uses this mode.

    Returns
    -------
        E6C         - ndarray
                     E6C primary-code sequence in bipolar format.
    """
    import re

    if flag != 1:
        raise ValueError("E6C tiered-code generation is not used by tracking")

    if not hasattr(generatePilotCode, "E6C_code"):
        # Read the hexadecimal memory-code table once.
        with open("./result_Cache/E6C_codes.dat", encoding="utf-8") as fid:
            codeText = fid.read()
        codeText = codeText[codeText.index("[")+1:codeText.rindex("]")]
        E6C_code = []
        for codeRow in codeText.split(";"):
            hexParts = re.findall(r"'([0-9A-F]+)'", codeRow)
            if hexParts:
                E6C_code.append("".join(hexParts))
        generatePilotCode.E6C_code = tuple(E6C_code)

    # Convert the requested code from hexadecimal to bipolar chips.
    # Python uses zero-based rows; MATLAB selects the PRN-th code row.
    code = generatePilotCode.E6C_code[int(PRN)-1]
    m = np.fromiter(
        ((int(value, 16) >> shift) & 1
         for value in code for shift in (3,2,1,0)), dtype=np.int8
        ).reshape(-1, 4)
    m1 = m.reshape(-1)
    E6Ccode = 1-2*m1[:5115]
    E6C = E6Ccode
    return E6C


#%% Galileo E6B/E6C-code sampling ------------------------------------------
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one Galileo E6 spreading code.

    Args
    ----
        settings      - object
                      Receiver settings.
        PRN           - int
                      PRN number of the sequence.
        sampleLen     - int
                      Number of output samples.
        component     - str
                      E6 component: ``"data"`` or ``"pilot"``.

    Returns
    -------
        codeSamples   - numpy.ndarray
                      Sampled E6 spreading code.
    """
    #--- Generate the selected E6 spreading code ---------------------------
    if component == "data":
        code = generateDataCode(int(PRN))
    elif component == "pilot":
        code = generatePilotCode(int(PRN))
    else:
        raise ValueError(f"Invalid E6 component: {component}")

    #=== Digitizing =========================================================
    #--- Make the index array used to read E6 code values -------------------
    codeValueIndex = np.ceil(
        np.arange(1, sampleLen + 1) * code.size /
        settings.samplesPerCode).astype(np.int32) - 1

    #--- Correct the last index due to rounding -----------------------------
    codeValueIndex[-1] = code.size - 1

    #--- Make the digitized version of the E6 code --------------------------
    return code[codeValueIndex]
