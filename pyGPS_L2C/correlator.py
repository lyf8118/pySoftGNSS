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
Generate, sample, and correlate GPS L2C CM/CL codes.

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
            int32Vector,      # L2CCodeTable
            ctypes.c_int,     # L2CCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L2CCodeTable, remCarrPhase,
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
            L2CCodeTable - numpy.ndarray
                        Concatenated guarded int32 CM/CL codes:
                        ``[CM(last, period, first),
                        CL(last, period, first)]``.
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
                        CM correlations followed by CL correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L2CCodeTable,
            L2CCodeTable.size // 2,
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
            int8Vector,       # L2CCodeTable
            ctypes.c_int,     # L2CCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L2CCodeTable, remCarrPhase,
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
            L2CCodeTable - numpy.ndarray
                        Concatenated guarded int8 CM/CL codes:
                        ``[CM(last, period, first),
                        CL(last, period, first)]``.
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
                        CM correlations followed by CL correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L2CCodeTable,
            L2CCodeTable.size // 2,
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
            int32Table,      # L2CCodeTable
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

    def corrEngine(self, settings, rawSignal, L2CCodeTable,
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
            L2CCodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded CM/CL code branches for every channel.
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
                        The first six rows are CM and the final six rows are
                        CL; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        codeLen = L2CCodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L2CCodeTable,
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
            int8Table,        # L2CCodeTable
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

    def corrEngine(self, settings, rawSignal, L2CCodeTable,
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
            L2CCodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded CM/CL code branches for every channel.
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
                        The first six rows are CM and the final six rows are
                        CL; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        codeLen = L2CCodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L2CCodeTable,
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
def corrPySerialQPSK(settings, rawSignal, L2CCodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one L2C channel with the local CM and CL codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one L2C code period. Complex samples are
                      represented by interleaved I and Q values.
        L2CCodeTable - numpy.ndarray
                      Guarded local-code vector containing the CM data and
                      CL pilot codes in that order.
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
                      CM data and CL pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = L2CCodeTable.size // 2
    CMCode = L2CCodeTable[:codeLen]
    CLCode = L2CCodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the CM data and CL pilot outputs.
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
    earlyCodeD = CMCode[tcode2]
    earlyCodeP = CLCode[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = CMCode[tcode2]
    lateCodeP = CLCode[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = CMCode[tcode2]
    promptCodeP = CLCode[tcode2]

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
def corrPyParallelQPSK(settings, rawSignal, L2CCodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active L2C channels against one shared IF block.

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
        L2CCodeTable - numpy.ndarray
                      One guarded CM/CL local-code row for each active
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
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for CM and CL.
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
        L2CCode = L2CCodeTable[channelNr, :]
        codeLen = L2CCode.size // 2
        CMCode = L2CCode[:codeLen]
        CLCode = L2CCode[codeLen:]

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
        earlyCodeD = CMCode[tcode]
        earlyCodeP = CLCode[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = CMCode[tcode]
        lateCodeP = CLCode[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = CMCode[tcode]
        promptCodeP = CLCode[tcode]

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


#%% GPS L2C primary-code generation helper
@lru_cache(maxsize=128)
def _generateL2CPrimaryCode(code_init, CodeLength):
    """Generate one un-interleaved L2C primary-code sequence.

    The cache key contains only the integer initial state and code length, so
    receiver settings objects do not need to be hashable.  A public generator
    always copies/interleaves this cached primary code into its return array.
    """
    # Register positions toggled by the output chip.  Array position zero is
    # the most-significant bit of the 27-stage integer register.
    RegPos = (3, 6, 8, 11, 14, 16, 18, 21, 22, 23, 24)
    feedbackMask = sum(1 << (26 - position) for position in RegPos)

    # Generate the code with a scalar 27-stage binary register.  Multiplying
    # a bipolar register stage by the output chip is equivalent to XOR in the
    # binary representation.  This avoids allocating a 27-element array for
    # every chip through np.roll.
    reg = int(code_init)
    primaryCode = np.empty(CodeLength, dtype=np.int8)
    for index in range(CodeLength):
        outputBit = reg & 1
        primaryCode[index] = 1 - 2 * outputBit
        reg = (reg >> 1) | (outputBit << 26)
        if outputBit:
            reg ^= feedbackMask

    # The cached array is internal and read-only.  Public generators return a
    # newly allocated interleaved array, preserving their previous semantics.
    primaryCode.setflags(write=False)
    return primaryCode


#%% GPS L2C CM-code generation
def generateCMcode(settings, PRN):
    """Generate one GPS L2C CM return-to-zero code.

    Args
    ----
        settings    - object
                    Receiver settings containing ``codeLength``.
        PRN         - int
                    PRN number of the sequence: 1--63 or 159--210.

    Returns
    -------
        CMcode      - numpy.ndarray
                    Desired L2C CM-code sequence in bipolar return-to-zero
                    chip form.
    """
    #--- Initial-state table from pages 9--11 and 62--63 of IS-GPS-200H ----
    # PRNs 64--158 do not exist for the L2 CM-/L2 CL-code.
    l2cm_init = (
        "742417664", "756014035", "002747144", "066265724", "601403471", "703232733",
        "124510070", "617316361", "047541621", "733031046", "713512145", "024437606",
        "021264003", "230655351", "001314400", "222021506", "540264026", "205521705",
        "064022144", "120161274", "044023533", "724744327", "045743577", "741201660",
        "700274134", "010247261", "713433445", "737324162", "311627434", "710452007",
        "722462133", "050172213", "500653703", "755077436", "136717361", "756675453",
        "435506112", "771353753", "226107701", "022025110", "402466344", "752566114",
        "702011164", "041216771", "047457275", "266333164", "713167356", "060546335",
        "355173035", "617201036", "157465571", "767360553", "023127030", "431343777",
        "747317317", "045706125", "002744276", "060036467", "217744147", "603340174",
        "326616775", "063240065", "111460621", "604055104", "157065232", "013305707",
        "603552017", "230461355", "603653437", "652346475", "743107103", "401521277",
        "167335110", "014013575", "362051132", "617753265", "216363634", "755561123",
        "365304033", "625025543", "054420334", "415473671", "662364360", "373446602",
        "417564100", "000526452", "226631300", "113752074", "706134401", "041352546",
        "664630154", "276524255", "714720530", "714051771", "044526647", "207164322",
        "262120161", "204244652", "202133131", "714351204", "657127260", "130567507",
        "670517677", "607275514", "045413633", "212645405", "613700455", "706202440",
        "705056276", "020373522", "746013617", "132720621", "434015513", "566721727",
        "140633660"
    )

    #--- Select the initial state for the specified PRN ---------------------
    PRN = int(PRN)
    if 1 <= PRN <= 63:
        shiftPos = PRN-1
    elif 159 <= PRN <= 210:
        shiftPos = PRN-96
    else:
        raise ValueError("GPS L2C PRNs 64 through 158 do not exist.")

    #--- Code length and initial state --------------------------------------
    CodeLength = settings.codeLength//2
    code_init = int(l2cm_init[shiftPos], 8)

    #--- Generate the CM code -----------------------------------------------
    CMcode = _generateL2CPrimaryCode(code_init, CodeLength)

    #--- Form the return-to-zero CM code ------------------------------------
    zeros = np.zeros(CodeLength, dtype=np.int8)
    return np.column_stack((CMcode, zeros)).ravel()


#%% GPS L2C CL-code generation
def generateCLcode(settings, PRN):
    """Generate one GPS L2C CL return-to-zero code.

    Args
    ----
        settings    - object
                    Receiver settings containing ``CLCodeLength``.
        PRN         - int
                    PRN number of the sequence: 1--63 or 159--210.

    Returns
    -------
        CLcode      - numpy.ndarray
                    Desired L2C CL-code sequence in bipolar return-to-zero
                    chip form.
    """
    #--- Initial-state table from pages 9--11 and 62--63 of IS-GPS-200H ----
    # PRNs 64--158 do not exist for the L2 CM-/L2 CL-code.
    l2cl_init = (
        "624145772", "506610362", "220360016", "710406104", "001143345", "053023326",
        "652521276", "206124777", "015563374", "561522076", "023163525", "117776450",
        "606516355", "003037343", "046515565", "671511621", "605402220", "002576207",
        "525163451", "266527765", "006760703", "501474556", "743747443", "615534726",
        "763621420", "720727474", "700521043", "222567263", "132765304", "746332245",
        "102300466", "255231716", "437661701", "717047302", "222614207", "561123307",
        "240713073", "101232630", "132525726", "315216367", "377046065", "655351360",
        "435776513", "744242321", "024346717", "562646415", "731455342", "723352536",
        "000013134", "011566642", "475432222", "463506741", "617127534", "026050332",
        "733774235", "751477772", "417631550", "052247456", "560404163", "417751005",
        "004302173", "715005045", "001154457", "605253024", "063314262", "066073422",
        "737276117", "737243704", "067557532", "227354537", "704765502", "044746712",
        "720535263", "733541364", "270060042", "737176640", "133776704", "005645427",
        "704321074", "137740372", "056375464", "704374004", "216320123", "011322115",
        "761050112", "725304036", "721320336", "443462103", "510466244", "745522652",
        "373417061", "225526762", "047614504", "034730440", "453073141", "533654510",
        "377016461", "235525312", "507056307", "221720061", "520470122", "603764120",
        "145604016", "051237167", "033326347", "534627074", "645230164", "000171400",
        "022715417", "135471311", "137422057", "714426456", "640724672", "501254540",
        "513322453"
    )

    #--- Select the initial state for the specified PRN ---------------------
    PRN = int(PRN)
    if 1 <= PRN <= 63:
        shiftPos = PRN-1
    elif 159 <= PRN <= 210:
        shiftPos = PRN-96
    else:
        raise ValueError("GPS L2C PRNs 64 through 158 do not exist.")

    #--- Code length and initial state --------------------------------------
    CodeLength = settings.CLCodeLength//2
    code_init = int(l2cl_init[shiftPos], 8)

    #--- Generate the CL code -----------------------------------------------
    CLcode = _generateL2CPrimaryCode(code_init, CodeLength)

    #--- Form the return-to-zero CL code ------------------------------------
    zeros = np.zeros(CodeLength, dtype=np.int8)
    return np.column_stack((zeros, CLcode)).ravel()


#%% GPS L2C-code sampling
def codeSampling(settings, PRN, sampleLen, component):
    """Digitize a GPS L2C code at the receiver sampling frequency.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.
        sampleLen   - int
                    Number of output samples.
        component   - str
                    ``"CM"`` for the data code or ``"CL"`` for the pilot
                    code.

    Returns
    -------
        codeSamples - numpy.ndarray
                    Sampled L2C code for the specified PRN and component.
    """
    #--- Generate the requested L2C code -----------------------------------
    if component == "CM":
        code = generateCMcode(settings, PRN)
        codeLength = settings.codeLength
    elif component == "CL":
        code = generateCLcode(settings, PRN)
        codeLength = settings.CLCodeLength
    else:
        raise ValueError('component must be "CM" or "CL".')

    #--- Find time constants -----------------------------------------------
    ts = 1/settings.samplingFreq   # Sampling period in seconds.
    tc = 1/settings.codeFreqBasis  # L2C chip period in seconds.

    #=== Digitizing =========================================================

    #--- Make the index array used to read L2C-code values -----------------
    # The index-array length depends on the sampling frequency: one entry
    # for every requested output sample.
    codeValueIndex = np.ceil(
        (ts*np.arange(sampleLen))/tc).astype(np.int32)-1

    #--- Correct the first and last indices for numerical rounding ---------
    codeValueIndex[0] = 0
    samplesPerComponent = round(
        settings.samplingFreq/(settings.codeFreqBasis/codeLength))
    if sampleLen == samplesPerComponent:
        codeValueIndex[-1] = codeLength-1

    #--- Make the digitized version of the requested L2C code --------------
    # The upsampled code selects one chip for every sampling instant.
    return code[codeValueIndex]
