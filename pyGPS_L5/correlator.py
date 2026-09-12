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
Generate, sample, and correlate GPS L5I/L5Q codes.

"""

import ctypes

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
            int32Vector,      # L5CCodeTable
            ctypes.c_int,     # L5CCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L5CCodeTable, remCarrPhase,
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
            L5CCodeTable - numpy.ndarray
                        Concatenated guarded int32 L5I/L5Q codes:
                        ``[L5I(last, period, first),
                        L5Q(last, period, first)]``.
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
                        L5I correlations followed by L5Q correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L5CCodeTable,
            L5CCodeTable.size // 2,
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
            int8Vector,       # L5CCodeTable
            ctypes.c_int,     # L5CCodeTable.size // 2
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L5CCodeTable, remCarrPhase,
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
            L5CCodeTable - numpy.ndarray
                        Concatenated guarded int8 L5I/L5Q codes:
                        ``[L5I(last, period, first),
                        L5Q(last, period, first)]``.
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
                        L5I correlations followed by L5Q correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L5CCodeTable,
            L5CCodeTable.size // 2,
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
            int32Table,      # L5CCodeTable
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

    def corrEngine(self, settings, rawSignal, L5CCodeTable,
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
            L5CCodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded L5I/L5Q code branches for every channel.
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
                        The first six rows are L5I and the final six rows are
                        L5Q; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        codeLen = L5CCodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L5CCodeTable,
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
            int8Table,        # L5CCodeTable
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

    def corrEngine(self, settings, rawSignal, L5CCodeTable,
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
            L5CCodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded L5I/L5Q code branches for every channel.
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
                        The first six rows are L5I and the final six rows are
                        L5Q; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        codeLen = L5CCodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L5CCodeTable,
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
def corrPySerialQPSK(settings, rawSignal, L5CCodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one L5 channel with the local L5I and L5Q codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one L5 code period. Complex samples are
                      represented by interleaved I and Q values.
        L5CCodeTable - numpy.ndarray
                      Guarded local-code vector containing the L5I data and
                      L5Q pilot codes in that order.
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
                      L5I data and L5Q pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = L5CCodeTable.size // 2
    L5CCodeD = L5CCodeTable[:codeLen]
    L5CCodeP = L5CCodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the L5I data and L5Q pilot outputs.
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
    earlyCodeD = L5CCodeD[tcode2]
    earlyCodeP = L5CCodeP[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = L5CCodeD[tcode2]
    lateCodeP = L5CCodeP[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = L5CCodeD[tcode2]
    promptCodeP = L5CCodeP[tcode2]

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
def corrPyParallelQPSK(settings, rawSignal, L5CCodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active L5 channels against one shared IF block.

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
        L5CCodeTable - numpy.ndarray
                      One guarded L5I/L5Q local-code row for each active
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
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for L5I and L5Q.
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
        L5CCode = L5CCodeTable[channelNr, :]
        codeLen = L5CCode.size // 2
        L5CCodeD = L5CCode[:codeLen]
        L5CCodeP = L5CCode[codeLen:]

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
        earlyCodeD = L5CCodeD[tcode]
        earlyCodeP = L5CCodeP[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = L5CCodeD[tcode]
        lateCodeP = L5CCodeP[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = L5CCodeD[tcode]
        promptCodeP = L5CCodeP[tcode]

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


# Common XA code shared by all L5I/L5Q PRNs.
_L5_XA_CODE = None


#%% GPS L5I-code generation -----------------------------------------------
def generateL5Icode(settings, PRN):
    """Generate one GPS L5I code.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.

    Returns
    -------
        L5Icode     - numpy.ndarray
                    Desired L5I code sequence in bipolar chip form.
    """
    # --- Code length --------------------------------------------------------
    CodeLength = settings.codeLength

    # --- Generate XA codes -------------------------------------------------
    global _L5_XA_CODE
    if _L5_XA_CODE is None:
        # Initial state of the XA register: 1 for 0 and -1 for 1, so
        # exclusive OR can be implemented by multiplication.
        xa_reg = np.ones(13, dtype=np.int8) * -1

        # XA-code output.
        XA = np.zeros(CodeLength, dtype=np.int8)

        # The XA register is reset to all ones; here -1 represents binary one.
        reset_state = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1],
            dtype=np.int8)

        for ind in range(CodeLength):
            XA[ind] = xa_reg[-1]
            if np.array_equal(xa_reg, reset_state):
                xa_reg.fill(-1)
            else:
                # Exclusive-OR operation for feedback.
                feedback = (xa_reg[8] * xa_reg[9] *
                            xa_reg[11] * xa_reg[12])
                # Shift the register to the right by one element.
                xa_reg[1:] = xa_reg[:-1]
                xa_reg[0] = feedback

        _L5_XA_CODE = XA

    XA = _L5_XA_CODE

    # --- Generate XBI codes ------------------------------------------------
    # Initial-state table from pages 5--7 and pages 29--33 of IS-GPS-705D.
    l5i_init = np.array([
        266, 365, 804, 1138, 1509, 1559, 1756, 2084,
        2170, 2303, 2527, 2687, 2930, 3471, 3940, 4132,
        4332, 4924, 5343, 5443, 5641, 5816, 5898, 5918,
        5955, 6243, 6345, 6477, 6518, 6875, 7168, 7187,
        7329, 7577, 7720, 7777, 8057, 5358, 3550, 3412,
        819, 4608, 3698, 962, 3001, 4441, 4937, 3717,
        4730, 7291, 2279, 7613, 5723, 7030, 1475, 2593,
        2904, 2056, 2757, 3756, 6205, 5053, 6437, 7789,
        2311, 7432, 5155, 1593, 5841, 5014, 1545, 3016,
        4875, 2119, 229, 7634, 1406, 4506, 1819, 7580,
        5446, 6053, 7958, 5267, 2956, 3544, 1277, 2996,
        1758, 3360, 2718, 3754, 7440, 2781, 6756, 7314,
        208, 5252, 696, 527, 1399, 5879, 6868, 217,
        7681, 3788, 1337, 2424, 4243, 5686, 1955, 4791,
        492, 1518, 6566, 5349, 506, 113, 1953, 2797,
        934, 3023, 3632, 1330, 4909, 4867, 1183, 3990,
        6217, 1224, 1733, 2319, 3928, 2380, 841, 5049,
        7027, 1197, 7208, 8000, 152, 6762, 3745, 4723,
        5502, 4796, 123, 8142, 5091, 7875, 330, 5272,
        4912, 374, 2045, 6616, 6321, 7605, 2570, 2419,
        1234, 1922, 4317, 110, 825, 958, 1089, 7813,
        6058, 7703, 6702, 1714, 6371, 2281, 1986, 6282,
        3201, 3760, 1056, 6233, 1150, 2823, 6250, 645,
        2401, 1639, 2946, 7091, 923, 7045, 6493, 1706,
        5836, 926, 6086, 950, 5905, 3240, 6675, 3197,
        1555, 3589, 4555, 5671, 6948, 4664, 2086, 5950,
        5521, 1515
    ], dtype=np.int32)

    assert 1 <= PRN <= 210, 'Invalid PRN code:'+str(PRN)

    # Initial state of the XBI register: 1 for 0 and -1 for 1, so exclusive
    # OR can be implemented by multiplication.
    xbi_reg = np.ones(13, dtype=np.int8) * -1

    # XBI-code output.
    XBI = np.zeros(CodeLength, dtype=np.int8)

    # XBI-code advance in chips for the start of the XBI code.
    resetPos = l5i_init[PRN - 1]

    # XBI initial state.
    for _ in range(resetPos):
        # Exclusive-OR operation for feedback.
        feedback = (xbi_reg[0] * xbi_reg[2] * xbi_reg[3] * xbi_reg[5] *
                    xbi_reg[6] * xbi_reg[7] * xbi_reg[11] * xbi_reg[12])
        # Shift the register to the right by one element.
        xbi_reg[1:] = xbi_reg[:-1]
        xbi_reg[0] = feedback

    # XBI output.
    for ind in range(CodeLength):
        XBI[ind] = xbi_reg[-1]
        # Exclusive-OR operation for feedback.
        feedback = (xbi_reg[0] * xbi_reg[2] * xbi_reg[3] * xbi_reg[5] *
                    xbi_reg[6] * xbi_reg[7] * xbi_reg[11] * xbi_reg[12])
        # Shift the register to the right by one element.
        xbi_reg[1:] = xbi_reg[:-1]
        xbi_reg[0] = feedback

    # --- L5I PRN-code output ----------------------------------------------
    L5Icode = XBI * XA
    return L5Icode


#%% GPS L5Q-code generation -----------------------------------------------
def generateL5Qcode(settings, PRN):
    """Generate one GPS L5Q code.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.

    Returns
    -------
        L5Qcode     - numpy.ndarray
                    Desired L5Q code sequence in bipolar chip form.
    """
    # --- Code length --------------------------------------------------------
    CodeLength = settings.codeLength

    # --- Generate XA codes -------------------------------------------------
    global _L5_XA_CODE
    if _L5_XA_CODE is None:
        # Initial state of the XA register: 1 for 0 and -1 for 1, so
        # exclusive OR can be implemented by multiplication.
        xa_reg = np.ones(13, dtype=np.int8) * -1

        # XA-code output.
        XA = np.zeros(CodeLength, dtype=np.int8)

        # The XA register is reset to all ones; here -1 represents binary one.
        reset_state = np.array(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1],
            dtype=np.int8)

        for ind in range(CodeLength):
            XA[ind] = xa_reg[-1]
            if np.array_equal(xa_reg, reset_state):
                xa_reg.fill(-1)
            else:
                # Exclusive-OR operation for feedback.
                feedback = (xa_reg[8] * xa_reg[9] *
                            xa_reg[11] * xa_reg[12])
                # Shift the register to the right by one element.
                xa_reg[1:] = xa_reg[:-1]
                xa_reg[0] = feedback

        _L5_XA_CODE = XA

    XA = _L5_XA_CODE

    # --- Generate XBQ codes ------------------------------------------------
    # Initial-state table from pages 5--7 and pages 29--33 of IS-GPS-705D.
    l5q_init = np.array([
        1701, 323, 5292, 2020, 5429, 7136, 1041, 5947,
        4315, 148, 535, 1939, 5206, 5910, 3595, 5135,
        6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961,
        7106, 5299, 4660, 276, 4389, 3783, 1591, 1601,
        749, 1387, 1661, 3210, 708, 4226, 5604, 6375,
        3056, 1772, 3662, 4401, 5218, 2838, 6913, 1685,
        1194, 6963, 5001, 6694, 991, 7489, 2441, 639,
        2097, 2498, 6470, 2399, 242, 3768, 1186, 5246,
        4259, 5907, 3870, 3262, 7387, 3069, 2999, 7993,
        7849, 4157, 5031, 5986, 4833, 5739, 7846, 898,
        2022, 7446, 6404, 155, 7862, 7795, 6121, 4840,
        6585, 429, 6020, 200, 1664, 1499, 7298, 1305,
        7323, 7544, 4438, 2485, 3387, 7319, 1853, 5781,
        1874, 7555, 2132, 6441, 6722, 1192, 2588, 2188,
        297, 1540, 4138, 5231, 4789, 659, 871, 6837,
        1393, 7383, 611, 4920, 5416, 1611, 2474, 118,
        1382, 1092, 7950, 7223, 1769, 4721, 1252, 5147,
        2165, 7897, 4054, 3498, 6571, 2858, 8126, 7017,
        1901, 181, 1114, 5195, 7479, 4186, 3904, 7128,
        1396, 4513, 5967, 2580, 2575, 7961, 2598, 4508,
        2090, 3685, 7748, 684, 913, 5558, 2894, 5858,
        6432, 3813, 3573, 7523, 5280, 3376, 7424, 2918,
        5793, 1747, 7079, 2921, 2490, 4119, 3373, 977,
        681, 4273, 5419, 5626, 1266, 5804, 2414, 6444,
        4757, 427, 5452, 5182, 6606, 6531, 4268, 3115,
        6835, 862, 4856, 2765, 37, 1943, 7977, 2512,
        4451, 4071
    ], dtype=np.int32)

    assert 1 <= PRN <= 210, 'Invalid PRN code:'+str(PRN)

    # Initial state of the XBQ register: 1 for 0 and -1 for 1, so exclusive
    # OR can be implemented by multiplication.
    xbq_reg = np.ones(13, dtype=np.int8) * -1

    # XBQ-code output.
    XBQ = np.zeros(CodeLength, dtype=np.int8)

    # XBQ-code advance in chips for the start of the XBQ code.
    resetPos = l5q_init[PRN - 1]

    # XBQ initial state.
    for _ in range(resetPos):
        # Exclusive-OR operation for feedback.
        feedback = (xbq_reg[0] * xbq_reg[2] * xbq_reg[3] * xbq_reg[5] *
                    xbq_reg[6] * xbq_reg[7] * xbq_reg[11] * xbq_reg[12])
        # Shift the register to the right by one element.
        xbq_reg[1:] = xbq_reg[:-1]
        xbq_reg[0] = feedback

    # XBQ output.
    for ind in range(CodeLength):
        XBQ[ind] = xbq_reg[-1]
        # Exclusive-OR operation for feedback.
        feedback = (xbq_reg[0] * xbq_reg[2] * xbq_reg[3] * xbq_reg[5] *
                    xbq_reg[6] * xbq_reg[7] * xbq_reg[11] * xbq_reg[12])
        # Shift the register to the right by one element.
        xbq_reg[1:] = xbq_reg[:-1]
        xbq_reg[0] = feedback

    # --- L5Q PRN-code output ----------------------------------------------
    L5Qcode = XBQ * XA
    return L5Qcode


#%% GPS L5-code sampling --------------------------------------------------
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one GPS L5 primary-code component.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.
        sampleLen   - int
                    Number of output samples.
        component   - str
                    ``"data"`` for L5I or ``"pilot"`` for L5Q.

    Returns
    -------
        codeSamples - numpy.ndarray
                    Sampled L5 primary code for the specified PRN.
    """
    #--- Generate the selected L5 primary code ----------------------------
    if component == "data":
        code = generateL5Icode(settings, PRN)
    elif component == "pilot":
        code = generateL5Qcode(settings, PRN)
    else:
        raise ValueError('component must be "data" or "pilot"')

    #--- Find time constants ----------------------------------------------
    ts = 1 / settings.samplingFreq   # Sampling period in seconds.
    tc = 1 / settings.codeFreqBasis # L5 chip period in seconds.

    #=== Digitizing ========================================================
    #--- Make the index array used to read L5-code values -----------------
    # The index-array length depends on the sampling frequency.
    codeValueIndex = np.ceil(
        (ts * np.arange(1, sampleLen + 1)) / tc).astype(np.int32) - 1

    #--- Correct the last index due to numerical rounding -----------------
    codeValueIndex[-1] = settings.codeLength - 1

    #--- Make the digitized version of the L5 code ------------------------
    return code[codeValueIndex]
