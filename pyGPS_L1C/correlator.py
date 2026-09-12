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
Generate, sample, and correlate GPS L1C data/pilot codes.

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
            int32Vector,      # L1CCodeTable
            ctypes.c_int,     # codeLen for one QPSK branch
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
        ]
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L1CCodeTable, remCarrPhase,
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
            L1CCodeTable - numpy.ndarray
                        Concatenated guarded int32 L1C data/L1C pilot codes:
                        ``[L1C data(last, period, first),
                        L1C pilot(last, period, first)]``.
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
                        L1C data correlations followed by L1C pilot correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        # The native ABI receives the guarded length of one QPSK branch.
        codeLen = L1CCodeTable.size // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L1CCodeTable,
            codeLen,
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
            int8Vector,       # L1CCodeTable
            ctypes.c_int,     # codeLen for one QPSK branch
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, L1CCodeTable, remCarrPhase,
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
            L1CCodeTable - numpy.ndarray
                        Concatenated guarded int8 L1C data/L1C pilot codes:
                        ``[L1C data(last, period, first),
                        L1C pilot(last, period, first)]``.
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
                        L1C data correlations followed by L1C pilot correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        # The native ABI receives the guarded length of one QPSK branch.
        codeLen = L1CCodeTable.size // 2

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L1CCodeTable,
            codeLen,
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
            int32Table,      # L1CCodeTable
            ctypes.c_int,    # codeLen for one QPSK branch
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

    def corrEngine(self, settings, rawSignal, L1CCodeTable,
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
            L1CCodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded L1C data/L1C pilot code branches for every channel.
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
                        The first six rows are L1C data and the final six rows are
                        L1C pilot; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        # Each table row concatenates two equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = L1CCodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L1CCodeTable,
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
            int8Table,        # L1CCodeTable
            ctypes.c_int,     # codeLen for one QPSK branch
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

    def corrEngine(self, settings, rawSignal, L1CCodeTable,
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
            L1CCodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded L1C data/L1C pilot code branches for every channel.
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
                        The first six rows are L1C data and the final six rows are
                        L1C pilot; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        # Each table row concatenates two equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = L1CCodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            L1CCodeTable,
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
def corrPySerialQPSK(settings, rawSignal, L1CCodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one L1C channel with the local L1C data and L1C pilot codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one L1C code period. Complex samples are
                      represented by interleaved I and Q values.
        L1CCodeTable - numpy.ndarray
                      Guarded local-code vector containing the L1C data and
                      pilot codes in that order.
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
                      L1C data and pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = L1CCodeTable.size // 2
    L1CCodeD = L1CCodeTable[:codeLen]
    L1CCodeP = L1CCodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the L1C data and pilot outputs.
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
    earlyCodeD = L1CCodeD[tcode2]
    earlyCodeP = L1CCodeP[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = L1CCodeD[tcode2]
    lateCodeP = L1CCodeP[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = L1CCodeD[tcode2]
    promptCodeP = L1CCodeP[tcode2]

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
def corrPyParallelQPSK(settings, rawSignal, L1CCodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active L1C channels against one shared IF block.

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
        L1CCodeTable - numpy.ndarray
                      One guarded L1C data/L1C pilot local-code row for each active
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
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for L1C data and L1C pilot.
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
        L1CCode = L1CCodeTable[channelNr, :]
        codeLen = L1CCode.size // 2
        L1CCodeD = L1CCode[:codeLen]
        L1CCodeP = L1CCode[codeLen:]

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
        earlyCodeD = L1CCodeD[tcode]
        earlyCodeP = L1CCodeP[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = L1CCodeD[tcode]
        lateCodeP = L1CCodeP[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = L1CCodeD[tcode]
        promptCodeP = L1CCodeP[tcode]

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



#%% GPS L1C data-code generation
def generateDataBOC11(settings, PRN):
    """Generate the GPS L1C data-channel BOC(1,1) primary code.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    Satellite PRN number.

    Returns
    -------
        L1CData     - numpy.ndarray
                    L1C data-channel code after extension insertion and
                    BOC(1,1) subcarrier modulation.
    """
    # L1C data primary-code table. Column 1 is the Weil phase difference w;
    # column 2 is the 7-chip extension insertion point p.
    wp_data = np.array((
        [5097, 181], [5110, 359], [5079, 72], [4403, 1110],
        [4121, 1480], [5043, 5034], [5042, 4622], [5104, 1],
        [4940, 4547], [5035, 826], [4372, 6284], [5064, 4195],
        [5084, 368], [5048, 1], [4950, 4796], [5019, 523],
        [5076, 151], [3736, 713], [4993, 9850], [5060, 5734],
        [5061, 34], [5096, 6142], [4983, 190], [4783, 644],
        [4991, 467], [4815, 5384], [4443, 801], [4769, 594],
        [4879, 4450], [4894, 9437], [4985, 4307], [5056, 5906],
        [4921, 378], [5036, 9448], [4812, 9432], [4838, 5849],
        [4855, 5547], [4904, 9546], [4753, 9132], [4483, 403],
        [4942, 3766], [4813, 3], [4957, 684], [4618, 9711],
        [4669, 333], [4969, 6124], [5031, 10216], [5038, 4251],
        [4740, 9893], [4073, 9884], [4843, 4627], [4979, 4449],
        [4867, 9798], [4964, 985], [5025, 4272], [4579, 126],
        [4390, 10024], [4763, 434], [4612, 1029], [4784, 561],
        [3716, 289], [4703, 638], [4851, 4353]
    ), dtype=np.int32)

    # Compute the Legendre/Jacobi sequence used as the base sequence for
    # Weil-code generation.
    N = 10223
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        residue = pow(ind, (N - 1) // 2, N)
        legendre[ind] = 1 if residue == 1 else 0

    # Read the PRN-specific phase difference and insertion point.
    p = wp_data[PRN - 1, 1]
    w = wp_data[PRN - 1, 0]

    # Data-channel Weil code: W(n) = L(n) xor L(n+w).
    Primary = np.logical_xor(
        legendre, legendre[(np.arange(N) + w) % N])

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Primary = 1 - 2 * Primary.astype(np.int8)

    #%% Insert the 7-chip extension sequence and apply BOC(1,1).
    extendedSequence = np.array(
        [1, -1, -1, 1, -1, 1, 1], dtype=np.int8)
    L1CWithExtension = np.concatenate(
        (Primary[:p - 1], extendedSequence, Primary[p - 1:]))

    L1CData = np.empty(L1CWithExtension.size * 2, dtype=np.int8)
    L1CData[::2] = L1CWithExtension
    L1CData[1::2] = -L1CWithExtension
    return L1CData


#%% GPS L1C pilot-code generation
def generatePilotTMBOC61(settings, PRN):
    """Generate the GPS L1C pilot-channel TMBOC(6,1,4/33) code.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    Satellite PRN number.

    Returns
    -------
        L1CPilot    - numpy.ndarray
                    L1C pilot-channel code after TMBOC modulation.
    """
    # L1C pilot primary-code table. Column 1 is the Weil phase difference w;
    # column 2 is the 7-chip extension insertion point p.
    wp_pilot = np.array((
        [5111, 412], [5109, 161], [5108, 1], [5106, 303],
        [5103, 207], [5101, 4971], [5100, 4496], [5098, 5],
        [5095, 4557], [5094, 485], [5093, 253], [5091, 4676],
        [5090, 1], [5081, 66], [5080, 4485], [5069, 282],
        [5068, 193], [5054, 5211], [5044, 729], [5027, 4848],
        [5026, 982], [5014, 5955], [5004, 9805], [4980, 670],
        [4915, 464], [4909, 29], [4893, 429], [4885, 394],
        [4832, 616], [4824, 9457], [4591, 4429], [3706, 4771],
        [5092, 365], [4986, 9705], [4965, 9489], [4920, 4193],
        [4917, 9947], [4858, 824], [4847, 864], [4790, 347],
        [4770, 677], [4318, 6544], [4126, 6312], [3961, 9804],
        [3790, 278], [4911, 9461], [4881, 444], [4827, 4839],
        [4795, 4144], [4789, 9875], [4725, 197], [4675, 1156],
        [4539, 4674], [4535, 10035], [4458, 4504], [4197, 5],
        [4096, 9937], [3484, 430], [3481, 5], [3393, 355],
        [3175, 909], [2360, 1622], [1852, 6284]
    ), dtype=np.int32)

    # Compute the Legendre/Jacobi sequence used as the base sequence for
    # Weil-code generation.
    N = 10223
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        residue = pow(ind, (N - 1) // 2, N)
        legendre[ind] = 1 if residue == 1 else 0

    # Read the PRN-specific phase difference and insertion point.
    p = wp_pilot[PRN - 1, 1]
    w = wp_pilot[PRN - 1, 0]

    # Pilot-channel Weil code: W(n) = L(n) xor L(n+w).
    Primary = np.logical_xor(
        legendre, legendre[(np.arange(N) + w) % N])

    # Convert to bipolar format and insert the 7-chip extension sequence.
    Primary = 1 - 2 * Primary.astype(np.int8)
    extendedSequence = np.array(
        [1, -1, -1, 1, -1, 1, 1], dtype=np.int8)
    L1CWithExtension = np.concatenate(
        (Primary[:p - 1], extendedSequence, Primary[p - 1:]))

    #%% Apply TMBOC(6,1,4/33) modulation.
    # Use 12 samples per chip so BOC(1,1) and BOC(6,1) share one grid.
    samplesPerChip = 12
    numChips = L1CWithExtension.size
    L1CPilot = np.empty(numChips * samplesPerChip, dtype=np.int8)

    # BOC(1,1): first half-chip is +1 and second half-chip is -1.
    subcarrierBOC11 = np.r_[
        np.ones(6, dtype=np.int8), -np.ones(6, dtype=np.int8)]

    # BOC(6,1): six alternating +1/-1 subcarrier cycles per chip.
    subcarrierBOC61 = np.tile(np.array([1, -1], dtype=np.int8), 6)

    # ICD TMBOC pattern: 4 of every 33 chips use BOC(6,1).
    tm_indices = (0, 4, 6, 29)
    for jj in range(numChips):
        currentSub = (subcarrierBOC61 if jj % 33 in tm_indices
                      else subcarrierBOC11)
        L1CPilot[jj * samplesPerChip:(jj + 1) * samplesPerChip] = (
            L1CWithExtension[jj] * currentSub)
    return L1CPilot


#%% GPS L1C pilot secondary-code generation
def generate2ndCode(PRN):
    """Generate the GPS L1C pilot secondary code in bipolar format.

    This implementation supports PRNs 1 to 63, which use the single-LFSR
    form.

    Args
    ----
        PRN         - int
                    Satellite PRN number.

    Returns
    -------
        Secondary   - numpy.ndarray
                    1800-chip L1C secondary-code sequence.
    """
    #%% L1C secondary-code parameter table.
    # Each PRN maps to the S1 polynomial and initial state, both stored as
    # octal strings. Parameters are from IS-GPS-800 Table 3.2-3.
    data_map = {
        1: ("5111", "3266"),     2: ("5421", "2040"),     3: ("5501", "1527"),
         4: ("5403", "3307"),     5: ("6417", "3756"),     6: ("6141", "3026"),
         7: ("6351", "0562"),     8: ("6501", "0420"),     9: ("6205", "3415"),
         10: ("6235", "0337"),     11: ("7751", "0265"),     12: ("6623", "1230"),
         13: ("6733", "2204"),     14: ("7627", "1440"),     15: ("5667", "2412"),
         16: ("5051", "3516"),     17: ("7665", "2761"),     18: ("6325", "3750"),
         19: ("4365", "2701"),     20: ("4745", "1206"),     21: ("7633", "1544"),
         22: ("6747", "1774"),     23: ("4475", "0546"),     24: ("4225", "2213"),
         25: ("7063", "3707"),     26: ("4423", "2051"),     27: ("6651", "3650"),
         28: ("4161", "1777"),     29: ("7237", "3203"),     30: ("4473", "1762"),
         31: ("5477", "2100"),     32: ("6163", "0571"),     33: ("7223", "3710"),
         34: ("6323", "3535"),     35: ("7125", "3110"),     36: ("7035", "1426"),
         37: ("4341", "0255"),     38: ("4353", "0321"),     39: ("4107", "3124"),
         40: ("5735", "0572"),     41: ("6741", "1736"),     42: ("7071", "3306"),
         43: ("4563", "1307"),     44: ("5755", "3763"),     45: ("6127", "1604"),
         46: ("4671", "1021"),     47: ("4511", "2624"),     48: ("4533", "0406"),
         49: ("5357", "0114"),     50: ("5607", "0077"),     51: ("6673", "3477"),
         52: ("6153", "1000"),     53: ("7565", "3460"),     54: ("7107", "2607"),
         55: ("6211", "2057"),     56: ("4321", "3467"),     57: ("7201", "0706"),
         58: ("4451", "2032"),     59: ("5411", "1464"),     60: ("5141", "0520"),
         61: ("7041", "1766"),     62: ("6637", "3270"),     63: ("4577", "0341")
    }

    #%% Decode the LFSR parameters for the requested PRN.
    poly_oct, init_oct = data_map[PRN]

    # The polynomial is represented with 12 bits. Drop the two fixed end
    # terms to obtain the 10 taps aligned with state[1:11].
    full_poly_bits = np.array(
        [int(bit) for bit in f"{int(poly_oct, 8):012b}"], dtype=np.int8)
    taps = full_poly_bits[1:11]

    # Per the ICD note, drop the leading zero from the 12-bit initial state.
    full_init_bits = np.array(
        [int(bit) for bit in f"{int(init_oct, 8):012b}"], dtype=np.int8)
    state = full_init_bits[1:].copy()

    #%% Generate the 1800-chip secondary sequence.
    code_len = 1800
    binary_seq = np.empty(code_len, dtype=np.int8)
    for index in range(code_len):
        output_bit = state[0]
        binary_seq[index] = output_bit
        feedback_val = output_bit ^ (
            np.sum(state[1:11] & taps) % 2)
        state[:-1] = state[1:]
        state[-1] = feedback_val

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Secondary = 1 - 2 * binary_seq
    return Secondary


#%% GPS L1C-code sampling
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one GPS L1C spreading waveform.

    Args
    ----
        settings      - object
                      Receiver settings.
        PRN           - int
                      PRN number of the sequence.
        sampleLen     - int
                      Number of output samples.
        component     - str
                      L1C component: ``"data"`` or ``"pilot"``.

    Returns
    -------
        codeSamples   - numpy.ndarray
                      Sampled L1C spreading waveform.
    """
    #--- Generate the selected L1C spreading waveform ---------------------
    if component == "data":
        code = generateDataBOC11(settings, PRN)
        codeSubchips = 2
    elif component == "pilot":
        code = generatePilotTMBOC61(settings, PRN)
        codeSubchips = 12
    else:
        raise ValueError(f"Invalid L1C component: {component}")

    #--- Find time constants ----------------------------------------------
    ts = 1 / settings.samplingFreq
    tc = 1 / settings.codeFreqBasis / codeSubchips

    #=== Digitizing ========================================================
    #--- Make index array to read L1C code values -------------------------
    codeValueIndex = np.ceil(
        ts / tc * np.arange(1, sampleLen + 1)).astype(np.int32) - 1

    #--- Correct the first and last indices due to rounding issues --------
    codeValueIndex[0] = 0
    codeValueIndex[-1] = settings.codeLength * codeSubchips - 1

    #--- Make the digitized version of the L1C code -----------------------
    return code[codeValueIndex]
