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
Generate, sample, and correlate BDS-3 B2a data and pilot codes.

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
            int32Vector,      # B2aCodeTable
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

    def corrEngine(self, settings, rawSignal, B2aCodeTable, remCarrPhase,
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
            B2aCodeTable - numpy.ndarray
                        Concatenated guarded int32 B2a data/B2a pilot codes:
                        ``[B2a data(last, period, first),
                        B2a pilot(last, period, first)]``.
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
                        B2a data correlations followed by B2a pilot correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        # The native ABI receives the guarded length of one QPSK branch.
        codeLen = B2aCodeTable.size // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B2aCodeTable,
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
            int8Vector,       # B2aCodeTable
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

    def corrEngine(self, settings, rawSignal, B2aCodeTable, remCarrPhase,
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
            B2aCodeTable - numpy.ndarray
                        Concatenated guarded int8 B2a data/B2a pilot codes:
                        ``[B2a data(last, period, first),
                        B2a pilot(last, period, first)]``.
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
                        B2a data correlations followed by B2a pilot correlations; each
                        branch is ``[I_E, Q_E, I_P, Q_P, I_L, Q_L]``.
        """

        # The native ABI receives the guarded length of one QPSK branch.
        codeLen = B2aCodeTable.size // 2

        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B2aCodeTable,
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
            int32Table,      # B2aCodeTable
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

    def corrEngine(self, settings, rawSignal, B2aCodeTable,
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
            B2aCodeTable - numpy.ndarray
                        C-contiguous int32 table containing concatenated
                        guarded B2a data/B2a pilot code branches for every channel.
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
                        The first six rows are B2a data and the final six rows are
                        B2a pilot; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """

        channelCnt = startIdx.size
        # Each table row concatenates two equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = B2aCodeTable.shape[1] // 2

        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B2aCodeTable,
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
            int8Table,        # B2aCodeTable
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

    def corrEngine(self, settings, rawSignal, B2aCodeTable,
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
            B2aCodeTable - numpy.ndarray
                        C-contiguous int8 table containing concatenated
                        guarded B2a data/B2a pilot code branches for every channel.
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
                        The first six rows are B2a data and the final six rows are
                        B2a pilot; each branch is ordered
                        ``I_E, Q_E, I_P, Q_P, I_L, Q_L``.
        """
        channelCnt = startIdx.size
        # Each table row concatenates two equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = B2aCodeTable.shape[1] // 2
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B2aCodeTable,
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
def corrPySerialQPSK(settings, rawSignal, B2aCodeTable, remCarrPhase,
                     carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one B2a channel with the local B2a data and B2a pilot codes.

    Args
    ----
        settings      - object
                      Receiver settings containing ``fileType`` and
                      ``dllCorrelatorSpacing``.
        rawSignal     - ndarray
                      IF samples for one B2a code period. Complex samples are
                      represented by interleaved I and Q values.
        B2aCodeTable - numpy.ndarray
                      Guarded local-code vector containing the B2a data and
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
                      B2a data and pilot branches, respectively.
    """
    # Split the shared QPSK table into its two equal guarded branches.
    codeLen = B2aCodeTable.size // 2
    B2aCodeD = B2aCodeTable[:codeLen]
    B2aCodeP = B2aCodeTable[codeLen:]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate the B2a data and pilot outputs.
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
    earlyCodeD = B2aCodeD[tcode2]
    earlyCodeP = B2aCodeP[tcode2]

    # Define index into late code vectors.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCodeD = B2aCodeD[tcode2]
    lateCodeP = B2aCodeP[tcode2]

    # Define index into prompt code vectors.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCodeD = B2aCodeD[tcode2]
    promptCodeP = B2aCodeP[tcode2]

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
def corrPyParallelQPSK(settings, rawSignal, B2aCodeTable, remCarrPhase,
                       carrPhaseStep, remCodePhase, codePhaseStep,
                       startIdx, chSampSize, isDataRead):
    """Correlate all active B2a channels against one shared IF block.

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
        B2aCodeTable - numpy.ndarray
                      One guarded B2a data/B2a pilot local-code row for each active
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
                      ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for B2a data and B2a pilot.
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
        B2aCode = B2aCodeTable[channelNr, :]
        codeLen = B2aCode.size // 2
        B2aCodeD = B2aCode[:codeLen]
        B2aCodeP = B2aCode[codeLen:]

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
        earlyCodeD = B2aCodeD[tcode]
        earlyCodeP = B2aCodeP[tcode]

        # Define index into Late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = B2aCodeD[tcode]
        lateCodeP = B2aCodeP[tcode]

        # Define index into Prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = B2aCodeD[tcode]
        promptCodeP = B2aCodeP[tcode]

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



# Common register-1 sequences shared by all PRNs.  The data and pilot
# components use different generator polynomials, so they keep separate
# caches indexed by the requested primary-code length.
_B2A_DATA_REG1_CACHE = {}
_B2A_PILOT_REG1_CACHE = {}


#%% BDS-3 B2a data-code generation
def generateB2aDataCode(settings, PRN):
    """Generate one of the BDS-3 satellite B2a data codes.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.

    Returns
    -------
        B2acode - numpy.ndarray
                    Vector containing the desired B2a data code
                    sequence in chips.
    """
    #--- Initial parameters for data code generation -----------------
    # Initial values of register No. 2.
    B2aData_reg2_ini = (
        0b1000000100101, 0b1000000110100, 0b1000010101101,
        0b1000101001111, 0b1000101010101, 0b1000110101110,
        0b1000111101110, 0b1000111111011, 0b1001100101001,
        0b1001111011010, 0b1010000110101, 0b1010001000100,
        0b1010001010101, 0b1010001011011, 0b1010001011100,
        0b1010010100011, 0b1010011110111, 0b1010100000001,
        0b1010100111110, 0b1010110101011, 0b1010110110001,
        0b1011001010011, 0b1011001100010, 0b1011010011000,
        0b1011010110110, 0b1011011110010, 0b1011011111111,
        0b1011100010010, 0b1011100111100, 0b1011110100001,
        0b1011111001000, 0b1011111010100, 0b1011111101011,
        0b1011111110011, 0b1100001010001, 0b1100010010100,
        0b1100010110111, 0b1100100010001, 0b1100100011001,
        0b1100110101011, 0b1100110110001, 0b1100111010010,
        0b1101001010101, 0b1101001110100, 0b1101011001011,
        0b1101101010111, 0b1110000110100, 0b1110010000011,
        0b1110010001011, 0b1110010100011, 0b1110010101000,
        0b1110100111011, 0b1110110010111, 0b1111001001000,
        0b1111010010100, 0b1111010011001, 0b1111011011010,
        0b1111011111000, 0b1111011111111, 0b1111110110101,
        0b0010000000010, 0b1101111110101, 0b0001111010010
    )

    # Code length.
    CodeLength = settings.codeLength

    #--- Generate XA codes ------------------------------------------------
    # Generate the PRN-independent register-1 output once per code length.
    # Initial state: 1 for 0 and -1 for 1, so exclusive OR can be
    # implemented by multiplication.
    register1Code = _B2A_DATA_REG1_CACHE.get(CodeLength)
    if register1Code is None:
        register1 = np.ones(13, dtype=np.int8) * -1
        register1Code = np.empty(CodeLength, dtype=np.int8)
        reset_index = 8190

        for ind in range(CodeLength):
            register1Code[ind] = register1[-1]
            feedback1 = (register1[0] * register1[4] *
                         register1[10] * register1[12])
            register1[1:] = register1[:-1]
            register1[0] = feedback1
            if ind + 1 == reset_index:
                register1.fill(-1)

        _B2A_DATA_REG1_CACHE[CodeLength] = register1Code

    # Initial state of the PRN-dependent register No. 2.
    initialState = np.array(
        [int(bit) for bit in f"{B2aData_reg2_ini[PRN - 1]:013b}"],
        dtype=np.int8)
    register2 = 1 - 2 * initialState

    # XA-code output.
    B2acode = np.zeros(CodeLength, dtype=np.int8)

    #--- Generate the B2a data channel codes -------------------------
    for ind in range(CodeLength):
        B2acode[ind] = register1Code[ind] * register2[-1]

        # Exclusive-OR operation for feedback.
        feedback2 = (register2[2] * register2[4] * register2[8] *
                     register2[10] * register2[11] * register2[12])

        # Shift the register right by one element.
        register2[1:] = register2[:-1]
        register2[0] = feedback2
    return B2acode


#%% BDS-3 B2a pilot-code generation
def generateB2aPilotCode(settings, PRN):
    """Generate one of the BDS-3 satellite B2a pilot codes.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.

    Returns
    -------
        B2aPilotcode - numpy.ndarray
                    Vector containing the desired B2a pilot code
                    sequence in chips.
    """
    #--- Initial parameters for pilot code generation -----------------
    # Initial values of register No. 2.
    B2aData_reg2_ini = (
        0b1000000100101, 0b1000000110100, 0b1000010101101,
        0b1000101001111, 0b1000101010101, 0b1000110101110,
        0b1000111101110, 0b1000111111011, 0b1001100101001,
        0b1001111011010, 0b1010000110101, 0b1010001000100,
        0b1010001010101, 0b1010001011011, 0b1010001011100,
        0b1010010100011, 0b1010011110111, 0b1010100000001,
        0b1010100111110, 0b1010110101011, 0b1010110110001,
        0b1011001010011, 0b1011001100010, 0b1011010011000,
        0b1011010110110, 0b1011011110010, 0b1011011111111,
        0b1011100010010, 0b1011100111100, 0b1011110100001,
        0b1011111001000, 0b1011111010100, 0b1011111101011,
        0b1011111110011, 0b1100001010001, 0b1100010010100,
        0b1100010110111, 0b1100100010001, 0b1100100011001,
        0b1100110101011, 0b1100110110001, 0b1100111010010,
        0b1101001010101, 0b1101001110100, 0b1101011001011,
        0b1101101010111, 0b1110000110100, 0b1110010000011,
        0b1110010001011, 0b1110010100011, 0b1110010101000,
        0b1110100111011, 0b1110110010111, 0b1111001001000,
        0b1111010010100, 0b1111010011001, 0b1111011011010,
        0b1111011111000, 0b1111011111111, 0b1111110110101,
        0b1010010000110, 0b0010111111000, 0b0001101010101
    )

    # Code length.
    CodeLength = settings.codeLength

    #--- Generate XA codes ------------------------------------------------
    # Generate the PRN-independent register-1 output once per code length.
    # Initial state: 1 for 0 and -1 for 1, so exclusive OR can be
    # implemented by multiplication.
    register1Code = _B2A_PILOT_REG1_CACHE.get(CodeLength)
    if register1Code is None:
        register1 = np.ones(13, dtype=np.int8) * -1
        register1Code = np.empty(CodeLength, dtype=np.int8)
        reset_index = 8190

        for ind in range(CodeLength):
            register1Code[ind] = register1[-1]
            feedback1 = (register1[2] * register1[5] *
                         register1[6] * register1[12])
            register1[1:] = register1[:-1]
            register1[0] = feedback1
            if ind + 1 == reset_index:
                register1.fill(-1)

        _B2A_PILOT_REG1_CACHE[CodeLength] = register1Code

    # Initial state of the PRN-dependent register No. 2.
    initialState = np.array(
        [int(bit) for bit in f"{B2aData_reg2_ini[PRN - 1]:013b}"],
        dtype=np.int8)
    register2 = 1 - 2 * initialState

    # XA-code output.
    B2aPilotcode = np.zeros(CodeLength, dtype=np.int8)

    #--- Generate the B2a pilot channel codes -------------------------
    for ind in range(CodeLength):
        B2aPilotcode[ind] = register1Code[ind] * register2[-1]

        # Exclusive-OR operation for feedback.
        feedback2 = (register2[0] * register2[4] * register2[6] *
                     register2[7] * register2[11] * register2[12])

        # Shift the register right by one element.
        register2[1:] = register2[:-1]
        register2[0] = feedback2
    return B2aPilotcode


#%% BDS-3 B2a-code sampling
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one BDS-3 B2a primary-code component.

    Args
    ----
        settings    - object
                    Receiver settings.
        PRN         - int
                    PRN number of the sequence.
        sampleLen   - int
                    Number of output samples.
        component   - str
                    ``"data"`` or ``"pilot"``.

    Returns
    -------
        codeSamples - numpy.ndarray
                    Sampled B2a primary code for the specified PRN.
    """
    #--- Generate the selected B2a primary code ---------------------------
    if component == "data":
        code = generateB2aDataCode(settings, PRN)
    elif component == "pilot":
        code = generateB2aPilotCode(settings, PRN)
    else:
        raise ValueError('component must be "data" or "pilot"')

    #--- Find time constants ----------------------------------------------
    ts = 1 / settings.samplingFreq   # Sampling period in seconds.
    tc = 1 / settings.codeFreqBasis # B2a chip period in seconds.

    #=== Digitizing ========================================================
    #--- Make the index array used to read B2a-code values ----------------
    # The length of the index array depends on the sampling frequency.
    codeValueIndex = np.ceil(
        ts / tc * np.arange(1, sampleLen + 1)).astype(np.int32) - 1

    #--- Correct the last index due to number-rounding issues -------------
    codeValueIndex[-1] = settings.codeLength - 1

    #--- Make the digitized version of the B2a code -----------------------
    return code[codeValueIndex]
