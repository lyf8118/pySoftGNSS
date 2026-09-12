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
Generate, sample, and correlate BDS-3 B1C codes.

"""

import ctypes
import numpy as np


#%% CPU SIMD correlator for channel-serial tracking
class CorrSIMDSerialQMBOC:
    """Call the channel-serial AVX2 QMBOC tracking correlator."""

    def __init__(self):
        """Load the SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDSerialQMBOC.dll"
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
            int32Vector,      # B1CCodeTable
            ctypes.c_int,     # codeLen for one QMBOC branch
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double]  # codePhaseStep
        
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # This is the clearup function.
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, B1CCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep):
        """Correlate one channel and one B1C code period.

        Args
        ----
            settings       - object
                           Receiver settings containing ``fileType``,
                           ``rShiftBits`` and ``dllCorrelatorSpacing``.
            rawSignal      - numpy.ndarray
                           Input int16 IF samples. Complex samples are stored
                           as interleaved I/Q values.
            B1CCodeTable   - numpy.ndarray
                           Three guarded int32 QMBOC code branches stored as
                           data BOC(1,1), pilot BOC(1,1), and pilot BOC(6,1).
            remCarrPhase   - float
                           Residual carrier phase in radians.
            carrPhaseStep  - float
                           Carrier phase increment in radians per sample.
            remCodePhase   - float
                           Residual code phase in BOC(6,1) samples.
            codePhaseStep  - float
                           Code phase increment per IF sample.

        Returns
        -------
            correValues    - numpy.ndarray
                           Eighteen data- and pilot-channel correlations. Each
                           group of six is ordered as
                           ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for data BOC(1,1),
                           pilot BOC(1,1), and pilot BOC(6,1), respectively.
        """
        # The native ABI receives the guarded length of one QMBOC branch.
        codeLen = B1CCodeTable.size // 3
        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B1CCodeTable,
            codeLen,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep)
        
        if not corrPointer:
            raise RuntimeError("SIMD QMBOC correlator initialization failed")

        # The C++ output is static and is overwritten by the next call.
        return np.ctypeslib.as_array(corrPointer, shape=(18,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the C++ engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-serial tracking
class CorrGPUSerialQMBOC:
    """Call the channel-serial CUDA QMBOC tracking correlator."""

    def __init__(self):
        """Load the CUDA DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrGPUSerialQMBOC.dll"
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
            int8Vector,       # B1CCodeTable
            ctypes.c_int,     # codeLen for one QMBOC branch
            ctypes.c_double,  # remCarrPhase
            ctypes.c_double,  # carrPhaseStep
            ctypes.c_double,  # remCodePhase
            ctypes.c_double,  # codePhaseStep
            ctypes.c_int]     # PRN
        
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, B1CCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep,
                   PRN):
        """Correlate one channel and one B1C code period on the GPU.

        Args
        ----
            settings       - object
                           Receiver settings containing ``fileType`` and
                           ``dllCorrelatorSpacing``.
            rawSignal      - numpy.ndarray
                           Input int16 IF samples. Complex samples are stored
                           as interleaved I/Q values.
            B1CCodeTable   - numpy.ndarray
                           Three guarded int8 QMBOC code branches.
            remCarrPhase   - float
                           Residual carrier phase in radians.
            carrPhaseStep  - float
                           Carrier phase increment in radians per sample.
            remCodePhase   - float
                           Residual code phase in BOC(6,1) samples.
            codePhaseStep  - float
                           Code phase increment per IF sample.
            PRN            - int
                           Current satellite PRN used for code caching.

        Returns
        -------
            correValues    - numpy.ndarray
                           Eighteen data- and pilot-channel correlations. Each
                           group of six is ordered as
                           ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for data BOC(1,1),
                           pilot BOC(1,1), and pilot BOC(6,1), respectively.
        """
        # The native ABI receives the guarded length of one QMBOC branch.
        codeLen = B1CCodeTable.size // 3
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B1CCodeTable,
            codeLen,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            int(PRN))
        
        if not corrPointer:
            raise RuntimeError("GPU serial QMBOC correlator failed")
        return np.ctypeslib.as_array(corrPointer, shape=(18,)).copy()

    def close(self):
        """Release the reusable buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% CPU SIMD correlator for channel-parallel tracking
class CorrSIMDParallelQMBOC:
    """Call the channel-parallel AVX2 QMBOC tracking correlator."""

    def __init__(self):
        """Load the parallel SIMD DLL and declare its ctypes interface.
        """
        dllPath = "../native_Correlators/corrSIMDParallelQMBOC.dll"
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
            int32Table,      # B1CCodeTable
            ctypes.c_int,    # codeLen for one QMBOC branch
            ctypes.c_int,    # channelCnt
            float64Vector,   # remCarrPhase
            float64Vector,   # carrPhaseStep
            float64Vector,   # remCodePhase
            float64Vector,   # codePhaseStep
            int32Vector,     # startIdx
            int32Vector,     # chSampSize
            ctypes.c_int]    # isDataRead
        
        self._simdLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # The clearup function.
        self._simdLibrary.corrEngineFree.argtypes = []
        self._simdLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, B1CCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep,
                   startIdx, chSampSize, isDataRead):
        """Correlate all active B1C channels against one shared IF block.

        Args
        ----
            settings       - object
                           Receiver settings containing ``fileType``,
                           ``rShiftBits`` and ``dllCorrelatorSpacing``.
            rawSignal      - numpy.ndarray
                           Shared input int16 IF signal.
            B1CCodeTable   - numpy.ndarray
                           C-contiguous int32 table containing three guarded
                           QMBOC branches per channel.
            remCarrPhase   - numpy.ndarray
                           Initial carrier phases in radians.
            carrPhaseStep  - numpy.ndarray
                           Carrier phase increments in radians per sample.
            remCodePhase   - numpy.ndarray
                           Initial code phases in BOC(6,1) samples.
            codePhaseStep  - numpy.ndarray
                           Code phase increments per IF sample.
            startIdx       - numpy.ndarray
                           Zero-based logical sample offsets in the shared
                           signal block.
            chSampSize     - numpy.ndarray
                           Code-period sample count for each channel.
            isDataRead     - int
                           Shared-data refresh flag: ``1`` for new data and
                           ``0`` to reuse the current block.

        Returns
        -------
            correValues    - numpy.ndarray
                           Correlation matrix with shape
                           ``(18, channelCnt)``. Each group of six rows is
                           ordered as ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for
                           data BOC(1,1), pilot BOC(1,1), and pilot BOC(6,1).
        """
        channelCnt = startIdx.size
        # Each table row concatenates three equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = B1CCodeTable.shape[1] // 3
        corrPointer = self._simdLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.rShiftBits,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B1CCodeTable,
            codeLen,
            channelCnt,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            startIdx,
            chSampSize,
            isDataRead)
        
        if not corrPointer:
            raise RuntimeError("SIMD parallel QMBOC correlator failed")
        corrValues = np.ctypeslib.as_array(corrPointer, shape=(18 * channelCnt,))
        return corrValues.reshape(channelCnt, 18).T.copy()

    def close(self):
        """Release the reusable buffers allocated by the SIMD engine.
        """
        self._simdLibrary.corrEngineFree()


#%% GPU correlator for channel-parallel tracking
class CorrGPUParallelQMBOC:
    """Call the channel-parallel CUDA QMBOC tracking correlator."""

    def __init__(self):
        """Load the parallel CUDA DLL and declare its ctypes interface.

        Returns
        -------
            None
        """
        dllPath = "../native_Correlators/corrGPUParallelQMBOC.dll"
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
            int8Table,        # B1CCodeTable
            ctypes.c_int,     # codeLen for one QMBOC branch
            ctypes.c_int,     # channelCnt
            float64Vector,    # remCarrPhase
            float64Vector,    # carrPhaseStep
            float64Vector,    # remCodePhase
            float64Vector,    # codePhaseStep
            int32Vector,      # startIdx
            int32Vector,      # chSampSize
            ctypes.c_int]     # isDataRead
        
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        # The clearup function.
        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, B1CCodeTable,
                   remCarrPhase, carrPhaseStep, remCodePhase, codePhaseStep,
                   startIdx, chSampSize, isDataRead):
        """Correlate all active B1C channels against one shared IF block.

        Args
        ----
            settings       - object
                           Receiver settings containing ``fileType`` and
                           ``dllCorrelatorSpacing``.
            rawSignal      - numpy.ndarray
                           Shared input int16 IF signal.
            B1CCodeTable   - numpy.ndarray
                           C-contiguous int8 table containing three guarded
                           QMBOC branches per channel.
            remCarrPhase   - numpy.ndarray
                           Residual carrier phases in radians.
            carrPhaseStep  - numpy.ndarray
                           Carrier phase increments in radians per sample.
            remCodePhase   - numpy.ndarray
                           Residual code phases in BOC(6,1) samples.
            codePhaseStep  - numpy.ndarray
                           Code phase increments per IF sample.
            startIdx       - numpy.ndarray
                           Zero-based logical sample offsets in ``rawSignal``.
            chSampSize     - numpy.ndarray
                           Logical sample count for each channel.
            isDataRead     - int
                           ``1`` uploads new data; ``0`` reuses the data.

        Returns
        -------
            correValues    - numpy.ndarray
                           Correlation matrix with shape
                           ``(18, channelCnt)``. Each group of six rows is
                           ordered as ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for
                           data BOC(1,1), pilot BOC(1,1), and pilot BOC(6,1).
        """
        channelCnt = startIdx.size
        # Each table row concatenates three equal-length guarded branches; the
        # native ABI receives the length of one branch.
        codeLen = B1CCodeTable.shape[1] // 3
        corrPointer = self._gpuLibrary.corrEngine(
            settings.fileType,
            rawSignal.size,
            settings.dllCorrelatorSpacing,
            rawSignal,
            B1CCodeTable,
            codeLen,
            channelCnt,
            remCarrPhase,
            carrPhaseStep,
            remCodePhase,
            codePhaseStep,
            startIdx,
            chSampSize,
            isDataRead )
        
        if not corrPointer:
            raise RuntimeError("GPU parallel QMBOC correlator failed")
        corrValues = np.ctypeslib.as_array(corrPointer, shape=(18 * channelCnt,))
        return corrValues.reshape(channelCnt, 18).T.copy()

    def close(self):
        """Release the reusable buffers allocated by the CUDA engine.
        """
        self._gpuLibrary.corrEngineFree()


#%% Python (Numpy) correlator for channel-serial tracking
def corrPySerialB1C(settings, rawSignal, B1CCodeTable, remCarrPhase,
                    carrPhaseStep, remCodePhase, codePhaseStep):
    """Correlate one BDS-3 B1C channel with the local QMBOC waveforms.

    Args
    ----
        settings       - object
                       Receiver settings containing ``fileType``,
                       ``dllCorrelatorSpacing`` and ``fullBandEn``.
        rawSignal      - ndarray
                       IF samples for one B1C code period. Complex samples
                       are represented by interleaved I and Q values.
        B1CCodeTable   - ndarray
                       Guarded local-code vector containing the data
                       BOC(1,1), pilot BOC(1,1), and pilot BOC(6,1)
                       waveforms in that order.
        remCarrPhase   - float
                       Residual carrier phase in radians.
        carrPhaseStep  - float
                       Carrier phase increment in radians per sample.
        remCodePhase   - float
                       Residual code phase in BOC(6,1) samples.
        codePhaseStep  - float
                       Code-phase increment per IF sample.
    Returns
    -------
        correValues    - ndarray
                       Eighteen accumulated I/Q correlations. Each group of
                       six is ordered as ``I_E, Q_E, I_P, Q_P, I_L, Q_L``
                       for data BOC(1,1), pilot BOC(1,1), and pilot
                       BOC(6,1), respectively.
    """
    # Split the shared QMBOC table into its three equal guarded branches.
    codeLen = B1CCodeTable.size // 3
    B1CDataCode = B1CCodeTable[:codeLen]
    B1CPilotCode = B1CCodeTable[codeLen : codeLen * 2]
    pilotBOC61 = B1CCodeTable[codeLen * 2 :]
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    # Allocate data BOC(1,1), pilot BOC(1,1), and pilot BOC(6,1) outputs.
    correValues = np.zeros(18, dtype=np.float64)

    # For complex data, form one complex vector from interleaved I/Q samples.
    if settings.fileType == 2:
        rawSignal = rawSignal[::2] + 1j * rawSignal[1::2]

    blksize = rawSignal.size

    # --- Generate local code replicas --------------------------------------
    # Time index for each sampling point.
    sampleInd = np.arange(blksize)
    codeInd = sampleInd * codePhaseStep
    # Python uses zero-based indices, so MATLAB's trailing +1 local-code
    # offset is not required below.

    # Define index into early code vector.
    tcode = remCodePhase - earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    earlyCode = B1CDataCode[tcode2]
    # Early local codes for pilot-channel signal tracking.
    pilotEarlyCode = B1CPilotCode[tcode2]
    p61_earlyCode = pilotBOC61[tcode2]

    # Define index into late code vector.
    tcode = remCodePhase + earlyLateSpc + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    lateCode = B1CDataCode[tcode2]
    # Late local codes for pilot-channel signal tracking.
    pilotLateCode = B1CPilotCode[tcode2]
    p61_lateCode = pilotBOC61[tcode2]

    # Define index into prompt code vector.
    tcode = remCodePhase + codeInd
    tcode2 = np.ceil(tcode).astype(np.int32)
    promptCode = B1CDataCode[tcode2]
    # Prompt local codes for pilot-channel signal tracking.
    pilotPromptCode = B1CPilotCode[tcode2]
    p61_promptCode = pilotBOC61[tcode2]

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
    correValues[:6] = (earlyCode.dot(iBasebandSignal),
                       earlyCode.dot(qBasebandSignal),
                       promptCode.dot(iBasebandSignal),
                       promptCode.dot(qBasebandSignal),
                       lateCode.dot(iBasebandSignal),
                       lateCode.dot(qBasebandSignal))

    # Get Early, Prompt, and Late accumulated values for pilot BOC(1,1).
    correValues[6:12] = (pilotEarlyCode.dot(iBasebandSignal),
                         pilotEarlyCode.dot(qBasebandSignal),
                         pilotPromptCode.dot(iBasebandSignal),
                         pilotPromptCode.dot(qBasebandSignal),
                         pilotLateCode.dot(iBasebandSignal),
                         pilotLateCode.dot(qBasebandSignal))

    if settings.fullBandEn == 1:
        # Correlation values for pilot BOC(6,1) spreading waveform.
        correValues[12:18] = (p61_earlyCode.dot(iBasebandSignal),
                              p61_earlyCode.dot(qBasebandSignal),
                              p61_promptCode.dot(iBasebandSignal),
                              p61_promptCode.dot(qBasebandSignal),
                              p61_lateCode.dot(iBasebandSignal),
                              p61_lateCode.dot(qBasebandSignal) )

    return correValues


#%% Python (Numpy) correlator for channel-parallel tracking
def corrPyParallelB1C(settings, rawSignal, B1CCodeTable, remCarrPhase,
                      carrPhaseStep, remCodePhase, codePhaseStep,
                      startIdx, chSampSize, isDataRead):
    """Correlate all active BDS-3 B1C channels against one signal block.

    This function has the same calling interface as the SIMD and GPU
    correlators and is used when ``settings.correlatorType == 0``. Each
    output column corresponds to one tracking channel.

    Args
    ----
        settings       - object
                       Receiver settings containing ``fileType``,
                       ``dllCorrelatorSpacing`` and ``fullBandEn``.
        rawSignal      - ndarray
                       Shared IF-signal block. Complex samples are stored as
                       interleaved I and Q values.
        B1CCodeTable   - ndarray
                       One guarded data/pilot/QMBOC local-code row for each
                       active channel.
        remCarrPhase   - ndarray
                       Residual carrier phase for each channel, in radians.
        carrPhaseStep  - ndarray
                       Carrier phase increment for each channel, in radians
                       per sample.
        remCodePhase   - ndarray
                       Residual code phase for each channel.
        codePhaseStep  - ndarray
                       Code-phase increment for each channel.
        startIdx       - ndarray
                       Start sample of each channel within ``rawSignal``.
        chSampSize     - ndarray
                       Number of samples in each channel's code period.
        isDataRead     - int
                       Shared-data refresh flag: ``1`` for new data and ``0``
                       to reuse the current block.
    Returns
    -------
        correValues    - ndarray
                       An ``18 x channelCnt`` matrix. Each column contains
                       ``I_E, Q_E, I_P, Q_P, I_L, Q_L`` for data BOC(1,1),
                       pilot BOC(1,1), and pilot BOC(6,1).
    """
    # For complex data, split the interleaved buffer [I0 Q0 I1 Q1 ...] into
    # I and Q views of the shared input block.
    if settings.fileType == 2:
        rawSignalI = rawSignal[::2]
        rawSignalQ = rawSignal[1::2]

    # Number of active tracking channels in this tracking epoch.
    channelCnt = startIdx.size
    # Output order per channel contains 18 data and pilot values. Each group
    # of six is [I_E, Q_E, I_P, Q_P, I_L, Q_L].
    correValues = np.zeros((18, channelCnt), dtype=np.float64)
    # Define early-late offset in chips.
    earlyLateSpc = settings.dllCorrelatorSpacing

    for channelNr in range(channelCnt):
        # Find the size of the current code period in whole samples for the
        # current channel and the start index of this channel block within the
        # shared rawSignal buffer.
        blksize = chSampSize[channelNr]
        codeStartIdx = startIdx[channelNr]

        # Get the three local-code branches for the current channel. The table
        # already contains the guard values required by ceil(tcode).
        B1CCode = B1CCodeTable[channelNr, :]
        codeLen = B1CCode.size // 3
        B1CCodeD = B1CCode[:codeLen]
        B1CCodeP = B1CCode[codeLen : codeLen * 2]
        pilotBOC61 = B1CCode[codeLen * 2 :]

        # Extract the current signal block to be processed by this channel.
        signalIndex = slice(codeStartIdx, codeStartIdx + blksize)
        if settings.fileType == 1:
            rawSignalBlock = rawSignal[signalIndex]
        else:
            rawSignalBlockI = rawSignalI[signalIndex]
            rawSignalBlockQ = rawSignalQ[signalIndex]
            rawSignalBlock = rawSignalBlockI + 1j * rawSignalBlockQ

        # --- Set up all the code phase tracking information -----------------
        sampleIndex = np.arange(blksize)
        codePhase = remCodePhase[channelNr] + codePhaseStep[channelNr] * sampleIndex

        # Define index into early code vectors.
        tcode = np.ceil(codePhase - earlyLateSpc).astype(np.int32)
        earlyCodeD = B1CCodeD[tcode]
        earlyCodeP = B1CCodeP[tcode]
        p61_earlyCode = pilotBOC61[tcode]

        # Define index into late code vectors.
        tcode = np.ceil(codePhase + earlyLateSpc).astype(np.int32)
        lateCodeD = B1CCodeD[tcode]
        lateCodeP = B1CCodeP[tcode]
        p61_lateCode = pilotBOC61[tcode]

        # Define index into prompt code vectors. Python uses zero-based
        # indices, so MATLAB's trailing +1 index offset is not required.
        tcode = np.ceil(codePhase).astype(np.int32)
        promptCodeD = B1CCodeD[tcode]
        promptCodeP = B1CCodeP[tcode]
        p61_promptCode = pilotBOC61[tcode]

        # --- Generate the carrier frequency to mix the signal to baseband ---
        # carrPhaseStep is already the carrier phase step in radians per
        # sample, so form the local carrier phase directly from sample indices
        # rather than from a time vector in seconds.
        trigarg = carrPhaseStep[channelNr] * sampleIndex + remCarrPhase[channelNr]
        carrsig = np.exp(-1j * trigarg)

        # --- Do correlation to generate the standard accumulated values -----
        # First mix to baseband.
        basebandSignal = carrsig * rawSignalBlock
        iBasebandSignal = basebandSignal.real
        qBasebandSignal = basebandSignal.imag

        # Get Early, Prompt, and Late values for the data channel. Output
        # order matches the SIMD/GPU channel-parallel correlator interface.
        correValues[:6, channelNr] = (earlyCodeD.dot(iBasebandSignal),
                                      earlyCodeD.dot(qBasebandSignal),
                                      promptCodeD.dot(iBasebandSignal),
                                      promptCodeD.dot(qBasebandSignal),
                                      lateCodeD.dot(iBasebandSignal),
                                      lateCodeD.dot(qBasebandSignal))

        # Get Early, Prompt, and Late values for pilot BOC(1,1).
        correValues[6:12, channelNr] = (earlyCodeP.dot(iBasebandSignal),
                                        earlyCodeP.dot(qBasebandSignal),
                                        promptCodeP.dot(iBasebandSignal),
                                        promptCodeP.dot(qBasebandSignal),
                                        lateCodeP.dot(iBasebandSignal),
                                        lateCodeP.dot(qBasebandSignal))

        if settings.fullBandEn == 1:
            # Correlation values for pilot BOC(6,1) spreading waveform.
            correValues[12:18, channelNr] = (p61_earlyCode.dot(iBasebandSignal),
                                             p61_earlyCode.dot(qBasebandSignal),
                                             p61_promptCode.dot(iBasebandSignal),
                                             p61_promptCode.dot(qBasebandSignal),
                                             p61_lateCode.dot(iBasebandSignal),
                                             p61_lateCode.dot(qBasebandSignal))

    return correValues


#%% Jacobi-symbol calculation
def JacobiSymbol(a, b):
    """Compute the Jacobi symbol ``(a/b)`` for integer arguments.

    When ``b`` is prime, the result is the Legendre symbol: one for a
    quadratic residue modulo ``b``, minus one for a quadratic nonresidue,
    and zero when ``a`` is divisible by ``b``.

    Args
    ----
        a           - int
                    First argument.
        b           - int
                    Positive odd second argument; it is the modulus when
                    prime.
    Returns
    -------
        result      - int
                    Jacobi symbol. The value ``-2`` indicates an invalid
                    nonpositive or even second argument.
    """
    # The second argument must be a positive odd integer.
    if b <= 0 or b % 2 == 0:
        return -2
    # Reduce the first argument modulo the second argument.
    a %= b
    result = 1
    while a:
        # Remove factors of two and apply the supplementary Jacobi law.
        while a % 2 == 0:
            a //= 2
            if b % 8 in (3, 5):
                result = -result
        # Swap the arguments and use quadratic reciprocity.
        a, b = b, a
        if a % 4 == 3 and b % 4 == 3:
            result = -result
        # Continue with the reduced remainder.
        a %= b
    return result if b == 1 else 0


#%% BDS-3 B1C pilot BOC(1,1) code generation
def generatePilotBOC11(settings, PRN):
    """Generate the BDS-3 B1C pilot BOC(1,1) spreading waveform.

    Args
    ----
        settings    - object
                    Receiver settings containing the primary-code length.
        PRN         - int
                    PRN number of the sequence.
    Returns
    -------
        B1CPilot    - numpy.ndarray
                    Pilot-channel B1C code expanded to the length of the
                    pilot BOC(6,1) waveform.
    """
    # Matrix of phase difference w and truncation point p for the B1C
    # pilot primary code.
    wp_pilot = np.array((
        [796, 7575],      [156, 2369],      [4198, 5688],       [3941, 539],
        [1374, 2270],     [1338, 7306],     [1833, 6457],       [2521, 6254],
        [3175, 5644],     [168, 7119],      [2715, 1402],       [4408, 5557],
        [3160, 5764],     [2796, 1073],     [459, 7001],        [3594, 5910],
        [4813, 10060],    [586, 2710],      [1428, 1546],       [2371, 6887],
        [2285, 1883],     [3377, 5613],     [4965, 5062],       [3779, 1038],
        [4547, 10170],    [1646, 6484],     [1430, 1718],       [607, 2535],
        [2118, 1158],     [4709, 526],      [1149, 7331],       [3283, 5844],
        [2473, 6423],     [1006, 6968],     [3670, 1280],       [1817, 1838],
        [771, 1989],      [2173, 6468],     [740, 2091],        [1433, 1581],
        [2458, 1453],     [3459, 6252],     [2155, 7122],       [1205, 7711],
        [413, 7216],      [874, 2113],      [2463, 1095],       [1106, 1628],
        [1590, 1713],     [3873, 6102],     [4026, 6123],       [4272, 6070],
        [3556, 1115],     [128, 8047],      [1200, 6795],       [130, 2575],
        [4494, 53],       [1871, 1729],     [3073, 6388],       [4386, 682],
        [4098, 5565],     [1923, 7160],     [1176, 2277]
    ), dtype=np.int32)

    #--- Compute the Jacobi symbols ---------------------------------------
    N = 10243
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        legendre[ind] = JacobiSymbol(ind, N)
    legendre[legendre == -1] = 0

    #--- Generate the B1C primary code ------------------------------------
    # The B1C primary-code length is 10230 chips.
    Primary = np.zeros(settings.codeLength, dtype=np.float64)
    p = wp_pilot[PRN - 1, 1]
    w = wp_pilot[PRN - 1, 0]
    ind = np.arange(10230, dtype=np.int32)
    k = (ind + p - 1) % N
    Primary[:10230] = np.logical_xor(
        legendre[k], legendre[(k + w) % N])

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Primary = 1 - 2 * Primary

    #--- Add the BOC(1,1) subcarrier -------------------------------------
    B1CPilot = np.empty(settings.codeLength * 2, dtype=np.float64)
    B1CPilot[0::2] = -Primary
    B1CPilot[1::2] = Primary

    #--- Expand to the length of the pilot BOC(6,1) waveform -------------
    return np.repeat(B1CPilot, 6)


#%% BDS-3 B1C pilot BOC(6,1) code generation
def generatePilotBOC61(settings, PRN):
    """Generate the BDS-3 B1C pilot BOC(6,1) spreading waveform.

    Args
    ----
        settings    - object
                    Receiver settings containing the primary-code length.
        PRN         - int
                    PRN number of the sequence.
    Returns
    -------
        B1CPilot    - numpy.ndarray
                    Pilot BOC(6,1) waveform in bipolar format.
    """
    # Matrix of phase difference w and truncation point p for the B1C
    # pilot primary code.
    wp_pilot = np.array((
        [796, 7575],      [156, 2369],      [4198, 5688],       [3941, 539],
        [1374, 2270],     [1338, 7306],     [1833, 6457],       [2521, 6254],
        [3175, 5644],     [168, 7119],      [2715, 1402],       [4408, 5557],
        [3160, 5764],     [2796, 1073],     [459, 7001],        [3594, 5910],
        [4813, 10060],    [586, 2710],      [1428, 1546],       [2371, 6887],
        [2285, 1883],     [3377, 5613],     [4965, 5062],       [3779, 1038],
        [4547, 10170],    [1646, 6484],     [1430, 1718],       [607, 2535],
        [2118, 1158],     [4709, 526],      [1149, 7331],       [3283, 5844],
        [2473, 6423],     [1006, 6968],     [3670, 1280],       [1817, 1838],
        [771, 1989],      [2173, 6468],     [740, 2091],        [1433, 1581],
        [2458, 1453],     [3459, 6252],     [2155, 7122],       [1205, 7711],
        [413, 7216],      [874, 2113],      [2463, 1095],       [1106, 1628],
        [1590, 1713],     [3873, 6102],     [4026, 6123],       [4272, 6070],
        [3556, 1115],     [128, 8047],      [1200, 6795],       [130, 2575],
        [4494, 53],       [1871, 1729],     [3073, 6388],       [4386, 682],
        [4098, 5565],     [1923, 7160],     [1176, 2277]
    ), dtype=np.int32)

    #--- Compute the Jacobi symbols ---------------------------------------
    N = 10243
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        legendre[ind] = JacobiSymbol(ind, N)
    legendre[legendre == -1] = 0

    #--- Generate the B1C primary code ------------------------------------
    # The B1C primary-code length is 10230 chips.
    Primary = np.zeros(settings.codeLength, dtype=np.float64)
    p = wp_pilot[PRN - 1, 1]
    w = wp_pilot[PRN - 1, 0]
    ind = np.arange(10230, dtype=np.int32)
    k = (ind + p - 1) % N
    Primary[:10230] = np.logical_xor(
        legendre[k], legendre[(k + w) % N])

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Primary = 1 - 2 * Primary

    #--- Add the BOC(6,1) subcarrier -------------------------------------
    subcarrier = np.tile((-1.0, 1.0), settings.codeLength * 6)
    B1CPilot = np.repeat(Primary, 12) * subcarrier
    return B1CPilot


#%% BDS-3 B1C data BOC(1,1) code generation
def generateDataBOC11(settings, PRN):
    """Generate the BDS-3 B1C data BOC(1,1) spreading waveform.

    Args
    ----
        settings    - object
                    Receiver settings containing the primary-code length.
        PRN         - int
                    PRN number of the sequence.
    Returns
    -------
        B1CData     - numpy.ndarray
                    Data-channel B1C code expanded to the length of the
                    pilot BOC(6,1) waveform.
    """
    # Matrix of phase difference w and truncation point p for the B1C
    # data primary code.
    wp_data = np.array((
        [2678, 699],       [4802, 694],       [958, 7318],       [859, 2127],
        [3843, 715],       [2232, 6682],      [124, 7850],       [4352, 5495],
        [1816, 1162],      [1126, 7682],      [1860, 6792],      [4800, 9973],
        [2267, 6596],      [424, 2092],       [4192, 19],        [4333, 10151],
        [2656, 6297],      [4148, 5766],      [243, 2359],       [1330, 7136],
        [1593, 1706],      [1470, 2128],      [882, 6827],       [3202, 693],
        [5095, 9729],      [2546, 1620],      [1733, 6805],      [4795, 534],
        [4577, 712],       [1627, 1929],      [3638, 5355],      [2553, 6139],
        [3646, 6339],      [1087, 1470],      [1843, 6867],      [216, 7851],
        [2245, 1162],      [726, 7659],       [1966, 1156],      [670, 2672],
        [4130, 6043],      [53, 2862],        [4830, 180],       [182, 2663],
        [2181, 6940],      [2006, 1645],      [1080, 1582],      [2288, 951],
        [2027, 6878],      [271, 7701],       [915, 1823],       [497, 2391],
        [139, 2606],       [3693, 822],       [2054, 6403],      [4342, 239],        
        [3342, 442],       [2592, 6769],      [1007, 2560],      [310, 2502],
        [4203, 5072],      [455, 7268],       [4318, 341]
    ), dtype=np.int32)

    #--- Compute the Jacobi symbols ---------------------------------------
    N = 10243
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        legendre[ind] = JacobiSymbol(ind, N)
    legendre[legendre == -1] = 0

    #--- Generate the B1C primary code ------------------------------------
    # The B1C primary-code length is 10230 chips.
    Primary = np.zeros(settings.codeLength, dtype=np.float64)
    p = wp_data[PRN - 1, 1]
    w = wp_data[PRN - 1, 0]
    ind = np.arange(10230, dtype=np.int32)
    k = (ind + p - 1) % N
    Primary[:10230] = np.logical_xor(
        legendre[k], legendre[(k + w) % N])

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Primary = 1 - 2 * Primary

    #--- Add the BOC(1,1) subcarrier -------------------------------------
    B1CData = np.empty(settings.codeLength * 2, dtype=np.float64)
    B1CData[0::2] = -Primary
    B1CData[1::2] = Primary

    #--- Expand to the length of the pilot BOC(6,1) waveform -------------
    return np.repeat(B1CData, 6)


#%% BDS-3 B1C pilot secondary-code generation
def generate2ndCode(PRN):
    """Generate the BDS-3 B1C pilot secondary code.

    Args
    ----
        PRN         - int
                    PRN number of the sequence.
    Returns
    -------
        Secondary   - numpy.ndarray
                    The 1800-chip B1C pilot secondary code in bipolar
                    format.
    """
    # Phase difference w and truncation point p for the B1C pilot secondary
    # codes, PRNs 1 through 63.
    wp_pilot = np.array((
        [269, 1889],       [1448, 1268],      [1028, 1593],      [1324, 1186],
        [822, 1239],       [5, 1930],         [155, 176],        [458, 1696],
        [310, 26],         [959, 1344],       [1238, 1271],      [1180, 1182],
        [1288, 1381],      [334, 1604],       [885, 1333],       [1362, 1185],
        [181, 31],         [1648, 704],       [838, 1190],       [313, 1646],
        [750, 1385],       [225, 113],        [1477, 860],       [309, 1656],
        [108, 1921],       [1457, 1173],      [149, 1928],       [322, 57],
        [271, 150],        [576, 1214],       [1103, 1148],      [450, 1458],
        [399, 1519],       [241, 1635],       [1045, 1257],      [164, 1687],
        [513, 1382],       [687, 1514],       [422, 1],          [303, 1583],
        [324, 1806],       [495, 1664],       [725, 1338],       [780, 1111],
        [367, 1706],       [882, 1543],       [631, 1813],       [37, 228],
        [647, 2871],       [1043, 2884],      [24, 1823],        [120, 75],
        [134, 11],         [136, 63],         [158, 1937],       [214, 22],
        [335, 1768],       [340, 1526],       [661, 1402],       [889, 1445],
        [929, 1680],       [1002, 1290],      [1149, 1245]
    ), dtype=np.int32)

    #--- Compute the Jacobi symbols ---------------------------------------
    N = 3607
    legendre = np.zeros(N, dtype=np.int8)
    for ind in range(1, N):
        legendre[ind] = JacobiSymbol(ind, N)
    legendre[legendre == -1] = 0

    #--- Generate the B1C secondary code ----------------------------------
    Secondary = np.zeros(1800, dtype=np.float64)
    p = wp_pilot[PRN - 1, 1]
    w = wp_pilot[PRN - 1, 0]
    ind = np.arange(1800, dtype=np.int32)
    k = (ind + p - 1) % N
    Secondary[:] = np.logical_xor(
        legendre[k], legendre[(k + w) % N])

    # Convert to bipolar format: 0 -> +1, 1 -> -1.
    Secondary = 1 - 2 * Secondary
    return Secondary
#%% BDS-3 B1C-code sampling
def codeSampling(settings, PRN, sampleLen, component):
    """Generate and sample one BDS-3 B1C spreading waveform.

    Args
    ----
        settings      - object
                      Receiver settings.
        PRN           - int
                      PRN number of the sequence.
        sampleLen     - int
                      Number of output samples.
        component     - str
                      B1C component: ``"data"``, ``"pilotBOC11"`` or
                      ``"pilotBOC61"``.

    Returns
    -------
        codeSamples   - numpy.ndarray
                      Sampled B1C spreading waveform.
    """
    #--- Generate the selected B1C spreading waveform -----------------------
    if component == "data":
        code = generateDataBOC11(settings, PRN)
    elif component == "pilotBOC11":
        code = generatePilotBOC11(settings, PRN)
    elif component == "pilotBOC61":
        code = generatePilotBOC61(settings, PRN)
    else:
        raise ValueError(f"Invalid B1C component: {component}")

    #--- Find time constants ------------------------------------------------
    ts = 1 / settings.samplingFreq
    tc = 1 / settings.codeFreqBasis / 12

    #=== Digitizing =========================================================
    #--- Make the index array used to read B1C code values ------------------
    # The length of the index array depends on the sampling frequency.
    codeValueIndex = np.ceil(
        ts * np.arange(1, sampleLen + 1) / tc).astype(np.int32)

    #--- Correct the first and last indices due to rounding -----------------
    codeValueIndex[-1] = settings.codeLength * 12
    codeValueIndex[0] = 1

    #--- Make the digitized version of the B1C code -------------------------
    # The upsampled code is formed by selecting code-chip values at each
    # sampling instant.
    return code[codeValueIndex - 1]
