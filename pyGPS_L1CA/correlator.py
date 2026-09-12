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
Generate, sample, and correlate GPS L1 C/A codes.

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
            ctypes.c_int,     # PRN
        ]
        self._gpuLibrary.corrEngine.restype = ctypes.POINTER(ctypes.c_double)

        self._gpuLibrary.corrEngineFree.argtypes = []
        self._gpuLibrary.corrEngineFree.restype = None

    def corrEngine(self, settings, rawSignal, caCode, remCarrPhase,
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
            PRN         - int
                        Current satellite PRN used for code caching.

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
            int(PRN),
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
class CorrGPUParallelFusedBPSK:
    """Call the fused channel-parallel CUDA BPSK tracking correlator."""

    def __init__(self):
        """Load the fused CUDA DLL and declare its ctypes interface.

        Returns
        -------
            None
        """
        dllPath = "../native_Correlators/corrGPUParallelFusedBPSK.dll"
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
        """Release buffers allocated by the fused CUDA engine.
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

#%% For GPS L1 C/A codes ------------------------------------------------------
def generateCAcode(settings,PRN):
    """Generate one period of the selected GPS L1 C/A code.

    Args
    ----
        settings    - object
                    Receiver settings containing ``codeLength``.
        PRN         - int
                    PRN number of the sequence, from 1 through 210.

    Returns
    -------
        CAcode      - numpy.ndarray
                    Desired C/A-code sequence in chips.
    """
    # --- Make the code shift array. The shift depends on the PRN number ------
    # The lookup table gives the G2 shift required for each PRN; for example,
    # PRN 19 uses ``g2s[19] == 471``.
    g2s = {
        1:    5,    2:    6,    3:    7,    4:    8,
        5:   17,    6:   18,    7:  139,    8:  140,
        9:  141,   10:  251,   11:  252,   12:  254,
        13:  255,   14:  256,   15:  257,   16:  258,
        17:  469,   18:  470,   19:  471,   20:  472,
        21:  473,   22:  474,   23:  509,   24:  512,
        25:  513,   26:  514,   27:  515,   28:  516,
        29:  859,   30:  860,   31:  861,   32:  862,
        33:  863,   34:  950,   35:  947,   36:  948,
        37:  950,
        38:   67,   39:  103,   40:   91,   41:   19,
        42:  679,   43:  225,   44:  625,   45:  946,
        46:  638,   47:  161,   48: 1001,   49:  554,
        50:  280,   51:  710,   52:  709,   53:  775,
        54:  864,   55:  558,   56:  220,   57:  397,
        58:   55,   59:  898,   60:  759,   61:  367,
        62:  299,   63: 1018,
        64:  729,   65:  695,   66:  780,   67:  801,
        68:  788,   69:  732,   70:   34,   71:  320,
        72:  327,   73:  389,   74:  407,   75:  525,
        76:  405,   77:  221,   78:  761,   79:  260,
        80:  326,   81:  955,   82:  653,   83:  699,
        84:  422,   85:  188,   86:  438,   87:  959,
        88:  539,   89:  879,   90:  677,   91:  586,
        92:  153,   93:  792,   94:  814,   95:  446,
        96:  264,   97: 1015,   98:  278,   99:  536,
        100:  819,  101:  156,  102:  957,  103:  159,
        104:  712,  105:  885,  106:  461,  107:  248,
        108:  713,  109:  126,  110:  807,  111:  279,
        112:  122,  113:  197,  114:  693,  115:  632,
        116:  771,  117:  467,  118:  647,  119:  203,
        120:  145,  121:  175,  122:   52,  123:   21,
        124:  237,  125:  235,  126:  886,  127:  657,
        128:  634,  129:  762,  130:  355,  131: 1012,
        132:  176,  133:  603,  134:  130,  135:  359,
        136:  595,  137:   68,  138:  386,  139:  797,
        140:  456,  141:  499,  142:  883,  143:  307,
        144:  127,  145:  211,  146:  121,  147:  118,
        148:  163,  149:  628,  150:  853,  151:  484,
        152:  289,  153:  811,  154:  202,  155: 1021,
        156:  463,  157:  568,  158:  904,  159:  670,
        160:  230,  161:  911,  162:  684,  163:  309,
        164:  644,  165:  932,  166:   12,  167:  314,
        168:  891,  169:  212,  170:  185,  171:  675,
        172:  503,  173:  150,  174:  395,  175:  345,
        176:  846,  177:  798,  178:  992,  179:  357,
        180:  995,  181:  877,  182:  112,  183:  144,
        184:  476,  185:  193,  186:  109,  187:  445,
        188:  291,  189:   87,  190:  399,  191:  292,
        192:  901,  193:  339,  194:  208,  195:  711,
        196:  189,  197:  263,  198:  537,  199:  663,
        200:  942,  201:  173,  202:  900,  203:   30,
        204:  500,  205:  935,  206:  556,  207:  373,
        208:   85,  209:  652,  210:  310,
        }

    assert 1 <= PRN <= 210, 'Invalid PRN code:'+str(PRN)

    # --- Pick right shift for the given PRN number ---------------------------
    g2shift = g2s[PRN]

    # --- Generate G1 code ----------------------------------------------------
    # Initialize g1 output to speed up the function
    g1 = np.zeros(settings.codeLength)
    # Load shift register
    reg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

    # Generate all G1 signal chips based on the G1 feedback polynomial
    for ind in range(settings.codeLength):
        g1[ind] = reg[9]
        reg = [reg[9] * reg[2]] + reg[0:9]

    # --- Generate G2 code ----------------------------------------------------
    # Initialize g2 output to speed up the function
    g2 = np.zeros(settings.codeLength)
    # Load shift register
    reg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    # Generate all G2 signal chips based on the G2 feedback polynomial
    for ind in range(settings.codeLength):
      g2[ind] = reg[9]
      reg = [reg[9] * reg[8]*reg[7]*reg[5]*reg[2]*reg[1]] + reg[0:9]

    # --- Shift G2 code -------------------------------------------------------
    # Circularly shift G2 by the PRN-specific number of chips.
    g2 = np.concatenate([g2[settings.codeLength-g2shift:],
                         g2[:settings.codeLength-g2shift]])

    # --- Form the one-chip-per-element bipolar C/A sequence ------------------
    return -(g1 * g2)

#%% GPS L1 C/A-code sampling
def codeSampling(settings,PRN,sampleLen):
    """Digitize a GPS L1 C/A code at the receiver sampling frequency.

    Args
    ----
        settings    - object
                    Receiver settings containing the sampling frequency,
                    code frequency and code length.
        PRN         - int
                    Specified PRN for the C/A code.
        sampleLen   - int
                    Number of output samples.

    Returns
    -------
        caCodesTable - numpy.ndarray
                     Digitized C/A-code samples for the specified PRN.
    """
    # Generate CA code for given PRN ------------------------------------------
    caCode= generateCAcode(settings,PRN)

    # Find time constants -----------------------------------------------------
    ts = 1/settings.samplingFreq           # Sampling period in sec
    tc = 1/settings.codeFreqBasis          # code chip period in sec

    # Digitizing --------------------------------------------------------------
    # Make indices to read C/A code values. The length of the index array
    # depends on the sampling frequency
    codeValueIndex = np.mod(np.floor(ts/tc * np.arange(sampleLen)),
                            settings.codeLength).astype(np.int32)

    # Make the digitized version of the C/A code ------------------------------
    # The "upsampled" code is made by selecting values from the C/A code chip
    # array for the time instances of each sample.
    return caCode[codeValueIndex]

