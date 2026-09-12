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

tracking.py - Module Description
--------------------------------
GLONASS L2OC code and carrier tracking.

"""

from copy import copy

import numpy as np

from commUtils import (
    WaitBar,
    calcCNoPld,
    calcLoopCoef,
)


#%% Initialize result structure
class Channel:
    """Store tracking results for one receiver channel.

    The results contain L2OCp prompt outputs, absolute code-period starting
    positions, and loop observations saved once per
    coherent integration interval.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize the results for one tracking channel.

        Args
        ----
            settings   - object
                       Receiver settings.
        Returns
        -------
            None
                Tracking-result arrays are stored in this object.
        """
        # Number of coherent tracking epochs to preallocate.
        numToProcess = round(
            settings.msToProcess / 1000 / settings.intTime)
        # PRN assigned to this channel.
        self.PRN = 0
        # Loop-completion flag; this is not a signal-lock detector.
        self.lockFlag = False
        # Absolute input-sample index at the start of each L2OC code epoch.
        self.absoluteSample = np.zeros(numToProcess)
        # Code-NCO frequency in chips/s.
        self.codeFreq = np.full(numToProcess, np.inf)
        # Carrier-NCO frequency in Hz.
        self.carrFreq = np.full(numToProcess, np.inf)
        # Data-branch in-phase Early, Prompt, and Late correlator outputs.
        self.I_E = np.zeros(numToProcess)
        self.I_P = np.zeros(numToProcess)
        self.I_L = np.zeros(numToProcess)
        # Data-branch quadrature Early, Prompt, and Late correlator outputs.
        self.Q_E = np.zeros(numToProcess)
        self.Q_P = np.zeros(numToProcess)
        self.Q_L = np.zeros(numToProcess)
        # Loop discriminators.
        self.dllDiscr = np.full(numToProcess, np.inf)
        self.dllDiscrFilt = np.full(numToProcess, np.inf)
        self.pllDiscr = np.full(numToProcess, np.inf)
        self.pllDiscrFilt = np.full(numToProcess, np.inf)
        # Residual code phase in chips and carrier phase in radians.
        self.remCodePhase = np.full(numToProcess, np.inf)
        self.remCarrPhase = np.full(numToProcess, np.inf)
        # Periodic C/N0 estimates in dB-Hz and PLL lock-detector results.
        cnoCount = numToProcess // settings.CNoVSMinterval
        self.DataCNo = np.zeros(cnoCount)
        self.DataPLD = np.zeros(cnoCount)

#%% Tracking engine
class TrackingEngine():
    """Perform GLONASS L2OC code and carrier tracking.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize the configured tracking channels.

        Args
        ----
            settings   - object
                       Receiver settings.
        Returns
        -------
            None
                Empty tracking channels are stored in this object.
        """
        self._settings = settings
        self.trackResults = [
                   Channel(settings) for _ in range(settings.numberOfChannels)]

    @property
    def settings(self):
        """Return the receiver-settings object retained by the engine.

        Returns
        -------
            settings   - object
                       Receiver-settings object retained by the engine.
        """
        return self._settings

    #%% Channel-serial tracking
    def trkChannelsSerial(self, acqResults, corrEngine):
        """Perform GLONASS L2OC code and carrier tracking for selected channels.

        The channels are processed using channel-serial tracking mode. Each
        correlator call processes the L2OCp branch on its fine TDM/BOC grid.

        Args
        ----
            acqResults - acquisition.AcqEngine
                       PRNs, carrier frequencies, code frequencies, and code
                       phases of all satellites to be tracked.
            corrEngine - callable or object
                       Correlator used for channel-serial tracking.
        Returns
        -------
            None
                Tracking results are stored in ``trackResults`` once per
                coherent integration interval. C/N0 and PLL-detector values
                are stored at the configured C/N0 interval.
        """
        from correlator import generateL2OcpBOCCode

        #--- Copy initial settings for all channels -----------------------
        settings = self.settings
        trkResults = self.trackResults
        # Determine how many acquired L2OC channels will be tracked.
        TrackedNr = min(sum(acqResults.acqFlag), settings.numberOfChannels)

        #%% Initialize tracking variables =================================
        # L2OCp is tracked over one complete 20 ms equivalent-code period.
        codeLength = settings.codeLength
        fineFactor = settings.L2OCFineFactor
        # Correlators use the fine L2OCp TDM/BOC grid.
        corrSettings = copy(settings)
        corrSettings.codeLength = settings.codeLength * fineFactor
        corrSettings.codeFreqBasis = settings.codeFreqBasis * fineFactor
        corrSettings.dllCorrelatorSpacing = (
            settings.dllCorrelatorSpacing * fineFactor)
        # Coherent integration times for the DLL and PLL.
        PDIcode = PDIcarr = settings.intTime
        # DLL and PLL loop-filter coefficients.
        tau1code, tau2code = calcLoopCoef(
            settings.dllNoiseBandwidth, settings.dllDampingRatio, 1.0)
        
        tau1carr, tau2carr = calcLoopCoef(
            settings.pllNoiseBandwidth, settings.pllDampingRatio, 1.0)

        # Number of preallocated coherent L2OCp integrations.
        numToProcess = trkResults[0].I_P.size

        # The GUI bar update period depends on the correlator backend.
        if settings.correlatorType == 0:
            # NumPy reference QPSK correlator.
            barUprate = 5
        else:
            # Compiled SIMD/GPU backends use less frequent GUI updates.
            barUprate = 200
            outputType = np.int16

        # Stored bytes per logical real or complex input sample.
        dataAdaptCoeff = settings.dataAdaptCoeff
        sampleType     = np.dtype(settings.dataType)
        bytesPerSample = dataAdaptCoeff * sampleType.itemsize

        # Open the raw IF recording once and seek for each serial channel.
        fid = open(settings.fileName, "rb")
        waitbar = WaitBar(numToProcess)

        #%% Start processing channels =====================================
        for channelNr in range(TrackedNr):
            # Mutable result container for the current L2OC channel.
            chResults = trkResults[channelNr]
            # Only process a channel for which acquisition succeeded.
            if not acqResults.acqFlag[channelNr]:
                continue

            # Save the tracked PRN.
            chResults.PRN = int(acqResults.PRN[channelNr])

            # Seek to skipNumberOfSamples plus the zero-based acquisition code phase.
            fid.seek(bytesPerSample * (settings.skipNumberOfSamples 
                                       + int(acqResults.codePhase[channelNr])))

            # The first QPSK-table branch carries the L2OCp TDM/BOC code;
            # the unused CSI branch is zero-filled to retain the common ABI.
            L2OCpCode = generateL2OcpBOCCode(chResults.PRN)
            L2OCpCode = np.concatenate(
                ([L2OCpCode[-1]], L2OCpCode, [L2OCpCode[0]]))
            L2CSICode = np.zeros(settings.codeLength*fineFactor)
            L2CSICode = np.concatenate(
                ([L2CSICode[-1]], L2CSICode, [L2CSICode[0]]))
            L2OCCodeTable = np.concatenate((L2OCpCode, L2CSICode))
            if settings.correlatorType == 1:
                L2OCCodeTable = np.asarray(L2OCCodeTable, dtype=np.int32)
            elif settings.correlatorType == 2:
                L2OCCodeTable = np.asarray(L2OCCodeTable, dtype=np.int8)

            #--- Perform various initializations --------------------------
            # Initial code and carrier NCO frequencies from acquisition.
            codeFreq = float(acqResults.codeFreq[channelNr])
            codeFreqBasis = float(acqResults.codeFreq[channelNr])
            carrFreq = float(acqResults.carrFreq[channelNr])
            carrFreqBasis = float(acqResults.carrFreq[channelNr])
            # Residual code phase in primary chips and carrier phase in rad.
            remCodePhase = 0.0
            remCarrPhase = 0.0
            # Code and carrier loop-filter memories.
            oldCodeNco = 0.0
            oldCodeError = 0.0
            oldCarrNco = 0.0
            oldCarrError = 0.0
            # C/N0 display and 0.5-0.5 smoothing state.
            CNoValue = np.zeros(3)
            tempCNoValue = np.zeros(3)
            # OC2 has 50 symbols, one per 20 ms coherent integration.
            OC2SignSeq = 1-2*np.asarray(settings.OC2, dtype=np.int8)
            oc2StartIdx = -1
            oc2EstFirst = round(0.3/settings.intTime)
            oc2EstLen = round(0.3/settings.intTime)
            oc2PromptBuf = np.zeros(oc2EstLen, dtype=np.complex128)

            #=== Process the requested coherent integrations ==============
            for loopCnt in range(numToProcess):
                #%% GUI update --------------------------------------------
                # Update the progress display periodically without repainting
                # it for every coherent L2OC integration interval.
                if ((loopCnt + 1) % barUprate == 0 or loopCnt + 1 == numToProcess):
                    
                    completedMs = round((loopCnt + 1) * settings.intTime * 1000)
                    totalMs = round(numToProcess * settings.intTime * 1000)
                    
                    trackingStatus = (
                        f"Tracking: Ch {channelNr + 1} of {TrackedNr}\n"
                        f"PRN: {chResults.PRN}\n"
                        f"Completed {completedMs} of {totalMs} msec\n"
                        f"L2OCp C/N0: {CNoValue[0]:.0f} (dB-Hz)")
                    
                    if not waitbar.update(loopCnt + 1, trackingStatus):
                        # Closing the progress display cancels tracking and
                        # exits the current processing run.
                        print("Progress bar closed, exiting...")
                        waitbar.close()
                        fid.close()
                        return

                #%% Read next block of data -------------------------------
                # Record the absolute sample index of this L2OC code epoch.
                chResults.absoluteSample[loopCnt] = (
                                                  fid.tell() / bytesPerSample)

                # Code phase step in primary-code chips per input sample.
                codePhaseStep = codeFreq / settings.samplingFreq
                # Samples required to complete the current primary code.
                blksize = int(np.ceil(
                                (codeLength - remCodePhase) / codePhaseStep))

                # Read one coherent integration of real or interleaved I/Q.
                if settings.correlatorType == 0:
                    # NumPy reference-correlator input.
                    rawSignal = np.fromfile(fid, settings.dataType,
                                            dataAdaptCoeff * blksize)
                else:
                    # SIMD/GPU backends consume int16 input samples.
                    rawSignal0 = np.fromfile(fid, settings.dataType, 
                                             dataAdaptCoeff * blksize)
                    rawSignal = rawSignal0.astype(outputType, copy=False)

                # Stop if the recording ends before a complete integration
                # interval is read.
                if rawSignal.size != dataAdaptCoeff * blksize:
                    print("Not able to read the specified number of samples "
                          "for tracking, exiting!")
                    fid.close()
                    waitbar.close()
                    return

                #%% Correlator implementation =============================
                # Carrier phase step in rad/sample.
                carrPhaseStep = (carrFreq * 2.0 * np.pi / settings.samplingFreq)
                # Convert the residual phase to the fine L2OCp grid.
                corrRemCodePhase = remCodePhase * fineFactor
                corrCodePhaseStep = codePhaseStep * fineFactor
                if settings.correlatorType == 0:
                    # NumPy reference QPSK correlator.
                    correValues = corrEngine(
                        corrSettings, rawSignal, L2OCCodeTable, remCarrPhase,
                        carrPhaseStep, corrRemCodePhase,
                        corrCodePhaseStep)
                elif settings.correlatorType == 1:
                    # CPU SIMD DLL QPSK correlator.
                    correValues = corrEngine.corrEngine(
                        corrSettings, rawSignal, L2OCCodeTable, remCarrPhase,
                        carrPhaseStep, corrRemCodePhase,
                        corrCodePhaseStep)
                elif settings.correlatorType == 2:
                    # CUDA DLL QPSK correlator.
                    correValues = corrEngine.corrEngine(
                        corrSettings, rawSignal, L2OCCodeTable, remCarrPhase,
                        carrPhaseStep, corrRemCodePhase,
                        corrCodePhaseStep,
                        chResults.PRN)

                # L2OCp Early/Prompt/Late I/Q outputs.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6]

                # Remove the 50-symbol OC2 overlay after its phase is known.
                if oc2StartIdx < 0:
                    if oc2EstFirst <= loopCnt < oc2EstFirst+oc2EstLen:
                        oc2PromptBuf[loopCnt-oc2EstFirst] = complex(I_P, Q_P)
                    if loopCnt == oc2EstFirst+oc2EstLen-1:
                        zDiff = oc2PromptBuf[1:]*np.conjugate(oc2PromptBuf[:-1])
                        metric = np.empty(50)
                        loopNow = np.arange(oc2EstFirst+1,
                                            oc2EstFirst+oc2EstLen)
                        loopPrev = loopNow-1
                        for phase in range(50):
                            transitions = (OC2SignSeq[(phase+loopNow) % 50] *
                                           OC2SignSeq[(phase+loopPrev) % 50])
                            metric[phase] = np.sum(np.real(zDiff*transitions))
                        oc2StartIdx = int(np.argmax(metric))
                        sortedMetric = np.sort(metric)[::-1]
                        metricRatio = sortedMetric[0]/max(
                            np.finfo(float).eps, abs(sortedMetric[1]))
                        print(f"\nPRN {chResults.PRN} automatic OC2 start "
                              f"index = {oc2StartIdx+1}, metric ratio = "
                              f"{metricRatio:.2f}")

                if oc2StartIdx >= 0:
                    oc2Index = (oc2StartIdx + round(
                        loopCnt*settings.intTime/0.02)) % 50
                    oc2Sign = OC2SignSeq[oc2Index]
                    I_E *= oc2Sign; Q_E *= oc2Sign
                    I_P *= oc2Sign; Q_P *= oc2Sign
                    I_L *= oc2Sign; Q_L *= oc2Sign

                #--- Save and update variables for current correlation ----
                # Save residual phases used by this correlation interval.
                chResults.remCodePhase[loopCnt] = remCodePhase
                chResults.remCarrPhase[loopCnt] = remCarrPhase
                # Carry residual phases into the next integration.
                remCodePhase = (
                    blksize * codePhaseStep
                    + remCodePhase - codeLength)
                remCarrPhase = np.fmod(
                    carrPhaseStep * blksize + remCarrPhase,
                    2 * np.pi)
                #%% Find PLL error and update carrier NCO ==================
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)

                # Second-order carrier-loop filter and NCO correction.
                carrNco = (oldCarrNco + (tau2carr / tau1carr) * 
                 (carrError - oldCarrError) + carrError * (PDIcarr / tau1carr))
                oldCarrNco = carrNco
                oldCarrError = carrError
                chResults.carrFreq[loopCnt] = carrFreq
                carrFreq = carrFreqBasis + carrNco

                #%% Find DLL error and update code NCO =====================
                E = np.sqrt(I_E ** 2 + Q_E ** 2)
                L = np.sqrt(I_L ** 2 + Q_L ** 2)
                codeError = (E - L) / (E + L)

                # Second-order code-loop filter and NCO correction.
                codeNco = (oldCodeNco + (tau2code / tau1code) * 
                 (codeError - oldCodeError) + codeError * (PDIcode / tau1code))
                oldCodeNco = codeNco
                oldCodeError = codeError
                chResults.codeFreq[loopCnt] = codeFreq
                codeFreq = codeFreqBasis - codeNco

                #%% Record measures for postprocessing ====================
                chResults.dllDiscr[loopCnt] = codeError
                chResults.dllDiscrFilt[loopCnt] = codeNco
                chResults.pllDiscr[loopCnt] = carrError
                chResults.pllDiscrFilt[loopCnt] = carrNco

                chResults.I_E[loopCnt] = I_E
                chResults.I_P[loopCnt] = I_P
                chResults.I_L[loopCnt] = I_L
                chResults.Q_E[loopCnt] = Q_E
                chResults.Q_P[loopCnt] = Q_P
                chResults.Q_L[loopCnt] = Q_L

                #%% C/N0 calculation ======================================
                if ((loopCnt + 1) % settings.CNoVSMinterval == 0):
                    CNoValue, pllDetector = calcCNoPld(chResults, 
                                                            settings, loopCnt)
                    
                    cnoCnt = (loopCnt + 1)// settings.CNoVSMinterval - 1
                    
                    # A 0.5-0.5 filter smooths the L2OCp C/N0.
                    chResults.DataCNo[cnoCnt] = (CNoValue[0] * 0.5 
                                                 + tempCNoValue[0] * 0.5)
                    
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                   
                tempCNoValue = CNoValue.copy()

            # Completing all requested integrations marks the channel done.
            chResults.lockFlag = True

        waitbar.close()
        fid.close()
        # Results remain available in self.trackResults.
        return


    #%% Channel-parallel tracking
    def trkChannelsParallel(self, acqResults, corrEngine):
        """Perform GLONASS L2OC tracking for all active channels.

        The channels are processed using channel-parallel tracking mode and
        share one IF-data buffer for every processing block.

        Args
        ----
            acqResults - acquisition.AcqEngine
                       PRNs, carrier frequencies, code frequencies, and code
                       phases of all satellites to be tracked.
            corrEngine - callable or object
                       Correlator used for channel-parallel tracking.
        Returns
        -------
            None
                Tracking results are stored in ``trackResults`` once per
                coherent integration interval. C/N0 and PLL-detector values
                are stored at the configured C/N0 interval.
        """
        from correlator import generateL2OcpBOCCode

        settings = self.settings
        
        #--- Copy result variables for all channels -----------------------
        # Channel objects contain the preallocated L2OCp tracking arrays.
        trkResults = self.trackResults
        
        # Number of active L2OC channels to be tracked.
        channelCnt = int(min(sum(acqResults.acqFlag), settings.numberOfChannels))
        
        if channelCnt == 0:
            return

        #%% Construct variables for tracking loops ========================
        # Previous DLL filter output (code-NCO command) [chips/s].
        oldCodeNco = np.zeros(channelCnt)
        # Previous DLL discriminator output.
        oldCodeError = np.zeros(channelCnt)
        # Previous PLL filter output (carrier-NCO command) [Hz].
        oldCarrNco = np.zeros(channelCnt)
        # Previous PLL discriminator output [cycles].
        oldCarrError = np.zeros(channelCnt)
        # Code-NCO frequencies initialized from acquisition [chips/s].
        codeFreq = np.asarray(
            acqResults.codeFreq[:channelCnt],
            dtype=np.float64).copy()
        # Fixed code-frequency bases for the loop update [chips/s].
        codeFreqBasis = codeFreq.copy()
        # Residual primary-code phase [chips].
        remCodePhase = np.zeros(channelCnt)
        # Residual carrier phase [rad].
        remCarrPhase = np.zeros(channelCnt)
        # Carrier-NCO frequencies initialized from acquisition [Hz].
        carrFreq = np.asarray(
            acqResults.carrFreq[:channelCnt],
            dtype=np.float64).copy()
        # Fixed acquired carrier-frequency bases [Hz].
        carrFreqBasis = carrFreq.copy()
        # Absolute zero-based sample of each channel's current code epoch.
        absoluteSample = ( int(settings.skipNumberOfSamples) + np.asarray(
            acqResults.codePhase[:channelCnt], dtype=np.int64))

        #%% Initialize tracking variables =================================
        # DLL and PLL coherent integration intervals [s].
        PDIcode = PDIcarr = settings.intTime
        #--- DLL variables ------------------------------------------------
        # DLL loop-filter coefficients.
        tau1code, tau2code = calcLoopCoef(settings.dllNoiseBandwidth,
                                          settings.dllDampingRatio, 1.0)
        #--- PLL variables ------------------------------------------------
        # PLL loop-filter coefficients.
        tau1carr, tau2carr = calcLoopCoef(settings.pllNoiseBandwidth,
                                          settings.pllDampingRatio, 1.0)

        # Equivalent L2OCp code length for one 20 ms integration.
        codeLength = settings.codeLength
        fineFactor = settings.L2OCFineFactor
        # Correlators use the fine L2OCp TDM/BOC grid.
        corrSettings = copy(settings)
        corrSettings.codeLength = settings.codeLength * fineFactor
        corrSettings.codeFreqBasis = settings.codeFreqBasis * fineFactor
        corrSettings.dllCorrelatorSpacing = (
            settings.dllCorrelatorSpacing * fineFactor)

        #%% Generate L2OC code tables for acquired signals =================
        # Each row contains guarded L2OCp and zero-filled CSI branches.
        branchLength = settings.codeLength * fineFactor + 2
        L2OCCodeTable = np.empty(
            (channelCnt, branchLength * 2), dtype=np.float64)

        for channelNr in range(channelCnt):
            chResults = trkResults[channelNr]
            chResults.PRN = int(acqResults.PRN[channelNr])
            L2OCpCode = generateL2OcpBOCCode(chResults.PRN)
            L2OCpCode = np.concatenate(
                ([L2OCpCode[-1]], L2OCpCode, [L2OCpCode[0]]))
            L2CSICode = np.zeros(settings.codeLength*fineFactor)
            L2CSICode = np.concatenate(
                ([L2CSICode[-1]], L2CSICode, [L2CSICode[0]]))
            L2OCCodeTable[channelNr] = np.concatenate((L2OCpCode, L2CSICode))

        if settings.correlatorType == 1:
            # SIMD shares an int32 QPSK code table across all channels.
            L2OCCodeTable = np.asarray(L2OCCodeTable, dtype=np.int32)
        elif settings.correlatorType == 2:
            # GPU shares an int8 QPSK code table across all channels.
            L2OCCodeTable = np.asarray(L2OCCodeTable, dtype=np.int8)

        #%% Initialize waitbar and variables for data reading =============
        # Number of coherent L2OC tracking intervals to process.
        numToProcess = trkResults[0].I_P.size
        # Update the GUI less often for the faster compiled correlators.
        if settings.correlatorType == 0:
            # NumPy reference QPSK correlator.
            barUprate = 5
        else:
            # Compiled SIMD/GPU QPSK correlators.
            barUprate = 100
            outputType = np.int16

        # Display values and per-channel C/N0 smoothing states.
        CNoValue = np.zeros(3)
        cnoSmoothState = np.zeros((channelCnt, 3))
        # Per-channel OC2 phase-estimation state.
        OC2SignSeq = 1-2*np.asarray(settings.OC2, dtype=np.int8)
        oc2StartIdx = np.full(channelCnt, -1, dtype=np.int32)
        oc2EstFirst = round(0.3/settings.intTime)
        oc2EstLen = round(0.3/settings.intTime)
        oc2PromptBuf = np.zeros(
            (channelCnt, oc2EstLen), dtype=np.complex128)

        #--- Variables for IF-signal reading ------------------------------
        # Stored scalars per logical sample: one real value or interleaved I/Q.
        dataAdaptCoeff = settings.dataAdaptCoeff
        # NumPy sample type, used to convert logical samples to byte offsets.
        sampleType = np.dtype(settings.dataType)
        # Number of stored bytes per logical real or complex sample.
        bytesPerSample = dataAdaptCoeff * sampleType.itemsize
        # One shared second of logical samples. Reuse the receiver's
        # samplesPerCode property and the 50 L2OCp periods per second.
        samplesPerCode = settings.samplesPerCode
        codePeriodsPerSec = round(settings.codeFreqBasis / settings.codeLength)
        samplesPerSec = samplesPerCode * codePeriodsPerSec
        # Inclusive ending sample and start of the current shared buffer.
        blkEndIdx = -1
        blkStartIdx = 0

        #--- Initialize waitbar -------------------------------------------
        # Open the IF record and create the channel-parallel progress display.
        fid = open(settings.fileName, "rb")
        waitbar = WaitBar(numToProcess)

        #%% Tracking processing for all channels ==========================
        # 1. Update the GUI.
        # 2. Read the next shared IF-data block when necessary.
        # 3. Correlate all active channels.
        # 4. Update every channel's tracking loops.
        # 5. Periodically estimate C/N0 and PLL detector outputs.
        for loopCnt in range(numToProcess):
            #%% GUI update -----------------------------------------------
            # Update the GUI periodically without repainting every L2OC
            # coherent integration interval.
            if ((loopCnt + 1) % barUprate == 0 or loopCnt + 1 == numToProcess):
                completedMs = round((loopCnt + 1) * settings.intTime * 1000)
                totalMs = round(numToProcess * settings.intTime * 1000)
                
                trackingStatus = (
                    f"Tracking {channelCnt} PRNs\n"
                    f"1st PRN: {trkResults[0].PRN}\n"
                    f"Completed {completedMs} of {totalMs} msec\n"
                    f"1st L2OCp C/N0: {CNoValue[0]:.0f} (dB-Hz)")
                
                if not waitbar.update(loopCnt + 1, trackingStatus):
                    # Closing the progress display cancels tracking and exits
                    # the current processing run.
                    print("Progress bar closed, exiting...")
                    waitbar.close()
                    fid.close()
                    return

            #%% Read next block of data -----------------------------------
            # Code phase increment [primary chips/sample] for each code NCO.
            codePhaseStep = codeFreq / settings.samplingFreq
            # Whole-sample coherent-integration length for every channel.
            chSampSize = np.ceil( (codeLength - remCodePhase)
                                 / codePhaseStep).astype(np.int32)

            # Flag passed to compiled correlators when rawSignal is refreshed.
            isDataRead = 0
            # Latest absolute sample required by any channel in this interval.
            lastSample = np.max( absoluteSample + chSampSize - 1)
            # Read another shared one-second block only when a channel window
            # extends beyond the current buffer.
            if blkEndIdx < lastSample:
                isDataRead = 1
                # Read from the earliest active channel so one buffer serves
                # every channel.
                blkStartIdx = int(np.min(absoluteSample))
                # Convert its logical-sample position to a file byte offset.
                fid.seek(bytesPerSample * blkStartIdx)

                if settings.correlatorType == 0:
                    # NumPy reference-correlator input.
                    rawSignal = np.fromfile( fid, settings.dataType,
                                            dataAdaptCoeff * samplesPerSec)
                else:
                    # SIMD/GPU backends consume int16 input samples.
                    rawSignal0 = np.fromfile(fid, settings.dataType,
                                             dataAdaptCoeff * samplesPerSec)
                    rawSignal = rawSignal0.astype(outputType, copy=False)

                # A short read means the requested tracking data is unavailable.
                if rawSignal.size != dataAdaptCoeff * samplesPerSec:
                    print("Not able to read the specified number of samples "
                          "for tracking, exiting!")
                    fid.close()
                    waitbar.close()
                    return

                # Inclusive absolute ending sample of the refreshed buffer.
                blkEndIdx = blkStartIdx + samplesPerSec - 1

            #%% Correlator implementation ---------------------------------
            # Carrier phase increment in radians/sample for every carrier NCO.
            carrPhaseStep = (
                carrFreq * 2.0 * np.pi / settings.samplingFreq)
           
            # Zero-based start of each channel interval within rawSignal.
            startIdx = np.asarray(absoluteSample - blkStartIdx,dtype=np.int32)
            # Convert each residual code phase to the fine L2OCp grid.
            corrRemCodePhase = remCodePhase * fineFactor
            corrCodePhaseStep = codePhaseStep * fineFactor
            # All backends return the common 12-row QPSK matrix; the first
            # six rows contain the L2OCp Early/Prompt/Late correlations.
            if settings.correlatorType == 0:
                # NumPy reference QPSK correlator.
                correValues = corrEngine(
                    corrSettings, rawSignal, L2OCCodeTable, remCarrPhase,
                    carrPhaseStep, corrRemCodePhase, corrCodePhaseStep, startIdx,
                    chSampSize, isDataRead)
            elif settings.correlatorType in (1, 2):
                # CPU SIMD or CUDA QPSK correlator.
                correValues = corrEngine.corrEngine(
                    corrSettings, rawSignal, L2OCCodeTable, remCarrPhase,
                    carrPhaseStep, corrRemCodePhase, corrCodePhaseStep, startIdx,
                    chSampSize, isDataRead)

            #%% Update tracking-loop variables for all channels ===========
            for channelNr in range(channelCnt):
                # Mutable tracking-result container for the current channel.
                chResults = trkResults[channelNr]

                # L2OCp Early, Prompt, and Late outputs.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6, channelNr]

                # Estimate and remove the OC2 phase independently per channel.
                if oc2StartIdx[channelNr] < 0:
                    if oc2EstFirst <= loopCnt < oc2EstFirst+oc2EstLen:
                        oc2PromptBuf[channelNr, loopCnt-oc2EstFirst] = complex(
                            I_P, Q_P)
                    if loopCnt == oc2EstFirst+oc2EstLen-1:
                        prompt = oc2PromptBuf[channelNr]
                        zDiff = prompt[1:]*np.conjugate(prompt[:-1])
                        metric = np.empty(50)
                        loopNow = np.arange(oc2EstFirst+1,
                                            oc2EstFirst+oc2EstLen)
                        loopPrev = loopNow-1
                        for phase in range(50):
                            transitions = (OC2SignSeq[(phase+loopNow) % 50] *
                                           OC2SignSeq[(phase+loopPrev) % 50])
                            metric[phase] = np.sum(np.real(zDiff*transitions))
                        oc2StartIdx[channelNr] = int(np.argmax(metric))

                if oc2StartIdx[channelNr] >= 0:
                    oc2Index = (oc2StartIdx[channelNr]+round(
                        loopCnt*settings.intTime/0.02)) % 50
                    oc2Sign = OC2SignSeq[oc2Index]
                    I_E *= oc2Sign; Q_E *= oc2Sign
                    I_P *= oc2Sign; Q_P *= oc2Sign
                    I_L *= oc2Sign; Q_L *= oc2Sign

                # Save the phases used for this correlation, then advance them
                # to the next L2OC coherent integration interval.
                chResults.remCodePhase[loopCnt] = remCodePhase[channelNr]
                chResults.remCarrPhase[loopCnt] = remCarrPhase[channelNr]
                remCodePhase[channelNr] = (chSampSize[channelNr] * codePhaseStep[channelNr] 
                                       + remCodePhase[channelNr] - codeLength)
                remCarrPhase[channelNr] = np.fmod(carrPhaseStep[channelNr]
                   * chSampSize[channelNr] + remCarrPhase[channelNr], 2 * np.pi)
                #%% Find PLL error and update carrier NCO ------------------
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)

                # PLL filter output and carrier-NCO correction [Hz].
                carrNco = (oldCarrNco[channelNr]+ (tau2carr / tau1carr)
                    * (carrError - oldCarrError[channelNr])
                    + carrError * (PDIcarr / tau1carr))
                oldCarrNco[channelNr] = carrNco
                oldCarrError[channelNr] = carrError
                # Save the carrier frequency used for this correlation.
                chResults.carrFreq[loopCnt] = carrFreq[channelNr]
                # Apply the carrier-NCO command for the next correlation.
                carrFreq[channelNr] = carrFreqBasis[channelNr] + carrNco

                #%% Find DLL error and update code NCO ---------------------
                E = np.sqrt(I_E ** 2 + Q_E ** 2)
                L = np.sqrt(I_L ** 2 + Q_L ** 2)
                codeError = (E - L) / (E + L)

                # DLL filter output and code-NCO correction [chips/s].
                codeNco = (oldCodeNco[channelNr] + (tau2code / tau1code)
                           * (codeError - oldCodeError[channelNr])
                           + codeError * (PDIcode / tau1code))
                oldCodeNco[channelNr] = codeNco
                oldCodeError[channelNr] = codeError
                # Save the code frequency used for this correlation.
                chResults.codeFreq[loopCnt] = codeFreq[channelNr]
                # Apply the code-NCO command for the next correlation.
                codeFreq[channelNr] = codeFreqBasis[channelNr] - codeNco

                #%% Record measures for postprocessing --------------------
                # Record this epoch's absolute sample and advance it.
                chResults.absoluteSample[loopCnt] = absoluteSample[channelNr]
                absoluteSample[channelNr] += chSampSize[channelNr]
                
                # Record loop and correlator observations.
                chResults.dllDiscr[loopCnt] = codeError
                chResults.dllDiscrFilt[loopCnt] = codeNco
                chResults.pllDiscr[loopCnt] = carrError
                chResults.pllDiscrFilt[loopCnt] = carrNco
                chResults.I_E[loopCnt] = I_E
                chResults.I_P[loopCnt] = I_P
                chResults.I_L[loopCnt] = I_L
                chResults.Q_E[loopCnt] = Q_E
                chResults.Q_P[loopCnt] = Q_P
                chResults.Q_L[loopCnt] = Q_L

                #%% C/N0 calculation --------------------------------------
                # Periodically estimate L2OCp C/N0 and PLL detector output.
                if ((loopCnt + 1) % settings.CNoVSMinterval == 0):
                    
                    currentCNoValue, pllDetector = (calcCNoPld(chResults, 
                                                        settings, loopCnt))
                    cnoCnt = (loopCnt + 1)// settings.CNoVSMinterval - 1
                            
                    # Apply the same 0.5-0.5 C/N0 smoothing used by the
                    # channel-serial L2OC tracking path.
                    chResults.DataCNo[cnoCnt] = (currentCNoValue[0] * 0.5                        
                                         + cnoSmoothState[channelNr, 0] * 0.5)
                    
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                    
                    if channelNr == 0:
                        CNoValue = currentCNoValue
                    cnoSmoothState[channelNr] = currentCNoValue

        # Completing the loop marks each active channel done.
        for channelNr in range(channelCnt):
            trkResults[channelNr].lockFlag = True
        waitbar.close()
        fid.close()
        # Results remain available in self.trackResults.
        return


    #%% Plot tracking results
    def plotTracking(self):
        """Plot the tracking results for all completed channels.

        Returns
        -------
            None
                Tracking-loop and data-channel C/N0 figures are displayed.
        """
        import matplotlib.pyplot as plt

        settings = self.settings

        #--- For all completed channels -----------------------------------
        for channelNr, chResults in enumerate(self.trackResults):
            if not chResults.lockFlag:
                continue

            #%% Select or create and clear the figure =====================
            # Offset receiver-owned figure numbers to avoid replacing figures
            # created elsewhere in the receiver.
            fig = plt.figure(channelNr + 200, figsize=(10, 7), clear=True)
            
            fig.suptitle(f"Channel {channelNr + 1} - PRN {chResults.PRN}")

            #%% Draw axes ==================================================
            spec = fig.add_gridspec(3, 3)
            # Row 1.
            h11 = fig.add_subplot(spec[0, 0])
            h12 = fig.add_subplot(spec[0, 1:])
            # Row 2.
            h21 = fig.add_subplot(spec[1, 0])
            h22 = fig.add_subplot(spec[1, 1:])
            # Row 3.
            h31 = fig.add_subplot(spec[2, 0])
            h32 = fig.add_subplot(spec[2, 1])
            h33 = fig.add_subplot(spec[2, 2])

            timeAxisInSeconds = ((np.arange(chResults.I_P.size) + 1)
                                 * settings.intTime)

            #%% Plot all figures ==========================================
            #--- Data prompt-correlator I/Q scatter plot ------------------
            h11.plot( chResults.I_P, chResults.Q_P, ".", markersize=2)
            h11.axis("equal")
            h11.set(title="Discrete-Time Scatter Plot",
                    xlabel="I prompt", ylabel="Q prompt")

            #--- L2OC navigation-message symbols -------------------------
            h12.plot(timeAxisInSeconds, chResults.I_P)
            h12.set(title="Bits of the navigation message")

            #--- Data-channel Early, Prompt and Late magnitudes -----------
            correlationResults = np.column_stack((
                np.hypot(chResults.I_E, chResults.Q_E),
                np.hypot(chResults.I_P, chResults.Q_P),
                np.hypot(chResults.I_L, chResults.Q_L)))
            
            h22.plot(timeAxisInSeconds, correlationResults,marker="*", markersize=2)
            h22.set(title="Correlation results")
            h22.legend(("Early", "Prompt", "Late"))

            #--- Raw and filtered PLL/DLL discriminator outputs -----------
            discriminatorPlots = ( (h21, chResults.pllDiscr, "r",
                                    "Raw PLL discriminator", "Cycles"),
                                  (h31, chResults.pllDiscrFilt, "b",
                                   "Filtered PLL discriminator", "Hz"),
                                  (h32, chResults.dllDiscr, "r",
                                   "Raw DLL discriminator", "Normalized error"),
                                  (h33, chResults.dllDiscrFilt, "b",
                                   "Filtered DLL discriminator", "Chips/s"))
            for axes, values, color, title, ylabel in discriminatorPlots:
                axes.plot(timeAxisInSeconds, values, color)
                axes.set(title=title, ylabel=ylabel)

            # Apply the common grid and time-axis formatting.
            timeAxes = (h12, h21, h22, h31, h32, h33)
            for axes in (h11,) + timeAxes:
                axes.grid(True)
            for axes in timeAxes:
                axes.set_xlabel("Time (s)")
                axes.margins(x=0)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            #--- Periodic data-channel C/N0 estimates ---------------------
            CNoInterval = settings.CNoVSMinterval * settings.intTime
            CNoTime = (np.arange(1, chResults.DataCNo.size + 1) * CNoInterval)
            CNoFig = plt.figure(channelNr + 300, figsize=(8, 4), clear=True)
            
            CNoAxes = CNoFig.add_subplot(1, 1, 1)
            CNoAxes.plot(CNoTime, chResults.DataCNo, ".-", markersize=4)
            CNoAxes.set(title=f"Data C/N0 estimation ({CNoInterval:g} s interval)",
                        xlabel="Time (s)", ylabel="C/N0 (dB-Hz)")
            CNoAxes.grid(True)
            CNoAxes.margins(x=0)
            CNoFig.tight_layout()

        plt.show()


    #%% Run tracking
    def trackingRun(self, acqResults):
        """Run GLONASS L2OC code and carrier tracking.

        Args
        ----
            acqResults - acquisition.AcqEngine
                         Prioritized acquisition results.

        Returns
        -------
            self       - TrackingEngine
                         This engine containing trackResults.
        """
        settings = self.settings
        TrackedNr = min(sum(acqResults.acqFlag),settings.numberOfChannels)
        if TrackedNr == 0:
            return self

        import correlator

        corrEngine = None
        try:
            #--- Channel-serial tracking ----------------------------------
            if settings.trkMode == 0:
                if settings.correlatorType == 0:
                    # Python reference correlator
                    corrEngine = correlator.corrPySerialQPSK
                elif settings.correlatorType == 1:
                    # SIMD correlator
                    corrEngine = correlator.CorrSIMDSerialQPSK()
                elif settings.correlatorType == 2:
                    # GPU correlator
                    corrEngine = correlator.CorrGPUSerialQPSK()

                self.trkChannelsSerial(acqResults, corrEngine)

            #--- Channel-parallel tracking --------------------------------
            elif settings.trkMode == 1:
                if settings.correlatorType == 0:
                    # Python reference correlator
                    corrEngine = correlator.corrPyParallelQPSK
                elif settings.correlatorType == 1:
                    # SIMD correlator
                    corrEngine = correlator.CorrSIMDParallelQPSK()
                elif settings.correlatorType == 2:
                    # GPU correlator
                    corrEngine = correlator.CorrGPUParallelQPSK()

                self.trkChannelsParallel(acqResults, corrEngine)
        finally:
            # Release persistent native buffers owned by SIMD/CUDA engines.
            if (corrEngine is not None) and (settings.correlatorType != 0):
                corrEngine.close()

        return self


#%% Module entry point
# This module defines tracking classes and performs no standalone processing.
if __name__ == "__main__":
    pass
