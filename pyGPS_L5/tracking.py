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
GPS L5 code and carrier tracking.

"""

import numpy as np

from commUtils import (
    WaitBar,
    calcCNoPld,
    calcLoopCoef,
    calcLoopCoefCarr,
)


#%% Initialize result structure
class Channel:
    """Store tracking results for one receiver channel.

    The results contain data- and pilot-channel prompt outputs, absolute L5
    primary-code starting positions, and loop observations saved once per
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
        # Absolute input-sample index at the start of each L5 code epoch.
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
        # Pilot-branch prompt correlator outputs.
        self.Pilot_I_P = np.zeros(numToProcess)
        self.Pilot_Q_P = np.zeros(numToProcess)
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
        self.PilotCNo = np.zeros(cnoCount)
        self.PilotPLD = np.zeros(cnoCount)
        self.L5C_CNo = np.zeros(cnoCount)

#%% Tracking engine
class TrackingEngine():
    """Perform GPS L5 code and carrier tracking.

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
        """Perform GPS L5 code and carrier tracking for selected channels.

        The channels are processed using channel-serial tracking mode. Each
        correlator call processes the L5I data and L5Q pilot branches on the
        common primary-code chip grid.

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
        from correlator import generateL5Icode, generateL5Qcode

        #--- Copy initial settings for all channels -----------------------
        settings = self.settings
        trkResults = self.trackResults
        # Determine how many acquired L5 channels will be tracked.
        TrackedNr = min(sum(acqResults.acqFlag), settings.numberOfChannels)

        #%% Initialize tracking variables =================================
        # L5I and L5Q use the primary-code chip grid.
        codeLength = settings.codeLength
        corrSettings = settings
        # Coherent integration times for the DLL and PLL.
        PDIcode = settings.intTime
        # DLL and third-order PLL loop-filter coefficients.
        tau1code, tau2code = calcLoopCoef(
            settings.dllNoiseBandwidth, settings.dllDampingRatio, 1.0)
        pf3, pf2, pf1 = calcLoopCoefCarr(settings)

        # Number of preallocated coherent L5 primary-code integrations.
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
            # Mutable result container for the current L5 channel.
            chResults = trkResults[channelNr]
            # Only process a channel for which acquisition succeeded.
            if not acqResults.acqFlag[channelNr]:
                continue

            # Save the tracked PRN.
            chResults.PRN = int(acqResults.PRN[channelNr])

            # Seek to skipNumberOfSamples plus the zero-based acquisition code phase.
            fid.seek(bytesPerSample * (settings.skipNumberOfSamples 
                                       + int(acqResults.codePhase[channelNr])))

            # Get L5I and L5Q codes sampled once per primary-code chip and
            # add one guard chip at each end for Early/Late indexing.
            L5ICode = generateL5Icode(
                settings, chResults.PRN)
            L5ICode = np.concatenate(([L5ICode[-1]], L5ICode, [L5ICode[0]]))
            L5QCode = generateL5Qcode(
                settings, chResults.PRN)
            L5QCode = np.concatenate(([L5QCode[-1]], L5QCode, [L5QCode[0]]))
            L5CCodeTable = np.concatenate((L5ICode, L5QCode))
            if settings.correlatorType == 1:
                L5CCodeTable = np.asarray(L5CCodeTable, dtype=np.int32)
            elif settings.correlatorType == 2:
                L5CCodeTable = np.asarray(L5CCodeTable, dtype=np.int8)

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
            d2CarrError = 0.0
            dCarrError = 0.0
            # C/N0 display and 0.5-0.5 smoothing state.
            CNoValue = np.zeros(3)
            tempCNoValue = np.zeros(3)

            #=== Process the requested coherent integrations ==============
            for loopCnt in range(numToProcess):
                #%% GUI update --------------------------------------------
                # Update the progress display periodically without repainting
                # it for every coherent L5 integration interval.
                if ((loopCnt + 1) % barUprate == 0 or loopCnt + 1 == numToProcess):
                    
                    completedMs = round((loopCnt + 1) * settings.intTime * 1000)
                    totalMs = round(numToProcess * settings.intTime * 1000)
                    
                    trackingStatus = (
                        f"Tracking: Ch {channelNr + 1} of {TrackedNr}\n"
                        f"PRN: {chResults.PRN}\n"
                        f"Completed {completedMs} of {totalMs} msec\n"
                        f"Data C/N0: {CNoValue[0]:.0f} (dB-Hz); "
                        f"Pilot C/N0: {CNoValue[1]:.0f} (dB-Hz)")
                    
                    if not waitbar.update(loopCnt + 1, trackingStatus):
                        # Closing the progress display cancels tracking and
                        # exits the current processing run.
                        print("Progress bar closed, exiting...")
                        waitbar.close()
                        fid.close()
                        return

                #%% Read next block of data -------------------------------
                # Record the absolute sample index of this L5 code epoch.
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
                # All correlators use primary-code chips for code phase.
                if settings.correlatorType == 0:
                    # NumPy reference QPSK correlator.
                    correValues = corrEngine(
                        corrSettings, rawSignal, L5CCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep)
                elif settings.correlatorType == 1:
                    # CPU SIMD DLL QPSK correlator.
                    correValues = corrEngine.corrEngine(
                        corrSettings, rawSignal, L5CCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep)
                elif settings.correlatorType == 2:
                    # CUDA DLL QPSK correlator.
                    correValues = corrEngine.corrEngine(
                        corrSettings, rawSignal, L5CCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep,
                        chResults.PRN)

                # Data and pilot Early/Prompt/Late I/Q outputs.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6]
                (pilot_I_E, pilot_Q_E, pilot_I_P, pilot_Q_P,
                 pilot_I_L, pilot_Q_L) = correValues[6:12]

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
                # L5Q is pi/2 ahead of L5I; rotate it back before combining.
                QI = ((pilot_I_P + 1j * pilot_Q_P) *
                      np.exp(-1j * np.pi / 2))
                carrErrorQ = (np.arctan(np.imag(QI) / np.real(QI)) /
                              (2.0 * np.pi))
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)
                carrError = (carrError + carrErrorQ) / 2

                # Third-order carrier-loop filter and NCO correction.
                d2CarrError += carrError * pf3
                dCarrError += d2CarrError + carrError * pf2
                carrNco = dCarrError + carrError * pf1
                chResults.carrFreq[loopCnt] = carrFreq
                carrFreq = carrFreqBasis + carrNco

                #%% Find DLL error and update code NCO =====================
                E = np.sqrt(I_E ** 2 + Q_E ** 2)
                L = np.sqrt(I_L ** 2 + Q_L ** 2)
                codeError = (E - L) / (E + L)
                pilotE = np.sqrt(pilot_I_E ** 2 + pilot_Q_E ** 2)
                pilotL = np.sqrt(pilot_I_L ** 2 + pilot_Q_L ** 2)
                codeErrorQ = (pilotE - pilotL) / (pilotE + pilotL)
                codeError = (codeError + codeErrorQ) / 2

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

                chResults.Pilot_I_P[loopCnt] = pilot_I_P
                chResults.Pilot_Q_P[loopCnt] = pilot_Q_P

                #%% C/N0 calculation ======================================
                if ((loopCnt + 1) % settings.CNoVSMinterval == 0):
                    CNoValue, pllDetector = calcCNoPld(chResults, 
                                                            settings, loopCnt)
                    
                    cnoCnt = (loopCnt + 1)// settings.CNoVSMinterval - 1
                    
                    # A 0.5-0.5 filter smooths data, pilot, and L5 C/N0.
                    chResults.DataCNo[cnoCnt] = (CNoValue[0] * 0.5 
                                                 + tempCNoValue[0] * 0.5)
                    
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                   
                    chResults.PilotCNo[cnoCnt] = (CNoValue[1] * 0.5 
                                                  + tempCNoValue[1] * 0.5)
                    
                    chResults.L5C_CNo[cnoCnt] = (CNoValue[2] * 0.5
                                                 + tempCNoValue[2] * 0.5)
                    
                    chResults.PilotPLD[cnoCnt] = pllDetector[1]
                    
                tempCNoValue = CNoValue.copy()

            # Completing all requested integrations marks the channel done.
            chResults.lockFlag = True

        waitbar.close()
        fid.close()
        # Results remain available in self.trackResults.
        return


    #%% Channel-parallel tracking
    def trkChannelsParallel(self, acqResults, corrEngine):
        """Perform GPS L5 tracking for all active channels.

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
        from correlator import generateL5Icode, generateL5Qcode

        settings = self.settings
        
        #--- Copy result variables for all channels -----------------------
        # Channel objects contain the preallocated data/pilot tracking arrays.
        trkResults = self.trackResults
        
        # Number of active L5 channels to be tracked.
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

        # Primary L5 code length and Early/Late spacing in primary chips.
        codeLength = settings.codeLength
        corrSettings = settings

        #%% Generate L5 code tables for acquired signals =================
        # Each row contains guarded L5I and L5Q primary-code branches.
        branchLength = codeLength + 2
        L5CCodeTable = np.empty(
            (channelCnt, branchLength * 2), dtype=np.float64)

        for channelNr in range(channelCnt):
            chResults = trkResults[channelNr]
            chResults.PRN = int(acqResults.PRN[channelNr])
            L5ICode = generateL5Icode(
                settings, chResults.PRN)
            L5ICode = np.concatenate(([L5ICode[-1]], L5ICode, [L5ICode[0]]))
            L5QCode = generateL5Qcode(
                settings, chResults.PRN)
            L5QCode = np.concatenate(([L5QCode[-1]], L5QCode, [L5QCode[0]]))
            L5CCodeTable[channelNr] = np.concatenate((L5ICode, L5QCode))

        if settings.correlatorType == 1:
            # SIMD shares an int32 QPSK code table across all channels.
            L5CCodeTable = np.asarray(L5CCodeTable, dtype=np.int32)
        elif settings.correlatorType == 2:
            # GPU shares an int8 QPSK code table across all channels.
            L5CCodeTable = np.asarray(L5CCodeTable, dtype=np.int8)

        #%% Initialize waitbar and variables for data reading =============
        # Number of coherent L5 tracking intervals to process.
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

        #--- Variables for IF-signal reading ------------------------------
        # Stored scalars per logical sample: one real value or interleaved I/Q.
        dataAdaptCoeff = settings.dataAdaptCoeff
        # NumPy sample type, used to convert logical samples to byte offsets.
        sampleType = np.dtype(settings.dataType)
        # Number of stored bytes per logical real or complex sample.
        bytesPerSample = dataAdaptCoeff * sampleType.itemsize
        # Number of logical samples per L5-code period and per second.
        samplesPerCode = settings.samplesPerCode
        samplesPerSec = samplesPerCode * 1000
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
            # Update the GUI periodically without repainting every L5
            # coherent integration interval.
            if ((loopCnt + 1) % barUprate == 0 or loopCnt + 1 == numToProcess):
                completedMs = round((loopCnt + 1) * settings.intTime * 1000)
                totalMs = round(numToProcess * settings.intTime * 1000)
                
                trackingStatus = (
                    f"Tracking {channelCnt} PRNs\n"
                    f"1st PRN: {trkResults[0].PRN}\n"
                    f"Completed {completedMs} of {totalMs} msec\n"
                    f"1st Data C/N0: {CNoValue[0]:.0f} (dB-Hz); "
                    f"1st Pilot C/N0: {CNoValue[1]:.0f} (dB-Hz)")
                
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
            # All backends return a 12 x channelCnt matrix containing L5I
            # and L5Q Early/Prompt/Late I/Q correlations.
            if settings.correlatorType == 0:
                # NumPy reference QPSK correlator.
                correValues = corrEngine(
                    corrSettings, rawSignal, L5CCodeTable, remCarrPhase,
                    carrPhaseStep, remCodePhase, codePhaseStep, startIdx,
                    chSampSize, isDataRead)
            elif settings.correlatorType in (1, 2):
                # CPU SIMD or CUDA QPSK correlator.
                correValues = corrEngine.corrEngine(
                    corrSettings, rawSignal, L5CCodeTable, remCarrPhase,
                    carrPhaseStep, remCodePhase, codePhaseStep, startIdx,
                    chSampSize, isDataRead)

            #%% Update tracking-loop variables for all channels ===========
            for channelNr in range(channelCnt):
                # Mutable tracking-result container for the current channel.
                chResults = trkResults[channelNr]

                # L5I and L5Q Early, Prompt, and Late outputs.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6, channelNr]
                (pilot_I_E, pilot_Q_E, pilot_I_P, pilot_Q_P,
                 pilot_I_L, pilot_Q_L) = correValues[6:12, channelNr]

                # Save the phases used for this correlation, then advance them
                # to the next L5 coherent integration interval.
                chResults.remCodePhase[loopCnt] = remCodePhase[channelNr]
                chResults.remCarrPhase[loopCnt] = remCarrPhase[channelNr]
                remCodePhase[channelNr] = (chSampSize[channelNr] * codePhaseStep[channelNr] 
                                       + remCodePhase[channelNr] - codeLength)
                remCarrPhase[channelNr] = np.fmod(carrPhaseStep[channelNr]
                   * chSampSize[channelNr] + remCarrPhase[channelNr], 2 * np.pi)

                #%% Find PLL error and update carrier NCO ------------------
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)
                QI = ((pilot_I_P + 1j * pilot_Q_P) *
                      np.exp(-1j * np.pi / 2))
                carrErrorQ = (np.arctan(np.imag(QI) / np.real(QI)) /
                              (2.0 * np.pi))
                carrError = (carrError + carrErrorQ) / 2

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
                pilotE = np.sqrt(pilot_I_E ** 2 + pilot_Q_E ** 2)
                pilotL = np.sqrt(pilot_I_L ** 2 + pilot_Q_L ** 2)
                codeErrorQ = (pilotE - pilotL) / (pilotE + pilotL)
                codeError = (codeError + codeErrorQ) / 2

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

                chResults.Pilot_I_P[loopCnt] = pilot_I_P
                chResults.Pilot_Q_P[loopCnt] = pilot_Q_P

                #%% C/N0 calculation --------------------------------------
                # Periodically estimate data, pilot, and combined L5 C/N0,
                # together with data/pilot PLL lock-detector values.
                if ((loopCnt + 1) % settings.CNoVSMinterval == 0):
                    
                    currentCNoValue, pllDetector = (calcCNoPld(chResults, 
                                                        settings, loopCnt))
                    cnoCnt = (loopCnt + 1)// settings.CNoVSMinterval - 1
                            
                    # Apply the same 0.5-0.5 C/N0 smoothing used by the
                    # channel-serial L5 tracking path.
                    chResults.DataCNo[cnoCnt] = (currentCNoValue[0] * 0.5                        
                                         + cnoSmoothState[channelNr, 0] * 0.5)
                    
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                    
                    chResults.PilotCNo[cnoCnt] = (currentCNoValue[1] * 0.5
                        + cnoSmoothState[channelNr, 1] * 0.5)
                    
                    chResults.L5C_CNo[cnoCnt] = (currentCNoValue[2] * 0.5
                                         + cnoSmoothState[channelNr, 2] * 0.5)
                    chResults.PilotPLD[cnoCnt] = pllDetector[1]

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

            #--- L5 navigation-message symbols ---------------------------
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
        """Run GPS L5 code and carrier tracking.

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
