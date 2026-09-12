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
BDS-3 B2a data- and pilot-channel code and carrier tracking.

"""

import numpy as np

from commUtils import WaitBar, calcCNoPld, calcLoopCoef

#%% Initialize result structure
class Channel:
    """Store tracking results for one receiver channel.

    The results contain data- and pilot-channel prompt outputs, absolute B2a
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
        # Absolute input-sample index at the start of each code period.
        self.absoluteSample = np.zeros(numToProcess)
        # Code-NCO frequency in chips/s.
        self.codeFreq = np.full(numToProcess, np.inf)
        # Carrier-NCO frequency in Hz.
        self.carrFreq = np.full(numToProcess, np.inf)
        # In-phase early, prompt and late correlator outputs.
        self.I_E = np.zeros(numToProcess)
        self.I_P = np.zeros(numToProcess)
        self.I_L = np.zeros(numToProcess)
        # Quadrature early, prompt and late correlator outputs.
        self.Q_E = np.zeros(numToProcess)
        self.Q_P = np.zeros(numToProcess)
        self.Q_L = np.zeros(numToProcess)
        # Pilot-channel prompt correlator outputs.
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
        # Periodic data, pilot and combined B2a C/N0 estimates and PLL lock
        # detector outputs.
        cnoCount = numToProcess // settings.CNoVSMinterval
        self.DataCNo = np.zeros(cnoCount)
        self.DataPLD = np.zeros(cnoCount)
        self.PilotCNo = np.zeros(cnoCount)
        self.PilotPLD = np.zeros(cnoCount)
        self.B2a_CNo = np.zeros(cnoCount)

#%% Tracking engine
class TrackingEngine():
    """Perform BDS-3 B2a code and carrier tracking.

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
        """Perform BDS-3 B2a code and carrier tracking for selected channels.
        The channels are processed using channel-serial tracking mode.

        Args
        ----
            acqResults - acquisition.AcqEngine
                       PRNs, carrier frequencies, and code phases of all
                       satellites to be tracked, prepared from acquisition
                       results.
            corrEngine - callable or object
                       Correlator used for channel-serial tracking.
        Returns
        -------
            None
                Tracking results are stored in ``trackResults``. They contain
                in-phase prompt outputs, absolute spreading-code starting
                positions, and other observations saved every millisecond.
                C/N0 is saved at the configured interval.
        """
        from correlator import generateB2aDataCode, generateB2aPilotCode

        #--- Copy initial settings for all channels -----------------------
        settings = self.settings
        trkResults = self.trackResults

        # --- Determine how many acquired channels will be tracked ------------
        TrackedNr = min(sum(acqResults.acqFlag), settings.numberOfChannels)

        #%% Initialize tracking variables =================================
        # Summation intervals for the code and carrier loops.
        PDIcode = PDIcarr = settings.intTime
        #--- DLL variables ------------------------------------------------
        # Calculate DLL filter coefficient values.
        tau1code, tau2code = calcLoopCoef(
            settings.dllNoiseBandwidth, settings.dllDampingRatio, 1.0)
        #--- PLL variables ------------------------------------------------
        # Calculate second-order carrier-loop filter coefficient values.
        tau1carr, tau2carr = calcLoopCoef(
            settings.pllNoiseBandwidth, settings.pllDampingRatio, 1.0)

        # Number of coherent tracking epochs processed by each channel.
        numToProcess = trkResults[0].I_P.size

        # The GUI bar update period depends on the correlator backend.
        if settings.correlatorType == 0:
            # NumPy reference correlator.
            barUprate = 200
        else:
            # Compiled SIMD/GPU backends use less frequent GUI updates.
            barUprate = 2000
            outputType = np.int16

        # Open the raw IF recording once and seek separately for each channel.
        fid = open(settings.fileName,"rb")
        waitbar = WaitBar(numToProcess)

        #%% Start processing channels =====================================
        for channelNr in range(TrackedNr):

            # Mutable result container for the current channel.
            chResults = trkResults[channelNr]
            # Process a channel only when acquisition was successful and a
            # nonzero PRN was assigned.
            if acqResults.acqFlag[channelNr] != True:
                continue

            # Save additional information: the PRN tracked by this channel.
            chResults.PRN = acqResults.PRN[channelNr]

            # Seek to the detected zero-based code phase. Convert logical
            # samples to stored bytes according to dataType and dataAdaptCoeff.
            if settings.dataType == 'int16':
                fid.seek(settings.dataAdaptCoeff * 2 * (settings.skipNumberOfSamples 
                                    + int(acqResults.codePhase[channelNr])))
            elif settings.dataType == 'int8':
                fid.seek(settings.dataAdaptCoeff * (settings.skipNumberOfSamples + 
                                    int(acqResults.codePhase[channelNr])))

            # Generate guarded B2a data and pilot primary-code branches.
            B2aCodeD = generateB2aDataCode(settings, chResults.PRN)
            B2aCodeD = np.concatenate(([B2aCodeD[-1]], B2aCodeD,
                                       [B2aCodeD[0]]))
            B2aCodeP = generateB2aPilotCode(settings, chResults.PRN)
            B2aCodeP = np.concatenate(([B2aCodeP[-1]], B2aCodeP,
                                       [B2aCodeP[0]]))
            B2aCodeTable = np.concatenate((B2aCodeD, B2aCodeP))

            if settings.correlatorType == 1:
                # The SIMD DLL consumes a guarded int32 code table.
                B2aCodeTable = np.asarray(B2aCodeTable, dtype=np.int32)
            elif settings.correlatorType == 2:
                # The CUDA DLL consumes a guarded int8 code table.
                B2aCodeTable = np.asarray(B2aCodeTable, dtype=np.int8)

            # --- Perform various initializations -----------------------------
            # Define initial code frequency basis of NCO
            codeFreq = acqResults.codeFreq[channelNr]
            codeFreqBasis = acqResults.codeFreq[channelNr]
            # Define residual code phase (in chips)
            remCodePhase  = 0.0
            # Define carrier frequency
            carrFreq = acqResults.carrFreq[channelNr]
            carrFreqBasis = acqResults.carrFreq[channelNr]
            # Define residual carrier phase
            remCarrPhase  = 0.0
            # code tracking loop parameters
            oldCodeNco   = 0.0
            oldCodeError = 0.0
            # Carrier/Costas-loop parameters.
            oldCarrNco = 0.0
            oldCarrError = 0.0

            # C/N0 computation and 0.5-0.5 smoothing state.
            CNoValue = np.zeros(3)
            tempCNoValue = np.zeros(3)

            # --- Process the number of specified code periods ----------------
            for loopCnt in range(numToProcess):

                #%% GUI update ============================================
                # Update the progress display periodically. This keeps the
                # interface responsive without repainting every millisecond.
                if ((loopCnt + 1) % barUprate == 0 or
                        loopCnt + 1 == numToProcess):
                    totalMs = round(numToProcess * settings.intTime * 1000)
                    trackingStatus = (
                        f'Ch {channelNr + 1}/{TrackedNr}, PRN {chResults.PRN}\n'
                        f'{round((loopCnt + 1) * settings.intTime * 1000)}'
                        f'/{totalMs} ms, '
                        f'Data C/N0 {CNoValue[0]:.1f} dB-Hz; '
                        f'Pilot C/N0 {CNoValue[1]:.1f} dB-Hz'
                    )
                    if not waitbar.update(loopCnt + 1, trackingStatus):
                        # Closing the progress display cancels tracking and
                        # exits the current processing run.
                        print('Progress bar closed, exiting...')
                        waitbar.close()
                        fid.close()
                        return

                #%% Read next block of data ================================
                # Record sample number (based on samples)
                if settings.dataType == 'int16':
                    chResults.absoluteSample[loopCnt] = (
                        fid.tell()/settings.dataAdaptCoeff/2)
                elif  settings.dataType == 'int8':
                    chResults.absoluteSample[loopCnt] = (
                        fid.tell()/settings.dataAdaptCoeff)

                # Update the phasestep based on code freq (variable) and
                # sampling frequency (fixed)
                codePhaseStep = codeFreq / settings.samplingFreq
                # Find the current code-period length in whole samples.
                blksize = np.ceil((settings.codeLength-remCodePhase) /
                                  codePhaseStep).astype(int)

                # Read the samples required for this tracking epoch.
                if settings.correlatorType == 0:
                    # NumPy reference-correlator input.
                    rawSignal = np.fromfile(fid, settings.dataType,
                                            settings.dataAdaptCoeff*blksize)
                else:
                    # SIMD/GPU backends consume int16 input samples.
                    rawSignal0 = np.fromfile(fid, settings.dataType,
                             settings.dataAdaptCoeff*blksize)
                    rawSignal = rawSignal0.astype(outputType, copy=False)

                # Stop if the recording ends before a complete epoch is read.
                if rawSignal.size != settings.dataAdaptCoeff*blksize:
                    print('''Not able to read the specified number of samples
                          for tracking, exiting!''')
                    fid.close()
                    waitbar.close()
                    return

                #%% Correlator implementation =============================
                # Divide the current carrier NCO by the sampling frequency to
                # obtain the carrier phase increment in radians per sample.
                carrPhaseStep = carrFreq * 2 * np.pi /settings.samplingFreq

                if settings.correlatorType == 0: # NumPy reference correlator
                    correValues = corrEngine(
                        settings, rawSignal, B2aCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep)
                elif settings.correlatorType == 1:  # CPU SIMD DLL backend
                    correValues = corrEngine.corrEngine(
                        settings, rawSignal, B2aCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep)
                elif settings.correlatorType == 2:  # CUDA DLL backend
                    correValues = corrEngine.corrEngine(
                        settings, rawSignal, B2aCodeTable, remCarrPhase,
                        carrPhaseStep, remCodePhase, codePhaseStep,
                        chResults.PRN)

                # Data-branch and pilot-branch Early/Prompt/Late I/Q values.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6]
                (pilot_I_E, pilot_Q_E, pilot_I_P, pilot_Q_P,
                 pilot_I_L, pilot_Q_L) = correValues[6:12]

                # --- Save and update variables for current correlation -------
                # Save remCodePhase for current correlation
                chResults.remCodePhase[loopCnt] = remCodePhase
                # Save remCarrPhase for current correlation
                chResults.remCarrPhase[loopCnt] = remCarrPhase
                # Remaining code phase for next tracking update
                remCodePhase = (blksize * codePhaseStep + remCodePhase -
                                settings.codeLength)
                # Remaining carrier phase for next tracking update
                remCarrPhase = np.fmod(carrPhaseStep * blksize + remCarrPhase, 2 * np.pi)

                #%% Find PLL error and update carrier NCO ==================
                # Data-channel carrier-loop discriminator.
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)

                if settings.pilotTRKflag == 1:
                    # The B2a pilot carrier is pi/2 ahead of the data carrier;
                    # rotate it back before forming the pilot discriminator.
                    QI = ((pilot_I_P + 1j * pilot_Q_P)
                          * np.exp(-1j * np.pi / 2))
                    carrErrorQ = (np.arctan(np.imag(QI) / np.real(QI))
                                  / (2.0 * np.pi))
                    carrError = (carrError + carrErrorQ) / 2

                # Second-order carrier-loop filter and NCO command.
                carrNco = (oldCarrNco
                           + (tau2carr / tau1carr)
                           * (carrError - oldCarrError)
                           + carrError * (PDIcarr / tau1carr))
                oldCarrNco = carrNco
                oldCarrError = carrError

                # Save carrier frequency for current correlation
                chResults.carrFreq[loopCnt] = carrFreq
                # Modify carrier freq based on NCO command
                carrFreq = carrFreqBasis + carrNco

                #%% Find DLL error and update code NCO =====================
                E = np.sqrt(I_E ** 2 + Q_E ** 2)
                L = np.sqrt(I_L ** 2 + Q_L ** 2)
                codeError = (E-L) / (E+L)
                if settings.pilotTRKflag == 1:
                    pilotE = np.sqrt(pilot_I_E**2 + pilot_Q_E**2)
                    pilotL = np.sqrt(pilot_I_L**2 + pilot_Q_L**2)
                    codeErrorQ = (pilotE - pilotL) / (pilotE + pilotL)
                    codeError = (codeError + codeErrorQ) / 2
                # Implement code loop filter and generate NCO command
                codeNco = oldCodeNco + (tau2code/tau1code) *   \
                    (codeError - oldCodeError) + codeError * (PDIcode/tau1code)
                oldCodeNco   = codeNco
                oldCodeError = codeError

                # Save code frequency for current correlation
                chResults.codeFreq[loopCnt] = codeFreq
                # Modify code freq based on NCO command
                codeFreq = codeFreqBasis - codeNco

                #%% Record measures for postprocessing =====================
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

                #%% C/N0 calculation =======================================
                if (loopCnt+1)% settings.CNoVSMinterval == 0:
                    CNoValue, pllDetector = calcCNoPld(
                        chResults, settings, loopCnt)
                    cnoCnt = (loopCnt + 1) // settings.CNoVSMinterval - 1
                    chResults.DataCNo[cnoCnt] = (0.5 * CNoValue[0]
                                                 + 0.5 * tempCNoValue[0])
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                    chResults.PilotCNo[cnoCnt] = (0.5 * CNoValue[1]
                                                  + 0.5 * tempCNoValue[1])
                    chResults.B2a_CNo[cnoCnt] = (0.5 * CNoValue[2]
                                                 + 0.5 * tempCNoValue[2])
                    chResults.PilotPLD[cnoCnt] = pllDetector[1]
                tempCNoValue = CNoValue.copy()

            # The requested loop completed. No signal-lock detector is applied.
            chResults.lockFlag = True
        waitbar.close()
        fid.close()
        # Results remain available in self.trackResults.
        return

    #%% Channel-parallel tracking
    def trkChannelsParallel(self, acqResults, corrEngine):
        """Perform BDS-3 B2a tracking for all active channels.

        The channels are processed using channel-parallel tracking mode.

        Args
        ----
            acqResults - acquisition.AcqEngine
                       PRNs, carrier frequencies, and code phases of all
                       satellites to be tracked, prepared from acquisition
                       results.
            corrEngine - callable or object
                       Correlator used for channel-parallel tracking.
        Returns
        -------
            None
                Tracking results are stored in ``trackResults``. They contain
                in-phase prompt outputs, absolute spreading-code starting
                positions, and other observations saved every millisecond.
                C/N0 is saved at the configured interval.
        """
        from correlator import generateB2aDataCode, generateB2aPilotCode

        settings = self.settings
        #--- Copy result variables for all channels -----------------------
        # Channel objects already contain the preallocated result arrays that
        # MATLAB constructs at the beginning of trkChannelsParallel.m.
        trkResults = self.trackResults
        # Number of active channels to be tracked.
        channelCnt = int(min(sum(acqResults.acqFlag), settings.numberOfChannels))

        if channelCnt == 0:
            return

        #%% Construct variables for tracking loops ========================
        # Previous DLL filter output (code-NCO command) [chips/s].
        oldCodeNco = np.zeros(channelCnt)
        # Previous DLL discriminator output.
        oldCodeError = np.zeros(channelCnt)
        # Second-order carrier-loop states.
        oldCarrNco = np.zeros(channelCnt)
        oldCarrError = np.zeros(channelCnt)
        # Code-NCO frequencies initialized from acquisition [chips/s].
        codeFreq = np.asarray(acqResults.codeFreq[:channelCnt],
                              dtype=np.float64).copy()
        # Fixed acquired code-frequency bases [chips/s].
        codeFreqBasis = codeFreq.copy()
        # Residual code phase [chips].
        remCodePhase = np.zeros(channelCnt)
        # Residual carrier phase [rad].
        remCarrPhase = np.zeros(channelCnt)
        # Carrier-NCO frequencies used for the current correlation [Hz].
        carrFreq = np.asarray(acqResults.carrFreq[:channelCnt],
                              dtype=np.float64).copy()
        # Acquired carrier-frequency bases [Hz].
        carrFreqBasis = carrFreq.copy()
        # Absolute zero-based sample of the next B2a-code start in the record.
        absoluteSample = (int(settings.skipNumberOfSamples) +
                 np.asarray(acqResults.codePhase[:channelCnt], dtype=np.int64))
        # Latest and previous data, pilot and combined C/N0 values.
        CNo = np.zeros((3, channelCnt))
        tempCNoValue = np.zeros((3, channelCnt))

        # Generate guarded data/pilot code tables for all active channels.
        branchCodeLen = settings.codeLength + 2
        B2aCodeTable = np.empty((channelCnt, 2 * branchCodeLen),
                                dtype=np.float64)

        for channelNr in range(channelCnt):
            # Result container and PRN for the current active channel.
            chResults = trkResults[channelNr]
            chResults.PRN = int(acqResults.PRN[channelNr])
            # Generate guarded B2a data and pilot primary-code branches.
            B2aCodeD = generateB2aDataCode(settings, chResults.PRN)
            B2aCodeP = generateB2aPilotCode(settings, chResults.PRN)
            B2aCodeTable[channelNr] = np.concatenate((
                [B2aCodeD[-1]], B2aCodeD, [B2aCodeD[0]],
                [B2aCodeP[-1]], B2aCodeP, [B2aCodeP[0]]))

        if settings.correlatorType == 1:
            # SIMD shares an int32 guarded-code table across all channels.
            B2aCodeTable = np.asarray(B2aCodeTable, dtype=np.int32)
        elif settings.correlatorType == 2:
            # GPU shares an int8 guarded-code table across all channels.
            B2aCodeTable = np.asarray(B2aCodeTable, dtype=np.int8)

        #%% Initialize tracking variables =================================
        # Number of coherent tracking epochs processed for all active channels.
        numToProcess = trkResults[0].I_P.size
        # DLL and PLL coherent integration intervals [s].
        PDIcode = PDIcarr = settings.intTime
        #--- DLL variables ------------------------------------------------
        # DLL loop-filter coefficients.
        tau1code, tau2code = calcLoopCoef(
            settings.dllNoiseBandwidth, settings.dllDampingRatio, 1.0)
        #--- PLL variables ------------------------------------------------
        # Second-order carrier-loop filter coefficients.
        tau1carr, tau2carr = calcLoopCoef(
            settings.pllNoiseBandwidth, settings.pllDampingRatio, 1.0)

        # Update the GUI less often for the faster compiled correlators.
        if settings.correlatorType == 0:
            barUprate = 50
        else:
            barUprate = 500
            outputType = np.int16

        #%% Initialize waitbar and variables for data reading =============
        #--- Variables for IF-signal reading ------------------------------
        # Stored scalars per logical sample: one real value or interleaved I/Q.
        dataAdaptCoeff = settings.dataAdaptCoeff
        # NumPy sample type, used to convert logical samples to byte offsets.
        sampleType = np.dtype(settings.dataType)
        # Number of logical samples per B2a-code period and per second.
        samplesPerCode = settings.samplesPerCode
        samplesPerSec = samplesPerCode * 1000
        # Inclusive absolute ending sample of the current IF-data block.
        blkEndIdx = -1
        # Absolute starting sample of the current IF-data block.
        blkStartIdx = 0

        #--- Initialize waitbar -------------------------------------------
        # Open the IF record and create the channel-parallel progress display.
        fid = open(settings.fileName, "rb")
        waitbar = WaitBar(numToProcess)

        #%% Tracking processing for all channels ==========================
        # For every requested code period:
        # 1. Update the GUI.
        # 2. Read a new shared IF-data block when required.
        # 3. Correlate all active channels.
        # 4. Update the tracking loops for all channels.
        # 5. Periodically estimate C/N0.
        for loopCnt in range(numToProcess):
            #%% GUI update ================================================
            # Update the GUI periodically without repainting every millisecond.
            if ((loopCnt + 1) % barUprate == 0 or
                    loopCnt + 1 == numToProcess):
                # Progress text reports the first active channel and C/N0.
                totalMs = round(numToProcess * settings.intTime * 1000)
                trackingStatus = (
                    f'Tracking {channelCnt} PRNs\n'
                    f'1st PRN: {trkResults[0].PRN}\n'
                    f'{round((loopCnt + 1) * settings.intTime * 1000)}'
                    f'/{totalMs} ms, '
                    f'1st data C/N0 {CNo[0, 0]:.1f} dB-Hz; '
                    f'pilot C/N0 {CNo[1, 0]:.1f} dB-Hz')
                if not waitbar.update(loopCnt + 1, trackingStatus):
                    # Closing the progress display cancels tracking and exits
                    # the current processing run.
                    print('Progress bar closed, exiting...')
                    waitbar.close()
                    fid.close()
                    return

            #%% Read next block of data ====================================
            # Code phase increment [chips/sample] from each current code NCO.
            codePhaseStep = codeFreq / settings.samplingFreq
            # Whole-sample length of the next code period for every channel.
            chSampSize = np.ceil(
                (settings.codeLength - remCodePhase) /
                codePhaseStep).astype(np.int32)

            # Flag passed to compiled correlators when rawSignal is refreshed.
            isDataRead = 0
            # Latest absolute sample required by any channel in this epoch.
            lastSample = np.max(absoluteSample + chSampSize - 1)
            # Read another shared one-second block only when a channel window
            # extends beyond the current buffer.
            if blkEndIdx < lastSample:
                isDataRead = 1
                # Start the new shared block at the earliest active channel.
                blkStartIdx = int(np.min(absoluteSample))
                # Convert its logical-sample position to a file byte offset.
                fid.seek(dataAdaptCoeff * blkStartIdx * sampleType.itemsize)
                # Read the shared one-second IF-data block.
                if settings.correlatorType == 0:
                    # NumPy reference-correlator input.
                    rawSignal = np.fromfile(fid, settings.dataType,
                                            dataAdaptCoeff * samplesPerSec)
                else:
                    # SIMD/GPU backends consume int16 input samples.
                    rawSignal0 = np.fromfile(fid, settings.dataType,
                                             dataAdaptCoeff * samplesPerSec)
                    rawSignal = rawSignal0.astype(outputType, copy=False)

                # A short read means the requested tracking data is unavailable.
                if rawSignal.size != dataAdaptCoeff * samplesPerSec:
                    print('Not able to read the specified number of samples '
                          'for tracking, exiting!')
                    fid.close()
                    waitbar.close()
                    return

                # Inclusive absolute ending sample of the refreshed buffer.
                blkEndIdx = blkStartIdx + samplesPerSec - 1

            #%% Correlator implementation =================================
            # Carrier phase increment in radians/sample for each carrier NCO.
            carrPhaseStep = (carrFreq * 2.0 * np.pi / settings.samplingFreq)
            # Zero-based start of each channel epoch within rawSignal.
            startIdx = np.asarray(absoluteSample - blkStartIdx, dtype=np.int32)

            # All backends return a 12 x channelCnt matrix containing the
            # data and pilot Early/Prompt/Late I/Q values.
            if settings.correlatorType == 0:
                # NumPy reference correlator.
                correValues = corrEngine(settings, rawSignal,
                      B2aCodeTable, remCarrPhase, carrPhaseStep, remCodePhase,
                      codePhaseStep, startIdx, chSampSize, isDataRead)
            elif settings.correlatorType == 1:
                # CPU SIMD correlator.
                correValues = corrEngine.corrEngine(settings, rawSignal,
                    B2aCodeTable, remCarrPhase, carrPhaseStep, remCodePhase,
                    codePhaseStep, startIdx, chSampSize, isDataRead)
            elif settings.correlatorType == 2:
                # GPU CUDA correlator.
                correValues = corrEngine.corrEngine(settings, rawSignal,
                    B2aCodeTable, remCarrPhase, carrPhaseStep, remCodePhase,
                    codePhaseStep, startIdx, chSampSize, isDataRead)

            #%% Update tracking-loop variables for all channels ===========
            for channelNr in range(channelCnt):
                # Mutable tracking-result container for the current channel.
                chResults = trkResults[channelNr]
                # Extract Early, Prompt, and Late I/Q correlator outputs.
                I_E, Q_E, I_P, Q_P, I_L, Q_L = correValues[:6, channelNr]
                (pilot_I_E, pilot_Q_E, pilot_I_P, pilot_Q_P,
                 pilot_I_L, pilot_Q_L) = correValues[6:12, channelNr]

                # Save the phases used for this correlation, then advance the
                # residual phases to the next code epoch.
                chResults.remCodePhase[loopCnt] = remCodePhase[channelNr]
                chResults.remCarrPhase[loopCnt] = remCarrPhase[channelNr]

                # Residual code phase for the next tracking update [chips].
                remCodePhase[channelNr] = (
                    chSampSize[channelNr] * codePhaseStep[channelNr] +
                    remCodePhase[channelNr] - settings.codeLength)

                # Residual carrier phase for the next update, wrapped to
                # (-2*pi, 2*pi) [rad], retaining the dividend sign.
                remCarrPhase[channelNr] = np.fmod(
                    carrPhaseStep[channelNr] * chSampSize[channelNr] +
                    remCarrPhase[channelNr], 2.0 * np.pi)

                #%% Find PLL error and update carrier NCO ------------------
                # PLL phase-discriminator output [cycles].
                carrError = np.arctan(Q_P / I_P) / (2.0 * np.pi)
                if settings.pilotTRKflag == 1:
                    QI = ((pilot_I_P + 1j * pilot_Q_P)
                          * np.exp(-1j * np.pi / 2))
                    carrErrorQ = (np.arctan(np.imag(QI) / np.real(QI))
                                  / (2.0 * np.pi))
                    carrError = (carrError + carrErrorQ) / 2
                # Second-order carrier-loop filter and NCO correction [Hz].
                carrNco = (oldCarrNco[channelNr]
                           + (tau2carr / tau1carr)
                           * (carrError - oldCarrError[channelNr])
                           + carrError * (PDIcarr / tau1carr))
                oldCarrNco[channelNr] = carrNco
                oldCarrError[channelNr] = carrError
                # Save the carrier frequency used for this correlation.
                chResults.carrFreq[loopCnt] = carrFreq[channelNr]
                # Apply the carrier-NCO command for the next correlation.
                carrFreq[channelNr] = carrFreqBasis[channelNr] + carrNco

                #%% Find DLL error and update code NCO ---------------------
                # Early and Late correlation magnitudes.
                E = np.sqrt(I_E ** 2 + Q_E ** 2)
                L = np.sqrt(I_L ** 2 + Q_L ** 2)
                # Normalized Early-minus-Late DLL discriminator output.
                codeError = (E - L) / (E + L)
                if settings.pilotTRKflag == 1:
                    pilotE = np.sqrt(pilot_I_E**2 + pilot_Q_E**2)
                    pilotL = np.sqrt(pilot_I_L**2 + pilot_Q_L**2)
                    codeErrorQ = (pilotE - pilotL) / (pilotE + pilotL)
                    codeError = (codeError + codeErrorQ) / 2
                # DLL filter output and code-NCO correction [chips/s].
                codeNco = (oldCodeNco[channelNr] + (tau2code / tau1code) *
                           (codeError - oldCodeError[channelNr]) +
                           codeError * (PDIcode / tau1code))
                # Preserve the filter state for the next tracking update.
                oldCodeNco[channelNr] = codeNco
                oldCodeError[channelNr] = codeError
                # Save the code frequency used for this correlation.
                chResults.codeFreq[loopCnt] = codeFreq[channelNr]
                # Apply the code-NCO command for the next correlation.
                codeFreq[channelNr] = codeFreqBasis[channelNr] - codeNco

                #%% Record measures for postprocessing --------------------
                # Record the current epoch and advance its absolute sample
                # position for the next millisecond.
                chResults.absoluteSample[loopCnt] = absoluteSample[channelNr]
                absoluteSample[channelNr] += chSampSize[channelNr]
                # Save discriminator outputs and filtered NCO commands.
                chResults.dllDiscr[loopCnt] = codeError
                chResults.dllDiscrFilt[loopCnt] = codeNco
                chResults.pllDiscr[loopCnt] = carrError
                chResults.pllDiscrFilt[loopCnt] = carrNco
                # Save Early, Prompt, and Late correlator outputs.
                chResults.I_E[loopCnt] = I_E
                chResults.I_P[loopCnt] = I_P
                chResults.I_L[loopCnt] = I_L
                chResults.Q_E[loopCnt] = Q_E
                chResults.Q_P[loopCnt] = Q_P
                chResults.Q_L[loopCnt] = Q_L
                chResults.Pilot_I_P[loopCnt] = pilot_I_P
                chResults.Pilot_Q_P[loopCnt] = pilot_Q_P

                #%% C/N0 calculation ======================================
                # Periodic variance-summing C/N0 estimate.
                if (loopCnt + 1) % settings.CNoVSMinterval == 0:
                    CNoValue, pllDetector = calcCNoPld(
                        chResults, settings, loopCnt)
                    cnoCnt = (loopCnt + 1) // settings.CNoVSMinterval - 1
                    CNo[:, channelNr] = CNoValue
                    chResults.DataCNo[cnoCnt] = (0.5 * CNoValue[0]
                        + 0.5 * tempCNoValue[0, channelNr])
                    chResults.DataPLD[cnoCnt] = pllDetector[0]
                    chResults.PilotCNo[cnoCnt] = (0.5 * CNoValue[1]
                        + 0.5 * tempCNoValue[1, channelNr])
                    chResults.B2a_CNo[cnoCnt] = (0.5 * CNoValue[2]
                        + 0.5 * tempCNoValue[2, channelNr])
                    chResults.PilotPLD[cnoCnt] = pllDetector[1]
                    tempCNoValue[:, channelNr] = CNoValue

        # A completed loop marks processing completion, not verified lock.
        for channelNr in range(channelCnt):
            trkResults[channelNr].lockFlag = True
        waitbar.close()
        fid.close()
        return

    #%% Plot tracking results
    def plotTracking(self):
        """Plot the tracking results for all completed channels.

        Returns
        -------
            None
                Tracking-result figures are displayed.
        """
        import matplotlib.pyplot as plt

        settings = self.settings

        #--- For all completed channels -----------------------------------
        for channelNr, chResults in enumerate(self.trackResults):
            if not chResults.lockFlag:
                continue

            #%% Select or create and clear the figure ======================
            # Offset receiver-owned figure numbers to avoid replacing figures
            # created elsewhere in the receiver.
            fig = plt.figure(channelNr + 200, figsize=(10, 7), clear=True)
            fig.suptitle(
                f'Channel {channelNr + 1} - PRN {chResults.PRN}')

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

            timeAxisInSeconds = (
                (np.arange(chResults.I_P.size) + 1) * settings.intTime)

            #%% Plot all figures ==========================================
            #--- Prompt correlator I/Q scatter plot -----------------------
            h11.plot(chResults.I_P, chResults.Q_P, '.', markersize=2)
            h11.axis('equal')
            h11.set(title='Discrete-Time Scatter Plot',
                    xlabel='I prompt', ylabel='Q prompt')

            #--- Navigation-message bits ----------------------------------
            h12.plot(timeAxisInSeconds, chResults.I_P)
            h12.set(title='Bits of the navigation message')

            #--- Early, Prompt and Late correlation magnitudes ------------
            correlationResults = np.column_stack((
                np.hypot(chResults.I_E, chResults.Q_E),
                np.hypot(chResults.I_P, chResults.Q_P),
                np.hypot(chResults.I_L, chResults.Q_L)))
            h22.plot(timeAxisInSeconds, correlationResults)
            h22.set(title='Correlation results', ylabel='Magnitude')
            h22.legend(('Early', 'Prompt', 'Late'))

            #--- Raw and filtered PLL/DLL discriminator outputs -----------
            discriminatorPlots = (
                (h21, chResults.pllDiscr, 'r',
                 'Raw PLL discriminator', 'Cycles'),
                (h31, chResults.pllDiscrFilt, 'b',
                 'Filtered PLL discriminator', 'Hz'),
                (h32, chResults.dllDiscr, 'r',
                 'Raw DLL discriminator', 'Normalized error'),
                (h33, chResults.dllDiscrFilt, 'b',
                 'Filtered DLL discriminator', 'Chips/s'))
            for axes, values, color, title, ylabel in discriminatorPlots:
                axes.plot(timeAxisInSeconds, values, color)
                axes.set(title=title, ylabel=ylabel)

            # Apply the common grid and time-axis formatting ------------------
            timeAxes = (h12, h21, h22, h31, h32, h33)
            for axes in (h11,) + timeAxes:
                axes.grid(True)
            for axes in timeAxes:
                axes.set_xlabel('Time (s)')
                axes.margins(x=0)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            #--- Periodic data, pilot and combined B2a C/N0 estimates -----
            CNoValue = np.column_stack((chResults.DataCNo,
                                        chResults.PilotCNo,
                                        chResults.B2a_CNo))
            CNoInterval = settings.CNoVSMinterval * settings.intTime
            CNoTime = np.arange(1, CNoValue.shape[0] + 1) * CNoInterval

            CNoFig = plt.figure(
                channelNr + 300, figsize=(8, 4), clear=True)
            CNoAxes = CNoFig.add_subplot(1, 1, 1)
            CNoAxes.plot(CNoTime, CNoValue, '.-', markersize=4)
            CNoAxes.legend(('Data', 'Pilot', 'Combined B2a'))
            CNoAxes.set(
                title=f'C/N0 estimation ({CNoInterval:g} s interval)',
                xlabel='Time (s)', ylabel='C/N0 (dB-Hz)')
            CNoAxes.grid(True)
            CNoAxes.margins(x=0)
            CNoFig.tight_layout()

        plt.show()

    #%% Run tracking
    def trackingRun(self, acqResults):
        """Run BDS-3 B2a code and carrier tracking.

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
