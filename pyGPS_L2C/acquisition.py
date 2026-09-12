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

acquisition.py - Module Description
-----------------------------------
GPS L2C acquisition on CPU or GPU.

"""
import numpy as np

#%% Acquisition results
class AcqEngine():
    """Perform GPS L2C acquisition and store its results.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize result arrays through the largest configured GPS L2C PRN.

        Args
        ----
            settings   - object
                       Receiver settings used by the acquisition methods.

        Returns
        -------
            None
                Acquisition-result arrays are stored in this object.
        """
        #%% Initialize acquisition results =================================
        # Allocate through the largest configured PRN for direct PRN - 1 indexing.
        svNum = int(np.max(settings.acqSatelliteList))
        # Refined carrier frequency [Hz], shape = (svNum,); inf if unavailable.
        self.carrFreq = np.full(svNum, np.inf)
        # Zero-based L2C-code phase [samples], shape = (svNum,); inf if unavailable.
        self.codePhase = np.full(svNum, np.inf)
        # Correlation peak ratio [dimensionless], shape = (svNum,); zero if unsearched.
        self.peakMetric = np.zeros(svNum)
        # PRN number, shape = (svNum,); zero denotes an unsearched result slot.
        self.PRN = np.zeros(svNum).astype(int)
        # Initial code-NCO frequency [Hz], shape = (svNum,); zero if unsearched.
        self.codeFreq = np.zeros(svNum)
        # Acquisition decision, shape = (svNum,); False means not acquired.
        self.acqFlag = np.full(svNum, False)
        # One-based CM-period index in the 75-period CL code, shape = (svNum,).
        self.CLCodePhase = np.zeros(svNum, dtype=np.int32)
        # Retain the caller's settings object by reference.
        self._settings = settings

    @property
    def settings(self):
        """Return the receiver-settings object retained by the engine.

        Returns
        -------
            settings   - object
                       Receiver-settings object retained by the engine.
        """
        return self._settings

    #%% Acquisition engine using CPU
    def acqProcessCPU(self):
        """Perform cold-start GPS L2C acquisition on the raw-signal file.

        The function searches for GPS L2C signals of all satellites listed
        in ``settings.acqSatelliteList`` and saves the code phase and frequency
        of each detected signal in the acquisition results.

        Returns
        -------
            None
                Results are stored in this object. ``carrFreq`` is
                ``numpy.inf`` if a signal is not detected for a PRN.
                ``codePhase`` is a zero-based sample offset.
        """
        import correlator
        from numpy import fft

        settings = self._settings

        #%% Read data for acquisition ======================================
        # Find number of samples per L2C spreading-code period.
        samplesPerCode = settings.samplesPerCode
        # Number of samples per L2C code chip.
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)

        with open(settings.fileName, "rb") as fid:
            # Move the starting point of processing. Can be used to start the
            # signal processing at any point in the data record (e.g. good for
            # long records or for signal processing in blocks).
            if settings.dataType == "int16":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == "int8":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)

            # Read 12 CM periods; fine carrier acquisition uses 10 periods.
            longSignal = np.fromfile(fid,settings.dataType,
                         settings.dataAdaptCoeff * 12 * samplesPerCode)

        # Remove the scalar-stream mean and convert interleaved I/Q samples
        # into a complex signal, as in acquisitionCPU.m.
        if settings.fileType == 2:
            longSignal = longSignal - np.mean(longSignal)
            longSignal = longSignal[::2] + 1j * longSignal[1::2]
            longSignal = np.asarray(longSignal, dtype=np.complex128)
        else:
            longSignal = np.asarray(longSignal, dtype=np.float64)

        #%% Acquisition initialization =====================================
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period.
        ts = 1 / settings.samplingFreq
        # Carrier phase time base for 10 CM code periods (200 ms).
        phasePoints = (
            np.arange(samplesPerCode * 10, dtype=np.float64) * 2 * np.pi * ts)

        # Number of frequency bins in the specified search band.
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        #--- Variables for fine acquisition -------------------------------
        # Code resampling index used by CM fine acquisition and CL search.
        # It maps each input sample to the corresponding local code chip.
        codeValueIndex = (np.arange(10 * samplesPerCode, dtype=np.float64) * ts
                          * settings.codeFreqBasis).astype(np.int32)
        codeValueIndex %= settings.codeLength

        # Double-precision phase coordinates for the continuous 200 ms
        # carrier used by the fine-frequency search.
        finePhasePoints = phasePoints

        # Reuse the zero-padded two-period CM buffer for every PRN.
        localCmCode = np.zeros(samplesPerCode * 2, dtype=np.float64)

        # Perform search for all listed PRN numbers.
        print("   (", end="", flush=True)

        #%% Coarse acquisition =============================================
        for PRN in settings.acqSatelliteList:
            # Generate the sampled CM data code for this PRN.
            cmCodesTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "CM")
            # Copy one code period into the reusable zero-padded replica.
            localCmCode[:samplesPerCode] = cmCodesTable[:samplesPerCode]

            #--- Perform DFT of the CM code -------------------------------
            cmCodeFreqDom = np.conjugate(fft.fft(localCmCode))

            # Initialize the global peak retained across frequency bins.
            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            for freqBinIndex in range(numberOfFreqBins):
                #--- Generate carrier-wave frequency grid -----------------
                # Compute the carrier frequency represented by this search bin.
                coarseFreqBin = (settings.IF - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                # Generate local sine and cosine for this frequency bin.
                sigCarr = np.exp(
                    -1j * coarseFreqBin * phasePoints[:2 * samplesPerCode])
                signal = longSignal[:2 * samplesPerCode]
                IQfreqDom = fft.fft(sigCarr * signal)
                results = np.abs(fft.ifft(IQfreqDom * cmCodeFreqDom))

                #--- Look for correlation peaks ---------------------------
                # Find the maximum correlation peak and its code phase.
                maxIndexTemp = int(np.argmax(results))
                maxPeakTemp = float(results[maxIndexTemp])
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            # Move a peak in the padded second period to the corresponding
            # code phase in the first CM period.
            if codePhaseMax >= samplesPerCode:
                codePhaseMax -= samplesPerCode

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Find the L2C code-phase exclusion range around the global peak.
            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip

            #--- Correct the exclusion range at array boundaries ----------
            # Wrap the retained range when the one-chip exclusion interval
            # crosses either boundary of the correlation array.
            if excludeIndex1 < 0:
                codePhaseRange = np.arange(excludeIndex2,
                                           samplesPerCode + excludeIndex1 + 1,)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = np.arange(excludeIndex2 - samplesPerCode,
                                           excludeIndex1 + 1,)
            else:
                codePhaseRange = np.hstack((np.arange(0, excludeIndex1 + 1),
                                    np.arange(excludeIndex2, samplesPerCode),))

            #--- Find the second-highest correlation peak -----------------
            # Find the largest correlation value outside the exclusion range.
            secondPeak = float(np.max(results[codePhaseRange]))
            # Save code-phase acquisition result.
            self.codePhase[PRN - 1] = codePhaseMax
            # Store the GLRT statistic.
            self.peakMetric[PRN - 1] = peakMax / secondPeak
            # Store the searched PRN.
            self.PRN[PRN - 1] = PRN

            # Move to the previous code start if fine acquisition would
            # exceed the available signal data.
            if codePhaseMax + 10 * samplesPerCode > longSignal.size:
                codePhaseMax -= samplesPerCode

            # A GLRT statistic above the threshold indicates that the L2C
            # signal has been found.
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                #%% Fine resolution frequency search ======================
                self.acqFlag[PRN - 1] = True
                # Indicate PRN number of the detected signal.
                print(f"{PRN:02d} ", end="", flush=True)

                #--- Prepare 200 ms code, carrier and input signals --------
                # Generate and resample the CM data code over 200 ms.
                L2CmCode = correlator.generateCMcode(settings, PRN)
                L2CmCode200ms = L2CmCode[codeValueIndex]
                sig200ms = longSignal[
                    codePhaseMax:codePhaseMax + 10 * samplesPerCode]
                # Coarse-frequency local carrier before the fine estimate.
                localCarr200cm = np.exp(-1j * freqMax * finePhasePoints)

                #--- Integration for each 1 ms segment over 200 ms ---------
                # Sum 200 adjacent 1 ms intervals after CM/carrier wipeoff.
                basebandSig = sig200ms * L2CmCode200ms * localCarr200cm
                samplesPer1ms = settings.samplesPerCode // 20
                sumPerCode = basebandSig[:200 * samplesPer1ms].reshape(
                    (200, samplesPer1ms)).sum(axis=1)

                #--- Find the fine carrier frequency ----------------------
                # Find the strongest spectral component after sign removal.
                maxPowerIndex = int(np.argmax(
                    np.abs(fft.fft(sumPerCode ** 2))))
                # Convert FFT-bin phase to carrier frequency correction. The
                # division by two compensates for the squaring operation.
                shiftAngle = np.angle(np.exp(-2 * np.pi * 1j * maxPowerIndex / 200)) / 2
                self.carrFreq[PRN - 1] = freqMax - shiftAngle / 0.001 / 2 / np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN - 1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN - 1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)

                # If IF is zero, change 0 Hz to 1 Hz to allow processing.
                if self.carrFreq[PRN - 1] == 0:
                    self.carrFreq[PRN - 1] = 1
                self.codePhase[PRN - 1] = codePhaseMax

                #--- Find the L2 CL code phase ----------------------------
                sigCarr = np.exp(
                    -1j * self.carrFreq[PRN - 1]
                    * phasePoints[:samplesPerCode])
                L2ClCode = correlator.generateCLcode(settings, PRN)
                powerArray = np.zeros(75)
                cmCodeIndex = codeValueIndex[:samplesPerCode]
                for index in range(75):
                    L2ClCodeSample = L2ClCode[
                        cmCodeIndex + settings.codeLength * index]
                    powerArray[index] = np.abs(np.sum(
                        sig200ms[:samplesPerCode] * L2ClCodeSample * sigCarr))
                self.CLCodePhase[PRN - 1] = int(np.argmax(powerArray)) + 1
            else:
                #--- No signal with this PRN -------------------------------
                print(". ", end="", flush=True)

        #%% Acquisition is over ===========================================
        print(")")

    #%% Acquisition engine using GPU
    def acqProcessGPU(self):
        """Perform cold-start GPS L2C acquisition on the raw-signal file.

        The function searches for GPS L2C signals of all satellites listed
        in ``settings.acqSatelliteList`` and saves the code phase and frequency
        of each detected signal in the acquisition results.

        Returns
        -------
            None
                Results are stored in this object. ``carrFreq`` is
                ``numpy.inf`` if a signal is not detected for a PRN.
                ``codePhase`` is a zero-based sample offset.
        """
        import correlator
        import cupy as xp
        from cupyx.scipy import fft as fftBackend

        settings = self._settings
        # Match the GPU acquisition architecture: bulk array processing uses
        # single precision on the device.
        realDataType = xp.float32
        complexDataType = xp.complex64

        #%% Read data for acquisition ======================================
        # Read and preprocess acquisition data on the host.
        # Find number of samples per L2C spreading-code period.
        samplesPerCode = settings.samplesPerCode
        # Number of samples per L2C code chip.
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)

        with open(settings.fileName, "rb") as fid:
            # Move to the configured processing start in the raw recording.
            if settings.dataType == "int16":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == "int8":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)

            # np.fromfile performs the disk read on the host. Fine carrier
            # acquisition uses 10 of the 12 CM periods read here.
            longSignal = np.fromfile(fid,settings.dataType,
                             settings.dataAdaptCoeff * 12 * samplesPerCode, )

        # Remove the scalar-stream mean, convert interleaved I/Q samples into
        # a complex signal, and pass the result to GPU memory.
        if settings.fileType == 2:
            longSignal = longSignal - np.mean(longSignal)
            longSignal = longSignal[::2] + 1j * longSignal[1::2]
            longSignal = xp.asarray(longSignal, dtype=complexDataType)
        else:
            longSignal = xp.asarray(longSignal, dtype=realDataType)

        #%% Acquisition initialization =====================================
        # GPU search coordinates and reusable buffers.
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period.
        ts = 1 / settings.samplingFreq
        # Carrier phase time base for 10 CM periods (200 ms).
        phasePoints = xp.arange(samplesPerCode * 10, dtype=realDataType)
        phasePoints *= realDataType(2 * np.pi * ts)

        # Number of frequency bins in the specified search band.
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        #--- Variables for fine acquisition -------------------------------
        # Code resampling index used by CM fine acquisition and CL search.
        # Calculate it on the host in float64: float32 can change chip indices
        # close to integer boundaries.
        codeValueIndex = (np.arange(10 * samplesPerCode, dtype=np.float64) * ts
                          * settings.codeFreqBasis).astype(np.int32)
        codeValueIndex %= settings.codeLength
        codeValueIndex = xp.asarray(codeValueIndex)

        # Generate only one 1 ms carrier segment for fine acquisition. The
        # start phase of every segment is compensated separately in float64.
        samplesPer1ms = settings.samplesPerCode // 20
        finePhasePoints = xp.arange(samplesPer1ms, dtype=realDataType)
        finePhasePoints *= realDataType(2 * np.pi * ts)

        # Reuse the zero-padded two-period CM buffer for every PRN.
        localCmCode = xp.zeros(samplesPerCode * 2, dtype=realDataType)

        # Perform search for all listed PRN numbers.
        print("   (", end="", flush=True)

        #%% Coarse acquisition =============================================
        for PRN in settings.acqSatelliteList:
            # Generate and sample the CM data code for this PRN.
            cmCodesTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "CM")
            localCmCode[:samplesPerCode] = xp.asarray(
                cmCodesTable[:samplesPerCode], dtype=realDataType)

            #--- Perform DFT of the CM code -------------------------------
            cmCodeFreqDom = xp.conjugate(fftBackend.fft(localCmCode))

            # Initialize the global peak retained across frequency bins.
            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            for freqBinIndex in range(numberOfFreqBins):
                #--- Generate carrier-wave frequency grid -----------------
                # Compute the carrier frequency represented by this search bin.
                coarseFreqBin = (settings.IF  - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                
                # Generate local sine and cosine for this frequency bin.
                sigCarr = xp.exp(
                    -1j * realDataType(coarseFreqBin)
                    * phasePoints[:2 * samplesPerCode]
                ).astype(complexDataType, copy=False)
                signal = longSignal[:2 * samplesPerCode]
                IQfreqDom = fftBackend.fft(sigCarr * signal)
                results = xp.abs(fftBackend.ifft(IQfreqDom * cmCodeFreqDom))

                #--- Look for correlation peaks ---------------------------
                # Find the maximum correlation peak and its code phase.
                # Transfer the peak magnitude only; transfer its code index
                # only if this bin becomes the new global maximum.
                maxPeakTemp = xp.max(results).item()
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = xp.argmax(results).item()
                    freqMax = coarseFreqBin

            # Move a peak in the padded second period to the first CM period.
            if codePhaseMax >= samplesPerCode:
                codePhaseMax -= samplesPerCode

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Find the L2C code-phase exclusion range around the global peak.
            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip

            #--- Correct the exclusion range at array boundaries ----------
            # Wrap the retained range when the one-chip exclusion interval
            # crosses either boundary of the correlation array.
            if excludeIndex1 < 0:
                codePhaseRange = xp.arange(excludeIndex2,
                                           samplesPerCode + excludeIndex1 + 1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = xp.arange(excludeIndex2 - samplesPerCode,
                                           excludeIndex1 + 1 )
            else:
                codePhaseRange = xp.hstack((xp.arange(0, excludeIndex1 + 1),
                                    xp.arange(excludeIndex2, samplesPerCode)))

            #--- Find the second-highest correlation peak -----------------
            # The largest retained value is the secondary peak in the ratio.
            secondPeak = xp.max(results[codePhaseRange]).item()
            # Save code-phase acquisition result.
            self.codePhase[PRN - 1] = codePhaseMax
            # Store the GLRT statistic.
            self.peakMetric[PRN - 1] = peakMax / secondPeak
            # Store the searched PRN.
            self.PRN[PRN - 1] = PRN

            # Move to the previous code start if fine acquisition would
            # exceed the available signal data.
            if codePhaseMax + 10 * samplesPerCode > longSignal.size:
                codePhaseMax -= samplesPerCode

            # A GLRT statistic above the threshold indicates that the L2C
            # signal has been found.
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                #%% Fine resolution frequency search ======================
                self.acqFlag[PRN - 1] = True
                # Indicate PRN number of the detected signal.
                print(f"{PRN:02d} ", end="", flush=True)

                #--- Prepare 200 ms code, carrier and input signals --------
                # Generate and resample the CM data code over 200 ms.
                L2CmCode = xp.asarray(
                    correlator.generateCMcode(settings, PRN),
                    dtype=realDataType)
                L2CmCode200ms = L2CmCode[codeValueIndex]
                sig200ms = longSignal[
                    codePhaseMax:codePhaseMax + 10 * samplesPerCode]
                
                # Reusable 1 ms coarse-frequency local carrier.
                localCarr1ms = xp.exp(
                    -1j * realDataType(freqMax) * finePhasePoints
                ).astype(complexDataType, copy=False)

                #--- Integration for each 1 ms segment over 200 ms ---------
                # Compute every 1 ms start phase on the host in float64. The
                # reduced phase avoids the accuracy loss of a continuous
                # 200 ms float32 carrier time base.
                initialPhase = np.fmod(
                    2 * np.pi * freqMax * np.arange(200, dtype=np.float64) *
                    samplesPer1ms * ts, 2 * np.pi)
                initialCarr = xp.asarray(
                    np.exp(-1j * initialPhase), dtype=complexDataType)

                # Sum 200 adjacent 1 ms intervals after CM/carrier wipeoff.
                fineUsedSamples = 200 * samplesPer1ms
                sig200msFine = sig200ms[:fineUsedSamples].reshape(
                    (200, samplesPer1ms))
                L2CmCode200ms = L2CmCode200ms[:fineUsedSamples].reshape(
                    (200, samplesPer1ms))
                sumPerCode = xp.sum(
                    sig200msFine * L2CmCode200ms * localCarr1ms, axis=1)
                sumPerCode *= initialCarr

                #--- Find the fine carrier frequency ----------------------
                # Find the strongest spectral component after sign removal.
                maxPowerIndex = xp.argmax(
                    xp.abs(fftBackend.fft(sumPerCode ** 2))).item()
                
                # Convert FFT-bin phase to carrier frequency correction. The
                # division by two compensates for the squaring operation.
                shiftAngle = np.angle(np.exp(-2 * np.pi * 1j * maxPowerIndex / 200)) / 2
                self.carrFreq[PRN - 1] = freqMax - shiftAngle / 0.001 / 2 / np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN - 1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN - 1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)

                # If IF is zero, change 0 Hz to 1 Hz to allow processing.
                if self.carrFreq[PRN - 1] == 0:
                    self.carrFreq[PRN - 1] = 1
                self.codePhase[PRN - 1] = codePhaseMax

                #--- Find the L2 CL code phase ----------------------------
                sigCarr = xp.exp(
                    -1j * realDataType(self.carrFreq[PRN - 1])
                    * phasePoints[:samplesPerCode]
                ).astype(complexDataType, copy=False)
                L2ClCode = xp.asarray(
                    correlator.generateCLcode(settings, PRN),
                    dtype=realDataType)
                powerArray = xp.zeros(75, dtype=realDataType)
                cmCodeIndex = codeValueIndex[:samplesPerCode]
                for index in range(75):
                    L2ClCodeSample = L2ClCode[
                        cmCodeIndex + settings.codeLength * index]
                    powerArray[index] = xp.abs(xp.sum(
                        sig200ms[:samplesPerCode] * L2ClCodeSample * sigCarr))
                self.CLCodePhase[PRN - 1] = int(xp.argmax(powerArray).item()) + 1
            else:
                #--- No signal with this PRN -------------------------------
                print(". ", end="", flush=True)

        #%% Acquisition is over ===========================================
        print(")")

    #%% Plot acquisition results
    def plotAcq(self):
        """Plot a bar chart of the acquisition results.

        No bars are shown for satellites not included in the acquisition list.

        Returns
        -------
            None
                The acquisition figure is displayed.
        """
        import matplotlib.pyplot as plt
        #--- Mark acquired signals ----------------------------------------
        validPrn = self.PRN > 0
        acqMask = validPrn & (self.carrFreq != np.inf)
        notAcqMask = validPrn & (self.carrFreq == np.inf)
        #%% Plot all results ==============================================
        fig, ax = plt.subplots()
        ax.bar(self.PRN[acqMask], self.peakMetric[acqMask], color = 'b', width = 0.85)
        ax.bar(self.PRN[notAcqMask], self.peakMetric[notAcqMask], color = 'r', width = 0.85)
        ax.legend(labels=['Acquired signals', 'Not acquired signals'])
        ax.set_xlabel('PRN number (no bar - SV is not in the acquisition list)')
        ax.set_ylabel('Acquisition Metric')
        ax.set_title('Acquisition results')
        plt.grid(True)
        plt.xticks(np.sort(self.PRN[validPrn]))
        plt.tight_layout()
        plt.show()
        return

    #%% Prepare tracking-channel priority
    def preRun(self):
        """Prepare tracking-channel priority from the acquisition results.

        The acquired signals are sorted according to signal strength. This
        function can be modified to use another satellite-selection algorithm
        or to introduce acquired-signal property offsets for testing.

        Returns
        -------
            None
                Acquisition-result arrays are reordered in place.
        """
        #--- Sort peaks and retain their index information ----------------
        # Sort peaks to find the strongest signals and retain their indices.
        PRNindexes = self.peakMetric.argsort()[::-1]
        self.peakMetric = self.peakMetric.take(PRNindexes)
        self.carrFreq = self.carrFreq.take(PRNindexes)
        self.codePhase = self.codePhase.take(PRNindexes)
        self.PRN = self.PRN.take(PRNindexes)
        self.acqFlag = self.acqFlag.take(PRNindexes)
        self.codeFreq = self.codeFreq.take(PRNindexes)
        self.CLCodePhase = self.CLCodePhase.take(PRNindexes)

    #%% Show channel status
    def showChannelStatus(self):
        """Print the status of all channels in a table.

        Returns
        -------
            None
                The channel-status table is written to standard output.
        """
        settings = self._settings
        print('\n*=========*=====*===============*===========*=============*=========*')
        print('| Channel | PRN |   Frequency   |  Doppler  | Code Offset | Status  |')
        print('*=========*=====*===============*===========*=============*========*')
        for channelNr in range(settings.numberOfChannels):
            if self.acqFlag[channelNr]:
                print('|    %2d   | %3d |  %2.5e  |   %5.0f   |    %6d   |    %1s    |' % (
                    channelNr,
                    self.PRN[channelNr],
                    self.carrFreq[channelNr],
                    self.carrFreq[channelNr] - settings.IF,
                    self.codePhase[channelNr],
                    'T' ))
            else:
                print('|    %2d   | --- | ------------  |   -----   |    ------   |   Off   |' %(channelNr))

        print('*=========*=====*===============*===========*=============*=========*\n')

    #%% Run acquisition
    def run(self):
        """Run acquisition, prioritize results, print status, and plot them.

        Returns
        -------
            None
                Acquisition results are stored in this object.
        """
        #%% Acquisition ====================================================
        # Acquisition process using CPU or GPU.
        if self.settings.gpuACQflag:
            self.acqProcessGPU()
        else:
            self.acqProcessCPU()
        #%% Prepare for the run ===========================================
        # Reorder all result arrays into tracking-channel priority.
        self.preRun()
        # Show the acq results
        self.showChannelStatus()
        # Plot acquisition results
        self.plotAcq()
