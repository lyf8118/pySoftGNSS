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
BDS-3 B1C acquisition on CPU or GPU.

"""
import numpy as np

#%% Acquisition results
class AcqEngine():
    """Perform BDS-3 B1C acquisition and store its results.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize result arrays through the largest configured B1C PRN.

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
        # Allocate through the largest configured PRN because result arrays
        # are indexed by PRN - 1.
        svNum = int(np.max(settings.acqSatelliteList))
        # Refined carrier frequencies [Hz]; inf means no valid estimate.
        # Shape: (svNum,).
        self.carrFreq = np.full(svNum, np.inf)
        # Zero-based B1C-code phases [samples]; inf means unavailable.
        # Shape: (svNum,).
        self.codePhase = np.full(svNum, np.inf)
        # Correlation peak ratios; zero means the PRN has not been searched.
        # Shape: (svNum,).
        self.peakMetric = np.zeros(svNum)
        # PRN numbers; zero marks an unsearched result slot. Shape: (svNum,).
        self.PRN = np.zeros(svNum).astype(int)
        # Initial code-NCO frequencies [Hz]; zero means not initialized.
        # Shape: (svNum,).
        self.codeFreq = np.zeros(svNum)
        # Acquisition decisions; False means not acquired. Shape: (svNum,).
        self.acqFlag = np.full(svNum, False)
        # Receiver settings retained by reference for all acquisition methods.
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
        """Perform cold-start BDS-3 B1C acquisition on the raw-signal file.

        The function searches for BDS-3 B1C signals of all satellites listed
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
        # Find number of samples per B1C spreading-code period.
        samplesPerCode = settings.samplesPerCode
        # Number of samples per B1C code chip.
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)

        with open(settings.fileName, "rb") as fid:
            # Move the starting point of processing. Can be used to start the
            # signal processing at any point in the data record (e.g. good for
            # long records or for signal processing in blocks).
            if settings.dataType == "int16":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == "int8":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)

            # Read 21 code periods so that at least 20 complete B1C periods
            # are available for the fine acquisition.
            longSignal = np.fromfile(fid,settings.dataType,
                         settings.dataAdaptCoeff * 21 * samplesPerCode)

        # Convert interleaved I/Q samples into a complex signal. CPU
        # acquisition uses double precision for the subsequent FFT search.
        if settings.fileType == 2:
            longSignal = longSignal[::2] + 1j * longSignal[1::2]
            longSignal = np.asarray(longSignal, dtype=np.complex128)
        else:
            longSignal = np.asarray(longSignal, dtype=np.float64)

        #%% Acquisition initialization =====================================
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period.
        ts = 1 / settings.samplingFreq
        # The carrier phase vector spans two B1C code periods and is reused
        # for every coarse-frequency bin.
        phasePoints = np.arange(samplesPerCode * 2, dtype=np.float64) * 2 * np.pi * ts

        # Number of frequency bins in the specified search band.
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        #--- Variables for fine acquisition -------------------------------
        # Code resampling index used by B1C pilot-code fine acquisition.
        # It maps each input sample to the corresponding local code chip.
        codeValueIndex = (np.arange(20 * samplesPerCode, dtype=np.float64) * ts
                          * settings.codeFreqBasis * 12).astype(np.int32)
        codeValueIndex %= settings.codeLength * 12

        # Double-precision phase coordinates for the continuous 200 ms
        # carrier used by the fine-frequency search.
        finePhasePoints = (
            np.arange(20 * samplesPerCode, dtype=np.float64) * 2 * np.pi * ts)

        # Reuse the large 20 ms buffers for every PRN.
        localData = np.zeros(samplesPerCode * 2, dtype=np.float64)
        localPilot = np.zeros(samplesPerCode * 2, dtype=np.float64)

        # Perform search for all listed PRN numbers.
        print("   (", end="", flush=True)

        #%% Coarse acquisition =============================================
        for PRN in settings.acqSatelliteList:
            # Generate B1C data codes and sample them at the sampling freq.
            dataPriTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "data")
            # Copy one code period into the reusable zero-padded replica.
            localData[:samplesPerCode] = dataPriTable[:samplesPerCode]

            #--- Perform DFT of B1C data and pilot codes -------------------
            # Conjugated data-code spectrum for frequency-domain correlation.
            dataPriFreqDom = np.conjugate(fft.fft(localData))

            # Generate the pilot replica only when pilot-assisted acquisition
            # is enabled.
            if settings.pilotACQflag:
                pilotPriTable = correlator.codeSampling(
                    settings, PRN, samplesPerCode, "pilotBOC11")
                localPilot[:samplesPerCode] = pilotPriTable[:samplesPerCode]
                pilotPriFreqDom = np.conjugate(fft.fft(localPilot))

            # Initialize the global peak retained across frequency bins.
            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            for freqBinIndex in range(numberOfFreqBins):
                #--- Generate carrier-wave frequency grid -----------------
                # Compute the carrier frequency represented by this search bin.
                coarseFreqBin = (settings.IF - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                # Generate local sine and cosine for this frequency bin.
                sigCarr = np.exp(-1j * coarseFreqBin * phasePoints)
                # "Remove carrier" from the signal.
                # Convert the baseband signal to frequency domain.
                IQfreqDom = fft.fft(sigCarr * longSignal[:samplesPerCode * 2])
                # Multiplication in frequency domain is correlation in time.
                convCodeIQ = IQfreqDom * dataPriFreqDom
                # Perform inverse DFT and store correlation results.
                results = np.abs(fft.ifft(convCodeIQ))

                if settings.pilotACQflag:
                    # Pilot signal components; data and pilot powers are
                    # combined with the configured 1:3 weighting.
                    convCodeIQ = IQfreqDom * pilotPriFreqDom
                    results = (results + 3*np.abs(fft.ifft(convCodeIQ))) / 4

                #--- Look for correlation peaks ---------------------------
                # Find the maximum correlation peak and its code phase.
                maxIndexTemp = int(np.argmax(results))
                maxPeakTemp = float(results[maxIndexTemp])
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Find the B1C code-phase exclusion range around the global peak.
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
            if codePhaseMax + 20 * samplesPerCode > longSignal.size:
                codePhaseMax -= samplesPerCode

            # A GLRT statistic above the threshold indicates that the B1C
            # signal has been found.
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                #%% Fine resolution frequency search ======================
                self.acqFlag[PRN - 1] = True
                # Indicate PRN number of the detected signal.
                print(f"{PRN:02d} ", end="", flush=True)

                #--- Prepare 200 ms code, carrier and input signals --------
                # Use the same branch selected for coarse acquisition.
                if settings.pilotACQflag:
                    b1cCode = correlator.generatePilotBOC11(settings, PRN)
                else:
                    b1cCode = correlator.generateDataBOC11(settings, PRN)
                b1cCode200ms = b1cCode[codeValueIndex]
                # Extract 20 B1C periods (200 ms) from the detected code phase.
                sig200ms = longSignal[codePhaseMax : codePhaseMax + 20 * samplesPerCode]
                # Coarse-frequency local carrier before the fine estimate.
                localCarr200cm = np.exp(-1j * freqMax * finePhasePoints)

                #--- Integration for each 1 ms segment over 200 ms ---------
                # Wipe off the B1C pilot code and coarse carrier from the
                # incoming signal.
                basebandSig = sig200ms * b1cCode200ms * localCarr200cm
                # Sum each 1 ms segment. Squaring removes 180-degree sign
                # transitions before the FFT-based frequency estimate.
                sumPerCode = basebandSig.reshape((200, -1)).sum(axis=1)

                #--- Find the fine carrier frequency ----------------------
                # Find the strongest spectral component after sign removal.
                maxPowerIndex = int(np.argmax(np.abs(fft.fft(sumPerCode**2))))
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
            else:
                #--- No signal with this PRN -------------------------------
                print(". ", end="", flush=True)

        #%% Acquisition is over ===========================================
        print(")")

    #%% Acquisition engine using GPU
    def acqProcessGPU(self):
        """Perform cold-start BDS-3 B1C acquisition on the raw-signal file.

        The function searches for BDS-3 B1C signals of all satellites listed
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
        # Find number of samples per B1C spreading-code period.
        samplesPerCode = settings.samplesPerCode
        # Number of samples per B1C code chip.
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)

        with open(settings.fileName, "rb") as fid:
            # Move to the configured processing start in the raw recording.
            if settings.dataType == "int16":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == "int8":
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)

            # np.fromfile performs the disk read on the host. Read 21 code
            # periods so that at least 20 complete periods remain.
            longSignal = np.fromfile(fid,settings.dataType,
                             settings.dataAdaptCoeff * 21 * samplesPerCode, )

        # Pass the IF data to GPU memory. Convert interleaved I/Q samples into
        # a complex signal before the device-side acquisition search.
        if settings.fileType == 2:
            longSignal = longSignal[::2] + 1j * longSignal[1::2]
            longSignal = xp.asarray(longSignal, dtype=complexDataType)
        else:
            longSignal = xp.asarray(longSignal, dtype=realDataType)

        #%% Acquisition initialization =====================================
        # GPU search coordinates and reusable buffers.
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period.
        ts = 1 / settings.samplingFreq
        # The phase vector spans two B1C code periods and is reused for every
        # coarse-frequency bin.
        phasePoints = xp.arange(samplesPerCode * 2, dtype=realDataType)
        phasePoints *= realDataType(2 * np.pi * ts)

        # Number of frequency bins in the specified search band.
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        #--- Variables for fine acquisition -------------------------------
        # Code resampling index used by B1C pilot-code fine acquisition.
        # Calculate it on the host in float64: float32 can change chip indices
        # close to integer boundaries.
        codeValueIndex = (np.arange(20 * samplesPerCode, dtype=np.float64) * ts
                          * settings.codeFreqBasis * 12).astype(np.int32)
        codeValueIndex %= settings.codeLength * 12
        codeValueIndex = xp.asarray(codeValueIndex)

        # Generate only one 1 ms carrier segment for fine acquisition. The
        # start phase of every segment is compensated separately in float64.
        samplesPer1ms = settings.samplesPerCode // 10
        finePhasePoints = xp.arange(samplesPer1ms, dtype=realDataType)
        finePhasePoints *= realDataType(2 * np.pi * ts)

        # Reuse the large 20 ms GPU buffers for every PRN.
        localData = xp.zeros(samplesPerCode * 2, dtype=realDataType)
        localPilot = xp.zeros(samplesPerCode * 2, dtype=realDataType)

        # Perform search for all listed PRN numbers.
        print("   (", end="", flush=True)

        #%% Coarse acquisition =============================================
        for PRN in settings.acqSatelliteList:
            # Generate B1C data codes and sample them at the sampling freq.
            dataPriTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "data")
            localData[:samplesPerCode] = xp.asarray(
                dataPriTable[:samplesPerCode], dtype=realDataType )

            #--- Perform DFT of B1C data and pilot codes -------------------
            # Conjugated data-code spectrum for frequency-domain correlation.
            dataPriFreqDom = xp.conjugate(fftBackend.fft(localData))

            # Generate the pilot replica only when pilot-assisted acquisition
            # is enabled.
            if settings.pilotACQflag:
                pilotPriTable = correlator.codeSampling(
                    settings, PRN, samplesPerCode, "pilotBOC11")
                localPilot[:samplesPerCode] = xp.asarray(
                    pilotPriTable[:samplesPerCode], dtype=realDataType)
                pilotPriFreqDom = xp.conjugate(fftBackend.fft(localPilot))

            # Initialize the global peak retained across frequency bins.
            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            for freqBinIndex in range(numberOfFreqBins):
                #--- Generate carrier-wave frequency grid -----------------
                # Compute the carrier frequency represented by this search bin.
                coarseFreqBin = (settings.IF  - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                
                # Generate local sine and cosine for this frequency bin.
                sigCarr = xp.exp(-1j * realDataType(coarseFreqBin) *
                                 phasePoints).astype(complexDataType, copy=False)
                # "Remove carrier" from the signal.
                # Convert the baseband signal to frequency domain.
                IQfreqDom = fftBackend.fft(sigCarr * longSignal[:samplesPerCode * 2])
                # Multiplication in frequency domain is correlation in time.
                convCodeIQ = IQfreqDom * dataPriFreqDom
                # Perform inverse DFT and store correlation results.
                results = xp.abs(fftBackend.ifft(convCodeIQ))

                if settings.pilotACQflag:
                    # Pilot signal components; data and pilot powers are
                    # combined with the configured 1:3 weighting.
                    convCodeIQ = IQfreqDom * pilotPriFreqDom
                    results = (results + 3*xp.abs(
                        fftBackend.ifft(convCodeIQ))) / 4

                #--- Look for correlation peaks ---------------------------
                # Find the maximum correlation peak and its code phase.
                # Transfer the peak magnitude only; transfer its code index
                # only if this bin becomes the new global maximum.
                maxPeakTemp = xp.max(results).item()
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = xp.argmax(results).item()
                    freqMax = coarseFreqBin

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Find the B1C code-phase exclusion range around the global peak.
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
            if codePhaseMax + 20 * samplesPerCode > longSignal.size:
                codePhaseMax -= samplesPerCode

            # A GLRT statistic above the threshold indicates that the B1C
            # signal has been found.
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                #%% Fine resolution frequency search ======================
                self.acqFlag[PRN - 1] = True
                # Indicate PRN number of the detected signal.
                print(f"{PRN:02d} ", end="", flush=True)

                #--- Prepare 200 ms code, carrier and input signals --------
                # Use the same branch selected for coarse acquisition.
                if settings.pilotACQflag:
                    b1cCode = correlator.generatePilotBOC11(settings, PRN)
                else:
                    b1cCode = correlator.generateDataBOC11(settings, PRN)
                b1cCode = xp.asarray(b1cCode, dtype=realDataType)
                b1cCode200ms = b1cCode[codeValueIndex]
                
                # Extract 20 B1C pilot periods (200 ms) from the detected
                # B1C pilot code phase.
                sig200ms = longSignal[codePhaseMax:codePhaseMax + 20 * samplesPerCode]
                
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

                # Wipe off pilot code and the reusable 1 ms carrier in a
                # two-dimensional operation, then restore each start phase.
                sig200ms = sig200ms.reshape((200, samplesPer1ms))
                b1cCode200ms = b1cCode200ms.reshape((200, samplesPer1ms))
                sumPerCode = xp.sum(
                    sig200ms * b1cCode200ms * localCarr1ms, axis=1)
                sumPerCode *= initialCarr

                #--- Find the fine carrier frequency ----------------------
                # Find the strongest spectral component after sign removal.
                maxPowerIndex = xp.argmax(xp.abs(fftBackend.fft(sumPerCode**2))).item()
                
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
        _, ax = plt.subplots()
        ax.bar(self.PRN[acqMask],self.peakMetric[acqMask],color="b",width=0.85 )
        ax.bar(self.PRN[notAcqMask],self.peakMetric[notAcqMask],color="r",
               width=0.85)
        ax.legend(labels=["Acquired signals", "Not acquired signals"])
        ax.set_xlabel("PRN number (no bar - SV is not in the acquisition list)")
        ax.set_ylabel("Acquisition Metric")
        ax.set_title("Acquisition results")
        plt.grid(True)
        plt.xticks(np.sort(self.PRN[validPrn]))
        plt.tight_layout()
        plt.show()

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

    #%% Show channel status
    def showChannelStatus(self):
        """Print the status of all channels in a table.

        Returns
        -------
            None
                The channel-status table is written to standard output.
        """
        settings = self._settings
        print("\n*=========*=====*===============*===========*=============*=========*")
        print("| Channel | PRN |   Frequency   |  Doppler  | Code Offset | Status  |")
        print("*=========*=====*===============*===========*=============*========*")
        for channelNr in range(settings.numberOfChannels):
            if self.acqFlag[channelNr]:
                print(
                    f"|    {channelNr:2d}   | {self.PRN[channelNr]:3d} |  "
                    f"{self.carrFreq[channelNr]:2.5e}  |   "
                    f"{self.carrFreq[channelNr] - settings.IF:5.0f}   |    "
                    f"{int(self.codePhase[channelNr]):6d}   |    T    |"
                )
            else:
                print(
                    f"|    {channelNr:2d}   | --- | ------------  |   -----   |    "
                    "------   |   Off   |"
                )

        print("*=========*=====*===============*===========*=============*=========*\n")

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
