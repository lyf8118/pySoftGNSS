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
BDS-3 B2a acquisition on CPU or GPU.

"""

import numpy as np

#%% Acquisition results
class AcqEngine():
    """Perform BDS-3 B2a acquisition and store its results.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize result arrays through the largest configured B2a PRN.

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
        # Zero-based B2a-code phases [samples]; inf means unavailable.
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
        """Perform cold-start acquisition on the configured raw-signal file.

        The function searches for BDS-3 B2a signals of all satellites listed
        in ``settings.acqSatelliteList``. It saves the code phase and frequency
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
        # Find number of samples per spreading code
        samplesPerCode = settings.samplesPerCode

        # At least 202 ms are required for the 200 ms fine-frequency search.
        # Two extra code periods allow a detected code phase in the second ms.
        codeLen = max(202, settings.acqNonCohTime + 2)

        with open(settings.fileName,"rb") as fid:
            # Move the starting point of processing. Can be used to start the
            # signal processing at any point in the data record (e.g. good for
            # long records or for signal processing in blocks).
            if settings.dataType == 'int16':
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == 'int8':
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)
            # Read data for acquisition.
            longSignal = np.fromfile(fid, settings.dataType,
                                     settings.dataAdaptCoeff*codeLen*samplesPerCode)

        # Convert interleaved I/Q samples into a complex signal. For complex
        # input, remove the scalar-stream mean as in acquisitionCPU.m.
        if settings.fileType == 2:
            longSignal = longSignal - np.mean(longSignal)
            longSignal = longSignal[::2] + longSignal[1::2] * 1j
            longSignal = np.asarray(longSignal, dtype=np.complex128)
        else:
            longSignal = np.asarray(longSignal, dtype=np.float64)

        #%% Acquisition initialization =====================================
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period
        ts = 1 / settings.samplingFreq
        # Actual duration represented by one sampled B2a code period.
        codePeriod = samplesPerCode * ts
        # Number of samples per B2a code chip
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)

        # The carrier phase vector spans 2 ms: one millisecond corresponds to
        # the local-code replica and the second to its zero padding.
        # Double-precision 2*pi-scaled time coordinates for the 2 ms carrier.
        phasePoints = 2 * np.pi * ts * np.arange(samplesPerCode * 2, dtype=np.float64)

        #--- Variables for fine acquisition -------------------------------
        # Double-precision phase coordinates for the continuous 200 ms carrier.
        finePhasePoints = np.arange(200 * samplesPerCode, dtype=np.float64) * 2 * np.pi * ts

        # Map the complete 200 ms fine-acquisition interval to B2a-code chips
        # once. The index is independent of PRN and is reused for every signal.
        codeValueIndex = (np.arange(200 * samplesPerCode, dtype=np.float64) * ts
                          * settings.codeFreqBasis ).astype(np.int32)
        codeValueIndex %= settings.codeLength

        # Number of frequency bins in the specified search band
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        # Reuse these buffers for every PRN/frequency bin to avoid repeated
        # allocation of large 2 ms arrays in the inner search loops.
        dataCodes2ms = np.zeros(samplesPerCode * 2, dtype=np.float64)
        pilotCodes2ms = np.zeros(samplesPerCode * 2, dtype=np.float64)
        results = np.empty(samplesPerCode * 2, dtype=np.float64)

        # Perform search for all listed PRN numbers ...
        print('   (',end="", flush=True)

        #%% Coarse acquisition =============================================
        for PRN in settings.acqSatelliteList:
            # Generate B2a data and pilot codes and sample them according to
            # the receiver sampling frequency.
            dataCodes2ms[:samplesPerCode] = correlator.codeSampling(
                settings, PRN, samplesPerCode, "data")
            pilotCodes2ms[:samplesPerCode] = correlator.codeSampling(
                settings, PRN, samplesPerCode, "pilot")
            #--- Perform DFT of B2a data and pilot codes ------------------
            dataCodeFreqDom = np.conjugate(fft.fft(dataCodes2ms))
            pilotCodeFreqDom = np.conjugate(fft.fft(pilotCodes2ms))

            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            for freqBinIndex in range(numberOfFreqBins):
                # Compute the carrier frequency represented by this search bin.
                coarseFreqBin = settings.IF - settings.acqSearchBand + \
                                settings.acqSearchStep * freqBinIndex
                # Generate the complex local carrier for this frequency bin.
                sigCarr = np.exp(-1j * coarseFreqBin * phasePoints)

                # Search results of one frequency bin and all code shifts
                results.fill(0.0)

                #--- Do non-coherent integration ------------------------------
                for nonCohIndex in range(settings.acqNonCohTime):
                    # Take a 2 ms input block for correlation.
                    signal = longSignal[nonCohIndex * samplesPerCode:
                                        (nonCohIndex + 2) * samplesPerCode]
                    # "Remove carrier" from the signal and convert the baseband
                    # signal to frequency domain
                    IQfreqDom = fft.fft(sigCarr * signal)
                    # Multiplication in the frequency domain is equivalent to
                    # correlation in the time domain after the inverse DFT.
                    convData = IQfreqDom * dataCodeFreqDom
                    convPilot = IQfreqDom * pilotCodeFreqDom
                    # Perform inverse DFT and non-coherently combine the data
                    # and pilot correlation magnitudes.
                    results += (np.abs(fft.ifft(convData))
                                + np.abs(fft.ifft(convPilot)))

                #--- Look for correlation peaks ---------------------------
                # Find the maximum correlation peak and its code phase.
                maxIndexTemp = int(np.argmax(results))
                maxPeakTemp = float(results[maxIndexTemp])

                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Match acquisitionCPU.m: use the results left by the final
            # frequency bin and exclude one chip around the global code phase.
            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip

            #--- Correct the exclusion range at array boundaries ----------
            # Correct the retained index range when the one-chip exclusion
            # interval crosses either boundary of the correlation array.
            if excludeIndex1 < 0:
                codePhaseRange = np.arange(excludeIndex2,
                                           samplesPerCode + excludeIndex1 + 1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = np.arange(excludeIndex2 - samplesPerCode,
                                           excludeIndex1 + 1)
            else:
                codePhaseRange = np.hstack((np.arange(0, excludeIndex1 + 1),
                                    np.arange(excludeIndex2, samplesPerCode)))

            #--- Find the second-highest correlation peak -----------------
            # Find the second-highest correlation peak outside the excluded
            # one-chip interval.
            secondPeak = float(np.max(results[codePhaseRange]))

            # Store the zero-based code-phase estimate.
            self.codePhase[PRN-1] = codePhaseMax
            # Store the peak ratio used as the GLRT acquisition statistic.
            self.peakMetric[PRN-1] = peakMax / secondPeak
            # Store the searched PRN.
            self.PRN[PRN-1] = PRN

            #%% Fine carrier-frequency search =============================
            if self.peakMetric[PRN-1] > settings.acqThreshold :
                # A GLRT statistic above the threshold indicates that the
                # signal has been found.
                self.acqFlag[PRN-1] = True
                # Indicate PRN number of the detected signal
                print(PRN,' ',end="", flush=True)
                #--- Prepare 200 ms code, carrier and input signals -----------
                # Generate the complete continuous 200 ms local carrier,
                # following acquisitionCPU.m.
                localCarr200cm = np.exp(-1j * freqMax * finePhasePoints)
                # Generate and sample the B2a data and pilot codes over the
                # complete 200 ms fine-acquisition interval.
                dataCode = correlator.generateB2aDataCode(settings, PRN)
                pilotCode = correlator.generateB2aPilotCode(settings, PRN)
                dataCode200ms = dataCode[codeValueIndex]
                pilotCode200ms = pilotCode[codeValueIndex]
                # Take 200 ms of input for fine-frequency acquisition.
                sig200ms = longSignal[ codePhaseMax:codePhaseMax +
                                                      200 * samplesPerCode]
                sig200ms = sig200ms.reshape((200, samplesPerCode))

                # Wipe off both local codes and the carrier, then integrate
                # each of the 200 code periods.
                #--- Integration for each of the 200 codes ----------------
                sumPerCodeData = np.empty(200, dtype=np.complex128)
                sumPerCodePilot = np.empty(200, dtype=np.complex128)

                for codePeriodIndex in range(200):
                    signalIndex = slice(codePeriodIndex * samplesPerCode,
                                        (codePeriodIndex + 1) * samplesPerCode)
                    localCarr = localCarr200cm[signalIndex]
                    sumPerCodeData[codePeriodIndex] = np.dot(
                        sig200ms[codePeriodIndex],
                        dataCode200ms[signalIndex] * localCarr)
                    sumPerCodePilot[codePeriodIndex] = np.dot(
                        sig200ms[codePeriodIndex],
                        pilotCode200ms[signalIndex] * localCarr)

                #--- Find the fine carrier frequency ----------------------
                # Index of the max power
                finePower = (np.abs(fft.fft(sumPerCodeData**2))
                             + np.abs(fft.fft(sumPerCodePilot**2)))
                maxPowerIndex = finePower.argmax()
                # FFT shift angle
                shiftAngle = np.angle(np.exp( -2*np.pi*1j* maxPowerIndex/200 ))/2
                self.carrFreq[PRN-1] = freqMax - shiftAngle/codePeriod/2/np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN-1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN-1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                # If IF is zero, use 1 Hz so the tracking channel remains active.
                if self.carrFreq[PRN-1] == 0:
                    self.carrFreq[PRN-1] = 1
            else:
                #--- No signal with this PRN ----------------------------------
                print('. ',end="", flush=True)

        #%% Acquisition is over ===========================================
        print(')')
    #%% Acquisition engine using GPU
    def acqProcessGPU(self):
        """Perform cold-start acquisition on the configured raw-signal file.

        The function searches for BDS-3 B2a signals of all satellites listed
        in ``settings.acqSatelliteList``. It saves the code phase and frequency
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
        # Match acquisitionGPU.m: bulk GPU processing uses single precision.
        realDataType = xp.float32
        complexDataType = xp.complex64

        #%% Read data for acquisition ======================================
        # Read and preprocess acquisition data on the host.
        # Number of samples in one nominal B2a code period.
        samplesPerCode = settings.samplesPerCode

        # At least 202 ms are required for the 200 ms fine-frequency search.
        # Two extra code periods allow a detected code phase in the second ms.
        codeLen = max(202, settings.acqNonCohTime + 2)

        with open(settings.fileName,"rb") as fid:
            # Move to the configured processing start in the raw recording.
            if settings.dataType == 'int16':
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples * 2)
            elif settings.dataType == 'int8':
                fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples)
            # np.fromfile performs the disk read on the host.
            longSignal = np.fromfile(fid, settings.dataType,
                             settings.dataAdaptCoeff*codeLen*samplesPerCode)

        # Pass the IF data to GPU memory. For complex input, remove the
        # scalar-stream mean before combining the interleaved I/Q samples.
        if settings.fileType == 2:
            longSignal = longSignal - np.mean(longSignal)
            longSignal = longSignal[::2] + longSignal[1::2] * 1j
            longSignal = xp.asarray(longSignal, dtype=complexDataType)
        else:
            longSignal = xp.asarray(longSignal, dtype=realDataType)

        #%% Acquisition initialization =====================================
        # GPU search coordinates and reusable buffers.
        #--- Variables for coarse acquisition -----------------------------
        # Sampling period and duration represented by one sampled code period.
        ts = 1 / settings.samplingFreq
        codePeriod = samplesPerCode * ts
        # One-chip exclusion width used when finding the secondary peak.
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)
        # The phase vector spans 2 ms: one millisecond corresponds to the
        # local-code replica and the second to its zero padding.
        # Single-precision 2*pi-scaled time coordinates for the 2 ms carrier.
        # Fine acquisition reuses the first 1 ms; applying
        # a separate start phase to every block avoids one large 200 ms phase.
        phasePoints = xp.arange(samplesPerCode * 2, dtype=realDataType)
        phasePoints *= realDataType(2 * np.pi * ts)
        finePhasePoints = phasePoints[:samplesPerCode]

        #--- Variables for fine acquisition -------------------------------
        # Map all 200 ms absolute sample positions to B2a-code chips once. The
        # mapping follows acquisitionGPU.m's double-precision host calculation:
        # direct float32 arithmetic changes chip indices near integer boundaries.
        codeValueIndex = (np.arange(200 * samplesPerCode, dtype=np.float64)* ts
                          * settings.codeFreqBasis).astype(np.int32)
        codeValueIndex %= settings.codeLength
        codeValueIndex = xp.asarray(codeValueIndex)

        # Number of coarse-frequency bins across the configured search band.
        numberOfFreqBins = round(settings.acqSearchBand * 2 /
                                 settings.acqSearchStep) + 1

        # Reuse the large 2 ms GPU buffers for every PRN/frequency bin.
        dataCodes2ms = xp.zeros(samplesPerCode * 2, dtype=realDataType)
        pilotCodes2ms = xp.zeros(samplesPerCode * 2, dtype=realDataType)
        results = xp.empty(samplesPerCode * 2, dtype=realDataType)

        # Search every configured satellite PRN.
        print('   (',end="", flush=True)

        #%% Coarse acquisition ============================================
        for PRN in settings.acqSatelliteList:
            # Generate one B2a data and pilot period and sample both at the
            # input sample rate.
            dataCodes2ms[:samplesPerCode] = xp.asarray(
                correlator.codeSampling(
                    settings, PRN, samplesPerCode, "data"),
                dtype=realDataType)
            pilotCodes2ms[:samplesPerCode] = xp.asarray(
                correlator.codeSampling(
                    settings, PRN, samplesPerCode, "pilot"),
                dtype=realDataType)
            #--- Perform DFT of B2a data and pilot codes ------------------
            dataCodeFreqDom = xp.conjugate(fftBackend.fft(dataCodes2ms))
            pilotCodeFreqDom = xp.conjugate(fftBackend.fft(pilotCodes2ms))

            codePhaseMax, freqMax, peakMax = 0, 0, 0
            #--- Make the correlation for all frequency bins --------------
            # Correlate the signal at every coarse-frequency bin.
            for freqBinIndex in range(numberOfFreqBins):
                # Frequency represented by the current coarse-search bin.
                coarseFreqBin = settings.IF - settings.acqSearchBand + \
                                settings.acqSearchStep * freqBinIndex
                # Complex carrier replica for this frequency bin.
                sigCarr = xp.exp(-1j * realDataType(coarseFreqBin) *
                             phasePoints).astype(complexDataType, copy=False)

                # Correlation magnitudes for this bin and every code shift.
                results.fill(0.0)

                #--- Do non-coherent integration --------------------------
                # Non-coherently combine overlapping 2 ms signal blocks.
                for nonCohIndex in range(settings.acqNonCohTime):
                    # Take a 2 ms input vector for correlation. Consecutive
                    # blocks advance by one code period.
                    signal = longSignal[nonCohIndex * samplesPerCode:
                                        (nonCohIndex + 2) * samplesPerCode]
                    # Wipe off this carrier and transform the 2 ms signal.
                    IQfreqDom = fftBackend.fft(sigCarr * signal)
                    # Multiplication by the conjugated code spectrum performs
                    # correlation after the inverse FFT.
                    convData = IQfreqDom * dataCodeFreqDom
                    convPilot = IQfreqDom * pilotCodeFreqDom
                    # Accumulate data- and pilot-correlation magnitudes.
                    results += (xp.abs(fftBackend.ifft(convData))
                                + xp.abs(fftBackend.ifft(convPilot)))

                #--- Look for correlation peaks ---------------------------
                # Transfer the peak magnitude only. Transfer its code index
                # only if this bin becomes the new global maximum.
                maxPeakTemp = xp.max(results).item()

                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = xp.argmax(results).item()
                    freqMax = coarseFreqBin

            #--- Find a one-chip-wide exclusion range around the peak -----
            # Following acquisitionGPU.m, use the correlation array left by the
            # final frequency bin and retain shifts outside +/-1 chip of the
            # global code peak. Wrap the retained range at the 1 ms boundary.
            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip

            #--- Correct the exclusion range at array boundaries ----------
            if excludeIndex1 < 0:
                codePhaseRange = xp.arange(
                    excludeIndex2,
                    samplesPerCode + excludeIndex1 + 1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = xp.arange(
                    excludeIndex2 - samplesPerCode,
                    excludeIndex1 + 1)
            else:
                codePhaseRange = xp.hstack((
                    xp.arange(0, excludeIndex1 + 1),
                    xp.arange(excludeIndex2, samplesPerCode)))

            #--- Find the second-highest correlation peak -----------------
            # The largest retained value is the secondary peak in the ratio.
            secondPeak = xp.max(results[codePhaseRange]).item()

            # Store host-side coarse results. Code phase remains zero-based.
            self.codePhase[PRN-1] = codePhaseMax
            # The peak ratio is the GLRT acquisition statistic.
            self.peakMetric[PRN-1] = peakMax / secondPeak
            self.PRN[PRN-1] = PRN

            #%% Fine carrier-frequency search =============================
            if self.peakMetric[PRN-1] > settings.acqThreshold :
                # A GLRT statistic above the threshold indicates that the
                # signal has been found.
                self.acqFlag[PRN-1] = True
                # Report the detected PRN.
                print(PRN,' ',end="", flush=True)
                #--- Prepare 200 ms code, carrier and input signals -------
                # Generate only one 1 ms complex64 carrier segment.
                localCarr1ms = xp.exp(-1j * realDataType(freqMax) *
                          finePhasePoints).astype(complexDataType, copy=False)
                # Keep one unsampled B2a data and pilot period on the GPU.
                dataCode = xp.asarray(
                    correlator.generateB2aDataCode(settings, PRN),
                    dtype=realDataType)
                pilotCode = xp.asarray(
                    correlator.generateB2aPilotCode(settings, PRN),
                    dtype=realDataType)

                # Generate the complete continuous 200 ms code replica in one
                # operation. Its values are only +/-1, so float32 is exact.
                dataCode200ms = dataCode[codeValueIndex].reshape(
                    (200, samplesPerCode))
                pilotCode200ms = pilotCode[codeValueIndex].reshape(
                    (200, samplesPerCode))
                # Align 200 ms of input at the detected zero-based code phase;
                # every row of the view is one sampled code period.
                sig200ms = longSignal[ codePhaseMax:codePhaseMax + 200 * samplesPerCode]
                sig200ms = sig200ms.reshape((200, samplesPerCode))
                # One coherent result is produced for every code period.
                #--- Integration for each of the 200 codes ----------------
                sumPerCodeData = xp.empty(200, dtype=complexDataType)
                sumPerCodePilot = xp.empty(200, dtype=complexDataType)

                for codePeriodIndex in range(200):
                    # Combine this millisecond of the continuous code with the
                    # reusable one-millisecond carrier replica.
                    localDataCarr = (dataCode200ms[codePeriodIndex]
                                     * localCarr1ms)
                    localPilotCarr = (pilotCode200ms[codePeriodIndex]
                                      * localCarr1ms)

                    # Compute this millisecond's carrier start phase on the host
                    # in float64 and reduce it modulo 2*pi. Double precision
                    # avoids the fine-bin shift observed with a float32 start.
                    initialPhase = np.fmod(2 * np.pi * freqMax *
                                    codePeriodIndex * codePeriod, 2 * np.pi)

                    # Only the resulting unit phasor enters GPU processing.
                    initialCarr = np.complex64(np.exp(-1j * initialPhase))

                    # The start phasor is constant within this block, so apply it
                    # after the dot product. Signal, replica and sum stay on GPU.
                    sumPerCodeData[codePeriodIndex] = xp.dot(
                        sig200ms[codePeriodIndex], localDataCarr) * initialCarr
                    sumPerCodePilot[codePeriodIndex] = xp.dot(
                        sig200ms[codePeriodIndex], localPilotCarr) * initialCarr

                #--- Find the fine carrier frequency ----------------------
                # Squaring removes the BPSK data-bit sign and doubles residual
                # carrier phase. A 200-point FFT locates that doubled frequency.
                # Index of the maximum power in the fine-frequency spectrum.
                finePower = (xp.abs(fftBackend.fft(sumPerCodeData**2))
                             + xp.abs(fftBackend.fft(sumPerCodePilot**2)))
                maxPowerIndex = finePower.argmax().item()
                # Wrap the FFT-bin phase and divide by two to undo the squaring.
                shiftAngle = np.angle(np.exp(-2*np.pi*1j*maxPowerIndex/200))/2
                # Convert phase per code period to hertz and refine the coarse
                # carrier estimate on the host in double precision.
                self.carrFreq[PRN-1] = freqMax - shiftAngle/codePeriod/2/np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN-1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN-1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                # If IF is zero, use 1 Hz so the tracking channel remains active.
                if self.carrFreq[PRN-1] == 0:
                    self.carrFreq[PRN-1] = 1
            else:
                #--- No signal with this PRN ----------------------------------
                print('. ',end="", flush=True)

        #%% Acquisition is over ===========================================
        print(')')


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
