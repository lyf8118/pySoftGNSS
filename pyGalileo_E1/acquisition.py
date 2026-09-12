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
Galileo data/pilot signal acquisition on CPU or GPU.

"""
import numpy as np


#%% Acquisition results
class AcqEngine():
    """Perform Galileo acquisition and store its results.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize acquisition results through the largest configured PRN.

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
        # Largest configured PRN and number of PRN-indexed result slots.
        svNum = int(np.max(settings.acqSatelliteList))
        # Refined carrier frequencies [Hz], shape ``(svNum,)``; inf means
        # unavailable.
        self.carrFreq = np.full(svNum, np.inf)
        # Zero-based E1-code phases [samples], shape ``(svNum,)``; inf means
        # unavailable.
        self.codePhase = np.full(svNum, np.inf)
        # Dimensionless first-to-second correlation-peak ratios,
        # shape ``(svNum,)``; zero means not searched.
        self.peakMetric = np.zeros(svNum)
        # PRN associated with each result slot; zero means not searched,
        # shape ``(svNum,)``.
        self.PRN = np.zeros(svNum, dtype=int)
        # Initial code-NCO frequencies [Hz], shape ``(svNum,)``; zero means
        # not initialized.
        self.codeFreq = np.zeros(svNum)
        # Acquisition decisions, shape ``(svNum,)``; False means not acquired.
        self.acqFlag = np.full(svNum, False)
        # Retain the caller's settings object by reference.
        self._settings = settings

    @property
    def settings(self):
        """Return the receiver-settings object retained by the engine.

        Returns
        -------
            settings   - object
                       Receiver settings.
        """
        return self._settings

    #%% Acquisition engine using CPU
    def acqProcessCPU(self):
        """Perform cold-start Galileo acquisition on the raw-signal file.

        The data and pilot primary-code correlations are combined
        non-coherently. Detected code phase and carrier frequency are stored
        in this acquisition engine; code phase is a zero-based sample offset.
        """
        import correlator
        from numpy import fft

        settings = self._settings
        samplesPerCode = settings.samplesPerCode
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)
        #%% Read data for acquisition =====================================
        with open(settings.fileName, "rb") as fid:
            scalarBytes = np.dtype(settings.dataType).itemsize
            fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples *
                     scalarBytes)
            longSignal = np.fromfile(
                fid, settings.dataType,
                settings.dataAdaptCoeff * settings.acqReadCodePeriods *
                samplesPerCode)

        # Convert interleaved I/Q samples into normal complex representation.
        if settings.fileType == 2:
            if settings.removeMean:
                longSignal = longSignal-np.mean(longSignal)
            longSignal = np.asarray(
                longSignal[::2]+1j*longSignal[1::2], dtype=np.complex128)
        else:
            longSignal = np.asarray(longSignal, dtype=np.float64)

        #%% Acquisition initialization ====================================
        ts = 1 / settings.samplingFreq
        phasePoints = (np.arange(samplesPerCode * 2, dtype=np.float64)
                       * 2 * np.pi * ts)
        numberOfFreqBins = (round(settings.acqSearchBand * 2 /
                                  settings.acqSearchStep) + 1)

        # Code indices and carrier phase for the continuous 200 ms fine search.
        fineSampleCount = settings.fineAcqPeriods * samplesPerCode
        codeValueIndex = (np.arange(fineSampleCount, dtype=np.float64) * ts
                          * settings.codeFreqBasis * settings.codeSubchips).astype(np.int32)
        codeValueIndex %= settings.codeLength * settings.codeSubchips
        finePhasePoints = (np.arange(fineSampleCount, dtype=np.float64)
                           * 2 * np.pi * ts)

        localData = np.zeros(samplesPerCode * 2, dtype=np.float64)
        localPilot = np.zeros(samplesPerCode * 2, dtype=np.float64)
        print("   (", end="", flush=True)

        #%% Coarse acquisition ============================================
        for PRN in settings.acqSatelliteList:
            dataTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "data")
            pilotTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "pilot")
            localData.fill(0); localPilot.fill(0)
            localData[:samplesPerCode] = dataTable
            localPilot[:samplesPerCode] = pilotTable
            dataFreqDom = np.conjugate(fft.fft(localData))
            pilotFreqDom = np.conjugate(fft.fft(localPilot))

            codePhaseMax, freqMax, peakMax = 0, 0.0, 0.0
            for freqBinIndex in range(numberOfFreqBins):
                coarseFreqBin = (settings.IF - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                sigCarr = np.exp(-1j * coarseFreqBin * phasePoints)
                signal = longSignal[:2 * samplesPerCode]
                IQfreqDom = fft.fft(sigCarr * signal)
                results = (np.abs(fft.ifft(IQfreqDom * dataFreqDom))
                           + np.abs(fft.ifft(IQfreqDom * pilotFreqDom)))

                maxIndexTemp = int(np.argmax(results))
                maxPeakTemp = float(results[maxIndexTemp])
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            #--- Find a one-chip-wide exclusion range around the peak -----
            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip
            if excludeIndex1 < 0:
                codePhaseRange = np.arange(excludeIndex2,
                                            samplesPerCode + excludeIndex1 + 1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = np.arange(excludeIndex2 - samplesPerCode,
                                            excludeIndex1 + 1)
            else:
                codePhaseRange = np.hstack((np.arange(0, excludeIndex1 + 1),
                                             np.arange(excludeIndex2, samplesPerCode)))

            secondPeak = float(np.max(results[codePhaseRange]))
            self.codePhase[PRN - 1] = codePhaseMax
            self.peakMetric[PRN - 1] = peakMax / secondPeak
            self.PRN[PRN - 1] = PRN

            if codePhaseMax + fineSampleCount > longSignal.size:
                codePhaseMax -= samplesPerCode

            #%% Fine carrier-frequency search =============================
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                self.acqFlag[PRN - 1] = True
                print(f"{PRN:02d} ", end="", flush=True)
                pilotCode = correlator.generatePilotCode(PRN, 1)
                pilotCodeFine = pilotCode[codeValueIndex]
                sigFine = longSignal[codePhaseMax:codePhaseMax + fineSampleCount]
                localCarrFine = np.exp(-1j * freqMax * finePhasePoints)

                basebandPilot = sigFine * pilotCodeFine * localCarrFine
                # E1 fine acquisition forms 200 one-millisecond pilot sums.
                sumPilot = basebandPilot.reshape((200, -1)).sum(axis=1)
                spectrum = np.abs(fft.fft(sumPilot**2))
                maxPowerIndex = int(np.argmax(spectrum))
                shiftAngle = np.angle(np.exp(-2 * np.pi * 1j * maxPowerIndex / 200)) / 2
                self.carrFreq[PRN - 1] = freqMax - shiftAngle / 0.001 / 2 / np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN - 1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN - 1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                if self.carrFreq[PRN - 1] == 0:
                    self.carrFreq[PRN - 1] = 1
                self.codePhase[PRN - 1] = codePhaseMax
            else:
                print(". ", end="", flush=True)
        print(")")

    #%% Acquisition engine using GPU
    def acqProcessGPU(self):
        """Perform cold-start Galileo acquisition using CuPy."""
        import correlator
        import cupy as xp
        from cupyx.scipy import fft as fftBackend

        settings = self._settings
        realDataType = xp.float32
        complexDataType = xp.complex64
        samplesPerCode = settings.samplesPerCode
        samplesPerChip = round(settings.samplingFreq / settings.codeFreqBasis)
        #%% Read data for acquisition =====================================
        with open(settings.fileName, "rb") as fid:
            scalarBytes = np.dtype(settings.dataType).itemsize
            fid.seek(settings.dataAdaptCoeff * settings.skipNumberOfSamples *
                     scalarBytes)
            longSignal = np.fromfile(
                fid, settings.dataType,
                settings.dataAdaptCoeff * settings.acqReadCodePeriods *
                samplesPerCode)

        # Convert interleaved I/Q samples into normal complex representation.
        if settings.fileType == 2:
            if settings.removeMean:
                longSignal = longSignal-np.mean(longSignal)
            longSignal = xp.asarray(
                longSignal[::2]+1j*longSignal[1::2], dtype=complexDataType)
        else:
            longSignal = xp.asarray(longSignal, dtype=realDataType)

        #%% Acquisition initialization ====================================
        ts = 1 / settings.samplingFreq
        phasePoints = xp.arange(samplesPerCode * 2, dtype=realDataType)
        phasePoints *= realDataType(2 * np.pi * ts)
        numberOfFreqBins = (round(settings.acqSearchBand * 2 /
                                  settings.acqSearchStep) + 1)

        fineSampleCount = settings.fineAcqPeriods * samplesPerCode
        # Generate indices in host float64 to avoid chip-boundary errors.
        codeValueIndex = (np.arange(fineSampleCount, dtype=np.float64) * ts
                          * settings.codeFreqBasis * settings.codeSubchips).astype(np.int32)
        codeValueIndex %= settings.codeLength * settings.codeSubchips
        codeValueIndex = xp.asarray(codeValueIndex)
        # Generate only one 1 ms carrier segment; compensate each segment's
        # start phase separately on the host in float64.
        samplesPer1ms = settings.samplesPerCode // 4
        finePhasePoints = xp.arange(samplesPer1ms, dtype=realDataType)
        finePhasePoints *= realDataType(2 * np.pi * ts)

        localData = xp.zeros(samplesPerCode * 2, dtype=realDataType)
        localPilot = xp.zeros(samplesPerCode * 2, dtype=realDataType)
        print("   (", end="", flush=True)

        #%% Coarse acquisition ============================================
        for PRN in settings.acqSatelliteList:
            dataTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "data")
            pilotTable = correlator.codeSampling(
                settings, PRN, samplesPerCode, "pilot")
            localData.fill(0); localPilot.fill(0)
            localData[:samplesPerCode] = xp.asarray(dataTable, dtype=realDataType)
            localPilot[:samplesPerCode] = xp.asarray(pilotTable, dtype=realDataType)
            dataFreqDom = xp.conjugate(fftBackend.fft(localData))
            pilotFreqDom = xp.conjugate(fftBackend.fft(localPilot))

            codePhaseMax, freqMax, peakMax = 0, 0.0, 0.0
            for freqBinIndex in range(numberOfFreqBins):
                coarseFreqBin = (settings.IF - settings.acqSearchBand
                                 + settings.acqSearchStep * freqBinIndex)
                sigCarr = xp.exp(-1j * realDataType(coarseFreqBin) * phasePoints)
                signal = longSignal[:2 * samplesPerCode]
                IQfreqDom = fftBackend.fft(sigCarr * signal)
                results = (xp.abs(fftBackend.ifft(IQfreqDom * dataFreqDom))
                           + xp.abs(fftBackend.ifft(IQfreqDom * pilotFreqDom)))

                maxPeakTemp = xp.max(results).item()
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = xp.argmax(results).item()
                    freqMax = coarseFreqBin

            excludeIndex1 = codePhaseMax - samplesPerChip
            excludeIndex2 = codePhaseMax + samplesPerChip
            if excludeIndex1 < 0:
                codePhaseRange = xp.arange(excludeIndex2,
                                            samplesPerCode + excludeIndex1 + 1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = xp.arange(excludeIndex2 - samplesPerCode,
                                            excludeIndex1 + 1)
            else:
                codePhaseRange = xp.hstack((xp.arange(0, excludeIndex1 + 1),
                                             xp.arange(excludeIndex2, samplesPerCode)))

            secondPeak = xp.max(results[codePhaseRange]).item()
            self.codePhase[PRN - 1] = codePhaseMax
            self.peakMetric[PRN - 1] = peakMax / secondPeak
            self.PRN[PRN - 1] = PRN
            if codePhaseMax + fineSampleCount > longSignal.size:
                codePhaseMax -= samplesPerCode

            #%% Fine carrier-frequency search =============================
            if self.peakMetric[PRN - 1] > settings.acqThreshold:
                self.acqFlag[PRN - 1] = True
                print(f"{PRN:02d} ", end="", flush=True)
                pilotCode = xp.asarray(correlator.generatePilotCode(PRN, 1),
                                       dtype=realDataType)
                sigFine = longSignal[codePhaseMax:codePhaseMax + fineSampleCount]
                localCarr1ms = xp.exp(
                    -1j * realDataType(freqMax) * finePhasePoints).astype(
                        complexDataType, copy=False)
                initialPhase = np.fmod(
                    2 * np.pi * freqMax * np.arange(200, dtype=np.float64) *
                    samplesPer1ms * ts, 2 * np.pi)
                initialCarr = xp.asarray(
                    np.exp(-1j * initialPhase), dtype=complexDataType)
                sigFine = sigFine.reshape((200, samplesPer1ms))
                pilotCodeFine = pilotCode[codeValueIndex].reshape(
                    (200, samplesPer1ms))
                sumPilot = xp.sum(
                    sigFine * pilotCodeFine * localCarr1ms, axis=1)
                sumPilot *= initialCarr
                spectrum = xp.abs(fftBackend.fft(sumPilot**2))
                maxPowerIndex = xp.argmax(spectrum).item()
                shiftAngle = np.angle(np.exp(-2 * np.pi * 1j * maxPowerIndex / 200)) / 2
                self.carrFreq[PRN - 1] = freqMax - shiftAngle / 0.001 / 2 / np.pi
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN - 1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN - 1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                if self.carrFreq[PRN - 1] == 0:
                    self.carrFreq[PRN - 1] = 1
                self.codePhase[PRN - 1] = codePhaseMax
            else:
                print(". ", end="", flush=True)
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
        ax.set_title(f'{self.settings.signalName} acquisition results')
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
        self.acqFlag =  self.acqFlag.take(PRNindexes)
        self.codeFreq =  self.codeFreq.take(PRNindexes)

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
