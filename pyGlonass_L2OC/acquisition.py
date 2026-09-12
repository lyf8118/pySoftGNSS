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
GLONASS L2OCp acquisition on CPU or GPU.

"""

import numpy as np

#%% Acquisition results
class AcqEngine():
    """Perform GLONASS L2OCp acquisition and store its results.

    Args
    ----
        settings   - object
                   Receiver settings.
    """

    def __init__(self, settings):
        """Initialize PRN-indexed results through the largest configured PRN.

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
        # Allocate PRN-indexed slots 1...max(acqSatelliteList); use PRN-1 below.
        svNum = int(np.max(settings.acqSatelliteList))
        # Refined carrier frequencies [Hz], shape (svNum,); inf means absent.
        self.carrFreq = np.full(svNum, np.inf)
        # Zero-based L2OCp-code phases [samples], shape (svNum,); inf means absent.
        self.codePhase = np.full(svNum, np.inf)
        # Acquisition peak ratios, shape (svNum,); zero means not evaluated.
        self.peakMetric = np.zeros(svNum)
        # PRN labels, shape (svNum,); zero means no PRN result is assigned.
        self.PRN = np.zeros(svNum).astype(int)
        # Initial code-NCO frequencies [Hz], shape (svNum,); zero means unset.
        self.codeFreq = np.zeros(svNum)
        # Acquisition decisions, shape (svNum,); False means not acquired.
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
        """Perform CPU cold-start acquisition for GLONASS L2OC signals.

        The search stores the L2OCp code phase, carrier frequency, and
        acquisition peak metric.
        """
        import correlator
        from numpy import fft

        settings = self.settings
        samplesPerCode = settings.samplesPerCode
        samplesPerChip = round(settings.samplingFreq/settings.codeFreqBasis)

        #%% Read data for acquisition =========================================
        sampleSize = np.dtype(settings.dataType).itemsize
        with open(settings.fileName, "rb") as fid:
            fid.seek(settings.dataAdaptCoeff*settings.skipNumberOfSamples*sampleSize)
            longSignal = np.fromfile(
                fid, settings.dataType,
                settings.dataAdaptCoeff*12*samplesPerCode)

        if settings.fileType == 2:
            longSignal = longSignal-np.mean(longSignal)
            longSignal = (longSignal[0::2] + 1j*longSignal[1::2]).astype(
                np.complex128, copy=False)
        else:
            longSignal = longSignal.astype(np.float64, copy=False)

        #%% Initialization ====================================================
        ts = 1/settings.samplingFreq
        phasePoints = (np.arange(10*samplesPerCode, dtype=np.float64)
                       * 2*np.pi*ts)
        numberOfFreqBins = round(
            settings.acqSearchBand*2/settings.acqSearchStep)+1

        # Map the complete 200 ms fine-search interval to the L2OCp fine grid.
        codeValueIndex = np.floor(
            np.arange(10*samplesPerCode, dtype=np.float64) *
            settings.codeFreqBasis*settings.L2OCFineFactor /
            settings.samplingFreq).astype(np.int32)
        codeValueIndex %= (settings.codeLength*
                           settings.L2OCFineFactor)

        localL2OCpCode = np.zeros(2*samplesPerCode, dtype=np.float64)

        print("   (", end="", flush=True)
        for PRN in settings.acqSatelliteList:
            #%% Coarse acquisition ===========================================
            L2OCpCodesTable = correlator.codeSampling(
                settings, PRN, samplesPerCode)
            localL2OCpCode[:samplesPerCode] = L2OCpCodesTable
            cmCodeFreqDom = np.conjugate(fft.fft(localL2OCpCode))

            codePhaseMax, freqMax, peakMax = 0, 0.0, 0.0
            for freqBinIndex in range(numberOfFreqBins):
                coarseFreqBin = (settings.IF-settings.acqSearchBand +
                                 settings.acqSearchStep*freqBinIndex)
                sigCarr = np.exp(-1j*coarseFreqBin *
                                 phasePoints[:2*samplesPerCode])
                IQfreqDom = fft.fft(
                    sigCarr*longSignal[:2*samplesPerCode])
                results = np.abs(fft.ifft(IQfreqDom*cmCodeFreqDom))
                maxIndexTemp = int(np.argmax(results))
                maxPeakTemp = float(results[maxIndexTemp])
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            # Move a peak found in the second code period to the equivalent
            # phase in the first L2OCp period.
            if codePhaseMax >= samplesPerCode:
                codePhaseMax -= samplesPerCode

            # Search the second peak outside a one-chip circular exclusion.
            excludeIndex1 = codePhaseMax-samplesPerChip
            excludeIndex2 = codePhaseMax+samplesPerChip
            if excludeIndex1 < 0:
                codePhaseRange = np.arange(
                    excludeIndex2, samplesPerCode+excludeIndex1+1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = np.arange(
                    excludeIndex2-samplesPerCode, excludeIndex1+1)
            else:
                codePhaseRange = np.r_[
                    0:excludeIndex1+1, excludeIndex2:samplesPerCode]

            secondPeak = float(np.max(results[codePhaseRange]))
            self.codePhase[PRN-1] = codePhaseMax
            self.peakMetric[PRN-1] = peakMax/secondPeak
            self.PRN[PRN-1] = PRN

            if self.peakMetric[PRN-1] > settings.acqThreshold:
                self.acqFlag[PRN-1] = True
                print(f"{PRN:02d} ", end="", flush=True)

                #%% Fine carrier-frequency search =============================
                L2OCpCode = correlator.generateL2OcpBOCCode(PRN)
                L2OCpCode200ms = L2OCpCode[codeValueIndex]
                sig200ms = longSignal[
                    codePhaseMax:codePhaseMax+10*samplesPerCode]
                localCarr200cm = np.exp(-1j*freqMax*phasePoints)
                basebandSig = sig200ms*L2OCpCode200ms*localCarr200cm

                samplesPer1ms = settings.samplesPerCode//20
                sumPerCode = basebandSig[:200*samplesPer1ms].reshape(
                    200, samplesPer1ms).sum(axis=1)
                maxPowerIndex = int(np.argmax(np.abs(fft.fft(sumPerCode**2))))
                shiftAngle = np.angle(
                    np.exp(-2*np.pi*1j*maxPowerIndex/200))/2
                self.carrFreq[PRN-1] = (
                    freqMax-shiftAngle/0.001/2/np.pi)
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN-1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN-1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                if self.carrFreq[PRN-1] == 0:
                    self.carrFreq[PRN-1] = 1

            else:
                print(". ", end="", flush=True)

        print(")")

    #%% Acquisition engine using GPU
    def acqProcessGPU(self):
        """Perform GPU cold-start acquisition for GLONASS L2OC signals."""
        import correlator
        import cupy as xp
        from cupyx.scipy import fft as fftBackend

        settings = self.settings
        realDataType = xp.float32
        complexDataType = xp.complex64
        samplesPerCode = settings.samplesPerCode
        samplesPerChip = round(settings.samplingFreq/settings.codeFreqBasis)

        #%% Read data for acquisition =========================================
        sampleSize = np.dtype(settings.dataType).itemsize
        with open(settings.fileName, "rb") as fid:
            fid.seek(settings.dataAdaptCoeff*settings.skipNumberOfSamples*sampleSize)
            hostSignal = np.fromfile(
                fid, settings.dataType,
                settings.dataAdaptCoeff*12*samplesPerCode)

        if settings.fileType == 2:
            hostSignal = hostSignal-np.mean(hostSignal)
            hostSignal = hostSignal[0::2] + 1j*hostSignal[1::2]
            longSignal = xp.asarray(hostSignal, dtype=complexDataType)
        else:
            longSignal = xp.asarray(hostSignal, dtype=realDataType)

        #%% Initialization ====================================================
        ts = 1/settings.samplingFreq
        phasePoints = xp.arange(
            10*samplesPerCode, dtype=realDataType)
        phasePoints *= realDataType(2*np.pi*ts)
        numberOfFreqBins = round(
            settings.acqSearchBand*2/settings.acqSearchStep)+1

        codeValueIndex = np.floor(
            np.arange(10*samplesPerCode, dtype=np.float64) *
            settings.codeFreqBasis*settings.L2OCFineFactor /
            settings.samplingFreq).astype(np.int32)
        codeValueIndex %= (settings.codeLength*
                           settings.L2OCFineFactor)
        codeValueIndex = xp.asarray(codeValueIndex)

        # Generate only one 1 ms carrier segment for fine acquisition. The
        # start phase of every segment is compensated separately in float64.
        samplesPer1ms = settings.samplesPerCode//20
        finePhasePoints = xp.arange(samplesPer1ms, dtype=realDataType)
        finePhasePoints *= realDataType(2*np.pi*ts)

        localL2OCpCode = xp.zeros(2*samplesPerCode, dtype=realDataType)

        print("   (", end="", flush=True)
        for PRN in settings.acqSatelliteList:
            #%% Coarse acquisition ===========================================
            L2OCpCodesTable = correlator.codeSampling(
                settings, PRN, samplesPerCode)
            localL2OCpCode[:samplesPerCode] = xp.asarray(
                L2OCpCodesTable, dtype=realDataType)
            cmCodeFreqDom = xp.conjugate(
                fftBackend.fft(localL2OCpCode))

            codePhaseMax, freqMax, peakMax = 0, 0.0, 0.0
            for freqBinIndex in range(numberOfFreqBins):
                coarseFreqBin = (settings.IF-settings.acqSearchBand +
                                 settings.acqSearchStep*freqBinIndex)
                sigCarr = xp.exp(
                    -1j*realDataType(coarseFreqBin) *
                    phasePoints[:2*samplesPerCode]).astype(
                        complexDataType, copy=False)
                IQfreqDom = fftBackend.fft(
                    sigCarr*longSignal[:2*samplesPerCode])
                results = xp.abs(fftBackend.ifft(
                    IQfreqDom*cmCodeFreqDom))
                maxIndexTemp = int(xp.argmax(results).item())
                maxPeakTemp = float(results[maxIndexTemp].item())
                if maxPeakTemp > peakMax:
                    peakMax = maxPeakTemp
                    codePhaseMax = maxIndexTemp
                    freqMax = coarseFreqBin

            if codePhaseMax >= samplesPerCode:
                codePhaseMax -= samplesPerCode

            excludeIndex1 = codePhaseMax-samplesPerChip
            excludeIndex2 = codePhaseMax+samplesPerChip
            if excludeIndex1 < 0:
                codePhaseRange = xp.arange(
                    excludeIndex2, samplesPerCode+excludeIndex1+1)
            elif excludeIndex2 >= samplesPerCode:
                codePhaseRange = xp.arange(
                    excludeIndex2-samplesPerCode, excludeIndex1+1)
            else:
                codePhaseRange = xp.concatenate((
                    xp.arange(excludeIndex1+1),
                    xp.arange(excludeIndex2, samplesPerCode)))

            secondPeak = float(xp.max(results[codePhaseRange]).item())
            self.codePhase[PRN-1] = codePhaseMax
            self.peakMetric[PRN-1] = peakMax/secondPeak
            self.PRN[PRN-1] = PRN

            if self.peakMetric[PRN-1] > settings.acqThreshold:
                self.acqFlag[PRN-1] = True
                print(f"{PRN:02d} ", end="", flush=True)

                #%% Fine carrier-frequency search =============================
                L2OCpCode = xp.asarray(
                    correlator.generateL2OcpBOCCode(PRN),
                    dtype=realDataType)
                L2OCpCode200ms = L2OCpCode[codeValueIndex]
                sig200ms = longSignal[
                    codePhaseMax:codePhaseMax+10*samplesPerCode]
                localCarr1ms = xp.exp(
                    -1j*realDataType(freqMax)*finePhasePoints).astype(
                        complexDataType, copy=False)

                initialPhase = np.fmod(
                    2*np.pi*freqMax*np.arange(200, dtype=np.float64)*
                    samplesPer1ms*ts, 2*np.pi)
                initialCarr = xp.asarray(
                    np.exp(-1j*initialPhase), dtype=complexDataType)
                fineUsedSamples = 200*samplesPer1ms
                sig200ms = sig200ms[:fineUsedSamples].reshape(
                    200, samplesPer1ms)
                L2OCpCode200ms = L2OCpCode200ms[:fineUsedSamples].reshape(
                    200, samplesPer1ms)
                sumPerCode = xp.sum(
                    sig200ms*L2OCpCode200ms*localCarr1ms, axis=1)
                sumPerCode *= initialCarr
                maxPowerIndex = int(xp.argmax(
                    xp.abs(fftBackend.fft(sumPerCode**2))).item())
                shiftAngle = np.angle(
                    np.exp(-2*np.pi*1j*maxPowerIndex/200))/2
                self.carrFreq[PRN-1] = (
                    freqMax-shiftAngle/0.001/2/np.pi)
                # Convert carrier Doppler to the corresponding code frequency.
                self.codeFreq[PRN-1] = settings.codeFreqBasis + (
                    (self.carrFreq[PRN-1] - settings.IF) /
                    settings.carrFreqBasis * settings.codeFreqBasis)
                if self.carrFreq[PRN-1] == 0:
                    self.carrFreq[PRN-1] = 1

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
