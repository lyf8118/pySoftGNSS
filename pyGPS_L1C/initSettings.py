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

initSettings.py - Module Description
------------------------------------
Initialize and save GPS L1C receiver settings.

Settings can be edited inside ``Settings`` or updated after initialization.
All settings are described in the class code.

"""

import numpy as np

class Settings(object):
    """Initialize and store GPS L1C receiver settings.

    Settings can be edited inside this class or updated after initialization.
    Edit this file to configure the receiver parameters. All settings are
    described in the class code.
    """

    def __init__(self):
        """Initialize the default receiver settings.

        Returns
        -------
            None
        """
        #%% Processing settings ===============================================
        # Process at least 36 s plus tracking transients so complete CNAV-2
        # frames are available to navigation decoding.
        self.msToProcess         = 40000          #[ms]
        # Maximum number of channels used for signal processing.
        self.numberOfChannels    = 15
        # Move the starting point of processing. Can be used to start the
        # signal processing at any point in the data record (e.g. for long
        # records). fseek function is used to move the file read point,
        # therefore the offset is expressed in logical time samples, not
        # directly in bytes. I/Q files contain two scalar values per sample.
        self.skipNumberOfSamples        = 0

        #%% Raw signal file name and related parameters =======================
        # Default raw recording used by the offline receiver.
        self.fileName           = "../IF_Data_Set/GPS L1C.iq"
        # Data type used to store each real scalar in the file.
        self.dataType           = 'int8'
        # File layouts:
        # 1 - real samples S0,S1,S2,...
        # 2 - interleaved I/Q samples I0,Q0,I1,Q1,I2,Q2,...
        self.fileType           = 2
        # Intermediate and sampling frequencies.
        self.IF                 = 0.0              # [Hz]
        self.samplingFreq       = 25.0e6           # [Hz]
        # Front-end bandwidth used by wideband-tracking weighting factors.
        self.FEBW               = 27.0e6           # [Hz]

        #%% L1C code parameters ==============================================
        # Nominal rate of the L1C primary code.
        self.codeFreqBasis      = 1.023e6         # [Hz]
        # Number of chips in one L1C primary-code period.
        self.codeLength         = 10230            # [Chips]
        # Carrier center frequency.
        self.carrFreqBasis      = 1575.42e6        # [Hz]

        #%% CNAV-2 LDPC decoder settings =====================================
        # Min-Sum decoder options. Set the offset to zero for standard
        # Min-Sum decoding.
        self.ldpcMaxIterations  = 50
        self.ldpcMsAlpha        = 1.0
        self.ldpcMsOffset       = 0.2

        #%% Acquisition settings ==============================================
        # Enable GPU acceleration for acquisition.
        self.gpuACQflag         = True               #  True - On; False - Off
        # Skip acquisition and restore the configured pickle file.
        self.skipAcquisition    = False
        # Acquisition pickle filename.
        self.acqPklName         = "./result_Cache/acqResults.pkl"
        # List of satellites to look for. Some satellites can be excluded to
        # speed up acquisition.
        self.acqSatelliteList   = np.arange(1,33)    # [PRN numbers]
        # One-sided band around IF searched for satellite signals.
        self.acqSearchBand      = 5000.              # [Hz]
        # Dimensionless primary-to-secondary correlation-peak threshold.
        self.acqThreshold       = 2.0
        # Frequency search step for coarse acquisition
        self.acqSearchStep      = 50                 # [Hz]
        # Enable acquisition using the pilot component.
        self.pilotACQflag       = True

        #%% Tracking loops settings ===========================================
        # Skip tracking and restore the configured pickle file.
        self.skipTracking       = False
        # Select channel scheduling for tracking.
        self.trkMode            = 1       # 0 - Channel-serial tracking;
                                          # 1 - Channel-parallel tracking;
        # Select the Python, SIMD or GPU correlator backend.
        self.correlatorType     = 2       # 0 - Python correlator;
                                          # 1 - SIMD correlator;
                                          # 2 - GPU correlator;
        # Number of right-shift bits for IF data to prevent SIMD-correlator
        # overflow when ADC valid bits occupy the high bits.
        if self.correlatorType == 1:
            self.rShiftBits     = 5       # 0 - No shift;
        # Pickle file used when tracking is skipped.
        self.trkPklName             = "./result_Cache/trackResults.pkl"
        # Code tracking loop parameters
        self.dllDampingRatio        = 0.7
        self.dllNoiseBandwidth      = 1.0         # [Hz]
        self.dllCorrelatorSpacing   = 0.07        # [primary-code chip]
        # Carrier tracking loop parameters
        self.pllDampingRatio        = 0.7
        self.pllNoiseBandwidth      = 10.         # [Hz]
        # Integration time for DLL and PLL
        self.intTime                = 0.01        # [s]
        # Enable tracking using the pilot component.
        self.pilotTRKflag           = True

        #%% Navigation solution settings ======================================
        # When enabled, restore saved navigation output instead of computing it.
        self.skipNavigation     = False
        # Pickle file used when navigation processing is skipped.
        self.navPklName         = "./result_Cache/navSolutions_eph.pkl"
        # Period between pseudorange and position calculations.
        self.navSolPeriod       = 200.0           # [ms]
        # Elevation mask to exclude signals from satellites at low elevation
        self.elevationMask      = 5.              # [degrees 0 - 90]
        # Enable/disable tropospheric range correction.
        self.useTropCorr        = True            # True - On; False - Off
        # True antenna position in the UTM system, if known. If all components
        # are NaN, plotting uses the mean estimated position as its reference.
        self.truePositionE      = float('nan')
        self.truePositionN      = float('nan')
        self.truePositionU      = float('nan')

        #%% Plot settings =====================================================
        # Enable/disable plotting of the tracking results for each channel
        self.plotTracking       = 1              # 0 - Off; 1 - On
        # Enable/disable plotting of the navigation solution
        self.plotNavigation     = 1              # 0 - Off; 1 - On

        #%% Constants =========================================================
        self._c                 = 299792458.     # Speed of light [m/s].
        # Initial assumed signal travel time [ms].
        self.startOffset       = 68.802

        #%% CNo Settings ======================================================
        # Number of prompt accumulations in each VSM C/N0 estimation window.
        self.CNoVSMinterval = 20

    #%% Methods below
    @property
    def c(self):
        """Return the speed of light.

        Returns
        -------
            c           - float
                        Speed of light in metres per second.
        """
        return self._c

    @property
    def samplesPerCode(self):
        """Return the number of samples per spreading code.

        Returns
        -------
            samplesPerCode - int
                           Rounded number of samples in one L1C-code period.
        """
        # Find the number of samples per spreading code.
        return round(self.samplingFreq/(self.codeFreqBasis / self.codeLength))

    @property
    def dataAdaptCoeff(self):
        """Return the stored values per logical signal sample.

        Returns
        -------
            dataAdaptCoeff - int
                           ``1`` for real samples or ``2`` for interleaved
                           complex samples.
        """
        # Convert logical sample counts to counts of stored scalar values.
        if self.fileType == 1:
            return 1
        else:
            return 2

    def probeData(self):
        """Plot raw data in the time and frequency domains and a histogram.

        The configured part of the raw signal is read and displayed before
        receiver processing starts.

        Returns
        -------
            None
        """
        import matplotlib.pyplot as plt
        from scipy.signal import welch
        from scipy.signal.windows import hamming

        #%% Generate plot of raw data =========================================
        with open(self.fileName, 'rb') as fid:
            # Move the starting point of processing. This can be used to start
            # signal processing at any point in a long data record.
            sampleSize = np.dtype(self.dataType).itemsize
            fid.seek(self.dataAdaptCoeff * self.skipNumberOfSamples * sampleSize, 0)

            # Read ten complete L1C-code periods of the input signal.
            sampleCount = self.dataAdaptCoeff * 10 * self.samplesPerCode
            data = np.fromfile(fid, self.dataType, sampleCount)

        if data.size < sampleCount:
            raise OSError('Could not read enough data from the data file.')

        #--- Initialization ---------------------------------------------------
        plt.figure(100)
        plt.clf()

        if self.fileType == 2:
            data = data[0::2] + 1j * data[1::2]

        samplesToPlot = round(self.samplesPerCode/2)
        timeScale = np.arange(samplesToPlot) / self.samplingFreq * 1000

        #--- Time-domain plot -------------------------------------------------
        if self.fileType == 1:
            axTime = plt.subplot(2, 2, 3)
            axTime.plot(timeScale, data[:samplesToPlot])
            axTime.set_title('Time domain plot')
            axTime.set_xlabel('Time (ms)')
            axTime.set_ylabel('Amplitude')
            axTime.axis('tight')
            axTime.grid()
        else:
            axTimeI = plt.subplot(3, 2, 4)
            axTimeI.plot(timeScale, np.real(data[:samplesToPlot]))
            axTimeI.set_title('Time domain plot (I)')
            axTimeI.set_xlabel('Time (ms)')
            axTimeI.set_ylabel('Amplitude')
            axTimeI.axis('tight')
            axTimeI.grid()

            axTimeQ = plt.subplot(3, 2, 3)
            axTimeQ.plot(timeScale, np.imag(data[:samplesToPlot]))
            axTimeQ.set_title('Time domain plot (Q)')
            axTimeQ.set_xlabel('Time (ms)')
            axTimeQ.set_ylabel('Amplitude')
            axTimeQ.axis('tight')
            axTimeQ.grid()

        #--- Frequency-domain plot -------------------------------------------
        rows = 2 if self.fileType == 1 else 3
        axFreq = plt.subplot2grid((rows, 2), (0, 0), colspan=2)
        freq, sigspec = welch(
            data, fs=self.samplingFreq, window=hamming(32768, sym=False),
            noverlap=2048, nfft=32768,
            return_onesided=(self.fileType == 1))
        if self.fileType == 2:
            freq = np.fft.fftshift(freq)
            sigspec = np.fft.fftshift(sigspec)
        axFreq.plot(freq/1e6, 10*np.log10(sigspec))
        axFreq.set_title('Frequency domain plot')
        axFreq.set_xlabel('Frequency (MHz)')
        axFreq.set_ylabel('Magnitude')
        axFreq.axis('tight')
        axFreq.grid()

        #--- Histogram --------------------------------------------------------
        if self.fileType == 1:
            axHist = plt.subplot(2, 2, 4)
            axHist.hist(data)
            axHist.set_title('Histogram')
            dmax = np.max(np.abs(data)) + 1
            axHist.set_xlim(-dmax, dmax)
            axHist.set_xlabel('Bin')
            axHist.set_ylabel('Number in bin')
            axHist.grid()
        else:
            axHistI = plt.subplot(3, 2, 6)
            axHistI.hist(np.real(data), bins='auto')
            axHistI.set_title('Histogram (I)')
            dmax = np.max(np.abs(data)) + 1
            axHistI.set_xlim(-dmax, dmax)
            axHistI.set_xlabel('Bin')
            axHistI.set_ylabel('Number in bin')
            axHistI.grid()

            axHistQ = plt.subplot(3, 2, 5)
            axHistQ.hist(np.imag(data), bins='auto')
            axHistQ.set_title('Histogram (Q)')
            axHistQ.set_xlim(-dmax, dmax)
            axHistQ.set_xlabel('Bin')
            axHistQ.set_ylabel('Number in bin')
            axHistQ.grid()

        plt.tight_layout()
        plt.show()
        print('  Raw IF data plotted ')

    def postProcessing(self):
        """Process the raw GPS L1C signal and plot the receiver results.

        First, acquisition identifies the satellites present in the configured
        data file. The code and carrier of each acquired signal are then
        tracked and their coherent-integration results are stored. Finally,
        the navigation messages are decoded, pseudoranges and receiver
        positions are calculated, and the requested result plots are produced.

        The processing sequence is:

        1. Acquire the GPS L1C signals.
        2. Initialize and track the active receiver channels.
        3. Decode navigation messages and calculate the navigation solution.
        4. Plot the tracking and navigation results.

        Returns
        -------
            None
        """
        import pickle, datetime
        #--- Acquisition ------------------------------------------------------
        # Perform acquisition unless it is disabled in the receiver settings;
        # otherwise restore the previously saved acquisition results.
        if not self.skipAcquisition:
            print('\nAcquisition is going...')
            startTime = datetime.datetime.now()

            # Run CPU or GPU acquisition according to gpuACQflag.
            import acquisition
            acqResults = acquisition.AcqEngine(self)
            acqResults.run()

            # Report acquisition elapsed time.
            print("   Elapsed time for acquisition is %s s\n" %(
                (datetime.datetime.now() - startTime).total_seconds()) )

            # Save the acquisition engine, including its result arrays.
            with open(self.acqPklName,"wb") as fp:
                pickle.dump(acqResults,fp)

        else:
            # Restore the complete acquisition engine and its result arrays.
            print("Skip acquisition and using pickle data")
            with open(self.acqPklName,"rb") as fp:
                acqResults = pickle.load(fp)

        #--- Tracking ---------------------------------------------------------
        # Process all active channels unless tracking results are restored.
        if not self.skipTracking:
            # Record and report the tracking start time.
            startTime = datetime.datetime.now()
            print('Tracking started at ',startTime.strftime("%Y-%m-%d %H:%M:%S"))

            # Run channel-serial or channel-parallel tracking according to
            # trkMode. The tracking results are retained by the engine.
            import tracking
            trackingEngine = tracking.TrackingEngine(self)
            trackingEngine.trackingRun(acqResults)
            trackResults = trackingEngine.trackResults

            # Report tracking elapsed time.
            print("   Elapsed time for tracking is %s s\n" %(
                (datetime.datetime.now() - startTime).total_seconds()) )

            # Save the complete tracking engine, including trackResults.
            with open(self.trkPklName,"wb") as fp:
                pickle.dump(trackingEngine,fp)
        else:
            # Restore the complete tracking engine and its result arrays.
            print("Skip tracking and using pickle data")
            with open(self.trkPklName,"rb") as fp:
                trackingEngine = pickle.load(fp)
            trackResults = trackingEngine.trackResults

        #--- Calculate navigation ---------------------------------------------
        # Decode navigation messages, calculate satellite positions, measure
        # pseudoranges, and determine the receiver position.
        import postNavigation
        navigationEngine = postNavigation.NavigationEngine(self)
        if not self.skipNavigation:
            navigationEngine.navigationRun(trackResults)
            navSolutions = navigationEngine.navSolutions
            eph = navigationEngine.eph
            with open(self.navPklName,"wb") as fp:
                pickle.dump((navSolutions,eph),fp)
        else:
            with open(self.navPklName,"rb") as fp:
                navSolutions, eph = pickle.load(fp)
            navigationEngine.navSolutions = navSolutions
            navigationEngine.eph = eph

        #--- Plot tracking ----------------------------------------------------
        # Plot the tracking results for all processed channels.
        if self.plotTracking:
            trackingEngine.plotTracking()

        #--- Plot navigation --------------------------------------------------
        # Plot the calculated navigation solution.
        if self.plotNavigation:
            navigationEngine.plotNavigation()
