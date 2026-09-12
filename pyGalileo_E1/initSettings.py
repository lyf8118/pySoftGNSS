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
Initialize and save Galileo E1 receiver settings.

Settings can be edited inside ``Settings`` or updated after initialization.
All settings are described in the class code.

"""

import numpy as np

class Settings(object):
    """Store receiver settings used by acquisition, tracking and navigation."""

    def __init__(self):
        """Initialize the default receiver settings.

        Returns
        -------
            None
        """
        #%% Processing settings ===============================================
        # Process 46 s so complete Galileo I/NAV pages are available.
        self.msToProcess         = 46000          #[ms]
        # Maximum number of channels used for signal processing.
        self.numberOfChannels    = 12
        # Move the starting point of processing. Can be used to start the
        # signal processing at any point in the data record (e.g. for long
        # records). fseek function is used to move the file read point,
        # therefore the offset is expressed in logical time samples, not
        # directly in bytes. I/Q files contain two scalar values per sample.
        self.skipNumberOfSamples        = 0

        #%% Raw signal file name and related parameters =======================
        # Default raw recording used by the offline receiver.
        self.fileName           = "../IF_Data_Set/L1CA_E1_B1C.bin"
        # Data type used to store each real scalar in the file.
        self.dataType           = 'int16'
        # File layouts:
        # 1 - real samples S0,S1,S2,...
        # 2 - interleaved I/Q samples I0,Q0,I1,Q1,I2,Q2,...
        self.fileType           = 2
        # Intermediate, sampling and code frequencies
        self.IF                 = 0.0              # [Hz]
        self.samplingFreq       = 30.0e6          # [Hz]
        self.codeFreqBasis      = 1.023e6         # [Hz]
        # Number of primary-code chips in one 4 ms E1B/E1C period.
        self.codeLength         = 4092            # [chips]
        # Galileo E1 carrier center frequency.
        self.carrFreqBasis      = 1575.42e6       # [Hz]
        self.signalName         = "Galileo E1"
        self.codeSubchips       = 12
        self.fineAcqPeriods     = 50
        self.acqReadCodePeriods = 51
        self.removeMean         = False

        #%% Acquisition settings ==============================================
        # Enable GPU acceleration for acquisition.
        self.gpuACQflag         = True               #  True - On; False - Off
        # Skip acquisition and restore the configured pickle file.
        self.skipAcquisition    = False
        # Acquisition pickle filename.
        self.acqPklName         = "./result_Cache/acqResults.pkl"
        # List of satellites to look for. Some satellites can be excluded to
        # speed up acquisition.
        self.acqSatelliteList   = np.arange(1,51)    # [PRN numbers]
        # One-sided band around IF searched for satellite signals.
        self.acqSearchBand      = 5000.              # [Hz]
        # Dimensionless primary-to-secondary correlation-peak threshold.
        self.acqThreshold       = 2.0
        # Frequency search step for coarse acquisition
        self.acqSearchStep      = 200                # [Hz]

        #%% Tracking loops settings ===========================================
        # Skip tracking and restore the configured pickle file.
        self.skipTracking       = False
        # Select channel scheduling for tracking.
        self.trkMode            = 1       # 0 - Channel-serial tracking;
                                          # 1 - Channel-parallel tracking;
        # E1 CBOC uses non-binary weights; the current native QPSK DLLs only
        # accept binary code tables, so use the Python reference correlator.
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
        self.dllCorrelatorSpacing   = 0.07        # [primary-code chips]
        # Carrier tracking loop parameters
        self.pllDampingRatio        = 0.7
        self.pllNoiseBandwidth      = 16.         # [Hz]
        # Integration time for DLL and PLL
        self.intTime                = 0.004       # [s]
        # Enable the E1C pilot branch in acquisition and tracking.
        self.pilotACQflag           = True
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
        self.CNoVSMinterval = 40

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
                           Rounded number of samples in one E1 code period.
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

        Returns
        -------
            None
        """
        import matplotlib.pyplot as plt
        from scipy.signal import welch
        from scipy.signal.windows.windows import hamming
        import sys

        # Validate the configured input filename and scalar data type.
        if not isinstance(self.fileName, str):
            raise TypeError('File name must be a string')
            sys.exit()

        if self.dataType not in ('int8', 'int16'):
            raise TypeError('Input IF dataType must be ', 'int8 or int16')
            sys.exit()

        #%% Generate plot of raw data =========================================
        try:
            with open(self.fileName, 'rb') as fid:
                # Move the starting point of processing. Can be used to start the
                # signal processing at any point in the data record (e.g. for long
                # records).
                if self.dataType == 'int16':
                    fid.seek(self.dataAdaptCoeff * self.skipNumberOfSamples * 2,0)
                elif self.dataType == 'int8':
                    fid.seek(self.dataAdaptCoeff * self.skipNumberOfSamples,0)

                try:
                    # Read ten complete E1-code periods of the input signal.
                    data = np.fromfile(fid, self.dataType,
                                       self.dataAdaptCoeff * 10 * self.samplesPerCode)
                except IOError:
                    # Report an input read failure.
                    print('Could not read enough data from the data file.')
                    sys.exit()

                #--- Generate plot of raw data --------------------------------
                plt.figure(100)
                plt.clf()

                #--- Frequency-domain plot ------------------------------------
                freq, Pxxf = welch(data, fs=self.samplingFreq/1e6,
                                   window = hamming(16384, False),
                                   nperseg=16384,noverlap=1024, nfft=16384)
                ax1 = plt.subplot(211)
                ax1.semilogy(freq, Pxxf)
                ax1.axis('tight')
                ax1.grid()
                ax1.set_xlabel('Frequency (MHz)')
                ax1.set_ylabel('PSD magnitude')
                ax1.set_title("Frequency domain plot")
                ax1.set_xlim(0,max(freq))

                #--- Time-domain plot -----------------------------------------
                timeScale = np.arange(0,25e-3,1/self.samplingFreq)*1000
                ax2 = plt.subplot(223)
                ax2.plot(timeScale[1:1000], data[1:1000])
                ax2.grid()
                ax2.axis('tight')
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Amplitude')
                ax2.set_title("Time domain plot")
                ax2.set_xticks(np.linspace(min(timeScale[1:1000]),
                                           max(timeScale[1:1000]), 4))

                #--- Histogram ------------------------------------------------
                ax3 = plt.subplot(224)
                ax3.hist(data, bins='auto', density=True)
                ax3.grid()
                ax3.set_xlabel('Bin')
                # Legacy label; density=True normalizes the histogram.
                ax3.set_ylabel('Number in bin')
                ax3.set_title("Histogram")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        #=== Error while opening the data file ================================
        except Exception as e:
            # There was an error, print it and exit
            print(repr(e))
            print('  Unable to read file ', self.fileName,'...')
            print('  (settings in "initSettings.py" to reconfigure)')
            sys.exit()

        print('  Raw IF data plotted ')

    def postProcessing(self):
        """Process the raw signal and plot the receiver results.

        Satellites are acquired, their code and carrier are tracked, and
        pseudoranges and receiver position solutions are calculated.

        Returns
        -------
            None
        """
        import pickle, datetime
        #--- Acquisition ------------------------------------------------------
        if not self.skipAcquisition:
            print('\nAcquisition is going...')
            startTime = datetime.datetime.now()

            # Run acquisition.
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
            print("Skip acquisition and using pickle data")
            with open(self.acqPklName,"rb") as fp:
                acqResults = pickle.load(fp)

        #--- Tracking ---------------------------------------------------------
        if not self.skipTracking:
            # Record and report the tracking start time.
            startTime = datetime.datetime.now()
            print('Tracking started at ',startTime.strftime("%Y-%m-%d %H:%M:%S"))

            # Run channel tracking.
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
            print("Skip tracking and using pickle data")
            with open(self.trkPklName,"rb") as fp:
                trackingEngine = pickle.load(fp)
            trackResults = trackingEngine.trackResults

        #--- Calculate navigation ---------------------------------------------
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
        if self.plotTracking:
            trackingEngine.plotTracking()

        #--- Plot navigation --------------------------------------------------
        if self.plotNavigation:
            navigationEngine.plotNavigation()






