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

postNavigation.py - Module Description
--------------------------------------
BDS-3 B2b B-CNAV3 navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
                       leastSquarePos, skyPlot, twosComp2dec)
from channelDecoder import LDPCDecoder

#%% BDS-3 B2b broadcast ephemeris
class Ephemeris:
    """Keep and decode one BDS-3 B2b B-CNAV3 broadcast ephemeris.

    The ephemeris for each PRN is decoded message by message. This object
    retains previously decoded messages until message types 10 and 30 provide
    the requisite orbit and clock parameters.
    """

    def __init__(self):
        """Initialize the common B-CNAV3 ephemeris structure.

        All fields are created in advance so that every satellite has the
        same structure when only one, or none, of the requisite messages has
        been decoded.

        Returns
        -------
            None
                Ephemeris fields and message-validity flags are initialized
                in this object.
        """
        # Flags for message-data decoding: elements 0, 1 and 2 correspond to
        # message types 10, 30 and 40, while element 3 represents all others.
        self.idValid = np.zeros(4, dtype=np.int32)
        # PRN number and complete-ephemeris flag.
        self.PRN = None
        self.flag = 0

        #%% Message type 10 ================================================
        # Seconds of week and ephemeris reference time.
        self.SOW = None
        self.t_oe = None
        # Satellite orbit type, semi-major-axis difference and its rate.
        self.SatType = None
        self.deltaA = self.ADot = None
        # Mean-motion difference and its rate.
        self.delta_n_0 = self.delta_n_0Dot = None
        # Mean anomaly, eccentricity and argument of perigee.
        self.M_0 = self.e = self.omega = None
        # Orbit-plane parameters and their rates.
        self.omega_0 = self.i_0 = None
        self.omegaDot = self.i_0Dot = None
        # Harmonic corrections to inclination, radius and argument of latitude.
        self.C_is = self.C_ic = None
        self.C_rs = self.C_rc = None
        self.C_us = self.C_uc = None
        # Data-, signal- and accuracy-integrity flags.
        self.DIFI = self.SIFI = self.AIFI = None

        #%% Message type 30 ================================================
        # Week number and satellite clock parameters.
        self.WN = self.t_oc = None
        self.a_0 = self.a_1 = self.a_2 = None
        # Group-delay differential of the B2bI component.
        self.T_GDB2bI = 0.0
        # Ionospheric model parameters.
        self.alpha1 = self.alpha2 = self.alpha3 = None
        self.alpha4 = self.alpha5 = self.alpha6 = None
        self.alpha7 = self.alpha8 = self.alpha9 = None
        # BDT-UTC parameters and satellite health.
        self.A_0UTC = self.A_1UTC = self.A_2UTC = None
        self.delta_t_LS = self.t_ot = self.WN_ot = None
        self.WN_LSF = self.DN = self.delta_t_LSF = None
        self.HS = None
        #%% Message type 40: BDT-to-other-GNSS time offset =================
        self.GNSS_ID = self.WN_0BGTO = self.t_0BGTO = None
        self.A_0BGTO = self.A_1BGTO = self.A_2BGTO = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary character sequence to an unsigned integer.

        Args
        ----
            bits   - str
                   Binary characters containing only ``'0'`` and ``'1'``.

        Returns
        -------
            int
                Unsigned decimal value of ``bits``; an empty string gives
                zero.
        """
        return int(bits, 2) if bits else 0

    @staticmethod
    def check_t(time):
        """Account for a beginning- or end-of-week crossover.

        Args
        ----
            time   - float
                   Time difference in seconds.

        Returns
        -------
            float
                Corrected time difference in seconds.
        """
        # Half of one BDT week [s].
        halfWeek = 302400
        # Correct differences that cross either week boundary.
        if time > halfWeek:
            return time-2*halfWeek
        if time < -halfWeek:
            return time+2*halfWeek
        return time

    @staticmethod
    def _crc24qCheck(bits):
        """Check a complete B-CNAV3 data block using CRC-24Q.

        Args
        ----
            bits   - array_like
                   Complete decoded message, including its 24 CRC bits.

        Returns
        -------
            bool
                ``True`` when polynomial division has a zero remainder.
        """
        # CRC-24Q generator polynomial without its leading x**24 term.
        register = 0
        polynomial = 0x864CFB
        # Shift the complete message through the 24-bit feedback register.
        for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
            topBit = (register >> 23) & 1
            register = ((register << 1) & 0xFFFFFF) | int(bit)
            if topBit:
                register ^= polynomial
        return register == 0

    def decodeEphemeris(self, navBitsBin):
        """Decode one 486-bit B-CNAV3 information block.

        The task is to select the necessary bits and convert them to decimal
        numbers. The first element must be the first bit of the message. For
        message contents refer to BDS-SIS-ICD-B2b-1.0. The same object is used
        on every call so that previously decoded messages are retained.

        Args
        ----
            navBitsBin - str
                       B-CNAV3 message represented by 486 binary characters.

        Returns
        -------
            None
                Decoded ephemeris fields and validity flags are stored in this
                object.
        """
        #%% Preparation for data-message decoding ==========================
        # Check that the input has the form required by the MATLAB decoder.
        if not isinstance(navBitsBin, str):
            raise TypeError("The parameter BITS must be a character array!")
        if len(navBitsBin) < 486:
            raise ValueError("The parameter BITS must contain 486 bits!")

        # Pi used by the BDS coordinate model.
        bdsPi = 3.1415926535898
        # Decode the six-bit message type.
        messageType = self.bin2dec(navBitsBin[0:6])

        #%% Message type 10 ================================================
        # Message type 10 provides the data required to calculate satellite
        # position and contains both parts of the broadcast ephemeris.
        if messageType == 10:
            # Mark message type 10 as decoded and retain the first SOW.
            self.idValid[0] = 10
            if self.SOW is None:
                self.SOW = self.bin2dec(navBitsBin[6:26])
            # Ephemeris reference time and satellite orbit type.
            self.t_oe = self.bin2dec(navBitsBin[30:41])*300
            satType = self.bin2dec(navBitsBin[41:43])
            self.SatType = {1: "GEO", 2: "IGSO", 3: "MEO"}.get(satType)
            # Semi-major-axis and mean-motion parameters.
            self.deltaA = twosComp2dec(navBitsBin[43:69])*2**-9
            self.ADot = twosComp2dec(navBitsBin[69:94])*2**-21
            self.delta_n_0 = twosComp2dec(navBitsBin[94:111])*2**-44*bdsPi
            self.delta_n_0Dot = twosComp2dec(navBitsBin[111:134])*2**-57*bdsPi
            # Keplerian orbit parameters.
            self.M_0 = twosComp2dec(navBitsBin[134:167])*2**-32*bdsPi
            self.e = self.bin2dec(navBitsBin[167:200])*2**-34
            self.omega = twosComp2dec(navBitsBin[200:233])*2**-32*bdsPi
            self.omega_0 = twosComp2dec(navBitsBin[233:266])*2**-32*bdsPi
            self.i_0 = twosComp2dec(navBitsBin[266:299])*2**-32*bdsPi
            self.omegaDot = twosComp2dec(navBitsBin[299:318])*2**-44*bdsPi
            self.i_0Dot = twosComp2dec(navBitsBin[318:333])*2**-44*bdsPi
            # Harmonic correction coefficients.
            self.C_is = twosComp2dec(navBitsBin[333:349])*2**-30
            self.C_ic = twosComp2dec(navBitsBin[349:365])*2**-30
            self.C_rs = twosComp2dec(navBitsBin[365:389])*2**-8
            self.C_rc = twosComp2dec(navBitsBin[389:413])*2**-8
            self.C_us = twosComp2dec(navBitsBin[413:434])*2**-30
            self.C_uc = twosComp2dec(navBitsBin[434:455])*2**-30
            # Data-, signal- and accuracy-integrity flags.
            self.DIFI = self.bin2dec(navBitsBin[455:456])
            self.SIFI = self.bin2dec(navBitsBin[456:457])
            self.AIFI = self.bin2dec(navBitsBin[457:458])

        #%% Message type 30 ================================================
        # Message type 30 contains clock, group-delay, ionospheric and
        # BDT-to-UTC parameters.
        elif messageType == 30:
            # Mark message type 30 as decoded and retain the first SOW.
            self.idValid[1] = 30
            if self.SOW is None:
                self.SOW = self.bin2dec(navBitsBin[6:26])
            # Week number, clock model and B2bI group-delay differential.
            self.WN = self.bin2dec(navBitsBin[26:39])
            self.t_oc = self.bin2dec(navBitsBin[43:54])*300
            self.a_0 = twosComp2dec(navBitsBin[54:79])*2**-34
            self.a_1 = twosComp2dec(navBitsBin[79:101])*2**-50
            self.a_2 = twosComp2dec(navBitsBin[101:112])*2**-66
            self.T_GDB2bI = twosComp2dec(navBitsBin[112:124])*2**-34
            # Ionospheric model parameters.
            self.alpha1 = self.bin2dec(navBitsBin[124:134])*2**-3
            self.alpha2 = twosComp2dec(navBitsBin[134:142])*2**-3
            self.alpha3 = self.bin2dec(navBitsBin[142:150])*2**-3
            self.alpha4 = self.bin2dec(navBitsBin[150:158])*2**-3
            self.alpha5 = self.bin2dec(navBitsBin[158:166])*2**-3
            self.alpha6 = twosComp2dec(navBitsBin[166:174])*2**-3
            self.alpha7 = twosComp2dec(navBitsBin[174:182])*2**-3
            self.alpha8 = twosComp2dec(navBitsBin[182:190])*2**-3
            self.alpha9 = twosComp2dec(navBitsBin[190:198])*2**-3
            # BDT-UTC and leap-second parameters.
            self.A_0UTC = twosComp2dec(navBitsBin[198:214])*2**-35
            self.A_1UTC = twosComp2dec(navBitsBin[214:227])*2**-51
            self.A_2UTC = twosComp2dec(navBitsBin[227:234])*2**-68
            self.delta_t_LS = twosComp2dec(navBitsBin[234:242])
            self.t_ot = self.bin2dec(navBitsBin[242:258])*2**4
            self.WN_ot = self.bin2dec(navBitsBin[258:271])
            self.WN_LSF = self.bin2dec(navBitsBin[271:284])
            self.DN = self.bin2dec(navBitsBin[284:287])
            self.delta_t_LSF = twosComp2dec(navBitsBin[287:295])
            # Satellite health.
            self.HS = self.bin2dec(navBitsBin[460:462])

        #%% Message type 40 ================================================
        # Message type 40 contains BDS-to-other-GNSS time-offset parameters.
        elif messageType == 40:
            # BDT-to-other-GNSS time-offset parameters.
            self.idValid[2] = 40
            if self.SOW is None:
                self.SOW = self.bin2dec(navBitsBin[6:26])
            self.GNSS_ID = self.bin2dec(navBitsBin[26:29])
            self.WN_0BGTO = self.bin2dec(navBitsBin[29:42])
            self.t_0BGTO = self.bin2dec(navBitsBin[42:58])*2**4
            self.A_0BGTO = twosComp2dec(navBitsBin[58:74])*2**-35
            self.A_1BGTO = twosComp2dec(navBitsBin[74:87])*2**-51
            self.A_2BGTO = twosComp2dec(navBitsBin[87:94])*2**-68
        else:
            # Other message types are identified but not decoded at present.
            self.idValid[3] = messageType
            self.SOW = self.bin2dec(navBitsBin[6:26])

        # Types 10 and 30 together provide complete orbit and clock data.
        self.flag = int(self.idValid[0] == 10 and self.idValid[1] == 30)

    @classmethod
    def NAVdecoding(cls, trackResult, settings):
        """Find B-CNAV3 frames, perform LDPC/CRC decoding and decode data.

        The method finds the first preamble occurrences in the prompt tracking
        stream, resolves the BPSK polarity ambiguity, performs B-CNAV3 LDPC
        decoding and CRC-24Q checking, and accumulates ephemeris messages.

        Args
        ----
            trackResult - tracking.Channel
                        Prompt I/Q tracking outputs for one satellite.
            settings    - initSettings.Settings
                        Receiver settings; retained for interface consistency
                        with the other navigation decoders.

        Returns
        -------
            eph            - Ephemeris
                           Decoded satellite ephemeris.
            firstSubFrame  - int or float
                           Starting position of the first valid B-CNAV3 frame
                           in the 1 ms tracking stream; infinity if none is
                           detected.
            TOW            - float
                           Seconds of week of the first valid message;
                           infinity if no valid frame is detected.
        """
        #--- Initialize ephemeris structure --------------------------------
        # Create the full structure even when none of the requisite messages
        # is decoded for this satellite.
        eph = cls()
        firstSubFrame = np.inf
        TOW = np.inf
        #%% Bit and frame synchronization ==================================
        # Sixteen-bit B-CNAV3 preamble in antipodal form.
        preambleBits = np.array(
            (-1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1), dtype=np.int8)
        #--- Extract raw soft information ----------------------------------
        rawSoftI = np.asarray(trackResult.I_P).reshape(-1)
        rawSoftQ = np.asarray(trackResult.Q_P).reshape(-1)
        # Make sure the I and Q branches have the same length.
        length = min(rawSoftI.size, rawSoftQ.size)
        rawSoftI = rawSoftI[:length]
        rawSoftQ = rawSoftQ[:length]
        # Generate hard decisions only for preamble synchronization.
        hardBits = np.where(rawSoftI > 0, 1, -1)
        #--- Correlate tracking output with the preamble --------------------
        # The ideal correlation peak is 16.
        correlation = np.correlate(hardBits, preambleBits, mode="valid")
        candidates = np.flatnonzero(np.abs(correlation) > 15.9)

        #%% B-CNAV3 decoding ================================================
        for candidate in candidates:
            # One complete B-CNAV3 frame contains 1000 soft tracking samples.
            if length-candidate < 1000:
                continue
            #--- Extract one complete B-CNAV3 frame -------------------------
            navSoftI = rawSoftI[candidate:candidate+1000].copy()
            navSoftQ = rawSoftQ[candidate:candidate+1000].copy()
            navHard = hardBits[candidate:candidate+1000].copy()

            #--- Polarity correction ---------------------------------------
            # BPSK phase ambiguity rotates the complete constellation by 180
            # degrees. Restore the expected preamble and soft-bit polarity.
            if not np.array_equal(navHard[:16], preambleBits):
                navHard = -navHard
                navSoftI = -navSoftI
                navSoftQ = -navSoftQ
            if not np.array_equal(navHard[:16], preambleBits):
                continue
            # Verify that the frame belongs to the tracked satellite.
            PRN = int("".join(map(
                str, (navHard[16:22] < 0).astype(np.uint8))), 2)
            if PRN != trackResult.PRN or not 6 <= PRN <= 58:
                continue
            eph.PRN = PRN

            #--- B2b B-CNAV3 LDPC/EMS decoding ------------------------------
            decodedBits = LDPCDecoder.B2bLDPCDecoder(navSoftI, navSoftQ)
            #--- CRC-24Q check ----------------------------------------------
            if not cls._crc24qCheck(decodedBits):
                continue
            #--- Decode ephemeris message by message ------------------------
            eph.decodeEphemeris("".join(decodedBits.astype(str)))
            # Retain the first valid frame position and its SOW.
            if np.isinf(firstSubFrame):
                firstSubFrame = int(candidate)
                TOW = eph.SOW if eph.SOW is not None else np.inf
        return eph, firstSubFrame, TOW

    def satpos(self, transmitTime):
        """Calculate B2b satellite position and clock correction.

        Args
        ----
            transmitTime - float
                         Signal transmission time in BDT seconds of week.

        Returns
        -------
            satPosition  - ndarray, shape (3,)
                         Satellite ECEF coordinates ``[X, Y, Z]`` in metres.
            satClkCorr   - float
                         Satellite clock correction in seconds.
        """
        #%% Initialize constants ===========================================
        # BDS coordinate-model constants.
        bdsPi = 3.1415926535898
        # Earth rotation rate [rad/s].
        OmegaE = 7.2921150e-5
        # Earth's universal gravitational parameter [m^3/s^2].
        mu = 3.986004418e14
        # Relativistic correction constant [s/m^(1/2)].
        F = -4.44280730904398e-10
        # Semi-major-axis references for MEO and IGSO/GEO satellites [m].
        A_ref_MEO = 27906100
        A_ref_IGSO_GEO = 42162200

        #%% Find initial satellite clock correction ========================
        # Time from clock reference epoch, corrected for week crossover.
        dt = self.check_t(transmitTime-self.t_oc)
        # Apply the clock model and B2bI group-delay correction.
        satClkCorr = (self.a_2*dt+self.a_1)*dt+self.a_0-self.T_GDB2bI
        time = transmitTime-satClkCorr

        #%% Find satellite position ========================================
        # Time from the ephemeris reference epoch.
        tk = self.check_t(time-self.t_oe)
        # Select the semi-major-axis reference for this orbit type.
        A_ref = A_ref_MEO if self.SatType == "MEO" else A_ref_IGSO_GEO
        # Restore semi-major axis at the reference and transmission epochs.
        A0 = A_ref+self.deltaA
        A = A0+self.ADot*tk
        # Reference mean motion and broadcast mean-motion correction.
        n0 = np.sqrt(mu/A0**3)
        delta_n = self.delta_n_0+0.5*self.delta_n_0Dot*tk
        # Mean anomaly reduced to one revolution.
        M = np.fmod(self.M_0+(n0+delta_n)*tk+2*bdsPi, 2*bdsPi)
        # Iteratively solve Kepler's equation for eccentric anomaly.
        E = M
        for _ in range(10):
            oldE = E
            E = M+self.e*np.sin(E)
            if abs(np.fmod(E-oldE, 2*bdsPi)) < 1e-12:
                break
        # Reduce eccentric anomaly to one revolution.
        E = np.fmod(E+2*bdsPi, 2*bdsPi)
        # Relativistic clock correction and true anomaly.
        dtr = F*self.e*np.sqrt(A0)*np.sin(E)
        nu = np.arctan2(np.sqrt(1-self.e**2)*np.sin(E), np.cos(E)-self.e)
        # Argument of latitude before harmonic corrections.
        phi = np.fmod(nu+self.omega, 2*bdsPi)
        # Correct argument of latitude, orbital radius and inclination.
        u = phi+self.C_uc*np.cos(2*phi)+self.C_us*np.sin(2*phi)
        r = A*(1-self.e*np.cos(E))+self.C_rc*np.cos(2*phi)+self.C_rs*np.sin(2*phi)
        i = self.i_0+self.i_0Dot*tk+self.C_ic*np.cos(2*phi)+self.C_is*np.sin(2*phi)
        # Longitude of ascending node relative to the Greenwich meridian.
        Omega = (self.omega_0+(self.omegaDot-OmegaE)*tk-OmegaE*self.t_oe)
        Omega = np.fmod(Omega+2*bdsPi, 2*bdsPi)
        # Satellite coordinates in the orbital plane.
        xp = r*np.cos(u)
        yp = r*np.sin(u)
        #--- Compute satellite ECEF coordinates ----------------------------
        satPosition = np.array((
            xp*np.cos(Omega)-yp*np.cos(i)*np.sin(Omega),
            xp*np.sin(Omega)+yp*np.cos(i)*np.cos(Omega),
            yp*np.sin(i)))
        #%% Include relativistic and group-delay terms in clock correction ==
        satClkCorr = ((self.a_2*dt+self.a_1)*dt+self.a_0
                      - self.T_GDB2bI+dtr)
        return satPosition, satClkCorr


#%% Navigation results
class NavSolutions:
    """Store measured pseudoranges and receiver navigation solutions.

    The results contain receiver clock error and receiver coordinates in
    several coordinate systems, including ECEF and UTM.

    Args
    ----
        numberOfChannels - int
                         Number of receiver channels.
        measNrSum   - int
                    Number of navigation-solution measurement points.
    """

    def __init__(self, numberOfChannels, measNrSum):
        """Allocate arrays for all navigation-solution measurement points.

        Args
        ----
            numberOfChannels - int
                             Number of receiver channels.
            measNrSum   - int
                        Number of navigation-solution measurement points.
        Returns
        -------
            None
                Navigation-result arrays are allocated in this object.
        """
        channelShape = (numberOfChannels, measNrSum)

        # Satellite information for each channel and measurement epoch.
        # NaN elevation/azimuth values keep unused satellites out of skyPlot.
        self.PRN          = np.zeros(channelShape, dtype=np.int16)
        self.el           = np.full(channelShape, np.nan)
        self.az           = np.full(channelShape, np.nan)
        self.transmitTime = np.full(channelShape, np.nan)
        self.satClkCorr   = np.full(channelShape, np.nan)
        self.rawP         = np.full(channelShape, np.nan)
        self.correctedP   = np.full(channelShape, np.nan)

        # Dilution of precision: GDOP, PDOP, HDOP, VDOP and TDOP.
        self.DOP = np.zeros((5, measNrSum))

        # Receiver ECEF position and clock bias [m].
        self.X  = np.full(measNrSum, np.nan)
        self.Y  = np.full(measNrSum, np.nan)
        self.Z  = np.full(measNrSum, np.nan)
        self.dt = np.full(measNrSum, np.nan)

        # Measurement location and receiver BDT.
        self.currMeasSample = np.full(measNrSum, -1, dtype=np.int64)
        self.localTime = np.full(measNrSum, np.nan)

        # WGS-84 geodetic and legacy ED50/UTM coordinates.
        self.latitude  = np.full(measNrSum, np.nan)
        self.longitude = np.full(measNrSum, np.nan)
        self.height    = np.full(measNrSum, np.nan)
        self.utmZone   = None
        self.E         = np.full(measNrSum, np.nan)
        self.N         = np.full(measNrSum, np.nan)
        self.U         = np.full(measNrSum, np.nan)


#%% Post-navigation engine
class NavigationEngine:
    """Calculate and plot receiver navigation solutions.

    Args
    ----
        settings    - object
                    Receiver settings.
    """

    def __init__(self, settings):
        """Initialize an empty post-navigation engine.

        Args
        ----
            settings    - object
                        Receiver settings.
        Returns
        -------
            None
                Receiver settings and empty navigation outputs are stored in
                this object.
        """
        self.settings = settings
        self.navSolutions = []
        self.eph = {}
        self.searchIndex = None

    def navigationRun(self, trackResults):
        """Calculate navigation solutions for the receiver.

        The method calculates pseudoranges and receiver positions. It then
        converts the ECEF solution to WGS-84 geodetic and UTM coordinates.

        Args
        ----
            trackResults - iterable of tracking.Channel
                         Results from the tracking function.
        Returns
        -------
            None
                Measured pseudoranges, receiver clock error, receiver
                coordinates, and satellite ephemerides are stored in
                ``navSolutions`` and ``eph``.
        """
        settings = self.settings

        # Reset output and per-run search state.
        self.navSolutions = []
        self.eph = {}
        # Per-channel cache used to accelerate consecutive sample searches.
        self.searchIndex = np.zeros(
            settings.numberOfChannels, dtype=np.int64)

        #%% Check whether enough data are available for a navigation solution
        # Satellite coordinates require complete B-CNAV3 ephemeris and clock
        # data. Tracking starts at an arbitrary point in the message, so at
        # least 24 seconds of tracking output are required.
        if settings.msToProcess < 24000:
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid B-CNAV3 frame in the 1 ms B2bI
        # primary-code stream. Infinity means no frame was decoded.
        subFrameStart = np.full(settings.numberOfChannels, np.inf)
        # TOW of the first valid message [s]; infinity indicates no detection.
        TOW = np.full(settings.numberOfChannels, np.inf)

        #--- Make a list excluding channels not in tracking lock ----------
        # Make a list containing only channels that remain in tracking lock.
        activeChnList = [
            channelNr for channelNr, channel in enumerate(trackResults)
            if channel.lockFlag ]

        #%% Decode ephemerides
        
        for channelNr in activeChnList.copy():
            #--- Get the PRN of the current channel ---------------------------
            PRN = trackResults[channelNr].PRN
            print(f"Decoding B-CNAV3 for PRN {PRN:02d} -----------------")

            currentEph, subFrameStart[channelNr], TOW[channelNr] = (
                Ephemeris.NAVdecoding(trackResults[channelNr], settings))
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all requisite B-CNAV3 messages from
            # subsequent positioning.
            if currentEph.flag != 1:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
                activeChnList.remove(channelNr)
            elif currentEph.HS != 0:
                print(f"    PRN {PRN:02d} is unhealthy and is excluded!")
                activeChnList.remove(channelNr)
            else:
                print(f"    The requisite messages for PRN {PRN:02d} "
                      "all decoded!")

        #%% Check whether at least four satellites remain
        # A three-dimensional receiver position and clock bias require four
        # or more satellites with decoded ephemerides.
        if len(activeChnList) < 4:
            print("Too few satellites with ephemeris data for position "
                  "calculations. Exiting!")
            return

        #%% Set measurement-time points and step
        # Find the common IF-sample interval with measurements available in
        # every ephemeris-ready channel. The one-sample margin at each end
        # keeps subsequent tracking-array searches inside their boundaries.
        sampleStart = max(trackResults[channelNr].absoluteSample[
            int(subFrameStart[channelNr])]
            for channelNr in activeChnList) + 1
        
        sampleEnd = min(trackResults[channelNr].absoluteSample[-1]
            for channelNr in activeChnList) - 1

        #--- Measurement step in units of IF samples --------------------------
        # Navigation-solution interval expressed in logical IF samples.
        measSampleStep = int(np.fix(
                            settings.samplingFreq*settings.navSolPeriod/1000))
        #--- Number of measurement points from start to end -------------------
        # Number of measurement epochs in the common sample interval.
        measNrSum = int(np.fix((sampleEnd-sampleStart)/measSampleStep))

        #%% Initialize navigation processing
        # Include all ephemeris-ready satellites in the first solution. No
        # receiver position is yet available for calculating elevation.
        satElev = np.full(settings.numberOfChannels, np.inf)

        # Filter activeChnList by elevation at every measurement epoch.

        # None marks the first position calculation. Thereafter localTime is
        # advanced by the navigation-solution interval.
        localTime = None

        # Preallocate results in MATLAB's channel-by-epoch orientation.
        self.navSolutions = NavSolutions(settings.numberOfChannels, measNrSum)
        navSolutions = self.navSolutions

        #%% Satellite and receiver position calculations ======================
        print("Positions are being computed. Please wait...")
        for currMeasNr in range(measNrSum):
            #%% Initialize the current measurement
            # Exclude satellites below the elevation mask using elevations
            # calculated at the preceding measurement epoch.
            activeChnList = [channelNr for channelNr in activeChnList
                             if satElev[channelNr] >= settings.elevationMask]
            
            # Save the PRNs used for this position calculation.
            navSolutions.PRN[activeChnList, currMeasNr] = [
                trackResults[channelNr].PRN
                for channelNr in activeChnList ]

            # Elevation, azimuth, transmit-time, and clock-correction arrays
            # were preallocated with NaN. Satellites excluded by the elevation
            # mask therefore do not jump to position (0, 0) in the sky plot.

            # Absolute IF-sample position of the current measurement epoch.
            currMeasSample = int(sampleStart+measSampleStep*currMeasNr)

            #%% Find pseudoranges
            # Raw pseudorange = (localTime - transmitTime) * c [m].
            # The pseudorange and transmit-time outputs are channel-ordered
            # vectors with one entry for every receiver channel.
            (rawP, transmitTime, localTime,_) = calculatePseudoranges(
                trackResults, subFrameStart, TOW, currMeasSample,
                localTime, activeChnList, settings, self.searchIndex)
            
            navSolutions.rawP[:, currMeasNr] = rawP
            
            navSolutions.transmitTime[activeChnList, currMeasNr] = (
                transmitTime[activeChnList])

            #%% Find satellite positions and clock corrections
            # Calculate the outputs in the same explicit channel/PRN order
            # as the pseudorange measurements.
            satPositions = np.zeros((3, len(activeChnList)))
            satClkCorr = np.zeros(len(activeChnList))
            
            for satNr, channelNr in enumerate(activeChnList):
                PRN = trackResults[channelNr].PRN
                satPositions[:, satNr], satClkCorr[satNr] = (
                    self.eph[PRN].satpos(transmitTime[channelNr]))
                
            navSolutions.satClkCorr[activeChnList, currMeasNr] = satClkCorr

            #%% Find receiver position
            # Elevation masking may reduce the current set; four or more
            # satellites are still required for a 3-D position and clock bias.
            if len(activeChnList) > 3:
                #=== Calculate receiver position ==============================
                # Correct pseudoranges for satellite clock errors.
                clkCorrRawP = (rawP[activeChnList] + satClkCorr * settings.c)

                # Estimate receiver position and clock bias.
                xyzdt, el, az, dop = leastSquarePos(
                                            satPositions, clkCorrRawP, settings)
                
                navSolutions.el[activeChnList, currMeasNr] = el
                navSolutions.az[activeChnList, currMeasNr] = az
                navSolutions.DOP[:, currMeasNr] = dop

                #=== Save results =============================================
                # Save receiver ECEF position [m].
                navSolutions.X[currMeasNr] = xyzdt[0]
                navSolutions.Y[currMeasNr] = xyzdt[1]
                navSolutions.Z[currMeasNr] = xyzdt[2]
                # As in MATLAB, store zero clock bias for the first solution;
                # subsequent values are the estimated range-form bias [m].
                navSolutions.dt[currMeasNr] = (
                    0 if currMeasNr == 0 else xyzdt[3])

                #=== Correct local time using clock-error estimation ==========
                # Correct receiver local time using clock bias [m].
                localTime -= xyzdt[3]/settings.c
                navSolutions.localTime[currMeasNr] = localTime
                
                # Save the absolute IF-sample location of this solution.
                navSolutions.currMeasSample[currMeasNr] = currMeasSample
                
                # Use current satellite elevations for the next epoch's mask.
                satElev = navSolutions.el[:, currMeasNr].copy()

                #=== Correct pseudoranges for clock errors ====================
                # Correct pseudoranges for satellite and receiver clocks.
                navSolutions.correctedP[activeChnList, currMeasNr] = (
                      rawP[activeChnList] + satClkCorr * settings.c - xyzdt[3])

                #%% Coordinate conversion
                #=== Convert to geodetic coordinates ==========================
                # Convert ECEF to WGS-84 geodetic coordinates.
                (navSolutions.latitude[currMeasNr], 
                 navSolutions.longitude[currMeasNr],
                 navSolutions.height[currMeasNr]) = cart2geo(
                    xyzdt[0], xyzdt[1], xyzdt[2], 5)

                #=== Convert to the UTM coordinate system =====================
                # Convert ECEF to the legacy ED50/UTM coordinates.
                # Select the UTM longitudinal zone.
                navSolutions.utmZone = findUtmZone(
                    navSolutions.latitude[currMeasNr], navSolutions.longitude[currMeasNr])
                
                (navSolutions.E[currMeasNr],navSolutions.N[currMeasNr],
                 navSolutions.U[currMeasNr]) = cart2utm(xyzdt[0], xyzdt[1],
                                                xyzdt[2],navSolutions.utmZone)
                # E, N and U contain the receiver position in the local
                # east, north and up coordinate directions.
            else:
                #--- Not enough satellites to find a 3-D position -------------
                # There are not enough satellites for a 3-D position. Result
                # arrays retain their preallocated NaN/zero values.
                print(f"   Measurement No. {currMeasNr+1}: Not enough "
                      "information for position solution.")

                # Missing solutions retain NaN and are automatically omitted
                # from the plots. DOP retains zeros because that is more
                # convenient for plotting; later processing must exclude NaN
                # values when computing statistics.

                # Known MATLAB limitation: positions/elevations of masked
                # satellites are not updated, so a rising satellite cannot
                # re-enter after it has been excluded by the elevation mask.

            # === Update local time by one measurement step ===================
            # Advance receiver time by one navigation-solution interval.
            localTime += measSampleStep/settings.samplingFreq

    def plotNavigation(self):
        """Plot coordinate variations over time and a receiver position.

        Receiver coordinates are plotted in the UTM system. Coordinate
        offsets are plotted when true UTM receiver coordinates are provided.

        Returns
        -------
            None
                The navigation solution stored in ``navSolutions`` is
                displayed using the receiver settings stored in ``settings``.
        """
        #%% Plot results if the necessary navigation data exist
        if (not self.navSolutions
                or not np.any(np.isfinite(self.navSolutions.E))):
            print("plotNavigation: No navigation data to plot.")
            return

        import matplotlib.pyplot as plt

        navSolutions = self.navSolutions
        settings     = self.settings

        #%% If no reference is provided, use the mean position ================
        # If no reference position is provided, use the mean solution.
        if (np.isnan(settings.truePositionE)
                or np.isnan(settings.truePositionN)
                or np.isnan(settings.truePositionU)):
            #=== Compute mean values ==========================================
            # Remove NaN values when calculating the mean; otherwise the
            # result returned by mean would also be NaN.
            refE = np.nanmean(navSolutions.E)
            refN = np.nanmean(navSolutions.N)
            refU = np.nanmean(navSolutions.U)
            refPointText = "Mean position"
        else:
            refE = settings.truePositionE
            refN = settings.truePositionN
            refU = settings.truePositionU
            refPointText = "Reference position"

        # Figure 300 follows MATLAB and avoids overwriting auto-numbered user
        # figures when navigation plots are closed and reopened.
        #=== Select or create and clear the figure ============================
        figure = plt.figure(300, figsize=(12, 8), clear=True)
        figure.canvas.manager.set_window_title("Navigation solutions")

        #--- Draw axes --------------------------------------------------------
        # Draw the three plotting axes.
        grid = figure.add_gridspec(4, 2)
        coordinateAxes = figure.add_subplot(grid[:2, :])
        positionAxes = figure.add_subplot(grid[2:, 0])
        skyAxes = figure.add_subplot(grid[2:, 1])

        #%% Plot all figures
        #--- Coordinate differences in the UTM system -------------------------
        # Coordinate variations in the legacy UTM system.
        coordinateAxes.plot(navSolutions.E-refE, label="E")
        coordinateAxes.plot(navSolutions.N-refN, label="N")
        coordinateAxes.plot(navSolutions.U-refU, label="U")
        coordinateAxes.set_title("Coordinate variations in UTM system")
        coordinateAxes.set_xlabel(
            f"Measurement period: {settings.navSolPeriod:g} ms")
        coordinateAxes.set_ylabel("Variations (m)")
        coordinateAxes.legend()
        coordinateAxes.grid(True)

        #--- Position plot in the UTM system ----------------------------------
        # Horizontal position offsets in the legacy UTM system.
        positionAxes.plot(navSolutions.E-refE, navSolutions.N-refN, "+",
                          label="Measurements")
        # Plot the reference point at the origin of the offset coordinates.
        positionAxes.plot(0, 0, "r+", markersize=10,
                          label=refPointText)
        positionAxes.set_title("Positions in UTM system")
        positionAxes.set_xlabel("East (m)")
        positionAxes.set_ylabel("North (m)")
        positionAxes.axis("equal")
        positionAxes.grid(True)
        positionAxes.legend()

        #--- Satellite sky plot -----------------------------------------------
        # Satellite sky plot. The utility follows MATLAB's
        # satellite-by-epoch layout and spherical elevation mapping.
        skyPlot(skyAxes, navSolutions.az, navSolutions.el,
                navSolutions.PRN[:, 0])
        skyAxes.set_title(
            f"Sky plot (mean PDOP: {np.mean(navSolutions.DOP[1]):.2f})")

        figure.tight_layout()
        plt.show()


__all__ = ["Ephemeris", "NavSolutions", "NavigationEngine"]
