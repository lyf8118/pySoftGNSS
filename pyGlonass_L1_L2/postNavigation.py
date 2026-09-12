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
GLONASS L1/L2 navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot,
)


#%% Broadcast ephemeris
class Ephemeris:
    """Keep a common GLONASS ephemeris structure for one satellite.

    This ensures that every satellite has the same structure when only one,
    or none, of the requisite navigation strings has been decoded.
    """

    def __init__(self):
        """Initialize the GLONASS ephemeris fields.

        Returns
        -------
            None
                Ephemeris fields are initialized in this object.
        """
        self.K: int | None = None            # GLONASS frequency channel

        #--- It is string 1 ---------------------------------------------------
        # It contains TOD, flag P1, and the x coordinate, velocity and
        # acceleration.
        self.P1:   int | None = None
        self.TOD: float | None = None
        self.xDis: float | None = None       # [km]
        self.xVel: float | None = None       # [km/s]
        self.xAcc: float | None = None       # [km/s^2]

        #--- It is string 2 ---------------------------------------------------
        # It contains ephemeris reference time, P2, health, and the y
        # coordinate, velocity and acceleration.
        self.P2: int | None = None
        self.B:  int | None = None
        self.tb: float | None = None         # [s]
        self.yDis: float | None = None       # [km]
        self.yVel: float | None = None       # [km/s]
        self.yAcc: float | None = None       # [km/s^2]

        #--- It is string 3 ---------------------------------------------------
        # It contains frequency offset, P, P3, health, and the z coordinate,
        # velocity and acceleration.
        self.P3: int | None = None
        self.gam: float | None = None
        self.P: int | None = None
        self.health: int | None = None
        self.zDis: float | None = None       # [km]
        self.zVel: float | None = None       # [km/s]
        self.zAcc: float | None = None       # [km/s^2]

        #--- It is string 4 ---------------------------------------------------
        # It contains the fourth part of the ephemeris parameters.
        self.tau_n: float | None = None      # [s]
        self.dtau: float | None = None       # [s]
        self.E: int | None = None            # [days]
        self.P4: int | None = None
        self.FT: int | None = None
        self.M: int | None = None
        self.n: int | None = None
        self.days: int | None = None         # [days]

        #--- It is string 5 ---------------------------------------------------
        # It contains the four-year interval number and the GLONASS-to-UTC(SU)
        # time-scale correction.
        self.N4: int | None = None
        self.tau_c: float | None = None
        self.validFlag: int | None = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary sequence to an unsigned decimal integer.

        Args
        ----
            bits        - array-like
                        Binary field containing only zero and one values.

        Returns
        -------
            value       - int
                        Unsigned decimal value of the binary field.
        """
        return int("".join(str(int(bit)) for bit in bits), 2)

    @staticmethod
    def checkPhase(string):
        """Check the phase of the supplied 85-bit string.

        The first bit of the string is used for the calculation.

        Args
        ----
            string      - array-like
                        Eighty-five bits from the navigation message. The
                        array must contain only zero and one values.

        Returns
        -------
            string      - numpy.ndarray
                        String with corrected polarity of the data bits.
        """
        string = np.asarray(string, dtype=np.int8).copy()
        if string[0] == 1:
            string = 1 - string
        return string

    @staticmethod
    def dataVerification(data):
        """Verify one GLONASS string with the ICD checking algorithm.

        This function uses the data-verification algorithm described in the
        GLONASS ICD. It does not correct the data; it only checks whether the
        supplied string is correct.

        Args
        ----
            data        - array-like
                        Eighty-five bits: 77 navigation bits and eight
                        Hamming-code checking bits.

        Returns
        -------
            status      - bool
                        True if the string is correct; otherwise False.
        """
        data = np.asarray(data, dtype=np.int8)
        index1 = np.array([9,10,12,13,15,17,19,20,22,24,26,28,30,32,34,35,
            37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,66,68,70,72,74,
            76,78,80,82,84]) - 1
        index2 = np.array([9,11,12,14,15,18,19,21,22,25,26,29,30,33,34,36,
            37,40,41,44,45,48,49,52,53,56,57,60,61,64,65,67,68,71,72,75,
            76,79,80,83,84]) - 1
        index3 = np.r_[9:12, 15:19, 22:26, 30:34, 37:41, 45:49, 53:57,
                       61:65, 68:72, 76:80, 84]
        index4 = np.r_[12:19, 26:34, 41:49, 57:65, 72:80]
        index5 = np.r_[19:34, 49:65, 80:85]
        index6 = np.arange(34, 65)

        C = np.empty(8, dtype=np.int8)
        C[0] = np.logical_xor(data[0], np.sum(data[index1]) % 2)
        C[1] = np.logical_xor(data[1], np.sum(data[index2]) % 2)
        C[2] = np.logical_xor(data[2], np.sum(data[index3]) % 2)
        C[3] = np.logical_xor(data[3], np.sum(data[index4]) % 2)
        C[4] = np.logical_xor(data[4], np.sum(data[index5]) % 2)
        C[5] = np.logical_xor(data[5], np.sum(data[index6]) % 2)
        C[6] = np.logical_xor(data[6], np.sum(data[65:85]) % 2)
        C[7] = np.logical_xor(np.sum(data[:8]) % 2,
                              np.sum(data[8:85]) % 2)
        return bool(not np.any(C) or (np.count_nonzero(C[:7]) == 1 and C[7]))

    @staticmethod
    def check_t(time):
        """Account for beginning- or end-of-day crossover.

        Args
        ----
            time        - float
                        Time in seconds.

        Returns
        -------
            corrTime    - float
                        Corrected time in seconds.
        """
        half_day = 43200.0
        if time > half_day:
            return time - 2*half_day
        if time < -half_day:
            return time + 2*half_day
        return time

    def decodeEphemeris(self, navBiBinaryBitsSamples):
        """Decode GLONASS ephemerides and TOD from 15 navigation strings.

        The stream contains 1275 bi-binary navigation bits represented by ten
        tracking accumulations per bi-binary half-bit. The first element must
        be the first bit of a string. The ID of the first string in the array
        is not important.

        Args
        ----
            navBiBinaryBitsSamples - array-like
                        Samples of the navigation message covering 15 strings.

        Returns
        -------
            TOD         - float
                        Time of day of the first string in the bit stream, in
                        seconds. Infinity is returned when string 1 is absent.

        Notes
        -----
            The decoded satellite ephemeris is stored in this object.
        """
        samples = np.asarray(navBiBinaryBitsSamples)
        if samples.size < 25500:
            raise ValueError("Fifteen complete GLONASS strings are required.")

        #--- Group every ten tracking values into one bi-binary half-bit ------
        # Sum the ten samples to obtain the best half-bit estimate, then
        # threshold the sums to binary zero and one values.
        navBiBinaryBits = samples[:25500].reshape(-1, 10).sum(axis=1) > 0

        #--- Convert from bi-binary symbols to relative code -----------------
        relNavBits = ((navBiBinaryBits[0::2].astype(np.int8) -
                       navBiBinaryBits[1::2].astype(np.int8) + 1) // 2)

        #--- Convert from relative code to data and checking bits -------------
        navBits = np.empty(1275, dtype=np.int8)
        navBits[0] = 0
        navBits[1:] = np.logical_xor(relNavBits[:-1], relNavBits[1:])

        #%% Decode all 15 strings =============================================
        lastStringID = 0
        for index in range(15):
            #--- Cut one 85-bit string ----------------------------------------
            string = self.checkPhase(navBits[85*index:85*(index+1)])
            #--- Correct the polarity of the string data bits -----------------
            stringID = self.bin2dec(string[1:5])
            lastStringID = stringID

            # Decode the string according to its ID. For details of the
            # string contents, refer to the GLONASS interface specification.

            if stringID == 1:
                #--- It is string 1 -------------------------------------------
                self.P1 = self.bin2dec(string[7:9])
                if self.P1 != 0:
                    self.P1 = (self.P1 + 1) * 15
                self.TOD = (self.bin2dec(string[9:14]) * 3600 +
                            self.bin2dec(string[14:20]) * 60 +
                            self.bin2dec(string[20:21]) * 30)
                self.xVel = (-1)**self.bin2dec(string[21:22]) * (
                    self.bin2dec(string[22:45]) * 2**-20)
                self.xAcc = (-1)**self.bin2dec(string[45:46]) * (
                    self.bin2dec(string[46:50]) * 2**-30)
                self.xDis = (-1)**self.bin2dec(string[50:51]) * (
                    self.bin2dec(string[51:77]) * 2**-11)

            elif stringID == 2:
                #--- It is string 2 -------------------------------------------
                self.B = self.bin2dec(string[5:6])
                self.P2 = self.bin2dec(string[8:9])
                self.tb = self.bin2dec(string[9:16]) * 15 * 60
                self.yVel = (-1)**self.bin2dec(string[21:22]) * (
                    self.bin2dec(string[22:45]) * 2**-20)
                self.yAcc = (-1)**self.bin2dec(string[45:46]) * (
                    self.bin2dec(string[46:50]) * 2**-30)
                self.yDis = (-1)**self.bin2dec(string[50:51]) * (
                    self.bin2dec(string[51:77]) * 2**-11)

            elif stringID == 3:
                #--- It is string 3 -------------------------------------------
                self.P3 = self.bin2dec(string[5:6])
                self.gam = (-1)**self.bin2dec(string[6:7]) * (
                    self.bin2dec(string[7:17]) * 2**-40)
                self.P = self.bin2dec(string[18:20])
                self.health = self.bin2dec(string[20:21])
                self.zVel = (-1)**self.bin2dec(string[21:22]) * (
                    self.bin2dec(string[22:45]) * 2**-20)
                self.zAcc = (-1)**self.bin2dec(string[45:46]) * (
                    self.bin2dec(string[46:50]) * 2**-30)
                self.zDis = (-1)**self.bin2dec(string[50:51]) * (
                    self.bin2dec(string[51:77]) * 2**-11)

            elif stringID == 4:
                #--- It is string 4 -------------------------------------------
                self.tau_n = (-1)**self.bin2dec(string[5:6]) * (
                    self.bin2dec(string[6:27]) * 2**-30)
                self.dtau = (-1)**self.bin2dec(string[27:28]) * (
                    self.bin2dec(string[28:32]) * 2**-30)
                self.E = self.bin2dec(string[32:37])
                self.P4 = self.bin2dec(string[51:52])
                self.FT = self.bin2dec(string[52:56])
                self.days = self.bin2dec(string[59:70])
                self.n = self.bin2dec(string[70:75])
                self.M = self.bin2dec(string[75:77])

            elif stringID == 5:
                #--- It is string 5 -------------------------------------------
                self.tau_c = (-1)**self.bin2dec(string[16:17]) * (
                    self.bin2dec(string[17:48]) * 2**-31)
                self.N4 = self.bin2dec(string[49:54])

            # Strings 6--15 contain almanac data and are not decoded here.

        if self.TOD is None:
            return np.inf
        #%% Compute the TOD of the first string in the array ==================
        # The transmitted TOD refers to the beginning of the frame containing
        # string 1. Correct it to the first string in this data block.
        return self.TOD - (15-lastStringID) * 2

    @classmethod
    def NAVdecoding(cls, I_P_InputBits):
        """Find and decode the first valid GLONASS navigation frame.

        The function finds the first time-mark occurrence in the bit stream.
        A candidate is verified from the two-second spacing between strings
        and the GLONASS data-verification algorithm.

        Args
        ----
            I_P_InputBits - array-like
                        Prompt correlator output from the tracking function.

        Returns
        -------
            eph         - Ephemeris
                        Decoded satellite ephemeris.
            firstSubFrame - float
                        Zero-based starting position of the first message in
                        the input prompt-I stream. Infinity indicates that no
                        valid time mark was detected.
            TOD         - float
                        Time of day of the first message in seconds. Infinity
                        indicates that no valid time mark was detected.
        """
        eph = cls()
        firstSubFrame = np.inf
        TOD = np.inf

        #%% Bit and frame synchronization =====================================
        # The search may start later in the tracking results to avoid noise
        # caused by tracking-loop transients.
        searchStartOffset = 0
        #--- Generate the time-mark pattern ----------------------------------
        preamble_bits = np.array([
            1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,1,-1,
            1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1])
        # Ten tracking values represent each bi-binary half-bit.
        preamble_ms = np.kron(preamble_bits, np.ones(10))

        # Threshold the tracking output and convert it to -1 and +1.
        bits = np.where(np.asarray(I_P_InputBits[searchStartOffset:]) > 0,
                        1, -1)
        if bits.size < preamble_ms.size:
            return eph, firstSubFrame, TOD

        # A valid-mode correlation index is the time-mark start. Data begins
        # 300 ms later, after the 30-bit time mark.
        # Correlate the tracking output with the time-mark pattern.
        tlmXcorrResult = np.correlate(bits, preamble_ms, mode="valid")
        timeMarkIndex = np.flatnonzero(np.abs(tlmXcorrResult) > 271)
        dataStartIndex = timeMarkIndex + searchStartOffset + 300

        #%% Analyze detected preamble-like patterns ===========================
        for candidate in dataStartIndex:
            # Verify that another time mark occurs one 2-s string later.
            if not np.any(dataStartIndex-candidate == 2000):
                continue
            if candidate + 1700 > len(I_P_InputBits):
                continue

            #--- Combine ten values of each bi-binary half-bit ----------------
            biBinaryBits = np.asarray(
                I_P_InputBits[candidate:candidate+1700]).reshape(170, 10).sum(1)
            biBinaryBits = (biBinaryBits > 0).astype(np.int8)

            #--- Convert bi-binary symbols to relative and data bits -----------
            relRevBits = ((biBinaryBits[0::2] - biBinaryBits[1::2] + 1) // 2)
            revBits = np.empty(85, dtype=np.int8)
            revBits[0] = 0
            revBits[1:] = np.logical_xor(relRevBits[:-1], relRevBits[1:])
            stringBits = revBits[::-1]

            # Verify the navigation data by the GLONASS checking algorithm.
            if not cls.dataVerification(stringBits):
                continue

            firstSubFrame = int(candidate)

            #--- Copy 15 strings, skipping each 300-ms time mark ---------------
            stringSamples = []
            completeFrame = True
            for stringIndex in range(15):
                stringStart = firstSubFrame + stringIndex*2000
                stringEnd = stringStart + 1700
                if stringEnd > len(I_P_InputBits):
                    completeFrame = False
                    break
                stringSamples.append(I_P_InputBits[stringStart:stringEnd])

            if completeFrame:
                TOD = eph.decodeEphemeris(np.concatenate(stringSamples))
            break

        if not np.isfinite(firstSubFrame):
            print("    Could not find valid preambles in this channel!")

        return eph, firstSubFrame, TOD

    @staticmethod
    def _stateDerivative(state, acceleration):
        """Return the GLONASS state derivative in the rotating ECEF frame.

        Args
        ----
            state       - numpy.ndarray
                        Position and velocity state ``[x, y, z, Vx, Vy, Vz]``
                        in SI units.
            acceleration - numpy.ndarray
                        Luni-solar acceleration ``[Ax, Ay, Az]`` in m/s^2.

        Returns
        -------
            derivative  - numpy.ndarray
                        Time derivative of the six-element state vector.
        """
        #--- Constants for satellite-position calculation --------------------
        omega = 7.292115e-5
        my = 3.986004418e14
        a = 6.378136e6
        J02 = 1.082657e-3

        x, y, z, Vx, Vy, Vz = state
        Ax, Ay, Az = acceleration
        r = np.sqrt(x*x + y*y + z*z)
        common = 1.5 * J02 * my * a*a / r**5

        dVx = (-my*x/r**3 - common*x*(1-5*z*z/r**2) +
               omega*omega*x + 2*omega*Vy + Ax)
        dVy = (-my*y/r**3 - common*y*(1-5*z*z/r**2) +
               omega*omega*y - 2*omega*Vx + Ay)
        dVz = (-my*z/r**3 - common*z*(3-5*z*z/r**2) + Az)
        return np.array([Vx, Vy, Vz, dVx, dVy, dVz])

    def satpos(self, transmitTime, tau_c=0.0):
        """Calculate one GLONASS satellite position and clock correction.

        The broadcast position, velocity and acceleration are propagated from
        tb to transmitTime with fourth-order Runge-Kutta integration. The
        broadcast coordinates are expressed in the PZ-90.11 reference frame.

        Args
        ----
            transmitTime - float
                        Signal transmission time in seconds.
            tau_c       - float
                        Difference between GLONASS time and UTC(SU), in
                        seconds.

        Returns
        -------
            satPosition - numpy.ndarray
                        Satellite ECEF position ``[X, Y, Z]`` in meters.
            satClkCorr  - float
                        Satellite clock correction in seconds.
        """
        #--- Find the time difference ----------------------------------------
        dt = self.check_t(transmitTime-self.tb)
        #--- Calculate the satellite clock correction ------------------------
        satClkCorr = -(self.tau_n + tau_c - self.gam*dt)
        #--- Find the integration time ---------------------------------------
        time = dt-satClkCorr

        # Broadcast position, velocity and acceleration converted to SI units.
        state = np.array([
            self.xDis*1e3, self.yDis*1e3, self.zDis*1e3,
            self.xVel*1e3, self.yVel*1e3, self.zVel*1e3], dtype=np.float64)
        acceleration = np.array(
            [self.xAcc, self.yAcc, self.zAcc], dtype=np.float64) * 1e3

        #--- Integrate forward or backward with Runge-Kutta ------------------
        # Use steps of at most 60 s and shorten the final step to the remaining
        # integration interval.
        integratedTime = 0.0
        while abs(time-integratedTime) > 1e-12:
            tau = np.sign(time-integratedTime) * min(
                60.0, abs(time-integratedTime))
            D1 = self._stateDerivative(state, acceleration)
            D2 = self._stateDerivative(state + 0.5*tau*D1, acceleration)
            D3 = self._stateDerivative(state + 0.5*tau*D2, acceleration)
            D4 = self._stateDerivative(state + tau*D3, acceleration)
            state += tau/6.0 * (D1 + 2*D2 + 2*D3 + D4)
            integratedTime += tau

        return state[:3], satClkCorr

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
                Result arrays are initialized in this object.
        """
        channelShape = (numberOfChannels, measNrSum)

        # Satellite information for each channel and measurement epoch.
        # NaN elevation/azimuth values keep unused satellites out of skyPlot.
        self.K          = np.zeros(channelShape, dtype=np.int16)
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

        # Measurement location and receiver GLONASS time.
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
                The navigation engine is initialized in this object.
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
        # GLONASS coordinates require strings 1--4, while string 5 provides
        # the time-scale correction. Fifteen two-second strings span one
        # complete 30 s frame; extra data cover a record starting mid-frame.
        if settings.msToProcess < 36000:
            # Show the error message and exit.
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid message in the 1 ms prompt-I
        # stream. Infinity indicates that no valid preamble was detected.
        subFrameStart = np.full(settings.numberOfChannels, np.inf)
        # TOD of the first valid message [s]; infinity indicates no detection.
        TOD = np.full(settings.numberOfChannels, np.inf)

        #--- Make a list excluding channels not in tracking lock ----------
        # Make a list containing only channels that remain in tracking lock.
        activeChnList = [
            channelNr for channelNr, channel in enumerate(trackResults)
            if channel.lockFlag ]

        #%% Decode ephemerides
        tau_c = 0.0
        
        for channelNr in activeChnList.copy():
            #--- Get the K of the current channel ---------------------------
            K = trackResults[channelNr].K
            print(f"Decoding NAV for K {K:02d} --------------------")

            currentEph, subFrameStart[channelNr], TOD[channelNr] = (
                            Ephemeris.NAVdecoding(trackResults[channelNr].I_P))
            currentEph.K = K
            self.eph[K]  = currentEph

            # tau_c broadcast by a GLONASS-M satellite is the more accurate
            # GLONASS-to-UTC(SU) time-scale correction.
            if (currentEph.M not in (None, 0)
                    and currentEph.tau_c is not None):
                tau_c = currentEph.tau_c

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without the requisite GLONASS strings.
            if (currentEph.P1 is None or currentEph.P2 is None
                    or currentEph.P3 is None or currentEph.P4 is None
                    or currentEph.N4 is None):
                print(f"    Ephemeris decoding fails for K {K:02d}!")
                activeChnList.remove(channelNr)
            elif currentEph.B != 0:
                print(f"    K {K:02d} is unhealthy and is excluded!")
                activeChnList.remove(channelNr)
            else:
                print(f"    The requisite messages for K {K:02d} "
                      "are all decoded!")

        #%% Check whether at least four satellites remain
        # A three-dimensional receiver position and clock bias require four
        # or more satellites with decoded ephemerides.
        if len(activeChnList) < 4:
            # Show the error message and exit.
            print("Too few satellites with ephemeris data for position "
                  "calculations. Exiting!")
            self.eph = {}
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
            
            # Save the frequency channels used for this position calculation.
            navSolutions.K[activeChnList, currMeasNr] = [
                trackResults[channelNr].K
                for channelNr in activeChnList ]

            # Absolute IF-sample position of the current measurement epoch.
            currMeasSample = int(sampleStart+measSampleStep*currMeasNr)

            #%% Find pseudoranges
            # Raw pseudorange = (localTime - transmitTime) * c [m].
            # The pseudorange and transmit-time outputs are channel-ordered
            # vectors with one entry for every receiver channel.
            (rawP, transmitTime, localTime,_) = calculatePseudoranges(
                trackResults, subFrameStart, TOD, currMeasSample,
                localTime, activeChnList, settings, self.searchIndex)
            
            navSolutions.rawP[:, currMeasNr] = rawP
            
            navSolutions.transmitTime[activeChnList, currMeasNr] = (
                transmitTime[activeChnList])

            #%% Find satellite positions and clock corrections
            # Calculate the outputs in the same explicit channel/K order
            # as the pseudorange measurements.
            satPositions = np.zeros((3, len(activeChnList)))
            satClkCorr = np.zeros(len(activeChnList))
            
            for satNr, channelNr in enumerate(activeChnList):
                K = trackResults[channelNr].K
                satPositions[:, satNr], satClkCorr[satNr] = (
                    self.eph[K].satpos(transmitTime[channelNr], tau_c))
                
            navSolutions.satClkCorr[activeChnList, currMeasNr] = (
                satClkCorr)

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

                # A non-finite DOP indicates singular/ill-conditioned
                # measurement geometry. Skip this solution but advance time.
                if not np.all(np.isfinite(dop)):
                    localTime += measSampleStep/settings.samplingFreq
                    continue

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
        """Plot coordinate variations over time and a position plot.

        Receiver coordinates are plotted in the UTM system. Coordinate
        offsets are plotted when true UTM receiver coordinates are provided.

        Returns
        -------
            None
                The navigation plots are displayed in figure 300.
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
                navSolutions.K[:, 0])
        skyAxes.set_title(
            f"Sky plot (mean PDOP: {np.mean(navSolutions.DOP[1]):.2f})")

        figure.tight_layout()
        plt.show()


__all__ = ["Ephemeris", "NavSolutions", "NavigationEngine"]
