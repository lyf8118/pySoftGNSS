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
GLONASS L3OCd navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot)


#%% Broadcast ephemeris
class Ephemeris:
    """Store the decoded GLONASS L3OCd immediate ephemeris of one SV.

    Message types 10, 11 and 12 supply the clock, time, position, velocity
    and acceleration parameters required for positioning.
    """

    def __init__(self):
        """Initialize fields carried by L3OCd message types 10, 11 and 12.

        Returns
        -------
            None
                Ephemeris fields and decoding-status flags are initialized in
                this object.
        """
        #%% Identification and time ========================================
        self.PRN: int | None = None
        self.SV_ID: int | None = None       # Satellite ID
        self.TOD: float | None = None       # Time of day [s]
        self.TOW: float | None = None       # Time of week [s]
        self.Type: int | None = None        # Message-type identifier

        #%% Message type 10 ================================================
        self.N4: int | None = None          # Four-year interval number
        self.NT: int | None = None          # Day in the four-year interval
        self.M: int | None = None           # Satellite modification flag
        self.Tb: float | None = None        # Index time [s]
        self.Tau: float | None = None       # Satellite clock bias [s]
        self.Gamma: float | None = None     # Frequency offset
        self.Beta: float | None = None      # Frequency drift rate [s^-1]
        self.Tau_c: float | None = None     # GLONASS-to-UTC(SU) correction [s]
        self.d_Tau_c: float | None = None   # Rate of change of Tau_c
        self.E_E: int | None = None         # Age of ephemeris data
        self.E_T: int | None = None         # Age of time/clock data
        self.F_E: int | None = None         # Ephemeris URA index
        self.F_T: int | None = None         # Time URA index
        self.Health: int | None = None      # Health flag
        self.DataValid: int | None = None   # Data-validity flag
        self.P1: int | None = None          # Service flag P1
        self.P2: int | None = None          # Sun-pointing/maneuver flag P2

        #%% Message types 11 and 12 ========================================
        # Position at instant Tb [km].
        self.X: float | None = None
        self.Y: float | None = None
        self.Z: float | None = None
        # Velocity at instant Tb [km/s].
        self.dX: float | None = None
        self.dY: float | None = None
        self.dZ: float | None = None
        # Luni-solar acceleration at instant Tb [km/s^2].
        self.ddX: float | None = None
        self.ddY: float | None = None
        self.ddZ: float | None = None

        #%% Message type 12 ================================================
        # Antenna phase-center offsets.
        self.Delta_X_pc: float | None = None
        self.Delta_Y_pc: float | None = None
        self.Delta_Z_pc: float | None = None
        # L3OCp-to-L3OCd time offset.
        self.Delta_Tau_L3: float | None = None
        # GPS-to-GLONASS time-offset correction.
        self.Tau_GPS: float | None = None

        #%% Decoding status ================================================
        self.idValid = np.zeros(3, dtype=bool)
        self.flag = False

    @staticmethod
    def bin2dec(bits):
        """Convert an MSB-first binary field to an unsigned decimal value.

        Args
        ----
            bits        - array-like
                        Binary field containing only zero and one values.

        Returns
        -------
            value       - int
                        Unsigned decimal value of the binary field.
        """
        value = 0
        for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
            value = (value << 1) | int(bit)
        return value

    @classmethod
    def signedBin2dec(cls, bits):
        """Convert a GLONASS sign-magnitude binary field to decimal.

        Args
        ----
            bits        - array-like
                        MSB-first sign-magnitude binary field.

        Returns
        -------
            value       - int
                        Signed decimal value of the binary field.
        """
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        magnitude = cls.bin2dec(bits[1:])
        return -magnitude if bits[0] else magnitude

    @staticmethod
    def check_t(time):
        """Account for a beginning- or end-of-day crossover.

        Args
        ----
            time        - float
                        Time in seconds.

        Returns
        -------
            corrTime    - float
                        Corrected time in seconds.
        """
        halfDay = 43200.0
        if time > halfDay:
            return time-2*halfDay
        if time < -halfDay:
            return time+2*halfDay
        return time

    def decodeString(self, navBits):
        """Decode ephemerides and TOD from one 300-bit L3OCd message.

        Args
        ----
            navBits    - array-like
                        One complete CRC-valid navigation message.

        Returns
        -------
            TOD        - float
                        Time of day of the decoded message, in seconds.

        Notes
        -----
            Common service fields are decoded for every message. Immediate
            ephemeris is assembled from message types 10, 11 and 12 and stored
            in this object.
        """
        navBits = np.asarray(navBits, dtype=np.uint8).reshape(-1)
        if navBits.size != 300:
            raise ValueError("GLONASS L3OCd message must contain 300 bits")

        #%% Parse common service fields ====================================
        # Message type.
        typeVal = self.bin2dec(navBits[20:26])
        self.Type = typeVal
        # TS. The L3OCd timestamp unit is three seconds.
        self.TOD = self.bin2dec(navBits[26:41])*3
        self.TOW = self.TOD
        # Satellite ID, health and data-validity attribute.
        self.SV_ID = self.bin2dec(navBits[41:47])
        self.Health = int(navBits[47])
        self.DataValid = int(navBits[48])

        #%% Parse data fields by message type ==============================
        if typeVal == 10:
            #--- Message type 10: first ephemeris part ---------------------
            # Four-year interval number N4 and day number NT.
            self.N4 = self.bin2dec(navBits[57:62])
            self.NT = self.bin2dec(navBits[62:73])
            # Ephemeris reference time Tb; the least significant bit is 90 s.
            self.Tb = self.bin2dec(navBits[82:92])*90
            # Clock and GLONASS-to-UTC(SU) time-correction parameters.
            self.Tau = self.signedBin2dec(navBits[122:154])*2**-38
            self.Gamma = self.signedBin2dec(navBits[154:173])*2**-48
            self.Tau_c = self.signedBin2dec(navBits[188:228])*2**-31
            self.idValid[0] = True

        elif typeVal == 11:
            #--- Message type 11: position and horizontal velocity --------
            # Position at instant Tb [km].
            self.X = self.signedBin2dec(navBits[57:97])*2**-20
            self.Y = self.signedBin2dec(navBits[97:137])*2**-20
            self.Z = self.signedBin2dec(navBits[137:177])*2**-20
            # X and Y velocity components at instant Tb [km/s].
            self.dX = self.signedBin2dec(navBits[177:212])*2**-30
            self.dY = self.signedBin2dec(navBits[212:247])*2**-30
            self.idValid[1] = True

        elif typeVal == 12:
            #--- Message type 12: vertical velocity and acceleration ------
            # Z velocity component at instant Tb [km/s].
            self.dZ = self.signedBin2dec(navBits[57:92])*2**-30
            # Luni-solar acceleration components at instant Tb [km/s^2].
            self.ddX = self.signedBin2dec(navBits[92:107])*2**-39
            self.ddY = self.signedBin2dec(navBits[107:122])*2**-39
            self.ddZ = self.signedBin2dec(navBits[122:137])*2**-39
            self.idValid[2] = True

        self.flag = bool(np.all(self.idValid))
        return self.TOD

    @staticmethod
    def _viterbiDecode(encodedBits):
        """Hard-decision Viterbi decode the rate-1/2 L3OCd code.

        The ICD convolutional encoder has constraint length seven and
        generator polynomials ``(133, 171)`` in octal form.

        Args
        ----
            encodedBits - array-like
                        Hard-decision convolutional-code symbols.

        Returns
        -------
            decodedBits - numpy.ndarray
                        Viterbi-decoded navigation bits.
        """
        encodedBits = np.asarray(encodedBits, dtype=np.uint8).reshape(-1)
        encodedBits = encodedBits[:encodedBits.size-encodedBits.size % 2]
        symbolCount = encodedBits.size//2
        stateCount = 64
        # Initialize the survivor-path metrics at the all-zero state.
        pathMetric = np.full(stateCount, np.inf)
        pathMetric[0] = 0.0
        previousState = np.zeros((symbolCount, stateCount), dtype=np.uint8)
        previousBit = np.zeros((symbolCount, stateCount), dtype=np.uint8)

        #--- Forward add-compare-select recursion -----------------------------
        for symbolIndex in range(symbolCount):
            received = encodedBits[2*symbolIndex:2*symbolIndex+2]
            nextMetric = np.full(stateCount, np.inf)
            for state in range(stateCount):
                if not np.isfinite(pathMetric[state]):
                    continue
                for inputBit in (0, 1):
                    register = (inputBit << 6) | state
                    output = np.array([
                        (register & 0o133).bit_count() & 1,
                        (register & 0o171).bit_count() & 1], dtype=np.uint8)
                    nextState = ((inputBit << 5) | (state >> 1)) & 0x3f
                    metric = pathMetric[state]+np.count_nonzero(output != received)
                    if metric < nextMetric[nextState]:
                        nextMetric[nextState] = metric
                        previousState[symbolIndex, nextState] = state
                        previousBit[symbolIndex, nextState] = inputBit
            pathMetric = nextMetric

        #--- Trace back the minimum-metric survivor path ----------------------
        decodedBits = np.empty(symbolCount, dtype=np.uint8)
        state = int(np.argmin(pathMetric))
        for symbolIndex in range(symbolCount-1, -1, -1):
            decodedBits[symbolIndex] = previousBit[symbolIndex, state]
            state = int(previousState[symbolIndex, state])
        return decodedBits

    @staticmethod
    def _crc24qCheck(navBits):
        """Check one complete L3OCd message using CRC-24Q.

        Args
        ----
            navBits     - array-like
                        Complete L3OCd navigation message.

        Returns
        -------
            crcOK       - bool
                        True when the CRC-24Q syndrome is zero.
        """
        # CRC-24Q generator polynomial: 0x1864CFB, excluding the implicit
        # x^24 term from the feedback register representation.
        crc = 0
        for bit in np.asarray(navBits, dtype=np.uint8):
            feedback = ((crc >> 23) & 1) ^ int(bit)
            crc = (crc << 1) & 0xffffff
            if feedback:
                crc ^= 0x864cfb
        return crc == 0

    @staticmethod
    def _preambleCandidates(decodedBits, preambleCorr, frameLenBits):
        """Find preamble-like patterns with a complete following message.

        Args
        ----
            decodedBits - numpy.ndarray
                        Viterbi-decoded navigation bits.
            preambleCorr - numpy.ndarray
                        Antipodal L3OCd preamble pattern.
            frameLenBits - int
                        Length of one L3OCd message in decoded bits.

        Returns
        -------
            candidates  - list of int
                        Zero-based preamble positions with a complete
                        following message.
        """
        lastStart = decodedBits.size-frameLenBits
        candidates = []
        antipodalDecoded = 1-2*decodedBits.astype(np.int8)
        for startIdx in range(lastStart+1):
            metric = np.dot(
                antipodalDecoded[startIdx:startIdx+preambleCorr.size],
                preambleCorr)
            if abs(metric) >= 18:
                candidates.append(startIdx)
        return candidates

    @classmethod
    def NAVdecoding(cls, I_P_InputBits):
        """Decode L3OCd ephemerides and first-message TOD.

        The five-chip Barker secondary code is first detected and stripped.
        The rate-1/2 convolutional code is then Viterbi decoded. CRC-valid
        message types 10, 11 and 12 provide the immediate ephemeris.

        Args
        ----
            I_P_InputBits - array-like
                          Data-channel prompt correlator outputs.

        Returns
        -------
            eph          - Ephemeris
                          Decoded satellite ephemeris.
            firstSubFrame - float
                          Zero-based first-message position in 1 ms tracking
                          epochs.
            TOD          - float
                          Time of day of the first valid message, in seconds.

        Notes
        -----
            The first encoded symbol may be the G1 or G2 output. Both
            alignments are searched so that Viterbi decoding begins with G1.
        """
        eph = cls()
        firstSubFrame = np.inf
        TOD = np.inf

        #%% Secondary-code stripping (de-Barker) ===========================
        # The L3OCd signal has a five-chip Barker secondary overlay. Detect
        # its phase and strip it to recover the convolutional-code symbols.
        BarkerPattern = np.array([-1., -1., -1., 1., -1.])
        BcLen = BarkerPattern.size
        maxEnergy = 0.0
        bestPhase = 0
        SymbolStream = np.empty(0)
        I_P_InputBits = np.asarray(I_P_InputBits, dtype=np.float64).reshape(-1)

        #--- Search for the best Barker phase ------------------------------
        for phase in range(BcLen):
            currentStream = I_P_InputBits[phase:]
            nSym = currentStream.size//BcLen
            if nSym < 600:
                continue
            reshaped = currentStream[:nSym*BcLen].reshape(nSym, BcLen)
            collapsedSymbols = reshaped@BarkerPattern
            # Use accumulated symbol magnitude to select the Barker phase.
            currentEnergy = np.sum(np.abs(collapsedSymbols))
            if currentEnergy > maxEnergy:
                maxEnergy = currentEnergy
                bestPhase = phase
                SymbolStream = collapsedSymbols

        if SymbolStream.size == 0:
            return eph, firstSubFrame, TOD
        # Remove DC offset from the symbol stream.
        SymbolStream = SymbolStream-np.mean(SymbolStream)

        #%% Viterbi decoding ===============================================
        # Take an even number of input symbols for rate-1/2 decoding.
        evenLen = SymbolStream.size-SymbolStream.size % 2
        encodedSymbols = SymbolStream[:evenLen]
        # Convert prompt symbols to hard-decision convolutional-code bits.
        dataBitsInput = (encodedSymbols < 0).astype(np.uint8)

        #--- Generate the preamble pattern --------------------------------
        preambleBin = np.array(
            [0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0],
            dtype=np.uint8)
        preambleCorr = 1-2*preambleBin.astype(np.int8)
        frameLenBits = 300

        #%% NAV data decoding ==============================================
        # The first encoded bit may be a G1 or G2 output. Try both alignments.
        for G1orG2 in range(2):
            lastOffset = G1orG2
            inputBits = dataBitsInput[
                G1orG2:dataBitsInput.size-lastOffset if lastOffset else None]
            decodedBits = cls._viterbiDecode(inputBits)

            # Find all preamble-like patterns. The correlation threshold of
            # 18 permits at most one polarity-independent preamble mismatch.
            for startIdx in cls._preambleCandidates(
                    decodedBits, preambleCorr, frameLenBits):
                navBits = decodedBits[startIdx:startIdx+frameLenBits].copy()

                # Correct message polarity from the preamble.
                if np.count_nonzero(navBits[:20] != preambleBin) > 10:
                    navBits = 1-navBits
                if np.count_nonzero(navBits[:20] != preambleBin) > 2:
                    continue
                # Detect errors in the complete input message using CRC-24Q.
                if not cls._crc24qCheck(navBits):
                    continue

                msgType = cls.bin2dec(navBits[20:26])
                if msgType not in (10, 11, 12):
                    print(f"  > Skipped Frame: Type {msgType} "
                          "(Not Ephemeris)")
                    continue

                #--- Save timing for the first valid message --------------
                if not np.isfinite(firstSubFrame):
                    firstSubFrame = (
                        startIdx*2*BcLen+bestPhase+G1orG2*BcLen)
                    TOD = cls.bin2dec(navBits[26:41])*3
                    eph.TOD = TOD
                    eph.TOW = TOD

                #--- Ephemeris decoding -----------------------------------
                eph.decodeString(navBits)
                print(f"Decoded L3OCd Frame: Type {msgType}, "
                      f"SV_ID {eph.SV_ID}")
                if eph.flag:
                    return eph, firstSubFrame, TOD

        if not np.isfinite(firstSubFrame):
            print("    Could not find valid L3OCd messages in this channel!")
        return eph, firstSubFrame, TOD

    @staticmethod
    def _stateDerivative(r, v, a_ls):
        """Return GLONASS differential-equation derivatives in rotating ECEF.

        Args
        ----
            r           - numpy.ndarray
                        Satellite ECEF position ``[x, y, z]`` in meters.
            v           - numpy.ndarray
                        Satellite ECEF velocity ``[vx, vy, vz]`` in m/s.
            a_ls        - numpy.ndarray
                        Luni-solar acceleration in m/s^2.

        Returns
        -------
            dr          - numpy.ndarray
                        Position derivative in m/s.
            dv          - numpy.ndarray
                        Velocity derivative in m/s^2.
        """
        #--- GLONASS constants -----------------------------------------------
        ae = 6378136.0
        mu = 398600.4418e9
        J2 = 1082.6257e-6
        omega = 7.292115e-5

        x, y, z = r
        vx, vy, vz = v
        r_sq = x*x+y*y+z*z
        r_mag = np.sqrt(r_sq)
        # Central-gravity and second-zonal-harmonic factors.
        c1 = -mu/r_sq/r_mag
        c2 = 1.5*J2*mu*ae**2/(r_sq*r_mag*r_sq)
        z_rat = 5*z*z/r_sq

        ax = c1*x-c2*x*(1-z_rat)+omega**2*x+2*omega*vy+a_ls[0]
        ay = c1*y-c2*y*(1-z_rat)+omega**2*y-2*omega*vx+a_ls[1]
        az = c1*z-c2*z*(3-z_rat)+a_ls[2]
        return v, np.array([ax, ay, az])

    def satpos(self, transmitTime):
        """Calculate one L3OCd satellite position and clock correction.

        The PZ-90.11 broadcast state vector is propagated from Tb to
        transmitTime.

        Args
        ----
            transmitTime - float
                        Signal transmission time in GLONASS time, in seconds.

        Returns
        -------
            satPosition - numpy.ndarray
                        Satellite ECEF position ``[X, Y, Z]`` in meters.
            satClkCorr  - float
                        Satellite clock correction in seconds.
        """
        #%% Find the initial satellite clock correction ======================
        # Calculate the time difference and account for day rollover.
        dt = self.check_t(transmitTime-self.Tb)
        # Satellite clock correction [s].
        satClkCorr = -self.Tau+self.Gamma*dt

        #%% Find the satellite position ======================================
        # State vector and perturbing acceleration in SI units.
        r = np.array([self.X, self.Y, self.Z], dtype=np.float64)*1000
        v = np.array([self.dX, self.dY, self.dZ], dtype=np.float64)*1000
        a_ls = np.array(
            [self.ddX, self.ddY, self.ddZ], dtype=np.float64)*1000

        # Integrate the state vector with a fourth-order Runge-Kutta method.
        # Shorten the final 30-s step to the remaining integration interval.
        t_current = 0.0
        while abs(t_current) < abs(dt):
            step = min(30.0, abs(dt)-abs(t_current))
            h = np.sign(dt)*step
            k1_r, k1_v = self._stateDerivative(r, v, a_ls)
            k2_r, k2_v = self._stateDerivative(
                r+0.5*h*k1_r, v+0.5*h*k1_v, a_ls)
            k3_r, k3_v = self._stateDerivative(
                r+0.5*h*k2_r, v+0.5*h*k2_v, a_ls)
            k4_r, k4_v = self._stateDerivative(r+h*k3_r, v+h*k3_v, a_ls)
            r += h/6*(k1_r+2*k2_r+2*k3_r+k4_r)
            v += h/6*(k1_v+2*k2_v+2*k3_v+k4_v)
            t_current += h

        return r, satClkCorr


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
        # L3OC positioning requires CRC-valid message types 10, 11 and 12.
        # At least 32 seconds accommodate arbitrary message alignment.
        if settings.msToProcess < 32000:
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

        for channelNr in activeChnList.copy():
            #--- Get the PRN of the current channel ---------------------------
            PRN = trackResults[channelNr].PRN
            print(f"Decoding NAV for PRN {PRN:02d} --------------------")

            currentEph, subFrameStart[channelNr], TOD[channelNr] = (
                            Ephemeris.NAVdecoding(trackResults[channelNr].I_P))
            currentEph.PRN = PRN
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all three immediate-data messages.
            if not currentEph.flag:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
                activeChnList.remove(channelNr)
            else:
                print(f"    Message types 10, 11 and 12 for PRN {PRN:02d} "
                      "all decoded!")

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

            # Save the PRNs used for this position calculation.
            navSolutions.PRN[activeChnList, currMeasNr] = [
                trackResults[channelNr].PRN
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
            # Calculate the outputs in the same explicit channel/PRN order
            # as the pseudorange measurements.
            satPositions = np.zeros((3, len(activeChnList)))
            satClkCorr = np.zeros(len(activeChnList))

            for satNr, channelNr in enumerate(activeChnList):
                PRN = trackResults[channelNr].PRN
                satPositions[:, satNr], satClkCorr[satNr] = (
                    self.eph[PRN].satpos(transmitTime[channelNr]))

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
                navSolutions.PRN[:, 0])
        skyAxes.set_title(
            f"Sky plot (mean PDOP: {np.mean(navSolutions.DOP[1]):.2f})")

        figure.tight_layout()
        plt.show()


__all__ = ["Ephemeris", "NavSolutions", "NavigationEngine"]
