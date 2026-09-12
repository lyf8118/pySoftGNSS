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
GLONASS L1OCd navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot,
)


#%% Broadcast ephemeris
class Ephemeris:
    """Store the decoded GLONASS L1OCd immediate ephemeris of one SV.

    String types 10, 11 and 12 supply the clock, time, position, velocity,
    acceleration and phase-center parameters needed for positioning.
    """

    def __init__(self):
        """Initialize all fields used by L1OCd string types 10, 11 and 12.

        Returns
        -------
            None
                Ephemeris fields and decoding-status flags are initialized in
                this object.
        """
        #%% Identification and time ==========================================
        self.PRN: int | None = None
        self.SV_ID: int | None = None       # Satellite ID
        self.TOD: float | None = None       # Time of day [s]
        self.TOW: float | None = None       # Time of week [s]
        self.Type: int | None = None        # String-type identifier

        #%% String type 10: time, clock and service data ======================
        self.N4: int | None = None          # Four-year interval number
        self.NT: int | None = None          # Day in the four-year interval
        self.M: int | None = None           # Satellite modification flag
        self.PS: int | None = None
        self.Tb: float | None = None        # Index time [s]
        self.Tau: float | None = None       # Satellite clock bias [s]
        self.Gamma: float | None = None     # Frequency offset
        self.Beta: float | None = None      # Frequency drift rate [s^-1]
        self.Tau_c: float | None = None     # GLONASS-to-MT correction [s]
        self.d_Tau_c: float | None = None   # Rate of change of Tau_c
        self.E_E: int | None = None         # Age of ephemeris data
        self.E_T: int | None = None         # Age of time/clock data
        self.R_E: int | None = None
        self.R_T: int | None = None
        self.F_E: int | None = None         # Ephemeris URA index
        self.F_T: int | None = None         # Time URA index
        self.Health: int | None = None      # Health flag
        self.DataValid: int | None = None   # Data-validity flag
        self.P1: int | None = None          # Service flag P1
        self.P2: int | None = None          # Sun-pointing/maneuver flag P2
        self.KP: int | None = None
        self.A: int | None = None

        #%% String types 11 and 12: state vector and corrections ==============
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
        # Antenna phase-center offsets.
        self.DeltaXpc: float | None = None
        self.DeltaYpc: float | None = None
        self.DeltaZpc: float | None = None
        # L2OCp-to-L1OCd and GPS-to-GLONASS time-offset corrections.
        self.DeltaTauL2: float | None = None
        self.TauGPS: float | None = None

        #%% Decoding status ===================================================
        # Types 10, 11 and 12 must all be decoded before positioning.
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
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        value = 0
        for bit in bits:
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
        """Decode one CRC-valid 250-bit GLONASS L1OCd string.

        Common service fields are decoded for every string. Immediate
        ephemeris is assembled from string types 10, 11 and 12.

        Args
        ----
            navBits     - array-like
                        One complete CRC-valid L1OCd navigation string.

        Returns
        -------
            TOD         - float
                        Time of day of the decoded string, in seconds.

        Notes
        -----
            The decoded fields and type-validity flags are stored in this
            object.
        """
        navBits = np.asarray(navBits, dtype=np.uint8).reshape(-1)
        if navBits.size != 250:
            raise ValueError("GLONASS L1OCd string must contain 250 bits")

        #%% Parse common service fields ======================================
        # String type.
        typeVal = self.bin2dec(navBits[12:18])
        self.Type = typeVal
        # TS. The L1OCd timestamp unit is two seconds.
        self.TOD = self.bin2dec(navBits[34:50])*2
        self.TOW = self.TOD
        # Satellite ID.
        self.SV_ID = self.bin2dec(navBits[18:24])
        # Health and data validity.
        self.Health = int(navBits[24])
        self.DataValid = int(navBits[25])
        # Service fields.
        self.P1 = self.bin2dec(navBits[26:30])
        self.P2 = int(navBits[30])
        self.KP = self.bin2dec(navBits[31:33])
        self.A = int(navBits[33])

        #%% Parse data fields according to the string type ====================
        if typeVal == 10:
            #--- String type 10 -------------------------------------------
            self.N4 = self.bin2dec(navBits[50:55])
            self.NT = self.bin2dec(navBits[55:66])
            self.M = self.bin2dec(navBits[66:69])
            self.PS = self.bin2dec(navBits[69:75])
            # Ephemeris reference time tb; the least significant bit is 90 s.
            self.Tb = self.bin2dec(navBits[75:85])*90
            self.E_E = self.bin2dec(navBits[85:93])
            self.E_T = self.bin2dec(navBits[93:101])
            self.R_E = self.bin2dec(navBits[101:103])
            self.R_T = self.bin2dec(navBits[103:105])
            self.F_E = self.signedBin2dec(navBits[105:110])
            self.F_T = self.signedBin2dec(navBits[110:115])
            # Clock and time-correction parameters.
            self.Tau = self.signedBin2dec(navBits[115:147])*2**-38
            self.Gamma = self.signedBin2dec(navBits[147:166])*2**-48
            self.Beta = self.signedBin2dec(navBits[166:181])*2**-57
            self.Tau_c = self.signedBin2dec(navBits[181:221])*2**-31
            self.d_Tau_c = self.signedBin2dec(navBits[221:234])*2**-49
            self.idValid[0] = True

        elif typeVal == 11:
            #--- String type 11 -------------------------------------------
            # Position at instant Tb.
            self.X = self.signedBin2dec(navBits[50:90])*2**-20
            self.Y = self.signedBin2dec(navBits[90:130])*2**-20
            self.Z = self.signedBin2dec(navBits[130:170])*2**-20
            # Velocity X component at instant Tb.
            self.dX = self.signedBin2dec(navBits[170:205])*2**-30
            # Phase-center corrections.
            self.DeltaXpc = (
                self.signedBin2dec(navBits[205:218])*2**-10)
            self.DeltaYpc = (
                self.signedBin2dec(navBits[218:231])*2**-10)
            self.idValid[1] = True

        elif typeVal == 12:
            #--- String type 12 -------------------------------------------
            # Phase-center correction.
            self.DeltaZpc = (
                self.signedBin2dec(navBits[50:63])*2**-10)
            # Velocity Y and Z components at instant Tb.
            self.dY = self.signedBin2dec(navBits[63:98])*2**-30
            self.dZ = self.signedBin2dec(navBits[98:133])*2**-30
            # Luni-solar acceleration components at instant Tb.
            self.ddX = self.signedBin2dec(navBits[133:148])*2**-39
            self.ddY = self.signedBin2dec(navBits[148:163])*2**-39
            self.ddZ = self.signedBin2dec(navBits[163:178])*2**-39
            # Inter-frequency and GPS time-correction parameters.
            self.DeltaTauL2 = (
                self.signedBin2dec(navBits[178:196])*2**-38)
            self.TauGPS = self.signedBin2dec(navBits[196:226])*2**-38
            self.idValid[2] = True

        self.flag = bool(np.all(self.idValid))
        return self.TOD

    @staticmethod
    def _removeOC1(I_P, ocPattern, stringLength):
        """Search the OC1 phase and collapse 2-ms samples to CE symbols.

        Args
        ----
            I_P         - array-like
                        Prompt correlator output, one value per 2 ms.
            ocPattern   - numpy.ndarray
                        Antipodal OC1 overlay pattern.
            stringLength - int
                        Normal L1OCd string length in decoded bits.

        Returns
        -------
            bestSymbols - numpy.ndarray
                        CE-symbol stream for the phase with maximum energy.
            bestPhase   - int
                        Zero-based phase of the selected OC1 pattern.
        """
        bestMetric = -np.inf
        bestPhase = 0
        bestSymbols = np.empty(0)
        minCESymbols = 2*stringLength

        for phase in range(ocPattern.size):
            values = I_P[phase:]
            pairCount = values.size//ocPattern.size
            if pairCount < minCESymbols:
                continue
            pairs = values[:pairCount*ocPattern.size].reshape(
                pairCount, ocPattern.size)
            ceSymbols = pairs@ocPattern
            metric = np.sum(np.abs(ceSymbols))
            if metric > bestMetric:
                bestMetric = metric
                bestPhase = phase
                bestSymbols = ceSymbols
        return bestSymbols, bestPhase

    @staticmethod
    def _viterbiDecode(encodedBits):
        """Hard-decision Viterbi decode the rate-1/2 L1OCd code.

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
        symbolCount = encodedBits.size//2
        stateCount = 64
        # Initialize the survivor-path metrics at the all-zero state.
        pathMetric = np.full(stateCount, np.inf)
        pathMetric[0] = 0.0
        previousState = np.zeros((symbolCount, stateCount), dtype=np.uint8)
        previousBit = np.zeros((symbolCount, stateCount), dtype=np.uint8)

        #--- Forward add-compare-select recursion -----------------------------
        for symbolIndex in range(symbolCount):
            received0, received1 = encodedBits[2*symbolIndex:2*symbolIndex+2]
            nextMetric = np.full(stateCount, np.inf)
            for state in range(stateCount):
                if not np.isfinite(pathMetric[state]):
                    continue
                for inputBit in (0, 1):
                    register = (inputBit << 6) | state
                    output0 = (register & 0o133).bit_count() & 1
                    output1 = (register & 0o171).bit_count() & 1
                    nextState = ((inputBit << 5) | (state >> 1)) & 0x3f
                    branchMetric = (int(output0 != int(received0)) +
                                    int(output1 != int(received1)))
                    metric = pathMetric[state]+branchMetric
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
    def _findPreambleCandidates(bits, preambleBits, stringLength):
        """Find exact preambles that leave one complete following string.

        Args
        ----
            bits        - numpy.ndarray
                        Viterbi-decoded navigation bits.
            preambleBits - numpy.ndarray
                        Binary L1OCd preamble pattern.
            stringLength - int
                        Length of one normal L1OCd string in bits.

        Returns
        -------
            candidateStarts - numpy.ndarray
                        Zero-based starting indices of exact preamble matches.
        """
        searchCount = bits.size-stringLength+1
        if searchCount <= 0:
            return np.empty(0, dtype=np.int64)
        candidates = [
            index for index in range(searchCount)
            if np.array_equal(
                bits[index:index+preambleBits.size], preambleBits)]
        return np.asarray(candidates, dtype=np.int64)

    @staticmethod
    def _crcRemainderMSB(codeBits, polynomial):
        """Perform MSB-first polynomial division over GF(2).

        Args
        ----
            codeBits    - array-like
                        Complete binary codeword.
            polynomial  - array-like
                        Generator-polynomial coefficients, MSB first.

        Returns
        -------
            remainder   - numpy.ndarray
                        CRC remainder from the polynomial division.
        """
        work = np.asarray(codeBits, dtype=np.uint8).copy()
        polynomial = np.asarray(polynomial, dtype=np.uint8)
        crcLength = polynomial.size-1
        for index in range(work.size-crcLength):
            # Subtraction over GF(2) is an exclusive-or operation.
            if work[index]:
                work[index:index+crcLength+1] ^= polynomial
        return work[-crcLength:]

    @classmethod
    def _checkL1OCdCRC250(cls, navBits):
        """Check a normal 250-bit L1OCd string by its CRC syndrome.

        Args
        ----
            navBits     - array-like
                        Complete L1OCd navigation string.

        Returns
        -------
            crcOK       - bool
                        True when the CRC syndrome is zero.
        """
        # g(x) = x^16 + x^14 + x^13 + x^11 + x^10 + x^9 + x^8
        #      + x^6 + x^5 + x + 1.
        polynomial = np.array(
            [1,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,1], dtype=np.uint8)
        return (len(navBits) == 250 and
                not np.any(cls._crcRemainderMSB(navBits, polynomial)))

    @classmethod
    def NAVdecoding(cls, I_P):
        """Decode GLONASS L1OCd navigation data.

        Args
        ----
            I_P         - array-like
                        Prompt correlator output, one value per 2 ms.

        Returns
        -------
            eph         - Ephemeris
                        Decoded satellite ephemeris.
            firstSubFrame - float
                        Zero-based start index of the first valid L1OCd string
                        in the 2-ms prompt stream. Infinity indicates that no
                        valid string was detected.
            TOD         - float
                        Time of day of the first valid string, in seconds.

        Notes
        -----
            L1OCd uses a 125-bps data stream, a rate-1/2 constraint-length-7
            convolutional encoder with ``(133, 171)`` octal generators, a
            250-symbol/s CE stream, the two-chip OC1 overlay, a 12-bit
            preamble, and a CRC-protected 250-bit normal string.
        """
        eph = cls()
        firstSubFrame = np.inf
        TOD = np.inf

        #%% Initialize ========================================================
        I_P = np.asarray(I_P, dtype=np.float64).reshape(-1)
        I_P -= np.mean(I_P)
        if I_P.size < 1200:
            return eph, firstSubFrame, TOD

        #%% ICD constants =====================================================
        preambleBits = np.array(
            [0,1,0,1,1,1,1,1,0,0,0,1], dtype=np.uint8)
        stringLength = 250
        #%% Remove OC1 ========================================================
        # One prompt value corresponds to 2 ms. Two adjacent values produce
        # one CE symbol after removal of the OC1 overlay.
        ceSymbols, bestOC1Phase = cls._removeOC1(
            I_P, np.array([1.0, -1.0]), stringLength)
        if ceSymbols.size == 0:
            return eph, firstSubFrame, TOD
        ceSymbols -= np.mean(ceSymbols)
        ceSymbols = ceSymbols[:ceSymbols.size-ceSymbols.size % 2]

        #%% Viterbi decoding and string search ================================
        # Resolve both the 180-degree carrier-phase ambiguity and whether the
        # first CE symbol is the first or second encoder output of a data bit.
        for polarity in (1.0, -1.0):
            hardCESymbols = (polarity*ceSymbols < 0).astype(np.uint8)
            for ceOffset in (0, 1):
                vitInput = hardCESymbols[ceOffset:]
                vitInput = vitInput[:vitInput.size-vitInput.size % 2]
                if vitInput.size < 2*stringLength:
                    continue
                decodedBits = cls._viterbiDecode(vitInput)
                candidateStarts = cls._findPreambleCandidates(
                    decodedBits, preambleBits, stringLength)

                for startIndex in candidateStarts:
                    navBits = decodedBits[
                        startIndex:startIndex+stringLength]
                    # Validate a complete normal string by its CRC syndrome.
                    if not cls._checkL1OCdCRC250(navBits):
                        continue

                    messageType = cls.bin2dec(navBits[12:18])
                    svID = cls.bin2dec(navBits[18:24])
                    if messageType not in (10, 11, 12):
                        print("  > Skipped L1OCd String: "
                              f"Type {messageType}, SV_ID {svID}, "
                              "CRC OK, not ephemeris.")
                        continue

                    if not np.isfinite(firstSubFrame):
                        # One decoded bit spans four 2-ms prompt samples.
                        firstSubFrame = (
                            bestOC1Phase+2*ceOffset+4*startIndex)
                        TOD = cls.bin2dec(navBits[34:50])*2

                    eph.decodeString(navBits)
                    print(f"Decoded L1OCd String: Type {messageType}, "
                          f"SV_ID {eph.SV_ID}")
                    if eph.flag:
                        return eph, firstSubFrame, TOD

        if not np.isfinite(firstSubFrame):
            print("    Could not find valid L1OCd strings in this channel!")
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
        omega = 7.2921151467e-5
        my = 3.986004418e14
        a = 6.378136e6
        J02 = 1.0826257e-3

        x, y, z, Vx, Vy, Vz = state
        Ax, Ay, Az = acceleration
        radius = np.sqrt(x*x+y*y+z*z)
        # Central-gravity and second-zonal-harmonic factors.
        common = 1.5*J02*my*a*a/radius**5
        dVx = (-my*x/radius**3-common*x*(1-5*z*z/radius**2) +
               omega*omega*x+2*omega*Vy+Ax)
        dVy = (-my*y/radius**3-common*y*(1-5*z*z/radius**2) +
               omega*omega*y-2*omega*Vx+Ay)
        dVz = (-my*z/radius**3-common*z*(3-5*z*z/radius**2)+Az)
        return np.array([Vx, Vy, Vz, dVx, dVy, dVz])

    def satpos(self, transmitTime):
        """Calculate one GLONASS satellite position and clock correction.

        The received signal time is first converted to the MT interval from
        Tb according to Appendix D of the GLONASS CDMA ICD. The PZ-90.11
        broadcast state is then propagated through that interval with
        fourth-order Runge-Kutta integration.

        Args
        ----
            transmitTime - float
                        Signal transmission time in seconds.

        Returns
        -------
            satPosition - numpy.ndarray
                        Satellite ECEF position ``[X, Y, Z]`` in meters.
            satClkCorr  - float
                        Satellite clock correction in seconds.
        """
        #--- Find the integration time ---------------------------------------
        # Transform the received signal time to the interval from the
        # broadcast ephemeris epoch according to Appendix D of the GLONASS
        # CDMA ICD.
        daySec = 86400.0
        timeDifference = transmitTime+self.Tau+self.Tau_c-self.Tb
        dayNumber = np.sign(timeDifference)*np.floor(
            abs(timeDifference/daySec)+0.5)
        timeDifference -= dayNumber*daySec
        deltaTb = timeDifference/(1+self.Gamma-self.d_Tau_c)

        #--- Calculate the satellite clock correction ------------------------
        # Tau_c and d_Tau_c convert GLONASS system time to MT and therefore
        # do not form part of the satellite clock correction returned to PVT.
        satClkCorr = -self.Tau+self.Gamma*deltaTb+self.Beta*deltaTb**2

        # Broadcast position, velocity and acceleration converted to SI units.
        state = np.array([
            self.X*1e3, self.Y*1e3, self.Z*1e3,
            self.dX*1e3, self.dY*1e3, self.dZ*1e3], dtype=np.float64)
        acceleration = np.array(
            [self.ddX, self.ddY, self.ddZ], dtype=np.float64)*1e3

        #--- Integrate forward or backward with Runge-Kutta ------------------
        # Use steps of at most 30 s and shorten the final step to the remaining
        # integration interval.
        integratedTime = 0.0
        while abs(deltaTb-integratedTime) > 1e-12:
            step = np.sign(deltaTb-integratedTime)*min(
                30.0, abs(deltaTb-integratedTime))
            D1 = self._stateDerivative(state, acceleration)
            D2 = self._stateDerivative(state+0.5*step*D1, acceleration)
            D3 = self._stateDerivative(state+0.5*step*D2, acceleration)
            D4 = self._stateDerivative(state+step*D3, acceleration)
            state += step/6.0*(D1+2*D2+2*D3+D4)
            integratedTime += step

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
        # L1OC positioning requires CRC-valid string types 10, 11 and 12.
        # A normal L1OCd string lasts two seconds; the longer record also
        # accommodates tracking transients and arbitrary starting position.
        if settings.msToProcess < 36000:
            # Show the error message and exit.
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid string in the 2 ms prompt-I
        # stream. Infinity indicates that no valid preamble was detected.
        subFrameStart = np.full(settings.numberOfChannels, np.inf)
        # TOD of the first valid string [s]; infinity indicates no detection.
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
            # Exclude satellites without all three immediate-data strings.
            if not currentEph.flag:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
                activeChnList.remove(channelNr)
            else:
                print(f"    String types 10, 11 and 12 for PRN {PRN:02d} "
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
