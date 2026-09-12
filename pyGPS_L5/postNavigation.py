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
GPS L5 CNAV decoding and position calculation.

"""

import numpy as np

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot, twosComp2dec,
)


#%% Broadcast ephemeris
class Ephemeris:
    """Keep a common CNAV ephemeris structure for one satellite.

    This ensures that the ephemeris of each satellite has a similar structure
    when only one or none of the three requisite messages is decoded.
    """

    def __init__(self):
        """Initialize a common CNAV ephemeris structure.

        Returns
        -------
            None
        """
        # Flags for message-data decoding. Entries 0--1 correspond to
        # message types 10 and 11, entries 2--9 to message types 30--37,
        # and entry 10 to all other message types.
        self.idValid = np.zeros(11, dtype=np.int16)
        # Satellite PRN number.
        self.PRN: int | None = None

        #--- Message type 10 -----------------------------------------------
        # Week number.
        self.weekNumber: int | None = None
        # Top time of ephemeris prediction.
        self.T_op: float | None = None
        # L2 health.
        self.health: int | None = None
        # ED accuracy index.
        self.URA_ED: int | None = None
        # Ephemeris-data reference time of week.
        self.t_oe: float | None = None
        # Semi-major-axis difference at reference time.
        self.deltaA: float | None = None
        # Change rate in semi-major axis.
        self.ADot: float | None = None
        # Mean-motion difference from computed value at reference time.
        self.delta_n_0: float | None = None
        # Rate of mean-motion difference from computed value.
        self.delta_n_0Dot: float | None = None
        # Mean anomaly at reference time.
        self.M_0: float | None = None
        # Eccentricity.
        self.e: float | None = None
        # Argument of perigee.
        self.omega: float | None = None

        #--- Message type 11 -----------------------------------------------
        # Longitude of ascending node of orbit plane at weekly epoch.
        self.omega_0: float | None = None
        # Inclination angle at reference time.
        self.i_0: float | None = None
        # Rate of right-ascension difference.
        self.delta_omegaDot: float | None = None
        # Rate of inclination angle.
        self.i_0Dot: float | None = None
        # Sine harmonic correction to the angle of inclination.
        self.C_is: float | None = None
        # Cosine harmonic correction to the angle of inclination.
        self.C_ic: float | None = None
        # Sine correction to the orbit radius.
        self.C_rs: float | None = None
        # Cosine correction to the orbit radius.
        self.C_rc: float | None = None
        # Sine harmonic correction to the argument of latitude.
        self.C_us: float | None = None
        # Cosine harmonic correction to the argument of latitude.
        self.C_uc: float | None = None

        #--- Message type 30 -----------------------------------------------
        # Clock-data reference time of week.
        self.t_oc: float | None = None
        # SV clock bias, drift, and drift-rate correction coefficients.
        self.a_f0: float | None = None
        self.a_f1: float | None = None
        self.a_f2: float | None = None
        # Group-delay differential correction terms.
        self.T_GD: float | None = None
        self.ISC_L5I: float | None = None
        # Ionospheric parameters.
        self.alpha0: float | None = None
        self.alpha1: float | None = None
        self.alpha2: float | None = None
        self.alpha3: float | None = None
        self.beta0: float | None = None
        self.beta1: float | None = None
        self.beta2: float | None = None
        self.beta3: float | None = None

        # Time of week of the first decoded message [s].
        self.TOW: float | None = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary character sequence to an unsigned integer.

        Args
        ----
            bits        - iterable of str
                        Binary characters containing only ``'0'`` or ``'1'``.
        Returns
        -------
            decimal     - int
                        Unsigned decimal value represented by ``bits``.
        """
        return int("".join(bits), 2)

    @staticmethod
    def check_t(time):
        """Account for beginning or end of week crossover.

        Args
        ----
            time        - float
                        Time in seconds.
        Returns
        -------
            corrTime    - float
                        Corrected time in seconds.
        """
        half_week = 302400  # [s]
        if time > half_week:
            return time-2*half_week
        if time < -half_week:
            return time+2*half_week
        return time

    @staticmethod
    def _viterbiDecode(codedBits):
        """Hard-decision decode the rate-1/2 convolutional code.

        The two generator polynomials are 171 and 133 in octal notation,
        corresponding to ``poly2trellis(7, [171 133])`` in MATLAB.

        Args
        ----
            codedBits   - array-like
                        Hard-decision convolutionally encoded bits.
        Returns
        -------
            decodedBits - numpy.ndarray
                        Viterbi-decoded CNAV bits.
        """
        # Rate-1/2 convolutional-code generator polynomials in octal form.
        polynomials = (0o171, 0o133)

        codedBits = np.asarray(codedBits, dtype=np.uint8)
        codedBits = codedBits[:codedBits.size-codedBits.size % 2]
        symbolCount = codedBits.size//2
        stateCount = 64

        # Viterbi traceback depth for vitdec(function).
        tblen = 35

        # Convert the CNAV-producing convolutional-code polynomials to a
        # trellis description. The new input enters the most-significant
        # register stage and the six memory elements shift toward the LSB.
        nextState = np.empty((stateCount, 2), dtype=np.uint8)
        output = np.empty((stateCount, 2, 2), dtype=np.uint8)
        for state in range(stateCount):
            for bit in (0, 1):
                register = (bit << 6) | state
                nextState[state, bit] = (bit << 5) | (state >> 1)
                output[state, bit, 0] = (register & polynomials[0]).bit_count() & 1
                output[state, bit, 1] = (register & polynomials[1]).bit_count() & 1

        maxMetric = np.iinfo(np.int16).max
        pathMetric = np.full(stateCount, maxMetric, dtype=np.int32)
        pathMetric[0] = 0
        previousState = np.zeros((symbolCount, stateCount), dtype=np.uint8)
        previousBit = np.zeros((symbolCount, stateCount), dtype=np.uint8)
        decodedBits = np.zeros(symbolCount, dtype=np.uint8)

        #--- Add-compare-select recursion ---------------------------------
        for symbolIndex in range(symbolCount):
            received = codedBits[2*symbolIndex:2*symbolIndex+2]
            newMetric = np.full(stateCount, maxMetric, dtype=np.int32)
            for state in range(stateCount):
                for bit in (0, 1):
                    target = int(nextState[state, bit])
                    metric = (pathMetric[state]
                              + np.count_nonzero(
                                  output[state, bit] != received))
                    if metric < newMetric[target]:
                        newMetric[target] = metric
                        previousState[symbolIndex, target] = state
                        previousBit[symbolIndex, target] = bit

            # Renormalize the state metrics and select the last state when
            # several survivor paths have the same minimum metric.
            pathMetric = newMetric-newMetric.min()
            minState = int(np.flatnonzero(pathMetric == 0)[-1])

            # In truncated mode, traceback starts after tblen symbols. The
            # current survivor path supplies the symbol tblen steps earlier.
            if symbolIndex >= tblen:
                state = minState
                for tracebackIndex in range(
                        symbolIndex, symbolIndex-tblen-1, -1):
                    decodedBit = previousBit[tracebackIndex, state]
                    state = int(previousState[tracebackIndex, state])
                decodedBits[symbolIndex-tblen] = decodedBit

        # Fill the last tblen symbols by tracing back from the final
        # minimum-metric state, as in MATLAB's truncated operating mode.
        state = minState
        for symbolIndex in range(symbolCount-1, symbolCount-tblen-1, -1):
            decodedBits[symbolIndex] = previousBit[symbolIndex, state]
            state = int(previousState[symbolIndex, state])
        return decodedBits

    @staticmethod
    def _crc24qCheck(bits):
        """Check one complete CNAV message with CRC-24Q.

        Args
        ----
            bits        - array-like
                        Complete 300-bit CNAV message including CRC.
        Returns
        -------
            valid       - bool
                        True when the CRC-24Q remainder is zero.
        """
        # CRC-24Q generator polynomial:
        # x^24+x^23+x^18+x^17+x^14+x^11+x^10+x^7+x^6+x^5+x^4+x^3+x+1.
        crc = 0
        polynomial = 0x864CFB
        for bit in np.asarray(bits, dtype=np.uint8):
            feedback = ((crc >> 23) & 1)^int(bit)
            crc = (crc << 1) & 0xFFFFFF
            if feedback:
                crc ^= polynomial
        return crc == 0

    @classmethod
    def _l5BitSync(cls, I_P_InputBits):
        """Wipe off the ten-chip NH code and form encoded CNAV symbols.

        Args
        ----
            I_P_InputBits - array-like
                          Data-prompt tracking outputs for one channel.
        Returns
        -------
            dataBits    - numpy.ndarray or None
                        NH-wiped encoded CNAV symbols.
            bitSyncPos  - int or None
                        Zero-based first CNAV-symbol boundary in the tracking
                        stream.
        """
        # Antipodal form of the ten-chip L5 Neuman-Hoffman code.
        NHCode = np.array([1, 1, 1, 1, -1, -1, 1, -1, 1, -1],
                          dtype=np.int8)

        I_P_InputBits = np.asarray(I_P_InputBits, dtype=np.float64)

        #%% Bit synchronization ============================================

        # Preamble search can be delayed to avoid noise due to tracking-loop
        # transients.
        searchStartOffset = 800

        # Take 0.5 s of tracking outputs and convert them to -1 and +1.
        I_P_Input = np.where(
            I_P_InputBits[searchStartOffset:searchStartOffset+501] > 0,
            1, -1)

        # Correlate the tracking outputs with the antipodal NH code.
        tlmXcorrResult = np.correlate(I_P_Input, NHCode, mode="valid")

        # Go through the first half of the correlation result to find the
        # first CNAV-data boundary.
        bitSyncPos = None
        for ind in range(I_P_Input.size//2):
            # Skip the final boundary value to avoid the correlation edge.
            navBits = np.abs(tlmXcorrResult[ind:-1:10])
            # Use 0.01 as the threshold because the values may be noninteger.
            if navBits.size and np.all(np.abs(navBits-10) < 0.01):
                bitSyncPos = ind+searchStartOffset
                break

        # Exclude the channel when no bit boundary was detected.
        if bitSyncPos is None:
            print("Could not find bit boundary in this channel!")
            return None, None

        #%% Prepare decoded data bits for processing =======================

        # Cut the input to an integer number of ten-code-period CNAV symbols.
        bitsNum = (I_P_InputBits.size-bitSyncPos)//10
        I_P_InputBits = np.where(
            I_P_InputBits[bitSyncPos:bitSyncPos+10*bitsNum] > 0, 1, -1)

        # Group every ten tracking outputs and wipe off the NH code.
        I_P_group = I_P_InputBits.reshape(-1, 10)
        dataBits = I_P_group@NHCode
        return dataBits, bitSyncPos

    def decodeEphemeris(self, navBitsBin):
        """Decode ephemeris and TOW from one GPS CNAV message.

        The ephemeris for one PRN is decoded message by message. The same
        object is therefore retained so that parameters decoded from earlier
        messages are not lost. CRC checking is completed before this method
        is called.

        Args
        ----
            navBitsBin  - sequence of str
                        One 300-bit navigation message containing only the
                        characters ``'0'`` and ``'1'``.
        Returns
        -------
            TOW         - float
                        Time of week at the start of the current message, in
                        seconds. The SV ephemeris is stored in this object.
        """
        #%% Check whether enough data are available ========================
        if len(navBitsBin) < 300:
            raise ValueError("The parameter navBitsBin must contain 300 bits!")

        # Pi used in the GPS coordinate system.
        gpsPi = 3.1415926535898

        #%% Decode messages needed to compute positioning ==================

        #--- Decode the message ID -----------------------------------------
        # For message contents, refer to IS-GPS-705.
        messageID = self.bin2dec(navBitsBin[14:20])
        self.PRN = self.bin2dec(navBitsBin[8:14])

        #--- Decode messages based on the message ID -----------------------
        # Select the necessary bits and convert them to decimal numbers.
        if messageID == 10:
            #--- It is message type 10 -------------------------------------
            # Message types 10 and 11 together provide the data required to
            # calculate the SV position. Type 10 contains the first part.
            self.idValid[0] = 10
            # Week number.
            self.weekNumber = self.bin2dec(navBitsBin[38:51])
            # Top time of ephemeris prediction.
            self.T_op = self.bin2dec(navBitsBin[54:65])*300
            # L2 health.
            self.health = self.bin2dec(navBitsBin[52:53])
            # ED accuracy index.
            self.URA_ED = twosComp2dec(navBitsBin[65:70])
            # Ephemeris-data reference time of week.
            self.t_oe = self.bin2dec(navBitsBin[70:81])*300
            # Semi-major-axis difference at reference time.
            self.deltaA = twosComp2dec(navBitsBin[81:107])*2**-9
            # Change rate in semi-major axis.
            self.ADot = twosComp2dec(navBitsBin[107:132])*2**-21
            # Mean-motion difference from computed value at reference time.
            self.delta_n_0 = twosComp2dec(navBitsBin[132:149])*2**-44*gpsPi
            # Rate of mean-motion difference from computed value.
            self.delta_n_0Dot = twosComp2dec(navBitsBin[149:172])*2**-57*gpsPi
            # Mean anomaly at reference time.
            self.M_0 = twosComp2dec(navBitsBin[172:205])*2**-32*gpsPi
            # Eccentricity.
            self.e = self.bin2dec(navBitsBin[205:238])*2**-34
            # Argument of perigee.
            self.omega = twosComp2dec(navBitsBin[238:271])*2**-32*gpsPi

        elif messageID == 11:
            #--- It is message type 11 -------------------------------------
            # It contains the second part of the ephemeris parameters.
            self.idValid[1] = 11
            # Ephemeris-data reference time of week.
            self.t_oe = self.bin2dec(navBitsBin[38:49])*300
            # Longitude of ascending node at weekly epoch.
            self.omega_0 = twosComp2dec(navBitsBin[49:82])*2**-32*gpsPi
            # Inclination angle at reference time.
            self.i_0 = twosComp2dec(navBitsBin[82:115])*2**-32*gpsPi
            # Rate of right-ascension difference.
            self.delta_omegaDot = twosComp2dec(navBitsBin[115:132])*2**-44*gpsPi
            # Rate of inclination angle.
            self.i_0Dot = twosComp2dec(navBitsBin[132:147])*2**-44*gpsPi
            # Harmonic correction terms.
            self.C_is = twosComp2dec(navBitsBin[147:163])*2**-30
            self.C_ic = twosComp2dec(navBitsBin[163:179])*2**-30
            self.C_rs = twosComp2dec(navBitsBin[179:203])*2**-8
            self.C_rc = twosComp2dec(navBitsBin[203:227])*2**-8
            self.C_us = twosComp2dec(navBitsBin[227:248])*2**-30
            self.C_uc = twosComp2dec(navBitsBin[248:269])*2**-30

        elif 30 <= messageID <= 37:
            #--- It is message type 30--37 ---------------------------------
            # Message type 30 contains clock, ionosphere, and group-delay
            # parameters. Types 31--37 contain, respectively, clock and
            # reduced almanac, EOP, UTC, differential correction, GGTO, text,
            # and midi almanac data. All carry the common clock-correction
            # parameters decoded below.
            self.idValid[messageID-28] = messageID
            # Clock-data reference time of week.
            self.t_oc = self.bin2dec(navBitsBin[60:71])*300
            # SV clock bias, drift, and drift-rate correction coefficients.
            self.a_f0 = twosComp2dec(navBitsBin[71:97])*2**-35
            self.a_f1 = twosComp2dec(navBitsBin[97:117])*2**-48
            self.a_f2 = twosComp2dec(navBitsBin[117:127])*2**-60

            if messageID == 30:
                #--- It is message type 30 ---------------------------------
                # It contains clock, ionosphere, and group-delay terms.
                self.T_GD = twosComp2dec(navBitsBin[127:140])*2**-35
                self.ISC_L5I = twosComp2dec(navBitsBin[166:179])*2**-35
                self.alpha0 = twosComp2dec(navBitsBin[192:200])*2**-30
                self.alpha1 = twosComp2dec(navBitsBin[200:208])*2**-27
                self.alpha2 = twosComp2dec(navBitsBin[208:216])*2**-24
                self.alpha3 = twosComp2dec(navBitsBin[216:224])*2**-24
                self.beta0 = twosComp2dec(navBitsBin[224:232])*2**11
                self.beta1 = twosComp2dec(navBitsBin[232:240])*2**14
                self.beta2 = twosComp2dec(navBitsBin[240:248])*2**16
                self.beta3 = twosComp2dec(navBitsBin[248:256])*2**16
            # Other terms in messages 31--37 are not decoded at present.

        else:
            # Other message types mainly contain reduced and midi almanacs,
            # UTC parameters, and related data. They are not decoded here.
            self.idValid[10] = messageID

        #%% Compute the time of week of the current message ================
        # The transmitted TOW belongs to the next message. Subtract one
        # six-second message interval to obtain the start of this message.
        TOW = self.bin2dec(navBitsBin[20:37])*6-6
        return TOW

    @classmethod
    def NAVdecoding(cls, I_P_InputBits):
        """Decode GPS L5 CNAV messages in one channel's tracking stream.

        The method performs NH-code synchronization, hard-decision Viterbi
        decoding, preamble search, CRC-24Q checking, message-by-message
        ephemeris decoding, and extraction of the first message TOW.

        Args
        ----
            I_P_InputBits - array-like
                          Output from the tracking function.
        Returns
        -------
            eph         - Ephemeris
                        SV ephemeris.
            firstSubFrame - int or float
                          Starting position of the first CNAV message in the
                          tracking stream, expressed as the number of 1 ms
                          L5 primary-code periods since tracking began. It is
                          infinity if no valid message is detected.
            TOW         - float
                        Time of week of the first message in seconds. It is
                        infinity if no valid message is detected.
        """
        # Eight-bit CNAV preamble in binary form.
        preambleCode = np.array(
            [1, 0, 0, 0, 1, 0, 1, 1], dtype=np.uint8)

        #--- Initialize the ephemeris structure ----------------------------
        # This ensures that the ephemeris of every SV has the same structure
        # when only one, or none, of the requisite messages is decoded.
        eph = cls()
        # Starting position and TOW of the first navigation message.
        firstSubFrame = np.inf
        TOW = np.inf

        #%% Bit synchronization ============================================
        dataBits, bitSyncPos = cls._l5BitSync(I_P_InputBits)
        if dataBits is None:
            return eph, firstSubFrame, TOW

        # Take an even number of input bits for convolutional decoding, then
        # threshold the outputs and convert them to zero and one.
        evenLen = dataBits.size-dataBits.size % 2
        dataBits = (dataBits[:evenLen] < 0).astype(np.uint8)

        #%% CNAV-data decoding =============================================
        # The first input may be the G1 or G2 output. Decode both alignments
        # to find the stream that starts with G1.
        for G1orG2 in range(2):
            alignedBits = dataBits[G1orG2:evenLen-G1orG2]
            decodedBits = cls._viterbiDecode(alignedBits)

            # Find and analyze every complete preamble-like pattern.
            for index in range(max(0, decodedBits.size-299)):
                preamble = decodedBits[index:index+8]
                if (not np.array_equal(preamble, preambleCode)
                        and not np.array_equal(
                            preamble, 1-preambleCode)):
                    continue

                #=== Read bits for CRC-24Q and ephemeris decoding ==========
                navBits = decodedBits[index:index+300].copy()

                # Correct the polarity of all data bits from the preamble.
                if not np.array_equal(navBits[:8], preambleCode):
                    navBits = 1-navBits

                # CRC-24Q must pass before the ephemeris is decoded.
                if not cls._crc24qCheck(navBits):
                    continue

                # Convert to the binary-character form used by the ephemeris
                # decoder and decode the current message.
                messageTOW = eph.decodeEphemeris(navBits.astype(str))

                # Save the position and TOW only for the first valid message.
                if not np.isfinite(firstSubFrame):
                    firstSubFrame = (index*2+G1orG2)*10+bitSyncPos
                    TOW = messageTOW
                    eph.TOW = TOW

            # Once a valid message is found, no alternate alignment is needed.
            if np.isfinite(firstSubFrame):
                break
        return eph, firstSubFrame, TOW

    def satpos(self, transmitTime):
        """Calculate one satellite's ECEF position and clock correction.

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
        #%% Initialize constants ===========================================
        gpsPi = 3.1415926535898
        # Earth rotation rate [rad/s].
        Omegae_dot = 7.2921151467e-5
        # Earth's universal gravitational constant [m^3/s^2].
        GM = 3.986005e14
        # Relativistic-correction constant [s/m^(1/2)].
        F = -4.442807633e-10
        # Reference semi-major axis [m].
        A_REF = 26559710.0
        # Reference rate of right ascension [rad/s].
        omegaDot_REF = -2.6e-9*gpsPi

        #%% 1 Transmitting-time correction --------------------------------

        # Find the time difference while accounting for week crossover.
        dt = self.check_t(transmitTime-self.t_oc)

        # Calculate the clock correction. The relativistic term is included
        # later after the eccentric anomaly has been calculated.
        satClkCorr = (self.a_f2*dt+self.a_f1)*dt+self.a_f0

        # Include group delay when message type 30 has been decoded. If the
        # recording is not long enough to decode type 30, this correction is
        # omitted from the PVT calculation. Refer to IS-GPS-705 for the
        # T_GD and ISC_L5I definitions.
        if self.idValid[2] == 30:
            satClkCorr = satClkCorr-self.T_GD+self.ISC_L5I

        # Correct the transmission time with the satellite clock error.
        time = transmitTime-satClkCorr

        #%% 2 Compute satellite position -----------------------------------
        # The processing sequence follows the broadcast-orbit algorithm in
        # Xie Gang, Principles of GNSS: GPS, GLONASS and Galileo, p. 248.

        #---- 2.1 Time from ephemeris reference time ----------------------
        tk = self.check_t(time-self.t_oe)

        #---- 2.2 Compute semi-major axis ---------------------------------
        # Unlike legacy GPS NAV, CNAV broadcasts a difference from the
        # reference semi-major axis and its rate of change.
        A_0 = A_REF+self.deltaA
        A = A_0+self.ADot*tk

        #---- 2.3 Compute mean motion -------------------------------------
        n0 = np.sqrt(GM/A_0**3)
        # Unlike legacy GPS NAV, CNAV supplies both the mean-motion
        # difference and its rate of change.
        delta_n = self.delta_n_0+0.5*self.delta_n_0Dot*tk
        n = n0+delta_n

        #---- 2.4 Mean-anomaly computation -------------------------------
        M = np.remainder(self.M_0+n*tk, 2*gpsPi)

        #---- 2.5 Eccentric anomaly --------------------------------------
        E = M
        for _ in range(10):
            E_old = E
            E = M+self.e*np.sin(E)
            dE = np.fmod(E-E_old, 2*gpsPi)
            if abs(dE) < 1e-12:
                break
        E = np.remainder(E, 2*gpsPi)

        #---- 2.6 True anomaly -------------------------------------------
        nu = np.arctan2(np.sqrt(1-self.e**2)*np.sin(E),
                        np.cos(E)-self.e)

        #---- 2.7 Argument of latitude -----------------------------------
        phi = np.remainder(nu+self.omega, 2*gpsPi)

        #---- 2.8--2.9 Correct latitude, radius, and inclination ----------
        u = phi+self.C_uc*np.cos(2*phi)+self.C_us*np.sin(2*phi)
        r = (A*(1-self.e*np.cos(E))
             + self.C_rc*np.cos(2*phi)+self.C_rs*np.sin(2*phi))
        i = (self.i_0+self.i_0Dot*tk
             + self.C_ic*np.cos(2*phi)+self.C_is*np.sin(2*phi))

        #---- 2.10 SV position in the orbital plane ----------------------
        xk1 = np.cos(u)*r
        yk1 = np.sin(u)*r

        #---- 2.11 Longitude of ascending node ---------------------------
        # Unlike legacy GPS NAV, CNAV broadcasts a difference from the
        # reference rate of right ascension.
        omegaDot = omegaDot_REF+self.delta_omegaDot
        Omega = np.remainder(
            self.omega_0+(omegaDot-Omegae_dot)*tk
            - Omegae_dot*self.t_oe, 2*gpsPi)

        #---- 2.12 Compute satellite coordinates in ECEF -----------------
        satPosition = np.array([
            xk1*np.cos(Omega)-yk1*np.cos(i)*np.sin(Omega),
            xk1*np.sin(Omega)+yk1*np.cos(i)*np.cos(Omega),
            yk1*np.sin(i),
        ])

        #%% 3 Include relativistic correction in clock correction ----------
        dtr = F*self.e*np.sqrt(A)*np.sin(E)
        satClkCorr += dtr
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

        # Measurement location and receiver GPS time.
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
        # At least three CNAV messages are needed: type 10, type 11, and any
        # one message from types 30--37. Each message is six seconds long, so
        # the requisite data occupy at least 18 seconds. Tracking can start in
        # the middle of a message; the MATLAB receiver therefore adds one
        # message interval and requires a record of at least 24 seconds.
        if settings.msToProcess < 24000:
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid message in the 1 ms prompt-I
        # stream. Infinity indicates that no valid preamble was detected.
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
            print(f"Decoding CNAV for PRN {PRN:02d} --------------------")

            (currentEph, subFrameStart[channelNr], TOW[channelNr]) = (
                Ephemeris.NAVdecoding(trackResults[channelNr].I_P))
            currentEph.PRN = PRN
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without requisite CNAV messages -------
            clockReady = np.any(
                currentEph.idValid[2:10] == np.arange(30, 38))
            if (currentEph.idValid[0] != 10
                    or currentEph.idValid[1] != 11
                    or not clockReady):
                # Print the same message-by-message decoding information as
                # the MATLAB post-navigation implementation.
                if currentEph.idValid[0] != 10:
                    print(f"    Message type 10 for PRN {PRN:02d} "
                          "not decoded.")
                if currentEph.idValid[1] != 11:
                    print(f"    Message type 11 for PRN {PRN:02d} "
                          "not decoded.")
                if not clockReady:
                    print(f"    None of message types 30--37 for PRN "
                          f"{PRN:02d} decoded.")
                print(f"    Channel for PRN {PRN:02d} excluded!")
                activeChnList.remove(channelNr)
            else:
                print(f"    Three requisite messages for PRN {PRN:02d} "
                      "all decoded!")

        #%% Check whether at least four satellites remain
        # A three-dimensional receiver position and clock bias require four
        # or more satellites with decoded ephemerides.
        if len(activeChnList) < 4:
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
