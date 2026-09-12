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
GPS L1 C/A navigation-message decoding and position calculation.

"""

import numpy as np
from scipy.signal import correlate

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot, twosComp2dec,
)


#%% Broadcast ephemeris
class Ephemeris:
    """Keep a common ephemeris structure for one satellite.

    This ensures that the ephemeris of each satellite has a similar structure
    when only one or none of the three requisite messages is decoded.
    """

    def __init__(self):
        """Initialize a common ephemeris structure for one satellite.

        Returns
        -------
            None
        """
        # Five-element marker retained from MATLAB. The entries are not used
        # as independent Boolean flags; readiness is tested from the IOD
        # fields.
        self.idValid = np.zeros(5, dtype=np.int8)
        # Satellite PRN number.
        self.PRN: int | None = None

        #--- It is subframe 1 -------------------------------------------------
        # It contains week number, SV clock corrections, health and accuracy.
        self.weekNumber: int | None = None
        self.accuracy:   int | None = None
        self.health:     int | None = None
        self.T_GD:     float | None = None
        self.IODC:       int | None = None
        self.t_oc:     float | None = None
        self.a_f2:     float | None = None
        self.a_f1:     float | None = None
        self.a_f0:     float | None = None

        #--- It is subframe 2 -------------------------------------------------
        # It contains the first part of the ephemeris parameters.
        self.IODE_sf2:   int | None = None
        self.C_rs:     float | None = None
        self.deltan:   float | None = None
        self.M_0:      float | None = None
        self.C_uc:     float | None = None
        self.e:        float | None = None
        self.C_us:     float | None = None
        self.sqrtA:    float | None = None
        self.t_oe:     float | None = None

        #--- It is subframe 3 -------------------------------------------------
        # It contains the second part of the ephemeris parameters.
        self.C_ic:     float | None = None
        self.omega_0:  float | None = None
        self.C_is:     float | None = None
        self.i_0:      float | None = None
        self.C_rc:     float | None = None
        self.omega:    float | None = None
        self.omegaDot: float | None = None
        self.IODE_sf3:   int | None = None
        self.iDot:     float | None = None

        #--- It is subframe 4 -------------------------------------------------
        # Subframe 4 contains almanac, ionospheric-model and UTC parameters,
        # together with the health of SVs PRN 25--32. It is not decoded at
        # present.

        #--- It is subframe 5 -------------------------------------------------
        # Subframe 5 contains the almanac and health of SVs PRN 1--24,
        # together with the almanac reference week number and time. It is
        # not decoded at present.

        # Time of week of the first decoded subframe [s].
        self.TOW: float | None = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary character sequence to an unsigned decimal integer.

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
    def navPartyChk(ndat):
        """Compute and check the parity bits on one GPS navigation word.

        The procedure is based on the flowchart in Figure 2-10 of the second
        edition of the GPS SPS Signal Specification.

        Args
        ----
            ndat        - array-like
                        A 32-bit array representing a 30-bit GPS navigation
                        word plus the two previous bits used in the parity
                        calculation.
        Returns
        -------
            status      - int
                        ``+1`` or ``-1`` if parity passes, or zero if parity
                        fails. ``+1`` means bits 1--24 have the correct
                        polarity; ``-1`` means they must be inverted.
        """
        ndat = np.asarray(ndat).copy()

        # To implement the exclusive-or operation using multiplication,
        # MATLAB represents a binary zero by -1 and a binary one by +1.
        # The corresponding truth table is:
        #
        #   a  b  parity XOR      a-sign  b-sign  product
        #   0  0       1            -1      -1        1
        #   0  1       0            -1       1       -1
        #   1  0       0             1      -1       -1
        #   1  1       1             1       1        1

        #--- Check whether the data bits must be inverted ---------------------
        # Check whether D30* requires inversion of the 24 data bits.
        if ndat[1] != 1:
            ndat[2:26] = -ndat[2:26]

        #--- Calculate the six parity bits ------------------------------------
        # Calculate the six parity bits. ndat[0:2] are D29* and D30*,
        # ndat[2:26] are d1--d24, and ndat[26:32] are received D25--D30.
        parity = np.empty(6)
        parity[0] = np.prod(ndat[[0, 2, 3, 4, 6, 7, 11, 12, 13,
                                  14, 15, 18, 19, 21, 24]])
        
        parity[1] = np.prod(ndat[[1, 3, 4, 5, 7, 8, 12, 13, 14,
                                  15, 16, 19, 20, 22, 25]])
        
        parity[2] = np.prod(ndat[[0, 2, 4, 5, 6, 8, 9, 13, 14,
                                  15, 16, 17, 20, 21, 23]])
        
        parity[3] = np.prod(ndat[[1, 3, 5, 6, 7, 9, 10, 14, 15,
                                  16, 17, 18, 21, 22, 24]])
        
        parity[4] = np.prod(ndat[[1, 2, 4, 6, 7, 8, 10, 11, 15,
                                  16, 17, 18, 19, 22, 23, 25]])
        
        parity[5] = np.prod(ndat[[0, 4, 6, 7, 9, 10, 11, 12, 14,
                                  16, 20, 23, 24, 25]])

        #--- Compare received parity with calculated parity -------------------
        # If parity passes, the output is -1 or +1 depending on whether the
        # data bits must be inverted. ndat[1] is D30*, the last bit of the
        # previous word. A parity failure returns zero.
        return int(-ndat[1]) if np.all(parity == ndat[26:32]) else 0

    @staticmethod
    def checkPhase(word, D30Star):
        """Check the parity of the supplied 30-bit word.
        The last parity bit of the previous word is used for the calculation.
        A note on the procedure is supplied by the GPS Standard Positioning
        Service Signal Specification.

        Args
        ----
            word       - iterable of str
                       An array with 30 bit long word from the navigation
                       message (a character array, must contain only '0' or '1').
            D30Star    - str
                       The last bit of the previous word.
        Returns
        -------
            word       - list of str
                       Word with corrected polarity of the data bits.
        """

        word = list(word)
        if D30Star == "1":
            # Data bits must be inverted
            word[:24] = ["1" if bit == "0" else "0"
                         for bit in word[:24]]
        return word

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
            return time - 2 * half_week
        if time < -half_week:
            return time + 2 * half_week
        return time

    def decodeEphemeris(self, bits, D30Star):
        """Decode ephemerides and TOW from the supplied bit stream.

        The bit stream must contain 1500 bits, and its first element must be
        the first bit of a subframe. The ID of the first subframe is not
        important. This function does not check parity.

        Args
        ----
            bits        - sequence of str
                        Bits of five navigation-message subframes. It must
                        contain only the characters ``'0'`` and ``'1'``.
            D30Star     - str
                        The last bit of the previous navigation word. Refer
                        to IS-GPS-200D for details of parity checking. It must
                        contain only ``'0'`` or ``'1'``.
        Returns
        -------
            TOW         - float
                        Time of week of the first subframe in the bit stream,
                        in seconds. The SV ephemeris is stored in this object.
        """
        #%% Check whether enough data are available
        # Check that five complete 300-bit subframes are available.
        if len(bits) < 1500:
            raise ValueError("The parameter BITS must contain 1500 bits!")

        #%% Initialize constants
        # Pi used in the GPS coordinate system.
        gpsPi = 3.1415926535898

        #%% Decode all five subframes
        for i in range(5):
            #--- Cut one subframe from the navigation-bit stream --------------
            subframe = list(bits[300*i:300*(i+1)])

            #--- Correct polarity of the data bits in all ten words -----------
            for ind in range(10):
                wordStart = 30 * ind
                wordEnd   = wordStart + 30
                subframe[wordStart:wordEnd] = self.checkPhase(
                                          subframe[wordStart:wordEnd], D30Star)
                D30Star = subframe[wordEnd-1]

            #--- Decode the subframe ID ---------------------------------------
            # Field definitions follow IS-GPS-200.
            subframeID = self.bin2dec(subframe[49:52])

            #--- Decode the subframe according to its ID ----------------------
            # The required bits are
            # selected and converted to decimal values as specified in the
            # GPS interface control document IS-GPS-200D.
            if subframeID == 1:
                #--- It is subframe 1 -----------------------------------------
                # It contains week number, SV clock corrections, health and
                # accuracy.
                self.weekNumber = self.bin2dec(subframe[60:70]) + 1024
                
                self.accuracy   = self.bin2dec(subframe[72:76])
                
                self.health     = self.bin2dec(subframe[76:82])
                
                self.T_GD       = twosComp2dec(subframe[196:204]) * 2**-31
                
                self.IODC   = self.bin2dec(subframe[82:84] + subframe[196:204])
                
                self.t_oc = self.bin2dec(subframe[218:234]) * 2**4
                
                self.a_f2 = twosComp2dec(subframe[240:248]) * 2**-55
                
                self.a_f1 = twosComp2dec(subframe[248:264]) * 2**-43
                
                self.a_f0 = twosComp2dec(subframe[270:292]) * 2**-31
                
                self.idValid[0] = 1

            elif subframeID == 2:
                #--- It is subframe 2 -----------------------------------------
                # It contains the first part of the ephemeris parameters.
                self.IODE_sf2 = self.bin2dec(subframe[60:68])
                
                self.C_rs     = twosComp2dec(subframe[68:84]) * 2**-5
                
                self.deltan   = (twosComp2dec(subframe[90:106]) * 2**-43 * gpsPi)
                
                self.M_0 = twosComp2dec(
                    subframe[106:114] + subframe[120:144]) * 2**-31 * gpsPi
                
                self.C_uc = twosComp2dec(subframe[150:166]) * 2**-29
                
                self.e    = self.bin2dec(
                    subframe[166:174] + subframe[180:204]) * 2**-33
                
                self.C_us = twosComp2dec(subframe[210:226]) * 2**-29
                
                self.sqrtA = self.bin2dec(
                    subframe[226:234] + subframe[240:264]) * 2**-19
                
                self.t_oe = self.bin2dec(subframe[270:286]) * 2**4
                # MATLAB writes marker 2 into the first entry, replacing the
                # subframe-1 marker; navigation therefore tests the IOD fields.
                self.idValid[0] = 2

            elif subframeID == 3:
                #--- It is subframe 3 -----------------------------------------
                # It contains the second part of the ephemeris parameters.
                self.C_ic = twosComp2dec(subframe[60:76]) * 2**-29
                
                self.omega_0 = twosComp2dec(
                    subframe[76:84] + subframe[90:114]) * 2**-31 * gpsPi
                
                self.C_is = twosComp2dec(subframe[120:136]) * 2**-29
                
                self.i_0 = twosComp2dec(
                    subframe[136:144] + subframe[150:174]) * 2**-31 * gpsPi
                
                self.C_rc = twosComp2dec(subframe[180:196]) * 2**-5
                
                self.omega = twosComp2dec(
                    subframe[196:204] + subframe[210:234]) * 2**-31 * gpsPi
                
                self.omegaDot = (twosComp2dec(subframe[240:264])
                                 * 2**-43 * gpsPi)
                
                self.IODE_sf3 = self.bin2dec(subframe[270:278])
                
                self.iDot = (twosComp2dec(subframe[278:292]) * 2**-43 * gpsPi)
                
                self.idValid[2] = 3

        # Subframe 4 contains almanac, ionospheric-model and UTC parameters,
        # and the health of SVs PRN 25--32. It is not decoded at present.
        # Subframe 5 contains the almanac and health of SVs PRN 1--24,
        # together with the almanac reference week number and time. It is
        # not decoded at present.

        # Compute TOW of the first supplied subframe. The HOW in the final
        # subframe contains the TOW of the following subframe, so subtract
        # the duration of the five decoded subframes (30 s).
        self.TOW = self.bin2dec(subframe[30:47]) * 6 - 30
        return self.TOW

    @classmethod
    def NAVdecoding(cls, I_P_InputBits):
        """Find the first preamble occurrence in one channel's bit stream.

        The preamble is verified by checking the six-second spacing between
        preambles and the parity of the first two words in a subframe.

        Args
        ----
            I_P_InputBits - array-like
                          Output from the tracking function.
        Returns
        -------
            eph         - Ephemeris
                        SV ephemeris.
            subFrameStart - int or float
                          Zero-based starting position of the first message,
                          counted in 1 ms C/A-code periods from the start of
                          tracking. It is ``numpy.inf`` if no preamble is found.
            TOW         - float
                        Time of week of the first message, in seconds. It is
                        set to ``numpy.inf`` if no valid preamble is found.
        """
        # Initialize a common ephemeris structure. This ensures that the
        # ephemeris of every SV has the same structure when only one, or none,
        # of the three requisite navigation messages is decoded.
        eph = cls()
        # Starting position and TOW of the first navigation message.
        subFrameStart = np.inf
        TOW = np.inf

        #%% Bit and frame synchronization
        # Preamble search may be delayed to avoid tracking-loop transients.
        searchStartOffset = 0

        #--- Generate the preamble pattern ------------------------------------
        # Generate the eight-bit preamble pattern.
        preamble_bits = np.array([1, -1, -1, -1, 1, -1, 1, 1])
        
        # Upsample the preamble to 20 values per bit so that it can be found
        # with the precision of one 1 ms tracking sample.
        preamble_ms = np.kron(preamble_bits, np.ones(20))

        #%% Correlate tracking output with the preamble
        # Read the tracking output containing navigation bits. The beginning
        # of the record can be skipped to avoid tracking-loop transients.
        # Threshold the prompt output and convert it to -1 and +1.
        bits = np.where(np.asarray(I_P_InputBits[searchStartOffset:]) > 0, 1, -1)

        # Correlate the tracking output with the preamble pattern.
        preamble_ms = np.pad(preamble_ms, (0, bits.size-preamble_ms.size))
        tlmXcorrResult = correlate(bits, preamble_ms)

        #--- Find all starting points of preamble-like patterns ---------------
        # Find all preamble-like patterns. An ideal 160-sample match gives a
        # magnitude of 160; the MATLAB threshold retains values above 153.
        # Zero delay is element bits.size-1 for two equal-length sequences.
        index = np.flatnonzero(
            np.abs(tlmXcorrResult[bits.size-1:]) > 153)
        index += searchStartOffset

        #%% Analyze detected preamble-like patterns
        # For every occurrence, find the time distance to the remaining
        # patterns. A candidate followed by another occurrence after 6000 ms
        # spans one subframe and is verified using the first two GPS words.
        for candidate in index:
            if np.any(index-candidate == 6000) and candidate >= 40:
                # === Re-read bit values for preamble verification ============
                # Re-read 62 bits: two bits from the preceding word are
                # required for parity, followed by the TLM and HOW words.
                # Combining the 20 prompt values per bit improves decisions
                # for noisy signals. The candidate index points to the start
                # of the TLM word.
                navWordSamples = I_P_InputBits[candidate-40:candidate+20*60]
                
                #--- Combine the 20 values of each bit ------------------------
                navWords = np.asarray(navWordSamples).reshape(62, 20).sum(1)
                # Threshold the combined bits and convert them to -1 and +1.
                navWords = np.where(navWords > 0, 1, -1)

                #--- Check the parity of the TLM and HOW words ----------------
                if (cls.navPartyChk(navWords[:32]) != 0
                        and cls.navPartyChk(navWords[30:62]) != 0):
                    # Parity is valid. Record the preamble start and skip the
                    # remaining candidates for this channel.
                    subFrameStart = int(candidate)
                    break

        # Exclude this channel from further navigation processing when no
        # valid preamble was detected.
        if np.isinf(subFrameStart):
            print("Could not find valid preambles in channel!")
            return eph, subFrameStart, TOW

        #%% Decode ephemerides
        # === Convert tracking output to navigation bits ======================
        # --- Copy five subframes from tracking output ------------------------
        # Convert the tracking output to navigation bits. Copy five complete
        # subframes together with D30* from the preceding word.
        navBitsSamples = I_P_InputBits[subFrameStart-20:subFrameStart+1500*20]
        # --- Group and sum the 20 values of every bit ------------------------
        # Group the 20 tracking values belonging to every bit and sum them to
        # obtain the best bit estimate.
        navBits = np.asarray(navBitsSamples).reshape(1501, 20).sum(1)

        #--- Threshold and convert to binary characters -----------------------
        # Threshold the decisions. The comparison produces one where the
        # condition is true and zero otherwise. Convert the result to the
        # binary-character representation expected by decodeEphemeris().
        navBitsBin = np.where(navBits > 0, "1", "0").tolist()

        #=== Decode ephemerides and TOW of the first subframe =================
        TOW = eph.decodeEphemeris(navBitsBin[1:1501], navBitsBin[0])
        return eph, subFrameStart, TOW

    def satpos(self, transmitTime):
        """Calculate satellite X, Y, and Z coordinates at transmit time.

        The coordinates are calculated from the ephemeris stored in this
        object.

        Args
        ----
            transmitTime - float
                         Satellite signal transmission time in seconds.
        Returns
        -------
            satPosition - numpy.ndarray
                        Satellite position in the ECEF coordinate system
                        ``[X, Y, Z]``.
            satClkCorr  - float
                        Satellite-clock correction in seconds.
        """
        from math import atan2

        #%% Initialize constants
        #--- GPS constants ----------------------------------------------------
        # GPS constants.
        # Pi used in the GPS coordinate system.
        gpsPi = 3.1415926535898
        
        #--- Constants for satellite-position calculation ---------------------
        Omegae_dot = 7.2921151467e-5   # Earth rotation rate [rad/s].
        GM = 3.986005e14               # Earth gravitational constant [m^3/s^2].
        F = -4.442807633e-10           # Relativistic constant [s/sqrt(m)].

        #%% Find initial satellite clock correction
        # Find the time difference from the clock-reference epoch, including
        # beginning- or end-of-week crossover.
        dt = self.check_t(transmitTime-self.t_oc)
        
        # Calculate the initial satellite-clock correction [s].
        satClkCorr = ((self.a_f2*dt+self.a_f1)*dt + self.a_f0-self.T_GD)
        time = transmitTime-satClkCorr

        #%% Find satellite position at corrected transmit time
        # Restore the semi-major axis.
        a = self.sqrtA*self.sqrtA
        
        # Correct time from the ephemeris reference epoch.
        tk = self.check_t(time-self.t_oe)
        
        # Calculate the initial mean motion.
        n0 = np.sqrt(GM/a**3)
        
        # Calculate the corrected mean motion.
        n = n0+self.deltan
        
        # Calculate the mean anomaly.
        # Reduce the mean anomaly to between 0 and 360 degrees.
        M = np.fmod(self.M_0+n*tk+2*gpsPi, 2*gpsPi)

        # Use the mean anomaly as the initial guess of eccentric anomaly.
        E = M
        #--- Iteratively compute the eccentric anomaly ------------------------
        for _ in range(10):
            E_old = E
            E     = M+self.e*np.sin(E)
            dE    = np.fmod(E-E_old, 2*gpsPi)
            if abs(dE) < 1e-12:
                # The requested precision has been reached.
                break
            
        # Reduce the eccentric anomaly to between 0 and 360 degrees.
        E = np.fmod(E+2*gpsPi, 2*gpsPi)

        # Calculate the relativistic correction.
        dtr = F*self.e*self.sqrtA*np.sin(E)
        
        # Calculate the true anomaly.
        nu = atan2(np.sqrt(1-self.e**2)*np.sin(E), np.cos(E)-self.e)
        
        # Compute the angle phi and reduce it to between 0 and 360 degrees.
        phi = np.fmod(nu+self.omega, 2*gpsPi)

        # Correct the argument of latitude.
        u = (phi+self.C_uc*np.cos(2*phi) + self.C_us*np.sin(2*phi))
        # Correct the orbital radius.
        
        r = (a*(1 - self.e*np.cos(E)) + self.C_rc*np.cos(2*phi)
                                    + self.C_rs*np.sin(2*phi))
        # Correct the inclination.
        
        i = (self.i_0 + self.iDot*tk + self.C_ic*np.cos(2*phi) 
                                             + self.C_is*np.sin(2*phi))

        # Calculate the satellite position in its orbital plane.
        xk1 = np.cos(u)*r
        yk1 = np.sin(u)*r
        
        # Compute the angle between the ascending node and the Greenwich
        # meridian.
        Omega = (self.omega_0 + (self.omegaDot - Omegae_dot) * tk
                 - Omegae_dot * self.t_oe)
        
        # Reduce the ascending-node angle to between 0 and 360 degrees.
        Omega = np.fmod(Omega + 2 * gpsPi, 2 * gpsPi)

        #--- Compute satellite coordinates ------------------------------------
        # Compute satellite coordinates in the ECEF system.
        satPosition = np.array([xk1 * np.cos(Omega)-yk1*np.cos(i)*np.sin(Omega),
                                xk1*np.sin(Omega)+yk1*np.cos(i)*np.cos(Omega),
                                yk1*np.sin(i),])

        #%% Include relativistic correction in the clock correction
        satClkCorr = ((self.a_f2*dt+self.a_f1)*dt + self.a_f0 - self.T_GD + dtr)
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
        # Satellite coordinates require at least subframes 1, 2 and 3.
        # Tracking starts at an arbitrary point, so the first received
        # subframe can be any of the five. One subframe lasts 6 s; therefore
        # five subframes require 30 s (30000 ms), with additional time for a
        # record that starts in the middle of a subframe.
        if settings.msToProcess < 36000:
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
            print(f"Decoding NAV for PRN {PRN:02d} --------------------")

            currentEph, subFrameStart[channelNr], TOW[channelNr] = (
                            Ephemeris.NAVdecoding(trackResults[channelNr].I_P))
            currentEph.PRN = PRN
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all three requisite subframes from
            # subsequent positioning.
            if (currentEph.IODC is None
                    or currentEph.IODE_sf2 is None
                    or currentEph.IODE_sf3 is None):
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
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
