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
GPS L1C CNAV-2 ephemeris decoding and position calculation.

"""

import numpy as np

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot, twosComp2dec,
)
from channelDecoder import BCHDecoder, LDPCDecoder
from correlator import generate2ndCode


#%% GPS L1C broadcast ephemeris
class Ephemeris:
    """Keep and decode one GPS L1C CNAV-2 broadcast ephemeris.

    This ensures that the ephemeris of each satellite has a similar structure
    when only one or none of the requisite CNAV-2 subframes is decoded.
    """

    def __init__(self):
        """Initialize the CNAV-2 ephemeris and validity fields.

        Returns
        -------
            None
        """
        #%% Initialization and flags
        # Flag indicating that the requisite CNAV-2 data were decoded.
        self.flag = 0
        # Time of week of the first decoded CNAV-2 frame [s].
        self.TOW = None
        # Satellite PRN number.
        self.PRN = None

        #%% Subframe 2: ephemeris and clock
        #--- General data fields ----------------------------------------------
        # GPS week number.
        self.WN = None
        # Interval time of week.
        self.ITOW = None
        # Data-sequence propagation time of week.
        self.t_op = None
        # L1C signal-health indicator.
        self.L1CHealth = None
        # Elevation-dependent user-range-accuracy index.
        self.URAEDIndex = None
        # Ephemeris reference time [s].
        self.t_oe = None

        #--- Orbit parameters -------------------------------------------------
        # Semi-major-axis difference at the reference time.
        self.deltaA = None
        # Change rate in the semi-major axis.
        self.ADot = None
        # Mean-motion difference from the computed value.
        self.delta_n_0 = None
        # Rate of change of the mean-motion difference.
        self.delta_n_0Dot = None
        # Mean anomaly at the reference time.
        self.M_0 = None
        # Eccentricity.
        self.e = None
        # Argument of perigee.
        self.omega = None
        # Longitude of the ascending node at the weekly epoch.
        self.omega_0 = None
        # Inclination angle at the reference time.
        self.i_0 = None
        # Rate of right ascension difference.
        self.omegaDot = None
        # Rate of inclination angle.
        self.IDOT = None

        #--- Harmonic correction terms ---------------------------------------
        # Sine and cosine corrections to the inclination angle.
        self.C_is = None
        self.C_ic = None
        # Sine and cosine corrections to the orbit radius.
        self.C_rs = None
        self.C_rc = None
        # Sine and cosine corrections to the argument of latitude.
        self.C_us = None
        self.C_uc = None

        #--- Accuracy and clock parameters -----------------------------------
        # Non-elevation-dependent user-range-accuracy indices.
        self.URANED0Index = None
        self.URANED1Index = None
        self.URANED2Index = None
        # SV clock-bias, drift and drift-rate correction coefficients.
        self.a_0 = None
        self.a_1 = None
        self.a_2 = None
        # Group-delay differential correction.
        self.T_GD = None
        # Inter-signal corrections for the L1C pilot and data components.
        self.ISC_L1Cp = None
        self.ISC_L1Cd = None
        # Integrity-status flag.
        self.ISF = None
        # CEI data-sequence propagation week number.
        self.WN_op = None

        #%% Subframe 3: variable data
        # Page-validity markers.
        self.PageID1 = None
        self.PageID2 = None
        # GPS/GNSS time-offset parameters.
        self.GGTO = {}
        # Earth-orientation parameters.
        self.EOP = {}

    @staticmethod
    def bin2dec(bits):
        """Convert a binary sequence to an unsigned decimal integer.

        Args
        ----
            bits        - array-like
                        Binary values containing only zero or one.
        Returns
        -------
            decimal     - int
                        Unsigned decimal value represented by ``bits``.
        """
        return int("".join(np.asarray(bits, dtype=str)), 2)

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
        if time > 302400:
            return time - 604800
        if time < -302400:
            return time + 604800
        return time

    def decodeEphemeris(self, navBitsBin):
        """Decode GPS L1C CNAV-2 ephemeris and TOW.

        The input must contain one 883-bit sequence: TOI(9), subframe 2(600),
        and subframe 3(274). BCH, LDPC and CRC checks are completed before
        this method is called.

        Args
        ----
            navBitsBin  - array-like
                        Bits of the navigation message containing only zero
                        and one. The ephemeris is decoded frame by frame so
                        this object retains parameters decoded previously.
        Returns
        -------
            None
                        TOW and the decoded SV ephemeris are stored in this
                        object.
        """
        #%% Check and prepare the supplied bit sequence
        # Keep the bits in one row as required by the field-conversion logic.
        bits = np.asarray(navBitsBin, dtype=np.int8).reshape(-1)
        # Pi used in the GPS coordinate system.
        gpsPi = 3.1415926535898
        u = self.bin2dec
        s = twosComp2dec

        #%% Decode subframe 1: Time of Interval
        # TOI is the number of 18-second epochs since the start of the current
        # two-hour interval-time-of-week period.
        TOI = u(bits[:9])

        #%% Decode subframe 2: ephemeris and clock
        # Subframe 2 contains 600 non-variable ephemeris and clock bits.
        sf2 = bits[9:609]
        #--- General data fields ----------------------------------------------
        # Week number.
        self.WN = u(sf2[0:13])
        # Interval time of week: number of two-hour epochs in the week.
        self.ITOW = u(sf2[13:21])
        # Data-sequence propagation time of week.
        self.t_op = u(sf2[21:32]) * 300
        # L1C signal health.
        self.L1CHealth = u(sf2[32:33])
        # Elevation-dependent user-range-accuracy index.
        self.URAEDIndex = s(sf2[33:38])
        # Ephemeris reference time; CNAV-2 uses t_oe as the clock-reference
        # time in the subsequent satellite-position calculation.
        self.t_oe = u(sf2[38:49]) * 300

        #--- Orbit parameters (IS-GPS-800 Table 3.5-1) ------------------------
        # Semi-major-axis difference and its change rate.
        self.deltaA = s(sf2[49:75]) * 2**-9
        self.ADot = s(sf2[75:100]) * 2**-21
        # Mean-motion difference and its change rate. Values transmitted in
        # semicircles are converted to radians.
        self.delta_n_0 = s(sf2[100:117]) * 2**-44 * gpsPi
        self.delta_n_0Dot = s(sf2[117:140]) * 2**-57 * gpsPi
        # Mean anomaly at the reference time.
        self.M_0 = s(sf2[140:173]) * 2**-32 * gpsPi
        # Eccentricity.
        self.e = u(sf2[173:206]) * 2**-34
        # Argument of perigee.
        self.omega = s(sf2[206:239]) * 2**-32 * gpsPi
        # Longitude of the ascending node at the weekly epoch.
        self.omega_0 = s(sf2[239:272]) * 2**-32 * gpsPi
        # Inclination angle at the reference time.
        self.i_0 = s(sf2[272:305]) * 2**-32 * gpsPi
        # Rate of right ascension difference.
        self.omegaDot = s(sf2[305:322]) * 2**-44 * gpsPi
        # Rate of inclination angle.
        self.IDOT = s(sf2[322:337]) * 2**-44 * gpsPi

        #--- Harmonic correction terms ---------------------------------------
        # These correction terms are expressed directly in radians or metres;
        # they are therefore not multiplied by pi.
        # Sine and cosine corrections to the inclination angle.
        self.C_is = s(sf2[337:353]) * 2**-30
        self.C_ic = s(sf2[353:369]) * 2**-30
        # Sine and cosine corrections to the orbit radius.
        self.C_rs = s(sf2[369:393]) * 2**-8
        self.C_rc = s(sf2[393:417]) * 2**-8
        # Sine and cosine corrections to the argument of latitude.
        self.C_us = s(sf2[417:438]) * 2**-30
        self.C_uc = s(sf2[438:459]) * 2**-30

        #--- Accuracy and clock parameters -----------------------------------
        # Non-elevation-dependent user-range-accuracy indices.
        self.URANED0Index = s(sf2[459:464])
        self.URANED1Index = u(sf2[464:467])
        self.URANED2Index = u(sf2[467:470])
        # SV clock-bias, drift and drift-rate correction coefficients.
        self.a_0 = s(sf2[470:496]) * 2**-35
        self.a_1 = s(sf2[496:516]) * 2**-48
        self.a_2 = s(sf2[516:526]) * 2**-60
        # Group-delay differential correction.
        self.T_GD = s(sf2[526:539]) * 2**-35
        # Inter-signal corrections for the L1C pilot and data components.
        self.ISC_L1Cp = s(sf2[539:552]) * 2**-35
        self.ISC_L1Cd = s(sf2[552:565]) * 2**-35
        # Integrity-status flag.
        self.ISF = u(sf2[565:566])
        # CEI data-sequence propagation week number.
        self.WN_op = u(sf2[566:574])

        #%% Decode subframe 3: variable data
        # Subframe 3 contains 274 variable page bits.
        sf3 = bits[609:883]
        # PRN and page identifier.
        self.PRN = u(sf3[0:8])
        pageID = u(sf3[8:14])
        if pageID == 1:
            #=== Page 1: UTC and ionosphere ==================================
            self.PageID1 = 1
            # UTC bias, drift and drift-rate coefficients.
            self.A0_n = s(sf3[14:30]) * 2**-35
            self.A1_n = s(sf3[30:43]) * 2**-51
            self.A2_n = s(sf3[43:50]) * 2**-68
            # Current and future leap-second parameters.
            self.delta_t_LS = s(sf3[50:58])
            self.t_ot = u(sf3[58:74]) * 16
            self.WN_ot = u(sf3[74:87])
            self.WN_LSF = u(sf3[87:100])
            self.DN = u(sf3[100:104])
            self.delta_t_LSF = s(sf3[104:112])
            # Klobuchar ionospheric-model alpha parameters.
            self.alpha0 = s(sf3[112:120]) * 2**-30
            self.alpha1 = s(sf3[120:128]) * 2**-27
            self.alpha2 = s(sf3[128:136]) * 2**-24
            self.alpha3 = s(sf3[136:144]) * 2**-24
            # Klobuchar ionospheric-model beta parameters.
            self.beta0 = s(sf3[144:152]) * 2**11
            self.beta1 = s(sf3[152:160]) * 2**14
            self.beta2 = s(sf3[160:168]) * 2**16
            self.beta3 = s(sf3[168:176]) * 2**16
            # Inter-signal corrections for L1 C/A, L2C, L5I5 and L5Q5.
            self.ISC_L1CA = s(sf3[176:189]) * 2**-35
            self.ISC_L2C = s(sf3[189:202]) * 2**-35
            self.ISC_L5I5 = s(sf3[202:215]) * 2**-35
            self.ISC_L5Q5 = s(sf3[215:228]) * 2**-35
        elif pageID == 2:
            #=== Page 2: GPS/GNSS time offset and Earth orientation ==========
            self.PageID2 = 2
            # GPS/GNSS time-offset parameters.
            self.GGTO = {
                "GNSS_ID": u(sf3[14:17]),
                "t_GGTO": u(sf3[17:33]) * 16,
                "WN_GGTO": u(sf3[33:46]),
                "A0_GGTO": s(sf3[46:62]) * 2**-35,
                "A1_GGTO": s(sf3[62:75]) * 2**-51,
                "A2_GGTO": s(sf3[75:82]) * 2**-68,
            }
            # Earth-orientation parameters.
            self.EOP = {
                "t_EOP": u(sf3[82:98]) * 16,
                "PM_X": s(sf3[98:119]) * 2**-20,
                "PM_X_dot": s(sf3[119:134]) * 2**-21,
                "PM_Y": s(sf3[134:155]) * 2**-20,
                "PM_Y_dot": s(sf3[155:170]) * 2**-21,
                "Delta_UTGPS": s(sf3[170:201]) * 2**-24,
                "Delta_UTGPS_dot": s(sf3[201:220]) * 2**-25,
            }

        #%% Compute TOW of the decoded frame
        # Combine the two-hour interval and the 18-second TOI epoch.
        self.TOW = self.ITOW * 7200 + TOI * 18
        # Mark the ephemeris as successfully decoded.
        self.flag = 1

    @staticmethod
    def _deinterleave(interleaved):
        """Undo the CNAV-2 38-by-46 block interleaver.

        Args
        ----
            interleaved - array-like
                        Interleaved CNAV-2 subframe-2 and subframe-3 symbols.
        Returns
        -------
            deinterleaved - numpy.ndarray
                          Symbols restored to their LDPC codeword order.
        """
        # Restore the transmitted 38-by-46 matrix and read it in the inverse
        # column order used by the CNAV-2 block interleaver.
        matrix = np.reshape(interleaved, (38, 46), order="F").T
        return matrix.reshape(-1, order="F")

    @classmethod
    def NAVdecoding(cls, trackResult, settings):
        """Decode GPS L1C CNAV-2 messages from prompt correlators.

        Pilot-overlay-code correlation supplies frame synchronization. The
        52-symbol first subframe is BCH decoded; the remaining symbols are
        deinterleaved and decoded by the H2 and H3 LDPC decoders.

        Args
        ----
            trackResult    - object
                           Tracking results for one satellite channel.
            settings       - object
                           Receiver settings.

        Returns
        -------
            eph            - Ephemeris
                           Decoded broadcast ephemeris.
            firstSubFrame  - int or float
                           Zero-based frame-start index in 10 ms tracking
                           epochs, or infinity when no valid frame is found.
            TOW            - float
                           TOW preceding the decoded frame, in seconds.
        """
        #--- Initialize the ephemeris structure -------------------------------
        # Use a common structure even when none of the requisite messages is
        # decoded for this satellite.
        eph = cls()
        # Starting position and TOW of the first valid navigation frame.
        firstSubFrame = np.inf
        TOW = np.inf
        # One CNAV-2 frame comprises 1800 symbols; the BCH-protected TOI
        # subframe comprises the first 52 symbols.
        frameLength = 1800
        toiLength = 52
        # Obtain the parity-check matrices for subframes 2 and 3.
        H2, H3 = LDPCDecoder.getH_information()

        #%% Frame synchronization using the pilot-channel overlay code
        # Select the prompt pilot component used by the tracking configuration.
        syncSoft = (trackResult.Pilot_I_P if settings.pilotTRKflag
                    else trackResult.Pilot_Q_P)
        # Threshold the prompt output and convert it to -1 and +1.
        syncBits = np.where(syncSoft > 0, 1, -1)
        # Generate the satellite-specific L1C pilot overlay code.
        secondary = generate2ndCode(trackResult.PRN)
        if syncBits.size < secondary.size:
            return eph, firstSubFrame, TOW

        # Correlate the tracking outputs with the overlay code and retain
        # candidate frame boundaries close to the ideal correlation peak.
        xcorrResult = np.correlate(syncBits, secondary, mode="valid")
        frameStarts = np.flatnonzero(np.abs(xcorrResult) >= 1750)
        # Read the data-channel prompt correlator outputs.
        dataSoftI = trackResult.I_P
        dataSoftQ = trackResult.Q_P

        #%% Decode the first valid CNAV-2 frame
        for frameStart in frameStarts:
            # Ensure that one complete frame remains in both prompt streams.
            frameEnd = frameStart + frameLength
            if frameEnd > min(dataSoftI.size, dataSoftQ.size):
                continue

            # Copy the current frame and make hard decisions for BCH decoding.
            frameSoftI = dataSoftI[frameStart:frameEnd].copy()
            frameSoftQ = dataSoftQ[frameStart:frameEnd].copy()
            hardBits = (frameSoftI > 0).astype(np.int8)

            #=== Decode subframe 1 with BCH(51,8) =============================
            # Retry the candidate with reversed polarity when the initial BCH
            # check fails.
            flag, decodedToi = BCHDecoder.BCH51_8Decoding(
                1 - 2 * hardBits[:toiLength])
            if not flag:
                hardBits = 1 - hardBits
                flag, decodedToi = BCHDecoder.BCH51_8Decoding(
                    1 - 2 * hardBits[:toiLength])
                if not flag:
                    continue
                frameSoftI *= -1
                frameSoftQ *= -1

            #=== Undo the subframe-2/3 block interleaver ======================
            deinterleavedI = cls._deinterleave(frameSoftI[toiLength:])
            deinterleavedQ = cls._deinterleave(frameSoftQ[toiLength:])
            # Divide the restored stream into the H2 and H3 LDPC codewords.
            sf2I, sf3I = deinterleavedI[:1200], deinterleavedI[1200:]
            sf2Q, sf3Q = deinterleavedQ[:1200], deinterleavedQ[1200:]

            #=== Decode subframes 2 and 3 with binary LDPC ====================
            # Decode subframe 2. If necessary, retry using the opposite signal
            # polarity and apply the same polarity to subframe 3.
            sf2, sf2Iter, sf2Error = LDPCDecoder.decode(
                sf2I, sf2Q, H2, settings)
            if sf2Error:
                sf2Neg, sf2IterNeg, sf2ErrorNeg = LDPCDecoder.decode(
                    -sf2I, -sf2Q, H2, settings)
                if not sf2ErrorNeg:
                    decodedToi[0] = 1 - decodedToi[0]
                    sf2, sf2Iter, sf2Error = sf2Neg, sf2IterNeg, False
                    sf3I, sf3Q = -sf3I, -sf3Q

            # Decode subframe 3 and reject the frame when either LDPC check
            # remains unsuccessful.
            sf3, sf3Iter, sf3Error = LDPCDecoder.decode(
                sf3I, sf3Q, H3, settings)
            if sf2Error or sf3Error:
                continue

            # Assemble TOI, subframe 2 and subframe 3 in the 883-bit layout
            # required by the ephemeris decoder.
            decodedNavBits = np.r_[decodedToi, sf2[:600], sf3[:274]]
            eph.decodeEphemeris(decodedNavBits.astype(np.int8))
            eph.PRN = trackResult.PRN
            # Save only the first valid frame and its preceding TOW.
            firstSubFrame = int(frameStart)
            TOW = eph.TOW - 18
            if settings.plotNavigation:
                print(f" PRN {trackResult.PRN:02d} CNAV-2 LDPC decoded: "
                      f"SF2 iter={sf2Iter}, SF3 iter={sf3Iter}")
            break
        return eph, firstSubFrame, TOW

    def satpos(self, transmitTime):
        """Calculate satellite X, Y, and Z coordinates at transmit time.

        The coordinates are calculated from the GPS L1C CNAV-2 ephemeris
        stored in this object.

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
        #%% Initialize constants
        # Pi used in the GPS coordinate system.
        gpsPi = 3.1415926535898
        OmegaE = 7.2921151467e-5       # Earth rotation rate [rad/s].
        mu = 3.986005e14               # Earth gravitational constant [m^3/s^2].
        F = -4.442807633e-10           # Relativistic constant [s/sqrt(m)].
        # Reference semi-major axis and right-ascension rate for GPS L1C.
        A_ref_L1C = 26559710
        Omega_dot_ref = -2.6e-9 * gpsPi

        #%% Find initial satellite clock correction
        # CNAV-2 uses the ephemeris epoch as the clock-reference epoch.
        #--- Find time difference ---------------------------------------------
        # Account for beginning- or end-of-week crossover.
        dt = self.check_t(transmitTime - self.t_oe)
        #--- Calculate clock correction ---------------------------------------
        # Calculate clock correction including the group-delay and pilot
        # inter-signal corrections. The pilot correction ISC_L1Cp is used;
        # data-component tracking would use ISC_L1Cd. Relativity is added
        # after the eccentric anomaly is known.
        satClkCorr = ((self.a_2 * dt + self.a_1) * dt + self.a_0
                      - self.T_GD + self.ISC_L1Cp)
        # Correct the transmission time using the initial clock correction.
        time = transmitTime - satClkCorr

        #%% Compute the satellite position
        # Time from the ephemeris reference epoch.
        tk = self.check_t(time - self.t_oe)
        #--- Semi-major axis --------------------------------------------------
        # Semi-major axis at the reference time and at transmission time.
        A0 = A_ref_L1C + self.deltaA
        A = A0 + self.ADot * tk
        #--- Mean motion ------------------------------------------------------
        # Initial and corrected mean motion.
        n0 = np.sqrt(mu / A0**3)
        n = n0 + self.delta_n_0 + 0.5 * self.delta_n_0Dot * tk
        #--- Mean anomaly -----------------------------------------------------
        # Mean anomaly reduced to one revolution.
        M = np.remainder(self.M_0 + n * tk + 2 * gpsPi, 2 * gpsPi)

        #--- Eccentric anomaly by iteration -----------------------------------
        # Use the mean anomaly as the initial estimate.
        E = M
        for _ in range(10):
            E_old = E
            E = M + self.e * np.sin(E)
            if abs(np.fmod(E - E_old, 2 * gpsPi)) < 1e-12:
                # The requested precision has been reached.
                break
        # Reduce the eccentric anomaly to one revolution.
        E = np.remainder(E + 2 * gpsPi, 2 * gpsPi)
        #--- Relativistic correction term -------------------------------------
        # Relativistic correction term.
        dtr = F * self.e * np.sqrt(A) * np.sin(E)
        #--- True anomaly -----------------------------------------------------
        # True anomaly.
        nu = np.arctan2(np.sqrt(1 - self.e**2) * np.sin(E),
                        np.cos(E) - self.e)
        #--- Argument of latitude ---------------------------------------------
        # Argument of latitude before harmonic corrections.
        Phi = np.remainder(nu + self.omega, 2 * gpsPi)
        #--- Second harmonic perturbations ------------------------------------
        sin2Phi = np.sin(2 * Phi)
        cos2Phi = np.cos(2 * Phi)
        # Correct the argument of latitude, orbit radius and inclination.
        u = Phi + self.C_uc * cos2Phi + self.C_us * sin2Phi
        r = (A * (1 - self.e * np.cos(E)) + self.C_rc * cos2Phi
             + self.C_rs * sin2Phi)
        i = (self.i_0 + self.IDOT * tk + self.C_ic * cos2Phi
             + self.C_is * sin2Phi)
        #--- Longitude of the ascending node ----------------------------------
        # Correct the longitude of the ascending node.
        OmegaDotCorrected = Omega_dot_ref + self.omegaDot
        Omega = np.remainder(
            self.omega_0 + (OmegaDotCorrected - OmegaE) * tk
            - OmegaE * self.t_oe + 2 * gpsPi, 2 * gpsPi)
        # Satellite position in the orbital plane.
        xp = r * np.cos(u)
        yp = r * np.sin(u)
        #--- Compute satellite coordinates in ECEF ----------------------------
        satPosition = np.array([
            xp * np.cos(Omega) - yp * np.cos(i) * np.sin(Omega),
            xp * np.sin(Omega) + yp * np.cos(i) * np.cos(Omega),
            yp * np.sin(i),
        ])

        #%% Final satellite-clock correction
        # Include the relativistic correction.
        satClkCorr = ((self.a_2 * dt + self.a_1) * dt + self.a_0
                      - self.T_GD + self.ISC_L1Cp + dtr)
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
        # Satellite coordinates require complete CNAV-2 ephemeris and clock
        # data. Tracking starts at an arbitrary point in the message, so at
        # least 36 seconds of tracking output are required.
        if settings.msToProcess < 36000:
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid CNAV-2 frame in the 10 ms L1C
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
            print(f"Decoding CNAV-2 for PRN {PRN:02d} -----------------")

            currentEph, subFrameStart[channelNr], TOW[channelNr] = (
                Ephemeris.NAVdecoding(trackResults[channelNr], settings))
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all requisite CNAV-2 messages from
            # subsequent positioning.
            if currentEph.flag != 1:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
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
