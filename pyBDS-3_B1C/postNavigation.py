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
BDS-3 B1C B-CNAV1 navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
                       leastSquarePos, skyPlot, twosComp2dec)
from channelDecoder import LDPCDecoder
from correlator import generate2ndCode

#%% BDS-3 B1C broadcast ephemeris
class Ephemeris:
    """Keep and decode one BDS-3 B1C satellite ephemeris.

    A common set of attributes is created for every satellite so that all
    ephemeris objects have the same structure even when only one, or none,
    of the required B-CNAV1 messages has been decoded for a given PRN.
    """

    def __init__(self):
        """Initialize a common B-CNAV1 ephemeris structure.

        Every field is initialized so that ephemerides from different
        satellites retain the same attribute layout while navigation pages
        are accumulated over time.

        Returns
        -------
            None
                All broadcast-ephemeris, clock, integrity and time-system
                fields are initialized in this object.
        """
        #%% Decode the first subframe =========================================
        # Satellite PRN number.
        self.PRN = None
        # Seconds of hour.
        self.SOH = None

        #%% Decode the second subframe ========================================
        # Week number.
        self.WN = None
        # Hours of week.
        self.HOW = None
        # Issue of data, clock.
        self.IODC = None
        # Issue of data, ephemeris.
        self.IODE = None

        #--- Ephemeris I ------------------------------------------------------
        # Ephemeris-data reference time of week.
        self.t_oe = None
        # Satellite type: GEO, IGSO, or MEO.
        self.SatType = None
        # Semi-major-axis difference at the reference time.
        self.deltaA = None
        # Change rate of the semi-major axis.
        self.ADot = None
        # Mean-motion difference from the computed value at reference time.
        self.delta_n_0 = None
        # Rate of the mean-motion difference from the computed value.
        self.delta_n_0Dot = None
        # Mean anomaly at reference time.
        self.M_0 = None
        # Eccentricity.
        self.e = None
        # Argument of perigee.
        self.omega = None

        #--- Ephemeris II -----------------------------------------------------
        # Longitude of the ascending node at the weekly epoch.
        self.omega_0 = None
        # Inclination angle at reference time.
        self.i_0 = None
        # Rate of the right-ascension difference.
        self.omegaDot = None
        # Rate of the inclination angle.
        self.i_0Dot = None
        # Sine harmonic correction to the angle of inclination.
        self.C_is = None
        # Cosine harmonic correction to the angle of inclination.
        self.C_ic = None
        # Sine correction to the orbit radius.
        self.C_rs = None
        # Cosine correction to the orbit radius.
        self.C_rc = None
        # Sine harmonic correction to the argument of latitude.
        self.C_us = None
        # Cosine harmonic correction to the argument of latitude.
        self.C_uc = None

        #--- Satellite clock-error parameters --------------------------------
        # Clock-data reference time of week.
        self.t_oc = None
        # Satellite-clock bias correction coefficient.
        self.a_0 = None
        # Satellite-clock drift correction coefficient.
        self.a_1 = None
        # Satellite-clock drift-rate correction coefficient.
        self.a_2 = None

        #--- Remaining fields of the second subframe -------------------------
        # Group-delay differential of the B2a pilot component.
        self.T_GDB2ap = None
        # Group-delay differential between B1C data and pilot components.
        self.ISC_B1Cd = None
        # Group-delay differential of the B1C pilot component.
        self.T_GDB1Cp = None

        #%% Decode the third subframe =========================================
        # Page identifiers decoded from subframe 3.
        self.PageID1 = None
        self.PageID3 = None
        # Satellite health state.
        self.HS = None
        # Data integrity flag.
        self.DIF = None
        # Signal integrity flag.
        self.SIF = None
        # Accuracy integrity flag.
        self.AIF = None
        # Signal-in-space monitoring accuracy index.
        self.SISMAI = None

        #--- Ionospheric parameters ------------------------------------------
        self.alpha1 = None
        self.alpha2 = None
        self.alpha3 = None
        self.alpha4 = None
        self.alpha5 = None
        self.alpha6 = None
        self.alpha7 = None
        self.alpha8 = None
        self.alpha9 = None

        #--- BDT-UTC parameters ----------------------------------------------
        # Bias, drift, and drift-rate coefficients of BDT relative to UTC.
        self.A_0UTC = None
        self.A_1UTC = None
        self.A_2UTC = None
        # Current or past leap-second count.
        self.delta_t_LS = None
        # Reference time of week and reference week number.
        self.t_ot = None
        self.WN_ot = None
        # Leap-second reference week and day numbers.
        self.WN_LSF = None
        self.DN = None
        # Current or future leap-second count.
        self.delta_t_LSF = None

        #--- BDT-GNSS time offset (BGTO) -------------------------------------
        # GNSS type identification.
        self.GNSS_ID = None
        # Reference week and reference time of week.
        self.WN_0BGTO = None
        self.t_0BGTO = None
        # Bias, drift, and drift-rate coefficients of BDT relative to GNSS.
        self.A_0BGTO = None
        self.A_1BGTO = None
        self.A_2BGTO = None

        # All required B-CNAV1 subframe messages have been decoded.
        self.flag = None
        # Time of week.
        self.TOW = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary character sequence to an unsigned integer.

        Args
        ----
            bits        - sequence of str
                        Binary characters containing only ``'0'`` or
                        ``'1'``.
        Returns
        -------
            decimal     - int
                        Unsigned decimal value represented by ``bits``.
        """
        return int("".join(bits), 2)

    @staticmethod
    def check_t(time):
        """Account for beginning or end of a BDT week crossover.

        Args
        ----
            time        - float
                        Time difference in seconds.
        Returns
        -------
            corrTime    - float
                        Corrected time difference in seconds.
        """
        # Half of one BDT week, in seconds.
        half_week = 302400
        if time > half_week:
            return time-2*half_week
        if time < -half_week:
            return time+2*half_week
        return time

    @staticmethod
    def _crc24qCheck(bits):
        """Check a complete navigation message with its CRC-24Q parity.

        Args
        ----
            bits        - ndarray
                        Complete binary message including the transmitted
                        24-bit CRC field.
        Returns
        -------
            valid       - bool
                        True when the CRC-24Q remainder is zero.
        """
        # CRC-24Q uses the lower 24 coefficients of its generator polynomial.
        crc = 0
        polynomial = 0x864CFB
        # Shift the complete message through the 24-bit feedback register.
        for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
            feedback = ((crc >> 23) & 1) ^ int(bit)
            crc = (crc << 1) & 0xFFFFFF
            if feedback:
                crc ^= polynomial
        return crc == 0

    def decodeEphemeris(self, navBitsBin):
        """Decode ephemeris and TOW from a B-CNAV1 bit stream.

        The input must contain one 878-bit B-CNAV1 frame beginning with
        subframe 1. BCH, LDPC, and CRC checks are assumed to have completed
        before this method is called. The ephemeris is decoded message by
        message into this object so that fields from earlier pages are not
        lost.

        Args
        ----
            navBitsBin  - str
                        Navigation-message bits containing only ``'0'`` or
                        ``'1'`` characters.
        Returns
        -------
            None
                Time of week is stored in ``TOW`` and the satellite
                ephemeris is stored in this object.

        Notes
        -----
        For further details on message contents, refer to the BDS-SIS-ICD-
        B1C specification.
        """
        #%% Check that the parameters are strings ============================
        if not isinstance(navBitsBin, str):
            raise TypeError("The parameter BITS must be a character array!")
        if len(navBitsBin) < 878:
            raise ValueError("The parameter BITS must contain 878 bits!")

        # Pi used by the BDS broadcast equations.
        bdsPi = 3.1415926535898

        #%% Decode the first subframe ========================================
        # Satellite PRN number.
        self.PRN = self.bin2dec(navBitsBin[0:6])
        if self.PRN < 1 or self.PRN > 63:
            return

        firstMessage = self.flag is None
        if firstMessage:
            # Seconds of hour.
            self.SOH = self.bin2dec(navBitsBin[6:14])*18

            #%% Decode the second subframe ===================================
            subFra2Bit = navBitsBin[14:53]
            # Week number.
            self.WN = self.bin2dec(subFra2Bit[0:13])
            # Hours of week.
            self.HOW = self.bin2dec(subFra2Bit[13:21])
            # Issue of data, clock.
            self.IODC = self.bin2dec(subFra2Bit[21:31])
            # Issue of data, ephemeris.
            self.IODE = self.bin2dec(subFra2Bit[31:39])

            #--- Ephemeris I --------------------------------------------------
            subFra2Bit = navBitsBin[53:256]
            # Ephemeris-data reference time of week.
            self.t_oe = self.bin2dec(subFra2Bit[0:11])*300
            # Satellite type.
            SatType = self.bin2dec(subFra2Bit[11:13])
            self.SatType = {1: "GEO", 2: "IGSO", 3: "MEO"}.get(SatType)
            # Semi-major-axis difference at the reference time.
            self.deltaA = twosComp2dec(subFra2Bit[13:39])*2**-9
            # Change rate of the semi-major axis.
            self.ADot = twosComp2dec(subFra2Bit[39:64])*2**-21
            # Mean-motion difference from the computed value at reference time.
            self.delta_n_0 = twosComp2dec(subFra2Bit[64:81])*2**-44*bdsPi
            # Rate of the mean-motion difference from the computed value.
            self.delta_n_0Dot = twosComp2dec(subFra2Bit[81:104])*2**-57*bdsPi
            # Mean anomaly at reference time.
            self.M_0 = (twosComp2dec(subFra2Bit[104:137])*2**-32*bdsPi)
            # Eccentricity.
            self.e = self.bin2dec(subFra2Bit[137:170])*2**-34
            # Argument of perigee.
            self.omega = (twosComp2dec(subFra2Bit[170:203])*2**-32*bdsPi)

            #--- Ephemeris II -------------------------------------------------
            subFra2Bit = navBitsBin[256:478]
            # Longitude of the ascending node at the weekly epoch.
            self.omega_0 = twosComp2dec(subFra2Bit[0:33])*2**-32*bdsPi
            # Inclination angle at reference time.
            self.i_0 = twosComp2dec(subFra2Bit[33:66])*2**-32*bdsPi
            # Rate of the right-ascension difference.
            self.omegaDot = twosComp2dec(subFra2Bit[66:85])*2**-44*bdsPi
            # Rate of the inclination angle.
            self.i_0Dot = twosComp2dec(subFra2Bit[85:100])*2**-44*bdsPi
            # Sine harmonic correction to the angle of inclination.
            self.C_is = twosComp2dec(subFra2Bit[100:116])*2**-30
            # Cosine harmonic correction to the angle of inclination.
            self.C_ic = twosComp2dec(subFra2Bit[116:132])*2**-30
            # Sine correction to the orbit radius.
            self.C_rs = twosComp2dec(subFra2Bit[132:156])*2**-8
            # Cosine correction to the orbit radius.
            self.C_rc = twosComp2dec(subFra2Bit[156:180])*2**-8
            # Sine harmonic correction to the argument of latitude.
            self.C_us = twosComp2dec(subFra2Bit[180:201])*2**-30
            # Cosine harmonic correction to the argument of latitude.
            self.C_uc = twosComp2dec(subFra2Bit[201:222])*2**-30

            #--- Satellite clock-error parameters ----------------------------
            subFra2Bit = navBitsBin[478:547]
            # Clock-data reference time of week.
            self.t_oc = self.bin2dec(subFra2Bit[0:11])*300
            # Satellite-clock bias correction coefficient.
            self.a_0 = twosComp2dec(subFra2Bit[11:36])*2**-34
            # Satellite-clock drift correction coefficient.
            self.a_1 = twosComp2dec(subFra2Bit[36:58])*2**-50
            # Satellite-clock drift-rate correction coefficient.
            self.a_2 = twosComp2dec(subFra2Bit[58:69])*2**-66

            #--- Remaining fields of the second subframe ---------------------
            subFra2Bit = navBitsBin[547:583]
            # Group-delay differential of the B2a pilot component.
            self.T_GDB2ap = twosComp2dec(subFra2Bit[0:12])*2**-34
            # Group delay between the B1C data and pilot components.
            self.ISC_B1Cd = twosComp2dec(subFra2Bit[12:24])*2**-34
            # Group-delay differential of the B1C pilot component.
            self.T_GDB1Cp = twosComp2dec(subFra2Bit[24:36])*2**-34

        #%% Decode the third subframe ========================================
        subFra2Bit = navBitsBin[614:878]
        # Page identifier.
        PageID = self.bin2dec(subFra2Bit[0:6])
        # Satellite health state.
        self.HS = self.bin2dec(subFra2Bit[6:8])
        # Data integrity flag.
        self.DIF = self.bin2dec(subFra2Bit[8:9])
        # Signal integrity flag.
        self.SIF = self.bin2dec(subFra2Bit[9:10])
        # Accuracy integrity flag.
        self.AIF = self.bin2dec(subFra2Bit[10:11])
        # Signal-in-space monitoring accuracy index.
        self.SISMAI = self.bin2dec(subFra2Bit[11:15])

        # Decode the remaining page-dependent fields of subframe 3.
        if PageID == 1:
            self.PageID1 = 1
            #--- Ionospheric parameters --------------------------------------
            tempBit = subFra2Bit[42:116]
            self.alpha1 = self.bin2dec(tempBit[0:10])*2**-3
            self.alpha2 = twosComp2dec(tempBit[10:18])*2**-3
            self.alpha3 = self.bin2dec(tempBit[18:26])*2**-3
            self.alpha4 = self.bin2dec(tempBit[26:34])*2**-3
            self.alpha5 = self.bin2dec(tempBit[34:42])*2**-3
            self.alpha6 = twosComp2dec(tempBit[42:50])*2**-3
            self.alpha7 = twosComp2dec(tempBit[50:58])*2**-3
            self.alpha8 = twosComp2dec(tempBit[58:66])*2**-3
            self.alpha9 = twosComp2dec(tempBit[66:74])*2**-3

            #--- BDT-UTC parameters ------------------------------------------
            tempBit = subFra2Bit[116:213]
            # Bias coefficient of the BDT time scale relative to UTC.
            self.A_0UTC = twosComp2dec(tempBit[0:16])*2**-35
            # Drift coefficient of the BDT time scale relative to UTC.
            self.A_1UTC = twosComp2dec(tempBit[16:29])*2**-51
            # Drift-rate coefficient of the BDT time scale relative to UTC.
            self.A_2UTC = twosComp2dec(tempBit[29:36])*2**-68
            # Current or past leap-second count.
            self.delta_t_LS = twosComp2dec(tempBit[36:44])
            # Reference time of week.
            self.t_ot = self.bin2dec(tempBit[44:60])*2**4
            # Reference week number.
            self.WN_ot = self.bin2dec(tempBit[60:73])
            # Leap-second reference week number.
            self.WN_LSF = self.bin2dec(tempBit[73:86])
            # Leap-second reference day number.
            self.DN = self.bin2dec(tempBit[86:89])
            # Current or future leap-second count.
            self.delta_t_LSF = twosComp2dec(tempBit[89:97])
        elif PageID == 3:
            self.PageID3 = 3
            #--- BDT-GNSS time offset (BGTO) ---------------------------------
            tempBit = subFra2Bit[158:226]
            # GNSS type identification.
            self.GNSS_ID = self.bin2dec(tempBit[0:3])
            # Reference week number.
            self.WN_0BGTO = self.bin2dec(tempBit[3:16])
            # Reference time of week.
            self.t_0BGTO = self.bin2dec(tempBit[16:32])*2**4
            # Bias coefficient of the BDT time scale relative to GNSS.
            self.A_0BGTO = twosComp2dec(tempBit[32:48])*2**-35
            # Drift coefficient of the BDT time scale relative to GNSS.
            self.A_1BGTO = twosComp2dec(tempBit[48:61])*2**-51
            # Drift-rate coefficient of the BDT time scale relative to GNSS.
            self.A_2BGTO = twosComp2dec(tempBit[61:68])*2**-68

        if firstMessage:
            # Time of week of the first decoded frame.
            self.TOW = self.HOW*3600+self.SOH
        # All required navigation data in this frame have been decoded.
        self.flag = 1

    @classmethod
    def NAVdecoding(cls, trackResult, settings):
        """Decode the BDS-3 B1C B-CNAV1 navigation message.

        Frame synchronization uses the pilot secondary code. One complete
        1800-symbol frame is extracted, decoded by the BCH and LDPC/EMS
        decoders, checked by CRC-24Q for SF2 and SF3, and finally converted
        into broadcast ephemeris fields.

        Args
        ----
            trackResult   - tracking.Channel
                          Tracking results for one BDS-3 B1C channel.
            settings      - object
                          Receiver settings.
        Returns
        -------
            eph           - Ephemeris
                          Decoded satellite ephemeris.
            firstSubFrame - float
                          Index of the first valid B-CNAV1 frame in the
                          tracking prompt stream. Infinity indicates that no
                          valid frame was found.
            TOW           - float
                          Time of week of the first valid frame, in seconds.
        """
        #--- Initialize the ephemeris structure ------------------------------
        eph = cls()
        firstSubFrame = np.inf
        TOW = np.inf

        #%% Frame synchronization using the pilot secondary code =============
        if settings.pilotACQflag == 1:
            syncBits_soft = trackResult.Pilot_I_P
        else:
            # In narrowband tracking the pilot phase is pi/2 ahead, so its
            # prompt power is on the Q branch.
            syncBits_soft = trackResult.Pilot_Q_P
        # Hard decisions are used only for frame synchronization.
        syncBits_hard = np.where(
            np.asarray(syncBits_soft).reshape(-1) > 0, 1, -1)
        # Generate the B1C pilot secondary code for this PRN.
        Secondary = generate2ndCode(trackResult.PRN)

        # Correlate tracking output with the pilot secondary code.
        XcorrResult = np.correlate(syncBits_hard, Secondary, mode="full")
        # SciPy's zero-lag element is len(Secondary)-1. Retaining zero and
        # positive lags matches the MATLAB xcorr slice.
        XcorrResult = XcorrResult[Secondary.size-1:]
        # Each complete B-CNAV1 frame contains 1800 symbols.
        index = np.flatnonzero(np.abs(XcorrResult) >= 1799.5)

        #%% Decode B1C data and extract ephemeris =============================
        raw_soft_bits_I = np.asarray(trackResult.I_P).reshape(-1)
        # The Q branch supplies noise/reference information for LLR creation.
        raw_soft_bits_Q = np.asarray(
            getattr(trackResult, "Q_P", np.zeros_like(raw_soft_bits_I))
        ).reshape(-1)
        # Use equal-length I and Q branches.
        minLen = min(raw_soft_bits_I.size, raw_soft_bits_Q.size)
        raw_soft_bits_I = raw_soft_bits_I[:minLen]
        raw_soft_bits_Q = raw_soft_bits_Q[:minLen]

        for candidate in index:
            # Ensure that one complete B-CNAV1 frame remains available.
            if raw_soft_bits_I.size-candidate < 1800:
                continue
            #--- Extract one complete B-CNAV1 frame ---------------------------
            navBits_soft_I = raw_soft_bits_I[candidate:candidate+1800]
            navBits_soft_Q = raw_soft_bits_Q[candidate:candidate+1800]
            #--- Run the B1C BCH and LDPC/EMS decoders ------------------------
            decodedNavBits = LDPCDecoder.B1CLDPCDecoder(
                navBits_soft_I, navBits_soft_Q)
            # An unsuccessful BCH decode returns no decoded frame.
            if decodedNavBits is None:
                continue

            #--- Check SF2 and SF3 with CRC-24Q -------------------------------
            decodedNav_SF2 = decodedNavBits[14:614]
            decodedNav_SF3 = decodedNavBits[614:878]
            if (cls._crc24qCheck(decodedNav_SF2)
                    and cls._crc24qCheck(decodedNav_SF3)):
                #--- Decode the ephemeris -------------------------------------
                navBitsChar = "".join(decodedNavBits.astype(str))
                eph.decodeEphemeris(navBitsChar)
                if np.isinf(TOW):
                    TOW = eph.TOW
                    firstSubFrame = candidate
        return eph, firstSubFrame, TOW

    def satpos(self, transmitTime):
        """Calculate BDS satellite ECEF position at transmission time.

        Args
        ----
            transmitTime  - float
                          Signal transmission time in seconds.
        Returns
        -------
            satPosition   - ndarray
                          Satellite position in the ECEF system,
                          ``[X, Y, Z]`` in metres.
            satClkCorr    - float
                          Satellite-clock correction in seconds.
        """
        #%% Initialize constants =============================================
        # Pi used in the BDS coordinate equations.
        bdsPi = 3.1415926535898
        # Earth rotation rate [rad/s].
        OmegaE = 7.2921150e-5
        # Earth's universal gravitational parameter [m^3/s^2].
        mu = 3.986004418e14
        # Relativistic correction constant [s/m^(1/2)].
        F = -4.44280730904398e-10
        # Semi-major-axis reference for MEO satellites [m].
        A_ref_MEO = 27906100
        # Semi-major-axis reference for IGSO and GEO satellites [m].
        A_ref_IGSO_GEO = 42162200

        #%% Find the initial satellite-clock correction =======================
        # Find the time difference and account for BDT week crossover.
        dt = self.check_t(transmitTime-self.t_oc)
        # Joint pilot/data tracking is used, so ISC_B1Cd is ignored.
        # Calculate the initial clock correction.
        satClkCorr = (
            (self.a_2*dt+self.a_1)*dt+self.a_0-self.T_GDB1Cp)
        time = transmitTime-satClkCorr

        #%% Find the satellite position =======================================
        # Time from the ephemeris reference epoch.
        tk = self.check_t(time-self.t_oe)

        # Select the reference semi-major axis for the satellite type.
        if self.SatType == "MEO":
            A_ref = A_ref_MEO
        elif self.SatType in ("IGSO", "GEO"):
            A_ref = A_ref_IGSO_GEO

        # Restore the semi-major axis.
        A0 = A_ref+self.deltaA
        A = A0+self.ADot*tk
        # Reference mean motion.
        n0 = np.sqrt(mu/A0**3)
        # Mean-motion difference from the computed value.
        delta_n = self.delta_n_0+0.5*self.delta_n_0Dot*tk
        # Corrected mean motion.
        n = n0+delta_n
        # Mean anomaly, reduced to one revolution.
        M = np.fmod(self.M_0+n*tk+2*bdsPi, 2*bdsPi)

        # Use the mean anomaly as the initial eccentric-anomaly estimate.
        E = M
        #--- Iteratively compute the eccentric anomaly -----------------------
        for _ in range(10):
            E_old = E
            E = M+self.e*np.sin(E)
            dE = np.fmod(E-E_old, 2*bdsPi)
            if abs(dE) < 1e-12:
                # Required precision has been reached.
                break
        # Reduce eccentric anomaly to one revolution.
        E = np.fmod(E+2*bdsPi, 2*bdsPi)

        # Compute the relativistic correction term.
        dtr = F*self.e*np.sqrt(A0)*np.sin(E)
        # Calculate the true anomaly.
        nu = np.arctan2(
            np.sqrt(1-self.e**2)*np.sin(E), np.cos(E)-self.e)
        # Argument of latitude before harmonic correction.
        Phi = np.fmod(nu+self.omega, 2*bdsPi)
        # Correct the argument of latitude.
        u = Phi+self.C_uc*np.cos(2*Phi)+self.C_us*np.sin(2*Phi)
        # Correct the orbit radius.
        r = (A*(1-self.e*np.cos(E))+self.C_rc*np.cos(2*Phi)
             + self.C_rs*np.sin(2*Phi))
        # Correct the inclination angle.
        i = (self.i_0+self.i_0Dot*tk+self.C_ic*np.cos(2*Phi)
             + self.C_is*np.sin(2*Phi))
        # Angle between the ascending node and the Greenwich meridian.
        Omega = (self.omega_0+(self.omegaDot-OmegaE)*tk
                 - OmegaE*self.t_oe)
        # Reduce the angle to one revolution.
        Omega = np.fmod(Omega+2*bdsPi, 2*bdsPi)

        #--- Compute satellite ECEF coordinates ------------------------------
        xp = r*np.cos(u)
        yp = r*np.sin(u)
        satPosition = np.array([xp*np.cos(Omega)-yp*np.cos(i)*np.sin(Omega),
                                xp*np.sin(Omega)+yp*np.cos(i)*np.cos(Omega),
                                yp*np.sin(i)])
        # Include the relativistic term in the final clock correction.
        satClkCorr = (self.a_2*dt+self.a_1)*dt+self.a_0-self.T_GDB1Cp+dtr
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
        # Satellite coordinates require the complete B-CNAV1 ephemeris and
        # clock data. Tracking starts at an arbitrary point in the message,
        # so retain the 36-second minimum record length used by the BDS MATLAB
        # postNavigation implementation.
        if settings.msToProcess < 36000:
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid B-CNAV1 frame in the 10 ms
        # B1C primary-code stream. Infinity means no frame was decoded.
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
            print(f"Decoding BCNAV1 for PRN {PRN:02d} -----------------")

            currentEph, subFrameStart[channelNr], TOW[channelNr] = (
                Ephemeris.NAVdecoding(trackResults[channelNr], settings))
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all requisite B-CNAV1 messages from
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
