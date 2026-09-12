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
Galileo E5a F/NAV-message decoding and position calculation.

"""

import numpy as np

from commUtils import (
    calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
    leastSquarePos, skyPlot, twosComp2dec,
)


#%% Broadcast ephemeris
class Ephemeris:
    """Decode Galileo F/NAV pages and calculate satellite coordinates.

    This class combines the functions of the MATLAB ``NAVdecoding``,
    ``ephemeris``, ``check_t`` and ``satpos`` routines. It stores one
    satellite's broadcast ephemeris, clock parameters and message status.
    """

    def __init__(self):
        """Initialize the common Galileo broadcast-ephemeris structure.

        Every satellite is initialized with the same fields so that partial
        decoding of the requisite F/NAV words still produces a consistent
        ephemeris object, as in MATLAB ``eph_structure_init``.

        Returns
        -------
            None
        """
        # Initialize all broadcast-orbit, clock, ionosphere and time fields.
        self.PRN = 0
        #--- F/NAV Issue of Data values ---------------------------------------
        self.IODnav1 = self.IODnav2 = None
        self.IODnav3 = self.IODnav4 = None
        #--- Ephemeris words 1-3 and GST --------------------------------------
        self.t_oe = self.M_0 = self.e = self.sqrtA = None
        self.Omega_0 = self.i_0 = self.omega = self.iDot = None
        self.OmegaDot = self.deltan = None
        self.CUC = self.CUS = self.CRC = self.CRS = None
        #--- SVID and satellite clock correction -------------------------------
        self.SVID = self.CIC = self.CIS = self.t_oc = None
        self.a_f0 = self.a_f1 = self.a_f2 = None
        #--- Ionosphere, BGD, signal health, data validity, and GST ------------
        self.a_i0 = self.a_i1 = self.a_i2 = None
        self.iono_SF1 = self.iono_SF2 = self.iono_SF3 = None
        self.iono_SF4 = self.iono_SF5 = None
        self.BGD_E1E5a = self.BGD_E1E5b = None
        self.E5a_HS = self.E5b_HS = self.E1b_HS = None
        self.E5a_DVS = self.E5b_DVS = self.E1B_DVS = None
        self.WN = None
        # Infinity denotes that no valid Time Of Week has yet been decoded.
        self.TOW = np.inf
        #--- GST-UTC conversion ------------------------------------------------
        self.A0 = self.A1 = self.delt_LS = None
        self.t_ot = self.WN_ot = self.WN_LSF = None
        self.DN = self.delt_LSF = None
        #--- GPS-GST conversion ------------------------------------------------
        self.A0_G = self.A1_G = self.t_og = self.WN_og = None
        self.flag = 0
        # Indicator set containing the requisite messages already decoded.
        self._wordValid = set()

    @staticmethod
    def _unsigned(bits):
        """Convert a most-significant-bit-first bit vector to an integer.

        Args
        ----
            bits - array_like
                 Binary values ordered from the most significant bit.

        Returns
        -------
            int
                Unsigned integer represented by ``bits``.
        """
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    @staticmethod
    def _viterbiDecode(symbols):
        """Hard-decision Viterbi decode the Galileo rate-1/2 code.

        The convolutional generators are 171 and inverted 133 in octal, as
        specified by the Galileo ICD and used by the MATLAB receiver.

        Args
        ----
            symbols - array_like
                    Hard-decision convolutionally encoded symbols.

        Returns
        -------
            numpy.ndarray
                Decoded binary information bits.
        """
        # Convert the Galileo convolutional-code polynomials to an explicit
        # 64-state trellis. The second branch G2 is inverted (see the ICD).
        symbols = np.asarray(symbols, dtype=np.uint8)
        symbolCnt = symbols.size//2
        pathMetric = np.full(64, np.inf)
        pathMetric[0] = 0
        predecessor = np.empty((symbolCnt, 64), dtype=np.uint8)
        decodedBit = np.empty((symbolCnt, 64), dtype=np.uint8)

        # Update the survivor path metrics for every received symbol pair.
        for symbolNr in range(symbolCnt):
            received = symbols[2*symbolNr:2*symbolNr+2]
            nextMetric = np.full(64, np.inf)
            for state in range(64):
                if not np.isfinite(pathMetric[state]):
                    continue
                for bit in (0, 1):
                    register = (bit << 6) | state
                    nextState = ((bit << 5) | (state >> 1)) & 0x3f
                    output0 = (register & 0o171).bit_count() & 1
                    output1 = ((register & 0o133).bit_count() & 1) ^ 1
                    branchMetric = (int(output0 != received[0])
                                    + int(output1 != received[1]))
                    metric = pathMetric[state]+branchMetric
                    if metric < nextMetric[nextState]:
                        nextMetric[nextState] = metric
                        predecessor[symbolNr, nextState] = state
                        decodedBit[symbolNr, nextState] = bit
            pathMetric = nextMetric

        # Trace back through the minimum-metric survivor path.
        state = int(np.argmin(pathMetric))
        bits = np.empty(symbolCnt, dtype=np.uint8)
        for symbolNr in range(symbolCnt-1, -1, -1):
            bits[symbolNr] = decodedBit[symbolNr, state]
            state = predecessor[symbolNr, state]
        return bits

    @staticmethod
    def _crc24q(bits):
        """Check a complete Galileo page with the CRC-24Q polynomial.

        Args
        ----
            bits - array_like
                 Complete decoded Galileo page, including its 24 CRC bits.

        Returns
        -------
            bool
                ``True`` when the page passes the CRC check.
        """
        # The polynomial matches the MATLAB ``comm.CRCDetector`` definition.
        remainder = np.asarray(bits, dtype=np.uint8).copy()
        polynomial = np.array(
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
        for bitNr in range(remainder.size-24):
            if remainder[bitNr]:
                remainder[bitNr:bitNr+25] ^= polynomial
        return not np.any(remainder[-24:])

    @staticmethod
    def check_t(time):
        """Account for beginning- or end-of-week crossover.

        Args
        ----
            time - float
                 Time difference in seconds.

        Returns
        -------
            float
                Corrected time difference in seconds.
        """
        # Half of one Galileo week [s].
        if time > 302400:
            time -= 604800
        elif time < -302400:
            time += 604800
        return time

    def _decodeFNavWord(self, navWord, pageIndex):
        """Decode one nominal Galileo F/NAV data word.

        Args
        ----
            navWord  - array_like
                     Binary F/NAV page after Viterbi decoding.
            pageIndex - int
                      Zero-based page index used to refer TOW to the first
                      decoded page.

        Returns
        -------
            None
                Decoded fields and word-valid flags are stored in this object.
        """
        #--- Decode the message type ------------------------------------------
        wordType = self._unsigned(navWord[:6])
        # Pi used in the Galileo coordinate system (the same value as GPS).
        galPi = 3.1415926535898

        #--- Decode the page according to its word type -----------------------
        if wordType == 1:
            # SVID, clock correction, ionosphere, BGD, signal health and GST.
            self.SVID = self._unsigned(navWord[6:12])
            self.IODnav1 = self._unsigned(navWord[12:22])
            self.t_oc = self._unsigned(navWord[22:36])*60
            self.a_f0 = twosComp2dec(navWord[36:67])*2**-34
            self.a_f1 = twosComp2dec(navWord[67:88])*2**-46
            self.a_f2 = twosComp2dec(navWord[88:94])*2**-59
            self.a_i0 = self._unsigned(navWord[102:113])*2**-2
            self.a_i1 = twosComp2dec(navWord[113:124])*2**-8
            self.a_i2 = twosComp2dec(navWord[124:138])*2**-15
            self.iono_SF1 = int(navWord[138])
            self.iono_SF2 = int(navWord[139])
            self.iono_SF3 = int(navWord[140])
            self.iono_SF4 = int(navWord[141])
            self.iono_SF5 = int(navWord[142])
            self.BGD_E1E5a = twosComp2dec(navWord[143:153])*2**-32
            self.E5a_HS = self._unsigned(navWord[153:155])
            self.WN = self._unsigned(navWord[155:167])
            self.TOW = self._unsigned(navWord[167:187])-pageIndex*10
            # Correct TOW to the time of the first decoded page.
            self.E5a_DVS = int(navWord[187])
            if self.E5a_HS != 1:
                self._wordValid.add(1)
        elif wordType == 2:
            # Ephemeris (1/3).
            self.IODnav2 = self._unsigned(navWord[6:16])
            self.M_0 = twosComp2dec(navWord[16:48])*2**-31*galPi
            self.OmegaDot = twosComp2dec(navWord[48:72])*2**-43*galPi
            self.e = self._unsigned(navWord[72:104])*2**-33
            self.sqrtA = self._unsigned(navWord[104:136])*2**-19
            self.Omega_0 = twosComp2dec(navWord[136:168])*2**-31*galPi
            self.iDot = twosComp2dec(navWord[168:182])*2**-43*galPi
            self._wordValid.add(2)
        elif wordType == 3:
            # Ephemeris (2/3).
            self.IODnav3 = self._unsigned(navWord[6:16])
            self.i_0 = twosComp2dec(navWord[16:48])*2**-31*galPi
            self.omega = twosComp2dec(navWord[48:80])*2**-31*galPi
            self.deltan = twosComp2dec(navWord[80:96])*2**-43*galPi
            self.CUC = twosComp2dec(navWord[96:112])*2**-29
            self.CUS = twosComp2dec(navWord[112:128])*2**-29
            self.CRC = twosComp2dec(navWord[128:144])*2**-5
            self.CRS = twosComp2dec(navWord[144:160])*2**-5
            self.t_oe = self._unsigned(navWord[160:174])*60
            self._wordValid.add(3)
        elif wordType == 4:
            # Ephemeris (3/3), GST-UTC and GPS-GST conversion parameters.
            self.IODnav4 = self._unsigned(navWord[6:16])
            self.CIC = twosComp2dec(navWord[16:32])*2**-29
            self.CIS = twosComp2dec(navWord[32:48])*2**-29
            self.A0 = twosComp2dec(navWord[48:80])*2**-30
            self.A1 = twosComp2dec(navWord[80:104])*2**-50
            self.delt_LS = twosComp2dec(navWord[104:112])
            self.t_ot = self._unsigned(navWord[112:120])*3600
            self.WN_ot = self._unsigned(navWord[120:128])
            self.WN_LSF = self._unsigned(navWord[128:136])
            self.DN = self._unsigned(navWord[136:139])
            self.delt_LSF = twosComp2dec(navWord[139:147])
            self.t_og = self._unsigned(navWord[147:155])*3600
            self.A0_G = twosComp2dec(navWord[155:171])*2**-35
            self.A1_G = twosComp2dec(navWord[171:183])*2**-51
            self.WN_og = self._unsigned(navWord[183:189])
            self._wordValid.add(4)

    def decodeFNav(self, navBits):
        """Decode the first five complete F/NAV pages in a bit stream.

        Args
        ----
            navBits - array_like
                    Binary F/NAV page symbols beginning at a preamble.

        Returns
        -------
            Ephemeris
                This object with all successfully decoded fields updated.
        """
        #%% Prepare related variables =========================================
        # Check the polarity of the data bits against the F/NAV sync pattern.
        syncBits = np.array(
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        #%% Decode messages ===================================================
        for pageIndex in range(5):
            #--- Pull out one complete page -----------------------------------
            pagePart = np.asarray(
                navBits[500*pageIndex:500*(pageIndex+1)], dtype=np.uint8)
            #--- Correct the polarity according to the preamble bits ----------
            if not np.array_equal(pagePart[:12], syncBits):
                pagePart = 1-pagePart
            #--- De-interleave and remove convolutional encoding --------------
            pageSymbols = np.reshape(np.reshape(pagePart[12:500], (61, 8), order="F").T, -1, order="F")
            page = self._viterbiDecode(pageSymbols)[:238]
            #--- Check the CRC ------------------------------------------------
            if not self._crc24q(page):
                continue
            #--- Decode the F/NAV word ----------------------------------------
            self._decodeFNavWord(page, pageIndex)
            # Stop after all four requisite messages have been decoded and
            # verify that their Issue of Data values are identical.
            if self._wordValid.issuperset(range(1, 5)):
                IODnav = (self.IODnav1, self.IODnav2,
                          self.IODnav3, self.IODnav4)
                self.flag = int(len(set(IODnav)) == 1)
                break
        return self

    @classmethod
    def NAVdecoding(cls, I_P_InputBits):
        """Find valid F/NAV preambles and decode one ephemeris set.

        Preamble candidates are found in the prompt-correlator stream and
        verified by page spacing. The E5a-I secondary code is removed before
        the F/NAV pages are channel decoded.

        Args
        ----
            I_P_InputBits - array_like
                          Prompt-I output from the tracking function.

        Returns
        -------
            tuple
                ``(eph, firstSubFrame, TOW)`` containing the decoded
                ephemeris, the first message position counted in 1 ms tracking
                epochs from the start of tracking, and Time Of Week in seconds.
                The final two values
                are infinity when no valid message is detected.
        """
        # Use the prompt correlator as the symbol stream.
        bits = np.where(np.asarray(I_P_InputBits) > 0, 1, -1).astype(np.int8)
        #--- Generate the synchronization and E5a-I secondary codes -----------
        secondaryCode = np.array([
            -1, 1, 1, 1, 1, -1, 1, 1, 1, 1,
            -1, 1, -1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
        preambleBits = np.array(
            [-1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1], dtype=np.int8)
        # Use int16 so the 240-symbol correlation peak does not overflow.
        preamble = np.kron(preambleBits, secondaryCode).astype(np.int16)
        # Correlate tracking output with the secondary-code-expanded preamble.
        correlation = np.correlate(bits, preamble, mode="full")
        correlation = correlation[preamble.size-1:]
        #%% Find all starting points of preamble-like patterns ================
        index = np.flatnonzero(np.abs(correlation) > 239.99)
        indexSet = set(index.tolist())

        # Analyze each detected preamble-like pattern.
        for startIndex in index:
            # Verify expected page spacing and available message length.
            if (startIndex+10000 not in indexSet
                    or startIndex+50000 > bits.size):
                continue
            # Remove the 20-symbol E5a-I secondary code to recover F/NAV.
            inputBits = bits[startIndex:startIndex+50000]
            navBits = (np.sum(
                inputBits.reshape(2500, 20)*secondaryCode, axis=1) < 0
                ).astype(np.uint8)
            syncBits = np.array(
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
            # Correct the polarity of all data bits from the preamble.
            if not np.array_equal(navBits[:12], syncBits):
                navBits = 1-navBits
            eph = cls().decodeFNav(navBits)
            if np.isfinite(eph.TOW):
                return eph, int(startIndex), float(eph.TOW)
        return cls(), np.inf, np.inf

    def satpos(self, transmitTime):
        """Calculate satellite ECEF coordinates at the transmission time.

        Args
        ----
            transmitTime - float
                         Signal transmission time in seconds.

        Returns
        -------
            tuple
                ``(satPosition, satClkCorr)`` containing the satellite ECEF
                position ``[X, Y, Z]`` in metres and clock correction in
                seconds.
        """
        #%% Initialize constants ==============================================
        # Galileo pi and constants for the satellite-position calculation.
        galPi = 3.1415926535898
        OmegaE = 7.2921151467e-5
        mu = 3.986004418e14
        F = -4.442807309e-10

        #%% Find the initial satellite clock correction -----------------------
        dt = self.check_t(transmitTime-self.t_oc)
        BGD_FIELD = "BGD_E1E5a"
        groupDelay = getattr(self, BGD_FIELD)*(1575.42e6/1176.45e6)**2
        # Calculate clock correction including the E1-E5a group delay.
        satClkCorr = (self.a_f2*dt+self.a_f1)*dt+self.a_f0-groupDelay
        time = transmitTime-satClkCorr
        #%% Find the satellite position ---------------------------------------
        # Restore the semi-major axis and calculate mean motion.
        A = self.sqrtA**2
        n = np.sqrt(mu/A**3)+self.deltan
        tk = self.check_t(time-self.t_oe)
        # Mean anomaly reduced to one revolution.
        M = np.remainder(self.M_0+n*tk+2*galPi, 2*galPi)
        # Initial guess and iterative solution of the eccentric anomaly.
        E = M
        for _ in range(10):
            oldE = E
            E = M+self.e*np.sin(E)
            if abs(E-oldE) < 1e-12:
                break
        E = np.remainder(E+2*galPi, 2*galPi)
        # Relativistic correction and true anomaly.
        dtr = F*self.e*self.sqrtA*np.sin(E)
        nu = np.arctan2(np.sqrt(1-self.e**2)*np.sin(E), np.cos(E)-self.e)
        # Correct argument of latitude, radius, and inclination.
        Phi = np.remainder(nu+self.omega, 2*galPi)
        u = Phi+self.CUC*np.cos(2*Phi)+self.CUS*np.sin(2*Phi)
        r = A*(1-self.e*np.cos(E))+self.CRC*np.cos(2*Phi)+self.CRS*np.sin(2*Phi)
        i = (self.i_0+self.iDot*tk+self.CIC*np.cos(2*Phi)
             + self.CIS*np.sin(2*Phi))
        # Angle between the ascending node and the Greenwich meridian.
        Omega = np.remainder(
            self.Omega_0+(self.OmegaDot-OmegaE)*tk-OmegaE*self.t_oe
            + 2*galPi, 2*galPi)
        #--- Compute satellite coordinates ------------------------------------
        xp = r*np.cos(u)
        yp = r*np.sin(u)
        satPosition = np.array([
            xp*np.cos(Omega)-yp*np.cos(i)*np.sin(Omega),
            xp*np.sin(Omega)+yp*np.cos(i)*np.cos(Omega),
            yp*np.sin(i)])
        #%% Include the relativistic term in the clock correction -------------
        satClkCorr = ((self.a_f2*dt+self.a_f1)*dt+self.a_f0
                      - groupDelay+dtr)
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

        # Measurement location and receiver Galileo System Time (GST).
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
        # A complete Galileo E5a F/NAV ephemeris set requires at least the
        # record length used by the corresponding MATLAB receiver.
        MIN_RECORD_MS = 50000
        if settings.msToProcess < MIN_RECORD_MS:
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
            # Exclude satellites without a complete, consistent F/NAV set.
            if currentEph.flag != 1:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
                activeChnList.remove(channelNr)
            else:
                print(f"    Requisite F/NAV words for PRN {PRN:02d} "
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
