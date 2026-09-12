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
BDS B3I D1/D2 navigation-message decoding and position calculation.

"""

import numpy as np

from commUtils import (calculatePseudoranges, cart2geo, cart2utm, findUtmZone,
                       leastSquarePos, skyPlot, twosComp2dec)

#%% BDS B3I broadcast ephemeris
class Ephemeris:
    """Keep and decode one BDS B3I D1 or D2 broadcast ephemeris.

    A common structure is retained for GEO, IGSO and MEO satellites so that
    navigation processing can use the same interface even when only part of
    the requisite navigation message has been decoded.
    """

    def __init__(self):
        """Initialize the BDS D1/D2 ephemeris structure.

        Returns
        -------
            None
                All clock, ionospheric and orbit fields are initialized so
                that every satellite has the same ephemeris structure before
                navigation-message decoding.
        """
        # Satellite PRN and complete-ephemeris flag.
        self.PRN = None
        self.flag = 0
        # SOW of the first decoded subframe.
        self.SOW = None
        # Satellite health, clock issue of data, user-range accuracy and week.
        self.SatH1 = None
        self.IODC = None
        self.URAI = None
        self.WN = None
        # Clock-data reference time and B3I group-delay correction.
        self.t_oc = None
        self.T_GD_1 = 0.0
        # Ionospheric-delay model parameters.
        self.alpha0 = self.alpha1 = self.alpha2 = self.alpha3 = None
        self.beta0 = self.beta1 = self.beta2 = self.beta3 = None
        # Satellite clock-correction parameters.
        self.a0 = self.a1 = self.a2 = None
        # Issue of data, ephemeris.
        self.IODE = None
        # Broadcast ephemeris parameters.
        self.deltan = None
        self.C_uc = self.C_us = None
        self.M_0 = self.e = self.sqrtA = None
        self.C_ic = self.C_is = None
        self.t_oe = None
        self.i_0 = self.iDot = None
        self.C_rc = self.C_rs = None
        self.omegaDot = self.omega_0 = self.omega = None

    @staticmethod
    def bin2dec(bits):
        """Convert a binary sequence to an unsigned decimal integer.

        Args
        ----
            bits   - str or array_like
                   Binary symbols containing only zero and one.

        Returns
        -------
            int
                Unsigned decimal value of ``bits``; an empty sequence gives
                zero.
        """
        if isinstance(bits, str):
            return int(bits, 2) if bits else 0
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        return int("".join(bits.astype(str)), 2) if bits.size else 0

    @staticmethod
    def bchdec(codeword):
        """Decode one or more BCH(15, 11) codewords.

        The BDS D1 and D2 navigation messages use a binary BCH(15, 11, 1)
        code. The decoder returns the first 11 systematic information bits
        and reports whether the received word contains no more than one bit
        error.

        Args
        ----
            codeword - array-like
                     Fifteen binary symbols per row.
        Returns
        -------
            decoded - ndarray
                    Eleven decoded information bits per row.
            cnumerr - ndarray or int
                    Number of corrected errors; ``-1`` marks an
                    uncorrectable word.
        """
        def remainder(word):
            """Return the BCH polynomial-division remainder.

            Args
            ----
                word   - array_like
                       One 15-bit BCH codeword.

            Returns
            -------
                ndarray
                    Four-bit remainder after division by the generator.
            """
            # Generator polynomial of the shortened BCH(15, 11) code.
            generator = np.array((1, 0, 0, 1, 1), dtype=np.int8)
            dividend = np.asarray(word, dtype=np.int8).copy()
            for index in range(11):
                if dividend[index]:
                    dividend[index:index+5] ^= generator
            return dividend[11:]

        # Work on a row matrix while preserving the one-word return form.
        words = np.asarray(codeword, dtype=np.int8)
        oneWord = words.ndim == 1
        words = words.reshape(-1, 15).copy()
        cnumerr = np.full(words.shape[0], -1, dtype=np.int8)
        # A valid word has zero syndrome; otherwise test every single-bit
        # correction because BCH(15, 11) corrects one error per codeword.
        for row, word in enumerate(words):
            if not np.any(remainder(word)):
                cnumerr[row] = 0
                continue
            for bitIndex in range(15):
                candidate = word.copy()
                candidate[bitIndex] ^= 1
                if not np.any(remainder(candidate)):
                    words[row] = candidate
                    cnumerr[row] = 1
                    break
        decoded = words[:, :11]
        if oneWord:
            return decoded[0], int(cnumerr[0])
        return decoded, cnumerr

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
    def _field(bits, *ranges):
        """Select MATLAB-style one-based bit ranges from a message.

        Args
        ----
            bits    - array_like
                    Complete binary navigation field.
            ranges  - int or tuple of int
                    Individual one-based positions or inclusive ranges.

        Returns
        -------
            str
                Selected bits concatenated in the supplied order.
        """
        selected = []
        for item in ranges:
            if isinstance(item, int):
                selected.append(bits[item-1])
            else:
                first, last = item
                selected.extend(bits[first-1:last])
        return "".join(str(int(bit)) for bit in selected)

    @staticmethod
    def _deinterleave(subframe, wordCount):
        """Deinterleave one BDS D1/D2 subframe as specified by the SIS ICD.

        Args
        ----
            subframe  - array_like
                      Interleaved binary subframe.
            wordCount - int
                      Number of 30-bit words to deinterleave.

        Returns
        -------
            ndarray
                Deinterleaved subframe bits.
        """
        deSubframe = list(subframe[:30])
        for word in range(1, wordCount):
            block = subframe[word*30:(word+1)*30]
            deSubframe.extend(block[0:22:2])
            deSubframe.extend(block[1:22:2])
            deSubframe.extend(block[22:30:2])
            deSubframe.extend(block[23:30:2])
        return np.asarray(deSubframe, dtype=np.int8)

    @classmethod
    def _decodeBchWords(cls, subframe, wordCount):
        """Decode the BCH-protected information fields in one subframe.

        Args
        ----
            subframe  - ndarray
                      Deinterleaved binary subframe; corrected information
                      bits are written into this array.
            wordCount - int
                      Number of 30-bit words to decode.

        Returns
        -------
            ndarray or None
                Corrected subframe, or ``None`` when any BCH word is
                uncorrectable.
        """
        decoded, cnumerr = cls.bchdec(subframe[15:30])
        if cnumerr < 0:
            return None
        subframe[15:26] = decoded
        for word in range(1, wordCount):
            first = word*30
            codeword = np.vstack((
                np.r_[subframe[first:first+11], subframe[first+22:first+26]],
                np.r_[subframe[first+11:first+22],
                      subframe[first+26:first+30]]))
            decoded, cnumerr = cls.bchdec(codeword)
            if np.any(cnumerr < 0):
                return None
            subframe[first:first+11] = decoded[0]
            subframe[first+11:first+22] = decoded[1]
        return subframe

    def decodeEphemeris(self, bits, PRN):
        """Decode ephemerides and SOW from a BDS D1 or D2 bit stream.

        ``bits`` must contain at least 1500 navigation bits. D1 decoding can
        use 1500--2100 bits (five to seven subframes), while D2 decoding uses
        15000 bits (fifty pages). Its first element must be the first bit of a
        subframe or page. The first supplied subframe/page ID is not
        important. GEO satellites use the D2 message; IGSO and MEO satellites
        use the D1 message. BCH correction is performed while each subframe
        or page is decoded.

        Args
        ----
            bits   - array_like
                   Navigation-message bits containing only zero and one.
            PRN    - int
                   Satellite PRN. GEO and MEO/IGSO satellites have different
                   message structures.

        Returns
        -------
            SOW    - float
                   Seconds of week of the first decoded subframe; infinity
                   when the stream cannot be decoded.
        """
        #--- Decode navigation message -------------------------------------
        # Select the required bits and convert them to decimal values. For
        # message contents refer to the BeiDou SIS ICD, version 1.0 (2012).
        bits = np.asarray(bits, dtype=np.int8).reshape(-1)
        # Eleven-bit BDS preamble and its inverted form.
        preamble = np.array((1,1,1,0,0,0,1,0,0,1,0), dtype=np.int8)
        invertedPreamble = 1-preamble
        if bits.size < 1500:
            return np.inf
        if np.array_equal(bits[:11], invertedPreamble):
            bits = 1-bits
        elif not np.array_equal(bits[:11], preamble):
            return np.inf

        self.PRN = PRN
        # Pi used by the BDS broadcast equations.
        bdsPi = 3.1415926535898
        SOW = np.inf

        #%% Ephemeris for GEO satellites ===================================
        if PRN <= 5 or 59 <= PRN <= 63:
            # D2 subframe 1 contains ten pages. Fifty 300-bit pages span the
            # complete five-subframe record collected by NAVdecoding.
            if bits.size < 15000:
                return np.inf
            # Keep each ten-page D2 cycle separate. This prevents split fields
            # and the IODC/IODE records from being assembled across cycles.
            cycles = {}

            # Examine all fifty D2 pages in the collected record.
            for pageIndex in range(50):
                subframe = bits[300*pageIndex:300*(pageIndex+1)]
                # Deinterleave and BCH-decode the five words of this page.
                subframe = self._deinterleave(subframe, 5)
                subframe = self._decodeBchWords(subframe, 5)
                if subframe is None:
                    continue
                subframeID = self.bin2dec(self._field(subframe, (16,18)))
                if subframeID != 1:
                    continue
                # Decode page number and SOW of the first usable page.
                pageNumber = self.bin2dec(self._field(subframe, (43,46)))
                if not 1 <= pageNumber <= 10:
                    continue
                if np.isinf(SOW):
                    SOW = self.bin2dec(self._field(subframe, (19,26), (31,42)))-0.6*pageIndex
                    self.SOW = SOW

                cycleKey = pageIndex-(pageNumber-1)
                cycle = cycles.setdefault(cycleKey, {
                    "eph": type(self)(),
                    "fragments": {name: ["", ""] for name in (
                        "a1", "C_uc", "e", "C_ic", "i_0",
                        "omegaDot", "omega")},
                    "valid": np.zeros(10, dtype=bool)})
                candidate = cycle["eph"]
                candidate.PRN = PRN
                fragments = cycle["fragments"]

                #--- Page 1: health, clock issue, week and group delay ------
                if pageNumber == 1:
                    candidate.SatH1 = self.bin2dec(self._field(subframe, 47))
                    candidate.IODC = self.bin2dec(self._field(subframe, (48,52)))
                    candidate.URAI = self.bin2dec(self._field(subframe, (61,64)))
                    candidate.WN = self.bin2dec(self._field(subframe, (65,77)))
                    candidate.t_oc = self.bin2dec(self._field(subframe, (78,82), (91,102)))*2**3
                    # B3I is the BDS clock-reference signal, so no inter-signal
                    # group-delay correction is applied to B3I observations.
                    candidate.T_GD_1 = 0.0
                #--- Page 2: ionospheric model parameters ------------------
                elif pageNumber == 2:
                    candidate.alpha0 = twosComp2dec(self._field(subframe, (47,52), (61,62)))*2**-30
                    candidate.alpha1 = twosComp2dec(self._field(subframe, (63,70)))*2**-27
                    candidate.alpha2 = twosComp2dec(self._field(subframe, (71,78)))*2**-24
                    candidate.alpha3 = twosComp2dec(self._field(subframe, (79,82), (91,94)))*2**-24
                    candidate.beta0 = twosComp2dec(self._field(subframe, (95,102)))*2**11
                    candidate.beta1 = twosComp2dec(self._field(subframe, (103,110)))*2**14
                    candidate.beta2 = twosComp2dec(self._field(subframe, (111,112), (121,126)))*2**16
                    candidate.beta3 = twosComp2dec(self._field(subframe, (127,134)))*2**16
                #--- Pages 3 through 10: clock and ephemeris parameters ----
                elif pageNumber == 3:
                    candidate.a0 = twosComp2dec(self._field(subframe, (101,112), (121,132)))*2**-33
                    fragments["a1"][0] = self._field(subframe, (133,136))
                elif pageNumber == 4:
                    fragments["a1"][1] = self._field(subframe, (47,52), (61,72))
                    candidate.a2 = twosComp2dec(self._field(subframe, (73,82), 91))*2**-66
                    candidate.IODE = self.bin2dec(self._field(subframe, (92,96)))
                    candidate.deltan = twosComp2dec(self._field(subframe, (97,112)))*2**-43*bdsPi
                    fragments["C_uc"][0] = self._field(subframe, (121,134))
                elif pageNumber == 5:
                    fragments["C_uc"][1] = self._field(subframe, (47,50))
                    candidate.M_0 = twosComp2dec(self._field(subframe, (51,52), (61,82), (91,98)))*2**-31*bdsPi
                    candidate.C_us = twosComp2dec(self._field(subframe, (99,112), (121,124)))*2**-31
                    fragments["e"][0] = self._field(subframe, (125,134))
                elif pageNumber == 6:
                    fragments["e"][1] = self._field(subframe, (47,52), (61,76))
                    candidate.sqrtA = self.bin2dec(self._field(subframe, (77,82), (91,112), (121,124)))*2**-19
                    fragments["C_ic"][0] = self._field(subframe, (125,134))
                elif pageNumber == 7:
                    fragments["C_ic"][1] = self._field(subframe, (47,52), (61,62))
                    candidate.C_is = twosComp2dec(self._field(subframe, (63,80)))*2**-31
                    candidate.t_oe = self.bin2dec(self._field(subframe, (81,82), (91,105)))*2**3
                    fragments["i_0"][0] = self._field(subframe, (106,112), (121,134))
                elif pageNumber == 8:
                    fragments["i_0"][1] = self._field(subframe, (47,52), (61,65))
                    candidate.C_rc = twosComp2dec(self._field(subframe, (66,82), 91))*2**-6
                    candidate.C_rs = twosComp2dec(self._field(subframe, (92,109)))*2**-6
                    fragments["omegaDot"][0] = self._field(subframe, (110,112), (121,136))
                elif pageNumber == 9:
                    fragments["omegaDot"][1] = self._field(subframe, (47,51))
                    candidate.omega_0 = twosComp2dec(self._field(subframe, 52, (61,82), (91,99)))*2**-31*bdsPi
                    fragments["omega"][0] = self._field(subframe, (100,112), (121,134))
                elif pageNumber == 10:
                    fragments["omega"][1] = self._field(subframe, (47,51))
                    candidate.iDot = twosComp2dec(self._field(subframe, 52, (61,73)))*2**-43*bdsPi
                # Record successful decoding of every requisite D2 page.
                cycle["valid"][pageNumber-1] = True

            completeCycle = next((cycles[key] for key in sorted(cycles)
                                  if np.all(cycles[key]["valid"])), None)
            if completeCycle is None:
                return SOW
            candidate = completeCycle["eph"]
            fragments = completeCycle["fragments"]
            # Combine split fields from one complete ten-page cycle only.
            fieldBits = "".join(fragments["a1"])
            if len(fieldBits) == 22:
                candidate.a1 = twosComp2dec(fieldBits)*2**-50
            fieldBits = "".join(fragments["C_uc"])
            if len(fieldBits) == 18:
                candidate.C_uc = twosComp2dec(fieldBits)*2**-31
            fieldBits = "".join(fragments["e"])
            if len(fieldBits) == 32:
                candidate.e = self.bin2dec(fieldBits)*2**-33
            fieldBits = "".join(fragments["C_ic"])
            if len(fieldBits) == 18:
                candidate.C_ic = twosComp2dec(fieldBits)*2**-31
            fieldBits = "".join(fragments["i_0"])
            if len(fieldBits) == 32:
                candidate.i_0 = twosComp2dec(fieldBits)*2**-31*bdsPi
            fieldBits = "".join(fragments["omegaDot"])
            if len(fieldBits) == 24:
                candidate.omegaDot = twosComp2dec(fieldBits)*2**-43*bdsPi
            fieldBits = "".join(fragments["omega"])
            if len(fieldBits) == 32:
                candidate.omega = twosComp2dec(fieldBits)*2**-31*bdsPi
            candidate.SOW = SOW
            candidate.flag = 1
            self.__dict__.update(candidate.__dict__)

        #%% Ephemeris for MEO and IGSO satellites ==========================
        elif 6 <= PRN <= 58:
            # Keep subframes 1, 2 and 3 within the same five-subframe cycle.
            cycles = {}
            for frameIndex in range(bits.size//300):
                subframe = bits[300*frameIndex:300*(frameIndex+1)]
                # Deinterleave and BCH-decode the first protected word.
                subframe = self._deinterleave(subframe, 10)
                decoded, cnumerr = self.bchdec(subframe[15:30])
                if cnumerr < 0:
                    continue
                subframe[15:26] = decoded
                subframeID = self.bin2dec(self._field(subframe, (16,18)))
                bchValid = subframeID in (1,2,3)
                # Decode all remaining BCH-protected word pairs.
                if bchValid:
                    for word in range(1, 5):
                        first = word*30
                        codeword = np.vstack((
                            np.r_[subframe[first:first+11],
                                  subframe[first+22:first+26]],
                            np.r_[subframe[first+11:first+22],
                                  subframe[first+26:first+30]]))
                        decoded, cnumerr = self.bchdec(codeword)
                        if np.any(cnumerr < 0):
                            bchValid = False
                            break
                        subframe[first:first+11] = decoded[0]
                        subframe[first+11:first+22] = decoded[1]
                if not bchValid:
                    continue
                frameKey = frameIndex-(subframeID-1)
                cycle = cycles.setdefault(frameKey, {
                    "eph": type(self)(), "valid": np.zeros(3, dtype=bool),
                    "toe": ["", ""]})
                candidate = cycle["eph"]
                candidate.PRN = PRN
                # SOW refers to the first supplied subframe.
                if np.isinf(SOW):
                    SOW = (self.bin2dec(self._field(subframe, (19,26), (31,42)))
                           - 6*frameIndex)
                    self.SOW = SOW

                #--- Subframe 1: clock, health and ionospheric parameters ---
                if subframeID == 1:
                    candidate.SatH1 = self.bin2dec(self._field(subframe, 43))
                    candidate.IODC = self.bin2dec(self._field(subframe, (44,48)))
                    candidate.URAI = self.bin2dec(self._field(subframe, (49,52)))
                    candidate.WN = self.bin2dec(self._field(subframe, (61,73)))
                    candidate.t_oc = self.bin2dec(self._field(subframe, (74,82), (91,98)))*2**3
                    # B3I is the BDS clock-reference signal.
                    candidate.T_GD_1 = 0.0
                    candidate.alpha0 = twosComp2dec(self._field(subframe, (127,134)))*2**-30
                    candidate.alpha1 = twosComp2dec(self._field(subframe, (135,142)))*2**-27
                    candidate.alpha2 = twosComp2dec(self._field(subframe, (151,158)))*2**-24
                    candidate.alpha3 = twosComp2dec(self._field(subframe, (159,166)))*2**-24
                    candidate.beta0 = twosComp2dec(self._field(subframe, (167,172), (181,182)))*2**11
                    candidate.beta1 = twosComp2dec(self._field(subframe, (183,190)))*2**14
                    candidate.beta2 = twosComp2dec(self._field(subframe, (191,198)))*2**16
                    candidate.beta3 = twosComp2dec(self._field(subframe, (199,202), (211,214)))*2**16
                    candidate.a2 = twosComp2dec(self._field(subframe, (215,225)))*2**-66
                    candidate.a0 = twosComp2dec(self._field(subframe, (226,232), (241,257)))*2**-33
                    candidate.a1 = twosComp2dec(self._field(subframe, (258,262), (271,287)))*2**-50
                    candidate.IODE = self.bin2dec(self._field(subframe, (288,292)))
                    cycle["valid"][0] = True
                #--- Subframe 2: first part of the broadcast ephemeris -----
                elif subframeID == 2:
                    candidate.deltan = twosComp2dec(self._field(subframe, (43,52), (61,66)))*2**-43*bdsPi
                    candidate.C_uc = twosComp2dec(self._field(subframe, (67,82), (91,92)))*2**-31
                    candidate.M_0 = twosComp2dec(self._field(subframe, (93,112), (121,132)))*2**-31*bdsPi
                    candidate.e = self.bin2dec(self._field(subframe, (133,142), (151,172)))*2**-33
                    candidate.C_us = twosComp2dec(self._field(subframe, (181,198)))*2**-31
                    candidate.C_rc = twosComp2dec(self._field(subframe, (199,202), (211,224)))*2**-6
                    candidate.C_rs = twosComp2dec(self._field(subframe, (225,232), (241,250)))*2**-6
                    candidate.sqrtA = self.bin2dec(self._field(subframe, (251,262), (271,290)))*2**-19
                    cycle["toe"][0] = self._field(subframe, (291,292))
                    cycle["valid"][1] = True
                #--- Subframe 3: remaining broadcast ephemeris fields ------
                elif subframeID == 3:
                    cycle["toe"][1] = self._field(subframe, (43,52), (61,65))
                    candidate.i_0 = twosComp2dec(self._field(subframe, (66,82), (91,105)))*2**-31*bdsPi
                    candidate.C_ic = twosComp2dec(self._field(subframe, (106,112), (121,131)))*2**-31
                    candidate.omegaDot = twosComp2dec(self._field(subframe, (132,142), (151,163)))*2**-43*bdsPi
                    candidate.C_is = twosComp2dec(self._field(subframe, (164,172), (181,189)))*2**-31
                    candidate.iDot = twosComp2dec(self._field(subframe, (190,202), 211))*2**-43*bdsPi
                    candidate.omega_0 = twosComp2dec(self._field(subframe, (212,232), (241,251)))*2**-31*bdsPi
                    candidate.omega = twosComp2dec(self._field(subframe, (252,262), (271,291)))*2**-31*bdsPi
                    cycle["valid"][2] = True

            completeCycle = next((cycles[key] for key in sorted(cycles)
                                  if np.all(cycles[key]["valid"])), None)
            if completeCycle is not None:
                candidate = completeCycle["eph"]
                toeBits = "".join(completeCycle["toe"])
                if len(toeBits) == 17:
                    candidate.t_oe = self.bin2dec(toeBits)*2**3
                candidate.SOW = SOW
                candidate.flag = 1
                self.__dict__.update(candidate.__dict__)
        return SOW

    @classmethod
    def NAVdecoding(cls, trackResult, settings):
        """Find the first valid D1/D2 preamble and decode its ephemeris.

        A preamble is verified by the six-second spacing between consecutive
        preambles and by BCH checking of the first word. GEO satellites use
        the D2 symbol rate, while IGSO/MEO satellites are demodulated with the
        20-chip Neumann-Hoffman code used by D1.

        Args
        ----
            trackResult - tracking.Channel
                        Prompt tracking output and PRN for one channel.
            settings    - initSettings.Settings
                        Receiver settings; retained for interface consistency
                        with the other navigation decoders.

        Returns
        -------
            eph            - Ephemeris
                           Decoded satellite ephemeris.
            subFrameStart  - int or float
                           Starting position of the first valid message in the
                           tracking stream; infinity when none is detected.
            SOW            - float
                           Seconds of week of the first message; infinity when
                           no valid preamble is detected.
        """
        #--- Initialize ephemeris structure --------------------------------
        # Every satellite receives the same field layout even when none of the
        # requisite navigation messages can be decoded.
        eph = cls()
        subFrameStart = np.inf
        SOW = np.inf
        # Skip the initial tracking transient before preamble search.
        searchStartOffset = 1000
        # BDS preamble and 20-chip Neumann-Hoffman code in antipodal form.
        preambleBits = np.array((1,1,1,-1,-1,-1,1,-1,-1,1,-1), dtype=np.int8)
        nhCode = np.array((-1,-1,-1,-1,-1,1,-1,-1,1,1,
                           -1,1,-1,1,-1,-1,1,1,1,-1), dtype=np.int8)
        PRN = trackResult.PRN
        # GEO D2 bits span two code periods; IGSO/MEO D1 bits span 20 periods.
        if PRN <= 5 or 59 <= PRN <= 63:
            preamble = np.kron(preambleBits, np.ones(2, dtype=np.int8))
            codePeriodsPerBit = 2
        else:
            preamble = np.kron(preambleBits, -nhCode)
            codePeriodsPerBit = 20

        #%% Search for a valid preamble ====================================
        prompt = np.asarray(trackResult.I_P).reshape(-1)
        if prompt.size <= searchStartOffset+preamble.size:
            return eph, subFrameStart, SOW
        # Hard decisions are used only for preamble synchronization.
        hardBits = np.where(prompt[searchStartOffset:] > 0, 1, -1)
        # Correlate the tracking output with the signal-specific preamble.
        correlation = np.correlate(hardBits, preamble, mode="valid")
        candidates = np.flatnonzero(
            np.abs(correlation) >= codePeriodsPerBit*10)+searchStartOffset

        # Verify six-second preamble spacing and the first BCH codeword.
        for candidate in candidates:
            if not np.any(candidates-candidate == 300*codePeriodsPerBit):
                continue
            stop = candidate+30*codePeriodsPerBit
            if stop > prompt.size:
                continue
            word = prompt[candidate:stop].reshape(30, codePeriodsPerBit).sum(1)
            word = np.where(word > 0, 1, 0)
            _, cnumerr = cls.bchdec(word[15:30])
            if cnumerr == 0:
                subFrameStart = int(candidate)
                break

        if np.isinf(subFrameStart):
            print(f"    Could not find valid preambles for PRN {PRN:02d}!")
            return eph, subFrameStart, SOW

        #%% Extract and demodulate the navigation record ===================
        # GEO D2 needs 50 pages (15,000 bits). D1 may inspect up to seven
        # subframes so that subframes 1, 2 and 3 can be selected from one
        # five-subframe cycle even when tracking starts at subframe 2 or 3.
        availablePeriods = prompt.size-int(subFrameStart)
        if PRN <= 5 or 59 <= PRN <= 63:
            navBitCount = 15000
            codePeriodsPerBit = 2
        else:
            frameCount = min(7, availablePeriods//(300*20))
            if frameCount < 5:
                print(f"    Insufficient navigation data for PRN {PRN:02d}!")
                return eph, np.inf, SOW
            navBitCount = 300*frameCount
            codePeriodsPerBit = 20
        recordLength = navBitCount*codePeriodsPerBit
        stop = int(subFrameStart)+recordLength
        if stop > prompt.size:
            print(f"    Insufficient navigation data for PRN {PRN:02d}!")
            return eph, np.inf, SOW
        navSamples = prompt[int(subFrameStart):stop]
        # Integrate each D2 symbol directly or remove the D1 NH code.
        if PRN <= 5 or 59 <= PRN <= 63:
            navBitsSamples = navSamples.reshape(-1, 2)
        else:
            demodulation = np.tile(nhCode, navBitCount)
            navBitsSamples = (navSamples*demodulation).reshape(-1, 20)
        # Make one hard decision per navigation bit and decode ephemeris.
        navBits = (navBitsSamples.sum(1) > 0).astype(np.int8)
        SOW = eph.decodeEphemeris(navBits, PRN)
        return eph, subFrameStart, SOW

    def satpos(self, transmitTime):
        """Calculate satellite position and clock correction.

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
        # Pi used by the BDS coordinate equations.
        bdsPi = 3.1415926535898
        # Earth rotation rate [rad/s].
        Omegae_dot = 7.2921150e-5
        # Earth's universal gravitational parameter [m^3/s^2].
        GM = 3.986004418e14
        # Relativistic correction constant [s/m^(1/2)].
        F = -4.442807633e-10
        #%% Find initial satellite clock correction ========================
        # Time from the clock reference epoch, corrected for week crossover.
        dt = self.check_t(transmitTime-self.t_oc)
        # Apply the broadcast clock model and B3I group-delay correction.
        satClkCorr = (self.a2*dt+self.a1)*dt+self.a0-self.T_GD_1
        time = transmitTime-satClkCorr
        #%% Find satellite position ========================================
        # Restore the semi-major axis and time from ephemeris reference epoch.
        A = self.sqrtA**2
        tk = self.check_t(time-self.t_oe)
        # Mean motion and mean anomaly reduced to one revolution.
        n0 = np.sqrt(GM/A**3)
        M = np.fmod(self.M_0+(n0+self.deltan)*tk+2*bdsPi, 2*bdsPi)
        # Iteratively solve Kepler's equation for eccentric anomaly.
        E = M
        for _ in range(10):
            oldE = E
            E = M+self.e*np.sin(E)
            if abs(np.fmod(E-oldE, 2*bdsPi)) < 1e-12:
                break
        # Reduce eccentric anomaly to one revolution.
        E = np.fmod(E+2*bdsPi, 2*bdsPi)
        # Relativistic correction and true anomaly.
        dtr = F*self.e*self.sqrtA*np.sin(E)
        nu = np.arctan2(np.sqrt(1-self.e**2)*np.sin(E), np.cos(E)-self.e)
        # Argument of latitude before harmonic corrections.
        phi = np.fmod(nu+self.omega, 2*bdsPi)
        # Correct argument of latitude, radius and inclination.
        u = phi+self.C_uc*np.cos(2*phi)+self.C_us*np.sin(2*phi)
        r = A*(1-self.e*np.cos(E))+self.C_rc*np.cos(2*phi)+self.C_rs*np.sin(2*phi)
        i = self.i_0+self.iDot*tk+self.C_ic*np.cos(2*phi)+self.C_is*np.sin(2*phi)
        # Longitude of ascending node differs for GEO and IGSO/MEO orbits.
        if self.PRN <= 5 or 59 <= self.PRN <= 63:
            Omega = self.omega_0+self.omegaDot*tk-Omegae_dot*self.t_oe
        else:
            Omega = (self.omega_0+(self.omegaDot-Omegae_dot)*tk
                     - Omegae_dot*self.t_oe)
        Omega = np.fmod(Omega+2*bdsPi, 2*bdsPi)
        #--- Compute satellite ECEF coordinates ----------------------------
        satPosition = np.array((
            np.cos(u)*r*np.cos(Omega)-np.sin(u)*r*np.cos(i)*np.sin(Omega),
            np.cos(u)*r*np.sin(Omega)+np.sin(u)*r*np.cos(i)*np.cos(Omega),
            np.sin(u)*r*np.sin(i)))
        # Rotate GEO coordinates by the five-degree orbit inclination model.
        if self.PRN <= 5 or 59 <= self.PRN <= 63:
            angleX = -5*bdsPi/180
            angleZ = Omegae_dot*tk
            Rx = np.array(((1,0,0), (0,np.cos(angleX),np.sin(angleX)),
                           (0,-np.sin(angleX),np.cos(angleX))))
            Rz = np.array(((np.cos(angleZ),np.sin(angleZ),0),
                           (-np.sin(angleZ),np.cos(angleZ),0), (0,0,1)))
            satPosition = Rz@Rx@satPosition
        # Include the relativistic term in the final clock correction.
        satClkCorr = (self.a2*dt+self.a1)*dt+self.a0-self.T_GD_1+dtr
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
        # Satellite coordinates require the complete D1/D2 ephemeris and
        # clock data. Tracking starts at an arbitrary point in the message,
        # so retain the 36-second minimum record length used by the BDS MATLAB
        # postNavigation implementation.
        if settings.msToProcess < 36000:
            print("Record is too short. Exiting!")
            return

        trackResults = list(trackResults)

        #%% Preallocate per-channel message timing
        # Starting position of the first valid D1/D2 frame in the 1 ms
        # B3I primary-code stream. Infinity means no frame was decoded.
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
            print(f"Decoding D1/D2 for PRN {PRN:02d} -----------------")

            currentEph, subFrameStart[channelNr], TOW[channelNr] = (
                Ephemeris.NAVdecoding(trackResults[channelNr], settings))
            self.eph[PRN]  = currentEph

            #--- Exclude satellites without the necessary nav data ------------
            # Exclude satellites without all requisite D1/D2 messages from
            # subsequent positioning.
            if currentEph.flag != 1:
                print(f"    Ephemeris decoding fails for PRN {PRN:02d}!")
                activeChnList.remove(channelNr)
            elif currentEph.SatH1 != 0:
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
