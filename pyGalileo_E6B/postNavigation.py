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
Galileo E6B High Accuracy Service page decoding.

The corresponding MATLAB receiver decodes HAS pages and time-of-hour data,
but it does not provide the complete broadcast orbit and clock information
required for a standalone E6B position solution.

"""

import numpy as np


#%% Galileo E6B HAS data
class Ephemeris:
    """Decode Galileo E6B pages and assemble receiver-wide HAS messages.

    The implementation follows MATLAB ``NAVdecoding`` and
    ``ephemeris_E6B``: it synchronizes and channel-decodes E6B pages,
    returns CRC-valid physical pages, buffers pages received from all
    satellites, and performs GF(256) Reed-Solomon erasure decoding after the
    required number of distinct Page IDs has been collected.
    """

    def __init__(self):
        """Initialize HAS results, page buffers, and finite-field tables.

        Returns
        -------
            None
        """
        # Initialize structure fields so unsuccessful or partial decoding
        # still returns the same HAS data structure for every satellite.
        self.PRN = 0
        self.flag = 0
        self.TOH = np.inf
        self.referenceGST = np.inf
        self.HAS = None
        self.HAS_Buffer = {}
        self.HASS = None
        self.completedMessages = []
        self._gfExp = self._gfLog = self._generator = None

    @staticmethod
    def _unsigned(bits):
        """Convert a most-significant-bit-first bit sequence to an integer.

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
        """Hard-decision decode the rate-1/2, constraint-length-7 code.

        The convolutional generators are 171 and inverted 133 in octal, as
        specified for Galileo and used by MATLAB ``poly2trellis``.

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
                    metric = (pathMetric[state]
                              + int(output0 != received[0])
                              + int(output1 != received[1]))
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
                 Decoded E6B page bits including the 24 CRC bits.

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

    def _initGF256(self):
        """Initialize GF(256) tables and load the official RS matrix.

        The field uses primitive polynomial 285. The systematic generator
        matrix has the MATLAB dimensions N=255 and K=32 and is cached for
        subsequent HAS messages.

        Returns
        -------
            None
        """
        if self._generator is not None:
            return
        #%% Initialize the GF(256) exponent and logarithm tables ===============
        self._gfExp = np.zeros(512, dtype=np.uint16)
        self._gfLog = np.zeros(256, dtype=np.int16)
        value = 1
        for exponent in range(255):
            self._gfExp[exponent] = value
            self._gfLog[value] = exponent
            value <<= 1
            if value & 256:
                value ^= 285
        self._gfExp[255:511] = self._gfExp[:256]

        #%% Load the systematic Reed-Solomon generator matrix =================
        matrixPath = ("./assets/Galileo-HAS-SIS-ICD_1.0_Annex_B_"
                      "Reed_Solomon_Generator_Matrix.txt")
        generator = np.loadtxt(matrixPath, delimiter=",", dtype=np.uint16)
        if (generator.shape != (255, 32)
                or np.any(generator > 255)
                or not np.array_equal(generator[:32], np.eye(32))):
            raise ValueError("Invalid Galileo HAS RS generator matrix")
        self._generator = generator.astype(np.uint8)

    def _gfMultiply(self, value1, value2):
        """Multiply two elements in the initialized GF(256) field.

        Args
        ----
            value1 - int
                   First GF(256) element.
            value2 - int
                   Second GF(256) element.

        Returns
        -------
            int
                Product in GF(256).
        """
        if value1 == 0 or value2 == 0:
            return 0
        return int(self._gfExp[
            int(self._gfLog[value1])+int(self._gfLog[value2])])

    def _gfInverse(self, value):
        """Return the multiplicative inverse of one GF(256) element.

        Args
        ----
            value - int
                  Nonzero GF(256) element.

        Returns
        -------
            int
                Multiplicative inverse in GF(256).
        """
        if value == 0:
            raise ArithmeticError("GF(256) division by zero")
        return int(self._gfExp[255-int(self._gfLog[value])])

    def _rsErasureDecode(self, rxBlock, pids, K):
        """Decode received RS erasure pages using their page identifiers.

        Args
        ----
            rxBlock - numpy.ndarray
                    ``K x 53`` array of received HAS page bytes.
            pids    - array_like
                    One-based Page IDs identifying the generator rows.
            K       - int
                    Number of information pages for this HAS message.

        Returns
        -------
            numpy.ndarray
                Decoded ``K x 53`` information-byte array.
        """
        # Select the generator rows belonging to the received Page IDs.
        coefficient = self._generator[np.asarray(pids)-1, :K].copy()
        # Invert the coefficient matrix by Gaussian elimination in GF(256).
        inverse = np.eye(K, dtype=np.uint8)
        for row in range(K):
            pivot = int(coefficient[row, row])
            if pivot == 0:
                candidates = np.flatnonzero(coefficient[row+1:, row])
                if candidates.size == 0:
                    raise ArithmeticError("Singular Reed-Solomon matrix")
                swapRow = row+1+int(candidates[0])
                coefficient[[row, swapRow]] = coefficient[[swapRow, row]]
                inverse[[row, swapRow]] = inverse[[swapRow, row]]
                pivot = int(coefficient[row, row])
            invPivot = self._gfInverse(pivot)
            for column in range(K):
                coefficient[row, column] = self._gfMultiply(
                    int(coefficient[row, column]), invPivot)
                inverse[row, column] = self._gfMultiply(
                    int(inverse[row, column]), invPivot)
            for otherRow in range(K):
                if otherRow == row:
                    continue
                factor = int(coefficient[otherRow, row])
                if factor:
                    for column in range(K):
                        coefficient[otherRow, column] ^= self._gfMultiply(
                            factor, int(coefficient[row, column]))
                        inverse[otherRow, column] ^= self._gfMultiply(
                            factor, int(inverse[row, column]))

        # Recover every byte column of the original information pages.
        decoded = np.zeros((K, 53), dtype=np.uint8)
        for row in range(K):
            for column in range(53):
                value = 0
                for index in range(K):
                    value ^= self._gfMultiply(
                        int(inverse[row, index]), int(rxBlock[index, column]))
                decoded[row, column] = value
        return decoded

    def decodePage(self, pageSymbols, rxTime, absoluteSample, PRN):
        """Decode one physical Galileo E6B page.

        Args
        ----
            pageSymbols - array_like
                        One E6B page in antipodal ``-1/+1`` form, beginning
                        with its 16-symbol synchronization pattern.
            rxTime     - float
                       Receiver time of the page in seconds.
            absoluteSample - float
                           Absolute IF-sample position of the preamble.
            PRN        - int
                       Source Galileo satellite number.

        Returns
        -------
            dict or None
                CRC-valid page record, or ``None`` when decoding fails.
        """
        #%% 0. Environment initialization =====================================
        # Input handling: force antipodal symbols into a compact array.
        pageSymbols = np.asarray(pageSymbols, dtype=np.int8)
        #%% 1. Physical-layer parameters ======================================
        preamble = np.array(
            [-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1],
            dtype=np.int8)
        #%% 2. Page processing =================================================
        # Synchronization check and polarity correction.
        if np.array_equal(pageSymbols[:16], preamble):
            dataSymbols = pageSymbols[16:]
        elif np.array_equal(pageSymbols[:16], -preamble):
            dataSymbols = -pageSymbols[16:]
        else:
            return None

        # De-interleave and remove the convolutional encoding.
        encodedBits = ((1-dataSymbols)//2).astype(np.uint8)
        decodedBits = self._viterbiDecode(
            np.reshape(np.reshape(encodedBits, (123, 8), order="F").T, -1, order="F"))[:486]
        # CRC check over the complete 486-bit decoded page.
        if not self._crc24q(decodedBits):
            return None

        #%% 3. HAS page parsing ===============================================
        hasPage = decodedBits[14:462]
        header = hasPage[:24]
        headerValue = self._unsigned(header)
        if headerValue == 0xAF3BC3:
            return None
        HASS = self._unsigned(header[:2])
        MT = self._unsigned(header[4:6])
        MID = self._unsigned(header[6:11])
        MSraw = self._unsigned(header[11:16])
        PID = self._unsigned(header[16:24])
        pageRecord = {"rxTime": float(rxTime),
                      "absoluteSample": float(absoluteSample),
                      "PRN": int(PRN), "HASS": HASS}
        # HASS=2/3 control pages are handled before validating fields that
        # do not participate in the service-status transition.
        if HASS in (2, 3):
            return pageRecord
        if PID < 1 or PID > 255:
            return None

        # Convert the 424-bit encoded payload to 53 bytes.
        payload = hasPage[24:]
        encodedBytes = np.array(
            [self._unsigned(payload[8*index:8*(index+1)])
             for index in range(53)], dtype=np.uint8)
        pageRecord.update({"MT": MT, "MID": MID, "MSraw": MSraw,
                           "PID": PID, "payload": encodedBytes})
        return pageRecord

    def _parseMT1Header(self, messageBits):
        """Parse the fixed 32-bit MT1 header without applying corrections."""
        TOH = self._unsigned(messageBits[:12])
        flagNames = ("Mask", "Orbit", "ClockFull", "ClockSubset",
                     "CodeBias", "PhaseBias")
        flags = {name: bool(messageBits[12+index])
                 for index, name in enumerate(flagNames)}
        return {"TOH": TOH, "Flags": flags,
                "Reserved": self._unsigned(messageBits[18:22]),
                "MaskID": self._unsigned(messageBits[22:27]),
                "IODSetID": self._unsigned(messageBits[27:32]),
                "RawBody": messageBits[32:].copy(),
                "BlocksDecoded": False}

    def _expireBuffers(self, rxTime):
        """Discard HAS message groups not completed within 150 seconds."""
        expired = [key for key, value in self.HAS_Buffer.items()
                   if rxTime-value["firstRxTime"] >= 150.0]
        for key in expired:
            del self.HAS_Buffer[key]

    def addPage(self, page):
        """Add one CRC-valid page to the receiver-wide HAS assembler."""
        rxTime = page["rxTime"]
        self._expireBuffers(rxTime)
        HASS = page["HASS"]

        # HASS=3 instructs users to stop using HAS and discard all messages.
        if HASS == 3:
            self.HAS_Buffer.clear()
            self.completedMessages.clear()
            self.HAS = None
            self.flag = 0
            self.TOH = np.inf
            self.referenceGST = np.inf
            self.HASS = HASS
            return None
        # HASS=2 is reserved and shall not enter the assembler.
        if HASS == 2:
            return None
        # Do not combine pages transmitted under different service states.
        if self.HASS in (0, 1) and self.HASS != HASS:
            self.HAS_Buffer.clear()
            self.HAS = None
            self.flag = 0
            self.TOH = np.inf
            self.referenceGST = np.inf
        self.HASS = HASS
        if page["MT"] != 1:
            return None

        key = (page["MT"], page["MID"], page["MSraw"], HASS)
        conflicts = [oldKey for oldKey in self.HAS_Buffer
                     if oldKey[0] == page["MT"]
                     and oldKey[1] == page["MID"]
                     and oldKey != key]
        for oldKey in conflicts:
            del self.HAS_Buffer[oldKey]

        K = page["MSraw"]+1
        messageBuffer = self.HAS_Buffer.setdefault(
            key, {"K": K, "firstRxTime": rxTime, "lastRxTime": rxTime,
                  "pages": {}, "sources": set()})
        messageBuffer["lastRxTime"] = rxTime
        messageBuffer["pages"].setdefault(page["PID"], page["payload"])
        messageBuffer["sources"].add(page["PRN"])
        if len(messageBuffer["pages"]) < K:
            return None

        self._initGF256()
        pids = sorted(messageBuffer["pages"])[:K]
        rxBlock = np.vstack([messageBuffer["pages"][pid] for pid in pids])
        decoded = self._rsErasureDecode(rxBlock, pids, K)
        messageBits = np.unpackbits(decoded.reshape(-1), bitorder="big")
        mt1 = self._parseMT1Header(messageBits)
        mt1.update({"HASS": HASS, "Operational": HASS == 1,
                    "MT": page["MT"], "MID": page["MID"],
                    "MSraw": page["MSraw"], "K": K, "PIDs": pids,
                    "Sources": sorted(messageBuffer["sources"]),
                    "FirstRxTime": messageBuffer["firstRxTime"],
                    "LastRxTime": messageBuffer["lastRxTime"],
                    "Data": messageBits})
        self.TOH = float(mt1["TOH"])
        self.HAS = mt1
        self.flag = 1
        self.completedMessages.append(mt1)
        del self.HAS_Buffer[key]
        return mt1

    @classmethod
    def NAVdecoding(cls, I_P_InputBits, absoluteSample, samplingFreq, PRN):
        """Find and physically decode all valid Galileo E6B pages.

        Preamble candidates are found in the prompt-correlator stream and
        verified by the spacing of consecutive 1000-symbol pages. Every
        CRC-valid physical page is returned to the receiver-wide assembler.

        Args
        ----
            I_P_InputBits - array_like
                          Prompt-I output from the tracking function.
            absoluteSample - array_like
                           Absolute IF-sample position of each tracking epoch.
            samplingFreq - float
                         IF sampling frequency in Hz.
            PRN        - int
                       Source Galileo satellite number.

        Returns
        -------
            list of dict
                CRC-valid page records with receiver time and source PRN.
        """
        # Use the prompt correlator as the antipodal symbol stream.
        bits = np.where(np.asarray(I_P_InputBits) > 0, 1, -1).astype(np.int8)
        #--- Generate the E6B synchronization pattern -------------------------
        preamble = np.array(
            [-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1],
            dtype=np.int8)
        # Correlate tracking output with the preamble.
        correlation = np.correlate(bits, preamble, mode="full")
        correlation = correlation[preamble.size-1:]
        #%% Find all starting points of preamble-like patterns ================
        index = np.flatnonzero(np.abs(correlation) > 15)
        indexSet = set(index.tolist())
        decoder = cls()
        pageRecords = []

        # Analyze each detected preamble-like pattern.
        for startIndex in index:
            # Check the expected spacing to the following page.
            if startIndex+1000 not in indexSet:
                continue
            #=== Read symbols for CRC checking and HAS decoding ================
            page = bits[startIndex:startIndex+1000]
            if page.size != 1000:
                continue
            # Correct page polarity according to its preamble.
            if not np.array_equal(page[:16], preamble):
                page = -page
            pageSample = float(absoluteSample[startIndex])
            pageRecord = decoder.decodePage(
                page, pageSample/samplingFreq, pageSample, PRN)
            if pageRecord is not None:
                pageRecords.append(pageRecord)
        return pageRecords


#%% Navigation results
class NavSolutions:
    """Store decoded E6B HAS results.

    E6B HAS time and correction data do not form an independent broadcast
    ephemeris, so receiver coordinates are intentionally not allocated.
    """

    def __init__(self):
        """Initialize the decoded HAS-message container.

        Returns
        -------
            None
        """
        self.HAS = {}


#%% Post-navigation engine
class NavigationEngine:
    """Decode Galileo E6B HAS pages from completed tracking channels.

    Args
    ----
        settings - object
                 Receiver settings.
    """

    def __init__(self, settings):
        """Initialize the Galileo E6B post-navigation engine.

        Args
        ----
            settings - object
                     Receiver settings.

        Returns
        -------
            None
        """
        self.settings = settings
        self.navSolutions = NavSolutions()
        self.eph = Ephemeris()

    def navigationRun(self, trackResults):
        """Decode E6B HAS pages without attempting a standalone PVT fix.

        The MATLAB E6B receiver supplies HAS corrections and Time Of Hour but
        not the broadcast orbit and clock fields required for an independent
        position solution.

        Args
        ----
            trackResults - iterable of tracking.Channel
                         Results from the tracking function.

        Returns
        -------
            None
                Receiver-wide HAS messages assembled from all satellites are
                stored in ``navSolutions`` and ``eph``.
        """
        #%% Initialize HAS decoding ===========================================
        # Reset the decoded HAS results for this run.
        self.navSolutions = NavSolutions()
        self.eph = Ephemeris()
        pageRecords = []
        #--- Make a list excluding channels not in tracking lock --------------
        for channel in trackResults:
            if not channel.lockFlag:
                continue
            #--- Decode pages for the current PRN ------------------------------
            print(f"Decoding HAS for PRN {channel.PRN:02d} --------------------")
            channelPages = Ephemeris.NAVdecoding(
                channel.I_P, channel.absoluteSample,
                self.settings.samplingFreq, channel.PRN)
            pageRecords.extend(channelPages)
            print(f"    {len(channelPages)} CRC-valid HAS pages found.")

        # Assemble common HAS messages after merging pages from all satellites.
        for page in sorted(
                pageRecords, key=lambda value: value["absoluteSample"]):
            message = self.eph.addPage(page)
            if message is not None:
                print(f"    HAS MT{message['MT']} MID {message['MID']:02d} "
                      f"decoded from {len(message['Sources'])} satellites.")

        # Build the public output only after all HASS state changes have been
        # processed. HASS=3 clears completedMessages inside the assembler.
        for message in self.eph.completedMessages:
            key = (message["MT"], message["MID"], message["MSraw"],
                   message["HASS"])
            self.navSolutions.HAS.setdefault(key, []).append(message)

        #%% Position calculation ==============================================
        # E6B HAS does not provide an independent broadcast ephemeris for PVT.
        print("Galileo E6B MATLAB source does not contain a complete standalone "
              "broadcast-ephemeris PVT implementation; position calculation "
              "is therefore not performed.")

    def plotNavigation(self):
        """Report that no standalone E6B navigation solution is available.

        Returns
        -------
            None
        """
        print("plotNavigation: Galileo E6B HAS decoding provides no standalone "
              "position solution to plot.")


__all__ = ["Ephemeris", "NavSolutions", "NavigationEngine"]
