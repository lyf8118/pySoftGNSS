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

channelDecoder.py - Module Description
--------------------------------------
BDS-3 B1C BCH and nonbinary LDPC channel decoding.

"""

import numpy as np

#%% BCH decoder
class BCHDecoder:
    """Decode the BCH-protected symbols in B-CNAV1 subframe 1."""

    @staticmethod
    def _bch_decode(bits, dataLength, codeLength, feedbackPos, threshold):
        """Decode a short BCH word by testing every source-data hypothesis.

        Args
        ----
            bits         - ndarray
                         Received bipolar ``(-1, +1)`` BCH symbols.
            dataLength   - int
                         Number of uncoded source symbols.
            codeLength   - int
                         Number of BCH-encoded symbols.
            feedbackPos  - ndarray
                         Feedback-register positions used by the encoder.
            threshold    - float
                         Minimum correlation required for successful
                         decoding.
        Returns
        -------
            flag         - int
                         Decoding status: one for success and zero for
                         failure.
            decodedBits  - ndarray
                         Decoded binary source symbols, or an empty array
                         when decoding fails.
        """
        bits = np.asarray(bits)
        # All hypotheses of the original symbols before BCH encoding.
        hypoBits = np.zeros((2**dataLength, dataLength), dtype=np.int8)
        # Correlation between each encoded hypothesis and received symbols.
        correValue = np.zeros(2**dataLength)

        #%% Decode each hypothesis ===========================================
        for hypothesis in range(2**dataLength):
            # Generate the current hypothesis in binary format.
            register = np.array(
                [int(bit) for bit in f"{hypothesis:0{dataLength}b}"],
                dtype=np.int8)
            hypoBits[hypothesis, :] = register
            # Convert polarity, 0 -> +1 and 1 -> -1, then reverse the register.
            register = np.where(register == 1, -1, 1)[::-1]
            # Symbols after BCH encoding of the current hypothesis.
            encoded = np.empty(codeLength, dtype=np.int8)

            # Encode the hypothesis using the BCH feedback register.
            for ind in range(codeLength):
                encoded[ind] = register[-1]
                # Exclusive-OR feedback is a product in bipolar format.
                feedback = np.prod(register[feedbackPos])
                # Shift the feedback register one position to the right.
                register = np.roll(register, 1)
                register[0] = feedback
            # Compute the hypothesis correlation value.
            correValue[hypothesis] = np.sum(encoded*bits)

        # Determine the decoded result from the maximum correlation.
        pos = int(np.argmax(correValue))
        if correValue[pos] >= threshold:
            return 1, hypoBits[pos, :].copy()
        return 0, np.empty(0, dtype=np.int8)

    @classmethod
    def BCH51_8Decoding(cls, bits):
        """Decode the BCH(51,8) symbols in B-CNAV1 subframe 1.

        A total of ``2**8`` hypotheses are tested to recover the eight
        original symbols.

        Args
        ----
            bits         - ndarray
                         Row vector of 51 bipolar ``(-1, +1)`` message
                         symbols.
        Returns
        -------
            flag         - int
                         One for successful decoding, otherwise zero.
            decodedBits  - ndarray
                         Eight decoded BCH source symbols.
        """
        # The correlation threshold determines decoding success and may be
        # reconfigured for a particular signal condition. The supplied array
        # contains the feedback-register positions of the BCH(51,8) encoder.
        return cls._bch_decode(
            bits, 8, 51, np.array([0, 3, 4, 5, 6, 7]), 50)

    @classmethod
    def BCH21_6Decoding(cls, bits):
        """Decode the BCH(21,6) symbols in B-CNAV1 subframe 1.

        A total of ``2**6`` hypotheses are tested to recover the six
        original symbols.

        Args
        ----
            bits         - ndarray
                         Row vector of 21 bipolar ``(-1, +1)`` message
                         symbols.
        Returns
        -------
            flag         - int
                         One for successful decoding, otherwise zero.
            decodedBits  - ndarray
                         Six decoded BCH source symbols.
        """
        # The correlation threshold determines decoding success and may be
        # reconfigured for a particular signal condition. The supplied array
        # contains the feedback-register positions of the BCH(21,6) encoder.
        return cls._bch_decode(
            bits, 6, 21, np.array([1, 3, 4, 5]), 20)

#%% LDPC decoder
class LDPCDecoder:
    """Decode the nonbinary LDPC codewords in B-CNAV1 subframes 2 and 3."""

    @staticmethod
    def getH_informationSF2():
        """Return the B-CNAV1 subframe-2 parity-check information."""
        H_idx = np.array([
            [11, 62, 102, 150], [4, 90, 131, 177], [13, 57, 135, 198],
            [6, 58, 106, 158], [10, 61, 101, 149], [17, 98, 151, 187],
            [21, 83, 119, 153], [40, 53, 132, 159], [34, 69, 125, 162],
            [23, 81, 117, 155], [34, 82, 130, 182], [2, 54, 102, 154],
            [36, 84, 120, 169], [28, 76, 124, 176], [29, 85, 112, 165],
            [25, 66, 121, 174], [35, 70, 126, 163], [35, 83, 131, 183],
            [33, 81, 129, 181], [30, 78, 126, 178], [1, 48, 143, 198],
            [41, 99, 147, 195], [37, 91, 139, 187], [16, 97, 150, 186],
            [38, 88, 136, 184], [9, 60, 100, 148], [47, 95, 138, 191],
            [24, 65, 120, 173], [8, 60, 108, 160], [39, 87, 123, 168],
            [46, 94, 137, 190], [31, 87, 114, 167], [19, 96, 149, 185],
            [23, 75, 119, 175], [24, 93, 111, 182], [1, 53, 101, 153],
            [18, 70, 114, 170], [25, 94, 108, 183], [36, 90, 138, 186],
            [45, 93, 136, 189], [37, 85, 121, 170], [32, 80, 128, 180],
            [10, 62, 110, 162], [41, 54, 133, 156], [9, 61, 109, 161],
            [40, 98, 146, 194], [31, 79, 127, 179], [5, 91, 128, 178],
            [11, 63, 111, 163], [47, 74, 141, 189], [0, 51, 142, 197],
            [51, 79, 146, 195], [6, 88, 129, 179], [44, 92, 139, 188],
            [15, 67, 105, 167], [14, 66, 104, 166], [2, 49, 140, 199],
            [16, 68, 112, 168], [42, 96, 144, 192], [20, 72, 116, 172],
            [46, 73, 140, 188], [26, 67, 122, 175], [39, 89, 137, 185],
            [33, 68, 124, 161], [27, 64, 123, 172], [3, 50, 141, 196],
            [0, 52, 100, 152], [19, 71, 115, 171], [20, 82, 118, 152],
            [26, 95, 109, 180], [18, 99, 148, 184], [3, 55, 103, 155],
            [30, 86, 113, 166], [32, 71, 127, 160], [49, 77, 144, 193],
            [22, 80, 116, 154], [44, 75, 142, 190], [7, 89, 130, 176],
            [4, 56, 104, 156], [50, 78, 145, 194], [7, 59, 107, 159],
            [12, 64, 106, 164], [14, 58, 132, 199], [8, 63, 103, 151],
            [17, 69, 113, 169], [13, 65, 107, 165], [29, 77, 125, 177],
            [21, 73, 117, 173], [12, 56, 134, 197], [28, 84, 115, 164],
            [48, 76, 147, 192], [43, 52, 135, 158], [15, 59, 133, 196],
            [38, 86, 122, 171], [45, 72, 143, 191], [5, 57, 105, 157],
            [22, 74, 118, 174], [43, 97, 145, 193], [42, 55, 134, 157],
            [27, 92, 110, 181],
        ], dtype=np.int16)
        H_ele = np.array([
            [35, 13, 51, 60], [1, 44, 53, 24], [15, 6, 1, 45],
            [30, 24, 1, 44], [24, 1, 44, 30], [33, 42, 14, 24],
            [30, 24, 1, 44], [1, 45, 15, 6], [24, 1, 44, 53],
            [44, 53, 24, 1], [1, 44, 53, 24], [45, 15, 6, 1],
            [35, 13, 18, 60], [6, 1, 45, 15], [1, 44, 30, 24],
            [30, 24, 1, 44], [1, 44, 30, 24], [30, 24, 1, 44],
            [35, 46, 56, 15], [44, 30, 24, 1], [1, 44, 53, 24],
            [45, 15, 6, 1], [51, 60, 35, 13], [44, 53, 24, 1],
            [24, 1, 44, 30], [1, 44, 53, 24], [1, 45, 15, 6],
            [44, 53, 24, 1], [24, 1, 44, 30], [24, 1, 44, 53],
            [33, 42, 14, 24], [24, 1, 44, 53], [1, 45, 15, 6],
            [24, 1, 44, 30], [30, 24, 1, 44], [27, 28, 30, 31],
            [30, 24, 1, 44], [45, 15, 6, 1], [53, 24, 1, 44],
            [44, 53, 24, 1], [1, 44, 30, 24], [38, 23, 22, 7],
            [53, 24, 1, 44], [5, 33, 42, 14], [5, 1, 45, 15],
            [1, 44, 30, 24], [6, 1, 45, 15], [29, 28, 30, 31],
            [44, 30, 24, 1], [24, 1, 44, 30], [1, 45, 15, 6],
            [35, 46, 56, 15], [24, 1, 44, 30], [45, 15, 6, 1],
            [24, 1, 44, 53], [45, 15, 6, 1], [1, 44, 30, 24],
            [42, 36, 12, 57], [1, 45, 15, 6], [1, 44, 30, 24],
            [53, 24, 1, 44], [1, 45, 15, 6], [30, 1, 44, 7],
            [24, 1, 44, 53], [53, 24, 1, 44], [1, 44, 30, 24],
            [44, 53, 24, 1], [6, 1, 45, 15], [54, 7, 38, 23],
            [53, 24, 1, 44], [44, 30, 24, 1], [1, 44, 53, 24],
            [6, 1, 45, 15], [38, 49, 11, 17], [1, 44, 53, 24],
            [45, 15, 6, 1], [6, 1, 45, 15], [1, 45, 15, 6],
            [17, 38, 49, 11], [30, 24, 1, 44], [1, 45, 15, 6],
            [57, 25, 9, 41], [6, 1, 45, 15], [1, 45, 15, 6],
            [53, 24, 1, 44], [24, 1, 44, 30], [26, 22, 14, 2],
            [6, 1, 45, 15], [30, 24, 1, 44], [44, 30, 24, 1],
            [41, 16, 29, 51], [1, 45, 15, 6], [24, 1, 44, 53],
            [1, 45, 15, 6], [44, 53, 24, 1], [1, 44, 53, 24],
            [42, 47, 37, 32], [24, 1, 44, 53], [44, 30, 24, 1],
            [53, 24, 1, 44],
        ], dtype=np.uint8)
        return H_idx, H_ele

    @staticmethod
    def getH_informationSF3():
        """Return the B-CNAV1 subframe-3 parity-check information."""
        H_idx = np.array([
            [14, 35, 56, 70], [1, 27, 45, 54], [2, 26, 61, 79],
            [12, 38, 52, 68], [17, 39, 44, 75], [17, 43, 67, 81],
            [0, 37, 70, 86], [18, 28, 67, 85], [1, 36, 71, 87],
            [12, 37, 57, 83], [7, 25, 51, 60], [11, 29, 55, 73],
            [23, 41, 63, 87], [9, 33, 59, 77], [23, 43, 58, 77],
            [9, 35, 49, 72], [22, 40, 62, 86], [5, 31, 49, 75],
            [0, 26, 44, 55], [16, 38, 45, 74], [6, 31, 51, 80],
            [3, 27, 60, 78], [13, 39, 53, 69], [2, 20, 46, 68],
            [4, 30, 48, 74], [19, 21, 63, 64], [19, 29, 66, 84],
            [3, 21, 47, 69], [4, 40, 53, 84], [10, 28, 54, 72],
            [8, 34, 48, 73], [15, 33, 47, 79], [14, 32, 46, 78],
            [15, 34, 57, 71], [6, 24, 50, 61], [22, 42, 59, 76],
            [11, 25, 65, 82], [13, 36, 56, 82], [10, 24, 64, 83],
            [5, 41, 52, 85], [7, 30, 50, 81], [8, 32, 58, 76],
            [16, 42, 66, 80], [18, 20, 62, 65],
        ], dtype=np.int16)
        H_ele = np.array([
            [30, 24, 1, 44], [51, 60, 35, 13], [6, 1, 45, 15],
            [24, 1, 44, 53], [1, 44, 53, 24], [13, 18, 60, 35],
            [44, 53, 24, 1], [15, 35, 46, 56], [44, 30, 24, 1],
            [33, 42, 14, 5], [45, 15, 6, 1], [24, 1, 44, 30],
            [18, 15, 32, 61], [45, 15, 6, 1], [44, 30, 24, 1],
            [61, 47, 20, 8], [45, 15, 6, 1], [39, 36, 34, 33],
            [53, 24, 1, 44], [15, 6, 1, 45], [34, 3, 55, 9],
            [1, 44, 30, 24], [40, 32, 61, 18], [15, 6, 1, 45],
            [1, 45, 15, 6], [34, 33, 45, 36], [53, 24, 1, 44],
            [24, 1, 44, 53], [44, 35, 31, 50], [1, 44, 53, 24],
            [30, 24, 1, 44], [44, 35, 61, 50], [6, 1, 45, 15],
            [53, 24, 1, 44], [30, 24, 1, 44], [1, 44, 53, 24],
            [55, 9, 34, 3], [15, 6, 1, 45], [37, 32, 52, 47],
            [12, 25, 36, 14], [24, 1, 44, 30], [2, 50, 22, 14],
            [15, 6, 1, 45], [1, 44, 53, 24],
        ], dtype=np.uint8)
        return H_idx, H_ele

    @staticmethod
    def init_table():
        """Initialize and return the GF(64) multiplication table."""
        N_GF = 6
        Q_GF = 2 ** N_GF
        GF_VEC = np.array((
            1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48, 35, 5, 10, 20, 40,
            19, 38, 15, 30, 60, 59, 53, 41, 17, 34, 7, 14, 28, 56, 51, 37,
            9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39,
            13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33,
        ), dtype=np.uint8)
        GF_POW = np.array((
            0, 0, 1, 6, 2, 12, 7, 26, 3, 32, 13, 35, 8, 48, 27, 18,
            4, 24, 33, 16, 14, 52, 36, 54, 9, 45, 49, 38, 28, 41, 19, 56,
            5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51, 37, 44, 55, 40,
            10, 61, 46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59, 57, 58,
        ), dtype=np.uint8)
        GF_MUL = np.zeros((Q_GF, Q_GF), dtype=np.uint8)
        for first in range(1, Q_GF):
            for second in range(1, Q_GF):
                power = (int(GF_POW[first])
                         + int(GF_POW[second])) % (Q_GF - 1)
                GF_MUL[first, second] = GF_VEC[power]
        return GF_MUL

    @staticmethod
    def graph_edge(H_idx, H_ele):
        """Expand sparse parity-check rows into Tanner-graph edge arrays.

        Args
        ----
            H_idx       - ndarray
                        Variable-node indices of the nonzero elements.
            H_ele       - ndarray
                        GF(64) coefficients of the nonzero elements.
        Returns
        -------
            ie          - ndarray
                        Check-node index of each Tanner-graph edge.
            je          - ndarray
                        Variable-node index of each edge.
            he          - ndarray
                        GF(64) coefficient of each edge.
            ne          - int
                        Total number of graph edges.
        """
        # Count the nonzero elements and fill the edge lists.
        ie = np.repeat(np.arange(H_idx.shape[0]), H_idx.shape[1])
        je = np.asarray(H_idx).reshape(-1)
        he = np.asarray(H_ele).reshape(-1)
        return ie, je, he, ie.size

    @staticmethod
    def _noiseEst(I_P, Q_P, colH):
        """Estimate SNR blindly and normalize the LDPC soft symbols.

        The moment method (M2M4) operates on the squared magnitude
        ``Z = I**2 + Q**2``. Prompt-correlation values are normalized for
        soft-decision decoding and formatted as six bits per GF(64) symbol.

        Args
        ----
            I_P         - ndarray
                        Prompt in-phase soft symbols.
            Q_P         - ndarray
                        Prompt quadrature soft/noise reference symbols.
            colH        - int
                        Number of GF(64) symbols in the LDPC codeword.
        Returns
        -------
            normIp      - ndarray
                        Normalized prompt values arranged as
                        ``colH x 6``.
            normSigma2  - float
                        Estimated single-branch noise variance normalized
                        by signal power.
        """
        #%% BDS-3 B1C LDPC ===================================================
        bitsPerSymbol = 6
        tiny = 1e-12
        I = np.asarray(I_P).reshape(-1)
        Q = np.asarray(Q_P).reshape(-1)

        # Squared magnitude used by the M2M4 moment estimator.
        Z = I**2+Q**2
        meanZ = np.mean(Z)
        # Population variances match the moment definitions used by MATLAB.
        varZ = np.var(Z)
        qVar = np.var(Q)

        # Use the complex-I/Q moment relation unless Q degenerates to zero.
        if qVar > tiny:
            # Complex-noise case:
            # meanZ = A^2 + 2*sigma^2 and
            # varZ = 4*A^2*sigma^2 + 4*sigma^4.
            Pav = np.sqrt(max(meanZ**2-varZ, 0))
            estNoiseVar = 0.5*max(meanZ-Pav, 0)
        else:
            # Real-noise case:
            # meanZ = A^2 + sigma^2 and
            # varZ = 4*A^2*sigma^2 + 2*sigma^4.
            Pav = np.sqrt(max(meanZ**2-varZ/2, 0))
            estNoiseVar = max(meanZ-Pav, 0)

        #=== Normalize for the decoder =======================================
        # Pav is the estimated signal power; its square root is the amplitude.
        estSignalAmp = np.sqrt(max(Pav, 0))
        # Avoid division by zero before normalizing the decoder input.
        estSignalAmp = max(estSignalAmp, tiny)
        normIp = np.asarray(I_P)/estSignalAmp
        normSigma2 = estNoiseVar/(estSignalAmp**2)
        #=== Format the decoder input ========================================
        return normIp.reshape(colH, bitsPerSymbol), normSigma2

    @classmethod
    def noiseEstSF2(cls, I_P, Q_P):
        """Estimate noise and normalize the 1200 B-CNAV1 SF2 soft bits.

        Args
        ----
            I_P         - ndarray
                        SF2 prompt in-phase soft symbols.
            Q_P         - ndarray
                        SF2 prompt quadrature reference symbols.
        Returns
        -------
            normIp      - ndarray
                        Normalized ``200 x 6`` decoder input.
            normSigma2  - float
                        Normalized noise variance.
        """
        return cls._noiseEst(I_P, Q_P, 200)

    @classmethod
    def noiseEstSF3(cls, I_P, Q_P):
        """Estimate noise and normalize the 528 B-CNAV1 SF3 soft bits.

        Args
        ----
            I_P         - ndarray
                        SF3 prompt in-phase soft symbols.
            Q_P         - ndarray
                        SF3 prompt quadrature reference symbols.
        Returns
        -------
            normIp      - ndarray
                        Normalized ``88 x 6`` decoder input.
            normSigma2  - float
                        Normalized noise variance.
        """
        return cls._noiseEst(I_P, Q_P, 88)

    @staticmethod
    def _initLlr(receivedSignal, sigmaSquared):
        """Calculate initial symbol metrics for nonbinary LDPC decoding.

        A nonnegative cost table is produced, where a lower value indicates
        a more probable GF(64) symbol and zero represents the hard-decision
        candidate.

        Args
        ----
            receivedSignal - ndarray
                           Normalized received bits, six per GF(64) symbol.
            sigmaSquared   - float
                           Normalized noise variance.
        Returns
        -------
            llrMatrix      - ndarray
                           Initial GF(64) symbol-cost matrix.
        """
        gfBits = 6
        gfSize = 64
        #--- Precompute the GF bit lookup table ------------------------------
        gfBitLut = ((np.arange(gfSize)[:, None]
                     >> np.arange(gfBits-1, -1, -1)) & 1).astype(np.int8)
        #--- Make hard decisions on the received bits ------------------------
        # Positive prompt correlation represents bit 1; nonpositive
        # correlation represents bit 0.
        hardDecisionBits = receivedSignal > 0
        #--- Build symbol metrics --------------------------------------------
        # Reliability scaling factor is 2/sigma^2.
        scalingFactor = np.divide(2.0, sigmaSquared)
        llrMatrix = np.empty((receivedSignal.shape[0], gfSize))

        for ind in range(receivedSignal.shape[0]):
            # Reliability weights of the six bits in the current symbol.
            bitReliabilities = (
                np.abs(receivedSignal[ind, :])*scalingFactor)
            # Candidate bits that disagree with the hard decision.
            bitMismatch = np.logical_xor(
                gfBitLut, hardDecisionBits[ind, :])
            # Sum the reliability of all mismatches to obtain symbol cost.
            llrMatrix[ind, :] = np.sum(
                bitReliabilities*bitMismatch, axis=1)
        return llrMatrix

    @classmethod
    def initLlrSF2(cls, receivedSignal, sigmaSquared):
        """Build the ``200 x 64`` initial symbol metrics for SF2.

        Args
        ----
            receivedSignal - ndarray
                           Normalized ``200 x 6`` SF2 soft bits.
            sigmaSquared   - float
                           Normalized noise variance.
        Returns
        -------
            llrMatrix      - ndarray
                           ``200 x 64`` nonbinary symbol-cost matrix.
        """
        return cls._initLlr(receivedSignal, sigmaSquared)

    @classmethod
    def initLlrSF3(cls, receivedSignal, sigmaSquared):
        """Build the ``88 x 64`` initial symbol metrics for SF3.

        Args
        ----
            receivedSignal - ndarray
                           Normalized ``88 x 6`` SF3 soft bits.
            sigmaSquared   - float
                           Normalized noise variance.
        Returns
        -------
            llrMatrix      - ndarray
                           ``88 x 64`` nonbinary symbol-cost matrix.
        """
        return cls._initLlr(receivedSignal, sigmaSquared)

    @staticmethod
    def permute_V2C(h, V2C, GF_MUL):
        """Permute a variable-to-check message by a GF(64) coefficient.

        Args
        ----
            h           - int
                        Nonzero parity-check coefficient.
            V2C         - ndarray
                        Variable-to-check cost message over GF(64).
        Returns
        -------
            V2C_p       - ndarray
                        Permuted variable-to-check message.
        """
        V2C_p = np.zeros(64, dtype=np.float32)
        for ind in range(64):
            # Map array indices to zero-based GF elements.
            V2C_p[GF_MUL[h, ind]] = V2C[ind]
        return V2C_p

    @staticmethod
    def permute_C2V(h, C2V, GF_MUL):
        """Permute a check-to-variable message by a GF(64) coefficient.

        Args
        ----
            h           - int
                        Nonzero parity-check coefficient.
            C2V         - ndarray
                        Check-to-variable cost message over GF(64).
        Returns
        -------
            C2V_p       - ndarray
                        Permuted check-to-variable message.
        """
        C2V_p = np.zeros(64, dtype=np.float32)
        for ind in range(64):
            C2V_p[ind] = C2V[GF_MUL[h, ind]]
        return C2V_p

    @staticmethod
    def ext_min_sum(L1, L2):
        """Apply the order-four extended min-sum check-node operation.

        Args
        ----
            L1          - ndarray or None
                        First GF(64) cost message. ``None`` initializes the
                        recursive check-node combination.
            L2          - ndarray
                        Second GF(64) cost message.
        Returns
        -------
            Ls          - ndarray
                        Combined GF(64) cost message.
        """
        NM_EMS = 4
        if L1 is None:
            return np.asarray(L2).copy()

        idx1 = np.argsort(L1, kind="stable")
        idx2 = np.argsort(L2, kind="stable")
        maxL = L1[idx1[NM_EMS-1]]+L2[idx2[NM_EMS-1]]
        Ls = np.full(64, maxL, dtype=np.float32)
        for i in range(NM_EMS):
            for j in range(NM_EMS):
                # XOR combines the two zero-based GF(64) symbol indices.
                xor_idx = np.bitwise_xor(idx1[i], idx2[j])
                value = L1[idx1[i]]+L2[idx2[j]]
                if value < Ls[xor_idx]:
                    Ls[xor_idx] = value
        return Ls

    @staticmethod
    def _syndromeIsZero(code, H_idx, H_ele, GF_MUL):
        """Test a GF(64) codeword against a sparse parity-check matrix.

        Args
        ----
            code        - ndarray
                        Candidate GF(64) codeword.
            H_idx       - ndarray
                        Nonzero parity-check column positions.
            H_ele       - ndarray
                        Nonzero parity-check GF(64) coefficients.
        Returns
        -------
            valid       - bool
                        True when every parity-check syndrome equals zero.
        """
        for row in range(H_idx.shape[0]):
            syndrome = 0
            for edge in range(H_idx.shape[1]):
                syndrome ^= int(GF_MUL[
                    int(code[H_idx[row, edge]]), int(H_ele[row, edge])])
            if syndrome != 0:
                return False
        return True

    @classmethod
    def _emsDecode(cls, L, H_idx, H_ele):
        """Decode a systematic nonbinary LDPC word with the EMS algorithm.

        Args
        ----
            L           - ndarray
                        Initial GF(64) symbol-cost matrix.
            H_idx       - ndarray
                        Nonzero parity-check column positions.
            H_ele       - ndarray
                        Nonzero parity-check GF(64) coefficients.
        Returns
        -------
            code        - ndarray
                        Decoded GF(64) codeword.

        Notes
        -----
        At most 15 iterations are performed. Decoding terminates early when
        every parity-check syndrome equals zero.
        """
        GF_MUL = cls.init_table()

        # Six bits form each of the 64 elements in GF(64).
        Q_GF = 64
        # Maximum number of EMS decoding iterations.
        MAX_ITER = 15
        n = L.shape[0]
        L = np.asarray(L).copy()
        code = np.argmin(L, axis=1)

        # Tanner-graph edges and check/variable messages.
        ie, je, he, ne = cls.graph_edge(H_idx, H_ele)
        V2C = np.zeros((ne, Q_GF), dtype=np.float32)
        C2V = np.zeros((ne, Q_GF), dtype=np.float32)

        checkEdges = [np.flatnonzero(ie == row)
                      for row in range(H_idx.shape[0])]
        variableEdges = [np.flatnonzero(je == col) for col in range(n)]

        # Initialize variable-to-check messages from channel costs.
        for edge in range(ne):
            V2C[edge, :] = cls.permute_V2C(
                int(he[edge]), L[int(je[edge]), :], GF_MUL)

        for _ in range(MAX_ITER):
            # Stop when the current GF(64) codeword satisfies all checks.
            if cls._syndromeIsZero(code, H_idx, H_ele, GF_MUL):
                break

            # Update check nodes.
            for edge in range(ne):
                Ls = None
                for other in checkEdges[int(ie[edge])]:
                    if other != edge:
                        Ls = cls.ext_min_sum(Ls, V2C[other, :])
                Ls = Ls-np.min(Ls)
                C2V[edge, :] = cls.permute_C2V(int(he[edge]), Ls, GF_MUL)

            # Update variable nodes.
            for edge in range(ne):
                variable = int(je[edge])
                Ls = L[variable, :].copy()
                for other in variableEdges[variable]:
                    if other != edge:
                        Ls += C2V[other, :]
                Ls -= np.min(Ls)
                V2C[edge, :] = cls.permute_V2C(int(he[edge]), Ls, GF_MUL)

            # Update posterior costs and hard GF(64) decisions.
            for variable in range(n):
                for edge in variableEdges[variable]:
                    L[variable, :] += C2V[edge, :]
                L[variable, :] -= np.min(L[variable, :])
                code[variable] = np.argmin(L[variable, :])
        return code

    @classmethod
    def B1C_EMS_decode_SF2(cls, L):
        """Decode the 200-symbol B-CNAV1 SF2 LDPC codeword.

        Args
        ----
            L           - ndarray
                        ``200 x 64`` initial symbol-cost matrix.
        Returns
        -------
            code        - ndarray
                        Decoded 200-symbol GF(64) codeword.
        """
        H_idx, H_ele = cls.getH_informationSF2()
        return cls._emsDecode(L, H_idx, H_ele)

    @classmethod
    def B1C_EMS_decode_SF3(cls, L):
        """Decode the 88-symbol B-CNAV1 SF3 LDPC codeword.

        Args
        ----
            L           - ndarray
                        ``88 x 64`` initial symbol-cost matrix.
        Returns
        -------
            code        - ndarray
                        Decoded 88-symbol GF(64) codeword.
        """
        H_idx, H_ele = cls.getH_informationSF3()
        return cls._emsDecode(L, H_idx, H_ele)

    @classmethod
    def B1CLDPCDecoder(cls, navBits_soft_I, navBits_soft_Q):
        """Decode one 1800-symbol BDS-3 B1C B-CNAV1 frame.

        Args
        ----
            navBits_soft_I - ndarray
                           Data-channel 1800-symbol B-CNAV1 soft sequence.
            navBits_soft_Q - ndarray
                           Q-branch 1800-symbol soft/noise reference
                           sequence.
        Returns
        -------
            decodedNavBits - ndarray or None
                           Decoded 878-bit B-CNAV1 message. ``None`` is
                           returned when BCH decoding fails.

        Notes
        -----
        B-CNAV1 subframe 1 contains BCH(21,6) in symbols 1--21 and
        BCH(51,8) in symbols 22--72. Symbols 73--1800 contain interleaved
        LDPC-coded SF2 and SF3 data. SF2 contains 1200 bits, or 200 GF(64)
        symbols, and SF3 contains 528 bits, or 88 GF(64) symbols. The decoded
        output comprises 14 SF1 bits, 600 SF2 bits, and 264 SF3 bits.
        """
        navBits_soft_I = np.asarray(navBits_soft_I).reshape(-1)
        navBits_soft_Q = np.asarray(navBits_soft_Q).reshape(-1)
        if navBits_soft_I.size < 1800 or navBits_soft_Q.size < 1800:
            return None

        navBits_soft_I = navBits_soft_I[:1800].copy()
        navBits_soft_Q = navBits_soft_Q[:1800].copy()
        #--- Hard decisions for BCH decoding and polarity determination ------
        bits_hard = (navBits_soft_I > 0).astype(np.int8)
        polarity = 1

        #=== SF1 BCH(21,6) decoding ==========================================
        flag, decodedBits = BCHDecoder.BCH21_6Decoding(1-2*bits_hard[:21])
        if flag == 0:
            # If BCH(21,6) fails, try the inverse data polarity.
            bits_hard = 1-bits_hard
            polarity = -1
            flag, decodedBits = BCHDecoder.BCH21_6Decoding(
                1-2*bits_hard[:21])
            if flag == 0:
                return None

        # Apply the detected polarity to the soft information.
        navBits_soft_I *= polarity
        navBits_soft_Q *= polarity
        decodedNavBits = np.zeros(878, dtype=np.int8)
        decodedNavBits[:6] = decodedBits

        #=== SF1 BCH(51,8) decoding ==========================================
        flag, decodedBits = BCHDecoder.BCH51_8Decoding(1-2*bits_hard[21:72])
        if flag == 0:
            return None
        decodedNavBits[6:14] = decodedBits

        #=== Deinterleave SF2 and SF3 ========================================
        # MATLAB reshape is column-major. After the selected rows are
        # transposed and reshaped, NumPy's default row-major flattening below
        # yields the same SF2/SF3 symbol order.
        tempBits_I = navBits_soft_I[72:].reshape(36, 48, order="F")
        tempBits_Q = navBits_soft_Q[72:].reshape(36, 48, order="F")
        Frame3Colum = np.arange(2, 35, 3)
        Frame2Colum = np.setdiff1d(np.arange(36), Frame3Colum)
        Frame2_soft_I = tempBits_I[Frame2Colum, :].reshape(-1)
        Frame2_soft_Q = tempBits_Q[Frame2Colum, :].reshape(-1)
        Frame3_soft_I = tempBits_I[Frame3Colum, :].reshape(-1)
        Frame3_soft_Q = tempBits_Q[Frame3Colum, :].reshape(-1)
        Frame2_hard = Frame2_soft_I > 0
        Frame3_hard = Frame3_soft_I > 0

        # Keep the deinterleaved systematic hard decisions as a fallback.
        # Both LDPC codes are systematic, so their first information bits
        # remain usable if EMS does not converge to a valid codeword.

        #=== Generate SF2 LLR soft information ===============================
        normIp, normSigma2 = cls.noiseEstSF2(
            Frame2_soft_I, Frame2_soft_Q)
        Symbol_LLR = cls.initLlrSF2(normIp, normSigma2)

        #=== Decode SF2 with EMS =============================================
        H_idx, H_ele = cls.getH_informationSF2()
        GF_MUL = cls.init_table()
        decoded_syms = cls.B1C_EMS_decode_SF2(Symbol_LLR)
        if cls._syndromeIsZero(decoded_syms, H_idx, H_ele, GF_MUL):
            # The first 100 GF(64) symbols are information symbols.
            decodedNav_SF2 = ((np.asarray(decoded_syms[:100])[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.int8).reshape(-1)
        else:
            # MATLAB retains systematic hard bits if EMS does not converge.
            decodedNav_SF2 = Frame2_hard[:600].astype(np.int8)

        #=== Generate SF3 LLR soft information ===============================
        normIp, normSigma2 = cls.noiseEstSF3(
            Frame3_soft_I, Frame3_soft_Q)
        Symbol_LLR = cls.initLlrSF3(normIp, normSigma2)

        #=== Decode SF3 with EMS =============================================
        H_idx, H_ele = cls.getH_informationSF3()
        GF_MUL = cls.init_table()
        decoded_syms = cls.B1C_EMS_decode_SF3(Symbol_LLR)
        if cls._syndromeIsZero(decoded_syms, H_idx, H_ele, GF_MUL):
            # The first 44 GF(64) symbols are information symbols.
            decodedNav_SF3 = ((np.asarray(decoded_syms[:44])[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.int8).reshape(-1)
        else:
            decodedNav_SF3 = Frame3_hard[:264].astype(np.int8)

        #=== Assemble the decoded B-CNAV1 message ============================
        decodedNavBits[14:614] = decodedNav_SF2
        decodedNavBits[614:878] = decodedNav_SF3
        return decodedNavBits


__all__ = ["BCHDecoder", "LDPCDecoder"]
