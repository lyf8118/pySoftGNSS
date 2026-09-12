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
Decode the non-binary LDPC channel code used by the BDS-3 B2a B-CNAV2
navigation message.

"""

import numpy as np


#%% LDPC decoder
class LDPCDecoder:
    """Decode the GF(64) B-CNAV2 LDPC code with extended min-sum."""

    @staticmethod
    def getH_information():
        """Return the B-CNAV2 GF(64) parity-check information."""
        H_idx = np.array([
            [19, 46, 49, 76], [22, 41, 68, 94], [8, 40, 60, 87],
            [5, 38, 70, 89], [23, 37, 58, 83], [6, 30, 54, 76],
            [1, 25, 49, 79], [8, 36, 51, 84], [11, 33, 59, 81],
            [16, 31, 65, 73], [21, 42, 75, 92], [4, 39, 71, 88],
            [5, 29, 53, 71], [20, 44, 54, 75], [15, 26, 66, 81],
            [16, 34, 64, 92], [15, 43, 56, 91], [14, 27, 67, 80],
            [12, 45, 69, 79], [3, 38, 56, 86], [20, 43, 74, 93],
            [4, 28, 52, 70], [7, 31, 55, 77], [13, 44, 68, 78],
            [17, 30, 64, 72], [9, 41, 61, 86], [19, 24, 67, 95],
            [21, 45, 55, 74], [18, 47, 48, 77], [17, 35, 65, 93],
            [18, 25, 66, 94], [0, 29, 62, 85], [13, 32, 63, 91],
            [1, 28, 63, 84], [9, 37, 50, 85], [3, 27, 51, 73],
            [22, 36, 59, 82], [6, 47, 60, 89], [2, 26, 50, 72],
            [0, 24, 48, 78], [14, 42, 57, 90], [7, 46, 61, 88],
            [23, 40, 69, 95], [2, 39, 57, 87], [11, 35, 52, 83],
            [12, 33, 62, 90], [10, 34, 53, 82], [10, 32, 58, 80],
        ], dtype=np.int32)
        H_ele = np.array([
            [1, 45, 15, 6], [18, 15, 32, 61], [24, 1, 44, 53],
            [44, 53, 24, 1], [45, 15, 6, 1], [32, 61, 18, 40],
            [30, 24, 1, 44], [24, 1, 44, 53], [1, 45, 15, 6],
            [33, 45, 36, 34], [44, 35, 31, 50], [24, 1, 44, 30],
            [1, 44, 53, 24], [3, 55, 9, 34], [30, 24, 1, 44],
            [39, 36, 34, 33], [6, 1, 45, 15], [1, 45, 15, 6],
            [15, 46, 45, 44], [15, 6, 1, 45], [44, 53, 24, 1],
            [6, 1, 45, 15], [26, 27, 37, 5], [24, 1, 44, 30],
            [45, 15, 6, 1], [35, 31, 50, 44], [32, 42, 47, 37],
            [44, 53, 24, 1], [24, 1, 44, 53], [22, 14, 2, 50],
            [45, 15, 6, 1], [53, 24, 1, 44], [57, 25, 9, 41],
            [6, 1, 45, 15], [24, 1, 44, 30], [1, 44, 53, 24],
            [30, 24, 1, 44], [45, 15, 6, 1], [6, 1, 45, 15],
            [44, 53, 24, 1], [9, 41, 57, 58], [24, 1, 44, 30],
            [1, 44, 30, 24], [7, 38, 23, 54], [35, 13, 51, 60],
            [6, 1, 45, 15], [33, 42, 14, 5], [1, 44, 30, 24],
        ], dtype=np.uint8)
        return H_idx, H_ele

    @staticmethod
    def init_table():
        """Initialize and return the GF(64) multiplication table."""
        N_GF = 6                  # Number of GF(q) bits.
        Q_GF = 2 ** N_GF          # Number of GF(q) elements.
        GF_VEC = np.array((
            1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48, 35, 5, 10, 20, 40,
            19, 38, 15, 30, 60, 59, 53, 41, 17, 34, 7, 14, 28, 56, 51, 37,
            9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39,
            13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33
        ), dtype=np.uint8)
        GF_POW = np.array((
            0, 0, 1, 6, 2, 12, 7, 26, 3, 32, 13, 35, 8, 48, 27, 18,
            4, 24, 33, 16, 14, 52, 36, 54, 9, 45, 49, 38, 28, 41, 19, 56,
            5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51, 37, 44, 55, 40,
            10, 61, 46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59, 57, 58
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
        """Generate the Tanner-graph edge lists.

        Args
        ----
            H_idx   - numpy.ndarray
                    Column indices of the nonzero parity-check elements.
            H_ele   - numpy.ndarray
                    Nonzero GF(64) parity-check elements.

        Returns
        -------
            ie      - numpy.ndarray
                    Check-node index of each edge.
            je      - numpy.ndarray
                    Variable-node index of each edge.
            he      - numpy.ndarray
                    GF(64) coefficient of each edge.
            ne      - int
                    Total number of Tanner-graph edges.
        """
        # Count the total number of nonzero elements.
        ne = H_idx.size

        # Fill the edge lists.
        ie = np.repeat(np.arange(H_idx.shape[0]), H_idx.shape[1])
        je = H_idx.reshape(-1)
        he = H_ele.reshape(-1)
        return ie, je, he, ne

    @staticmethod
    def noiseEst(I_P, Q_P):
        """Estimate SNR and normalize the prompt values.

        Performs blind signal-to-noise-ratio estimation using the moment
        method (M2M4) based on the squared magnitude Z = I**2 + Q**2.

        Args
        ----
            I_P         - numpy.ndarray
                        Prompt I-branch soft sequence.
            Q_P         - numpy.ndarray
                        Prompt Q-branch noise-reference sequence.

        Returns
        -------
            normIp      - numpy.ndarray
                        Normalized 96-by-6 decoder input.
            normSigma2  - float
                        Normalized single-branch noise variance.
        """
        #%% BDS-3 B2a LDPC ===================================================
        colH = 96
        bitsPerSymbol = 6

        tiny = 1e-12
        I = np.asarray(I_P, dtype=np.float64).reshape(-1)
        Q = np.asarray(Q_P, dtype=np.float64).reshape(-1)

        Z = I**2 + Q**2
        meanZ = np.mean(Z)
        varZ = np.var(Z)          # Population variance for moment matching.
        qVar = np.var(Q)

        if qVar > tiny:
            # Complex-noise / IQ-noise case:
            # meanZ = A^2 + 2*sigma^2;
            # varZ = 4*A^2*sigma^2 + 4*sigma^4.
            Pav = np.sqrt(max(meanZ**2 - varZ, 0.0))
            estNoiseVar = 0.5 * max(meanZ - Pav, 0.0)
        else:
            # Q degenerates to zero (real case):
            # meanZ = A^2 + sigma^2;
            # varZ = 4*A^2*sigma^2 + 2*sigma^4.
            Pav = np.sqrt(max(meanZ**2 - varZ / 2, 0.0))
            estNoiseVar = max(meanZ - Pav, 0.0)

        estSignalAmp = np.sqrt(max(Pav, 0.0))
        estSignalAmp = max(estSignalAmp, tiny)

        #=== Normalize for the decoder =======================================
        normIp = I_P / estSignalAmp
        normSigma2 = estNoiseVar / estSignalAmp**2

        #=== Format the decoder input ========================================
        normIp = np.asarray(normIp).reshape(colH, bitsPerSymbol)
        return normIp, normSigma2

    @staticmethod
    def initLlr(receivedSignal, sigmaSquared):
        """Calculate initial symbol metrics for nonbinary LDPC decoding.

        The result is a nonnegative table in which a lower value indicates
        higher probability and zero identifies the hard-decision candidate.

        Args
        ----
            receivedSignal - ndarray
                           Normalized bit soft values arranged as ``96 x 6``.
            sigmaSquared   - float
                           Normalized single-branch noise variance.
        Returns
        -------
            llrMatrix      - ndarray
                           Initial ``96 x 64`` GF(64) symbol metrics.
        """
        #%% BDS-3 B2a B-CNAV2 LDPC ===========================================
        gfBits = 6
        gfSize = 64
        colH = 96

        # Preallocate output matrix.
        llrMatrix = np.empty((colH, gfSize), dtype=np.float32)

        #--- Precompute the GF bit lookup table -------------------------------
        gfBitLut = ((np.arange(gfSize)[:, None]
                     >> np.arange(gfBits - 1, -1, -1)) & 1).astype(np.uint8)

        #--- Make hard decisions on the received bits -------------------------
        # BPSK mapping rule: positive -> 0, negative -> 1.
        hardDecisionBits = receivedSignal < 0

        #--- Build symbol metrics ---------------------------------------------
        # Reliability scaling factor: 2 / sigma^2.
        scalingFactor = 2 / max(sigmaSquared, 1e-12)
        for symbolNr in range(colH):
            bitReliabilities = (np.abs(receivedSignal[symbolNr])
                                * scalingFactor)
            bitMismatch = gfBitLut != hardDecisionBits[symbolNr]
            llrMatrix[symbolNr] = np.sum(
                bitReliabilities * bitMismatch, axis=1)
        return llrMatrix

    @staticmethod
    def permute_V2C(h, V2C, GF_MUL):
        """Permute a variable-to-check message by a GF(64) coefficient.

        Args
        ----
            h       - int
                    Nonzero GF(64) parity-check coefficient.
            V2C     - ndarray
                    Variable-to-check message.
            GF_MUL  - ndarray
                    GF(64) multiplication table.
        """
        Q_GF = 64
        V2C_p = np.zeros(Q_GF, dtype=np.float32)
        V2C_p[GF_MUL[h]] = V2C
        return V2C_p

    @staticmethod
    def permute_C2V(h, C2V, GF_MUL):
        """Permute a check-to-variable message by a GF(64) coefficient.

        Args
        ----
            h       - int
                    Nonzero GF(64) parity-check coefficient.
            C2V     - ndarray
                    Check-to-variable message.
            GF_MUL  - ndarray
                    GF(64) multiplication table.
        """
        return C2V[GF_MUL[h]].astype(np.float32, copy=False)

    @staticmethod
    def ext_min_sum(L1, L2):
        """Apply the order-four extended min-sum check-node operation.

        Args
        ----
            L1, L2  - ndarray
                    Input GF(64) cost vectors.
        Returns
        -------
            Ls      - ndarray
                    Output GF(64) cost vector.
        """
        NM_EMS = 4
        Q_GF = 64
        if L1 is None:
            return L2.copy()

        idx1 = np.argsort(L1)[:NM_EMS]
        idx2 = np.argsort(L2)[:NM_EMS]
        maxL = L1[idx1[-1]] + L2[idx2[-1]]
        Ls = np.full(Q_GF, maxL, dtype=np.float32)

        for firstIdx in idx1:
            for secondIdx in idx2:
                xor_idx = firstIdx ^ secondIdx
                cost = L1[firstIdx] + L2[secondIdx]
                if cost < Ls[xor_idx]:
                    Ls[xor_idx] = cost
        return Ls

    @staticmethod
    def _syndromeIsZero(code, H_idx, H_ele, GF_MUL):
        """Test a GF(64) codeword against a sparse parity-check matrix.

        Args
        ----
            code    - ndarray
                    GF(64) hard-decision codeword.
            H_idx, H_ele - ndarray
                    Sparse parity-check column indices and coefficients.
            GF_MUL  - ndarray
                    GF(64) multiplication table.
        Returns
        -------
            bool
                    True when every syndrome symbol is zero.
        """
        for row in range(H_idx.shape[0]):
            syndrome = 0
            for edge in range(H_idx.shape[1]):
                syndrome ^= int(GF_MUL[
                    code[H_idx[row, edge]], H_ele[row, edge]])
            if syndrome:
                return False
        return True

    @classmethod
    def B2a_EMS_decode(cls, L):
        """Decode 96 GF(64) B-CNAV2 symbols with the EMS algorithm.

        Args
        ----
            L       - ndarray
                    Initial ``96 x 64`` symbol metrics.
        Returns
        -------
            code    - ndarray
                    Decoded GF(64) codeword.
        """
        GF_MUL = cls.init_table()
        H_idx, H_ele = cls.getH_information()

        N_GF = 6                  # Number of GF(q) bits.
        Q_GF = 2 ** N_GF         # Number of GF(q) elements.
        MAX_ITER = 15            # Maximum number of decoding iterations.
        n = 96

        # Minimum-cost symbol index in each row.
        code = np.argmin(L, axis=1).astype(np.uint8)

        # Tanner-graph edges.
        ie, je, he, ne = cls.graph_edge(H_idx, H_ele)
        V2C = np.empty((ne, Q_GF), dtype=np.float32)
        C2V = np.zeros_like(V2C)

        for edge in range(ne):
            V2C[edge] = cls.permute_V2C(
                he[edge], L[je[edge]], GF_MUL)

        for _ in range(MAX_ITER):
            # Check whether the decoded codeword satisfies the parity
            # equations.
            if cls._syndromeIsZero(code, H_idx, H_ele, GF_MUL):
                break

            # Update check nodes.
            for edge in range(ne):
                Ls = None
                for neighbour in np.flatnonzero(ie == ie[edge]):
                    if neighbour != edge:
                        Ls = cls.ext_min_sum(Ls, V2C[neighbour])
                Ls -= np.min(Ls)
                C2V[edge] = cls.permute_C2V(he[edge], Ls, GF_MUL)

            # Update variable nodes.
            for edge in range(ne):
                Ls = L[je[edge]].copy()
                for neighbour in np.flatnonzero(je == je[edge]):
                    if neighbour != edge:
                        Ls += C2V[neighbour]
                Ls -= np.min(Ls)
                V2C[edge] = cls.permute_V2C(he[edge], Ls, GF_MUL)

            # Update LLRs and GF(q) symbols.
            for symbolNr in range(n):
                for edge in np.flatnonzero(je == symbolNr):
                    L[symbolNr] += C2V[edge]
                L[symbolNr] -= np.min(L[symbolNr])
                code[symbolNr] = np.argmin(L[symbolNr])
        return code

    @classmethod
    def B2aLDPCDecoder(cls, navBits_soft_I, navBits_soft_Q):
        """Decode one B-CNAV2 frame into 288 information bits.

        Args
        ----
            navBits_soft_I - numpy.ndarray
                            600-bit B-CNAV2 soft sequence after
                            secondary-code despreading.
            navBits_soft_Q - numpy.ndarray
                            600-bit Q-branch soft/noise-reference sequence.

        Returns
        -------
            decodedNavBits - numpy.ndarray
                            288 decoded navigation bits for CRC and
                            ephemeris decoding.
        """
        #=== Generate LLR soft-information matrix =============================
        # B-CNAV2 frame before LDPC decoding:
        #   1  - 24   : preamble;
        #   25 - 600  : LDPC-coded payload, 576 bits = 96 GF(64) symbols.
        payload_I = np.asarray(navBits_soft_I).reshape(-1)[24:600]
        payload_Q = np.asarray(navBits_soft_Q).reshape(-1)[24:600]

        # Blind SNR estimation and soft-information normalization.
        normIp, normSigma2 = cls.noiseEst(payload_I, payload_Q)

        # Generate the 96-by-64 LLR cost matrix required for nonbinary LDPC
        # decoding.
        Symbol_LLR = cls.initLlr(normIp, normSigma2)

        #=== EMS decoding =====================================================
        decoded_symbols_96 = cls.B2a_EMS_decode(Symbol_LLR).reshape(-1)

        # The first 48 GF(64) symbols carry 288 information bits.
        decoded_symbols_48 = decoded_symbols_96[:48]

        #=== Map decoded symbols back to a bit stream =========================
        decodedNavBits = ((decoded_symbols_48[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.uint8).reshape(-1)
        return decodedNavBits


__all__ = ["LDPCDecoder"]
