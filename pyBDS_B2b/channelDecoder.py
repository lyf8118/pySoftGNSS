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
Decode the non-binary LDPC channel code used by the BDS-3 B2b B-CNAV3
navigation message.

"""

import numpy as np


#%% LDPC decoder
class LDPCDecoder:
    """Decode the GF(64) B-CNAV3 LDPC code with extended min-sum."""

    @staticmethod
    def getH_information():
        """Return the B-CNAV3 GF(64) parity-check information."""
        H_idx = np.array([
            [19, 67, 109, 130], [26, 71, 104, 132], [13, 42, 101, 146],
            [23, 61, 113, 126], [22, 60, 112, 128], [3, 45, 84, 126],
            [20, 77, 88, 158], [0, 42, 81, 123], [22, 75, 107, 143],
            [17, 59, 95, 140], [21, 77, 106, 142], [10, 52, 91, 133],
            [33, 73, 113, 156], [8, 46, 105, 146], [16, 63, 114, 124],
            [36, 56, 121, 161], [36, 78, 110, 148], [25, 58, 117, 136],
            [38, 55, 120, 160], [28, 69, 86, 159], [40, 67, 118, 152],
            [27, 71, 85, 161], [30, 39, 93, 154], [18, 66, 108, 129],
            [8, 50, 89, 131], [0, 49, 115, 151], [38, 80, 109, 147],
            [37, 54, 122, 159], [32, 79, 97, 120], [24, 69, 102, 133],
            [7, 45, 107, 145], [16, 58, 94, 139], [25, 70, 103, 134],
            [28, 73, 101, 154], [30, 80, 98, 121], [13, 55, 90, 136],
            [29, 74, 99, 155], [19, 76, 87, 157], [39, 66, 117, 151],
            [7, 49, 88, 130], [23, 76, 105, 141], [37, 79, 108, 149],
            [31, 78, 96, 122], [4, 46, 85, 127], [27, 72, 100, 153],
            [34, 74, 111, 157], [6, 47, 106, 144], [9, 60, 96, 141],
            [3, 65, 104, 149], [35, 72, 112, 158], [1, 50, 116, 152],
            [34, 51, 83, 138], [20, 68, 110, 131], [32, 41, 95, 153],
            [4, 63, 102, 147], [41, 68, 119, 150], [31, 40, 94, 155],
            [5, 64, 103, 148], [15, 65, 116, 123], [11, 62, 98, 143],
            [17, 64, 115, 125], [12, 54, 92, 135], [26, 59, 118, 137],
            [2, 44, 83, 125], [21, 62, 111, 127], [29, 70, 84, 160],
            [12, 44, 100, 145], [33, 53, 82, 140], [1, 43, 82, 124],
            [5, 47, 86, 128], [15, 57, 93, 138], [24, 57, 119, 135],
            [14, 43, 99, 144], [2, 48, 114, 150], [14, 56, 91, 137],
            [6, 48, 87, 129], [35, 52, 81, 139], [10, 61, 97, 142],
            [18, 75, 89, 156], [11, 53, 92, 134], [9, 51, 90, 132],
        ], dtype=np.int32)
        
        H_ele = np.array([
            [46, 45, 44, 15], [58, 56, 60, 62], [54, 7, 38, 23],
            [26, 22, 14, 2], [35, 1, 31, 44], [16, 63, 20, 9],
            [42, 47, 37, 32], [63, 13, 54, 10], [1, 21, 25, 7],
            [41, 48, 2, 27], [46, 25, 22, 48], [60, 24, 4, 50],
            [25, 11, 7, 1], [13, 27, 56, 8], [60, 48, 2, 27],
            [53, 35, 16, 13], [20, 16, 63, 9], [43, 47, 18, 20],
            [9, 41, 57, 58], [37, 53, 61, 29], [19, 24, 42, 14],
            [15, 24, 50, 37], [37, 53, 61, 29], [51, 59, 63, 47],
            [63, 26, 41, 12], [44, 51, 35, 13], [27, 56, 8, 43],
            [38, 12, 25, 51], [2, 46, 56, 35], [43, 58, 19, 49],
            [49, 21, 7, 35], [13, 29, 53, 61], [32, 49, 58, 19],
            [32, 49, 58, 19], [53, 40, 61, 18], [50, 54, 60, 62],
            [23, 25, 30, 16], [27, 37, 5, 26], [42, 14, 24, 33],
            [5, 31, 51, 30], [6, 45, 56, 19], [1, 45, 15, 6],
            [24, 50, 37, 15], [46, 58, 18, 6], [9, 3, 43, 29],
            [17, 32, 58, 37], [30, 1, 44, 7], [1, 44, 30, 24],
            [43, 34, 48, 57], [47, 20, 33, 26], [28, 4, 52, 44],
            [40, 21, 44, 17], [52, 17, 24, 61], [43, 34, 48, 57],
            [42, 14, 24, 33], [8, 43, 27, 56], [58, 19, 32, 49],
            [18, 6, 61, 21], [29, 7, 10, 16], [43, 22, 41, 20],
            [9, 3, 63, 43], [33, 45, 36, 34], [8, 43, 27, 56],
            [15, 32, 18, 61], [36, 19, 3, 57], [56, 8, 46, 13],
            [38, 23, 55, 22], [27, 5, 2, 62], [5, 26, 27, 37],
            [39, 9, 30, 48], [62, 54, 56, 60], [46, 44, 14, 15],
            [24, 23, 45, 11], [29, 41, 10, 16], [29, 7, 10, 16],
            [39, 56, 30, 48], [18, 40, 32, 61], [9, 3, 63, 43],
            [15, 1, 42, 45], [11, 60, 6, 49], [22, 15, 12, 33],
        ], dtype=np.uint8)
        return H_idx, H_ele

    @staticmethod
    def init_table():
        """Initialize and return the GF(64) multiplication table."""
        N_GF = 6                  # Number of GF(q) bits.
        Q_GF = 2 ** N_GF         # Number of GF(q) elements.
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
                        Normalized 162-by-6 decoder input.
            normSigma2  - float
                        Normalized single-branch noise variance.
        """
        #%% BDS-3 B2b LDPC ===================================================
        colH = 162
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
                           Normalized bit soft values arranged as ``162 x 6``.
            sigmaSquared   - float
                           Normalized single-branch noise variance.
        Returns
        -------
            llrMatrix      - ndarray
                           Initial ``162 x 64`` GF(64) symbol metrics.
        """
        #%% BDS-3 B2b B-CNAV3 LDPC ===========================================
        gfBits = 6
        gfSize = 64
        colH = 162

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
        """Permute one variable-node-to-check-node GF(64) message."""
        Q_GF = 64
        V2C_p = np.zeros(Q_GF, dtype=np.float32)
        V2C_p[GF_MUL[h]] = V2C
        return V2C_p

    @staticmethod
    def permute_C2V(h, C2V, GF_MUL):
        """Permute one check-node-to-variable-node GF(64) message."""
        return C2V[GF_MUL[h]].astype(np.float32, copy=False)

    @staticmethod
    def ext_min_sum(L1, L2):
        """Combine two truncated GF(64) EMS messages."""
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
        """Return whether all GF(64) parity-check equations are satisfied."""
        for row in range(H_idx.shape[0]):
            syndrome = 0
            for edge in range(H_idx.shape[1]):
                syndrome ^= int(GF_MUL[
                    code[H_idx[row, edge]], H_ele[row, edge]])
            if syndrome:
                return False
        return True

    @classmethod
    def B2b_EMS_decode(cls, L):
        """Decode 162 GF(64) B-CNAV3 symbols with the EMS algorithm."""
        GF_MUL = cls.init_table()
        H_idx, H_ele = cls.getH_information()

        N_GF = 6                  # Number of GF(q) bits.
        Q_GF = 2 ** N_GF         # Number of GF(q) elements.
        MAX_ITER = 15            # Maximum number of decoding iterations.
        n = 162

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
    def B2bLDPCDecoder(cls, navBits_soft_I, navBits_soft_Q):
        """Decode one B-CNAV3 frame into 486 information bits.

        Args
        ----
            navBits_soft_I - numpy.ndarray
                            1000-bit B-CNAV3 soft sequence after
                            secondary-code despreading.
            navBits_soft_Q - numpy.ndarray
                            1000-bit Q-branch soft/noise-reference sequence.

        Returns
        -------
            decodedNavBits - numpy.ndarray
                            486 decoded navigation bits for CRC and
                            ephemeris decoding.
        """
        #=== Generate LLR soft-information matrix =============================
        # B-CNAV3 frame before LDPC decoding:
        #   1  - 28    : preamble;
        #   29 - 1000  : LDPC-coded payload, 972 bits = 162 GF(64) symbols.
        payload_I = np.asarray(navBits_soft_I).reshape(-1)[28:1000]
        payload_Q = np.asarray(navBits_soft_Q).reshape(-1)[28:1000]

        # Blind SNR estimation and soft-information normalization.
        normIp, normSigma2 = cls.noiseEst(payload_I, payload_Q)

        # Generate the 162-by-64 LLR cost matrix required for nonbinary LDPC
        # decoding.
        Symbol_LLR = cls.initLlr(normIp, normSigma2)

        #=== EMS decoding =====================================================
        decoded_symbols_162 = cls.B2b_EMS_decode(Symbol_LLR).reshape(-1)

        # The first 81 GF(64) symbols carry 486 information bits.
        decoded_symbols_81 = decoded_symbols_162[:81]

        #=== Map decoded symbols back to a bit stream =========================
        decodedNavBits = ((decoded_symbols_81[:, None] >> np.arange(5, -1, -1)) & 1).astype(np.uint8).reshape(-1)
        return decodedNavBits


__all__ = ["LDPCDecoder"]
