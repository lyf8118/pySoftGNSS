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

commUtils.py - Module Description
---------------------------------
Common numerical and display utilities for the software receiver.

"""

#%% Imports
import numpy as np


#%% Tracking progress display
class WaitBar:
    """Display a cancellable progress window for tracking.

    Args
    ----
        total       - int
                    Maximum progress value.
    """

    def __init__(self, total):
        """Initialize the tracking progress window.

        Args
        ----
            total       - int
                        Maximum progress value.
        Returns
        -------
            None
        """
        import tkinter as tk
        from tkinter import ttk

        self.cancelled = False
        self.window = tk.Tk()
        self.window.title("Tracking")
        self.window.minsize(400, 110)

        self.label = ttk.Label(self.window)
        self.label.pack(pady=10)
        self.bar = ttk.Progressbar(self.window, maximum=total, length=360)
        self.bar.pack()
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

    def update(self, value, text):
        """Update the displayed progress.

        Args
        ----
            value       - int
                        Current progress value.
            text        - str
                        Text displayed above the progress bar.
        Returns
        -------
            continueFlag - bool
                         False after cancellation; otherwise True.
        """
        self.label["text"] = text
        self.bar["value"] = value
        self.window.update()
        return not self.cancelled

    def cancel(self):
        """Record a cancellation request and hide the window.
        """
        self.cancelled = True
        self.window.withdraw()

    def close(self):
        """Close the progress window.
        """
        self.window.destroy()


#%% Tracking-loop and signal-quality utilities
def calcLoopCoef(LBW, zeta, k):
    """Find loop coefficients used in PLLs and DLLs.

    Args
    ----
        LBW         - float
                    Loop noise bandwidth.
        zeta        - float
                    Damping ratio.
        k           - float
                    Loop gain.
    Returns
    -------
        tau1, tau2  - float
                    Loop-filter coefficients.
    """
    # Solve the natural frequency from loop noise bandwidth and damping.
    Wn = LBW*8*zeta/(4*zeta**2+1)

    # Solve for tau1 and tau2.
    tau1 = k/(Wn*Wn)
    tau2 = 2.0*zeta/Wn
    return tau1, tau2


def calcLoopCoefCarr(settings):
    """Find third-order loop coefficients used by the carrier PLL.

    The loop improves carrier-tracking performance under dynamic conditions.
    See Kaplan and Hegarty, *Satellite Signal Acquisition, Tracking, and Data
    Demodulation*.

    Args
    ----
        settings    - object
                    Receiver settings.
    Returns
    -------
        pf3, pf2, pf1 - float
                    Three loop-filter coefficients.
    """
    # Loop noise bandwidth: the third-order loop remains stable at LBW <= 18 Hz.
    LBW = settings.pllNoiseBandwidth

    # Summation interval.
    intTime = settings.intTime

    # Loop constant coefficients.
    a3 = 1.1
    b3 = 2.4

    # Solve the natural frequency.
    Wn = LBW/0.7845

    # Solve for pf3, pf2 and pf1.
    pf3 = Wn**3*intTime**2
    pf2 = a3*Wn**2*intTime
    pf1 = b3*Wn
    return pf3, pf2, pf1


def CNoVSM(I, Q, T):
    """Calculate C/N0 using the variance-summing method.

    Args
    ----
        I           - array-like
                    Prompt in-phase values from tracking.
        Q           - array-like
                    Prompt quadrature values from tracking.
        T           - float
                    Tracking accumulation interval in seconds.
    Returns
    -------
        CNo         - float
                    Estimated C/N0 for the supplied I and Q values.
    """
    # Calculate prompt power.
    Z = I**2+Q**2

    # Calculate the mean and variance of the power.
    Zm = np.mean(Z)
    Zv = np.var(Z, ddof=1)

    # Calculate the average carrier power.
    Pav = np.emath.sqrt(Zm**2-Zv)

    # Calculate the variance of the noise.
    Nv = 0.5*(Zm-Pav)

    # Calculate C/N0 and convert it to dB-Hz.
    CNo = 10*np.log10(np.abs((1/T)*Pav/(2*Nv)))
    return float(np.real(CNo))


def calcCNoPld(chResults, settings, loopCnt):
    """Calculate C/N0 and the PLL lock-detector output.

    Args
    ----
        chResults    - Channel
                      Correlation values accumulated for one channel.
        settings     - object
                      Receiver settings.
        loopCnt      - int
                      Zero-based iteration index for C/N0 calculation.
    Returns
    -------
        CNo          - numpy.ndarray
                      Data, pilot, and whole E6B-signal C/N0 in dB-Hz.
        pllDetector  - numpy.ndarray
                      PLL lock-detector outputs for data and pilot channels.
    """
    CNo = np.zeros(3)
    pllDetector = np.zeros(2)
    firstIndex = loopCnt - settings.CNoVSMinterval + 1
    lastIndex = loopCnt + 1
    T = settings.intTime

    #--- C/N0 and PLL detector estimation for the data channel -----------
    I_P = chResults.I_P[firstIndex:lastIndex]
    Q_P = chResults.Q_P[firstIndex:lastIndex]
    dataCNo = 10**(CNoVSM(I_P, Q_P, T)/10)
    CNo[0] = 10*np.log10(dataCNo)
    NBP = ((np.sum(I_P[I_P > 0])-np.sum(I_P[I_P < 0]))**2
           + np.sum(Q_P)**2)
    NBD = ((np.sum(I_P[I_P > 0])-np.sum(I_P[I_P < 0]))**2
           - np.sum(Q_P)**2)
    pllDetector[0] = NBD/NBP

    #--- C/N0 and PLL detector estimation for the pilot channel ----------
    Q_P = chResults.Pilot_I_P[firstIndex:lastIndex]
    I_P = chResults.Pilot_Q_P[firstIndex:lastIndex]
    pilotCNo = 10**(CNoVSM(I_P, Q_P, T)/10)
    CNo[1] = 10*np.log10(pilotCNo)
    NBP = ((np.sum(I_P[I_P > 0])-np.sum(I_P[I_P < 0]))**2
           + np.sum(Q_P)**2)
    NBD = ((np.sum(I_P[I_P > 0])-np.sum(I_P[I_P < 0]))**2
           - np.sum(Q_P)**2)
    pllDetector[1] = NBD/NBP

    # C/N0 estimation for the complete Galileo E6B signal.
    CNo[2] = 10*np.log10(dataCNo+pilotCNo)
    return CNo, pllDetector


#%% Public interface
__all__ = [
    # Progress display.
    "WaitBar",
    # Tracking-loop and signal-quality utilities.
    "calcLoopCoef", "calcLoopCoefCarr", "CNoVSM", "calcCNoPld",
]
