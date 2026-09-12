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


#%% L2OCp C/N0 and PLL-lock detector
def calcCNoPld(chResults, settings, loopCnt):
    """Calculate L2OCp C/N0 and its PLL lock-detector output."""
    CNo = np.zeros(3)
    pllDetector = np.zeros(2)
    firstIndex = loopCnt-settings.CNoVSMinterval+1
    lastIndex = loopCnt+1
    I_P = chResults.I_P[firstIndex:lastIndex]
    Q_P = chResults.Q_P[firstIndex:lastIndex]
    CNo[0] = CNoVSM(I_P, Q_P, settings.intTime)
    CNo[2] = CNo[0]
    narrowI = np.sum(I_P[I_P > 0])-np.sum(I_P[I_P < 0])
    NBP = narrowI**2+np.sum(Q_P)**2
    NBD = narrowI**2-np.sum(Q_P)**2
    pllDetector[0] = NBD/NBP
    return CNo, pllDetector


#%% Public interface
__all__ = [
    # Progress display.
    "WaitBar",
    # Tracking-loop and signal-quality utilities.
    "calcLoopCoef", "CNoVSM", "calcCNoPld",
]
