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
Report the navigation status of the tracking-only GLONASS L2OCp receiver.

"""


#%% Broadcast ephemeris placeholder
class Ephemeris:
    """Store the L2OCp tracking-only status for one satellite.

    The present receiver tracks the pilot component. It does not decode the
    L2 CSI navigation message, so this class records only satellite identity
    and the unavailable-decoding status.
    """

    def __init__(self, PRN=0):
        """Initialize a placeholder because L2 CSI is not decoded.

        Args
        ----
            PRN         - int
                        Satellite slot number associated with the tracked
                        channel.

        Returns
        -------
            None
                Satellite identity and decoding-status fields are initialized
                in this object.
        """
        self.PRN = int(PRN)
        self.SVID = int(PRN)
        self.navDecoded = False
        self.flag = False
        self.decodeStatus = (
            "L2OCp pilot tracking only; L2 CSI navigation message "
            "is not decoded.")


#%% Navigation results
class NavSolutions:
    """Store the empty L2OCp tracking-only navigation result.

    No position solution is available until L2 CSI navigation-message
    decoding is implemented.
    """

    def __init__(self):
        """Initialize the unavailable navigation-solution status.

        Returns
        -------
            None
                ``solutionAvailable`` is initialized to False.
        """
        self.solutionAvailable = False


#%% Post-navigation engine
class NavigationEngine:
    """Handle the tracking-only L2OCp post-processing stage.

    Args
    ----
        settings        - object
                        Receiver settings.
    """

    def __init__(self, settings):
        """Initialize the GLONASS L2OCp post-navigation engine.

        Args
        ----
            settings    - object
                        Receiver settings.

        Returns
        -------
            None
                The tracking-only navigation engine is initialized in this
                object.
        """
        self.settings = settings
        self.navSolutions = NavSolutions()
        self.eph = {}

    def navigationRun(self, trackResults):
        """Record tracked satellites and skip unavailable L2 CSI navigation.

        Args
        ----
            trackResults - iterable of tracking.Channel
                        Results from the tracking function.

        Returns
        -------
            self        - NavigationEngine
                        Tracking-only post-navigation engine containing a
                        placeholder ephemeris entry for each locked channel.
        """
        #%% Initialize tracking-only navigation results ======================
        self.navSolutions = NavSolutions()
        self.eph = {}
        #--- Save satellites that remain in tracking lock --------------------
        for channel in trackResults:
            if channel.lockFlag:
                self.eph[channel.PRN] = Ephemeris(channel.PRN)

        # L2 CSI is not decoded, so pseudorange and position computation are
        # intentionally skipped for this pilot-tracking receiver.
        print("GLONASS L2OCp tracking-only mode: no L2 CSI ephemeris "
              "decoded, navigation solution skipped.")
        return self

    def plotNavigation(self):
        """Skip plotting because no L2OCp navigation solution is available.

        Returns
        -------
            None
                No navigation figure is created in tracking-only mode.
        """
        return


__all__ = ["Ephemeris", "NavSolutions", "NavigationEngine"]
