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

main.py - Module Description
----------------------------
Initialize the settings of the GPS L1 C/A receiver.

The raw data are plotted first. Processing starts after the user confirms
that the receiver is ready.

"""

import os
# Clear the terminal display.
os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import initSettings
    # --- Initialize constants, settings --------------------------------------
    settings = initSettings.Settings()

    # Probe the input data, then ask whether processing should start.
    settings.probeData()

    # Read the user's start/exit choice.
    gnssStart = eval(input('Enter "1" to initiate GNSS processing or "0" to exit: '))

    # Start things rolling ====================================================
    if gnssStart:
        settings.postProcessing()
