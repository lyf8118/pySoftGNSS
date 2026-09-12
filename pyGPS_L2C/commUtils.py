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


#%% Data/pilot C/N0 and PLL-lock detector
def calcCNoPld(chResults, settings, loopCnt):
    """Calculate data, pilot, and combined L2C C/N0 values.

    Args
    ----
        chResults  - Channel
                    Correlation values accumulated for one channel.
        settings   - object
                    Receiver settings.
        loopCnt    - int
                    Zero-based current tracking-interval index.
    Returns
    -------
        CNo         - numpy.ndarray
                    Data, pilot, and combined L2C C/N0 values in dB-Hz.
        pllDetector - numpy.ndarray
                    Data- and pilot-channel PLL lock-detector outputs.
    """
    CNo = np.zeros(3)
    pllDetector = np.zeros(2)
    firstIndex = loopCnt - settings.CNoVSMinterval + 1
    lastIndex = loopCnt + 1
    T = settings.intTime

    #--- Data-channel estimates ------------------------------------------
    I_P = chResults.I_P[firstIndex:lastIndex]
    Q_P = chResults.Q_P[firstIndex:lastIndex]
    dataCNo = 10 ** (CNoVSM(I_P, Q_P, T) / 10)
    CNo[0] = 10 * np.log10(dataCNo)
    NBP = ((np.sum(I_P[I_P > 0]) - np.sum(I_P[I_P < 0])) ** 2
           + np.sum(Q_P) ** 2)
    NBD = ((np.sum(I_P[I_P > 0]) - np.sum(I_P[I_P < 0])) ** 2
           - np.sum(Q_P) ** 2)
    pllDetector[0] = NBD / NBP

    #--- CL pilot-channel estimates --------------------------------------
    Q_P = chResults.Pilot_I_P[firstIndex:lastIndex]
    I_P = chResults.Pilot_Q_P[firstIndex:lastIndex]
    pilotCNo = 10 ** (CNoVSM(I_P, Q_P, T) / 10)
    CNo[1] = 10 * np.log10(pilotCNo)
    NBP = ((np.sum(I_P[I_P > 0]) - np.sum(I_P[I_P < 0])) ** 2
           + np.sum(Q_P) ** 2)
    NBD = ((np.sum(I_P[I_P > 0]) - np.sum(I_P[I_P < 0])) ** 2
           - np.sum(Q_P) ** 2)
    pllDetector[1] = NBD / NBP

    #--- Combined L2C estimate -------------------------------------------
    CNo[2] = 10 * np.log10(dataCNo + pilotCNo)
    return CNo, pllDetector


#%% Navigation-plot utilities
def skyPlot(hAxis, az, el, prn, line_style="auto"):
    """Plot a sky view from the receiver perspective.

    Args
    ----
        hAxis      - matplotlib.axes.Axes
                   Axes that receive the sky plot.
        az         - array-like
                   Satellite azimuth angles. Each row contains one satellite
                   and the columns contain calculated azimuth values.
        el         - array-like
                   Satellite elevation angles arranged like ``az``.
        prn        - array-like
                   PRN numbers of the satellites.
        line_style - str
                   Line style and color used for all satellite positions.
    Returns
    -------
        hpol       - list
                   Handles of the plotted satellite tracks.
    """
    #%% Normalize input-array shapes ==========================================
    az = np.atleast_2d(az)
    el = np.atleast_2d(el)
    prn = np.asarray(prn).reshape(-1)

    #%% Prepare axis ==========================================================
    #--- Plot white background and horizon ------------------------------------
    # Prepare the axes and plot the white background with the horizon.
    hAxis.set_facecolor("white")
    th = np.linspace(0, 2*np.pi, 101)
    xunit = np.cos(th)
    yunit = np.sin(th)
    hAxis.plot(90*yunit, 90*xunit, color="0.35", zorder=0)

    #%% Plot and annotate spokes in degrees ===================================
    # Plot spokes and annotate the twelve azimuth sectors. Zero degrees is
    # north and azimuth grows clockwise in the Cartesian coordinates.
    rt = 1.1*90
    for angle in np.arange(0, 360, 30):
        angleRad = np.deg2rad(angle)
        hAxis.plot([0, 90*np.sin(angleRad)],
                   [0, 90*np.cos(angleRad)], ":", color="0.65",
                   linewidth=0.5, zorder=1)
        hAxis.text(rt*np.sin(angleRad), rt*np.cos(angleRad), str(angle),
                   ha="center", va="center")

    #%% Plot elevation grid ===============================================
    # Plot elevation grid lines and tick text every 15 degrees using the
    # SoftGNSS spherical-radius mapping.
    for elevation in range(0, 91, 15):
        elevationSpherical = 90*np.cos(np.deg2rad(elevation))
        
        hAxis.plot(yunit*elevationSpherical,
                   xunit*elevationSpherical, ":", color="0.65",
                   linewidth=0.5, zorder=1)
        
        hAxis.text(0, elevationSpherical, str(elevation),
                   ha="center", va="center", backgroundcolor="white")

    #%% Transform elevation to distance from the plot centre ==================
    elSpherical = 90*np.cos(np.deg2rad(el))
    #--- Transform data to Cartesian coordinates ------------------------------
    # Transform the azimuth/elevation data to Cartesian sky-view coordinates.
    xx = elSpherical*np.sin(np.deg2rad(az))
    yy = elSpherical*np.cos(np.deg2rad(az))

    #%% Plot data on top of the grid ==========================================
    # Use the default line style unless the caller supplies one. A supplied
    # style is used for every satellite.
    line_style = ".-" if line_style == "auto" else line_style
    hpol = hAxis.plot(xx.T, yy.T, line_style, zorder=2)

    #--- Mark and label the latest satellite position -------------------------
    # Mark and label the latest valid position of every satellite. The
    # leading blank in the label places the PRN beside the final point and
    # keeps a constant visual offset when the plot is zoomed.
    for i in range(len(prn)):
        valid = np.isfinite(xx[i]) & np.isfinite(yy[i])
        if prn[i] != 0 and np.any(valid):
            last = np.flatnonzero(valid)[-1]
            
            color = hpol[i].get_color()
            
            hAxis.plot(xx[i, last], yy[i, last], "o",
                       color=color, markersize=7, zorder=3)
            
            hAxis.text(xx[i, last], yy[i, last], f"  {int(prn[i])}",
                       color="b")

    #--- Set axis limits and reserve space for the title ----------------------
    hAxis.set_xlim(-95, 95)
    hAxis.set_ylim(-90, 101)
    #--- Keep equal aspect ratio and hide Cartesian axes ----------------------
    # Preserve equal data aspect ratio and hide the Cartesian axes.
    hAxis.set_aspect("equal")
    hAxis.axis("off")
    return hpol


#%% Pseudorange computation
def calculatePseudoranges(trackResults, subFrameStart, TOW, currMeasSample,
                          localTime, channelList, settings, searchIndex):
    """Find relative pseudoranges for the satellites in ``channelList``.

    The pseudoranges contain an unknown receiver clock offset that is found
    by the least-squares position procedure.

    Args
    ----
        trackResults  - sequence
                      Output from the tracking function.
        subFrameStart - array-like
                      Zero-based first-message positions counted in 20 ms
                      L2C CM-code periods from the start of tracking.
        TOW           - array-like
                      Time of week of the first subframe, in seconds.
        currMeasSample - int
                       Current measurement sample location.
        localTime     - float or None
                      Local GPST at the measurement time.
        channelList   - sequence
                      Channels to process.
        settings      - object
                      Receiver settings.
        searchIndex   - numpy.ndarray
                      Per-channel search positions used by the Python port.
    Returns
    -------
        pseudoranges  - numpy.ndarray
                      Relative pseudoranges to the satellites.
        transmitTime  - numpy.ndarray
                      Transmitting times at the measurement time.
        localTime     - float
                      Local GPST at the measurement time.
        carrFreqPerSat - numpy.ndarray
                       Carrier frequencies at the measurement time.
    """
    # Transmitting time and carrier frequency of all channels at the current
    # measurement-sample location.
    transmitTime = np.full(settings.numberOfChannels, np.inf)
    carrFreqPerSat = np.full(settings.numberOfChannels, np.inf)

    # Use the saved indices to accelerate successive searches.
    if localTime is None:
        searchIndex.fill(0)

    # Process all channels in the requested list.
    for channelNr in channelList:
        # Find the prompt-integration interval containing the measurement
        # point location.
        index = int(searchIndex[channelNr])
        while (trackResults[channelNr].absoluteSample[index] <= currMeasSample):
            index += 1
        
        searchIndex[channelNr] = index
        index -= 1

        # Update the phase step from code frequency and sampling frequency,
        # then find code phase from the PRN start to the measurement point.
        codePhaseStep = (trackResults[channelNr].codeFreq[index]
                         / settings.samplingFreq)
        
        codePhase = (trackResults[channelNr].remCodePhase[index]
                     + codePhaseStep * (currMeasSample
                    - trackResults[channelNr].absoluteSample[index]))

        # Form transmitting time [s].  codePhase/codeLength is the fractional
        # PRN period; index-subFrameStart is the integer number of periods.
        transmitTime[channelNr] = (
            (codePhase / settings.codeLength + index - subFrameStart[channelNr])
            * settings.codeLength / settings.codeFreqBasis + TOW[channelNr])
        
        carrFreqPerSat[channelNr] = trackResults[channelNr].carrFreq[index]

    # At the first fix, initialize local time from transmit time and the
    # assumed propagation delay settings.startOffset [ms].
    if localTime is None:
        maxTime = np.max(transmitTime[channelList])
        localTime = maxTime+settings.startOffset/1000

    # Convert signal travel time to distance.  Both times are seconds, so
    # multiplication by the speed of light [m/s] gives metres.
    pseudoranges = (localTime-transmitTime)*settings.c
    return pseudoranges, transmitTime, localTime, carrFreqPerSat


#%% Position solution and propagation corrections
def e_r_corr(traveltime, X_sat):
    """Rotate satellite ECEF coordinates for Earth rotation during travel.

    Args
    ----
        traveltime  - float
                    Signal travel time.
        X_sat       - array-like
                    Satellite ECEF coordinates.
    Returns
    -------
        X_sat_rot   - numpy.ndarray
                    Rotated satellite ECEF coordinates.
    """
    # Earth rotation rate [rad/s].
    Omegae_dot = 7.292115147e-5

    # Find the Earth rotation angle accumulated during signal travel.
    omegatau = Omegae_dot*traveltime

    # Form the ECEF z-axis rotation matrix.
    R3 = np.array([[np.cos(omegatau), np.sin(omegatau), 0],
                   [-np.sin(omegatau), np.cos(omegatau), 0], [0, 0, 1],])
    
    # Rotate the satellite coordinates from transmit to receive time.
    X_sat_rot = R3@X_sat
    
    return X_sat_rot


def togeod(a, finv, X, Y, Z):
    """Calculate geodetic coordinates from Cartesian coordinates.

    The linear quantities ``X``, ``Y``, ``Z`` and ``a`` must use the same
    unit. Angular outputs are decimal degrees and height retains that unit.

    Args
    ----
        a           - float
                    Semi-major axis of the reference ellipsoid.
        finv        - float
                    Inverse flattening of the reference ellipsoid.
        X, Y, Z     - float
                    Cartesian coordinates.
    Returns
    -------
        dphi       - float
                   Geodetic latitude.
        dlambda    - float
                   Geodetic longitude.
        h          - float
                   Height above the reference ellipsoid.
    """
    from math import atan2

    h = 0
    tolsq = 1e-10
    maxit = 10
    
    # Compute the radians-to-degrees factor.
    rtd = 180/np.pi

    # Compute squared eccentricity and its complement.
    esq = 0 if finv < 1e-20 else (2-1/finv)/finv
    oneesq = 1-esq

    # First guess: P is distance from the spin axis; longitude follows
    # directly from the Cartesian coordinates.
    P = np.sqrt(X**2 + Y**2)
    dlambda = atan2(Y, X)*rtd if P > 1e-20 else 0
    if dlambda < 0:
        dlambda += 360

    # r is distance from the Cartesian origin.
    r = np.sqrt(P**2+Z**2)
    sinphi = Z/r if r > 1e-20 else 0
    dphi = np.arcsin(sinphi)
    if r < 1e-20:
        return dphi, dlambda, h

    # Initial height is the distance from the origin minus an approximate
    # distance from the origin to the ellipsoid surface.
    h = r-a*(1-sinphi*sinphi/finv)

    # Iteratively refine geodetic latitude and ellipsoidal height.
    for i in range(1, maxit+1):
        sinphi = np.sin(dphi)
        cosphi = np.cos(dphi)
        # Radius of curvature in the prime-vertical direction.
        N_phi = a/np.sqrt(1-esq*sinphi*sinphi)

        # Residuals in distance from the spin axis and in Z.
        dP = P-(N_phi+h)*cosphi
        dZ = Z-(N_phi*oneesq+h)*sinphi

        # Update height and latitude.
        h += sinphi*dZ+cosphi*dP
        dphi += (cosphi*dZ-sinphi*dP)/(N_phi+h)

        # Test for convergence; warn if the final iteration is reached.
        if dP*dP+dZ*dZ < tolsq:
            break
        if i == maxit:
            print(f"Problem in TOGEOD, did not converge in {i} "
                  "iterations")

    return dphi*rtd, dlambda, h


def topocent(X, dx):
    """Transform a vector into a topocentric system with origin at ``X``.

    Args
    ----
        X           - array-like
                    Three-element ECEF coordinate of the origin.
        dx          - array-like
                    Three-element ECEF difference vector.
    Returns
    -------
        Az          - float
                    Azimuth from north, positive clockwise, in degrees.
        El          - float
                    Elevation angle in degrees.
        D           - float
                    Vector length in the same unit as the inputs.
    """
    from math import atan2

    # Find the geodetic coordinates of the local origin on WGS-84.
    dtr = np.pi/180
    phi, lambda_, h = togeod(6378137, 298.257223563, X[0], X[1], X[2])

    cl = np.cos(lambda_*dtr)
    sl = np.sin(lambda_*dtr)
    cb = np.cos(phi*dtr)
    sb = np.sin(phi*dtr)
    
    # Construct the ECEF-to-local transformation matrix.
    F = np.array([ [-sl, -sb*cl, cb*cl],
                  [cl, -sb*sl, cb*sl],
                  [0, cb, sb], ])

    # Transform dx into local east, north and up components.
    local_vector = F.T@dx
    E = local_vector[0]
    N = local_vector[1]
    U = local_vector[2]
    hor_dis = np.sqrt(E**2+N**2)
    
    if hor_dis < 1e-20:
        Az, El = 0, 90
    else:
        Az = atan2(E, N)/dtr
        El = atan2(U, hor_dis)/dtr
        
    if Az < 0:
        Az += 360

    # The vector length retains the linear unit used by the input.
    D = np.linalg.norm(dx)
    return Az, El, D


def tropo(sinel, hsta, p, tkel, hum, hp, htkel, hhum):
    """Calculate the Goad-Goodman tropospheric correction.

    The correction is subtracted from pseudoranges and carrier phases.

    Args
    ----
        sinel       - float
                    Sine of the satellite elevation angle.
        hsta        - float
                    Station height in kilometres.
        p           - float
                    Atmospheric pressure in mbar at height ``hp``.
        tkel        - float
                    Surface temperature in kelvin at height ``htkel``.
        hum         - float
                    Humidity in percent at height ``hhum``.
        hp          - float
                    Pressure-measurement height in kilometres.
        htkel       - float
                    Temperature-measurement height in kilometres.
        hhum        - float
                    Humidity-measurement height in kilometres.
    Returns
    -------
        ddr         - float
                    Range correction in metres.
    """
    # Semi-major axis of the Earth ellipsoid [km].
    a_e = 6378.137
    b0 = 7.839257e-5
    tlapse = -6.5

    # Reduce the measured temperature, humidity and pressure to sea level.
    tkhum = tkel + tlapse * (hhum - htkel)
    atkel = 7.5 * (tkhum - 273.15) / (237.3 + tkhum - 273.15)
    e0    = 0.0611 * hum * 10 ** atkel
    tksea = tkel - tlapse * htkel
    em    = -978.77 / (2.8704e6 * tlapse * 1e-5)
    tkelh = tksea + tlapse * hhum
    e0sea = e0 * (tksea / tkelh) ** (4 * em)
    tkelp = tksea + tlapse * hp
    psea  = p * (tksea / tkelp) ** em

    if sinel < 0:
        sinel = 0

    # Initialize the dry-component refractivity and atmospheric top height.
    ddr = 0
    refsea  = 77.624e-6/tksea
    htop    = 1.1385e-5/refsea
    refsea *= psea
    ref     = refsea * ((htop-hsta) / htop) ** 4

    # Accumulate the dry and wet atmospheric components.
    for component in range(2):
        rtop = ((a_e + htop) ** 2-(a_e + hsta) ** 2 * (1 - sinel ** 2))

        # Check for physically inconsistent geometry.
        if rtop < 0:
            rtop = 0
        rtop = np.sqrt(rtop)-(a_e+hsta)*sinel
        a = -sinel/(htop-hsta)
        b = -b0*(1-sinel**2)/(htop-hsta)
        rn = np.array([rtop**i for i in range(2, 10)])
        alpha = np.array([2*a,            
                          2*a**2+4*b/3,
                          a*(a**2+3*b),
                          a**4/5+2.4*a**2*b+1.2*b**2,
                          2*a*b*(a**2+3*b)/3,
                          b**2*(6*a**2+4*b)*1.428571e-1,
                          0,
                          0, ])
        
        if b**2 > 1e-35:
            alpha[6] = a*b**3/2
            alpha[7] = b**4/9

        dr = rtop
        dr += alpha@rn
        ddr += dr*ref*1000

        if component == 0:
            # After the dry component, initialize the wet refractivity.
            refsea = (371900e-6/tksea-12.92e-6)/tksea
            htop = 1.1385e-5*(1255/tksea+0.05)/refsea
            ref = refsea*e0sea*((htop-hsta)/htop)**4

    return ddr


def leastSquarePos(satpos, obs, settings):
    """Calculate the least-squares receiver-position solution.

    Args
    ----
        satpos      - numpy.ndarray
                    Satellite ECEF positions, one satellite per column.
        obs         - array-like
                    Pseudoranges corrected for satellite clock error.
        settings    - object
                    Receiver settings.
    Returns
    -------
        pos         - numpy.ndarray
                    Receiver ECEF position and clock error ``[X, Y, Z, dt]``.
        el          - numpy.ndarray
                    Satellite elevation angles in degrees.
        az          - numpy.ndarray
                    Satellite azimuth angles in degrees.
        dop         - numpy.ndarray
                    Dilutions of precision ``[GDOP, PDOP, HDOP, VDOP, TDOP]``.
    """
    # Initialization: start the receiver at the centre of the Earth.
    nmbOfIterations = 10
    dtr             = np.pi/180
    pos             = np.zeros(4)
    X               = satpos.copy()
    nmbOfSatellites = satpos.shape[1]

    A   = np.zeros((nmbOfSatellites, 4))
    omc = np.zeros(nmbOfSatellites)
    az  = np.zeros(nmbOfSatellites)
    el  = np.zeros(nmbOfSatellites)

    # Iteratively find receiver position and clock bias.
    for iter in range(nmbOfIterations):
        for i in range(nmbOfSatellites):
            if iter == 0:
                # MATLAB initializes the first linearization with a fixed
                # two-metre tropospheric correction.
                Rot_X = X[:, i]
                trop = 2
            else:
                # Update the observation equations on later iterations.
                rho2 = np.sum((X[:, i]-pos[:3])**2)
                traveltime = np.sqrt(rho2)/settings.c

                # Correct satellite position for Earth rotation: convert
                # ECEF coordinates at signal-transmit time to the ECEF frame
                # orientation at signal-receive time.
                Rot_X = e_r_corr(traveltime, X[:, i])

                # Find satellite azimuth and elevation at the receiver.
                az[i], el[i], _ = topocent(pos[:3], Rot_X-pos[:3])

                if settings.useTropCorr:
                    # Calculate the Goad-Goodman tropospheric correction.
                    trop = tropo(np.sin(el[i]*dtr), 0.0, 1013.0, 293.0,
                                 50.0, 0.0, 0.0, 0.0)
                else:
                    # Do not calculate or apply tropospheric correction.
                    trop = 0

            # Apply range, clock and tropospheric corrections.
            delta = Rot_X-pos[:3]
            distance = np.linalg.norm(delta)
            omc[i] = obs[i]-distance-pos[3]-trop

            # Construct the geometry/design matrix A.
            A[i, :3] = -delta/distance
            A[i, 3] = 1

        # Exit gracefully when satellite geometry is rank deficient.
        if np.linalg.matrix_rank(A) != 4:
            print("Cannot get a converged solution!")
            return np.zeros(4), el, az, np.full(5, np.inf)

        # Find and apply the position update in the least-squares sense.
        x = np.linalg.lstsq(A, omc, rcond=None)[0]
        pos += x

    # Calculate dilution of precision from the final geometry matrix.
    Q = np.linalg.inv(A.T@A)
    
    dop = np.array([np.sqrt(np.trace(Q)),                    # GDOP
                    np.sqrt(Q[0, 0]+Q[1, 1]+Q[2, 2]),        # PDOP
                    np.sqrt(Q[0, 0]+Q[1, 1]),                # HDOP
                    np.sqrt(Q[2, 2]),                        # VDOP
                    np.sqrt(Q[3, 3]), ])                     # TDOP
   
    return pos, el, az, dop


#%% Coordinate transformations
def cart2geo(X, Y, Z, i):
    """Convert Cartesian coordinates to geographical coordinates.

    The reference-ellipsoid choices are International 1924, International
    1967, WGS-72, GRS-80 and WGS-84 for selector values 1 through 5.

    Args
    ----
        X, Y, Z     - float
                    Cartesian coordinates.
        i           - int
                    Reference-ellipsoid selector.
    Returns
    -------
        phi         - float
                    Geographical latitude.
        lambda_     - float
                    Geographical longitude.
        h           - float
                    Height above the selected ellipsoid.
    """
    from math import atan, atan2

    # 1: International 1924, 2: International 1967, 3: WGS-72,
    # 4: GRS-80, 5: WGS-84.
    a = np.array([6378388, 6378160, 6378135, 6378137, 6378137])
    f = np.array([1/297, 1/298.247, 1/298.26,
                  1/298.257222101, 1/298.257223563])
    i -= 1

    # Calculate longitude, second eccentricity and curvature constant, then
    # form the initial latitude estimate.
    lambda_ = atan2(Y, X)
    ex2 = (2-f[i])*f[i]/(1-f[i])**2
    c = a[i]*np.sqrt(1+ex2)
    phi = atan(Z/(np.sqrt(X**2+Y**2)*(1-(2-f[i]))*f[i]))

    # Iteratively approximate latitude and ellipsoidal height.
    h, oldh = 0.1, 0
    iterations = 0
    while abs(h-oldh) > 1e-12:
        oldh = h
        N    = c/np.sqrt(1+ex2*np.cos(phi)**2)
        phi  = atan(Z/(np.sqrt(X**2+Y**2) * (1-(2-f[i])*f[i]*N/(N+h))))
        h    = np.sqrt(X**2+Y**2)/np.cos(phi)-N
        iterations += 1
        
        if iterations > 100:
            print("Failed to approximate h with desired precision. "
                  f"h-oldh: {h-oldh:e}.")
            break

    # Convert latitude and longitude from radians to decimal degrees.
    return phi*180/np.pi, lambda_*180/np.pi, h


def findUtmZone(latitude, longitude):
    """Find the UTM zone number for a longitude and latitude.

    Longitude must be between -180 and 180 degrees and latitude between
    -80 and 84 degrees. Both inputs must be decimal degrees.

    Args
    ----
        latitude    - float
                    Latitude in decimal degrees.
        longitude   - float
                    Longitude in decimal degrees.
    Returns
    -------
        utmZone     - int
                    UTM zone number.
    """
    # Check longitude and latitude bounds.
    if longitude > 180 or longitude < -180:
        raise ValueError("Longitude value exceeds limits (-180:180).")
    if latitude > 84 or latitude < -80:
        raise ValueError("Latitude value exceeds limits (-80:84).")

    # Zone numbering starts at 180 degrees west.
    utmZone = int(np.fix((180+longitude)/6)+1)

    # Apply the Norway and Svalbard special zones.
    if latitude > 72:
        if 0 <= longitude < 9:
            utmZone = 31
        elif 9 <= longitude < 21:
            utmZone = 33
        elif 21 <= longitude < 33:
            utmZone = 35
        elif 33 <= longitude < 42:
            utmZone = 37
    elif 56 <= latitude < 64 and 3 <= longitude < 12:
        utmZone = 32
    return utmZone


def clsin(ar, degree, argument):
    """Perform Clenshaw summation of the sine of an argument.

    Args
    ----
        ar          - array-like
                    Series coefficients.
        degree      - int
                    Number of coefficients to use.
        argument    - float
                    Series argument.
    Returns
    -------
        result      - float
                    Value of the sine series.
    """
    # Clenshaw backward recurrence.
    cos_arg = 2*np.cos(argument)
    hr1 = 0
    hr = 0
    for t in range(degree-1, -1, -1):
        hr2 = hr1
        hr1 = hr
        hr = ar[t]+cos_arg*hr1-hr2
    result = hr*np.sin(argument)
    return result


def clksin(ar, degree, arg_real, arg_imag):
    """Perform Clenshaw summation of a sine with a complex argument.

    Args
    ----
        ar          - array-like
                    Series coefficients.
        degree      - int
                    Number of coefficients to use.
        arg_real    - float
                    Real part of the argument.
        arg_imag    - float
                    Imaginary part of the argument.
    Returns
    -------
        re          - float
                    Real part of the result.
        im          - float
                    Imaginary part of the result.
    """
    # Sine/cosine of the real argument and hyperbolic functions of the
    # imaginary argument form the real and imaginary parts of 2*cos(z).
    sin_arg_r  = np.sin(arg_real)
    cos_arg_r  = np.cos(arg_real)
    sinh_arg_i = np.sinh(arg_imag)
    cosh_arg_i = np.cosh(arg_imag)
    r          = 2*cos_arg_r*cosh_arg_i
    i          = -2*sin_arg_r*sinh_arg_i
    hr1        = hr = hi1 = hi = 0

    # Coupled real/imaginary Clenshaw backward recurrence.
    for t in range(degree-1, -1, -1):
        hr2, hr1 = hr1, hr
        hi2, hi1 = hi1, hi
        z        = ar[t]+r*hr1-i*hi-hr2
        hi       = i*hr1+r*hi1-hi2
        hr       = z

    # Form sin(z) and complete the final complex multiplication.
    r  = sin_arg_r*cosh_arg_i
    i  = cos_arg_r*sinh_arg_i
    re = r*hr-i*hi
    im = r*hi+i*hr
    return re, im


def cart2utm(X, Y, Z, zone):
    """Transform ITRF96 Cartesian coordinates to UTM coordinates.

    Args
    ----
        X, Y, Z     - float
                    Cartesian coordinates referenced to ITRF96.
        zone        - int
                    UTM zone of the position.
    Returns
    -------
        E           - float
                    UTM easting.
        N           - float
                    UTM northing.
        U           - float
                    UTM up coordinate.
    """
    from math import atan2, atanh

    # International 1924 ellipsoid, valid for ED50.
    a   = 6378388
    f   = 1/297
    ex2 = (2-f)*f/(1-f)**2
    c   = a*np.sqrt(1+ex2)

    # Fixed scale/rotation/translation from ITRF96 to ED50.
    vec = np.array([X, Y, Z-4.5])
    alpha = .756e-6
    R = np.array([[1, -alpha, 0],        
                  [alpha, 1, 0],        
                  [0, 0, 1], ])
    trans = np.array([89.5, 93.8, 127.6])
    scale = 0.9999988
    v = scale*(R@vec)+trans

    # Calculate longitude and preliminary latitude/height in ED50, then
    # iteratively refine latitude and ellipsoidal height.
    L = atan2(v[1], v[0])
    N1 = 6395000
    B = atan2(v[2]/((1-f)**2*N1), np.linalg.norm(v[:2])/N1)
    U, oldU = 0.1, 0
    while abs(U-oldU) > 1e-4:
        oldU = U
        N1 = c/np.sqrt(1+ex2*np.cos(B)**2)
        B = atan2(v[2]/((1-f)**2*N1+U), np.linalg.norm(v[:2])/(N1+U))
        U = np.linalg.norm(v[:2])/np.cos(B)-N1

    # Normalized meridian quadrant (Koenig and Weise formulation).
    m0 = 0.0004
    n = f/(2-f)
    m = n**2*(1/4+n*n/64)
    w = a*(-n-m0+m*(1-m0))/(1+n)
    Q_n = a+w
    E0 = 500000
    L0 = (zone-30)*6-3

    bg = np.array([ -3.37077907e-3, 4.73444769e-6, 
                   -8.29914570e-9, 1.58785330e-11,])
    gtu = np.array([8.41275991e-4, 7.67306686e-7,        
                    1.21291230e-9, 2.48508228e-12,])

    # Ellipsoidal latitude/longitude to spherical latitude/longitude.
    neg_geo = B < 0
    Bg_r = abs(B)
    res_clensin = clsin(bg, 4, 2*Bg_r)
    Bg_r += res_clensin
    L0 *= np.pi/180
    Lg_r = L-L0
    # Spherical latitude/longitude to complementary spherical northing and
    # easting.
    cos_BN = np.cos(Bg_r)
    Np = atan2(np.sin(Bg_r), np.cos(Lg_r)*cos_BN)
    Ep = atanh(np.sin(Lg_r)*cos_BN)

    # Spherical normalized N/E to ellipsoidal N/E.
    Np *= 2
    Ep *= 2
    dN, dE = clksin(gtu, 4, Np, Ep)
    Np /= 2
    Ep /= 2
    Np += dN
    Ep += dE
    N = Q_n*Np
    E = Q_n*Ep+E0
    # Apply the southern-hemisphere false northing.
    if neg_geo:
        N = -N+20000000
    return E, N, U


#%% Navigation-data conversion
def twosComp2dec(binaryNumber):
    """Convert a two's-complement binary number to an integer.

    Args
    ----
        binaryNumber - sequence of str
                     Row vector of zeros and ones.
    Returns
    -------
        intNumber    - int
                     Integer represented by ``binaryNumber``.
    """
    #--- Normalize the input to a binary string -------------------------------
    if not isinstance(binaryNumber, str):
        binaryNumber = "".join(str(int(bit)) for bit in binaryNumber)

    #--- Convert from binary form to a decimal number -------------------------
    intNumber = int(binaryNumber, 2)

    #--- If the number is negative, correct the result ------------------------
    # If the sign bit is one, correct the result to two's-complement form.
    if binaryNumber[0] == "1":
        intNumber -= 2**len(binaryNumber)
    return intNumber


#%% Public interface
__all__ = [
    # Progress display.
    "WaitBar",
    # Tracking-loop and signal-quality utilities.
    "calcLoopCoef", "CNoVSM", "calcCNoPld",
    # Navigation plotting.
    "skyPlot",
    # Pseudorange computation.
    "calculatePseudoranges",
    # Position solution and propagation corrections.
    "e_r_corr", "togeod", "topocent", "tropo", "leastSquarePos",
    # Coordinate transformations.
    "cart2geo", "findUtmZone", "clsin", "clksin", "cart2utm",
    # Navigation-data conversion.
    "twosComp2dec",
]
