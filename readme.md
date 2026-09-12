An Open-Source Python GNSS SDR Toolbox with SIMD/GPU Acceleration： pySoftGNSS 
===============================================================================



Overview
-------------------------------------------------------------------------------
pySoftGNSS is an open-source Python-based toolbox for post-processing recorded
GNSS intermediate-frequency (IF) signals. It follows a common receiver
processing architecture with mSoftGNSS and organizes signal-processing
functions into Python modules and classes. Supporting GPS, Galileo, GLONASS,
and BeiDou signals, it provides signal-dependent acquisition, data/pilot
tracking, navigation-message decoding, pseudorange generation, and positioning.
CPU and GPU acquisition use NumPy/SciPy and CuPy, respectively. Both
channel-serial and channel-parallel tracking support NumPy-based,
SIMD-accelerated, and GPU-accelerated correlators. Channel-parallel tracking
reuses shared IF data buffers and batches correlation tasks across channels.
C++ and CUDA C++ correlators are compiled into dynamic-link libraries and
accessed through ctypes.CDLL, while scheduling, tracking-loop control,
navigation decoding, and positioning remain in Python. This design preserves
readable, configurable high-level processing while accelerating computationally
intensive sample-level operations. With modernized-signal support, LDPC
decoding, and integrated visualization, pySoftGNSS provides an extensible
platform for GNSS algorithm development, receiver prototyping, and reproducible
evaluation.



Authors
-------------------------------------------------------------------------------
* Yafeng Li
    * E-Mail: <lyf8118@126.com>
    * Wechat: lyf8118521
    * QQ Group for Technical Discussions on GNSS Software Receivers: 147304049


* Dennis Akos  
    * E-Mail: <dma@colorado.edu>
    * HP: <http://www.colorado.edu/aerospace/dennis-akos>



Features
-------------------------------------------------------------------------------
* GNSS signal processing functions written in Python
    * Local code generation
    * CPU and GPU acquisition
    * Channel-serial and channel-parallel tracking
    * Python/NumPy, SIMD DLL, and GPU/CUDA DLL correlators
    * Data/pilot tracking for modernized signals
    * Navigation-message decoding (including LDPC decoder)
    * Pseudorange generation
    * Position/clock-bias calculation and result plotting where implemented
* SIMD/GPU acceleration and correlator options
    * settings.gpuACQflag selects CPU or CuPy GPU-array acquisition.
    * settings.trkMode selects channel-serial or channel-parallel tracking.
    * settings.correlatorType = 0 uses Python/NumPy reference correlators.
    * settings.correlatorType = 1 uses CPU SIMD DLL correlators.
    * settings.correlatorType = 2 uses CUDA GPU DLL correlators.
    * Python/NumPy reference correlators and signal-specific local-code
      generators are implemented in each receiver's correlator.py.
    * Shared DLL correlator backends include BPSK, QPSK, and QMBOC
      SIMD and CUDA variants, called through ctypes.CDLL.
* Supported signals
    * GPS L1 C/A
    * GPS L1C
    * GPS L2C (data + pilot)
    * GPS L5 (data + pilot)
    * Galileo E1 (data + pilot)
    * Galileo E5a (data + pilot)
    * Galileo E5b (data + pilot)
    * Galileo E6B (HAS page/message assembly; no standalone positioning)
    * GLONASS L1OF
    * GLONASS L2OF
    * GLONASS L1OC
    * GLONASS L2OC (pilot tracking only)
    * GLONASS L3OC
    * BeiDou B1I
    * BeiDou B3I
    * BDS-3 B1C (data + pilot)
    * BDS-3 B2a (data + pilot)
    * BeiDou B2b (data)
* RF binary-file post processing
    * Supports real-sample and I/Q-sample IF files configured by
      settings.fileType.
    * Supports int8 and int16 input sample formats configured by
      settings.dataType.
    * LimeSDR complex IF and NUT4NT real-sample examples are documented below;
      match the settings to the recording metadata, not just the file name.
    * The bundled accelerated correlators are Windows x64 DLLs, loaded
      from Python without requiring MATLAB.
	


Directory and Files
-------------------------------------------------------------------------------
Root folders
    ./Doc                     Documentation, ICD material, papers, and receiver
                              summary documents.
    ./IF_Data_Set             Optional location for IF data files and metadata.
    ./native_Correlators      Shared SIMD and CUDA correlator source files
                              and compiled .dll binaries.

Receiver folders
    Each receiver folder is self-contained except for the shared ./native_Correlators,
    ./IF_Data_Set, and ./Doc folders. The current receiver folders are:

    ./pyGPS_L1CA              GPS L1 C/A SDR receiver
    ./pyGPS_L1C               GPS L1C SDR receiver
    ./pyGPS_L2C               GPS L2C SDR receiver
    ./pyGPS_L5                GPS L5 SDR receiver
    ./pyGalileo_E1            Galileo E1 SDR receiver
    ./pyGalileo_E5a           Galileo E5a SDR receiver
    ./pyGalileo_E5b           Galileo E5b SDR receiver
    ./pyGalileo_E6B           Galileo E6B SDR receiver
    ./pyGlonass_L1_L2         GLONASS L1/L2 SDR receiver
    ./pyGlonass_L1OC          GLONASS L1OC SDR receiver
    ./pyGlonass_L2OC          GLONASS L2OC SDR receiver
    ./pyGlonass_L3OC          GLONASS L3OC SDR receiver
    ./pyBDS_B1I               BeiDou B1I/B2I SDR receiver
    ./pyBDS_B3I               BeiDou B3I SDR receiver
    ./pyBDS-3_B1C             BDS-3 B1C SDR receiver
    ./pyBDS-3_B2a             BDS-3 B2a SDR receiver
    ./pyBDS_B2b               BeiDou B2b SDR receiver

Common layout inside each receiver folder
    ./main.py                 Receiver startup script. It selects the receiver
                              directory, loads settings, probes IF data, and
                              asks whether processing should start.
    ./initSettings.py         Settings class for receiver configuration,
                              probeData(), and top-level postProcessing().
    ./acquisition.py          AcqEngine class with CPU and CuPy GPU acquisition,
                              result preparation, and acquisition plotting.
    ./tracking.py             Channel and TrackingEngine classes with
                              channel-serial/channel-parallel tracking and plotting.
    ./postNavigation.py       Ephemeris, NavSolutions, and NavigationEngine classes
                              for navigation decoding and position computation
                              where implemented.
    ./correlator.py           Local-code generation/sampling, Python/NumPy
                              correlators, and ctypes DLL wrapper classes.
    ./channelDecoder.py       LDPC decoder classes and associated BCH decoders;
                              present only in pyBDS-3_B1C, pyBDS-3_B2a,
                              pyBDS_B2b, and pyGPS_L1C.
    ./commUtils.py            Receiver-local common math and positioning utilities:
                              pseudoranges, coordinate transforms, least-squares
                              positioning, loop coefficients, and progress display.
    ./result_Cache            Acquisition, tracking, and navigation/ephemeris
                              results saved as .pkl files. Keep this directory.
                              Galileo E1 also requires E1b.dat/E1c.dat here;
                              E6B requires E6B_codes.dat/E6C_codes.dat here.
    ./assets                  Galileo E6B HAS Reed-Solomon generator matrix;
                              present in pyGalileo_E6B.

Shared correlator backends in ./native_Correlators
    corrSIMDSerial*.cpp       SIMD serial tracking correlator source files.
    corrSIMDParallel*.cpp     SIMD channel-parallel tracking correlator source files.
    corrGPUSerial*.cu         CUDA serial tracking correlator source files.
    corrGPUParallel*.cu       CUDA channel-parallel tracking correlator source files.
    corrGPUParallelFused*.cu  CUDA fused channel-parallel tracking correlator
                              source files where available.
    *.dll                     Compiled DLLs used by settings.correlatorType.
    CMakeLists.txt            Build configuration for all 13 correlator DLLs.
    CMakeSettings.json        Visual Studio CMake Debug/Release configurations.
    ./build                   CMake intermediate files; DLL outputs remain
                              directly in ./native_Correlators.

Correlator selection
    settings.correlatorType = 0
                              Python/NumPy reference correlators in the receiver's
                              correlator.py. These are easiest to inspect
                              and debug.
    settings.correlatorType = 1
                              CPU SIMD DLL correlators from ./native_Correlators.
                              These accelerate tracking without requiring CUDA.
    settings.correlatorType = 2
                              CUDA GPU DLL correlators from ./native_Correlators.
                              These are intended for high-throughput serial or
                              channel-parallel tracking.



Software Dependencies
-------------------------------------------------------------------------------
* Python 3 with NumPy, SciPy, and Matplotlib is required. Using the bundled
  Windows x64 DLLs also requires 64-bit Python.
* SciPy provides signal-processing and numerical utilities; Matplotlib displays
  IF spectra and receiver results.
* Navigation decoders are implemented in postNavigation.py and, for the four
  LDPC-based receivers, channelDecoder.py. MATLAB toolboxes are not required.
* CuPy with a compatible CUDA installation and NVIDIA GPU/driver is needed
  for GPU acquisition. Native CUDA DLL correlators require a compatible
  NVIDIA GPU and driver; CuPy is not needed for the DLL calls themselves.
* An AVX2-capable CPU is needed for SIMD DLL correlators. Rebuilding all DLLs
  with the supplied CMake project requires CMake 4.2 or later, Windows x64,
  Microsoft Visual C++, and a compatible CUDA Toolkit.
  The default CUDA architecture is 86-real; adjust CMAKE_CUDA_ARCHITECTURES
  for other GPUs. Built DLLs are placed directly in native_Correlators.
* Tkinter is used for tracking progress windows and should be available in
  the Python environment.
* CPU-only operation is possible by selecting Python/NumPy or SIMD correlators
  and disabling GPU acquisition in each receiver's initSettings.py.
  
  
 
How to use
-------------------------------------------------------------------------------
* Step 1: Enter the receiver folder for the target signal, for example
          "pyBDS-3_B1C" or "pyGPS_L1CA".
* Step 2: Put the IF data file under "IF_Data_Set", or set an absolute file
          path in settings.fileName.
* Step 3: Configure processing, IF-data, acquisition, tracking, correlator, and
          navigation parameters in "initSettings.py".
          Set skipAcquisition, skipTracking, and skipNavigation to False for a
          fresh run; True reloads the corresponding saved .pkl results.
* Step 4: Select CPU/GPU acquisition with settings.gpuACQflag, channel-serial or
          channel-parallel tracking with settings.trkMode, and NumPy/SIMD/GPU
          correlation with settings.correlatorType.
* Step 5: Run "python main.py" from the receiver folder. Inspect the IF-data
          plots, then enter 1 to start processing (or 0 to exit).
* Step 6: Review acquisition, tracking, navigation, position, and plotting outputs
          generated by Settings.postProcessing() and postNavigation.py.
          Results are saved in the receiver's result_Cache folder as configured
          by acqPklName, trkPklName, and navPklName.



Implementation details
-------------------------------------------------------------------------------
The receiver set follows the SoftGNSS processing architecture and organizes
its Python implementation into modules and classes. It supports modern GNSS
signals, CPU/GPU acquisition, channel-serial and channel-parallel tracking,
and shared SIMD/CUDA DLL correlator backends called through ctypes.CDLL.

See the reference for the BDS-3 B1C/B2a receiver design:
Li, Y., Shivaramaiah, N.C. & Akos, D.M. Design and implementation of an open-source 
BDS-3 B1C/B2a SDR receiver. GPS Solut 23, 60 (2019). 
https://doi.org/10.1007/s10291-019-0853-z


	   
Test signal (collected by LimeSDR) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C.bin
       -- for GPS L1C/A, Galielo E1 and BDS B1C 
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1NdZeVrEdEjDnJiV91Cg0qg?pwd=bvh5 password: bvh5
    * GPS L1C.iq
       -- for GPS L1C 
       -- dataType: int8
       -- fileType: 8 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 25e6 Hz
       -- link: https://pan.baidu.com/s/1hZ4S8R4VWB5OtPr9VyPCYQ?pwd=tv8b password: tv8b
    * GPS_L2.bin
       -- for GPS L2C
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1KqdyWyuVPAgx0EoUdoLwag?pwd=5tya password: 5tya
    * L5_E5a_B2a.bin
       -- for GPS L5, Galielo E5a and BDS B2a
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1rX2lguJgMmgjAdZIJK1FNg?pwd=bnzm password: bnzm
    
    * E5b_B2b.bin   
       -- for Galielo E5b, BDS B2b
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/128dwHIKUG9iM0MigS_7vJw?pwd=x4uq password: x4uq
    * GAL_E6.bin  
       -- for Galielo E6
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1OrHtpNG-dQxSmmVc9rX6tA?pwd=qfx8 password: qfx8
    * GLO_L1OC_L1.bin 
       -- for GLONASS L1OF  
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 1.005e6 Hz
       -- sampling Frequency: 30e6 Hz
       -- GLONASS L1OC
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1w-fYtmM3oW_pxR4ZH_HGyw?pwd=js93 password: js93
    * GLO_L2OC_L2.bin  
       -- for GLONASS L2OF 
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: -2.06e6 Hz
       -- sampling Frequency: 30e6 Hz
    
       -- for GLONASS L2OC 
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1jefXiCkeI9qHSqwZ_RXr2g?pwd=33w6 password: 33w6
    * GLO_L3OC.bin  
       -- for GLONASS L3OC	
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1ySNHn3W66Q6n9fxERR7nOQ?pwd=289m password: 289m
    * BDS_B1I.bin
       -- for BDS B1I 
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1qV-TCtROGHL8iJZlqw5mZA?pwd=39vd password: 39vd
    * BDS_B3I.bin       
       -- for BDS B3I    
       -- dataType: int16
       -- fileType: 16 bit complex samples
       -- IF: 0 Hz
       -- sampling Frequency: 30e6 Hz
       -- link: https://pan.baidu.com/s/1T_g1to1e8OW5OFtHlQsZeg?pwd=6w1u password: 6w1u 



Test signal (collected by NUT4NT) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C_real.bin
       -- for GPS L1 C/A, Galileo E1 and BDS B1C
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -14.58e6 Hz
       -- sampling Frequency: 53e6 Hz
       -- link: https://pan.baidu.com/s/1cFGCBwcWA0vHBeiL7WU_2Q?pwd=r8au password: r8au
    * B1I_real.bin
       -- for BDS B1I
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -28.902e6 Hz
       -- sampling Frequency: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1y4LfFZSO-Qm3Tvshvg_Gzg?pwd=86ii password: 86ii
    * GLO_L1OF_L1OC_real.bin
       -- for GLONASS L1OF
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +12e6 Hz
       -- sampling Frequency: 53e6 Hz
    
       -- for GLONASS L1OC
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +10.995e6 Hz
       -- sampling Frequency: 53e6 Hz
       -- link: https://pan.baidu.com/s/1-fxXe6VW36izl2ve3dMQsw?pwd=j7p9 password: j7p9
    * GLO_L2OF_L2OC_real.bin
       -- for GLONASS L2OF
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -14e6 Hz
       -- sampling Frequency: 70e6 Hz
    
       -- for GLONASS L2OC
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -11.94e6 Hz
       -- sampling Frequency: 70e6 Hz
       -- link: https://pan.baidu.com/s/1P-xWe-aAUu3esymnbsPd-w?pwd=pnr8 password: pnr8
    * L2C_real.bin
       -- for GPS L2C (L2CM and L2CL)
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -7.4e6 Hz
       -- sampling Frequency: 53e6 Hz
       -- link: https://pan.baidu.com/s/16OZPphh6SAObSKf1t2ykPA?pwd=ss7k password: ss7k
    * E5b_B2b_GLO_L3OC_real.bin
       -- for Galileo E5b, BDS B2b
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +17.14e6 Hz
       -- sampling Frequency: 70e6 Hz
    
       -- for GLONASS L3OC
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +12.025e6 Hz
       -- sampling Frequency: 70e6 Hz
       -- link: https://pan.baidu.com/s/1Z-LK5UZ_DQgQk654cY0LgA?pwd=puzw password: puzw
    * L5_E5a_B2a_real.bin
       -- for GPS L5, Galileo E5a and BDS B2a
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: -13.55e6 Hz
       -- sampling Frequency: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1xlftENWf76QwEVzzmxSUDw?pwd=m74w password: m74w
    * B3I_E6_real.bin
       -- for BDS B3I
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +8.52e6 Hz
       -- sampling Frequency: 70e6 Hz
    
       -- for Galileo E6B
       -- dataType: int8
       -- fileType: 8 bit real samples
       -- IF: +18.75e6 Hz
       -- sampling Frequency: 70e6 Hz
       -- link: https://pan.baidu.com/s/14q3K1MKNnAtxz64mUu9DKw?pwd=unn8 password: unn8
