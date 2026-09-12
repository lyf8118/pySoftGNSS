An Open-Source Python GNSS SDR Toolbox with SIMD/GPU Acceleration： pySoftGNSS 
===============================================================================



Overview
-------------------------------------------------------------------------------
pySoftGNSS是一个基于Python的开源工具箱，用于对录制的GNSS中频（IF）信号进行后处理。
它与mSoftGNSS遵循共同的接收机处理架构，并以Python模块和类组织信号处理功能。
工具箱支持GPS、Galileo、GLONASS 和北斗信号，根据各信号已实现的处理能力，提供捕获、
数据/导频跟踪、导航电文解码、伪距生成和定位功能。CPU和GPU捕获分别使用NumPy/SciPy和CuPy。
通道串行和通道并行两种跟踪模式均支持NumPy相关器、SIMD加速相关器和GPU加速相关器。
通道并行跟踪复用共享IF数据缓冲，并批量执行各通道的相关任务。C++和CUDA C++相关器编译为
动态链接库，通过ctypes.CDLL调用；调度、跟踪环路控制、导航解码和定位仍在Python中完成。
这种设计在保留高层处理可读性和可配置性的同时，加速了计算密集的样本级运算。
结合现代化信号支持、LDPC解码和集成的可视化功能，pySoftGNSS为GNSS算法开发、
接收机原型开发和可重复评估提供了可扩展的平台。



Authors
-------------------------------------------------------------------------------
* Yafeng Li
    * E-Mail: <lyf8118@126.com>
    * Wechat: lyf8118521
    * GNSS软件接收机技术讨论QQ群：147304049


* Dennis Akos  
    * E-Mail: <dma@colorado.edu>
    * HP: <http://www.colorado.edu/aerospace/dennis-akos>




Features
-------------------------------------------------------------------------------
* 基于 Python 实现的 GNSS 信号处理功能
    * 本地码生成
    * CPU 和 GPU 捕获
    * 通道串行和通道并行跟踪
    * Python/NumPy、SIMD DLL 和 CUDA DLL 相关器
    * 现代化信号的数据/导频跟踪
    * 导航电文解码（包括 LDPC 解码器）
    * 伪距生成
    * 按已实现的处理范围进行位置/钟差解算和结果绘图
* SIMD/GPU 加速和相关器选项
    * settings.gpuACQflag 用于选择 CPU 捕获或 CuPy GPU-array 捕获。
    * settings.trkMode 用于选择通道串行跟踪或通道并行跟踪。
    * settings.correlatorType = 0 表示使用 Python/NumPy 参考相关器。
    * settings.correlatorType = 1 表示使用 CPU SIMD DLL 相关器。
    * settings.correlatorType = 2 表示使用 CUDA GPU DLL 相关器。
    * Python/NumPy 参考相关器和各信号的本地码生成函数，
      位于各接收机的 correlator.py 中。
    * 共用 DLL 相关器后端包含 BPSK、QPSK 和 QMBOC 的
      SIMD 和 CUDA 实现，通过 ctypes.CDLL 调用。
* 支持的信号
    * GPS L1 C/A
    * GPS L1C
    * GPS L2C (data + pilot)
    * GPS L5 (data + pilot)
    * Galileo E1 (data + pilot)
    * Galileo E5a (data + pilot)
    * Galileo E5b (data + pilot)
    * Galileo E6B（HAS 页/电文组帧，不独立定位）
    * GLONASS L1OF
    * GLONASS L2OF
    * GLONASS L1OC
    * GLONASS L2OC（仅导频跟踪）
    * GLONASS L3OC
    * BeiDou B1I/B2I
    * BeiDou B3I
    * BDS-3 B1C (data + pilot)
    * BDS-3 B2a (data + pilot)
    * BeiDou B2b (data)
* 支持 RF 二进制文件后处理
    * 通过 settings.fileType 配置实采样或 I/Q 复采样 IF 数据。
    * 通过 settings.dataType 配置 int8 或 int16 输入采样格式。
    * README 后面保留 LimeSDR 复数 IF 和 NUT4NT 实采样数据示例；
      参数应与录制数据的元数据一致，不能只修改文件名。
    * 随附的加速相关器为 Windows x64 DLL，由 Python 加载，
      不需要 MATLAB。



Directory and Files
-------------------------------------------------------------------------------
根目录
    ./Doc                     文档、ICD、论文和各接收机说明材料。
    ./IF_Data_Set             可放置待处理 IF 数据及对应 metadata 文件。
    ./native_Correlators      所有接收机共用的 SIMD/CUDA 相关器源码和
                              已编译的 .dll 文件。

接收机目录
    每个接收机目录都是相对独立的软件接收机；除 ./native_Correlators、
    ./IF_Data_Set 和 ./Doc 外，各接收机代码互不依赖。当前接收机目录如下：

    ./pyGPS_L1CA              GPS L1 C/A 软件接收机
    ./pyGPS_L1C               GPS L1C 软件接收机
    ./pyGPS_L2C               GPS L2C 软件接收机
    ./pyGPS_L5                GPS L5 软件接收机
    ./pyGalileo_E1            Galileo E1 软件接收机
    ./pyGalileo_E5a           Galileo E5a 软件接收机
    ./pyGalileo_E5b           Galileo E5b 软件接收机
    ./pyGalileo_E6B           Galileo E6B 软件接收机
    ./pyGlonass_L1_L2         GLONASS L1/L2 软件接收机
    ./pyGlonass_L1OC          GLONASS L1OC 软件接收机
    ./pyGlonass_L2OC          GLONASS L2OC 软件接收机
    ./pyGlonass_L3OC          GLONASS L3OC 软件接收机
    ./pyBDS_B1I               北斗 B1I软件接收机
    ./pyBDS_B3I               北斗 B3I 软件接收机
    ./pyBDS-3_B1C             北斗三号 B1C 软件接收机
    ./pyBDS-3_B2a             北斗三号 B2a 软件接收机
    ./pyBDS_B2b               北斗 B2b 软件接收机

每个接收机目录的通用结构
    ./main.py                 接收机启动脚本：切换到接收机目录、读取配置、探测 IF
                              数据，并询问是否启动处理。
    ./initSettings.py         Settings 类：接收机参数配置、probeData() 和顶层
                              postProcessing() 处理调度。
    ./acquisition.py          AcqEngine 类：CPU/CuPy GPU 捕获、结果整理和捕获绘图。
    ./tracking.py             Channel 和 TrackingEngine 类：通道串行/并行跟踪和绘图。
    ./postNavigation.py       Ephemeris、NavSolutions 和 NavigationEngine 类：
                              按已实现的处理范围进行导航电文解码和定位解算。
    ./correlator.py           本地码生成/采样、Python/NumPy 相关器和 ctypes DLL 接口类。
    ./channelDecoder.py       LDPC 解码器及相关 BCH 解码器类；仅存在于
                              pyBDS-3_B1C、pyBDS-3_B2a、pyBDS_B2b 和 pyGPS_L1C。
    ./commUtils.py            当前接收机本地公共数学/定位工具函数：伪距计算、
                              坐标转换、最小二乘定位、环路参数计算和进度显示。
    ./result_Cache            保存捕获、跟踪和导航/星历结果的 .pkl 文件，请保留该目录。
                              Galileo E1 还需保留其中的 E1b.dat/E1c.dat；
                              E6B 还需保留其中的 E6B_codes.dat/E6C_codes.dat。
    ./assets                  Galileo E6B HAS 的 Reed-Solomon 生成矩阵，
                              位于 pyGalileo_E6B 中。

./native_Correlators 中的共用相关器后端
    corrSIMDSerial*.cpp       SIMD 通道串行跟踪相关器源码。
    corrSIMDParallel*.cpp     SIMD 通道并行跟踪相关器源码。
    corrGPUSerial*.cu         CUDA 通道串行跟踪相关器源码。
    corrGPUParallel*.cu       CUDA 通道并行跟踪相关器源码。
    corrGPUParallelFused*.cu  CUDA fused 通道并行跟踪相关器源码。
    *.dll                     已编译 DLL 文件，由 settings.correlatorType 选择调用。
    CMakeLists.txt            全部 13 个相关器 DLL 的编译配置。
    CMakeSettings.json        Visual Studio CMake 的 Debug/Release 配置。
    ./build                   CMake 编译中间文件；DLL 仍直接输出到
                              ./native_Correlators 中。

相关器选择
    settings.correlatorType = 0
                              使用各接收机 correlator.py 中的 Python/NumPy 参考相关器，
                              便于阅读、调试和对比。
    settings.correlatorType = 1
                              使用 ./native_Correlators 中的 CPU SIMD DLL 相关器，
                              不依赖 CUDA 即可加速跟踪。
    settings.correlatorType = 2
                              使用 ./native_Correlators 中的 CUDA GPU DLL 相关器，
                              用于高吞吐的通道串行或通道并行跟踪。

Software Dependencies
-------------------------------------------------------------------------------
* 需要 Python 3、NumPy、SciPy 和 Matplotlib。使用随附的 Windows x64 DLL
  还需要 64 位 Python。
* SciPy 提供信号处理和数值计算工具；Matplotlib 用于 IF 频谱和接收机结果绘图。
* 导航解码在 postNavigation.py 中实现；四个使用 LDPC 的接收机还使用
  channelDecoder.py，不需要 MATLAB 工具箱。
* GPU 捕获需要 CuPy、兼容的 CUDA 环境及 NVIDIA GPU/驱动。
  原生 CUDA DLL 相关器需要兼容的 NVIDIA GPU 和驱动；
  DLL 调用本身不依赖 CuPy。
* SIMD DLL 需要支持 AVX2 的 CPU。使用随附 CMake 工程重新编译全部 DLL，
  需要 CMake 4.2 或以上、Windows x64、Microsoft Visual C++ 和兼容的 CUDA Toolkit。
  默认 CUDA 架构为 86-real；其他 GPU 应调整 CMAKE_CUDA_ARCHITECTURES。
  编译生成的 DLL 直接放在 native_Correlators 中。
* 跟踪进度窗口使用 Tkinter，Python 环境中需提供该组件。
* 如果只需要 CPU 运行，可在各接收机 initSettings.py 中关闭 GPU 捕获，并选择
  Python/NumPy 相关器或 SIMD 相关器。
  
  
 
How to use
-------------------------------------------------------------------------------
* Step 1: 进入目标信号对应的接收机目录，例如 "pyBDS-3_B1C" 或 "pyGPS_L1CA"。
* Step 2: 将 IF 数据放入 "IF_Data_Set"，或在 settings.fileName 中设置数据文件
          的绝对路径。
* Step 3: 在 "initSettings.py" 中配置处理时长、IF 数据、捕获、跟踪、相关器和
          导航解算参数。重新处理时，将 skipAcquisition、skipTracking 和
          skipNavigation 设为 False；设为 True 时读取对应的 .pkl 结果。
* Step 4: 通过 settings.gpuACQflag 选择 CPU/GPU 捕获，通过 settings.trkMode
          选择通道串行/并行跟踪，通过 settings.correlatorType 选择 Python/NumPy、
          SIMD DLL 或 GPU DLL 相关器。
* Step 5: 在当前接收机目录下运行 "python main.py"，查看 IF 数据图后，
          输入 1 启动处理，或输入 0 退出。
* Step 6: 通过 Settings.postProcessing() 和 postNavigation.py 生成并查看捕获、
          跟踪、导航、定位和绘图结果。结果按 acqPklName、trkPklName 和
          navPklName 的配置保存在当前接收机的 result_Cache 目录中。



Implementation details
-------------------------------------------------------------------------------
本软件接收机套件遵循 SoftGNSS 的处理架构，以 Python 模块和类组织实现，
支持多种现代化 GNSS 信号、CPU/GPU 捕获、通道串行/并行跟踪，以及通过
ctypes.CDLL 调用的共用 SIMD/CUDA DLL 相关器后端。

BDS-3 B1C/B2a 接收机设计可参考：
Li, Y., Shivaramaiah, N.C. & Akos, D.M. Design and implementation of an open-source
BDS-3 B1C/B2a SDR receiver. GPS Solut (2019) 23: 60.
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
