cmake_minimum_required(VERSION 4.2)

# RTX A5500 uses compute capability 8.6. Override this cache value when
# building for a different GPU, for example: -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86-real CACHE STRING "CUDA architectures to build")
endif()

project(PythonSDRCorrelators LANGUAGES CXX CUDA)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "These correlator DLLs require Windows and Microsoft Visual C++.")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Configure the project for the x64 platform.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Python loads the DLLs from this directory. Keep import libraries, PDB files,
# object files, and generated Visual Studio files inside the build directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pdb")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_CURRENT_SOURCE_DIR}")
    set("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}/lib")
    set("CMAKE_PDB_OUTPUT_DIRECTORY_${config_upper}" "${CMAKE_BINARY_DIR}/pdb")
endforeach()

add_library(corrSIMDSerialBPSK SHARED corrSIMDSerialBPSK.cpp)
add_library(corrSIMDParallelBPSK SHARED corrSIMDParallelBPSK.cpp)
add_library(corrSIMDSerialQPSK SHARED corrSIMDSerialQPSK.cpp)
add_library(corrSIMDParallelQPSK SHARED corrSIMDParallelQPSK.cpp)
add_library(corrSIMDSerialQMBOC SHARED corrSIMDSerialQMBOC.cpp)
add_library(corrSIMDParallelQMBOC SHARED corrSIMDParallelQMBOC.cpp)

foreach(target
        corrSIMDSerialBPSK
        corrSIMDParallelBPSK
        corrSIMDSerialQPSK
        corrSIMDParallelQPSK
        corrSIMDSerialQMBOC
        corrSIMDParallelQMBOC)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/arch:AVX2>)
    set_target_properties(${target} PROPERTIES PREFIX "")
endforeach()

add_library(corrGPUSerialBPSK SHARED corrGPUSerialBPSK.cu)
add_library(corrGPUParallelFusedBPSK SHARED corrGPUParallelFusedBPSK.cu)
add_library(corrGPUParallelBPSK SHARED corrGPUParallelBPSK.cu)
add_library(corrGPUSerialQPSK SHARED corrGPUSerialQPSK.cu)
add_library(corrGPUParallelQPSK SHARED corrGPUParallelQPSK.cu)
add_library(corrGPUSerialQMBOC SHARED corrGPUSerialQMBOC.cu)
add_library(corrGPUParallelQMBOC SHARED corrGPUParallelQMBOC.cu)

foreach(target
        corrGPUSerialBPSK
        corrGPUParallelFusedBPSK
        corrGPUParallelBPSK
        corrGPUSerialQPSK
        corrGPUParallelQPSK
        corrGPUSerialQMBOC
        corrGPUParallelQMBOC)
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Static)
endforeach()

message(STATUS "Correlator DLL output directory: ${CMAKE_CURRENT_SOURCE_DIR}")
