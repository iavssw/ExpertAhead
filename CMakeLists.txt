cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(heteroMosaic)

# Check if the environmental variable is set
if(NOT DEFINED ENV{HOME_LIBS})
    message(FATAL_ERROR "HOME_LIBS environmental variable is not set. Please set it before running CMake.")
endif()
# Set CMake variable based on the environmental variable
set(HOME_LIBS "$ENV{HOME_LIBS}")

# Find Python
find_package(Python COMPONENTS Interpreter REQUIRED)

# Try to find Torch from Python environment first
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    message(STATUS "Found PyTorch in Python environment: ${TORCH_PYTHON_PATH}")
else()
    # Fallback to LibTorch path
    list(APPEND CMAKE_PREFIX_PATH "${HOME_LIBS}/libtorch_7.1.0")
    set(TORCH_PYTHON_PATH "${HOME_LIBS}/libtorch_7.1.0")
endif()

# Try to find pybind11 from Python environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    message(STATUS "Found pybind11 CMake directory: ${PYBIND11_CMAKE_DIR}")
endif()

# Set the XDNA driver path
set(DRIVER_PATH "${HOME_LIBS}/xdna-driver-05-19-25")

# Set ROCm path
set(ROCM_PATH "/opt/rocm-7.1.1")
link_directories(${ROCM_PATH}/lib)

# Find required packages
find_package(Torch REQUIRED)

# Set runtime output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include directories for the project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Enable HIP language and set architectures using rocminfo
execute_process(
    COMMAND rocminfo
    OUTPUT_VARIABLE ROCMINFO_OUT
    RESULT_VARIABLE ROCMINFO_RESULT
    ERROR_VARIABLE ROCMINFO_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT ROCMINFO_RESULT EQUAL 0)
    message(FATAL_ERROR "rocminfo failed or is not installed: ${ROCMINFO_ERROR}")
endif()

# Extract GFX version (e.g., gfx90a)
if(ROCMINFO_OUT MATCHES "Name:[ \t]+(gfx[0-9a-f]+)")
    set(CURRENT_GFX "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Could not extract GFX version from rocminfo output")
endif()

# Check for CDNA architecture (gfx90a, gfx908, gfx942, etc.)
# If GFX version starts with "gfx9" or CDNA_DEVICE env var is set, enable CDNA_DEVICE
if(CURRENT_GFX MATCHES "^gfx9" OR "$ENV{CDNA_DEVICE}" STREQUAL "1")
    message(STATUS "CDNA Architecture detected (${CURRENT_GFX}) or forced. Disabling custom kernels.")
    add_compile_definitions(CDNA_DEVICE=1)
endif()

# Whitelist of GFX versions that DON'T need HSA_OVERRIDE
set(WHITELIST "gfx950;gfx1201;gfx1101;gfx1200;gfx1030;gfx942;gfx90a;gfx908;gfx1151")

# Check if current GFX is in whitelist using regex for compatibility
if(";${WHITELIST};" MATCHES ";${CURRENT_GFX};")
    set(HETEROMOSAIC_HIP_ARCH_DEFAULT "${CURRENT_GFX}")
else()
    set(HETEROMOSAIC_HIP_ARCH_DEFAULT "gfx1100")
    set(ENV{HSA_OVERRIDE_GFX_VERSION} "11.0.0")
    message(STATUS "GFX ${CURRENT_GFX} not in whitelist. Setting HSA_OVERRIDE_GFX_VERSION=11.0.0 and targeting gfx1100")
endif()

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "${HETEROMOSAIC_HIP_ARCH_DEFAULT}" CACHE STRING "HIP architectures" FORCE)
endif()
message(STATUS "Using HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
enable_language(HIP)

# Include subdirectories
add_subdirectory(src/hipkernels)
add_subdirectory(src/npu_matmul_func)
add_subdirectory(src/microbenchmarks)

# Implementation libraries
add_subdirectory(src/unified_llm_w4a16_predict)
add_subdirectory(src/unified_llm_w4a16_base)

### PyBind module now lives in src/unified_llm_w4a16_predict
