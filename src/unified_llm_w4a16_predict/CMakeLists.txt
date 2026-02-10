set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/rocm-7.1.1/include")

add_library(unified_llm_w4a16_hetero_lib STATIC
    unified_llm_w4a16.cpp
    npuSetup.cpp
    helper.cpp
)

# Set ROCM_PATH if not defined
# Set ROCM_PATH if not defined
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH /opt/rocm-7.1.1)
endif()

find_package(hip REQUIRED)

message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")

target_include_directories(unified_llm_w4a16_hetero_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    /opt/rocm-7.1.1/include
)


include_directories(
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
)
target_include_directories(unified_llm_w4a16_hetero_lib PUBLIC
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
)

# Dependencies
find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)

target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC 
    "${TORCH_LIBRARIES}"
    npu_matmul_func
    hipkernels
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(unified_llm_w4a16_hetero_lib PUBLIC 
    __HIP_PLATFORM_AMD__=1
)

target_compile_options(unified_llm_w4a16_hetero_lib PRIVATE 
    -march=native
    -O3
    -march=native
    -O3
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(unified_llm_w4a16_hetero_lib PROPERTIES CXX_STANDARD 17)

# PyBind module (combined into this directory)
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get PyTorch path from Python environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Try to find Torch from Python environment first
if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
    message(STATUS "Found PyTorch in Python environment: ${TORCH_PYTHON_PATH}")
endif()

# Find Torch
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch not found in Python environment, using standalone LibTorch")
endif()

# Strip invalid C++ flag if present
if(CMAKE_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
if(TORCH_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# Get pybind11 include directory
get_target_property(PYBIND11_INCLUDE_DIR pybind11::headers INTERFACE_INCLUDE_DIRECTORIES)
if(NOT PYBIND11_INCLUDE_DIR)
    get_target_property(PYBIND11_INCLUDE_DIR pybind11::pybind11 INTERFACE_INCLUDE_DIRECTORIES)
endif()
if(NOT PYBIND11_INCLUDE_DIR)
    set(PYBIND11_INCLUDE_DIR ${Python_INCLUDE_DIRS})
endif()

# Create Python module
add_library(unified_llm_w4a16_hetero_libtorch MODULE
    python_bindings.cpp
)

# Set Python module properties
set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
    OUTPUT_NAME "unified_llm_w4a16_hetero_libtorch"
)

# Include directories
target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${DRIVER_PATH}/src/include/uapi/drm_local
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add PyTorch include directories if found in Python environment
if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

# Link libraries
if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    link_directories("${TORCH_PYTHON_PATH}/lib")
endif()
link_directories(${ROCM_PATH}/lib)

find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(NOT HSA_RUNTIME_LIB)
    message(FATAL_ERROR "hsa-runtime64 not found in ${ROCM_PATH}/lib")
endif()

target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE
    unified_llm_w4a16_hetero_lib
    "${TORCH_LIBRARIES}"
    pybind11::module
    pybind11::windows_extras
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
)
# Note: Don't use pybind11::lto with HIP - causes linking issues

# Manually find and link torch_python for torch/extension.h support
if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PYTHON_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIBRARY)
        target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE ${TORCH_PYTHON_LIBRARY})
        message(STATUS "Found torch_python library: ${TORCH_PYTHON_LIBRARY}")
    else()
        message(WARNING "torch_python library not found in ${TORCH_PYTHON_PATH}/lib")
    endif()
endif()

# Set rpath
if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH "${TORCH_PYTHON_PATH}/lib"
        BUILD_RPATH "${TORCH_PYTHON_PATH}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
else()
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Set compile definitions
target_compile_definitions(unified_llm_w4a16_hetero_libtorch PRIVATE
    VERSION_INFO=${pybind11_VERSION}
    __HIP_PLATFORM_AMD__=1
)

# Set output directory
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16")
set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
)
