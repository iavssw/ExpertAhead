if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH /opt/rocm-7.1.1)
endif()

find_package(hip REQUIRED)
find_package(Torch REQUIRED)

add_executable(torchVShipblaslt torchVShipblaslt.cpp)

target_include_directories(torchVShipblaslt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
    ${TORCH_INCLUDE_DIRS}
)

find_library(HIPBLASLT_LIB hipblaslt PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(NOT HIPBLASLT_LIB)
    message(FATAL_ERROR "hipblaslt not found in ${ROCM_PATH}/lib")
endif()

# Strip invalid C++ flag if present
if(CMAKE_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
if(TORCH_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

target_link_libraries(torchVShipblaslt PRIVATE
    "${TORCH_LIBRARIES}"
    ${HIPBLASLT_LIB}
    amdhip64
)

target_compile_definitions(torchVShipblaslt PRIVATE
    __HIP_PLATFORM_AMD__=1
)

target_compile_options(torchVShipblaslt PRIVATE
    -O3
    -march=native
)

set_target_properties(torchVShipblaslt PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(test_layout_hip test_layout_hip.cpp)
target_include_directories(test_layout_hip PRIVATE
    ${CMAKE_SOURCE_DIR}/src/unified_llm_w4a16_base
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_layout_hip PRIVATE
    unified_llm_w4a16_base_lib
    "${TORCH_LIBRARIES}"
    amdhip64
)
target_compile_definitions(test_layout_hip PRIVATE
    __HIP_PLATFORM_AMD__=1
)
target_compile_options(test_layout_hip PRIVATE
    -O3 -march=native
)
set_target_properties(test_layout_hip PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)

add_executable(test_layout_hip_3d test_layout_hip_3d.cpp)
target_include_directories(test_layout_hip_3d PRIVATE
    ${CMAKE_SOURCE_DIR}/src/unified_llm_w4a16_base
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(test_layout_hip_3d PRIVATE
    unified_llm_w4a16_base_lib
    "${TORCH_LIBRARIES}"
    amdhip64
)
target_compile_definitions(test_layout_hip_3d PRIVATE
    __HIP_PLATFORM_AMD__=1
)
target_compile_options(test_layout_hip_3d PRIVATE
    -O3 -march=native
)
set_target_properties(test_layout_hip_3d PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)



