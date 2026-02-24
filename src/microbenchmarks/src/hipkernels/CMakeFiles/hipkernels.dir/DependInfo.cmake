
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/home/michael/heteroPredict/src/hipkernels/embedding.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/embedding.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/flash_attn_decode.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/flash_attn_decode.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/lm_head.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/lm_head.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/rmsnorm.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/rmsnorm.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/rope.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/rope.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/w4a16_gemm_unpacked.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/w4a16_gemm_unpacked.cu.o"
  "/home/michael/heteroPredict/src/hipkernels/w4a16_gemv_unpacked.cu" "/home/michael/heteroPredict/src/microbenchmarks/src/hipkernels/CMakeFiles/hipkernels.dir/w4a16_gemv_unpacked.cu.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "HIPBLASLT_USE_ROCROLLER"
  "RDNA3=1"
  "USE_C10D_GLOO"
  "USE_C10D_NCCL"
  "USE_DISTRIBUTED"
  "USE_PROF_API=1"
  "USE_RPC"
  "USE_TENSORPIPE"
  "__HIP_PLATFORM_AMD__"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/home/michael/heteroPredict/include"
  "/opt/rocm/include/hip"
  "/home/michael/libraries/libtorch_7.1.0/include"
  "/home/michael/libraries/libtorch_7.1.0/include/torch/csrc/api/include"
  "/opt/rocm-7.1.1/include/hiprand"
  "/opt/rocm-7.1.1/include/rocrand"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
