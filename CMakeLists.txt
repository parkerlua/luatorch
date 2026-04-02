cmake_minimum_required(VERSION 3.18)
project(luatorch C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CUDA_STANDARD 17)

# auto detect gpu architecture if not specified
# cmake 3.24+ supports "native", older versions need manual setting
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(DETECTED_GPUS)
    if(DETECTED_GPUS)
        string(REGEX REPLACE "[ \t]+" ";" DETECTED_GPUS "${DETECTED_GPUS}")
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_GPUS})
        message(STATUS "luatorch: auto-detected gpu architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
        message(STATUS "luatorch: could not detect gpu, building for common architectures")
    endif()
endif()

# cpu source files
set(CPU_SOURCES
    csrc/tensor.c
    csrc/ops/arithmetic.c
    csrc/ops/matmul.c
    csrc/ops/activation.c
    csrc/ops/loss.c
)

# cuda source files
set(CUDA_SOURCES
    csrc/cuda/tensor.cu
    csrc/cuda/memory_pool.cu
    csrc/cuda/ops/arithmetic.cu
    csrc/cuda/ops/matmul.cu
    csrc/cuda/ops/activation.cu
    csrc/cuda/ops/loss.cu
    csrc/cuda/ops/fused.cu
    csrc/cuda/ops/cast.cu
    csrc/cuda/ops/flash_attention.cu
)

# find cublas
find_package(CUDAToolkit REQUIRED)

# optional nccl for distributed training
option(LUATORCH_USE_NCCL "Enable NCCL for multi-GPU support" OFF)
set(NCCL_SOURCES "")
if(LUATORCH_USE_NCCL)
    find_library(NCCL_LIBRARY nccl HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
    find_path(NCCL_INCLUDE nccl.h HINTS /usr/include /usr/local/include)
    if(NCCL_LIBRARY AND NCCL_INCLUDE)
        message(STATUS "luatorch: NCCL found, multi-GPU support enabled")
        set(NCCL_SOURCES csrc/distributed/nccl_wrapper.cu)
        add_definitions(-DLUATORCH_HAS_NCCL)
    else()
        message(WARNING "luatorch: NCCL requested but not found, distributed training disabled")
        message(WARNING "  install nccl: sudo apt install libnccl-dev")
    endif()
endif()

# build as shared library
add_library(luatorch SHARED ${CPU_SOURCES} ${CUDA_SOURCES} ${NCCL_SOURCES})

# link libraries
target_link_libraries(luatorch
    CUDA::cudart
    CUDA::cublas
    m
    pthread
)

# link nccl if found
if(NCCL_LIBRARY AND LUATORCH_USE_NCCL)
    target_link_libraries(luatorch ${NCCL_LIBRARY})
    target_include_directories(luatorch PRIVATE ${NCCL_INCLUDE})
endif()

# include paths
target_include_directories(luatorch PRIVATE csrc csrc/cuda csrc/distributed)

# output naming
set_target_properties(luatorch PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "luatorch"
    CUDA_SEPARABLE_COMPILATION ON
)

# install
install(TARGETS luatorch DESTINATION lib)
