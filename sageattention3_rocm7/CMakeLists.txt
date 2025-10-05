cmake_minimum_required(VERSION 3.21)
project(SageAttention3_ROCm LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(SAGE_ATTN_DEBUG "Enable debug output" OFF)
option(BUILD_TESTS "Build test executables" ON)
option(BUILD_PYTHON_MODULE "Build Python extension module" OFF)

# ROCm setup - Allow override via environment or CMake variable
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm installation")
if(NOT ROCM_PATH)
    if(WIN32)
        set(ROCM_PATH "C:/Program Files/AMD/ROCm/7.0")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

message(STATUS "ROCm Path: ${ROCM_PATH}")

# Find HIP
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

# Find rocWMMA
find_path(ROCWMMA_INCLUDE_DIR
    NAMES rocwmma/rocwmma.hpp
    PATHS ${ROCM_PATH}/include
    REQUIRED
)

message(STATUS "Found rocWMMA headers: ${ROCWMMA_INCLUDE_DIR}")

# Find Python and PyTorch if building Python module
if(BUILD_PYTHON_MODULE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_FIND_RESULT
    )

    if(NOT TORCH_FIND_RESULT EQUAL 0)
        message(FATAL_ERROR "Could not find PyTorch. Please ensure PyTorch is installed.")
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    find_package(Torch REQUIRED)
endif()

# Target architecture
set(GPU_TARGETS "gfx1151" CACHE STRING "GPU targets to compile for (RDNA3.5)")
message(STATUS "GPU Targets: ${GPU_TARGETS}")

# Source files
set(HIP_SOURCES
    hip/attention_forward.hip.cpp
    hip/quantization/int4_ops.hip.cpp
)

# Set source properties for HIP
set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

# Create HIP library target
hip_add_library(sage_attention_rocm7 SHARED ${HIP_SOURCES})

# Include directories
target_include_directories(sage_attention_rocm7 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/include
    ${ROCM_PATH}/include
    ${ROCWMMA_INCLUDE_DIR}
    ${HIP_INCLUDE_DIRS}
    ${ROCBLAS_INCLUDE_DIRS}
)

if(BUILD_PYTHON_MODULE)
    target_include_directories(sage_attention_rocm7 PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
endif()

# Compile definitions
target_compile_definitions(sage_attention_rocm7 PRIVATE
    __HIP_PLATFORM_AMD__=1
    WARP_SIZE=32
)

if(SAGE_ATTN_DEBUG)
    target_compile_definitions(sage_attention_rocm7 PRIVATE SAGE_ATTN_DEBUG=1)
endif()

# HIP compile options
set(HIP_COMPILE_OPTIONS
    --offload-arch=${GPU_TARGETS}
    -O3
    -ffast-math
    -fPIC
    -munsafe-fp-atomics
    -std=c++17
)

# Apply compile options
target_compile_options(sage_attention_rocm7 PRIVATE ${HIP_COMPILE_OPTIONS})

# Set HIP flags globally for hip_add_library
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --offload-arch=${GPU_TARGETS}")
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -O3 -ffast-math -fPIC")
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -munsafe-fp-atomics")
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -std=c++17")

# Link libraries
target_link_libraries(sage_attention_rocm7
    hip::device
    roc::rocblas
)

if(BUILD_PYTHON_MODULE)
    target_link_libraries(sage_attention_rocm7 ${TORCH_LIBRARIES})
endif()

# Windows-specific settings
if(WIN32)
    # Work around Windows SDK library issues
    target_compile_definitions(sage_attention_rocm7 PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )

    # Ensure proper linkage on Windows
    set_target_properties(sage_attention_rocm7 PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    # Add Windows-specific compiler flags
    target_compile_options(sage_attention_rocm7 PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    )
endif()

# Set output name
set_target_properties(sage_attention_rocm7 PROPERTIES
    OUTPUT_NAME "sage_attention_rocm7"
    POSITION_INDEPENDENT_CODE ON
)

# Build test executable
if(BUILD_TESTS)
    set(TEST_SOURCES
        test_int4_quantization.hip.cpp
    )

    set_source_files_properties(${TEST_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

    hip_add_executable(test_int4_quantization ${TEST_SOURCES})

    target_include_directories(test_int4_quantization PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hip/include
        ${ROCM_PATH}/include
        ${ROCWMMA_INCLUDE_DIR}
    )

    target_link_libraries(test_int4_quantization
        sage_attention_rocm7
        hip::device
        roc::rocblas
    )

    target_compile_options(test_int4_quantization PRIVATE ${HIP_COMPILE_OPTIONS})
endif()

# Installation rules
install(TARGETS sage_attention_rocm7
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    hip/quantization/int4_ops.h
    hip/include/common.h
    DESTINATION include/sage_attention_rocm7
)

# Print configuration summary
message(STATUS "=== SageAttention3 ROCm7 Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "HIP Compiler: ${HIP_HIPCC_EXECUTABLE}")
message(STATUS "HIP Version: ${HIP_VERSION}")
message(STATUS "GPU Targets: ${GPU_TARGETS}")
message(STATUS "ROCm Path: ${ROCM_PATH}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Python Module: ${BUILD_PYTHON_MODULE}")
if(BUILD_PYTHON_MODULE)
    message(STATUS "Python: ${Python3_EXECUTABLE}")
    message(STATUS "PyTorch: ${TORCH_LIBRARIES}")
endif()
message(STATUS "Compile Flags: ${HIP_HIPCC_FLAGS}")
message(STATUS "==========================================")
