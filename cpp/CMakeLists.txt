cmake_minimum_required(VERSION 3.24)
project(mnist_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(MNIST_ENABLE_CUDA "Build the CUDA engine when a CUDA compiler is found" ON)

add_executable(mnist_infer src/main.cpp src/model_io.cpp src/cpu_engine.cpp)
target_include_directories(mnist_infer PRIVATE include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_infer PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(mnist_infer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:fast>)
else()
    target_compile_options(mnist_infer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
endif()

if(MNIST_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER)
    # "native" = the GPU in this machine. Override with e.g. -DCMAKE_CUDA_ARCHITECTURES=86
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(mnist_infer PRIVATE src/cuda_engine.cu)
    target_compile_definitions(mnist_infer PRIVATE MNIST_HAVE_CUDA)
    # Debug: device debug info (-G) so Nsight can step through kernels.
    # Release: optimised, with line info so Nsight Compute can map time to source lines.
    target_compile_options(mnist_infer PRIVATE
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3;-lineinfo>")
    target_link_libraries(mnist_infer PRIVATE CUDA::cudart)
    message(STATUS "CUDA engine: enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA engine: disabled (no CUDA compiler found)")
endif()
