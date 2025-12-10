#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME pytorchVectorAdd)

project(${_TARGET_NAME} LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

find_package(Torch COMPONENTS pytorch)

if(Torch_FOUND)
    # HIP container setting c++17, so let us correct is even for CUDA or CPU torch too
    set_property(TARGET torch PROPERTY CXX_STANDARD ${alpaka_CXX_STANDARD})
    set_property(TARGET torch_library PROPERTY CXX_STANDARD ${alpaka_CXX_STANDARD})
    if(TARGET alpaka::hip)
        # torch_hip a dependency of torch for AMD is setting in the compile flags hard -std=c++17 so we need to set it again to revert this hard coded standard setting
        target_compile_options(torch_hip INTERFACE -std=c++${alpaka_CXX_STANDARD})
    endif()
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    #-------------------------------------------------------------------------------
    # Add executable.

    add_library(${_TARGET_NAME} SHARED src/vectorAddTorch.cpp)
    target_link_libraries(${_TARGET_NAME} PUBLIC alpaka::alpaka "${TORCH_LIBRARIES}" Python3::Python)
    target_include_directories(${_TARGET_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
    # Required if using AT_DISPATCH macros or registering ops
    target_include_directories(${_TARGET_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}/torch/csrc/api/include")

    alpaka_finalize(${_TARGET_NAME})

    set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
endif()
