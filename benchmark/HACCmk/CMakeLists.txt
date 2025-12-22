#
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(HACCmk LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

if(TARGET OpenMP::OpenMP_CXX)
    add_executable(HACCmk_orig "src/main_orig.cpp" "src/Step10_orig.cpp")
    target_include_directories(HACCmk_orig PRIVATE "src")
    target_link_libraries(HACCmk_orig PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(HACCmk_orig PRIVATE Catch2::Catch2WithMain)

    add_executable(HACCmk_sa "src/main_sa.cpp")
    target_include_directories(HACCmk_sa PRIVATE "src")
    target_link_libraries(HACCmk_sa PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(HACCmk_sa PRIVATE Catch2::Catch2WithMain)

    #Run as a ctest
    if(alpaka_CI)
        # Only run for release builds since this is a benchmark
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            add_test(NAME HACCmk_orig COMMAND "HACCmk_orig")
            add_test(NAME HACCmk_sa COMMAND "HACCmk_sa")
        endif()
    else()
        # For a normal benchmark test, number of samples should be equal to the default value.
        add_test(NAME HACCmk_orig COMMAND "HACCmk_orig")
        add_test(NAME HACCmk_sa COMMAND "HACCmk_sa")
    endif()
else()
    message(STATUS "No OpenMP found, HACCmk_orig is not build.")
endif()

add_executable(HACCmk_alpaka "src/main_alpaka.cpp")
target_include_directories(HACCmk_alpaka PRIVATE "src")
target_link_libraries(HACCmk_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(HACCmk_alpaka PRIVATE Catch2::Catch2WithMain)
alpaka_finalize(HACCmk_alpaka)

#Run as a ctest
if(alpaka_CI)
    # Only run for release builds since this is a benchmark
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(NAME HACCmk_alpaka COMMAND "HACCmk_alpaka")
    endif()
else()
    # For a normal benchmark test, number of samples should be equal to the default value.
    add_test(NAME HACCmk_alpaka COMMAND "HACCmk_alpaka")
endif()
