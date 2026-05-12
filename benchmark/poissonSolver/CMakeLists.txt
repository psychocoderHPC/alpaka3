#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(poissonSolver LANGUAGES CXX)

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

set(_POISSON_SOLVER_MAIN "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp")

if(EXISTS "${_POISSON_SOLVER_MAIN}")
    add_executable(poissonSolver "${_POISSON_SOLVER_MAIN}")
    target_include_directories(poissonSolver PRIVATE "src")
    target_link_libraries(poissonSolver PRIVATE alpaka::alpaka Catch2::Catch2)
    alpaka_finalize(poissonSolver)
    set_target_properties(poissonSolver PROPERTIES FOLDER benchmarks/poissonSolver)

    add_test(NAME poissonSolver COMMAND "poissonSolver")
else()
    message(STATUS "Skipping benchmark target poissonSolver because src/main.cpp is not present yet.")
endif()
