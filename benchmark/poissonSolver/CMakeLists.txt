#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(poissonSolver LANGUAGES CXX)

set(
    alpaka_BENCHMARK_POISSON_SOLVER_DIMS
    "2"
    CACHE STRING "Compile-time number of dimensions for the Poisson solver benchmark"
)
set_property(CACHE alpaka_BENCHMARK_POISSON_SOLVER_DIMS PROPERTY STRINGS 1 2 3 4)

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
    target_compile_definitions(
        poissonSolver
        PRIVATE ALPAKA_BENCHMARK_POISSON_SOLVER_DIMS=${alpaka_BENCHMARK_POISSON_SOLVER_DIMS}
    )
    target_link_libraries(poissonSolver PRIVATE alpaka::alpaka Catch2::Catch2)
    alpaka_finalize(poissonSolver)
    set_target_properties(poissonSolver PROPERTIES FOLDER benchmarks/poissonSolver)

    if(alpaka_BENCHMARK_POISSON_SOLVER_DIMS STREQUAL "1")
        add_test(
            NAME poissonSolver
            COMMAND "poissonSolver" --size-x 10001 --max-steps 400 --epsilon 1e-7 --preconditioner chebyshev
                    --preconditioner-max-steps 24
        )
    elseif(alpaka_BENCHMARK_POISSON_SOLVER_DIMS STREQUAL "2")
        add_test(NAME poissonSolver COMMAND "poissonSolver" --size-x 321 --size-y 321 --max-steps 200 --epsilon 1e-8)
        add_test(
            NAME poissonSolver_jacobi
            COMMAND "poissonSolver" --size-x 257 --size-y 193 --max-steps 200 --epsilon 1e-8 --preconditioner jacobi
                    --preconditioner-max-steps 4
        )
    elseif(alpaka_BENCHMARK_POISSON_SOLVER_DIMS STREQUAL "3")
        add_test(
            NAME poissonSolver
            COMMAND "poissonSolver" --size-x 65 --size-y 49 --size-z 33 --max-steps 200 --epsilon 1e-8
        )
    elseif(alpaka_BENCHMARK_POISSON_SOLVER_DIMS STREQUAL "4")
        add_test(
            NAME poissonSolver
            COMMAND "poissonSolver" --size-x 33 --size-y 25 --size-z 17 --size-w 9 --max-steps 200 --epsilon 1e-8
        )
    endif()
else()
    message(STATUS "Skipping benchmark target poissonSolver because src/main.cpp is not present yet.")
endif()
