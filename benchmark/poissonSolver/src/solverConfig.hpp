/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <string_view>

#ifndef ALPAKA_BENCHMARK_POISSON_SOLVER_DIMS
#    define ALPAKA_BENCHMARK_POISSON_SOLVER_DIMS 2u
#endif

namespace poisson
{
    using IdxType = uint32_t;
    using Real = double;

    inline constexpr uint32_t dimensions = static_cast<uint32_t>(ALPAKA_BENCHMARK_POISSON_SOLVER_DIMS);
    static_assert(dimensions >= 1u && dimensions <= 4u, "Poisson solver dimensions must be in [1, 4].");

    using Extent = alpaka::Vec<IdxType, dimensions>;
    using RealVec = alpaka::Vec<Real, dimensions>;

    inline constexpr std::array<std::string_view, 4u> dimensionLabels{"x", "y", "z", "w"};

    ALPAKA_FN_HOST_ACC constexpr auto alpakaDimFromUserDim(uint32_t const userDim) -> uint32_t
    {
        return dimensions - 1u - userDim;
    }
} // namespace poisson
