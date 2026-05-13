/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "solverConfig.hpp"

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace poisson
{
    struct Options
    {
        std::array<IdxType, dimensions> sizes;
        uint32_t maxSteps;
        uint32_t preconditionerMaxSteps;
        double epsilon;
        std::string preconditioner;
    };

    inline constexpr uint32_t defaultMaxSteps = 50u;
    inline constexpr uint32_t defaultPreconditionerMaxSteps = 24u;
    inline constexpr double defaultEpsilon = 1.0e-10;
    inline constexpr char defaultPreconditioner[] = "none";

    inline constexpr auto defaultSizes()
    {
        std::array<IdxType, dimensions> values{};
        values.fill(65u);
        return values;
    }

    inline auto addSizeOptions(auto cli, Options& options)
    {
        using namespace Catch::Clara;

        cli = cli | Opt(options.sizes[0], "sizeX")["--size-x"]("Grid size in x direction");
        if constexpr(dimensions >= 2u)
            cli = cli | Opt(options.sizes[1], "sizeY")["--size-y"]("Grid size in y direction");
        if constexpr(dimensions >= 3u)
            cli = cli | Opt(options.sizes[2], "sizeZ")["--size-z"]("Grid size in z direction");
        if constexpr(dimensions >= 4u)
            cli = cli | Opt(options.sizes[3], "sizeW")["--size-w"]("Grid size in w direction");
        return cli;
    }

    inline int parseCmd(int argc, char* argv[], Options& options)
    {
        options = Options{
            .sizes = defaultSizes(),
            .maxSteps = defaultMaxSteps,
            .preconditionerMaxSteps = defaultPreconditionerMaxSteps,
            .epsilon = defaultEpsilon,
            .preconditioner = defaultPreconditioner};

        Catch::Session session;

        using namespace Catch::Clara;

        auto cli = addSizeOptions(session.cli(), options)
                   | Opt(options.maxSteps, "maxSteps")["--max-steps"]("Maximum number of solver iterations")
                   | Opt(options.epsilon, "epsilon")["--epsilon"]("Convergence threshold")
                   | Opt(options.preconditioner, "preconditioner")["--preconditioner"](
                       "Preconditioner to use: none/off, jacobi-iter (alias: jacobi), or chebyshev "
                       "(aliases: g-ci, g-nocomm-ci)")
                   | Opt(options.preconditionerMaxSteps, "preconditionerMaxSteps")["--preconditioner-max-steps"](
                       "Maximum number of preconditioner iterations");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
        {
            return rc;
        }

        auto failPositive = [](char const* name)
        {
            std::cerr << "Error: " << name << " must be greater than zero.\n";
            return EXIT_FAILURE;
        };

        if(options.maxSteps == 0u)
        {
            return failPositive("max-steps");
        }
        if(options.preconditionerMaxSteps == 0u)
        {
            return failPositive("preconditioner-max-steps");
        }
        if(options.epsilon <= 0.0)
        {
            std::cerr << "Error: epsilon must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        for(uint32_t dim = 0u; dim < dimensions; ++dim)
        {
            if(options.sizes[dim] < 3u)
            {
                std::cerr << "Error: size-" << dimensionLabels[dim] << " must be at least 3.\n";
                return EXIT_FAILURE;
            }
        }

        std::transform(
            options.preconditioner.begin(),
            options.preconditioner.end(),
            options.preconditioner.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            });

        if(options.preconditioner == "off")
        {
            options.preconditioner = "none";
        }
        else if(options.preconditioner == "jacobi")
        {
            options.preconditioner = "jacobi-iter";
        }
        else if(options.preconditioner == "diagonal-jacobi")
        {
            options.preconditioner = "jacobi-iter";
        }
        else if(options.preconditioner == "g-ci" || options.preconditioner == "g-nocomm-ci")
        {
            options.preconditioner = "chebyshev";
        }

        if(
            options.preconditioner != "none" && options.preconditioner != "jacobi-iter"
            && options.preconditioner != "chebyshev")
        {
            std::cerr
                << "Error: preconditioner must be one of: none, off, jacobi-iter, jacobi, diagonal-jacobi,"
                   " chebyshev, g-ci, g-nocomm-ci.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
} // namespace poisson
