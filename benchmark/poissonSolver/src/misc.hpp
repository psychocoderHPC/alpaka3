/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace poisson
{
    struct Options
    {
        uint32_t sizeX;
        uint32_t sizeY;
        uint32_t maxSteps;
        uint32_t preconditionerMaxSteps;
        double epsilon;
        std::string preconditioner;
    };

    inline constexpr uint32_t defaultSizeX = 64u;
    inline constexpr uint32_t defaultSizeY = 64u;
    inline constexpr uint32_t defaultMaxSteps = 50u;
    inline constexpr uint32_t defaultPreconditionerMaxSteps = 8u;
    inline constexpr double defaultEpsilon = 1.0e-8;
    inline constexpr char defaultPreconditioner[] = "none";

    inline int parseCmd(int argc, char* argv[], Options& options)
    {
        options = Options{
            .sizeX = defaultSizeX,
            .sizeY = defaultSizeY,
            .maxSteps = defaultMaxSteps,
            .preconditionerMaxSteps = defaultPreconditionerMaxSteps,
            .epsilon = defaultEpsilon,
            .preconditioner = defaultPreconditioner};

        Catch::Session session;

        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(options.sizeX, "sizeX")["--size-x"]("Grid size in x direction")
                   | Opt(options.sizeY, "sizeY")["--size-y"]("Grid size in y direction")
                   | Opt(options.maxSteps, "maxSteps")["--max-steps"]("Maximum number of solver iterations")
                   | Opt(options.epsilon, "epsilon")["--epsilon"]("Convergence threshold")
                   | Opt(options.preconditioner, "preconditioner")["--preconditioner"](
                       "Preconditioner to use: none/off or jacobi")
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

        if(options.sizeX == 0u)
        {
            return failPositive("size-x");
        }
        if(options.sizeY == 0u)
        {
            return failPositive("size-y");
        }
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
        if(options.sizeX < 3u)
        {
            std::cerr << "Error: size-x must be at least 3.\n";
            return EXIT_FAILURE;
        }
        if(options.sizeY < 3u)
        {
            std::cerr << "Error: size-y must be at least 3.\n";
            return EXIT_FAILURE;
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

        if(options.preconditioner != "none" && options.preconditioner != "jacobi")
        {
            std::cerr << "Error: preconditioner must be one of: none, off, jacobi.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
} // namespace poisson
