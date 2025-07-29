/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include "gpu_integrator.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <typeinfo>

using namespace alpaka;

constexpr uint64_t init_seed = 147634;
// Numerical Recipes, ranqd1
constexpr uint64_t rand_modulus = (uint64_t(1) << 32);
constexpr uint64_t rand_multiplier = 1'664'525;
constexpr uint64_t rand_increment = 1'013'904'223;

uint64_t randc(uint64_t& seed)
{
    seed = (rand_multiplier * seed + rand_increment) & (rand_modulus - 1);
    return seed;
}

double uniform_rand(uint64_t& seed, double min, double max)
{
    auto val = randc(seed) / ((double) rand_modulus) * (max - min) + min;
    return val;
}

void set_params(
    size_t size,
    size_t band_width,
    double min,
    double max,
    concepts::MdSpan auto amplitude_lincomb,
    concepts::MdSpan auto t_offset,
    concepts::MdSpan auto t_scale)
{
    uint64_t seed = init_seed;
    for(size_t i = 0; i < size; i++)
    {
        t_offset[i] = uniform_rand(seed, min, max);
        t_scale[i] = uniform_rand(seed, min, max);
        // amplitude[i] = uniform_rand(min, max);

        for(int j = -((int) band_width / 2); j < (((int) band_width + 1) / 2); j++)
        {
            if((int) i + j >= 0 && i + j < size)
                amplitude_lincomb[Vec{i, i + j}] = uniform_rand(seed, min, max);
        }
    }
}

struct Rhs
{
    ALPAKA_FN_ACC void operator()(
        [[maybe_unused]] auto const& acc,
        size_t x_size,
        double t,
        concepts::MdSpan auto dxdt,
        concepts::MdSpan auto amplitude_lincomb,
        concepts::MdSpan auto t_offset,
        concepts::MdSpan auto t_scale,
        size_t y) const
    {
#if USE_ALPAKA
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{x_size}))
        {
#    if 1
            // we need to select only one row of the amplitude, currently there is no shortcut for it
            auto shiftedAmplitude = makeMdSpan(
                &amplitude_lincomb[alpaka::Vec{i, 0}],
                alpaka::Vec{1, x_size},
                amplitude_lincomb.getPitches(),
                amplitude_lincomb.getAlignment());
            // single thread is doing the transform reduce
            auto allThreads = onAcc::SimdAlgo{onAcc::WorkerGroup{Vec<int, 2>::all(0), Vec<int, 2>::all(1)}};
            auto ret = allThreads.transformReduce(
                acc,
                alpaka::Vec{0, x_size},
                double{0},
                std::plus{},
                [&](auto const&, auto&& amplitudePtr) constexpr
                {
                    auto packageOffset = amplitudePtr.getIdx();
                    ALPAKA_TYPEOF(amplitudePtr.load())
                    sinSimd(
                        [&](auto const& w) constexpr
                        {
                            auto jIdx = packageOffset.x() + w;
                            return math::sin(t * t_scale[jIdx] + t_offset[jIdx]);
                        });

                    return amplitudePtr.load() * sinSimd;
                },
                shiftedAmplitude);

#    else
            double ret = 0.0;
            for(size_t j = 0; j < x_size; j++)
            {
                ret += amplitude_lincomb[Vec{i, j}] * math::sin(t * t_scale[j] + t_offset[j]);
            }
#    endif

            dxdt[Vec{y, i}] = ret;
        }
#else
        for(size_t i = 0; i < x_size; i++)
        {
            // dxdt[i] = amplitude[i] * std::sin(t * t_scale[i] + t_offset[i]);

            dxdt[Vec{y, i}] = 0;
            for(size_t j = 0; j < x_size; j++)
            {
                dxdt[Vec{y, i}] += amplitude_lincomb[Vec{i, j}] * std::sin(t * t_scale[j] + t_offset[j]);
            }
        }
#endif
    }
};

namespace mio
{
    void log_debug(std::string_view s)
    {
        std::cout << s << "\n";
    }
} // namespace mio

auto example(auto const deviceSpec, auto const exec, int numElements) -> int
{
    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

#if USE_ALPAKA
    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << deviceSpec.getApi().getName()
              << " on " << onHost::getName(devAcc) << std::endl;
#endif

    mio::log_debug("Enter the world of memilio");

    int const size = numElements, band_width = 2;

    auto t_offset = onHost::allocHost<double>(size);
    auto t_scale = onHost::allocHost<double>(size);

    auto amplitude_lincomb = onHost::allocHost<double>(Vec{size, size});
    onHost::memset(queue, t_offset, 0);
    onHost::memset(queue, t_scale, 0);
    onHost::memset(queue, amplitude_lincomb, 0);
    onHost::wait(queue);

    set_params(size, band_width, -3.0, 3.0, amplitude_lincomb.getMdSpan(), t_offset.getMdSpan(), t_scale.getMdSpan());
    mio::log_debug("Params Set");

    auto t_offset_dev = onHost::allocMirror(devAcc, t_offset);
    auto t_scale_dev = onHost::allocMirror(devAcc, t_scale);
    auto amplitude_lincomb_dev = onHost::allocMirror(devAcc, amplitude_lincomb);
    onHost::memcpy(queue, t_offset_dev, t_offset);
    onHost::memcpy(queue, t_scale_dev, t_scale);
    onHost::memcpy(queue, amplitude_lincomb_dev, amplitude_lincomb);

    onHost::wait(queue);

    double const abs_tol = 1e-3, rel_tol = 1e-8, min_dt = 1e-2, max_dt = 1e+2;

    auto m_kt_values = onHost::allocHost<double>(Vec{size_t{tableau().entries_low.dim()}, size});
    auto m_kt_values_dev = onHost::allocMirror(devAcc, m_kt_values);

    Monstrosity stepper{
        exec,
        queue,
        abs_tol,
        rel_tol,
        min_dt,
        max_dt,
        std::vector<double>(size),
        std::vector<double>(size),
        m_kt_values,
        m_kt_values_dev};

    mio::log_debug("Core Set");

    // OdeIntegrator<double> integrator(core);

    mio::log_debug("Integrator Set");

    // TimeSeries<double> results(0, Eigen::VectorXd::Zero(size));

    double dt = 0.1;

    double t = 0.0;
    std::vector<double> x(size, 0.0);
    std::vector<double> x2(size, 0.0);

    std::cout << "\n";

    mio::log_debug("Results Set");
    mio::log_debug("Integrating...");


    auto const beginT = std::chrono::high_resolution_clock::now();
    while(t < 100 * M_PI)
    {
        stepper.step(Rhs{}, x, t, dt, x2, amplitude_lincomb_dev, t_offset_dev, t_scale_dev);
        for(size_t i = 0; i < size; i++)
        {
            x[i] = x2[i];
            x2[i] = 0;
        }
    }
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << std::endl;

    mio::log_debug("Integration Finished");

    double result = 0.0;
    for(size_t i = 0; i < x.size(); ++i)
        result += x[i];


    mio::log_debug("Exit Main");

    if(size == 100)
    {
        double expected = -861.197;
        std::cout << "result " << result << " abs(error)=" << std::abs(expected - result) << std::endl;

        return std::abs(expected - result) < 1e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    else
    {
        std::cout << "result " << result << " NOT validated" << std::endl;
        return EXIT_SUCCESS;
    }
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    size_t numElements = 100;

    int opt;
    while((opt = getopt(argc, argv, "hn:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }
#if USE_ALPAKA
    // Execute the example with all backends
    return executeForEachIfHasDevice(
        [=](auto const& backend) { return example(backend[object::deviceSpec], backend[object::exec], numElements); },
        onHost::allBackends(onHost::enabledApis));
#else
    example(onHost::DeviceSpec{api::host, deviceKind::cpu}, exec::cpuSerial, numElements);
#endif
}
