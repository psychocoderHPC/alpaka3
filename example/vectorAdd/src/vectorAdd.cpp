/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <typeinfo>

#define R123_NO_CUDA_DEVICE_RANDOM 1

#include "gpu_integrator.h"

#include <cmath>
#include <iostream>
#include <memory>

using namespace alpaka;

constexpr uint64_t init_seed = 147634;
// Numerical Recipes, ranqd1
constexpr uint64_t rand_modulus = (uint64_t(1) << 32);
constexpr uint64_t rand_multiplier = 1'664'525;
constexpr uint64_t rand_increment = 1'013'904'223;

constexpr size_t problem_size = 100;

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
    alpaka::concepts::MdSpan auto amplitude_lincomb,
    alpaka::concepts::MdSpan auto t_offset,
    alpaka::concepts::MdSpan auto t_scale)
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
        alpaka::concepts::MdSpan auto dxdt,
        alpaka::concepts::MdSpan auto amplitude_lincomb,
        alpaka::concepts::MdSpan auto t_offset,
        alpaka::concepts::MdSpan auto t_scale,
        size_t y) const
    {
#if USE_ALPAKA
        for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{x_size}))
        {
            double ret = 0.0;
            for(size_t j = 0; j < x_size; j++)
            {
                ret += amplitude_lincomb[Vec{i, j}] * alpaka::math::sin(t * t_scale[j] + t_offset[j]);
            }
            dxdt[alpaka::Vec{y, i}] = ret;
        }
#else
        for(size_t i = 0; i < nProblem; i++)
        {
            // dxdt[i] = amplitude[i] * std::sin(t * t_scale[i] + t_offset[i]);

            dxdt[alpaka::Vec{y, i}] = 0;
            for(size_t j = 0; j < nProblem; j++)
            {
                dxdt[alpaka::Vec{y, i}] += amplitude_lincomb[Vec{i, j}] * std::sin(t * t_scale[j] + t_offset[j]);
            }
        }
#endif
    }
};

// void rhs2(Eigen::Ref<const Eigen::VectorXd> x, double t, Eigen::Ref<Eigen::VectorXd> dxdt) {
//     for (size_t i = 0; i < x.size(); i++) {
//         // dxdt[i] = amplitude[i] * std::sin(t * t_scale[i] + t_offset[i]);

//         dxdt[i] = 0;
//         for (size_t j = 0; j < x.size(); j++) {
//             dxdt[i] += amplitude_lincomb[i][j] * std::sin(t * t_scale[j] + t_offset[j]);
//         }
//     }
// }


namespace mio
{
    void log_debug(std::string_view s)
    {
        std::cout << s << "\n";
    }
} // namespace mio

template<typename T_Cfg>
auto example(T_Cfg const& cfg)
{
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];


    // Select a device
    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Get the host device for allocating memory on the host.
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);
#if USE_ALPAKA
    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << " on "
              << alpaka::onHost::getName(devAcc) << std::endl;
#endif

    // using namespace mio;
    // set_log_level(LogLevel::off);

    mio::log_debug("Enter the world of memilio");

    int const size = problem_size, band_width = 2;

    // // Guard the CUDA test with proper CUDA error handling
    // // cudaError_t cudaStatus = cudaSetDevice(0);
    // // if (cudaStatus != cudaSuccess) {
    // //     std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    // //     std::cout << "CUDA test failed! Continuing without CUDA." << std::endl;
    // // }
    // // else {
    // //     std::cout << "CUDA initialization succeeded." <<  std::endl;

    // //     cudaDeviceReset();
    // // }

    // // TODO: nvidia-x-markers??

    auto t_offset = onHost::alloc<double>(devHost, size);
    auto t_scale = onHost::alloc<double>(devHost, size);

    auto amplitude_lincomb = onHost::alloc<double>(devHost, Vec{size, size});
    alpaka::onHost::memset(queue, t_offset, 0);
    alpaka::onHost::memset(queue, t_scale, 0);
    alpaka::onHost::memset(queue, amplitude_lincomb, 0);
    alpaka::onHost::wait(queue);

    set_params(size, band_width, -3.0, 3.0, amplitude_lincomb.getMdSpan(), t_offset.getMdSpan(), t_scale.getMdSpan());
    mio::log_debug("Params Set");

    auto t_offset_dev = onHost::allocMirror(devAcc, t_offset);
    auto t_scale_dev = onHost::allocMirror(devAcc, t_scale);
    auto amplitude_lincomb_dev = onHost::allocMirror(devAcc, amplitude_lincomb);
    onHost::memcpy(queue, t_offset_dev, t_offset);
    onHost::memcpy(queue, t_scale_dev, t_scale);
    onHost::memcpy(queue, amplitude_lincomb_dev, amplitude_lincomb);

    alpaka::onHost::wait(queue);

    // print(t_offset);
    // print(t_scale);
    // print(amplitude_lincomb);
    // std::cout << "\n";

    // // std::cout << amplitude_lincomb << "\n";

    double const abs_tol = 1e-3, rel_tol = 1e-8, min_dt = 1e-2, max_dt = 1e+2;
    // // auto core = std::make_shared<mio::ControlledStepperWrapper<double,
    // boost::numeric::odeint::runge_kutta_cash_karp54>>(abs_tol, rel_tol, min_dt, max_dt); auto core =
    // std::make_shared<mio::RKIntegratorCore<double>>(abs_tol, rel_tol, min_dt, max_dt);

    auto m_kt_values = onHost::alloc<double>(devHost, Vec{size_t{tableau().entries_low.dim()}, size});
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
        // print(x);
        // print(x2);
        for(size_t i = 0; i < size; i++)
        {
            x[i] = x2[i];
            x2[i] = 0;
        }
        // std::cin.ignore();
    }
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for kernel execution: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << std::endl;

    // integrator.advance(rhs, tmax, dt, results);

    mio::log_debug("Integration Finished");

    // if (size < 5)
    //     results.print_table();
    // else
    //     std::cout << "Num time steps: " << results.get_num_time_points() << "\n";

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

auto main(int argc, char* argv[]) -> int
{
#if USE_ALPAKA
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
#else
    example(Dict{DictEntry{object::api, api::cpu}, DictEntry{object::exec, exec::cpuSerial}});
#endif
}
