/* Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "naive_cpu.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace alpaka;

struct GMemCoalescedKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta)
        const
    {
        using IndexType = typename ALPAKA_TYPEOF(out.getExtents())::index_type;

        for(auto ndIndex : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            auto [col, row] = ndIndex;

            float tmp = 0.0;
            for(IndexType i = 0; i < in1.getExtents().y(); ++i)
            {
                tmp += in1[Vec2D{i, row}] * in2[Vec2D{col, i}];
            }
            out[ndIndex] = alpha * tmp + beta * out[ndIndex];
        }
    }
};

int testGMemCoalescedKernel(onHost::concepts::Device auto device, auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // tolerance
    constexpr float epsilon = 0.0001f;

    constexpr Vec2D in1_size = {1024, 256};
    constexpr Vec2D in2_size = {256, 1024};
    constexpr Vec2D out_size = {in1_size.x(), in2_size.y()};
    constexpr size_t flopCount = in1_size.y() * out_size.product() * 2u + 2u * out_size.product();
    static_assert(in1_size.y() == in2_size.x());
    static_assert(in1_size.x() == out_size.x());
    static_assert(in2_size.y() == out_size.y());
    float alpha = 1.0;
    float beta = 0.5;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = onHost::allocHost<float>(in1_size);
    auto in2_h = onHost::allocHost<float>(in2_size);
    auto out_h = onHost::allocHost<float>(out_size);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < in1_size.x(); ++i)
        for(uint32_t j = 0; j < in1_size.y(); ++j)
            in1_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t i = 0; i < in2_size.x(); ++i)
        for(uint32_t j = 0; j < in2_size.y(); ++j)
            in2_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t i = 0; i < out_size.x(); ++i)
        for(uint32_t j = 0; j < out_size.y(); ++j)
            out_h[Vec2D{j, i}] = 0.;

    // run the test the given device
    onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = onHost::allocMirror(device, in1_h);
    auto in2_d = onHost::allocMirror(device, in2_h);
    auto out_d = onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    onHost::memcpy(queue, in1_d, in1_h);
    onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the si
    onHost::memset(queue, out_d, 0x00);

    auto frameSpec = onHost::FrameSpec{Vec2D{8, 8}, Vec2D{32, 32}};

    std::cout << "Testing GMemCoalescedKernel with scalar indices with a grid of " << frameSpec << "\n";

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    queue.enqueue(
        computeExec,
        frameSpec,
        GMemCoalescedKernel{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        alpha,
        beta);

    onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(endT - beginT).count();
    std::cout << "Time for kernel execution: " << duration << " s" << std::endl;
    std::cout << "  - flop count : " << flopCount << std::endl;
    std::cout << "  - performance: " << static_cast<double>(flopCount) / 1.e12 / duration << " tflop/s" << std::endl;

    // copy the results from the device to the host
    onHost::memcpy(queue, out_h, out_d);

    // check the results
    auto cpu_out = onHost::allocHostMirror(out_h);
    onHost::memset(queue, cpu_out, 0x00);

    // wait for all the operations to complete
    onHost::wait(queue);

    // Perform a naive CPU matrix multiplication to compare the results
    naive_matrix_mult(in1_h, in2_h, cpu_out, alpha, beta);

    bool mismatch = false;
    for(uint32_t i = 0; i < out_size.product(); ++i)
    {
        auto lIdx = mapToND(out_size, i);
        if(!(std::abs(out_h[lIdx] - cpu_out[lIdx]) < epsilon))
        {
            std::cout << "MISMATCH at " << lIdx << " kernel=" << out_h[lIdx] << " cpu=" << cpu_out[lIdx] << std::endl;
            mismatch = true;
        }
        assert(std::abs(out_h[lIdx] - cpu_out[lIdx]) < epsilon);
    }

    if(!mismatch)
        std::cout << "success\n";

    return mismatch ? EXIT_FAILURE : EXIT_SUCCESS;
}

int example(auto const cfg)
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto computeExec = cfg[object::exec];

    std::cout << "Using alpaka accelerator: " << core::demangledName(computeExec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return EXIT_SUCCESS;
    }

    // use the first device
    onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << onHost::getName(device) << "\n\n";

    return testGMemCoalescedKernel(device, computeExec);
}

auto main() -> int
{
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& cfg) { return example(cfg); },
        onHost::allBackends(onHost::enabledApis));
}
