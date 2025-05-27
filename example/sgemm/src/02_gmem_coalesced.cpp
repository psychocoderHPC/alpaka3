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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const A, auto const B, auto out, float alpha, float beta) const
    {
        using IndexType = typename ALPAKA_TYPEOF(out.getExtents())::index_type;

        for(auto ndIndex : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            auto [col, row] = ndIndex;

            float tmp = 0.0;
            for(IndexType i = 0; i < A.getExtents().y(); ++i)
            {
                tmp += A[Vec2D{i, row}] * B[Vec2D{col, i}];
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
    constexpr float epsilon = 1e-4;

    constexpr Vec2D A_size = {1024, 256};
    constexpr Vec2D B_size = {256, 1024};
    constexpr Vec2D C_size = {A_size.x(), B_size.y()};
    constexpr size_t flopCount = A_size.y() * C_size.product() * 2u + 2u * C_size.product();
    static_assert(A_size.y() == B_size.x());
    static_assert(A_size.x() == C_size.x());
    static_assert(B_size.y() == C_size.y());
    float alpha = 1.0;
    float beta = 0.5;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto A_h = onHost::allocHost<float>(A_size);
    auto B_h = onHost::allocHost<float>(B_size);
    auto C_h = onHost::allocHost<float>(C_size);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < A_size.x(); ++i)
        for(uint32_t j = 0; j < A_size.y(); ++j)
            A_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t i = 0; i < B_size.x(); ++i)
        for(uint32_t j = 0; j < B_size.y(); ++j)
            B_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t i = 0; i < C_size.x(); ++i)
        for(uint32_t j = 0; j < C_size.y(); ++j)
            C_h[Vec2D{j, i}] = 0.;

    // run the test the given device
    onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto A_d = onHost::allocMirror(device, A_h);
    auto B_d = onHost::allocMirror(device, B_h);
    auto C_d = onHost::allocMirror(device, C_h);

    // copy the input data to the device; the size is known from the buffer objects
    onHost::memcpy(queue, A_d, A_h);
    onHost::memcpy(queue, B_d, B_h);

    // fill the output buffer with zeros; the si
    onHost::memset(queue, C_d, 0x00);

    auto frameSpec = onHost::FrameSpec{Vec2D{8, 8}, Vec2D{32, 32}};

    std::cout << "Testing GMemCoalescedKernel with scalar indices with a grid of " << frameSpec << "\n";

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    queue.enqueue(
        computeExec,
        frameSpec,
        GMemCoalescedKernel{},
        A_d.getMdSpan(),
        B_d.getMdSpan(),
        C_d.getMdSpan(),
        alpha,
        beta);

    onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(endT - beginT).count();
    std::cout << "Time for kernel execution: " << duration << " s" << std::endl;
    std::cout << "  - flop count : " << flopCount << std::endl;
    std::cout << "  - performance: " << static_cast<double>(flopCount) / 1.e12 / duration << " tflop/s" << std::endl;

    // copy the results from the device to the host
    onHost::memcpy(queue, C_h, C_d);

    // check the results
    auto cpu_out = onHost::allocHostMirror(C_h);
    onHost::memset(queue, cpu_out, 0x00);

    // wait for all the operations to complete
    onHost::wait(queue);

    // Perform a naive CPU matrix multiplication to compare the results
    naive_matrix_mult(A_h, B_h, cpu_out, alpha, beta);

    bool mismatch = false;
    for(uint32_t i = 0; i < C_size.product(); ++i)
    {
        auto lIdx = mapToND(C_size, i);
        if(!(std::abs(C_h[lIdx] - cpu_out[lIdx]) < epsilon))
        {
            std::cout << "MISMATCH at " << lIdx << " kernel=" << C_h[lIdx] << " cpu=" << cpu_out[lIdx] << std::endl;
            mismatch = true;
        }
        assert(std::abs(C_h[lIdx] - cpu_out[lIdx]) < epsilon);
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
