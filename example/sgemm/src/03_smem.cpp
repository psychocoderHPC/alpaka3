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

struct SMemKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta)
        const
    {
        auto numFramesMD = acc[frame::count];
        auto frameExtentMD = acc[frame::extent];

        using IndexType = typename ALPAKA_TYPEOF(numFramesMD)::index_type;

        // Go over each tile
        for(auto tileIndexMD : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numFramesMD}))
        {
            // this seems like way too much shared memory usage
            auto sharedIn1Tile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, frameExtentMD);
            auto sharedIn2Tile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, frameExtentMD);

            auto tmp = onAcc::declareSharedMdArray<float, uniqueId()>(acc, frameExtentMD);

            // iterate through input buffers with stride of smem size
            // Assumption: frameExtent is quadratic, problem size is dividable by frameExtent
            for(IndexType chunkStride = 0; chunkStride < in1.getExtents().y(); chunkStride += frameExtentMD.y())
            {
                // populate smem
                for(auto tileElemIndexMD :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtentMD}))
                { // buffer access: in[col, row]
                    // TODO here we could use memory coalescing
                    sharedIn1Tile[tileElemIndexMD] = in1[Vec2D{
                        chunkStride + tileElemIndexMD.y(),
                        frameExtentMD.x() * tileIndexMD.x() + tileElemIndexMD.x()}];
                    sharedIn2Tile[tileElemIndexMD] = in2[Vec2D{
                        frameExtentMD.y() * tileIndexMD.y() + tileElemIndexMD.y(),
                        chunkStride + tileElemIndexMD.x()}];
                    if(chunkStride == 0)
                        tmp[tileElemIndexMD] = 0.;
                }

                /* This call is equal to `onAcc::syncBlockThreads(acc)`
                 *
                 * The synchronization is required because we will use for the second loop over frame element indicis a
                 * different traversing schema. Therefore, you should not assume any thread and data element relation.
                 */
                alpaka::onAcc::syncBlockThreads(acc);

                for(auto tileElemIndexMD :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtentMD}))
                {
                    for(IndexType i = 0; i < frameExtentMD.y(); i++)
                    {
                        tmp[tileElemIndexMD] += sharedIn1Tile[Vec2D{i, tileElemIndexMD.x()}]
                                                * sharedIn2Tile[Vec2D{tileElemIndexMD.y(), i}];
                    }
                }

                alpaka::onAcc::syncBlockThreads(acc);
            }

            for(auto tileElemIndexMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtentMD}))
            {
                out[tileIndexMD * frameExtentMD + tileElemIndexMD]
                    = alpha * tmp[tileElemIndexMD] + beta * out[tileIndexMD * frameExtentMD + tileElemIndexMD];
            }
        }
    }
};

int testGMemNaiveKernel(onHost::concepts::Device auto device, auto computeExec)
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

    int const frameExtent1D = 16;
    auto frameExtent = CVec<uint32_t, frameExtent1D, frameExtent1D>{};
    int framecountX = std::ceil(out_size.x() / frameExtent.x());
    int framecountY = std::ceil(out_size.y() / frameExtent.y());
    auto frameSpec = onHost::FrameSpec{Vec2D{framecountY, framecountX}, frameExtent};

    // Assumption for this kernel: Our SMEM is quadratic and cleanly divides sizes of out buffer
    static_assert(out_size.x() % frameExtent.x() == 0);
    static_assert(out_size.y() % frameExtent.y() == 0);
    static_assert(frameExtent.x() == frameExtent.y());

    std::cout << "Testing SMemKernel with scalar indices with a grid of " << frameSpec << "\n";

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    queue.enqueue(
        computeExec,
        frameSpec,
        SMemKernel{},
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

    return testGMemNaiveKernel(device, computeExec);
}

auto main() -> int
{
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& cfg) { return example(cfg); },
        onHost::allBackends(onHost::enabledApis));
}
