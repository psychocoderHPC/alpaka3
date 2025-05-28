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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const A, auto const B, auto out, float alpha, float beta) const
    {
        concepts::CVector auto frameExtentMD = acc[frame::extent];
        concepts::Vector auto threadIdxMD = acc[layer::thread].idx();
        concepts::CVector auto threadBlockExtentMD = acc[layer::thread].count();

        using IndexType = typename ALPAKA_TYPEOF(out.getExtents())::index_type;

        // Go over each tile
        for(auto tileOffsetMD : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{ALPAKA_TYPEOF(out.getExtents())::all(0), out.getExtents(), frameExtentMD}))
        {
            // this seems like way too much shared memory usage
            auto sharedATile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, frameExtentMD);
            auto sharedBTile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, frameExtentMD);

            constexpr auto independentThread = onAcc::WorkerGroup{CVec<uint32_t, 0, 0>{}, CVec<uint32_t, 1, 1>{}};

            constexpr concepts::CVector auto elemPerThread = CVec<
                uint32_t,
                frameExtentMD.y() / threadBlockExtentMD.y(),
                frameExtentMD.x() / threadBlockExtentMD.x()>{};
            using TVecX = alpaka::Vec<float, elemPerThread.x()>;
            using TVec = alpaka::Vec<TVecX, elemPerThread.y()>;

            TVec tmp = {0}; // TVec::all(TVecX::all(0.f));

            // iterate through input buffers with stride of smem size
            // Assumption: frameExtent is quadratic, problem size is dividable by frameExtent
            for(IndexType chunkOffset = 0; chunkOffset < A.getExtents().x(); chunkOffset += frameExtentMD.x())
            {
                // populate smem
                for(auto tileElemIndexMD :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtentMD}))
                {
                    sharedATile[tileElemIndexMD]
                        = A[Vec2D{tileOffsetMD.y() + tileElemIndexMD.y(), tileElemIndexMD.x() + chunkOffset}];
                    sharedBTile[tileElemIndexMD]
                        = B[Vec2D{tileElemIndexMD.y() + chunkOffset, tileOffsetMD.x() + tileElemIndexMD.x()}];
                }

                /* This call is equal to `onAcc::syncBlockThreads(acc)`
                 *
                 * The synchronization is required because we will use for the second loop over frame element indicis a
                 * different traversing schema. Therefore, you should not assume any thread and data element relation.
                 */
                alpaka::onAcc::syncBlockThreads(acc);

                for(auto tElemIdxMD : onAcc::makeIdxMap(acc, independentThread, IdxRange{elemPerThread}))
                {
                    for(IndexType k = 0; k < frameExtentMD.x(); k++)
                    {
                        concepts::Vector auto tileElemIndexMD = tElemIdxMD * threadBlockExtentMD + threadIdxMD;
                        tmp[tElemIdxMD.y()][tElemIdxMD.x()]
                            += sharedATile[Vec2D{tileElemIndexMD.y(), k}] * sharedBTile[Vec2D{k, tileElemIndexMD.x()}];
                    }
                }

                alpaka::onAcc::syncBlockThreads(acc);
            }

            for(auto tElemIdxMD : onAcc::makeIdxMap(acc, independentThread, IdxRange{elemPerThread}))
            {
                concepts::Vector auto tileElemIndexMD = tElemIdxMD * threadBlockExtentMD + threadIdxMD;
                out[tileOffsetMD + tileElemIndexMD]
                    = alpha * tmp[tElemIdxMD.y()][tElemIdxMD.x()] + beta * out[tileOffsetMD + tileElemIndexMD];
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
    constexpr float epsilon = 1e-4;

    constexpr Vec2D A_size = {256, 1024};
    constexpr Vec2D B_size = {1024, 256};
    constexpr Vec2D C_size = {A_size.y(), B_size.x()};
    constexpr size_t flopCount = static_cast<size_t>(A_size.x()) * C_size.product() * 2u + 2u * C_size.product();
    static_assert(A_size.x() == B_size.y());
    static_assert(A_size.y() == C_size.y());
    static_assert(B_size.x() == C_size.x());
    float alpha = 1.0;
    float beta = 0.5;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto A_h = onHost::allocHost<float>(A_size);
    auto B_h = onHost::allocHost<float>(B_size);
    auto C_h = onHost::allocHost<float>(C_size);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t j = 0; j < A_size.y(); ++j)
        for(uint32_t i = 0; i < A_size.x(); ++i)
            A_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t j = 0; j < B_size.y(); ++j)
        for(uint32_t i = 0; i < B_size.x(); ++i)
            B_h[Vec2D{j, i}] = dist(rand);
    for(uint32_t j = 0; j < C_size.y(); ++j)
        for(uint32_t i = 0; i < C_size.x(); ++i)
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

    uint32_t const frameExtent1D = 16;
    concepts::CVector auto frameExtent = CVec<uint32_t, frameExtent1D, frameExtent1D>{};
    concepts::Vector auto framecount = divExZero(C_size, frameExtent);
    auto frameSpec = onHost::FrameSpec{framecount, frameExtent};

    // Assumption for this kernel: Our SMEM is quadratic and cleanly divides sizes of out buffer
    static_assert(C_size.x() % frameExtent.x() == 0);
    static_assert(C_size.y() % frameExtent.y() == 0);
    static_assert(frameExtent.x() == frameExtent.y());

    std::cout << "Testing SMemKernel with scalar indices with a grid of " << frameSpec << "\n";

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    queue.enqueue(computeExec, frameSpec, SMemKernel{}, A_d, B_d, C_d, alpha, beta);

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

    return testGMemNaiveKernel(device, computeExec);
}

auto main() -> int
{
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& cfg) { return example(cfg); },
        onHost::allBackends(onHost::enabledApis));
}
