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

struct VectorizedNonQuadraticElementsKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        auto const A,
        auto const B,
        auto out,
        float alpha,
        float beta,
        concepts::CVector auto chunkExtent,
        concepts::CVector auto bk) const
    {
        concepts::Vector auto threadIdxMD = acc[layer::thread].idx();
        concepts::CVector auto threadsInBlock = acc[layer::thread].count();

        constexpr auto elemPerWorker = chunkExtent / threadsInBlock;
        auto elemPerThread = CVec<uint32_t, elemPerWorker.y(), elemPerWorker.x()>{};

        using IndexType = typename ALPAKA_TYPEOF(out.getExtents())::index_type;

        // Go over each tile
        for(auto tileOffsetMD : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{ALPAKA_TYPEOF(out.getExtents())::all(0), out.getExtents(), chunkExtent}))
        {
            // this seems like way too much shared memory usage
            concepts::CVector auto sBExtent = CVec<uint32_t, bk.x(), chunkExtent.x()>{};
            concepts::CVector auto sAExtent = CVec<uint32_t, chunkExtent.y(), bk.x()>{};
            // shared a matrix is stored transposed to support shared to register vector loads
            auto sharedATile = onAcc::declareSharedMdArray<float, uniqueId()>(
                acc,
                CVec<uint32_t, sAExtent.y() * sAExtent.x() / 4u, 4u>{});
            auto sharedBTile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, sBExtent);

            static_assert(sAExtent.x() == sBExtent.y());

            constexpr uint32_t regLoadElem = 1u;
            /** This definition is required to pass the CUDA compiler evaluation if cuda and host executes are used,
             * not sure why it is not required for A. */
            using RegBArrayType = float[regLoadElem][elemPerThread.x()];

            float regA[regLoadElem][elemPerThread.y()] = {0};
            float regB[regLoadElem][elemPerThread.x()] = {0};
            float regC[elemPerThread.y()][elemPerThread.x()] = {0};
            auto regMdA = MdSpanArray<float[regLoadElem][elemPerThread.y()], Alignment<16u>>{regA};
            auto regMdB = MdSpanArray<RegBArrayType, Alignment<16u>>{regB};
            auto regMdC = MdSpanArray<float[elemPerThread.y()][elemPerThread.x()], Alignment<16u>>{regC};


            for(IndexType chunkOffset = 0; chunkOffset < A.getExtents().x(); chunkOffset += bk.x())
            {
                auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInBlock};
#if 0
                simdGrid.template concurrent<16u, Alignment<16>>(
                    acc,
                    sAExtent,
                    [&](auto const&, auto const& a) constexpr
                    {
                        auto globalMemValue = a[Vec2D{tileOffsetMD.y(), chunkOffset}].load();

                        for(auto i = 0u; i < globalMemValue.dim(); ++i)
                        {
                            aTransposed[a.getIdx() + Vec2D{0, i}] = globalMemValue[i];
                        }
                    },
                    A);
#else
                for(auto tileElemIndexMD : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{Vec2D::all(0u), sAExtent, Vec2D{4u, 4u}}))
                {
                    auto a0 = SimdPtr{
                        A,
                        Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 0u, chunkOffset + tileElemIndexMD.x()},
                        Alignment<16u>{},
                        CVec<
                            uint32_t,
                            4u>{}}.load();
                    auto a1 = SimdPtr{
                        A,
                        Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 1u, chunkOffset + tileElemIndexMD.x()},
                        Alignment<16u>{},
                        CVec<
                            uint32_t,
                            4u>{}}.load();
                    auto a2 = SimdPtr{
                        A,
                        Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 2u, chunkOffset + tileElemIndexMD.x()},
                        Alignment<16u>{},
                        CVec<
                            uint32_t,
                            4u>{}}.load();
                    auto a3 = SimdPtr{
                        A,
                        Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 3u, chunkOffset + tileElemIndexMD.x()},
                        Alignment<16u>{},
                        CVec<
                            uint32_t,
                            4u>{}}.load();

                    for(auto i = 0u; i < 4; ++i)
                    {
                        auto sAPtr = SimdPtr{
                            sharedATile,
                            Vec2D{sAExtent.x() * tileElemIndexMD.y() / 4u + tileElemIndexMD.x() + i, 0u},
                            Alignment<16u>{},
                            CVec<uint32_t, 4u>{}};
                        auto foo = Simd<float, 4u, Alignment<16u>>{a0[i], a1[i], a2[i], a3[i]};
                        static_assert(std::is_same_v<decltype(sAPtr.load()), decltype(foo)>);
                        sAPtr = foo;
                    }
                    // aTransposed[tileElemIndexMD]
                    //     = A[Vec2D{tileOffsetMD.y() + tileElemIndexMD.y(), chunkOffset + tileElemIndexMD.x()}];
                }
#endif

                simdGrid.template concurrent<16u, Alignment<16>>(
                    acc,
                    sBExtent,
                    [&](auto const&, auto sharedB, auto const& b) constexpr {
                        sharedB = b[Vec2D{chunkOffset, tileOffsetMD.x()}].load();
                    },
                    sharedBTile,
                    B);


                /* This call is equal to `onAcc::syncBlockThreads(acc)`
                 *
                 * The synchronization is required because we will use for the second loop over frame element indicis a
                 * different traversing schema. Therefore, you should not assume any thread and data element relation.
                 */
                alpaka::onAcc::syncBlockThreads(acc);
                for(uint32_t dotIdx = 0; dotIdx < sAExtent.x(); dotIdx += regLoadElem)
                {
                    for(uint32_t d = 0u; d < regLoadElem; ++d)
                        for(uint32_t k = 0u; k < elemPerThread.y(); k += 4)
                        {
                            auto regAPtr = SimdPtr{regMdA, Vec2D{d, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                            auto sAPtr = SimdPtr{
                                sharedATile,
                                Vec2D{
                                    sAExtent.x() * (threadIdxMD.y() * elemPerThread.y() / 4 + k / 4) + dotIdx + d,
                                    0u},
                                Alignment<16u>{},
                                CVec<uint32_t, 4u>{}};
                            regAPtr = sAPtr.load();
                        }

                    for(uint32_t d = 0u; d < regLoadElem; ++d)
                        for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                        {
                            auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                            auto sBPtr = SimdPtr{
                                sharedBTile,
                                Vec2D{dotIdx + d, threadIdxMD.x() * 4 + k * threadsInBlock.x()},
                                Alignment<16u>{},
                                CVec<uint32_t, 4u>{}};
                            regBPtr = sBPtr.load();
                        }

                    for(uint32_t d = 0u; d < regLoadElem; ++d)
                        for(uint32_t j = 0u; j < elemPerThread.y(); ++j)
                        {
                            for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                            {
                                auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                                auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                                regCPtr = regCPtr.load() + regMdA[Vec2D{d, j}] * regBPtr.load();
                            }
                        }
                }

                alpaka::onAcc::syncBlockThreads(acc);
            }

            for(uint32_t j = 0u; j < elemPerThread.y(); ++j)
            {
                for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                {
                    concepts::Vector auto cTileOffsetMD
                        = tileOffsetMD + threadIdxMD * Vec2D{elemPerThread.y(), 4} + Vec2D{j, k * threadsInBlock.x()};
                    auto cSimdPtr = SimdPtr{out, cTileOffsetMD, Alignment<16>{}, CVec<uint32_t, 4u>{}};
                    auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                    cSimdPtr = alpha * regCPtr.load() + beta * cSimdPtr.load();
                }
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

    constexpr uint32_t bk = 16;
    constexpr auto elemPerThread = CVec<uint32_t, 8u, 8u>{};
    concepts::CVector auto frameExtent = CVec<uint32_t, 4, 32>{};
    concepts::CVector auto chunkExtent
        = CVec<uint32_t, frameExtent.y() * elemPerThread.y(), frameExtent.x() * elemPerThread.x()>{};
    concepts::Vector auto framecount = divExZero(C_size, chunkExtent);
    // workaround: we need fewer threads than the chunk extent has element, we will use frameExtent, currently the
    // number of threads in FrameSpec can not have a different type than the frameExtent
    auto frameSpec = onHost::FrameSpec{framecount, frameExtent};

    // Assumption for this kernel: Our SMEM is quadratic and cleanly divides sizes of out buffer
    static_assert(C_size.x() % chunkExtent.x() == 0);
    static_assert(C_size.y() % chunkExtent.y() == 0);

    std::cout << "Testing SMemThreadOversubscriptionNonQuadraticKernel with scalar indices with a grid of "
              << frameSpec << "and chunk extent=" << chunkExtent << "elements per thread=" << elemPerThread << "\n";

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    constexpr uint32_t repeat = 2;
    for(uint32_t i = 0; i < repeat; ++i)
    {
        queue.enqueue(
            computeExec,
            frameSpec,
            VectorizedNonQuadraticElementsKernel{},
            A_d,
            B_d,
            C_d,
            alpha,
            beta,
            chunkExtent,
            CVec<uint32_t, bk>{});
    }

    onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(endT - beginT).count();
    std::cout << "Time for kernel execution: " << duration << " s" << std::endl;
    std::cout << "  - flop count : " << flopCount << std::endl;
    std::cout << "  - performance: "
              << static_cast<double>(flopCount) / 1.e12 / (duration / static_cast<double>(repeat)) << " tflop/s"
              << std::endl;
    if(A_size.x() <= 1024u)
    {
        // copy the results from the device to the host
        onHost::memcpy(queue, C_h, C_d);

        // check the results
        auto cpu_out = onHost::allocHostMirror(C_h);
        onHost::memset(queue, cpu_out, 0x00);

        // wait for all the operations to complete
        onHost::wait(queue);

        for(uint32_t i = 0; i < repeat; ++i)
        {
            // Perform a naive CPU matrix multiplication to compare the results
            naive_matrix_mult(A_h, B_h, cpu_out, alpha, beta);
        }
        bool mismatch = false;
        for(uint32_t i = 0; i < C_size.product(); ++i)
        {
            auto lIdx = mapToND(C_size, i);
            if(!(std::abs(C_h[lIdx] - cpu_out[lIdx]) < epsilon))
            {
                std::cout << std::scientific << "MISMATCH at " << lIdx << " kernel=" << C_h[lIdx]
                          << " cpu=" << cpu_out[lIdx] << " error=" << C_h[lIdx] - cpu_out[lIdx] << std::endl;
                mismatch = true;
            }
            assert(std::abs(C_h[lIdx] - cpu_out[lIdx]) < epsilon);
        }

        if(!mismatch)
            std::cout << "success\n";

        return mismatch ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    else
        return EXIT_SUCCESS;
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
