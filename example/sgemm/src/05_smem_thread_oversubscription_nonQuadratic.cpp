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

struct SMemThreadOversubscriptionNonQuadraticKernel
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
        concepts::CVector auto elemPerThread,
        concepts::CVector auto bk) const
    {
        concepts::Vector auto threadIdxMD = acc[layer::thread].idx();
        //    concepts::CVector auto frameExtent = acc[frame::extent];


        using IndexType = typename ALPAKA_TYPEOF(out.getExtents())::index_type;

        // Go over each tile
        for(auto tileOffsetMD : onAcc::makeIdxMap(
                acc,
                onAcc::worker::blocksInGrid,
                IdxRange{ALPAKA_TYPEOF(out.getExtents())::all(0), out.getExtents(), chunkExtent}))
        {
            constexpr uint32_t numElem = elemPerThread.x();

            // this seems like way too much shared memory usage
            concepts::CVector auto sBExtent = CVec<uint32_t, bk.x(), chunkExtent.x()>{};
            concepts::CVector auto sAExtent = CVec<uint32_t, chunkExtent.y(), bk.x()>{};

            auto sharedATile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, sAExtent);
            auto sharedBTile = onAcc::declareSharedMdArray<float, uniqueId()>(acc, sBExtent);

            static_assert(sAExtent.x() == sBExtent.y());

#if 0
            using TVecA = alpaka::Vec<float, numElem>;
            using TVecB = alpaka::Vec<float, numElem>;
            using TVecCX = alpaka::Vec<float, numElem>;
            using TVecC = alpaka::Vec<TVecCX, numElem>;

            TVecA tmpA = {0};
            TVecB tmpB = {0};
            TVecC tmpC = TVecC::all(TVecCX::all(0));
#else
            float regA[numElem] = {0};
            float regB[numElem] = {0};
            float regC[numElem][numElem] = {0};
            auto regMdA = MdSpanArray<float[numElem], Alignment<16u>>{regA};
            auto regMdB = MdSpanArray<float[numElem], Alignment<16u>>{regB};
            auto regMdC = MdSpanArray<float[numElem][numElem], Alignment<16u>>{regC};
#endif


            // iterate through input buffers with stride of smem size
            // Assumption: frameExtent is quadratic, problem size is dividable by frameExtent
            for(IndexType chunkOffset = 0; chunkOffset < A.getExtents().x(); chunkOffset += sAExtent.x())
            {
                // std::cout << "------A----\n";
                // populate smem
#if 0
                for(auto tileElemIndexMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sAExtent}))
                {
                    sharedATile[tileElemIndexMD]
                        = A[Vec2D{tileOffsetMD.y() + tileElemIndexMD.y(), tileElemIndexMD.x() + chunkOffset}];
                }
                [[maybe_unused]] auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInBlock};
#else
                auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInBlock};
                simdGrid.template concurrent<16u, Alignment<16>>(
                    acc,
                    sAExtent,
                    [&](auto const&, auto sharedA, auto const& a) constexpr {
                        sharedA = a[Vec2D{tileOffsetMD.y(), chunkOffset}].load();
                    },
                    sharedATile,
                    A);
#endif

#if 0
                // std::cout << "------B----\n";
                for(auto tileElemIndexMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sBExtent}))
                {
                    sharedBTile[tileElemIndexMD]
                        = B[Vec2D{tileElemIndexMD.y() + chunkOffset, tileOffsetMD.x() + tileElemIndexMD.x()}];
                    //    printf("B %u,%u ->
                    //    %f\n",tileElemIndexMD.y(),tileElemIndexMD.x(),sharedBTile[tileElemIndexMD]);
                    //  printf("B -> %f\n",sharedBTile[Vec2D{1u,2u}]);
                    //   std::cout << sharedBTile[tileElemIndexMD] << ",";
                    //  if(tileElemIndexMD.x() == sBExtent.x() - 1)
                    //      std::cout << "\n";
                }
                // std::cout << "-----------\n";
#else
                simdGrid.template concurrent<16u, Alignment<16>>(
                    acc,
                    sBExtent,
                    [&](auto const&, auto sharedB, auto const& b) constexpr {
                        sharedB = b[Vec2D{chunkOffset, tileOffsetMD.x()}].load();
                    },
                    sharedBTile,
                    B);
#endif


                /* This call is equal to `onAcc::syncBlockThreads(acc)`
                 *
                 * The synchronization is required because we will use for the second loop over frame element indicis a
                 * different traversing schema. Therefore, you should not assume any thread and data element relation.
                 */
                alpaka::onAcc::syncBlockThreads(acc);
                for(uint32_t dotIdx = 0; dotIdx < sAExtent.x(); dotIdx += 1)
                {
                    //  std::cout << "------A cache----" << numElem << "\n";
                    for(uint32_t k = 0u; k < numElem; ++k)
                    {
                        regMdA[k] = sharedATile[Vec2D{threadIdxMD.y() * numElem + k, dotIdx}];
                        //   if(threadIdxMD.x() == 1u)
                        //      printf("A %u -> %f\n", k,tmpA[k]);
                        //    std::cout << tmpA[k] << ",";
                    }

                    // std::cout << "\n------B cache----" << numElem << "\n";
#if 0
                    for(uint32_t k = 0u; k < numElem; ++k)
                    {
                        regMdB[k] = sharedBTile[Vec2D{dotIdx, threadIdxMD.x() * numElem + k}];
                    }
#else

                    for(uint32_t k = 0u; k < numElem; k += 4)
                    {
                        auto regBPtr = SimdPtr{regMdB, Vec1D{k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                        auto sBPtr = SimdPtr{
                            sharedBTile,
                            Vec2D{dotIdx, threadIdxMD.x() * numElem + k},
                            Alignment<16u>{},
                            CVec<uint32_t, 4u>{}};
                        regBPtr = sBPtr.load();
                    }
#endif

#if 0
                    //  std::cout << "\n------C cache----" << numElem << "\n";
                    for(uint32_t j = 0u; j < numElem; ++j)
                        for(uint32_t i = 0u; i < numElem; ++i)
                        {
                            regMdC[Vec2D{j,i}] += regMdA[j] * regMdB[i];
                            //     std::cout << tmpC[j][i] << ", " << tmpA[j] << "," << tmpB[i] << "\n";
                            //     if(i == numElem - 1)
                            //         std::cout << "\n";
                        }
#else
                    for(uint32_t j = 0u; j < numElem; ++j)
                    {
                        for(uint32_t k = 0u; k < numElem; k += 4)
                        {
                            auto regBPtr = SimdPtr{regMdB, Vec1D{k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                            auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                            regCPtr = regCPtr.load() + regMdA[j] * regBPtr.load();
                        }
                    }
#endif
                    //  std::cout << "------\n";
                }

                alpaka::onAcc::syncBlockThreads(acc);
            }

#if 0
            // std::cout << "------C out----" << elemPerThreadC << "\n";
            for(uint32_t j = 0u; j < numElem; ++j)
                for(uint32_t i = 0u; i < numElem; ++i)
                {
                    concepts::Vector auto tileElemIndexMD = threadIdxMD * numElem + alpaka::Vec{j, i};

                    out[tileOffsetMD + tileElemIndexMD]
                        = alpha * regMdC[Vec2D{j,i}] + beta * out[tileOffsetMD + tileElemIndexMD];
                    //    std::cout << tileOffsetMD + tileElemIndexMD << "|" << out[tileOffsetMD + tileElemIndexMD] <<
                    //    ","; if(tElemIdxMD.x() == elemPerThreadC.x() - 1)
                    //        std::cout << "\n";
                }
#else
            concepts::Vector auto cTileOffsetMD = tileOffsetMD + threadIdxMD * numElem;
            for(uint32_t j = 0u; j < numElem; ++j)
            {
                for(uint32_t k = 0u; k < numElem; k += 4)
                {
                    auto cSimdPtr = SimdPtr{out, cTileOffsetMD + Vec2D{j,k}, Alignment<16>{}, CVec<uint32_t, 4u>{}};
                    auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<16u>{}, CVec<uint32_t, 4u>{}};
                    cSimdPtr = alpha * regCPtr.load() + beta * cSimdPtr.load();
                }
            }

#endif
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

    constexpr uint32_t bk = 8;
    constexpr uint32_t elemPerThread = 8u;
    concepts::CVector auto frameExtent = CVec<uint32_t, 8, 8>{};
    concepts::CVector auto chunkExtent
        = CVec<uint32_t, frameExtent.y() * elemPerThread, frameExtent.x() * elemPerThread>{};
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

    constexpr uint32_t repeat = 50;
    for(uint32_t i = 0; i < repeat; ++i)
    {
        queue.enqueue(
            computeExec,
            frameSpec,
            SMemThreadOversubscriptionNonQuadraticKernel{},
            A_d,
            B_d,
            C_d,
            alpha,
            beta,
            chunkExtent,
            CVec<uint32_t, elemPerThread>{},
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
