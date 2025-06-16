/* Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "naive_cpu.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


#if ALPAKA_HAS_CUBLAS
#    include <cublas_v2.h>
#endif

#if ALPAKA_HAS_HIPBLAS
#    include <hipblas/hipblas.h>
#endif

using namespace alpaka;
using DataType = size_t;
constexpr uint32_t alignment = uint32_t{sizeof(DataType) * 4u};

ALPAKA_FN_INLINE constexpr void loadAToShared(
    auto const& acc,
    concepts::MdSpan auto const& sharedATile,
    concepts::MdSpan auto const& A,
    concepts::Vector auto const& tileOffsetMD,
    concepts::Vector auto const& sAExtent,
    auto aXOffset)
{
    for(auto tileElemIndexMD :
        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec2D::all(0u), sAExtent, Vec2D{4u, 4u}}))
    {
        auto a0 = SimdPtr{
            A,
            Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 0u, aXOffset + tileElemIndexMD.x()},
            Alignment<alignment>{},
            CVec<
                uint32_t,
                4u>{}}.load();
        auto a1 = SimdPtr{
            A,
            Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 1u, aXOffset + tileElemIndexMD.x()},
            Alignment<alignment>{},
            CVec<
                uint32_t,
                4u>{}}.load();
        auto a2 = SimdPtr{
            A,
            Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 2u, aXOffset + tileElemIndexMD.x()},
            Alignment<alignment>{},
            CVec<
                uint32_t,
                4u>{}}.load();
        auto a3 = SimdPtr{
            A,
            Vec2D{tileOffsetMD.y() + tileElemIndexMD.y() + 3u, aXOffset + tileElemIndexMD.x()},
            Alignment<alignment>{},
            CVec<
                uint32_t,
                4u>{}}.load();

        auto numTiles = sAExtent / 4u;
        auto tileIdx = tileElemIndexMD / 4u;
        auto tileSlot = linearize(numTiles, tileIdx);
        for(auto i = 0u; i < 4; ++i)
        {
            auto sAPtr = SimdPtr{
                sharedATile,
                Vec2D{tileSlot + numTiles.product() * i, 0u},
                Alignment<alignment>{},
                CVec<uint32_t, 4u>{}};
            auto foo = Simd<DataType, 4u, Alignment<alignment>>{a0[i], a1[i], a2[i], a3[i]};
            static_assert(std::is_same_v<decltype(sAPtr.load()), decltype(foo)>);
            sAPtr = foo;
        }
    }
}

ALPAKA_FN_INLINE constexpr void loadBToShared(
    auto const& acc,
    concepts::MdSpan auto const& sharedBTile,
    concepts::MdSpan auto const& B,
    concepts::Vector auto const& tileOffsetMD,
    concepts::Vector auto const& sBExtent,
    auto bYOffset)
{
    auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInBlock};
    simdGrid.template concurrent<alignment, Alignment<alignment>>(
        acc,
        sBExtent,
        [&](auto const&, auto sharedB, auto const& b) constexpr {
            sharedB = b[Vec2D{bYOffset, tileOffsetMD.x()}].load();
        },
        sharedBTile,
        B);
}

ALPAKA_FN_INLINE constexpr void computeCTile(
    auto const& acc,
    auto& regMdC,
    concepts::MdSpan auto const& sharedATile,
    concepts::MdSpan auto const& sharedBTile,
    concepts::Vector auto const& threadIdxMD,
    concepts::CVector auto threadsInBlock,
    concepts::Vector auto const& sAExtent,
    concepts::CVector auto elemPerThread)
{
    constexpr uint32_t regLoadElem = 1u;

#define DO_FULL 0

#if DO_FULL == 0
    DataType regA[regLoadElem][4u];
    auto regMdA = MdSpanArray<DataType[regLoadElem][4u], Alignment<alignment>>{regA};
#elif DO_FULL == 2
    DataType regA[regLoadElem][elemPerThread.y()];
    auto regMdA = MdSpanArray<DataType[regLoadElem][elemPerThread.y()], Alignment<alignment>>{regA};
#endif
    DataType regB[regLoadElem][elemPerThread.x()];

    /** This definition is required to pass the CUDA compiler evaluation if cuda and host executes are used,
     * not sure why it is not required for A. */
    constexpr auto numBRegElem = elemPerThread.x();
    using RegBArrayType = DataType[regLoadElem][numBRegElem];
    auto regMdB = MdSpanArray<RegBArrayType, Alignment<alignment>>{regB};
    for(uint32_t dotIdx = 0; dotIdx < sAExtent.x(); dotIdx += regLoadElem)
    {
#if DO_FULL == 0
        for(uint32_t d = 0u; d < regLoadElem; ++d)
            for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
            {
                auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                auto sBPtr = SimdPtr{
                    sharedBTile,
                    Vec2D{dotIdx + d, threadIdxMD.x() * 4 + k * threadsInBlock.x()},
                    Alignment<alignment>{},
                    CVec<uint32_t, 4u>{}};
                regBPtr = sBPtr.load();
            }

        for(uint32_t d = 0u; d < regLoadElem; ++d)
            for(uint32_t kk = 0u; kk < elemPerThread.y(); kk += 4)
            {
                auto numTiles = sAExtent / 4u;
                auto tileIdx = Vec2D{threadIdxMD.y() * elemPerThread.y() + kk, dotIdx + d} / 4u;
                auto tileSlot = linearize(numTiles, tileIdx);

                auto regAPtr = SimdPtr{regMdA, Vec2D{d, 0}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                auto sAPtr = SimdPtr{
                    sharedATile,
                    Vec2D{tileSlot + numTiles.product() * ((dotIdx + d) % 4), 0u},
                    Alignment<alignment>{},
                    CVec<uint32_t, 4u>{}};
                regAPtr = sAPtr.load();
                for(uint32_t j = 0u; j < 4; ++j)
                {
                    for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                    {
                        auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                        auto regCPtr = SimdPtr{regMdC, Vec2D{kk + j, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                        regCPtr = regCPtr.load() + regMdA[Vec2D{d, j}] * regBPtr.load();
                    }
                }
            }
#elif DO_FULL == 2
        for(uint32_t d = 0u; d < regLoadElem; ++d)
            for(uint32_t k = 0u; k < elemPerThread.y(); k += 4)
            {
                auto numTiles = sAExtent / 4u;
                auto tileIdx = Vec2D{threadIdxMD.y() * elemPerThread.y() + k, dotIdx + d} / 4u;
                auto tileSlot = linearize(numTiles, tileIdx);

                auto regAPtr = SimdPtr{regMdA, Vec2D{d, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                auto sAPtr = SimdPtr{
                    sharedATile,
                    Vec2D{tileSlot + numTiles.product() * ((dotIdx + d) % 4), 0u},
                    Alignment<alignment>{},
                    CVec<uint32_t, 4u>{}};
                regAPtr = sAPtr.load();
            }
        for(uint32_t d = 0u; d < regLoadElem; ++d)
            for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
            {
                auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                auto sBPtr = SimdPtr{
                    sharedBTile,
                    Vec2D{dotIdx + d, (threadIdxMD.x() * 4u + k / 4u * threadsInBlock.x() * 4u)},
                    Alignment<alignment>{},
                    CVec<uint32_t, 4u>{}};
                regBPtr = sBPtr.load();
            }

        for(uint32_t d = 0u; d < regLoadElem; ++d)
            for(uint32_t j = 0u; j < elemPerThread.y(); ++j)
            {
                for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                {
                    auto regBPtr = SimdPtr{regMdB, Vec2D{d, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                    auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                    regCPtr = regCPtr.load() + regMdA[Vec2D{d, j}] * regBPtr.load();
                }
            }
#endif
    }
}

struct VectorizedNonQuadraticElementsKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        auto const A,
        auto const B,
        auto out,
        DataType alpha,
        DataType beta,
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
            auto sharedATile = onAcc::declareSharedMdArray<DataType, uniqueId()>(
                acc,
                CVec<uint32_t, sAExtent.y() * sAExtent.x() / 4u, 4u>{});
            auto sharedBTile = onAcc::declareSharedMdArray<DataType, uniqueId()>(acc, sBExtent);


            static_assert(sAExtent.x() == sBExtent.y());

            DataType regC[elemPerThread.y()][elemPerThread.x()] = {0};
            auto regMdC = MdSpanArray<DataType[elemPerThread.y()][elemPerThread.x()], Alignment<alignment>>{regC};

            for(IndexType chunkOffset = 0; chunkOffset < A.getExtents().x(); chunkOffset += bk.x())
            {
                loadAToShared(acc, sharedATile, A, tileOffsetMD, sAExtent, chunkOffset);
                loadBToShared(acc, sharedBTile, B, tileOffsetMD, sBExtent, chunkOffset);

                alpaka::onAcc::syncBlockThreads(acc);

                computeCTile(
                    acc,
                    regMdC,
                    sharedATile,
                    sharedBTile,
                    threadIdxMD,
                    threadsInBlock,
                    sAExtent,
                    elemPerThread);

                alpaka::onAcc::syncBlockThreads(acc);
            }

            for(uint32_t j = 0u; j < elemPerThread.y(); ++j)
            {
                for(uint32_t k = 0u; k < elemPerThread.x(); k += 4)
                {
                    concepts::Vector auto cTileOffsetMD = tileOffsetMD + threadIdxMD * Vec2D{elemPerThread.y(), 4}
                                                          + Vec2D{j, k / 4u * threadsInBlock.x() * 4u};
                    auto cSimdPtr = SimdPtr{out, cTileOffsetMD, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                    auto regCPtr = SimdPtr{regMdC, Vec2D{j, k}, Alignment<alignment>{}, CVec<uint32_t, 4u>{}};
                    cSimdPtr = alpha * regCPtr.load() + beta * cSimdPtr.load();
                }
            }
        }
    }
};

bool equal([[maybe_unused]] auto idx, auto a, auto b)
{
    bool isEqual = false;
    double relativeError = 1.0;
    if constexpr(std::integral<ALPAKA_TYPEOF(a)> && std::integral<ALPAKA_TYPEOF(b)>)
    {
        isEqual = (a == b);
    }
    else
    {
        // relative tolerance
        constexpr double epsilon = 1e-4;
        relativeError = 1.0 - std::abs(static_cast<double>(a) / static_cast<double>(b));
        isEqual = relativeError < epsilon;
    }
    if(!isEqual)
    {
        std::cout << std::scientific << std::setprecision(std::numeric_limits<DataType>::max_digits10)
                  << "MISMATCH at " << idx << " kernel=" << a << " cpu=" << b
                  << (relativeError != 1.0 ? (std::string(" error=") + std::to_string(relativeError)) : "")
                  << std::endl;
    }
    assert(isEqual);
    return isEqual;
}

int verifyResults(auto queue, auto C_d, auto CReference_h)
{
    auto C_h = onHost::allocHostMirror(C_d);

    onHost::memcpy(queue, C_h, C_d);
    // wait for all the operations to complete
    onHost::wait(queue);


    bool isValid = true;
    for(uint32_t i = 0; i < C_h.getExtents().product() || !isValid; ++i)
    {
        auto lIdx = mapToND(C_h.getExtents(), i);
        isValid = equal(lIdx, C_h[lIdx], CReference_h[lIdx]);
    }

    if(isValid)
        std::cout << "success validated!\n";
    else
        return EXIT_FAILURE;


    return EXIT_SUCCESS;
}

int testGMemNaiveKernel(onHost::concepts::Device auto device, auto computeExec)
{
    // random number generator with a gaussian distribution
#if 0
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<DataType> dist{0.0001f, 1.f};
#endif
    constexpr Vec2D A_size = {256, 1024};
    constexpr Vec2D B_size = {1024, 256};
    constexpr Vec2D C_size = {A_size.y(), B_size.x()};
    constexpr size_t flopCount = static_cast<size_t>(A_size.x()) * C_size.product() * 2u + 2u * C_size.product();
    static_assert(A_size.x() == B_size.y());
    static_assert(A_size.y() == C_size.y());
    static_assert(B_size.x() == C_size.x());
    DataType alpha = 5;
    DataType beta = 3;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto A_h = onHost::allocHost<DataType>(A_size);
    auto B_h = onHost::allocHost<DataType>(B_size);
    auto C_h = onHost::allocHost<DataType>(C_size);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t j = 0; j < A_size.y(); ++j)
        for(uint32_t i = 0; i < A_size.x(); ++i)
            A_h[Vec2D{j, i}] = i + j * A_size.x(); // dist(rand);
    for(uint32_t j = 0; j < B_size.y(); ++j)
        for(uint32_t i = 0; i < B_size.x(); ++i)
            B_h[Vec2D{j, i}] = i + j * B_size.x(); // dist(rand);
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

    constexpr uint32_t repeat = 2;
    constexpr uint32_t bk = 16;
    constexpr auto elemPerThread = CVec<uint32_t, 16u, 8u>{};
    concepts::CVector auto frameExtent = CVec<uint32_t, 8, 16>{};
    concepts::CVector auto chunkExtent
        = CVec<uint32_t, frameExtent.y() * elemPerThread.y(), frameExtent.x() * elemPerThread.x()>{};
    concepts::Vector auto framecount = divExZero(C_size, chunkExtent);
    // workaround: we need fewer threads than the chunk extent has element, we will use frameExtent, currently the
    // number of threads in FrameSpec can not have a different type than the frameExtent
    auto frameSpec = onHost::FrameSpec{framecount, frameExtent};

    // Assumption for this kernel: Our SMEM is quadratic and cleanly divides sizes of out buffer
    static_assert(C_size.x() % chunkExtent.x() == 0);
    static_assert(C_size.y() % chunkExtent.y() == 0);
    static_assert(bk >= 4);
    static_assert(A_size.x() >= 4 && A_size.y() >= 4);

    std::cout << "Testing SMemThreadOversubscriptionNonQuadraticKernel with scalar indices with a grid of "
              << frameSpec << "and chunk extent=" << chunkExtent << "elements per thread=" << elemPerThread << "\n";

    auto const callSgemm = [&]()
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
    };

#if ALPAKA_HAS_CUBLAS == 1
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    cublasSetStream(handle, queue.getNativeHandle());

    auto C_blas_d = onHost::allocMirror(device, C_h);
    onHost::memset(queue, C_blas_d, 0x00);
    onHost::wait(queue);

    int M = A_d.getExtents().y();
    int N = A_d.getExtents().x();
    int K = B_d.getExtents().x();

    auto const callSgemmCUBlas = [&]()
    {
        stat = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            K,
            M,
            N,
            &alpha,
            B_d.data(),
            B_d.getPitches().y() / sizeof(DataType),
            A_d.data(),
            A_d.getPitches().y() / sizeof(DataType),
            &beta,
            C_blas_d.data(),
            C_blas_d.getPitches().y() / sizeof(DataType));
    };

    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cublasSgemm failed with error code: " << stat << std::endl;
        return EXIT_FAILURE;
    }

    // warmup call CUBlas, result is used for validation
    callSgemmCUBlas();
#endif
#if ALPAKA_HAS_HIPBLAS == 1
    hipblasHandle_t handle;
    // required to avoid `device not ready` in HIp which is not reset by hipblas
    onHost::wait(queue);
    hipblasStatus_t stat = hipblasCreate(&handle);
    if(stat != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "hipblasCreate failed with error code: " << stat << std::endl;
        return EXIT_FAILURE;
    }
    stat = hipblasSetStream(handle, queue.getNativeHandle());
    if(stat != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "hipblasSetStream failed with error code: " << stat << std::endl;
        return EXIT_FAILURE;
    }

    auto C_blas_d = onHost::allocMirror(device, C_h);
    onHost::memset(queue, C_blas_d, 0x00);
    onHost::wait(queue);

    int M = A_d.getExtents().y();
    int N = A_d.getExtents().x();
    int K = B_d.getExtents().x();

    auto const callSgemmHipBlas = [&]()
    {
        stat = hipblasSgemm(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            K,
            M,
            N,
            &alpha,
            B_d.data(),
            B_d.getPitches().y() / sizeof(DataType),
            A_d.data(),
            A_d.getPitches().y() / sizeof(DataType),
            &beta,
            C_blas_d.data(),
            C_blas_d.getPitches().y() / sizeof(DataType));
    };

    if(stat != HIPBLAS_STATUS_SUCCESS)
    {
        std::cerr << "hipblasSgemm failed with error code: " << stat << std::endl;
        return EXIT_FAILURE;
    }

    // warmup call CUBlas, result is used for validation
    callSgemmHipBlas();
#endif

    // warmup call, result is used for validation
    callSgemm();

    onHost::wait(queue);

    int err = EXIT_SUCCESS;
#if ALPAKA_HAS_CUBLAS != 1 && ALPAKA_HAS_HIPBLAS != 1
    if(A_h.getExtents().x() <= 1024u)
    {
        // check the results
        auto cpuReference = onHost::allocHostMirror(C_h);
        auto host = onHost::makeHostDevice();
        auto q = host.makeQueue();
        onHost::memset(q, cpuReference, 0x00);
        onHost::wait(q);
        // wait for all the operations to complete
        onHost::wait(queue);

        naive_matrix_mult<DataType>(A_h, B_h, cpuReference, alpha, beta);
        std::cout << "validation against native implementation!\n";
        err = verifyResults(queue, C_d, cpuReference);
    }
    else
        std::cout << "validation skipped, matrix to large!\n";
#elif ALPAKA_HAS_CUBLAS == 1

    if(A_h.getExtents().x() <= 4096u)
    {
        std::cout << "validation against cuBlas!\n";
        auto blasReference_h = onHost::allocHostMirror(C_h);

        onHost::memcpy(queue, blasReference_h, C_blas_d);
        onHost::wait(queue);
        err = verifyResults(queue, C_d, blasReference_h);
    }
    else
        std::cout << "validation skipped, matrix to large!\n";
#elif ALPAKA_HAS_HIPBLAS == 1

    if(A_h.getExtents().x() <= 4096u)
    {
        std::cout << "validation against hipBlas!\n";
        auto blasReference_h = onHost::allocHostMirror(C_h);

        onHost::memcpy(queue, blasReference_h, C_blas_d);
        onHost::wait(queue);
        err = verifyResults(queue, C_d, blasReference_h);
    }
    else
        std::cout << "validation skipped, matrix to large!\n";
#endif

    if(err == EXIT_SUCCESS)
    {
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        for(uint32_t i = 0; i < repeat; ++i)
        {
            callSgemm();
        }
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(endT - beginT).count();


#if ALPAKA_HAS_CUBLAS == 1
        onHost::wait(queue);
        auto const beginTBlas = std::chrono::high_resolution_clock::now();
        for(uint32_t i = 0; i < repeat; ++i)
        {
            callSgemmCUBlas();
        }
        onHost::wait(queue);
        auto const endTBlas = std::chrono::high_resolution_clock::now();
        double durationBlas = std::chrono::duration<double>(endTBlas - beginTBlas).count();
        std::cout << "cuBlas" << std::endl;
        std::cout << "  - kernel execution: " << durationBlas << " s" << std::endl;
        std::cout << "  - performance     : "
                  << static_cast<double>(flopCount) / 1.e12 / (durationBlas / static_cast<double>(repeat))
                  << " tflop/s" << std::endl;
#endif
#if ALPAKA_HAS_HIPBLAS == 1
        onHost::wait(queue);
        auto const beginTBlas = std::chrono::high_resolution_clock::now();
        for(uint32_t i = 0; i < repeat; ++i)
        {
            callSgemmHipBlas();
        }
        onHost::wait(queue);
        auto const endTBlas = std::chrono::high_resolution_clock::now();
        double durationBlas = std::chrono::duration<double>(endTBlas - beginTBlas).count();
        std::cout << "hipBlas" << std::endl;
        std::cout << "  - kernel execution: " << durationBlas << " s" << std::endl;
        std::cout << "  - performance     : "
                  << static_cast<double>(flopCount) / 1.e12 / (durationBlas / static_cast<double>(repeat))
                  << " tflop/s" << std::endl;
#endif

        std::cout << "alpaka" << std::endl;
        std::cout << "  - kernel execution: " << duration << " s" << std::endl;
        std::cout << "  - performance     : "
                  << static_cast<double>(flopCount) / 1.e12 / (duration / static_cast<double>(repeat)) << " tflop/s"
                  << std::endl;
        std::cout << "  - flop count      : " << flopCount << std::endl;
    }
    return err;
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
