/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace alpaka;

namespace
{
    // BEGIN-TUTORIAL-chunkedKernel
    struct ChunkedVectorAddKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IMdSpan auto out,
            concepts::IDataSource auto const& in0,
            concepts::IDataSource auto const& in1) const
        {
            auto frameExtent = acc[frame::extent];
            auto linearNumFrames = Vec{acc[frame::count].product()};
            auto linearFrameExtent = Vec{frameExtent.product()};

            for(auto linearFrameIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::linearBlocksInGrid, IdxRange{linearNumFrames}))
            {
                auto tile = onAcc::declareSharedMdArray<int, uniqueId()>(acc, frameExtent);

                for(auto linearFrameElem :
                    onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInBlock, IdxRange{linearFrameExtent}))
                {
                    auto globalIdx = linearFrameIdx * frameExtent + linearFrameElem;
                    tile[linearFrameElem] = in0[globalIdx];
                }

                onAcc::syncBlockThreads(acc);

                for(auto linearFrameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::linearThreadsInBlock,
                        IdxRange{linearFrameExtent},
                        onAcc::traverse::tiled))
                {
                    auto globalIdx = linearFrameIdx * frameExtent + linearFrameElem;
                    out[globalIdx] = tile[linearFrameElem] + in1[globalIdx];
                }

                onAcc::syncBlockThreads(acc);
            }
        }
    };
    // END-TUTORIAL-chunkedKernel
} // namespace

TEST_CASE("tutorial chunked frames kernel", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::blocking);

    std::array<int, 8u> hostIn0{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8u> hostIn1{10, 10, 10, 10, 10, 10, 10, 10};
    std::array<int, 8u> hostOut{};

    auto in0Buffer = onHost::allocLike(device, hostIn0);
    auto in1Buffer = onHost::allocLike(device, hostIn1);
    auto outBuffer = onHost::allocLike(device, hostIn0);

    onHost::memcpy(queue, in0Buffer, hostIn0);
    onHost::memcpy(queue, in1Buffer, hostIn1);

    // BEGIN-TUTORIAL-chunkedLaunch
    constexpr auto frameExtent = CVec<uint32_t, 4u>{};
    auto numFrames = Vec{static_cast<uint32_t>(hostOut.size())} / frameExtent;
    auto frameSpec = onHost::FrameSpec{numFrames, frameExtent};

    queue.enqueue(frameSpec, KernelBundle{ChunkedVectorAddKernel{}, outBuffer, in0Buffer, in1Buffer});
    // END-TUTORIAL-chunkedLaunch

    onHost::memcpy(queue, hostOut, outBuffer);
    onHost::wait(queue);

    CHECK(hostOut[0] == 10);
    CHECK(hostOut[1] == 11);
    CHECK(hostOut[2] == 12);
    CHECK(hostOut[3] == 13);
    CHECK(hostOut[4] == 14);
    CHECK(hostOut[5] == 15);
    CHECK(hostOut[6] == 16);
    CHECK(hostOut[7] == 17);
}
