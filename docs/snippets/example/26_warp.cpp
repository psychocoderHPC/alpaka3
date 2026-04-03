/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

namespace
{
    // BEGIN-TUTORIAL-warpKernel
    struct WarpSumKernel
    {
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            concepts::IDataSource auto const& in,
            concepts::IMdSpan auto out) const
        {
            auto const warpSize = onAcc::warp::getSize(acc);
            auto const idxInWarp = onAcc::warp::getLaneIdx(acc);
            auto const workSize = pCast<uint32_t>(out.getExtents());
            for(auto [blockBase] :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{0u, workSize, warpSize}))
            {
                auto value = in[Vec{blockBase + idxInWarp}];
                for(uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
                    value += onAcc::warp::shflDown(acc, value, offset);

                if(onAcc::warp::getLaneIdx(acc) == 0u)
                {
                    out[blockBase / warpSize] = value;
                }
            }
        }
    };

    // END-TUTORIAL-warpKernel
} // namespace

TEST_CASE("tutorial warp shuffle reduction", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::blocking);
    auto selector = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu});
    auto const warpSize = selector.getDeviceProperties(0).warpSize;

    auto const blocks = 2u;
    auto const threadsPerBlock = warpSize;

    std::vector<uint32_t> hostInput(blocks * threadsPerBlock);
    std::vector<uint32_t> hostOutput(blocks, 0u);
    std::vector<uint32_t> expectedOutput(blocks, 0u);

    for(uint32_t blockIdx = 0; blockIdx < blocks; ++blockIdx)
    {
        for(uint32_t laneIdx = 0; laneIdx < warpSize; ++laneIdx)
        {
            auto const value = blockIdx * warpSize + laneIdx + 1u;
            hostInput[blockIdx * threadsPerBlock + laneIdx] = value;
            expectedOutput[blockIdx] += value;
        }
    }

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);
    onHost::memset(queue, outputBuffer, 0x00);

    // BEGIN-TUTORIAL-warpLaunch
    auto frameSpec = onHost::FrameSpec{Vec{blocks}, Vec{threadsPerBlock}};
    queue.enqueue(frameSpec, KernelBundle{WarpSumKernel{}, inputBuffer, outputBuffer});
    // END-TUTORIAL-warpLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t blockIdx = 0; blockIdx < blocks; ++blockIdx)
        CHECK(hostOutput[blockIdx] == expectedOutput[blockIdx]);
}
