/* Copyright 2026 OpenAI
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
        ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IDataSource auto const& in, concepts::IMdSpan auto out) const
        {
            auto const threadsPerBlock = acc[layer::thread].count().product();
            auto const warpSize = onAcc::warp::getSize(acc);
            auto const blockBase = acc[layer::block].idx().x() * threadsPerBlock;

            for(auto [localThread] :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{threadsPerBlock}}))
            {
                auto value = in[Vec{blockBase + localThread}];

                for(uint32_t offset = warpSize / 2u; offset > 0u; offset /= 2u)
                    value += onAcc::warp::shflDown(acc, value, offset);

                if(onAcc::warp::getLaneIdx(acc) == 0u)
                {
                    out[Vec{acc[layer::block].idx().x()}] = value;
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

    std::vector<uint32_t> hostInput(blocks * threadsPerBlock, 1u);
    std::vector<uint32_t> hostOutput(blocks, 0u);

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

    for(auto sum : hostOutput)
        CHECK(sum == warpSize);
}
