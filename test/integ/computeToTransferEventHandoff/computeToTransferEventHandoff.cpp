/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

struct MultiplyKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto data, std::uint32_t factor) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            data[idx.x()] *= factor;
    }
};

TEMPLATE_LIST_TEST_CASE("compute to transfer event handoff", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    onHost::Queue computeQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue transferQueue = device.makeQueue(queueKind::blocking);
    onHost::Event computeFinished = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t factor = 3u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceBuffer);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i + 1u;
    onHost::fill(computeQueue, deviceBuffer, 0u);

    // Queue A performs the compute work and records the point where the transformed buffer is ready.
    onHost::memcpy(computeQueue, deviceBuffer, hostInput);
    computeQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{MultiplyKernel{}, deviceBuffer, factor});
    computeQueue.enqueue(computeFinished);

    // Queue B stays separate so the device->host copy only starts after the event establishes cross-queue ordering.
    transferQueue.waitFor(computeFinished);
    onHost::memcpy(transferQueue, hostOutput, deviceBuffer);
    onHost::wait(transferQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == hostInput[i] * factor);
    }
}
