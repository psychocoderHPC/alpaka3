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

struct AddOffsetInPlaceKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto data, std::uint32_t offset) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            data[idx.x()] += offset;
    }
};

TEMPLATE_LIST_TEST_CASE("cross queue event re enqueue", "", TestApis)
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
    onHost::Queue transferQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue transformQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue copyBackQueue = device.makeQueue(queueKind::blocking);
    onHost::Event readyEvent = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 29u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceBuffer);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i * 11u + 5u;
    onHost::fill(transferQueue, deviceBuffer, 0u);

    onHost::memcpy(transferQueue, deviceBuffer, hostInput);
    transferQueue.enqueue(readyEvent);

    // Queue B first consumes the transfer-complete event, then publishes the same event name again for its later
    // kernel completion point.
    transformQueue.waitFor(readyEvent);
    transformQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetInPlaceKernel{}, deviceBuffer, offset});
    transformQueue.enqueue(readyEvent);

    // If the event stayed bound to queue A's first enqueue, this copy could observe only the transferred data and
    // miss queue B's transformation.
    copyBackQueue.waitFor(readyEvent);
    onHost::memcpy(copyBackQueue, hostOutput, deviceBuffer);
    onHost::wait(copyBackQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == hostInput[i] + offset);
    }
}
