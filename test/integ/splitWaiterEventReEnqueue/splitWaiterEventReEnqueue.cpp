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

TEMPLATE_LIST_TEST_CASE("split waiter event re enqueue", "", TestApis)
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
    onHost::Queue producerQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue firstConsumerQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue secondConsumerQueue = device.makeQueue(queueKind::blocking);
    onHost::Event readyEvent = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 23u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostBeforeKernel = onHost::allocHostLike(deviceBuffer);
    auto hostAfterKernel = onHost::allocHostLike(deviceBuffer);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i * 7u + 3u;
    onHost::fill(producerQueue, deviceBuffer, 0u);

    onHost::memcpy(producerQueue, deviceBuffer, hostInput);
    producerQueue.enqueue(readyEvent);

    // Queue B waits before the event is re-enqueued, so it must stay tied to the original transfer-complete point.
    firstConsumerQueue.waitFor(readyEvent);
    onHost::memcpy(firstConsumerQueue, hostBeforeKernel, deviceBuffer);

    producerQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetInPlaceKernel{}, deviceBuffer, offset});

    // Reusing the same event publishes a later completion point for subsequent waiters without moving existing ones.
    producerQueue.enqueue(readyEvent);

    // If this later wait reused the first enqueue instead, it could copy stale pre-kernel data instead of the update.
    secondConsumerQueue.waitFor(readyEvent);
    onHost::memcpy(secondConsumerQueue, hostAfterKernel, deviceBuffer);

    onHost::wait(firstConsumerQueue);
    onHost::wait(secondConsumerQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostBeforeKernel[i] == hostInput[i]);
        CHECK(hostAfterKernel[i] == hostInput[i] + offset);
    }
}
