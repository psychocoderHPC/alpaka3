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

TEMPLATE_LIST_TEST_CASE("event re enqueue", "", TestApis)
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
    onHost::Queue consumerQueue = device.makeQueue(queueKind::blocking);
    onHost::Event readyEvent = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 19u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceBuffer);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i * 5u + 2u;
    onHost::fill(producerQueue, deviceBuffer, 0u);

    // The first record marks "transfer finished", but the second record must move the same event forward to the
    // later kernel completion point.
    onHost::memcpy(producerQueue, deviceBuffer, hostInput);
    producerQueue.enqueue(readyEvent);
    producerQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetInPlaceKernel{}, deviceBuffer, offset});
    producerQueue.enqueue(readyEvent);

    // If waitFor latched onto the old enqueue, this copy could observe pre-kernel data instead of the transformed
    // data.
    consumerQueue.waitFor(readyEvent);
    onHost::memcpy(consumerQueue, hostOutput, deviceBuffer);
    onHost::wait(consumerQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == hostInput[i] + offset);
    }
}
