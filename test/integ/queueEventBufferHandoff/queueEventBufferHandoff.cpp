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

struct AddOffsetAfterHandoffKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto data, std::uint32_t offset) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            data[idx.x()] += offset;
    }
};

TEMPLATE_LIST_TEST_CASE("queue + event buffer handoff", "", TestApis)
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
    onHost::Event dataReady = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 11u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceBuffer);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i * 2u;
    onHost::fill(producerQueue, deviceBuffer, 0u);

    // Queue A produces the shared device buffer and records the point where the handoff becomes valid.
    onHost::memcpy(producerQueue, deviceBuffer, hostInput);
    producerQueue.enqueue(dataReady);

    // Queue B is intentionally separate so this test covers cross-queue ordering instead of plain FIFO execution.
    consumerQueue.waitFor(dataReady);
    consumerQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetAfterHandoffKernel{}, deviceBuffer, offset});
    onHost::memcpy(consumerQueue, hostOutput, deviceBuffer);
    onHost::wait(consumerQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == hostInput[i] + offset);
    }
}
