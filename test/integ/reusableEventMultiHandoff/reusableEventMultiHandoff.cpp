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

struct AddOffsetKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto input,
        alpaka::concepts::IMdSpan auto output,
        std::uint32_t offset) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            output[idx.x()] = input[idx.x()] + offset;
    }
};

TEMPLATE_LIST_TEST_CASE("reusable event multi handoff", "", TestApis)
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
    onHost::Queue transformQueue = device.makeQueue(queueKind::blocking);
    onHost::Queue mirrorQueue = device.makeQueue(queueKind::blocking);
    onHost::Event transferFinished = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 17u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceInput = onHost::alloc<std::uint32_t>(device, extent);
    auto deviceOutput = onHost::alloc<std::uint32_t>(device, extent);
    auto hostMirror = onHost::allocHostLike(deviceInput);
    auto hostOutput = onHost::allocHostLike(deviceOutput);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i * 3u + 1u;
    onHost::fill(producerQueue, deviceInput, 0u);
    onHost::fill(producerQueue, deviceOutput, 0u);

    // Queue A publishes the point where the device input buffer is safe for other queues to consume.
    onHost::memcpy(producerQueue, deviceInput, hostInput);
    producerQueue.enqueue(transferFinished);

    // Queue B reuses that same event before launching a kernel that depends on the copied device input.
    transformQueue.waitFor(transferFinished);
    transformQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetKernel{}, deviceInput, deviceOutput, offset});
    onHost::memcpy(transformQueue, hostOutput, deviceOutput);

    // Queue C reuses the same event again so this memcpy cannot race the original host->device transfer.
    mirrorQueue.waitFor(transferFinished);
    onHost::memcpy(mirrorQueue, hostMirror, deviceInput);

    onHost::wait(transformQueue);
    onHost::wait(mirrorQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostMirror[i] == hostInput[i]);
        CHECK(hostOutput[i] == hostInput[i] + offset);
    }
}
