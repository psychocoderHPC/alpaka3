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

struct WriteIntermediateKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto input,
        alpaka::concepts::IMdSpan auto intermediate,
        std::uint32_t addend) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            intermediate[idx.x()] = input[idx.x()] + addend;
    }
};

struct ConsumeIntermediateKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto intermediate,
        alpaka::concepts::IMdSpan auto output,
        std::uint32_t factor) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            output[idx.x()] = intermediate[idx.x()] * factor;
    }
};

TEMPLATE_LIST_TEST_CASE("compute to compute event handoff", "", TestApis)
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
    onHost::Event intermediateReady = device.makeEvent();

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t addend = 5u;
    constexpr std::uint32_t factor = 2u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceInput = onHost::alloc<std::uint32_t>(device, extent);
    auto intermediateBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto deviceOutput = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceOutput);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i + 1u;
    onHost::fill(producerQueue, intermediateBuffer, 0u);
    onHost::fill(consumerQueue, deviceOutput, 0u);

    onHost::memcpy(producerQueue, deviceInput, hostInput);

    // Queue A produces intermediate data and records the exact point where queue B may safely consume it.
    producerQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{WriteIntermediateKernel{}, deviceInput, intermediateBuffer, addend});
    producerQueue.enqueue(intermediateReady);

    // Queue B is separate on purpose: without the event, the dependent kernel could observe incomplete data.
    consumerQueue.waitFor(intermediateReady);
    consumerQueue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{ConsumeIntermediateKernel{}, intermediateBuffer, deviceOutput, factor});
    onHost::memcpy(consumerQueue, hostOutput, deviceOutput);
    onHost::wait(consumerQueue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == (hostInput[i] + addend) * factor);
    }
}
