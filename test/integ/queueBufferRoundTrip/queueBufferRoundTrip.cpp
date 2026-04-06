/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

struct AddOffsetKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto data, std::uint32_t offset) const
    {
        for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
            data[idx.x()] += offset;
    }
};

TEMPLATE_LIST_TEST_CASE("queue + memcpy + kernel round trip", "", TestApis)
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
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    constexpr Vec extent = Vec{32u};
    constexpr auto frameExtent = CVec<std::uint32_t, 8u>{};
    constexpr std::uint32_t offset = 7u;

    auto hostInput = onHost::alloc<std::uint32_t>(onHost::makeHostDevice(), extent);
    auto deviceBuffer = onHost::alloc<std::uint32_t>(device, extent);
    auto hostOutput = onHost::allocHostLike(deviceBuffer);

    // This test intentionally keeps the data flow obvious:
    // fill on host, copy to device, run one kernel, copy back, then validate the final values.
    for(std::uint32_t i = 0; i < extent.x(); ++i)
        hostInput[i] = i;
    onHost::fill(queue, deviceBuffer, 99u);

    onHost::memcpy(queue, deviceBuffer, hostInput);
    queue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameExtent, frameExtent},
        KernelBundle{AddOffsetKernel{}, deviceBuffer, offset});
    onHost::memcpy(queue, hostOutput, deviceBuffer);
    onHost::wait(queue);

    for(std::uint32_t i = 0; i < extent.x(); ++i)
    {
        CAPTURE(i);
        CHECK(hostOutput[i] == i + offset);
    }
}
