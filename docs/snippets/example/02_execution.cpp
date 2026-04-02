/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEST_CASE("tutorial enumerate backends and executors", "[docs]")
{
    // BEGIN-TUTORIAL-enumerateDeviceSpec
    auto deviceSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};
    auto selector = onHost::makeDeviceSelector(deviceSpec);

    auto numDevices = selector.getDeviceCount();
    REQUIRE(numDevices >= 1u);

    auto properties = selector.getDeviceProperties(0u);
    auto device = selector.makeDevice(0u);
    // END-TUTORIAL-enumerateDeviceSpec

    CHECK(properties.warpSize >= 1u);
    CHECK(!device.getName().empty());

    std::size_t numVisitedBackends = 0u;
    // BEGIN-TUTORIAL-enumerateBackends
    onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            ++numVisitedBackends;

            auto backendDeviceSpec = backend[object::deviceSpec];
            auto backendExec = backend[object::exec];
            auto backendSelector = onHost::makeDeviceSelector(backendDeviceSpec);
            auto backendDevice = backendSelector.makeDevice(0u);
            auto backendQueue = backendDevice.makeQueue();

            backendQueue.enqueueHostFn([]() noexcept {});
            onHost::wait(backendQueue);

            alpaka::unused(backendExec);
            return EXIT_SUCCESS;
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
    // END-TUTORIAL-enumerateBackends

    CHECK(numVisitedBackends >= 1u);
}
