/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEST_CASE("tutorial events and synchronization", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue0 = device.makeQueue();
    auto queue1 = device.makeQueue();
    auto event = device.makeEvent();
    int value = 0;

    // BEGIN-TUTORIAL-eventCreation
    queue0.enqueueHostFn([&value]() { value = 41; });
    queue0.enqueue(event);
    // END-TUTORIAL-eventCreation

    // BEGIN-TUTORIAL-eventWait
    queue1.waitFor(event);
    queue1.enqueueHostFn([&value]() { value += 1; });
    onHost::wait(queue1);
    // END-TUTORIAL-eventWait

    CHECK(event.isComplete());
    CHECK(value == 42);
}
