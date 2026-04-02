/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

TEST_CASE("tutorial views and subviews", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue();

    std::vector<int> hostData{0, 1, 2, 3, 4, 5, 6, 7};

    // BEGIN-TUTORIAL-viewCreation
    auto hostView = makeView(hostData);
    auto middleView = hostView.getSubView(std::size_t{2}, std::size_t{4});
    // END-TUTORIAL-viewCreation

    CHECK(hostView.getExtents().x() == 8u);
    CHECK(middleView.getExtents().x() == 4u);
    CHECK(middleView[Vec{std::size_t{0}}] == 2);
    CHECK(middleView[Vec{std::size_t{3}}] == 5);

    // BEGIN-TUTORIAL-viewCopy
    auto deviceBuffer = onHost::allocLike(device, hostView);
    onHost::memcpy(queue, deviceBuffer, hostView);

    auto hostSlice = onHost::allocHost<int>(4u);
    onHost::memcpy(queue, hostSlice, deviceBuffer.getSubView(Vec{std::size_t{2}}, Vec{std::size_t{4}}));
    onHost::wait(queue);
    // END-TUTORIAL-viewCopy

    CHECK(hostSlice[Vec{0u}] == 2);
    CHECK(hostSlice[Vec{1u}] == 3);
    CHECK(hostSlice[Vec{2u}] == 4);
    CHECK(hostSlice[Vec{3u}] == 5);
}
