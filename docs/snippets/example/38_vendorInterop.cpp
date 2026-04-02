/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

using namespace alpaka;

namespace vendorTutorial
{
    // BEGIN-TUTORIAL-vendorSymbol
    ALPAKA_FN_SYMBOL(AffineTransform, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);
    // END-TUTORIAL-vendorSymbol

    // BEGIN-TUTORIAL-vendorFallback
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void fnRegister(AffineTransform::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>)
    {
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void fnDispatch(
        AffineTransform::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        float scale,
        float shift,
        alpaka::concepts::IMdSpan auto&& input)
    {
        alpaka::onHost::transform(
            ALPAKA_FORWARD(queue),
            ALPAKA_FORWARD(output),
            ScalarFunc{[=] ALPAKA_FN_ACC(float const& value)
                       { return scale * value + shift; }},
            ALPAKA_FORWARD(input));
    }
    // END-TUTORIAL-vendorFallback

    // BEGIN-TUTORIAL-vendorHost
    constexpr void fnRegister(AffineTransform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
    {
    }

    constexpr void fnDispatch(
        AffineTransform::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
        auto&& queue,
        alpaka::concepts::IMdSpan auto&& output,
        float scale,
        float shift,
        alpaka::concepts::IMdSpan auto&& input)
    {
        auto outPtr = output.data();
        queue.enqueueHostFn(
            [=]()
            {
                std::transform(
                    input.data(),
                    input.data() + input.getExtents().x(),
                    outPtr,
                    [=](float value)
                    { return scale * value + shift; });
            });
    }
    // END-TUTORIAL-vendorHost
} // namespace vendorTutorial

TEST_CASE("tutorial vendor interop dispatch", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::blocking);

    std::array<float, 5u> hostInput{1.f, 2.f, 3.f, 4.f, 5.f};
    std::array<float, 5u> hostOutput{};

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-vendorCall
    if(vendorTutorial::AffineTransform::isRegistered(queue))
    {
        vendorTutorial::AffineTransform::call(queue, outputBuffer, 2.0f, 0.5f, inputBuffer);
    }
    // END-TUTORIAL-vendorCall

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    CHECK(vendorTutorial::AffineTransform::isRegistered(queue));
    CHECK(hostOutput[0] == 2.5f);
    CHECK(hostOutput[1] == 4.5f);
    CHECK(hostOutput[2] == 6.5f);
    CHECK(hostOutput[3] == 8.5f);
    CHECK(hostOutput[4] == 10.5f);
}
