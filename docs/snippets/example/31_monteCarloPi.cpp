/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

namespace
{
    // BEGIN-TUTORIAL-piKernel
    struct MonteCarloPiKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan auto hits, uint32_t seed) const
        {
            for(auto [idx] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{hits.getExtents()}))
            {
                rand::engine::Philox4x32x10 engine(seed + idx);
                auto uniform = rand::distribution::UniformReal{0.0f, 1.0f, rand::interval::co};
                auto x = uniform(engine);
                auto y = uniform(engine);
                hits[idx] = (x * x + y * y <= 1.0f) ? 1u : 0u;
            }
        }
    };
    // END-TUTORIAL-piKernel
} // namespace

TEST_CASE("tutorial monte carlo pi", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::blocking);
    auto exec = exec::cpuSerial;

    constexpr uint32_t numSamples = 16384u;
    auto hitBuffer = onHost::alloc<uint32_t>(device, Vec{numSamples});
    auto hitCountBuffer = onHost::alloc<uint32_t>(device, Vec{1u});
    auto hostHitCount = onHost::allocHostLike(hitCountBuffer);

    auto frameSpec = onHost::getFrameSpec<uint32_t>(device, hitBuffer.getExtents());

    // BEGIN-TUTORIAL-piLaunch
    queue.enqueue(frameSpec, KernelBundle{MonteCarloPiKernel{}, hitBuffer, 2026u});
    onHost::reduce(queue, exec, 0u, hitCountBuffer, std::plus{}, hitBuffer);
    // END-TUTORIAL-piLaunch

    onHost::memcpy(queue, hostHitCount, hitCountBuffer);
    onHost::wait(queue);

    // BEGIN-TUTORIAL-piEstimate
    auto estimatedPi = 4.0f * static_cast<float>(hostHitCount[0]) / static_cast<float>(numSamples);
    // END-TUTORIAL-piEstimate

    CHECK(estimatedPi == Catch::Approx(3.14159f).margin(0.15f));
}
