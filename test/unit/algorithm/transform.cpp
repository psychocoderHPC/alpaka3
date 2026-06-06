/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Cross stencil
 *
 * N-dimensional asymmetric cross stencil. The positive direction is weighted with 5 and the negative with 3.
 * The middle value is not weighted.
 */
struct CrossStencil
{
    constexpr auto operator()(concepts::SimdPtr auto const& in) const
    {
        using SimdPtrType = ALPAKA_TYPEOF(in);
        using VecIdxType = typename SimdPtrType::IdxType;

        // Load the value the simd pointer is pointing to.
        auto result = in.load();

        for(uint32_t d = 0u; d < VecIdxType::dim(); ++d)
        {
            /* Shift relative to the current element pointed to into the negative or positive direction within a
             * dimension.
             */
            concepts::Vector auto negative = VecIdxType::fill(0);
            concepts::Vector auto positive = VecIdxType::fill(0);
            negative[d] = -1;
            positive[d] = 1;

            result += in[negative].load() * 3;
            result += in[positive].load() * 5;
        }

        return result;
    }
};

template<typename T_DataType, typename T_StencilFn>
void testStencilTransform(auto cfg, concepts::Vector auto extentMd)
{
    using DataType = T_DataType;
    using Extent = std::decay_t<decltype(extentMd)>;

    auto deviceSpec = cfg[object::deviceSpec];
    alpaka::concepts::Executor auto exec = cfg[object::exec];

    auto computeDevSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!computeDevSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device computeDev = computeDevSelector.makeDevice(0);
    onHost::Queue computeQueue = computeDev.makeQueue();

    INFO("device spec: " << getName(deviceSpec));
    INFO("device name: " << computeDev.getName());
    INFO("executor   : " << exec.getName());
    INFO("extents    : " << extentMd);

    auto const halo = Extent::fill(1);
    auto const innerExtent = extentMd - Extent::fill(2);

    onHost::SharedBuffer computeIn = onHost::allocDeferred<DataType>(computeQueue, extentMd);
    // The output does not need helo's
    onHost::SharedBuffer computeOut = onHost::allocDeferred<DataType>(computeQueue, innerExtent);

    onHost::SharedBuffer hostIn = onHost::allocLike(onHost::makeHostDevice(), computeIn);
    onHost::SharedBuffer hostOut = onHost::allocLike(onHost::makeHostDevice(), computeOut);

    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            /* Use a fake size of 1000 for each direction, this would make the index human.
             * Accidentally switching the index order z,y,x in SImdPtr during the stencil access will be detracted
             * and easy to debug. e.g. (y,x) -> (13,5) will become 13005
             */
            int32_t value = alpaka::linearize(ALPAKA_TYPEOF(extentMd)::fill(1000), idx);

            hostIn[idx] = value;
        });

    onHost::memcpy(computeQueue, computeIn, hostIn);

    /* For the stencil operation we work only on the inner volume, this allows performing the stencil operation
     * without handling out of memory access at the boundaries.
     */
    auto shiftedComputeIn = computeIn.getView().getSubView(halo, innerExtent);

    onHost::transform(computeQueue, exec, computeOut, StencilFunc{T_StencilFn{}}, shiftedComputeIn);
    onHost::memcpy(computeQueue, hostOut, computeOut);
    onHost::wait(computeQueue);

    meta::ndLoopIncIdx(
        innerExtent,
        [&](concepts::Vector auto outIdx)
        {
            concepts::Vector auto const inIdx = outIdx + halo;

            DataType expected = hostIn[inIdx];

            for(uint32_t d = 0u; d < extentMd.dim(); ++d)
            {
                // absolut offset from the origin of hostIn including helo
                auto negative = inIdx;
                auto positive = inIdx;

                --negative[d];
                ++positive[d];

                expected += hostIn[negative] * 3;
                expected += hostIn[positive] * 5;
            }

            REQUIRE(hostOut[outIdx] == expected);
        });
}

/** Validate the transform stencil feature.
 *
 * Additionally, this checks take care that SimdPtr::operator[]() works as expected.
 */
TEMPLATE_LIST_TEST_CASE("transform SimdPtr cross stencil", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int32_t;

    auto extentMdList = std::make_tuple(Vec{4096}, Vec{128, 2048}, Vec{16, 128, 1024});

    std::apply(
        [&](auto... extents) { (testStencilTransform<DataType, CrossStencil>(cfg, extents), ...); },
        extentMdList);
}
