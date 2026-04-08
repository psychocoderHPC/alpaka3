/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <type_traits>
#include <vector>

using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("mdIterator", "")
{
    constexpr auto numElements = CVec<size_t, 17, 31>{};
    alpaka::concepts::IBuffer auto span = onHost::allocHost<uint32_t>(numElements);

    size_t counter = 0u;
    for(uint32_t& v : span)
        v = counter++;

    // validate by using the forward iterator
    size_t refence = 0u;
    for(uint32_t v : span)
    {
        CHECK(v == refence);
        ++refence;
    }

    // validate without using the forward iterator
    meta::ndLoopIncIdx(
        numElements,
        [&](alpaka::concepts::Vector<size_t, 2> auto idx) { CHECK(span[idx] == linearize(numElements, idx)); });
}

TEST_CASE("mdIterator getMdSpan host coverage", "[mem][mdIterator][mdspan]")
{
    auto collectValues = []<typename T_Range>(T_Range&& range)
    {
        using Value = std::remove_cvref_t<decltype(*range.begin())>;
        std::vector<Value> values;
        for(auto&& value : range)
            values.push_back(value);
        return values;
    };

    SECTION("View::getMdSpan preserves layout, write-through, and const iteration")
    {
        // The MdSpan returned from a host view should alias the same storage and iterate in the same linear order.
        alignas(32) std::array<int, 2 * 3 * 4> storage{};
        for(std::size_t i = 0; i < storage.size(); ++i)
            storage[i] = static_cast<int>(i);

        auto view = alpaka::makeView(api::host, storage.data(), alpaka::Vec{2u, 3u, 4u}, alpaka::Alignment<32>{});
        auto mdSpan = view.getMdSpan();
        auto const constMdSpan = view.getConstView().getMdSpan();

        REQUIRE(mdSpan.getExtents() == view.getExtents());
        REQUIRE(mdSpan.getPitches() == view.getPitches());
        REQUIRE(mdSpan.data() == view.data());
        STATIC_REQUIRE(std::is_same_v<decltype(mdSpan.getAlignment()), alpaka::Alignment<32>>);
        STATIC_REQUIRE(std::is_same_v<decltype(constMdSpan.getAlignment()), alpaka::Alignment<32>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(*mdSpan.begin())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(*constMdSpan.begin())>>);

        auto const sampleIdx = alpaka::Vec{1u, 2u, 3u};
        mdSpan[sampleIdx] = 777;
        REQUIRE(view[sampleIdx] == 777);
        REQUIRE(&mdSpan[sampleIdx] == &view[sampleIdx]);
        REQUIRE(&constMdSpan[sampleIdx] == &view[sampleIdx]);

        auto const visited = collectValues(mdSpan);
        REQUIRE(visited.size() == storage.size());
        REQUIRE(visited.front() == 0);
        REQUIRE(visited.back() == 777);
        REQUIRE(collectValues(constMdSpan) == visited);
    }
}
