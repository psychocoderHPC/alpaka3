/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <type_traits>
#include <utility>
#include <vector>

using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("mdIterator host coverage", "[mem][mdIterator][iterator]")
{
    auto collectValues = []<typename T_Range>(T_Range&& range)
    {
        using Value = std::remove_cvref_t<decltype(*range.begin())>;
        std::vector<Value> values;
        for(auto&& value : range)
            values.push_back(value);
        return values;
    };

    SECTION("zero-extent MdSpan and View are empty")
    {
        // Zero extents should terminate immediately instead of yielding a bogus first element.
        std::array<uint32_t, 1> storage{42u};
        auto const emptyMdSpan = alpaka::makeMdSpan(storage.data(), alpaka::Vec{0u, 3u});
        auto const emptyView = alpaka::makeView(api::host, storage.data(), alpaka::Vec{0u, 3u});

        REQUIRE(emptyMdSpan.begin() == emptyMdSpan.end());
        REQUIRE(emptyView.begin() == emptyView.end());
        REQUIRE(collectValues(emptyMdSpan).empty());
        REQUIRE(collectValues(emptyView).empty());
    }

    SECTION("1D MdSpan iteration stays linear")
    {
        // A linear buffer should be visited in storage order on the host path.
        std::array<uint32_t, 5> storage{3u, 6u, 9u, 12u, 15u};
        auto span = alpaka::makeMdSpan(storage.data(), alpaka::Vec{storage.size()});

        REQUIRE(collectValues(span) == std::vector<uint32_t>{3u, 6u, 9u, 12u, 15u});
    }

    SECTION("3D View iteration follows linearize order")
    {
        // The iterator must keep the last dimension fastest so traversal matches linearized indexing.
        auto const extents = alpaka::Vec{2u, 3u, 4u};
        auto buffer = onHost::allocHost<uint32_t>(extents);
        auto view = buffer.getView();

        meta::ndLoopIncIdx(
            extents,
            [&](alpaka::concepts::Vector<uint32_t, 3> auto idx)
            { view[idx] = static_cast<uint32_t>(linearize(extents, idx)); });

        auto const visited = collectValues(view);
        REQUIRE(visited.size() == extents.product());
        REQUIRE(visited.front() == 0u);
        REQUIRE(visited.back() == extents.product() - 1u);

        for(uint32_t linearIdx = 0; linearIdx < visited.size(); ++linearIdx)
            CHECK(visited[linearIdx] == linearIdx);
    }

    SECTION("mutable View iteration writes back into the underlying buffer")
    {
        // Writing through a non-const iterator must update the storage the view refers to.
        auto const extents = alpaka::Vec{2u, 3u};
        auto buffer = onHost::allocHost<int>(extents);
        auto view = buffer.getView();

        int nextValue = 10;
        for(int& value : view)
        {
            value = nextValue;
            nextValue += 5;
        }

        REQUIRE(view[alpaka::Vec{0u, 0u}] == 10);
        REQUIRE(view[alpaka::Vec{0u, 2u}] == 20);
        REQUIRE(view[alpaka::Vec{1u, 0u}] == 25);
        REQUIRE(view[alpaka::Vec{1u, 2u}] == 35);
        CHECK(buffer[alpaka::Vec{1u, 1u}] == 30);
    }

    SECTION("const View iteration is read-only and preserves order")
    {
        // Const iteration should expose const references while traversing the same order as mutable iteration.
        std::array<int, 6> storage{2, 4, 6, 8, 10, 12};
        auto view = alpaka::makeView(api::host, storage.data(), alpaka::Vec{2u, 3u});
        auto const constView = view.getConstView();

        static_assert(!std::is_const_v<std::remove_reference_t<decltype(*view.begin())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(*constView.begin())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(*std::as_const(view).begin())>>);

        REQUIRE(collectValues(view) == std::vector<int>{2, 4, 6, 8, 10, 12});
        REQUIRE(collectValues(constView) == collectValues(view));
    }

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

    SECTION("packed makeView derives packed pitches and getMdSpan keeps that layout")
    {
        // The packed raw-pointer overload must derive pitches from extents instead of accepting a padded layout, and
        // the immediate MdSpan view should expose the same packed contract.
        auto const extents = alpaka::Vec{2u, 3u, 4u};
        auto const packedPitches = alpaka::calculatePitchesFromExtents<int>(extents);
        auto const explicitPitches = alpaka::Vec{80u, 20u, 4u};
        alignas(64) std::array<int, 2 * 3 * 4> storage{};
        for(std::size_t i = 0; i < storage.size(); ++i)
            storage[i] = static_cast<int>(i);

        auto packedView = alpaka::makeView(api::host, storage.data(), extents, alpaka::Alignment<64>{});
        auto mdSpan = packedView.getMdSpan();

        REQUIRE(packedView.getApi() == api::host);
        REQUIRE(packedView.getExtents() == extents);
        REQUIRE(packedView.getPitches() == packedPitches);
        REQUIRE(packedView.getPitches() != explicitPitches);
        REQUIRE(packedView.data() == storage.data());
        STATIC_REQUIRE(std::is_same_v<decltype(packedView.getAlignment()), alpaka::Alignment<64>>);

        storage[15] = 1500;
        storage[23] = 2300;
        auto const sampleIdx = alpaka::Vec{1u, 0u, 3u};
        REQUIRE(packedView[sampleIdx] == 1500);
        REQUIRE(&packedView[sampleIdx] == &storage[15]);

        REQUIRE(mdSpan.getExtents() == packedView.getExtents());
        REQUIRE(mdSpan.getPitches() == packedView.getPitches());
        REQUIRE(mdSpan.data() == packedView.data());
        STATIC_REQUIRE(std::is_same_v<decltype(mdSpan.getAlignment()), alpaka::Alignment<64>>);

        mdSpan[sampleIdx] = 777;
        REQUIRE(packedView[sampleIdx] == 777);
        REQUIRE(storage[15] == 777);
        REQUIRE(storage[23] == 2300);
        REQUIRE(&mdSpan[sampleIdx] == &packedView[sampleIdx]);
    }

    SECTION("pre-increment and post-increment advance one element at a time")
    {
        // Forward-iterator increments need to preserve the old value for post-increment and return self for
        // pre-increment.
        std::array<int, 4> storage{7, 11, 13, 17};
        auto span = alpaka::makeMdSpan(storage.data(), alpaka::Vec{storage.size()});

        auto iter = span.begin();
        REQUIRE(*iter == 7);

        auto& preIncrement = ++iter;
        REQUIRE(&preIncrement == &iter);
        REQUIRE(*iter == 11);

        auto postIncrement = iter++;
        REQUIRE(*postIncrement == 11);
        REQUIRE(*iter == 13);

        ++iter;
        REQUIRE(*iter == 17);
        ++iter;
        REQUIRE(iter == span.end());
    }
}
