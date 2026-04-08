/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/testMacros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <type_traits>

TEST_CASE("1D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int x = 10;
    auto buffer0 = alpaka::onHost::allocHost<int>(x);
    for(auto i = 0; i < x; ++i)
    {
        buffer0[i] = i;
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, single integer, define extent")
    {
        constexpr int sub_size = 8;
        auto sub_view0 = view0.getSubView(sub_size);
        REQUIRE(sub_view0.getExtents() == alpaka::Vec{sub_size});
        for(auto i = 0; i < sub_size; ++i)
        {
            REQUIRE_MESSAGE(sub_view0[i] == i, "i=" << i);
        }
    }

    SECTION("getSubView, single integer, define offset and extent")
    {
        constexpr int offset = 2;
        constexpr int end = 7;
        auto sub_view0 = view0.getSubView(offset, end);

        REQUIRE(sub_view0.getExtents() == alpaka::Vec{end});
        for(auto i = 0; i < sub_view0.getExtents()[0]; ++i)
        {
            REQUIRE_MESSAGE(sub_view0[i] == i + offset, "i=" << i);
        }
    }

    SECTION("getSubView, zero extent in 1D stays empty and does not iterate")
    {
        // Zero-sized 1D subviews should preserve the requested extent and produce an empty host iteration range.
        auto sub_view0 = view0.getSubView(0);

        REQUIRE(sub_view0.getExtents() == alpaka::Vec{0});

        auto count = 0;
        for([[maybe_unused]] auto&& value : sub_view0)
        {
            alpaka::unused(value);
            ++count;
        }
        REQUIRE(count == 0);
    }
}

TEST_CASE("3D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int z = 3;
    constexpr int y = 5;
    constexpr int x = 4;
    alpaka::Vec total_extents{z, y, x};
    REQUIRE(total_extents.z() == z);
    REQUIRE(total_extents.y() == y);
    REQUIRE(total_extents.x() == x);

    auto buffer0 = alpaka::onHost::allocHost<int>(total_extents);

    for(auto vec : alpaka::IdxRange{total_extents})
    {
        buffer0[vec] = vec.z() * 100 + vec.y() * 10 + vec.x();
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, define extent")
    {
        alpaka::Vec extents_subview0{z - 1, y - 1, x - 1};
        auto sub_view0 = view0.getSubView(extents_subview0);
        REQUIRE(sub_view0.getApi() == view0.getApi());

        STATIC_REQUIRE(sub_view0.dim() == extents_subview0.dim());
        REQUIRE(sub_view0.getExtents() == extents_subview0);

        for(auto vec : alpaka::IdxRange{extents_subview0})
        {
            REQUIRE_MESSAGE(sub_view0[vec] == vec.z() * 100 + vec.y() * 10 + vec.x(), "vec=" << vec);
        }
    }

    SECTION("getSubView, define offset and extent")
    {
        alpaka::Vec offset_subview0{1, 2, 1};
        alpaka::Vec extents_subview0{z - 1, y - 3, x - 1};

        auto sub_view0 = view0.getSubView(offset_subview0, extents_subview0);
        REQUIRE(sub_view0.getApi() == view0.getApi());

        STATIC_REQUIRE(sub_view0.dim() == extents_subview0.dim());
        REQUIRE(sub_view0.getExtents() == extents_subview0);

        REQUIRE(sub_view0[alpaka::Vec{0, 0, 0}] == view0[offset_subview0]);
        REQUIRE(
            sub_view0[alpaka::Vec{0, 1, 0}]
            == view0[alpaka::Vec{offset_subview0.z(), offset_subview0.y() + 1, offset_subview0.x()}]);
        REQUIRE(
            sub_view0[extents_subview0 - alpaka::Vec{1, 1, 1}]
            == view0[offset_subview0 + extents_subview0 - alpaka::Vec{1, 1, 1}]);

        REQUIRE(offset_subview0.x() + extents_subview0.x() <= view0.getExtents().x());
        REQUIRE(offset_subview0.y() + extents_subview0.y() <= view0.getExtents().y());
        REQUIRE(offset_subview0.z() + extents_subview0.z() <= view0.getExtents().z());

        auto counter = offset_subview0;
        for(auto vec : alpaka::IdxRange{sub_view0.getExtents()})
        {
            REQUIRE_MESSAGE(sub_view0[vec] == counter.z() * 100 + counter.y() * 10 + counter.x(), "vec=" << vec);
            counter.x()++;
            if(counter.x() == offset_subview0.x() + extents_subview0.x())
            {
                counter.y()++;
                counter.x() = offset_subview0.x();
                if(counter.y() == offset_subview0.y() + extents_subview0.y())
                {
                    counter.z()++;
                    counter.y() = offset_subview0.y();
                }
            }
        }
    }

    SECTION("getSubView, zero extent in 3D stays empty and does not iterate")
    {
        // Zero in any dimension should still create a valid empty subview for host iteration.
        alpaka::Vec extents_subview0{z, 0, x};
        auto sub_view0 = view0.getSubView(extents_subview0);

        REQUIRE(sub_view0.getExtents() == extents_subview0);

        auto count = 0;
        for([[maybe_unused]] auto&& value : sub_view0)
        {
            alpaka::unused(value);
            ++count;
        }
        REQUIRE(count == 0);
    }

    SECTION("getSubView with offset writes through to the parent storage")
    {
        // Offset subviews must alias the parent data so writes land at shifted coordinates in the original view.
        alpaka::Vec offset_subview0{1, 2, 1};
        alpaka::Vec extents_subview0{1, 2, 2};
        auto sub_view0 = view0.getSubView(offset_subview0, extents_subview0);

        for(auto vec : alpaka::IdxRange{extents_subview0})
        {
            sub_view0[vec] = 900 + static_cast<int>(alpaka::linearize(extents_subview0, vec));
        }

        for(auto vec : alpaka::IdxRange{extents_subview0})
        {
            auto const parent_idx = offset_subview0 + vec;
            REQUIRE(view0[parent_idx] == 900 + static_cast<int>(alpaka::linearize(extents_subview0, vec)));
        }
    }

    SECTION("const getSubView with offset stays read-only and reads shifted values")
    {
        // `getSubView() const` should propagate constness to the returned view while still reading the shifted region.
        auto const& const_view0 = view0;
        alpaka::Vec offset_subview0{1, 2, 1};
        alpaka::Vec extents_subview0{2, 2, 3};
        auto sub_view0 = const_view0.getSubView(offset_subview0, extents_subview0);

        static_assert(std::is_const_v<std::remove_pointer_t<decltype(sub_view0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(sub_view0[alpaka::Vec{0, 0, 0}])>>);

        for(auto vec : alpaka::IdxRange{extents_subview0})
        {
            auto const parent_idx = offset_subview0 + vec;
            REQUIRE(sub_view0[vec] == view0[parent_idx]);
        }
    }
}

TEST_CASE("alpaka::View::getSubView keeps pitches, pointer, and alignment contracts", "[mem][view][SubDataStorage]")
{
    alignas(32) std::array<int, 2 * 3 * 4> storage{};
    for(std::size_t i = 0; i < storage.size(); ++i)
    {
        storage[i] = static_cast<int>(i);
    }

    auto view0 = alpaka::makeView(alpaka::api::host, storage.data(), alpaka::Vec{2, 3, 4}, alpaka::Alignment<32>{});

    SECTION("offset subviews preserve pitches and drop to plain alignment")
    {
        // A shifted origin can break stronger alignment guarantees, but the pitch layout must still match the parent.
        auto const offset_subview0 = view0.getSubView(alpaka::Vec{1, 1, 1}, alpaka::Vec{1, 2, 3});

        REQUIRE(offset_subview0.getPitches() == view0.getPitches());
        REQUIRE(offset_subview0.data() == &view0[alpaka::Vec{1, 1, 1}]);

        static_assert(std::is_same_v<decltype(offset_subview0.getAlignment()), alpaka::Alignment<>>);
    }

    SECTION("extent-only subviews keep the original pointer and alignment")
    {
        // Cropping only by extent must not move the pointer or weaken the parent alignment contract.
        auto const sub_view0 = view0.getSubView(alpaka::Vec{2, 2, 3});

        REQUIRE(sub_view0.data() == view0.data());
        REQUIRE(sub_view0.getPitches() == view0.getPitches());

        static_assert(std::is_same_v<decltype(sub_view0.getAlignment()), alpaka::Alignment<32>>);
    }
}

TEST_CASE(
    "alpaka::makeView(any) keeps host metadata and rebuilt subviews alias the source",
    "[mem][view][SubDataStorage]")
{
    auto const extents = alpaka::Vec{2u, 3u, 4u};
    auto buffer = alpaka::onHost::allocHost<int>(extents);
    for(auto idx : alpaka::IdxRange{extents})
    {
        buffer[idx] = static_cast<int>(alpaka::linearize(extents, idx));
    }

    auto rebuiltFromBuffer = alpaka::makeView(buffer);

    alignas(32) std::array<int, 2 * 3 * 4> storage{};
    for(std::size_t i = 0; i < storage.size(); ++i)
    {
        storage[i] = static_cast<int>(100 + i);
    }

    auto sourceView = alpaka::makeView(alpaka::api::host, storage.data(), extents, alpaka::Alignment<32>{});
    auto rebuiltFromView = alpaka::makeView(sourceView);

    SECTION("rebuilding from a host buffer preserves the parent metadata and alias")
    {
        // `makeView(buffer)` should hand back a plain host view with the same layout contract as the owning buffer.
        REQUIRE(rebuiltFromBuffer.getApi() == buffer.getApi());
        REQUIRE(rebuiltFromBuffer.getExtents() == buffer.getExtents());
        REQUIRE(rebuiltFromBuffer.getPitches() == buffer.getPitches());
        REQUIRE(rebuiltFromBuffer.data() == buffer.data());
        STATIC_REQUIRE(std::is_same_v<decltype(rebuiltFromBuffer.getAlignment()), decltype(buffer.getAlignment())>);

        auto const sampleIdx = alpaka::Vec{1u, 2u, 3u};
        rebuiltFromBuffer[sampleIdx] = 777;
        REQUIRE(buffer[sampleIdx] == 777);
        REQUIRE(&rebuiltFromBuffer[sampleIdx] == &buffer[sampleIdx]);
    }

    SECTION("rebuilding from a host view preserves layout and narrowed subviews still alias")
    {
        // `makeView(view)` must keep the original view metadata, and follow-on subviews should still reference the
        // same shifted region.
        REQUIRE(rebuiltFromView.getApi() == sourceView.getApi());
        REQUIRE(rebuiltFromView.getExtents() == sourceView.getExtents());
        REQUIRE(rebuiltFromView.getPitches() == sourceView.getPitches());
        REQUIRE(rebuiltFromView.data() == sourceView.data());
        STATIC_REQUIRE(std::is_same_v<decltype(rebuiltFromView.getAlignment()), alpaka::Alignment<32>>);

        auto const offset = alpaka::Vec{1u, 1u, 1u};
        auto const subExtents = alpaka::Vec{1u, 2u, 3u};
        auto subView = rebuiltFromView.getSubView(offset, subExtents);

        REQUIRE(subView.getExtents() == subExtents);
        REQUIRE(subView.getPitches() == rebuiltFromView.getPitches());
        REQUIRE(subView.data() == &rebuiltFromView[offset]);
        STATIC_REQUIRE(std::is_same_v<decltype(subView.getAlignment()), alpaka::Alignment<>>);

        auto const localIdx = alpaka::Vec{0u, 1u, 2u};
        subView[localIdx] = 2026;
        REQUIRE(rebuiltFromView[offset + localIdx] == 2026);
        REQUIRE(sourceView[offset + localIdx] == 2026);
    }
}

TEST_CASE(
    "alpaka::View::getSubView(BoundaryDirection) covers host lower upper core and const aliasing",
    "[mem][view][SubDataStorage]")
{
    alpaka::Vec totalExtents{4, 5, 6};
    auto buffer0 = alpaka::onHost::allocHost<int>(totalExtents);

    for(auto idx : alpaka::IdxRange{totalExtents})
    {
        buffer0[idx] = idx.z() * 100 + idx.y() * 10 + idx.x();
    }

    auto view0 = buffer0.getView();
    auto const lowerHalos = alpaka::Vec{1u, 2u, 1u};
    auto const upperHalos = alpaka::Vec{2u, 1u, 3u};
    using HaloVec = std::remove_cvref_t<decltype(lowerHalos)>;
    auto const makeBoundary = [&](auto const& boundaries)
    { return alpaka::BoundaryDirection<3, HaloVec, HaloVec>{boundaries, lowerHalos, upperHalos}; };

    SECTION("lower boundary keeps the origin region")
    {
        // LOWER should select the leading halo without shifting the origin in any chosen dimension.
        auto subView = view0.getSubView(makeBoundary(
            alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::LOWER}));

        auto const expectedExtents = alpaka::Vec{1, 2, 1};
        auto const expectedOffset = alpaka::Vec{0, static_cast<int>(lowerHalos.y()), 0};

        REQUIRE(subView.getExtents() == expectedExtents);

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            REQUIRE(subView[idx] == view0[expectedOffset + idx]);
        }
    }

    SECTION("upper boundary maps to the shifted tail region")
    {
        // UPPER should start from the tail extents minus the upper halo sizes.
        auto subView = view0.getSubView(makeBoundary(
            alpaka::Vec{alpaka::BoundaryType::UPPER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE}));

        auto const expectedExtents = alpaka::Vec{2, 1, 2};
        auto const expectedOffset = alpaka::Vec{
            totalExtents.z() - static_cast<int>(upperHalos.z()),
            totalExtents.y() - static_cast<int>(upperHalos.y()),
            static_cast<int>(lowerHalos.x())};

        REQUIRE(subView.getExtents() == expectedExtents);

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            REQUIRE(subView[idx] == view0[expectedOffset + idx]);
        }
    }

    SECTION("middle boundary returns the asymmetric interior")
    {
        // MIDDLE should crop by the lower and upper halos independently in each dimension.
        auto subView = view0.getSubView(alpaka::makeCoreBoundaryDirection<3>(lowerHalos, upperHalos));

        auto const expectedOffset = alpaka::Vec{
            static_cast<int>(lowerHalos.z()),
            static_cast<int>(lowerHalos.y()),
            static_cast<int>(lowerHalos.x())};
        auto const expectedExtents = alpaka::Vec{
            totalExtents.z() - static_cast<int>(lowerHalos.z()) - static_cast<int>(upperHalos.z()),
            totalExtents.y() - static_cast<int>(lowerHalos.y()) - static_cast<int>(upperHalos.y()),
            totalExtents.x() - static_cast<int>(lowerHalos.x()) - static_cast<int>(upperHalos.x())};

        REQUIRE(subView.getExtents() == expectedExtents);

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            REQUIRE(subView[idx] == view0[expectedOffset + idx]);
        }
    }

    SECTION("degenerate middle boundary with a consumed dimension stays empty")
    {
        // If halos consume a dimension completely, the middle region should be a valid zero-extent subview.
        auto subView
            = view0.getSubView(alpaka::makeCoreBoundaryDirection<3>(alpaka::Vec{1u, 2u, 3u}, alpaka::Vec{2u, 3u, 3u}));

        REQUIRE(subView.getExtents() == alpaka::Vec{1, 0, 0});

        auto count = 0;
        for([[maybe_unused]] auto&& value : subView)
        {
            alpaka::unused(value);
            ++count;
        }
        REQUIRE(count == 0);
    }

    SECTION("const boundary subviews stay read-only and still read shifted values")
    {
        // The boundary-direction overload must propagate constness just like the offset overload does.
        auto const& constView0 = view0;
        auto subView = constView0.getSubView(makeBoundary(
            alpaka::Vec{alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::UPPER}));

        auto const expectedOffset = alpaka::Vec{
            totalExtents.z() - static_cast<int>(upperHalos.z()),
            static_cast<int>(lowerHalos.y()),
            totalExtents.x() - static_cast<int>(upperHalos.x())};
        auto const expectedExtents = alpaka::Vec{2, 2, 3};

        REQUIRE(subView.getExtents() == expectedExtents);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView[alpaka::Vec{0, 0, 0}])>>);

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            REQUIRE(subView[idx] == view0[expectedOffset + idx]);
        }
    }

    SECTION("non-const boundary subviews preserve mutability and alias the parent")
    {
        // Non-const parents should keep writable element access so boundary views can update the original storage.
        auto subView = view0.getSubView(makeBoundary(
            alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE}));

        auto const expectedOffset
            = alpaka::Vec{0, totalExtents.y() - static_cast<int>(upperHalos.y()), static_cast<int>(lowerHalos.x())};
        auto const expectedExtents = alpaka::Vec{1, 1, 2};

        REQUIRE(subView.getExtents() == expectedExtents);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subView.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subView[alpaka::Vec{0, 0, 0}])>>);

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            subView[idx] = 700 + static_cast<int>(alpaka::linearize(expectedExtents, idx));
        }

        for(auto idx : alpaka::IdxRange{expectedExtents})
        {
            REQUIRE(view0[expectedOffset + idx] == 700 + static_cast<int>(alpaka::linearize(expectedExtents, idx)));
        }
    }
}
