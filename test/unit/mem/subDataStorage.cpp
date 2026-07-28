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
        constexpr int subSize = 8;
        auto subView0 = view0.getSubView(subSize);
        REQUIRE(subView0.getExtents() == alpaka::Vec{subSize});
        for(auto i = 0; i < subSize; ++i)
        {
            REQUIRE_MESSAGE(subView0[i] == i, "i=" << i);
        }
    }

    SECTION("getSubView, single integer, define offset and extent")
    {
        constexpr int offset = 2;
        constexpr int end = 7;
        auto subView0 = view0.getSubView(offset, end);

        REQUIRE(subView0.getExtents() == alpaka::Vec{end});
        for(auto i = 0; i < subView0.getExtents()[0]; ++i)
        {
            REQUIRE_MESSAGE(subView0[i] == i + offset, "i=" << i);
        }
    }

    SECTION("getSubView, zero extent in 1D stays empty and does not iterate")
    {
        // Zero-sized 1D subviews should preserve the requested extent and produce an empty host iteration range.
        auto subView0 = view0.getSubView(0);

        REQUIRE(subView0.getExtents() == alpaka::Vec{0});

        auto count = 0;
        for([[maybe_unused]] auto&& value : subView0)
            ++count;

        REQUIRE(count == 0);
    }
}

TEST_CASE("3D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int z = 3;
    constexpr int y = 5;
    constexpr int x = 4;
    alpaka::Vec totalExtents{z, y, x};
    REQUIRE(totalExtents.z() == z);
    REQUIRE(totalExtents.y() == y);
    REQUIRE(totalExtents.x() == x);

    auto buffer0 = alpaka::onHost::allocHost<int>(totalExtents);

    for(auto vec : alpaka::IdxRange{totalExtents})
    {
        buffer0[vec] = vec.z() * 100 + vec.y() * 10 + vec.x();
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, define extent")
    {
        alpaka::Vec extentsSubview0{z - 1, y - 1, x - 1};
        auto subView0 = view0.getSubView(extentsSubview0);
        REQUIRE(subView0.getApi() == view0.getApi());

        STATIC_REQUIRE(subView0.dim() == extentsSubview0.dim());
        REQUIRE(subView0.getExtents() == extentsSubview0);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            REQUIRE_MESSAGE(subView0[vec] == vec.z() * 100 + vec.y() * 10 + vec.x(), "vec=" << vec);
        }
    }

    SECTION("getSubView, define offset and extent")
    {
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{z - 1, y - 3, x - 1};

        auto subView0 = view0.getSubView(offsetSubview0, extentsSubview0);
        REQUIRE(subView0.getApi() == view0.getApi());

        STATIC_REQUIRE(subView0.dim() == extentsSubview0.dim());
        REQUIRE(subView0.getExtents() == extentsSubview0);

        REQUIRE(subView0[alpaka::Vec{0, 0, 0}] == view0[offsetSubview0]);
        REQUIRE(
            subView0[alpaka::Vec{0, 1, 0}]
            == view0[alpaka::Vec{offsetSubview0.z(), offsetSubview0.y() + 1, offsetSubview0.x()}]);
        REQUIRE(
            subView0[extentsSubview0 - alpaka::Vec{1, 1, 1}]
            == view0[offsetSubview0 + extentsSubview0 - alpaka::Vec{1, 1, 1}]);

        REQUIRE(offsetSubview0.x() + extentsSubview0.x() <= view0.getExtents().x());
        REQUIRE(offsetSubview0.y() + extentsSubview0.y() <= view0.getExtents().y());
        REQUIRE(offsetSubview0.z() + extentsSubview0.z() <= view0.getExtents().z());

        auto counter = offsetSubview0;
        for(auto vec : alpaka::IdxRange{subView0.getExtents()})
        {
            REQUIRE_MESSAGE(subView0[vec] == counter.z() * 100 + counter.y() * 10 + counter.x(), "vec=" << vec);
            counter.x()++;
            if(counter.x() == offsetSubview0.x() + extentsSubview0.x())
            {
                counter.y()++;
                counter.x() = offsetSubview0.x();
                if(counter.y() == offsetSubview0.y() + extentsSubview0.y())
                {
                    counter.z()++;
                    counter.y() = offsetSubview0.y();
                }
            }
        }
    }

    SECTION("getSubView, zero extent in 3D stays empty and does not iterate")
    {
        // Zero in any dimension should still create a valid empty subview for host iteration.
        alpaka::Vec extentsSubview0{z, 0, x};
        auto subView0 = view0.getSubView(extentsSubview0);

        REQUIRE(subView0.getExtents() == extentsSubview0);

        auto count = 0;
        for([[maybe_unused]] auto&& value : subView0)
            ++count;

        REQUIRE(count == 0);
    }

    SECTION("getSubView with offset writes through to the parent storage")
    {
        // Offset subviews must alias the parent data so writes land at shifted coordinates in the original view.
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{1, 2, 2};
        auto subView0 = view0.getSubView(offsetSubview0, extentsSubview0);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            subView0[vec] = 900 + static_cast<int>(alpaka::linearize(extentsSubview0, vec));
        }

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            auto const parentIdx = offsetSubview0 + vec;
            REQUIRE(view0[parentIdx] == 900 + static_cast<int>(alpaka::linearize(extentsSubview0, vec)));
        }
    }

    SECTION("const getSubView with offset stays read-only and reads shifted values")
    {
        // `getSubView() const` should propagate constness to the returned view while still reading the shifted region.
        auto const& constView0 = view0;
        alpaka::Vec offsetSubview0{1, 2, 1};
        alpaka::Vec extentsSubview0{2, 2, 3};
        auto subView0 = constView0.getSubView(offsetSubview0, extentsSubview0);

        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[alpaka::Vec{0, 0, 0}])>>);

        for(auto vec : alpaka::IdxRange{extentsSubview0})
        {
            auto const parentIdx = offsetSubview0 + vec;
            REQUIRE(subView0[vec] == view0[parentIdx]);
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
        auto const offsetSubview0 = view0.getSubView(alpaka::Vec{1, 1, 1}, alpaka::Vec{1, 2, 3});

        REQUIRE(offsetSubview0.getPitches() == view0.getPitches());
        REQUIRE(offsetSubview0.data() == &view0[alpaka::Vec{1, 1, 1}]);

        STATIC_REQUIRE(offsetSubview0.getAlignment().get<int>() == alpaka::Alignment<>::get<int>());
    }

    SECTION("extent-only subviews keep the original pointer and alignment")
    {
        // Cropping only by extent must not move the pointer or weaken the parent alignment contract.
        auto const subView0 = view0.getSubView(alpaka::Vec{2, 2, 3});

        REQUIRE(subView0.data() == view0.data());
        REQUIRE(subView0.getPitches() == view0.getPitches());

        STATIC_REQUIRE(subView0.getAlignment().get<int>() == alpaka::Alignment<32>::get<int>());
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
    {
        return alpaka::BoundaryDirection<3, HaloVec, HaloVec>{
            boundaries,
            lowerHalos,
            upperHalos};
    };

    SECTION("lower boundary keeps the origin region")
    {
        // LOWER should select the leading halo without shifting the origin in any chosen dimension.
        auto subView = view0.getSubView(
            makeBoundary(alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::LOWER}));

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
        auto subView = view0.getSubView(
            makeBoundary(alpaka::Vec{alpaka::BoundaryType::UPPER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE}));

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
        auto subView = view0.getSubView(alpaka::makeCoreBoundaryDirection<3>(alpaka::Vec{1u, 2u, 3u}, alpaka::Vec{2u, 3u, 3u}));

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
        auto subView = constView0.getSubView(
            makeBoundary(alpaka::Vec{alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::UPPER}));

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
        auto subView = view0.getSubView(
            makeBoundary(alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE}));

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
