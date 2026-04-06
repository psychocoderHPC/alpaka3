/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

TEST_CASE("IdxRange::begin() and end()", "[mem][IdxRange][iterator]")
{
    SECTION("only end extents")
    {
        constexpr int y = 4;
        for(int counter = 0; auto vec : alpaka::IdxRange(alpaka::Vec{y, 3}))
        {
            REQUIRE(vec == alpaka::Vec{counter / (y - 1), counter % (y - 1)});
            counter++;
        }
    }

    SECTION("begin and end extents")
    {
        constexpr int y = 4;
        constexpr int x = 3;
        constexpr int y_begin = 2;
        constexpr int x_begin = 1;
        for(auto expected_vec = alpaka::Vec{y_begin, x_begin};
            auto vec : alpaka::IdxRange(alpaka::Vec{y_begin, x_begin}, alpaka::Vec{y, x}))
        {
            REQUIRE(vec == expected_vec);
            expected_vec.x()++;
            if(expected_vec.x() == x)
            {
                expected_vec.y()++;
                expected_vec.x() = x_begin;
            }
        }
    }

    SECTION("begin and end extents and stride")
    {
        constexpr int y = 15;
        constexpr int x = 12;
        constexpr int y_begin = 2;
        constexpr int x_begin = 0;
        constexpr int y_stride = 4;
        constexpr int x_stride = 3;
        for(auto expected_vec = alpaka::Vec{y_begin, x_begin};
            auto vec :
            alpaka::IdxRange(alpaka::Vec{y_begin, x_begin}, alpaka::Vec{y, x}, alpaka::Vec{y_stride, x_stride}))
        {
            REQUIRE(vec == expected_vec);
            expected_vec.x() += x_stride;
            if(expected_vec.x() >= x)
            {
                expected_vec.y() += y_stride;
                expected_vec.x() = x_begin;
            }
        }
    }
}

TEST_CASE("IdxRange edge cases and transforms", "[mem][IdxRange][iterator]")
{
    auto collectVisited = []<typename T_Range>(T_Range const& range)
    {
        using Idx = std::decay_t<decltype(*range.begin())>;
        std::vector<Idx> visited;
        for(auto const idx : range)
            visited.push_back(idx);
        return visited;
    };

    SECTION("empty range yields no elements")
    {
        // A range with identical begin/end must be empty rather than yielding a sentinel element.
        auto const range = alpaka::IdxRange(alpaka::Vec{2, 3}, alpaka::Vec{2, 3});

        REQUIRE(range.begin() == range.end());
        REQUIRE(collectVisited(range).empty());
        REQUIRE(range.distance() == alpaka::Vec{0, 0});
        REQUIRE(decltype(range)::dim() == 2u);
    }

    SECTION("1D stride stops before overshooting end")
    {
        // Non-divisible spans must stop at the last valid point instead of stepping past the end bound.
        auto const range = alpaka::IdxRange(alpaka::Vec{1}, alpaka::Vec{10}, alpaka::Vec{4});

        REQUIRE(collectVisited(range) == std::vector{alpaka::Vec{1}, alpaka::Vec{5}, alpaka::Vec{9}});
        REQUIRE(range.distance() == alpaka::Vec{9});
        REQUIRE(decltype(range)::dim() == 1u);
    }

    SECTION("3D traversal keeps x fastest and carries into y then z")
    {
        // This locks down the multidimensional visitation order for host-side iteration.
        auto const range = alpaka::IdxRange(alpaka::Vec{1, 2, 3}, alpaka::Vec{3, 4, 5});

        REQUIRE(
            collectVisited(range)
            == std::vector{
                alpaka::Vec{1, 2, 3},
                alpaka::Vec{1, 2, 4},
                alpaka::Vec{1, 3, 3},
                alpaka::Vec{1, 3, 4},
                alpaka::Vec{2, 2, 3},
                alpaka::Vec{2, 2, 4},
                alpaka::Vec{2, 3, 3},
                alpaka::Vec{2, 3, 4}});
        REQUIRE(range.distance() == alpaka::Vec{2, 2, 2});
        REQUIRE(decltype(range)::dim() == 3u);
    }

    SECTION("translations preserve stride and shift every visited point")
    {
        // Shifting a range should translate begin/end and all visited indices without altering stride.
        auto const original = alpaka::IdxRange(alpaka::Vec{2, 1}, alpaka::Vec{8, 7}, alpaka::Vec{3, 2});
        auto const shiftedRight = original >> alpaka::Vec{4, 5};
        auto const shiftedLeft = original << alpaka::Vec{1, 1};

        REQUIRE(original.m_stride == shiftedRight.m_stride);
        REQUIRE(original.m_stride == shiftedLeft.m_stride);
        REQUIRE(original.distance() == shiftedRight.distance());
        REQUIRE(original.distance() == shiftedLeft.distance());
        REQUIRE(decltype(original)::dim() == decltype(shiftedRight)::dim());
        REQUIRE(decltype(original)::dim() == decltype(shiftedLeft)::dim());
        REQUIRE(
            collectVisited(shiftedRight)
            == std::vector{
                alpaka::Vec{6, 6},
                alpaka::Vec{6, 8},
                alpaka::Vec{6, 10},
                alpaka::Vec{9, 6},
                alpaka::Vec{9, 8},
                alpaka::Vec{9, 10}});
        REQUIRE(
            collectVisited(shiftedLeft)
            == std::vector{
                alpaka::Vec{1, 0},
                alpaka::Vec{1, 2},
                alpaka::Vec{1, 4},
                alpaka::Vec{4, 0},
                alpaka::Vec{4, 2},
                alpaka::Vec{4, 4}});
    }

    SECTION("stride scaling changes visited points but not begin/end")
    {
        // Scaling the stride must sparsify iteration while keeping the same logical bounds.
        auto const original = alpaka::IdxRange(alpaka::Vec{1, 2}, alpaka::Vec{8, 11}, alpaka::Vec{2, 3});
        auto const scaled = original % 2;

        REQUIRE(scaled.m_begin == original.m_begin);
        REQUIRE(scaled.m_end == original.m_end);
        REQUIRE(scaled.m_stride == alpaka::Vec{4, 6});
        REQUIRE(scaled.distance() == original.distance());
        REQUIRE(decltype(original)::dim() == decltype(scaled)::dim());
        REQUIRE(
            collectVisited(original)
            == std::vector{
                alpaka::Vec{1, 2},
                alpaka::Vec{1, 5},
                alpaka::Vec{1, 8},
                alpaka::Vec{3, 2},
                alpaka::Vec{3, 5},
                alpaka::Vec{3, 8},
                alpaka::Vec{5, 2},
                alpaka::Vec{5, 5},
                alpaka::Vec{5, 8},
                alpaka::Vec{7, 2},
                alpaka::Vec{7, 5},
                alpaka::Vec{7, 8}});
        REQUIRE(
            collectVisited(scaled)
            == std::vector{alpaka::Vec{1, 2}, alpaka::Vec{1, 8}, alpaka::Vec{5, 2}, alpaka::Vec{5, 8}});
    }
}

TEST_CASE("IdxRange boundary slicing keeps bounds, emptiness, and stride", "[mem][IdxRange][boundary]")
{
    auto collectVisited = []<typename T_Range>(T_Range const& range)
    {
        using Idx = std::decay_t<decltype(*range.begin())>;
        std::vector<Idx> visited;
        for(auto const idx : range)
            visited.push_back(idx);
        return visited;
    };

    SECTION("lower upper and middle slices keep the expected subrange values")
    {
        // Boundary slices should map directly onto the parent begin/end coordinates in each selected dimension.
        auto const parent = alpaka::IdxRange(alpaka::Vec{2, 5, 7}, alpaka::Vec{11, 17, 21}, alpaka::Vec{3, 4, 5});
        auto const lowerHalos = alpaka::Vec{2u, 3u, 4u};
        auto const upperHalos = alpaka::Vec{1u, 5u, 2u};
        using HaloVec = std::remove_cvref_t<decltype(lowerHalos)>;
        auto const makeBoundary = [&](auto const& data)
        { return alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{data, lowerHalos, upperHalos}; };

        // LOWER should keep the parent origin in selected dimensions while the core stays cropped by both halos.
        auto const lower = alpaka::makeDirectionSubRange(
            parent,
            makeBoundary(
                alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::LOWER}));
        REQUIRE(lower.m_begin == alpaka::Vec{2, 8, 7});
        REQUIRE(lower.m_end == alpaka::Vec{4, 12, 11});
        REQUIRE(lower.m_stride == parent.m_stride);
        REQUIRE(collectVisited(lower) == std::vector{alpaka::Vec{2, 8, 7}});

        // UPPER should anchor the selected dimensions against the parent tail without disturbing stride.
        auto const upper = alpaka::makeDirectionSubRange(
            parent,
            makeBoundary(
                alpaka::Vec{alpaka::BoundaryType::UPPER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::MIDDLE}));
        REQUIRE(upper.m_begin == alpaka::Vec{10, 12, 11});
        REQUIRE(upper.m_end == alpaka::Vec{11, 17, 19});
        REQUIRE(upper.m_stride == parent.m_stride);
        REQUIRE(
            collectVisited(upper)
            == std::vector{
                alpaka::Vec{10, 12, 11},
                alpaka::Vec{10, 12, 16},
                alpaka::Vec{10, 16, 11},
                alpaka::Vec{10, 16, 16}});

        // MIDDLE should crop by asymmetric halos in every dimension and preserve the sparse parent walk.
        auto const middle
            = alpaka::makeDirectionSubRange(parent, alpaka::makeCoreBoundaryDirection<3u>(lowerHalos, upperHalos));
        REQUIRE(middle.m_begin == alpaka::Vec{4, 8, 11});
        REQUIRE(middle.m_end == alpaka::Vec{10, 12, 19});
        REQUIRE(middle.m_stride == parent.m_stride);
        REQUIRE(
            collectVisited(middle)
            == std::vector{
                alpaka::Vec{4, 8, 11},
                alpaka::Vec{4, 8, 16},
                alpaka::Vec{7, 8, 11},
                alpaka::Vec{7, 8, 16}});
    }

    SECTION("empty core stays empty when halos consume a dimension")
    {
        // If halos eat the full span of one dimension, the middle slice must become empty instead of yielding garbage.
        auto const parent = alpaka::IdxRange(alpaka::Vec{1, 3, 5}, alpaka::Vec{7, 9, 11}, alpaka::Vec{2, 3, 2});
        auto const middle = alpaka::makeDirectionSubRange(
            parent,
            alpaka::makeCoreBoundaryDirection<3u>(alpaka::Vec{1u, 2u, 3u}, alpaka::Vec{1u, 4u, 1u}));

        REQUIRE(middle.m_begin == alpaka::Vec{2, 5, 8});
        REQUIRE(middle.m_end == alpaka::Vec{6, 5, 10});
        REQUIRE(middle.m_stride == parent.m_stride);
        REQUIRE(middle.begin() == middle.end());
        REQUIRE(collectVisited(middle).empty());
    }

    SECTION("signed ranges keep their signed coordinate type")
    {
        // Signed parent coordinates must survive boundary slicing so negative starts stay representable.
        auto const parent = alpaka::IdxRange(alpaka::Vec{-5, -4}, alpaka::Vec{6, 9}, alpaka::Vec{4, 3});
        auto const lowerHalos = alpaka::Vec{2u, 3u};
        auto const upperHalos = alpaka::Vec{1u, 2u};
        auto const middle
            = alpaka::makeDirectionSubRange(parent, alpaka::makeCoreBoundaryDirection<2u>(lowerHalos, upperHalos));

        static_assert(std::is_same_v<typename decltype(middle)::IdxType, int>);
        REQUIRE(middle.m_begin == alpaka::Vec{-3, -1});
        REQUIRE(middle.m_end == alpaka::Vec{5, 7});
        REQUIRE(middle.m_stride == parent.m_stride);
        REQUIRE(
            collectVisited(middle)
            == std::vector{
                alpaka::Vec{-3, -1},
                alpaka::Vec{-3, 2},
                alpaka::Vec{-3, 5},
                alpaka::Vec{1, -1},
                alpaka::Vec{1, 2},
                alpaka::Vec{1, 5}});
    }
}
