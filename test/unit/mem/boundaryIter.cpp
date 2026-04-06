/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <type_traits>
#include <vector>

using namespace alpaka;

TEST_CASE("BoundaryIter host coverage", "[mem][BoundaryIter][iterator]")
{
    SECTION("containers enumerate every direction once and stop at the OOB sentinel")
    {
        // The iterator should walk all 3^dim boundary states in lexicographic order without duplicates.
        auto const lowerHalos = alpaka::Vec{1u, 2u, 3u};
        auto const upperHalos = alpaka::Vec{4u, 5u, 6u};
        auto const directions = alpaka::BoundaryDirectionsContainer{lowerHalos, upperHalos};

        using Direction = std::remove_cvref_t<decltype(*directions.begin())>;
        std::vector<Direction> visited;
        for(auto it = directions.begin(); it != directions.end(); ++it)
        {
            auto const direction = *it;
            REQUIRE(direction.lowerHaloSize == lowerHalos);
            REQUIRE(direction.upperHaloSize == upperHalos);

            bool duplicate = false;
            for(auto const& previous : visited)
            {
                if(previous == direction)
                {
                    duplicate = true;
                    break;
                }
            }
            REQUIRE_FALSE(duplicate);
            visited.push_back(direction);
        }

        REQUIRE(visited.size() == directions.length());
        REQUIRE(directions.length() == 27u);
        REQUIRE(visited.front().data[0] == alpaka::BoundaryType::LOWER);
        REQUIRE(visited.front().data[1] == alpaka::BoundaryType::LOWER);
        REQUIRE(visited.front().data[2] == alpaka::BoundaryType::LOWER);
        REQUIRE(directions.begin() != directions.end());

        auto iter = directions.begin();
        for(uint32_t i = 0; i < directions.length(); ++i)
            ++iter;
        REQUIRE(iter == directions.end());
        auto const endIter = directions.end();
        REQUIRE((*endIter).data[0] == alpaka::BoundaryType::OOB);
        REQUIRE((*endIter).data[1] == alpaka::BoundaryType::OOB);
        REQUIRE((*endIter).data[2] == alpaka::BoundaryType::OOB);
    }

    SECTION("classification helpers match explicit directions")
    {
        // Explicit boundary patterns should map to the expected geometric classification.
        auto const lowerHalos = alpaka::Vec{1u, 1u, 1u};
        auto const upperHalos = alpaka::Vec{2u, 2u, 2u};
        using HaloVec = std::remove_cvref_t<decltype(lowerHalos)>;

        auto const vertex = alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{
            alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::UPPER, alpaka::BoundaryType::LOWER},
            lowerHalos,
            upperHalos};
        auto const edge = alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{
            alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::UPPER},
            lowerHalos,
            upperHalos};
        auto const face = alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{
            alpaka::Vec{alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::LOWER},
            lowerHalos,
            upperHalos};
        auto const cell = alpaka::makeCoreBoundaryDirection<3u>(lowerHalos, upperHalos);

        REQUIRE(vertex.boundaryDimensionality() == 0u);
        REQUIRE(vertex.isVertex());
        REQUIRE_FALSE(vertex.isEdge());
        REQUIRE_FALSE(vertex.isFace());
        REQUIRE_FALSE(vertex.isCell());
        REQUIRE_FALSE(vertex.isInterior());

        REQUIRE(edge.boundaryDimensionality() == 1u);
        REQUIRE(edge.isEdge());
        REQUIRE_FALSE(edge.isVertex());
        REQUIRE_FALSE(edge.isFace());
        REQUIRE_FALSE(edge.isCell());
        REQUIRE_FALSE(edge.isInterior());

        REQUIRE(face.boundaryDimensionality() == 2u);
        REQUIRE(face.isFace());
        REQUIRE_FALSE(face.isVertex());
        REQUIRE_FALSE(face.isEdge());
        REQUIRE_FALSE(face.isCell());
        REQUIRE_FALSE(face.isInterior());

        REQUIRE(cell.boundaryDimensionality() == 3u);
        REQUIRE(cell.isCell());
        REQUIRE(cell.isInterior());
        REQUIRE_FALSE(cell.isVertex());
        REQUIRE_FALSE(cell.isEdge());
        REQUIRE_FALSE(cell.isFace());
    }

    SECTION("the OOB sentinel stays unclassified")
    {
        // The end sentinel is not a real boundary and should not masquerade as a vertex or interior cell.
        auto const directions = alpaka::makeBoundaryDirIterator<2u>();
        auto const endIter = directions.end();
        auto const oob = *endIter;

        REQUIRE(oob.isOutOfBounds());
        REQUIRE(oob.boundaryDimensionality() == 0u);
        REQUIRE_FALSE(oob.isVertex());
        REQUIRE_FALSE(oob.isEdge());
        REQUIRE_FALSE(oob.isFace());
        REQUIRE_FALSE(oob.isCell());
        REQUIRE_FALSE(oob.isInterior());
    }

    SECTION("core direction helpers always produce a middle direction")
    {
        // All makeCoreBoundaryDirection overloads should agree on the middle state and requested halo sizes.
        auto const lowerHalos = alpaka::Vec{2u, 3u};
        auto const upperHalos = alpaka::Vec{4u, 5u};
        auto const symmetricHalos = alpaka::Vec{7u, 8u};

        auto const explicitCore = alpaka::makeCoreBoundaryDirection<2u>(lowerHalos, upperHalos);
        auto const symmetricCore = alpaka::makeCoreBoundaryDirection<2u>(symmetricHalos);
        constexpr auto defaultCore = alpaka::makeCoreBoundaryDirection<2u>();

        REQUIRE(explicitCore.data[0] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(explicitCore.data[1] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(explicitCore.lowerHaloSize == lowerHalos);
        REQUIRE(explicitCore.upperHaloSize == upperHalos);

        REQUIRE(symmetricCore.data[0] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(symmetricCore.data[1] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(symmetricCore.lowerHaloSize == symmetricHalos);
        REQUIRE(symmetricCore.upperHaloSize == symmetricHalos);

        REQUIRE(defaultCore.data[0] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(defaultCore.data[1] == alpaka::BoundaryType::MIDDLE);
        REQUIRE(defaultCore.lowerHaloSize == alpaka::fillCVec<uint32_t, 2u, 1u>());
        REQUIRE(defaultCore.upperHaloSize == alpaka::fillCVec<uint32_t, 2u, 1u>());
    }

    SECTION("factory overloads preserve halos and infer dimensions from inputs")
    {
        // The convenience factories should keep the supplied halo values whether they come from vectors or a view.
        auto const symmetricHalos = alpaka::Vec{3u, 4u};
        auto const asymmetricLowerHalos = alpaka::Vec{1u, 2u, 3u};
        auto const asymmetricUpperHalos = alpaka::Vec{4u, 5u, 6u};
        int storage[2 * 3 * 4]{};
        auto const view = alpaka::makeView(alpaka::api::host, storage, alpaka::Vec{2u, 3u, 4u});

        auto const symmetricDirections = alpaka::makeBoundaryDirIterator(symmetricHalos);
        auto const asymmetricDirections = alpaka::makeBoundaryDirIterator(asymmetricLowerHalos, asymmetricUpperHalos);
        auto const defaultDirections = alpaka::makeBoundaryDirIterator(view);

        STATIC_REQUIRE(std::remove_cvref_t<decltype(symmetricDirections)>::dim() == 2u);
        STATIC_REQUIRE(std::remove_cvref_t<decltype(asymmetricDirections)>::dim() == 3u);
        STATIC_REQUIRE(std::remove_cvref_t<decltype(defaultDirections)>::dim() == 3u);

        REQUIRE((*symmetricDirections.begin()).lowerHaloSize == symmetricHalos);
        REQUIRE((*symmetricDirections.begin()).upperHaloSize == symmetricHalos);
        REQUIRE((*asymmetricDirections.begin()).lowerHaloSize == asymmetricLowerHalos);
        REQUIRE((*asymmetricDirections.begin()).upperHaloSize == asymmetricUpperHalos);
        REQUIRE((*defaultDirections.begin()).lowerHaloSize == alpaka::fillCVec<uint32_t, 3u, 1u>());
        REQUIRE((*defaultDirections.begin()).upperHaloSize == alpaka::fillCVec<uint32_t, 3u, 1u>());
    }

    SECTION("stream output stays human-readable for representative directions")
    {
        // Stable text output keeps host-side diagnostics and examples easy to understand.
        auto const lowerHalos = alpaka::Vec{1u, 1u, 1u};
        auto const upperHalos = alpaka::Vec{2u, 2u, 2u};
        using HaloVec = std::remove_cvref_t<decltype(lowerHalos)>;
        auto const edge = alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{
            alpaka::Vec{alpaka::BoundaryType::LOWER, alpaka::BoundaryType::MIDDLE, alpaka::BoundaryType::UPPER},
            lowerHalos,
            upperHalos};
        auto const oob = alpaka::BoundaryDirection<3u, HaloVec, HaloVec>{
            alpaka::fillCVec<alpaka::BoundaryType, 3u, alpaka::BoundaryType::OOB>(),
            lowerHalos,
            upperHalos};

        std::ostringstream edgeText;
        edgeText << edge;
        REQUIRE(edgeText.str() == "v-^ (edge)   ");

        std::ostringstream oobText;
        oobText << oob;
        REQUIRE(oobText.str() == "xxx");
    }
}
