/* Copyright 2023 Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/trait.hpp"

#include <cstdint>
#include <exception>
#include <type_traits>

namespace alpaka::onHost
{
    template<typename TView>
    struct ViewAccessOps
    {
    public:
        static consteval uint32_t dim()
        {
            return alpaka::trait::getDim_v<TView>;
        }

        /** pointer to data */
        decltype(auto) data()
        {
            return onHost::data(*static_cast<TView*>(this));
        }

        /** pointer to data */
        decltype(auto) data() const
        {
            return onHost::data(*static_cast<TView const*>(this));
        }

        auto getMdSpan() const
        {
            return alpaka::MdSpan{
                data(),
                getExtents(*static_cast<TView const*>(this)),
                getPitches(*static_cast<TView const*>(this))};
        }

        auto getMdSpan()
        {
            return alpaka::MdSpan{
                data(),
                getExtents(*static_cast<TView const*>(this)),
                getPitches(*static_cast<TView const*>(this))};
        }

        /** access 1-dimensional data with a scalar index
         *
         * @{
         */
        decltype(auto) operator[](std::integral auto idx) const requires(dim() == 1u)
        {
            return data()[idx];
        }

        decltype(auto) operator[](std::integral auto idx) requires(dim() == 1u)
        {
            return data()[idx];
        }

        /** @} */

        /** access M-dimensional data with a vector index
         *
         * @{
         */
        decltype(auto) operator[](alpaka::concepts::Vector auto idx) const
        {
            return getMdSpan()[idx];
        }

        decltype(auto) operator[](alpaka::concepts::Vector auto idx)
        {
            return getMdSpan()[idx];
        }

        /** @} */
    };
} // namespace alpaka::onHost
