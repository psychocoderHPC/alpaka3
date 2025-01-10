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
    private:
        using value_type = alpaka::trait::GetValueType<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using size_type = alpaka::trait::GetSizeType_t<TView>;

    public:
        static consteval uint32_t dim()
        {
            return alpaka::trait::GetExtentType_t<TView>::dim();
        }

        /** pointer to data */
        auto data() -> pointer
        {
            return onHost::data(*static_cast<TView*>(this));
        }

        /** pointer to data */
        auto data() const -> const_pointer
        {
            return onHost::data(*static_cast<TView const*>(this));
        }

        auto operator*() -> reference requires(dim() <= 1u)
        {
            return *data();
        }

        auto operator*() const -> const_reference requires(dim() <= 1u)
        {
            return *data();
        }

        auto operator->() -> pointer requires(dim() == 0u)
        {
            return data();
        }

        auto operator->() const -> const_pointer requires(dim() == 0u)
        {
            return data();
        }

        /** access 1-dimensional data with a scalar index
         *
         * @{
         */
        auto operator[](std::integral auto idx) const -> const_reference requires(dim() == 1u)
        {
            return data()[idx];
        }

        auto operator[](std::integral auto idx) -> reference requires(dim() == 1u)
        {
            return data()[idx];
        }

        /** @} */

        auto getMdSpan() const
        {
            auto* ptr = data();
            auto& view = *static_cast<TView const*>(this);
            return alpaka::MdSpan{
                ptr,
                alpaka::trait::GetExtentType_t<TView>::all(0),
                onHost::getExtents(view),
                onHost::getPitches(view)};
        }

        /** access M-dimensional data with a vector index
         *
         * @{
         */
        auto operator[](alpaka::concepts::Vector auto idx) const -> const_reference
        {
            return getMdSpan()[idx];
        }

        auto operator[](alpaka::concepts::Vector auto idx) -> reference
        {
            return getMdSpan()[idx];
        }

        /** @} */
    };
} // namespace alpaka::onHost
