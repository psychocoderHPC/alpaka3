/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/trait.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    /** memory owning view
     *
     * This view is only holding a handle to the real data, copying the view is cheap.
     * Owning mean that it is guaranteed that the lifetime of the data is defined by the lifetime of the view.
     */
    template<typename T_Datahandle, typename T_Extents, typename T_Offsets = T_Extents>
    struct View
    {
    public:
        using value_type = alpaka::trait::GetValueType_t<typename T_Datahandle::element_type>;
        using size_type = alpaka::trait::GetSizeType_t<typename T_Datahandle::element_type>;

        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;

        /** creates a view
         *
         * @param data handle to the physical data
         * @param extents M-dimensional extents in elements of the view. Must be <= number of elements in the data
         * handle.
         *
         * @{
         */
        View(T_Datahandle data, T_Extents const& extents)
            : m_data(std::move(data))
            , m_extents(extents)
            , m_offsets(T_Offsets::all(0))
        {
        }

        /**
         * @param offset M-dimensional offset in elements with respect to the pointer of data handle.
         */
        View(T_Datahandle data, T_Offsets const& offset, T_Extents const& extents)
            : m_data(std::move(data))
            , m_extents(extents)
            , m_offsets(offset)
        {
        }

        /** @} */

        /** creates a view
         *
         * Extents will be derived from the data handle.
         *
         * @param data handle to the physical data
         */
        View(T_Datahandle data) : m_data(std::move(data)), m_extents(m_data->m_extents), m_offsets(T_Offsets::all(0))
        {
        }

        View(View const&) = default;
        View(View&&) = default;

        /** shallow copy the handle
         *
         * It is not copying the data the view is pointing to.
         */
        View& operator=(View const&) = default;

        static consteval uint32_t dim()
        {
            return T_Extents::dim();
        }

        /** get the number of elements for each dimension */
        auto getExtents() const
        {
            return m_extents;
        }

        /** Get the distance in bytes to move to the next element in the corresponding dimension. */
        auto getPitches() const
        {
            return alpaka::onHost::getPitches(m_data);
        }

        /** pointer to data */
        pointer data()
        {
            return getMdSpan().data();
        }

        /** pointer to data */
        const_pointer data() const
        {
            return getMdSpan().data();
        }

        auto getMdSpan() const
        {
            auto* ptr = onHost::data(m_data);
            return alpaka::MdSpan{ptr, m_offsets, m_extents, getPitches()};
        }

        /** access 1-dimensional data with a scalar index
         *
         * @{
         */
        const_reference operator[](std::integral auto idx) const requires(dim() == 1u)
        {
            return data()[idx];
        }

        reference operator[](std::integral auto idx) requires(dim() == 1u)
        {
            return data()[idx];
        }

        /** @} */

        /** access M-dimensional data with a vector index
         *
         * @{
         */
        const_reference operator[](alpaka::concepts::Vector auto idx) const
        {
            return getMdSpan()[idx];
        }

        reference operator[](alpaka::concepts::Vector auto idx)
        {
            return getMdSpan()[idx];
        }

        /** @} */

    private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        friend struct internal::Memcpy;

        T_Datahandle m_data;
        T_Extents m_extents;
        T_Extents m_offsets;

        friend struct internal::Data;
        friend struct alpaka::internal::GetApi;
    };

    template<typename T_Datahandle>
    ALPAKA_FN_HOST_ACC View(T_Datahandle)
        -> View<T_Datahandle, alpaka::trait::GetExtentType_t<typename T_Datahandle::element_type>>;

} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename... T_Args>
    struct GetApi::Op<onHost::View<T_Args...>>
    {
        decltype(auto) operator()(auto&& buffer) const
        {
            return onHost::getApi(buffer.m_data);
        }
    };
} // namespace alpaka::internal

namespace alpaka::trait
{

    template<typename T_Datahandle, typename... T_Args>
    struct GetValueType<onHost::View<T_Datahandle, T_Args...>>
    {
        using type = trait::GetValueType_t<typename T_Datahandle::element_type>;
    };

    template<typename T_Datahandle, typename... T_Args>
    struct GetExtentType<onHost::View<T_Datahandle, T_Args...>>
    {
        using type = trait::GetExtentType_t<typename T_Datahandle::element_type>;
    };

    template<typename T_Datahandle, typename... T_Args>
    struct GetSizeType<onHost::View<T_Datahandle, T_Args...>>
    {
        using type = trait::GetSizeType_t<trait::GetExtentType_t<typename T_Datahandle::element_type>>;
    };
} // namespace alpaka::trait
