/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/mem/PitchedPtr.hpp"

#include <cstdint>
#include <memory>

namespace alpaka::onHost
{
    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct UnMangedData : PitchedPtr<T_Type, T_Extents, T_Pitches>
    {
    public:
        UnMangedData(T_BaseHandle base, PitchedPtr<T_Type, T_Extents, T_Pitches> const pitchedPtr)
            : PitchedPtr<T_Type, T_Extents, T_Pitches>{std::move(pitchedPtr)}
            , m_base{std::move(base)}
        {
        }

        UnMangedData(UnMangedData const&) = default;
        UnMangedData(UnMangedData&&) = default;

        UnMangedData& operator=(UnMangedData const&) = default;
        UnMangedData& operator=(UnMangedData&&) = default;

        ~UnMangedData()
        {
        }

    private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        T_BaseHandle m_base;

        friend struct alpaka::internal::GetApi;
    };
} // namespace alpaka::onHost

namespace alpaka::internal
{

    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct GetApi::Op<onHost::UnMangedData<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        decltype(auto) operator()(auto&& data) const
        {
            return onHost::getApi(data.m_base);
        }
    };
} // namespace alpaka::internal

namespace alpaka::trait
{
    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct GetValueType<onHost::UnMangedData<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        using type = GetValueType_t<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>;
    };

    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct GetDim<onHost::UnMangedData<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        static constexpr uint32_t value = getDim_v<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>;
    };
#if 0
    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct GetExtentType<onHost::UnMangedData<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        using type = GetExtentType_t<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>;
    };

    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct GetSizeType<onHost::UnMangedData<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        using type = trait::GetSizeType_t<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>;
    };
#endif
} // namespace alpaka::trait
