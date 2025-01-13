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
    template<typename T_BaseHandle, typename T_PitchedPtr>
    struct MangedData
        : T_PitchedPtr
        , std::enable_shared_from_this<MangedData<T_BaseHandle, T_PitchedPtr>>
    {
    public:
        MangedData(T_BaseHandle base, T_PitchedPtr const pitchedPtr) : T_PitchedPtr{std::move(pitchedPtr)}, m_base{std::move(base)}
        {
        }

        MangedData(MangedData const&) = default;
        MangedData(MangedData&&) = default;

        MangedData& operator=(MangedData const &) = default;
        MangedData& operator=(MangedData&&) = default;

        ~MangedData()
        {
            this->free();
        }

        std::shared_ptr<MangedData> getSharedPtr()
        {
            return this->shared_from_this();
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

    template<typename T_BaseHandle, typename T_PitchedPtr>
    struct GetApi::Op<onHost::MangedData<T_BaseHandle, T_PitchedPtr>>
    {
        decltype(auto) operator()(auto&& data) const
        {
            return onHost::getApi(data.m_base);
        }
    };
} // namespace alpaka::internal

namespace alpaka::trait
{
    template<typename T_BaseHandle, typename T_PitchedPtr>
    struct GetValueType<onHost::MangedData<T_BaseHandle, T_PitchedPtr>>
    {
        using type = GetValueType_t<T_PitchedPtr>;
    };
#if 0
    template<typename T_BaseHandle, typename T_PitchedPtr>
    struct GetExtentType<onHost::MangedData<T_BaseHandle, T_PitchedPtr>>
    {
        using type = GetExtentType_t<T_PitchedPtr>;
    };

    template<typename T_BaseHandle, typename T_PitchedPtr>
    struct GetSizeType<onHost::MangedData<T_BaseHandle, T_PitchedPtr>>
    {
        using type = trait::GetSizeType_t<T_PitchedPtr>;
    };
#endif
} // namespace alpaka::trait
