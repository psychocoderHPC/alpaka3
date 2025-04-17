/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onHost
{
    template<typename T_Api, alpaka::device::concepts::DeviceTag T_DeviceTag>
    struct DeviceSelector
    {
    public:
        DeviceSelector(T_Api api, T_DeviceTag deviceTag)
            : m_platform(internal::makePlatform(api))
            , m_deviceTag(deviceTag)
        {
        }

        uint32_t getDeviceCount() const
        {
            return internal::GetDeviceCount::Op<ALPAKA_TYPEOF(*m_platform.get()), T_DeviceTag>{}(*m_platform.get());
        }

        auto makeDevice(uint32_t idx)
        {
            return internal::MakeDevice::Op<ALPAKA_TYPEOF(*m_platform.get()), T_DeviceTag>{}(
                *m_platform.get(),
                idx,
                m_deviceTag);
        }

    private:
        ALPAKA_TYPEOF(onHost::internal::MakePlatform::Op<T_Api>{}(T_Api{})) m_platform;
        T_DeviceTag m_deviceTag;
    };
} // namespace alpaka::onHost
