/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    namespace object
    {
        struct Api
        {
        };

        constexpr Api api;

        ALPAKA_TAG(exec);

        ALPAKA_TAG(device);

        ALPAKA_TAG(dynSharedMemBytes);
    } // namespace object

    namespace device
    {
        namespace detail
        {
            struct DeviceBaseTag
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_DeviceTag>
            struct IsDeviceTag : std::is_base_of<detail::DeviceBaseTag,T_DeviceTag>
            {
            };
        } // namespace trait

        template<typename T_DeviceTag>
        constexpr bool isDeviceTag_v = trait::IsDeviceTag<T_DeviceTag>::value;

        namespace concepts
        {
            template<typename T_DeviceTag>
            concept DeviceTag = isDeviceTag_v<T_DeviceTag>;
        }

        struct Cpu : detail::DeviceBaseTag
        {
            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};

        struct AmdGpu : detail::DeviceBaseTag
        {
            static std::string getName()
            {
                return "AmdGpu";
            }
        };

        constexpr auto amdGpu = AmdGpu{};

        struct NvidiaGpu : detail::DeviceBaseTag
        {
            static std::string getName()
            {
                return "NvidiaGpu";
            }
        };

        constexpr auto nvidiaGpu = NvidiaGpu{};

        struct IntelGpu : detail::DeviceBaseTag
        {
            static std::string getName()
            {
                return "IntelGpu";
            }
        };

        constexpr auto intelGpu = IntelGpu{};

        constexpr auto allDevices = std::make_tuple(cpu,amdGpu,nvidiaGpu,intelGpu);

    } // namespace device
#if 0
    namespace vendor
    {
        namespace detail
        {
            struct VendoreBaseTag
            {
            };
        } // namespace detail

        namespace trait
        {
            template<typename T_VendorTag>
            struct IsVendorTag : std::is_base_of<T_VendorTag, detail::VendoreBaseTag>
            {
            };
        } // namespace trait

        template<typename T_DeviceTag>
        constexpr bool isVendorTag_v = trait::IsVendorTag<T_DeviceTag>::value;

        namespace concepts
        {
            template<typename T_VendorTag>
            concept VendorTag = isVendorTag_v<T_VendorTag>;
        }

        struct Nvidia : detail::VendoreBaseTag
        {
            static std::string getName()
            {
                return "Nvidia";
            }
        };

        struct AMD : detail::VendoreBaseTag
        {
            static std::string getName()
            {
                return "AMD";
            }
        };

        struct Intel : detail::VendoreBaseTag
        {
            static std::string getName()
            {
                return "Intel";
            }
        };

        struct Any : detail::VendoreBaseTag
        {
            static std::string getName()
            {
                return "Any";
            }
        };
        constexpr auto any = Any{};
    } // namespace vendor
#endif
    namespace layer
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(block);
        ALPAKA_TAG(shared);
        ALPAKA_TAG(dynShared);
    } // namespace layer

    namespace frame
    {
        ALPAKA_TAG(count);
        ALPAKA_TAG(extent);
    } // namespace frame

    namespace action
    {
        ALPAKA_TAG(sync);
    } // namespace action

    struct Empty
    {
    };

    namespace exec
    {


        namespace traits
        {
            template<typename T_Mapping>
            struct IsSeqExecutor : std::false_type
            {
            };

            template<typename T_Exec>
            constexpr bool isSeqExecutor_v = IsSeqExecutor<T_Exec>::value;

        } // namespace traits
    } // namespace exec

    /** check if a executor can only be used with a single thred per block
     *
     * @return true if a block can only have a single thread, else false
     */
    template<typename T_Exec>
    consteval bool isSeqExecutor(T_Exec exec)
    {
        return exec::traits::isSeqExecutor_v<T_Exec>;
    }
} // namespace alpaka
