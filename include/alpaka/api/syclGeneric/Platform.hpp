/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "Device.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/core/Sycl.hpp"
#    include "alpaka/internal/interface.hpp"

#    include <sycl/sycl.hpp>

#    include <map>
#    include <memory>
#    include <numeric>
#    include <optional>

namespace alpaka
{
    namespace detail
    {
        template<typename T_DeviceKind>
        struct SYCLDeviceSelector;

        struct Context
        {
            Context() = default;

            sycl::platform getPlatformByName(std::string const& platformName)
            {
                auto platforms = sycl::platform::get_platforms();

                for(auto const& platform : platforms)
                {
                    if(platform.get_info<sycl::info::platform::name>() == platformName)
                    {
                        return platform;
                    }
                }

                throw std::runtime_error("Platform not found");
            }

            auto getContext(sycl::platform platform)
            {
                std::string platformName = platform.get_info<sycl::info::platform::name>();
                if(contextMap.contains(platformName))
                {
                    return contextMap[platformName];
                }

                std::vector<sycl::device> devices;
                try
                {
                    devices = platform.get_devices();
                }
                catch(...)
                {
                    devices.clear();
                }
                if(devices.size())
                {
                    auto context = sycl::context{
                        platform.get_devices(),
                        [](sycl::exception_list exceptions)
                        {
                            auto ss_err = std::stringstream{};
                            ss_err << "Caught asynchronous SYCL exception(s):\n";
                            for(std::exception_ptr e : exceptions)
                            {
                                try
                                {
                                    std::rethrow_exception(e);
                                }
                                catch(sycl::exception const& err)
                                {
                                    ss_err << err.what() << " (" << err.code() << ")\n";
                                }
                            }
                            throw std::runtime_error(ss_err.str());
                        }};
                    return contextMap[platformName] = context;
                }
                return sycl::context{};
            }

            std::map<std::string, sycl::context> contextMap;
        };
    } // namespace detail

    namespace onHost

    {
        namespace syclGeneric
        {
            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct Platform : std::enable_shared_from_this<Platform<T_ApiInterface, T_DeviceKind>>
            {
            public:
                Platform() : contextManager{make_sharedSingleton<detail::Context>()}
                {
                    try
                    {
                        syclPlatform = sycl::platform{detail::SYCLDeviceSelector<T_DeviceKind>{}};
                        syclDevices = syclPlatform->get_devices();
                        devices.resize(syclDevices.size());
                        syclContext = contextManager->getContext(syclPlatform.value());
                    }
                    catch(...)
                    {
                        syclContext.reset();
                        syclPlatform.reset();
                        syclDevices.clear();
                        devices.clear();
                    }
                }

                Platform(Platform const&) = delete;
                Platform& operator=(Platform const&) = delete;

                Platform(Platform&&) = delete;
                Platform& operator=(Platform&&) = delete;

                std::shared_ptr<Platform<T_ApiInterface, T_DeviceKind>> getSharedPtr()
                {
                    return this->shared_from_this();
                }

                auto getContext() const
                {
                    if(!syclContext.has_value())
                        throw std::runtime_error("The underlying SYCL context is invalid.");
                    return syclContext.value();
                }

                uint32_t getDeviceCount() const
                {
                    constexpr bool isSupportedDev = trait::IsDeviceSupportedBy::
                        Op<T_DeviceKind, ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<Platform>()))>::value;
                    if constexpr(isSupportedDev)
                    {
                        auto numDevices = devices.size();
                        return static_cast<uint32_t>(numDevices);
                    }
                    return 0u;
                }

                Handle<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>> makeDevice(uint32_t const& idx)
                {
                    uint32_t const numDevices = getDeviceCount();
                    if(idx >= numDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for SYCL device with index " << idx
                              << " because there are only " << numDevices << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    std::lock_guard<std::mutex> lk{deviceGuard};

                    if(auto sharedPtr = devices[idx].lock())
                    {
                        return sharedPtr;
                    }

                    auto newDevice = std::make_shared<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>(
                        std::move(getSharedPtr()),
                        syclDevices[idx],
                        idx);
                    devices[idx] = newDevice;
                    return newDevice;
                }

                static constexpr auto getName()
                {
                    return onHost::demangledName<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>();
                }

                friend struct internal::GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>;
                friend struct onHost::internal::MakeDevice;

            private:
                friend struct onHost::internal::IsDataAccessible;
                friend struct GetDeviceProperties;

                // The context manager is required to be able to use the same sycl context for different device types
                std::shared_ptr<alpaka::detail::Context> contextManager;
                std::optional<sycl::context> syclContext;
                // native sycl platform for the corresponding device kind this platform is representing
                std::optional<sycl::platform> syclPlatform;
                // native sycl devices for the corresponding device kind this platform is representing
                std::vector<sycl::device> syclDevices;
                // alpaka devices for the internal hierarchy
                std::vector<std::weak_ptr<syclGeneric::Device<Platform<T_ApiInterface, T_DeviceKind>>>> devices;

                std::mutex deviceGuard;

                void _()
                {
                    static_assert(internal::concepts::Platform<Platform>);
                }
            };
        } // namespace syclGeneric

        namespace internal
        {
            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct MakeDevice::Link<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>
            {
                Handle<syclGeneric::Device<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>> operator()(
                    syclGeneric::Platform<T_ApiInterface, T_DeviceKind>& platform,
                    std::pair<sycl::device, sycl::context> const& nativeHandle,
                    bool syncBeforeDestroy)
                {
                    /* if there  is at least one alpaka manged device in the list we can not link device for the
                     * deviceKind because it is requiring to overwrite the sycl context to guarantee that external
                     * allocated memory can be uses by alpaka */
                    bool canApplyContext = true;
                    bool hasAlreadyLinkedDevices = false;
                    for(auto& deviceWeakPtr : platform.devices)
                    {
                        if(deviceWeakPtr.use_count() != 0)
                        {
                            if(auto sharedPtr = deviceWeakPtr.lock())
                            {
                                if(sharedPtr->m_manageDevice)
                                    canApplyContext = false;
                                else
                                    hasAlreadyLinkedDevices = true;
                            }
                        }
                    }

                    /* Currently we search the context for the device kind by our self and here show only if the
                     * context is equal to the context provided by the user. Maybe this is not correct and we should
                     * create the alpaka sycl platform already with the user provided context.
                     * @todo revisit linking existing devices and see if the way how we handle linking of CUDA/HIP
                     * devices is the right way for SYCL too.
                     */
                    if(!canApplyContext)
                    {
                        throw std::runtime_error(
                            "For this device kind there are already alpaka managed devices in the list. Linking "
                            "external devices is not possible.");
                    }
                    if(!hasAlreadyLinkedDevices)
                    {
                        platform.syclPlatform = std::get<0>(nativeHandle).get_platform();
                        platform.syclDevices = platform.syclPlatform->get_devices();
                        platform.devices.resize(platform.syclDevices.size());
                        std::string platformName = platform.syclPlatform->template get_info<sycl::info::platform::name>();
                        platform.contextManager->contextMap[platformName] = std::get<1>(nativeHandle);
                        platform.syclContext = platform.contextManager->getContext(platform.syclPlatform.value());
                    }

                    uint32_t const numDevices = platform.getDeviceCount();
                    // search if we know the user provided device within our context
                    uint32_t idx = numDevices;
                    for(uint32_t d = 0; d < numDevices; ++d)
                    {
                        if(platform.syclDevices[d] == std::get<0>(nativeHandle))
                        {
                            idx = d;
                            break;
                        }
                    }

                    if(idx == numDevices)
                        throw std::runtime_error(std::string("Device handle provided not found."));

                    std::lock_guard<std::mutex> lk{platform.deviceGuard};

                    if(auto sharedPtr = platform.devices[idx].lock())
                    {
                        return sharedPtr;
                    }

                    auto newDevice = std::make_shared<syclGeneric::Device<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>>(
                        std::move(platform.getSharedPtr()),
                        std::get<0>(nativeHandle),
                        idx,
                        syncBeforeDestroy);
                    platform.devices[idx] = newDevice;
                    return newDevice;
                }
            };

            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct MakeDevice::Unlink<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>
            {
                void operator()(
                    syclGeneric::Platform<T_ApiInterface, T_DeviceKind>& platform,
                    std::pair<sycl::device, sycl::context> nativeHandle) const
                {
                    std::lock_guard<std::mutex> lk{platform.deviceGuard};

                    uint32_t const numDevices = platform.getDeviceCount();
                    // search if we know the user provided device within our context
                    uint32_t idx = numDevices;
                    for(uint32_t d = 0; d < numDevices; ++d)
                    {
                        if(platform.syclDevices[d] == std::get<0>(nativeHandle))
                        {
                            idx = d;
                            break;
                        }
                    }
                    if(idx == numDevices)
                        throw std::runtime_error(std::string("Device handle provided not found."));

                    auto sharedPtr = platform.devices[idx].lock();
                    if(!sharedPtr)
                        throw std::runtime_error(
                            std::string("Try to unlink unknown device") + std::to_string(idx) + "'");
                    if(sharedPtr->m_manageDevice)
                        throw std::runtime_error(
                            std::string("Unlinking an alpaka managed device with id '") + std::to_string(idx)
                            + "' is not allowed.");

                    platform.devices[idx].reset();
                }
            };

            template<typename T_ApiInterface, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface, T_DeviceKind>>
            {
                DeviceProperties operator()(
                    syclGeneric::Platform<T_ApiInterface, T_DeviceKind> const& platform,
                    uint32_t deviceIdx) const
                {
                    if(deviceIdx >= platform.syclDevices.size())
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device properties for SYCL device with index " << deviceIdx
                              << " because there are only " << platform.getDeviceCount() << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }
                    sycl::device const dev = platform.syclDevices[deviceIdx];

                    auto prop = DeviceProperties{};
                    prop.m_name = dev.get_info<sycl::info::device::name>();
                    prop.m_maxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_group_size>();
                    std::vector<std::size_t> wrap_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
                    // @todo do not reduce wrap size to a single value, return all values
                    prop.m_warpSize = static_cast<uint32_t>(std::reduce(
                        wrap_sizes.begin(),
                        wrap_sizes.end(),
                        std::size_t{0},
                        [](std::size_t a, std::size_t b)
                        {
                            // The CPU runtime supports a sub-group size of 64, but the SYCL implementation
                            // currently does not
                            return std::max(a, b) <= 32 ? std::max(a, b) : 32;
                        }));
                    prop.m_multiProcessorCount = dev.get_info<sycl::info::device::max_compute_units>();

                    return prop;
                }
            };
        } // namespace internal

    } // namespace onHost
} // namespace alpaka
#endif
