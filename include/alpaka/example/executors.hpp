/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/tag.hpp"

#include <algorithm>

namespace alpaka::onHost
{
    constexpr auto getExecutorsList(auto const apiAndDeviceTagList)
    {
        using DevSelectorType
            = decltype(makeDeviceSelector(apiAndDeviceTagList[object::api], apiAndDeviceTagList[object::device]));
        using DeviceType = decltype(std::declval<DevSelectorType>().makeDevice(0));
        using AutoDeviceMappings = decltype(supportedMappings(std::declval<DeviceType>()));
        return AutoDeviceMappings{};
    }

    constexpr auto getDevicesFor(auto const api)
    {
        return std::apply(
            [api](auto... devTag) constexpr {
                return std::make_tuple(Dict{DictEntry{object::api, api}, DictEntry{object::device, devTag}}...);
            },
            supportedDevices(api));
    }

    constexpr auto createBackendsFor(auto const apiDeviceDict)
    {
        return std::apply(
            [apiDeviceDict](auto... executor) constexpr
            {
                return std::make_tuple(Dict{
                    DictEntry{object::api, apiDeviceDict[object::api]},
                    DictEntry{object::device, apiDeviceDict[object::device]},
                    DictEntry{object::exec, executor}}...);
            },
            getExecutorsList(apiDeviceDict));
    }

    constexpr auto createBackendList(auto const apiDeviceDictList)
    {
        return std::apply(
            [](auto... apiDeviceDict) constexpr
            { return std::tuple_cat(createBackendsFor(apiDeviceDict)...); },
            apiDeviceDictList);
    }

    consteval auto allBackends(auto const usedApis)
    {
        return std::apply(
            [](auto... api) constexpr { return std::tuple_cat(createBackendList(getDevicesFor(api))...); },
            usedApis);
    }
} // namespace alpaka::onHost
