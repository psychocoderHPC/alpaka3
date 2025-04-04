/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/Vec.hpp"
#    include "alpaka/core/Assert.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/core/Vectorize.hpp"
#    include "alpaka/tag.hpp"

#    include <sycl/sycl.hpp>

#    include <functional>

namespace alpaka::onAcc
{
    namespace syclGeneric
    {

        template<typename T_NumBlocks, auto TDim>
        class BlockLayer
        {
            using TIdx = typename T_NumBlocks::type;

            sycl::nd_item<TDim> const& m_item;

        public:
            BlockLayer(sycl::nd_item<TDim> const& item) : m_item(item)
            {
            }

            constexpr auto idx() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_group(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_group(0), m_item.get_group(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{m_item.get_group(0), m_item.get_group(1), m_item.get_group(2)};
                }
            }

            constexpr auto count() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_group_range(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_group_range(0), m_item.get_group_range(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{
                        m_item.get_group_range(0),
                        m_item.get_group_range(1),
                        m_item.get_group_range(2)};
                }
            }
        };

        template<typename T_NumThreads, auto TDim>
        class ThreadLayer
        {
            using TIdx = typename T_NumThreads::type;

            sycl::nd_item<TDim> const& m_item;

        public:
            ThreadLayer(sycl::nd_item<TDim> const& item) : m_item(item)
            {
            }

            constexpr auto idx() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_local_id(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_local_id(0), m_item.get_local_id(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{m_item.get_local_id(0), m_item.get_local_id(1), m_item.get_local_id(2)};
                }
            }

            constexpr auto count() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_local_range(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_local_range(0), m_item.get_local_range(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{
                        m_item.get_local_range(0),
                        m_item.get_local_range(1),
                        m_item.get_local_range(2)};
                }
            }

            constexpr auto count() const requires alpaka::concepts::CVector<T_NumThreads>
            {
                return T_NumThreads{};
            }
        };

        template<auto TDim>
        class Sync
        {
            sycl::nd_item<TDim> const& m_item;

        public:
            Sync(sycl::nd_item<TDim> const& item) : m_item(item)
            {
            }

            void operator()() const
            {
                m_item.barrier();
            }
        };

        namespace detail
        {
            //! Implementation of static block shared memory provider.
            //!
            //! externally allocated fixed-size memory, likely provided by BlockSharedMemDynMember.
            class BlockSharedMemStMemberImpl
            {
                struct MetaData
                {
                    //! pointer to allocated data
                    uint8_t* ptr = nullptr;
                    //! Unique id if the next data chunk.
                    size_t id = std::numeric_limits<size_t>::max();
                };

                static constexpr uint32_t metaDataSize = sizeof(MetaData);

            public:
#    ifndef NDEBUG
                BlockSharedMemStMemberImpl(std::uint8_t* mem, uint32_t capacity)
                    : m_mem(reinterpret_cast<MetaData*>(mem))
                    , m_capacity(capacity / metaDataSize)
                {
                    ALPAKA_ASSERT_ACC((m_mem == nullptr) == (m_capacity == 0u));
                }
#    else
                BlockSharedMemStMemberImpl(std::uint8_t* mem, uint32_t) : m_mem(reinterpret_cast<MetaData*>(mem))
                {
                }
#    endif

                /** number of bytes required for bookkeeping of mayNumberOfAllocations unique allocations
                 *
                 * @param maxNumUniqueAllocations number of unique allocation a user is allowed to perform
                 * @return bytes required to store lookup meta data
                 */
                static consteval uint32_t sizeLookupBufferInBytes(uint32_t maxNumUniqueAllocations)
                {
                    return metaDataSize * maxNumUniqueAllocations;
                }

                template<typename T>
                T* alloc(size_t id) const
                {
                    auto group = sycl::ext::oneapi::this_work_item::get_work_group<1>();
                    T* data = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(group);

                    MetaData& metaDataEntry = m_mem[m_numEntries];
                    ++m_numEntries;
                    ALPAKA_ASSERT_ACC(m_numEntries <= m_capacity);

                    // Update meta data with id and pointer to the current allocation
                    if(group.get_local_linear_id() == 0u)
                    {
                        // only one thread must update the pointer in shared memory
                        metaDataEntry.ptr = reinterpret_cast<uint8_t*>(data);
                    }
                    metaDataEntry.id = id;

                    return data;
                }

                //! Give the pointer to an exiting variable
                //!
                //! @tparam T type of the variable
                //! @param id unique id of the variable
                //! @return nullptr if variable with id not exists
                template<typename T>
                auto getVarPtr(size_t id) const -> T*
                {
                    // Iterate over metadata
                    for(uint32_t off = 0u; off < m_numEntries; ++off)
                    {
                        MetaData& metaDataEntry = m_mem[off];

                        if(metaDataEntry.id == id)
                            return reinterpret_cast<T*>(metaDataEntry.ptr);
                    }

                    // Variable not found.
                    return nullptr;
                }

            private:
                //! Number unqiue meta data entries stored
                mutable uint32_t m_numEntries = 0u;

                //! Memory layout
                //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
                //! Size of padding can be zero if data after padding is already aligned.
                MetaData* const m_mem;
#    ifndef NDEBUG
                //! max number of meta data entries
                uint32_t const m_capacity;
#    endif
            };
        } // namespace detail

        class StaticSharedMemory : private detail::BlockSharedMemStMemberImpl
        {
        public:
            /** number of bytes required for bookkeeping of mayNumberOfAllocations unique allcoations
             *
             * @param maxNumUniqueAllocations number of unique allocation a user is allowed to perform
             * @return bytes required to store lookup meta data
             */
            static consteval uint32_t sizeLookupBufferInBytes(uint32_t maxNumUniqueAllocations)
            {
                return detail::BlockSharedMemStMemberImpl::sizeLookupBufferInBytes(maxNumUniqueAllocations);
            }

            StaticSharedMemory(StaticSharedMemory const&) = delete;

            /** Construct shared memory allocator
             * @param accessor local memory accessor to store lookup meta data
             *                 bytes required to store N unique allocation can be calculated with
             * sizeLookupBufferInBytes()
             */
            StaticSharedMemory(sycl::local_accessor<std::byte> const& accessor)
                : BlockSharedMemStMemberImpl(
                    reinterpret_cast<std::uint8_t*>(accessor.get_multi_ptr<sycl::access::decorated::no>().get()),
                    static_cast<uint32_t>(accessor.size()))

            {
            }

            using Base = detail::BlockSharedMemStMemberImpl;

            template<typename T, size_t T_unique>
            T& allocVar()
            {
                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    data = Base::template alloc<T>(T_unique);
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }
        };

        class DynamicSharedMemory
        {
            sycl::local_accessor<std::byte> const& m_accessor;

        public:
            DynamicSharedMemory(sycl::local_accessor<std::byte> const& accessor) : m_accessor(accessor)
            {
            }

            template<typename T, size_t>
            T* allocDynamic(uint32_t)
            {
                return reinterpret_cast<T*>(m_accessor.get_multi_ptr<sycl::access::decorated::no>().get());
            }

            constexpr size_t byte_size() noexcept
            {
                return m_accessor.byte_size();
            }
        };
    } // namespace syclGeneric

    template<typename T_Executor, typename T_Api, typename T_NumBlocks, typename T_NumThreads, auto TDim>
    auto makeSyclGenericAccDict(
        sycl::nd_item<TDim> const& work_item,
        onAcc::syclGeneric::StaticSharedMemory& static_shared_memory,
        onAcc::syclGeneric::DynamicSharedMemory& dynamic_shared_memory)
    {
        static_assert(TDim > 0);
        static_assert(TDim <= 3, "more the 3 dimensions are not supported");
        return Dict{
            DictEntry(layer::block, onAcc::syclGeneric::BlockLayer<T_NumBlocks, TDim>{work_item}),
            DictEntry(layer::thread, onAcc::syclGeneric::ThreadLayer<T_NumThreads, TDim>{work_item}),
            DictEntry(layer::shared, std::ref(static_shared_memory)),
            DictEntry(layer::dynShared, std::ref(dynamic_shared_memory)),
            DictEntry(object::dynSharedMemBytes, dynamic_shared_memory.byte_size()),
            DictEntry(action::sync, onAcc::syclGeneric::Sync{work_item}),
            DictEntry(object::api, T_Api{}),
            DictEntry(object::exec, T_Executor{})};
    };

} // namespace alpaka::onAcc

#endif
