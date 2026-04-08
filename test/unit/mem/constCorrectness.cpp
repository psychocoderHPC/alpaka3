/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/mem/concepts/detail/CopyConstructableDataSource.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <iostream>

using namespace alpaka;

/** @file test the behaviour for constness of buffers/views/MdSpan
 *
 * We run two different test, one for access with 1D memory and the usage of scalars to access the memory and one for
 * multidimensional memory, because there are specializations for 1D in our implementations.
 */

/** validate that the buffer is mutable
 *
 * If we use non-outer const buffers/views in the function signature it should be possible to call the function even if
 * the buffer/view had an outer const before. The reason why we can not transform the outer const into an inner const
 * is that we requires this within @c KernelBundle to forward the kernel user arguments in a tuple to the user kernel.
 */
void requiresMutableBufferMd(concepts::IMdSpan auto buffer)
{
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>);
    static_assert(!std::is_const_v<std::remove_reference_t<decltype(buffer[Vec{0, 0}])>>);
}

TEST_CASE("mdspan inner const copy constructor", "[mem][mdspan][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutMdSpan>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutMdSpan>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstMdSpan>);

    MutMdSpan mut_mdspan(ptr, extents, pitches);
    ConstMdSpan const_mdspan(const_ptr, extents, pitches);

    STATIC_REQUIRE(std::constructible_from<MutMdSpan, MutMdSpan&>);
    [[maybe_unused]] MutMdSpan mut_mdspan_copy(mut_mdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, ConstMdSpan&>);
    [[maybe_unused]] ConstMdSpan const_mdspan_copy(const_mdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, MutMdSpan&>);
    [[maybe_unused]] ConstMdSpan mut_to_const_mdspan(mut_mdspan);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutMdSpan, ConstMdSpan&>);
    // should not compile
    // MdSpan const_to_mud_mdspan(const_mdspan);
}

TEST_CASE("mdspan inner const assignment operator", "[mem][mdspan][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan mut_mdspan(ptr, extents, pitches);
    ConstMdSpan const_mdspan(const_ptr, extents, pitches);

    STATIC_REQUIRE(concepts::AssignableFrom<MutMdSpan&, MutMdSpan>);
    [[maybe_unused]] MutMdSpan mut_mdspan2 = mut_mdspan;
    mut_mdspan2 = mut_mdspan;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpan&, ConstMdSpan>);
    [[maybe_unused]] ConstMdSpan const_mdspan2 = const_mdspan;
    const_mdspan2 = const_mdspan;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpan&, MutMdSpan>);
    [[maybe_unused]] ConstMdSpan const_mdspan3 = mut_mdspan;
    const_mdspan3 = mut_mdspan;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutMdSpan&, ConstMdSpan>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutMdSpan mut_mdspan3 = const_mdspan;
    // mut_mdspan3 = const_mdspan;
}

TEST_CASE("mdspan inner const move operator", "[mem][mdspan][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan copy_mut_mdspan(ptr, extents, pitches);
    ConstMdSpan copy_const_mdspan(const_ptr, extents, pitches);

    [[maybe_unused]] MutMdSpan copy_mut_mdspan2(std::move(copy_mut_mdspan));
    [[maybe_unused]] ConstMdSpan copy_const_mdspan2(std::move(copy_const_mdspan));
    [[maybe_unused]] ConstMdSpan copy_const_mdspan3(std::move(copy_mut_mdspan2));
    // should not compile
    // MutMdSpan copy_mut_mdspan3(std::move(copy_const_mdspan3));

    MutMdSpan assign_mut_mdspan(ptr, extents, pitches);
    ConstMdSpan assign_const_mdspan(const_ptr, extents, pitches);

    MutMdSpan assign_mut_mdspan2 = std::move(assign_mut_mdspan);
    [[maybe_unused]] ConstMdSpan assign_const_mdspan2 = std::move(assign_const_mdspan);
    [[maybe_unused]] ConstMdSpan assign_const_mdspan3 = std::move(assign_mut_mdspan2);
    // should not compile
    // MutMdSpan assign_mut_mdspan3 = std::move(assign_const_mdspan3);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void funcCopyByValue(TMdSpan mdspan)
{
    STATIC_REQUIRE(std::same_as<typename decltype(mdspan)::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void funcReference(TMdSpan& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void funcConstReference(TMdSpan const& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void funcUniversalRef(TMdSpan&& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

TEST_CASE("function calls with mdspan object", "[mem][mdspan][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdspan{ptr, extents, pitches};
    funcCopyByValue<int, int&>(mdspan);
    funcReference<int, int&>(mdspan);
    funcConstReference<int, int const&>(mdspan);
    funcUniversalRef<int, int&>(mdspan);
    funcUniversalRef<int, int&>(MdSpan{ptr, extents, pitches});

    MdSpan const const_mdspan{ptr, extents, pitches};
    funcCopyByValue<int, int&>(const_mdspan);
    funcReference<int, int const&>(const_mdspan);
    funcConstReference<int, int const&>(const_mdspan);

    MdSpan mdspan_inner_const{const_ptr, extents, pitches};
    funcCopyByValue<int const, int const&>(mdspan_inner_const);
    funcReference<int const, int const&>(mdspan_inner_const);
    funcConstReference<int const, int const&>(mdspan_inner_const);
    funcUniversalRef<int const, int const&>(mdspan_inner_const);
    funcUniversalRef<int const, int const&>(MdSpan{const_ptr, extents, pitches});

    MdSpan const const_mdspan_inner_const{const_ptr, extents, pitches};
    funcCopyByValue<int const, int const&>(const_mdspan_inner_const);
    funcReference<int const, int const&>(const_mdspan_inner_const);
    funcConstReference<int const, int const&>(const_mdspan_inner_const);
    funcUniversalRef<int const, int const&>(const_mdspan_inner_const);
}

TEST_CASE("MdSpanArray inner const copy constructor", "[mem][mdspanarray][correctness][copyConstruct]")
{
    int static_data[2][2] = {{0, 0}, {0, 0}};
    int const const_static_data[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(static_data), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(const_static_data), size_t, alpaka::Alignment<16>>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutMdSpanArray>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutMdSpanArray>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstMdSpanArray>);

    MutMdSpanArray mut_md_span_array(static_data);
    ConstMdSpanArray const_md_span_array(const_static_data);

    STATIC_REQUIRE(std::constructible_from<MutMdSpanArray, MutMdSpanArray&>);
    [[maybe_unused]] MutMdSpanArray mut_md_span_array_copy(mut_md_span_array);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpanArray, ConstMdSpanArray&>);
    [[maybe_unused]] ConstMdSpanArray const_md_span_array_copy(const_md_span_array);


    STATIC_REQUIRE(std::constructible_from<ConstMdSpanArray, MutMdSpanArray&>);
    [[maybe_unused]] ConstMdSpanArray mut_to_const_md_span_array(mut_md_span_array);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutMdSpanArray, ConstMdSpanArray&>);
    // should not compile
    // MutMdSpanArray const_to_mut_md_span_array(const_md_span_array);
}

TEST_CASE("MdSpanArray inner const assignment operator", "[mem][mdspanarray][correctness][copyConstruct]")
{
    int static_data[2][2] = {{0, 0}, {0, 0}};
    int const const_static_data[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(static_data), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(const_static_data), size_t, alpaka::Alignment<16>>;

    MutMdSpanArray mut_md_span_array(static_data);
    ConstMdSpanArray const_md_span_array(const_static_data);

    STATIC_REQUIRE(concepts::AssignableFrom<MutMdSpanArray&, MutMdSpanArray>);
    [[maybe_unused]] MutMdSpanArray mut_md_span_arra2 = mut_md_span_array;
    mut_md_span_arra2 = mut_md_span_array;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpanArray&, ConstMdSpanArray>);
    [[maybe_unused]] ConstMdSpanArray const_md_span_array2 = const_md_span_array;
    const_md_span_array2 = const_md_span_array;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpanArray&, MutMdSpanArray>);
    [[maybe_unused]] ConstMdSpanArray const_md_span_array3 = mut_md_span_array;
    const_md_span_array3 = mut_md_span_array;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutMdSpanArray&, ConstMdSpanArray>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutMdSpanArray mut_view3 = const_md_span_array;
    // mut_view3 = const_md_span_array;
}

TEST_CASE(
    "MdSpanArray inner const move constructor and move assignment operator",
    "[mem][mdspanarray][correctness][moveConstruct]")
{
    int static_data[2][2] = {{0, 0}, {0, 0}};
    int const const_static_data[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(static_data), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(const_static_data), size_t, alpaka::Alignment<16>>;

    MutMdSpanArray constr_mut_md(static_data);
    ConstMdSpanArray constr_const_md(const_static_data);

    MutMdSpanArray constr_mut_md2(std::move(constr_mut_md));
    [[maybe_unused]] ConstMdSpanArray constr_const_md2(std::move(constr_const_md));
    [[maybe_unused]] ConstMdSpanArray constr_const_md3(std::move(constr_mut_md2));
    // should not compile
    // MutMdSpanArray constr_mut_md3(std::move(constr_const_md3));

    MutMdSpanArray assign_mut_md(static_data);
    ConstMdSpanArray assign_const_md(const_static_data);

    MutMdSpanArray assign_mut_md2 = std::move(assign_mut_md);
    [[maybe_unused]] ConstMdSpanArray assign_const_md2 = std::move(assign_const_md);
    [[maybe_unused]] ConstMdSpanArray assign_const_md3 = std::move(assign_mut_md2);
    // should not compile
    // MutMdSpanArray assign_mut_md3 = std::move(assign_const_md3);
}

TEST_CASE("View inner const copy constructor", "[mem][view][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutView>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutView>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstView>);

    MutView mut_view(api::host, ptr, extents, pitches);
    ConstView const_view(api::host, const_ptr, extents, pitches);

    STATIC_REQUIRE(std::constructible_from<MutView, MutView&>);
    [[maybe_unused]] MutView mut_view_copy(mut_view);

    STATIC_REQUIRE(std::constructible_from<ConstView, ConstView&>);
    [[maybe_unused]] ConstView const_view_copy(const_view);

    STATIC_REQUIRE(std::constructible_from<ConstView, MutView&>);
    [[maybe_unused]] ConstView mut_to_const_view(mut_view);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutView, ConstView&>);
    // should not compile
    // MutView const_to_mud_view(const_view);
}

TEST_CASE("View inner const assignment operator", "[mem][view][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView mut_view(api::host, ptr, extents, pitches);
    ConstView const_view(api::host, const_ptr, extents, pitches);

    STATIC_REQUIRE(concepts::AssignableFrom<MutView&, MutView>);
    [[maybe_unused]] MutView mut_view2 = mut_view;
    mut_view2 = mut_view;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstView&, ConstView>);
    [[maybe_unused]] ConstView const_view2 = const_view;
    const_view2 = const_view;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstView&, MutView>);
    [[maybe_unused]] ConstView const_view3 = mut_view;
    const_view3 = mut_view;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutView&, ConstView>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutView mut_view3 = const_view;
    // mut_view3 = const_view;
}

TEST_CASE("View inner const move constructor and move assignment operator", "[mem][view][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView construct_mut_view(api::host, ptr, extents, pitches);
    ConstView construct_const_view(api::host, const_ptr, extents, pitches);

    MutView construct_mut_view2(std::move(construct_mut_view));
    [[maybe_unused]] ConstView construct_const_view2(std::move(construct_const_view));
    [[maybe_unused]] ConstView construct_const_view3(std::move(construct_mut_view2));
    // should not compile
    // MutView construct_mut_view3(std::move(construct_const_view3));

    MutView assign_mut_view(api::host, ptr, extents, pitches);
    ConstView assign_const_view(api::host, const_ptr, extents, pitches);

    MutView assign_mut_view2 = std::move(assign_mut_view);
    [[maybe_unused]] ConstView assign_const_view2 = std::move(assign_const_view);
    [[maybe_unused]] ConstView assign_const_view3 = std::move(assign_mut_view2);
    // should not compile
    // MutView assign_mut_view3 = std::move(assign_const_view3);
}

TEST_CASE("sharedBuffer inner const copy constructor", "[mem][sharedBuffer][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutSharedBuffer>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutSharedBuffer>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstSharedBuffer>);

    MutSharedBuffer mut_shared_buffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer const_shared_buffer(api::host, const_ptr, extents, pitches, [] {});

    STATIC_REQUIRE(std::constructible_from<MutSharedBuffer, MutSharedBuffer&>);
    [[maybe_unused]] MutSharedBuffer mut_shared_buffer_copy(mut_shared_buffer);

    STATIC_REQUIRE(std::constructible_from<ConstSharedBuffer, ConstSharedBuffer&>);
    [[maybe_unused]] ConstSharedBuffer const_shared_buffer_copy(const_shared_buffer);

    STATIC_REQUIRE(std::constructible_from<ConstSharedBuffer, MutSharedBuffer&>);
    [[maybe_unused]] ConstSharedBuffer mut_to_const_shared_buffer(mut_shared_buffer);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutSharedBuffer, ConstSharedBuffer&>);
    // should not compile
    // MutSharedBuffer const_to_mud_shared_buffer(const_shared_buffer);
}

TEST_CASE("sharedBuffer inner const assignment operator", "[mem][sharedBuffer][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    MutSharedBuffer mut_shared_buffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer const_shared_buffer(api::host, const_ptr, extents, pitches, [] {});

    STATIC_REQUIRE(concepts::AssignableFrom<MutSharedBuffer&, MutSharedBuffer>);
    [[maybe_unused]] MutSharedBuffer mut_shared_buffer2 = mut_shared_buffer;
    mut_shared_buffer2 = mut_shared_buffer;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstSharedBuffer&, ConstSharedBuffer>);
    [[maybe_unused]] ConstSharedBuffer const_shared_buffer2 = const_shared_buffer;
    const_shared_buffer2 = const_shared_buffer;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstSharedBuffer&, MutSharedBuffer>);
    [[maybe_unused]] ConstSharedBuffer const_shared_buffer3 = mut_shared_buffer;
    const_shared_buffer3 = mut_shared_buffer;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutSharedBuffer&, ConstSharedBuffer>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutSharedBuffer mut_shared_buffer3 = const_shared_buffer;
    // mut_shared_buffer3 = const_shared_buffer;
}

TEST_CASE("sharedBuffer inner const move operator", "[mem][sharedBuffer][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    MutSharedBuffer copy_mut_shared_buffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer copy_const_shared_buffer(api::host, const_ptr, extents, pitches, [] {});

    MutSharedBuffer copy_mut_shared_buffer2(std::move(copy_mut_shared_buffer));
    ConstSharedBuffer copy_const_shared_buffer2(std::move(copy_const_shared_buffer));
    [[maybe_unused]] ConstSharedBuffer copy_const_shared_buffer3(std::move(copy_mut_shared_buffer2));
    // should not compile
    // MutSharedBuffer copy_mut_shared_buffer3(std::move(copy_const_shared_buffer3));

    MutSharedBuffer assign_mut_shared_buffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer assign_const_shared_buffer(api::host, const_ptr, extents, pitches, [] {});

    MutSharedBuffer assign_mut_shared_buffer2 = std::move(assign_mut_shared_buffer);
    ConstSharedBuffer assign_const_shared_buffer2 = std::move(assign_const_shared_buffer);
    [[maybe_unused]] ConstSharedBuffer assign_const_shared_buffer3 = std::move(assign_mut_shared_buffer2);
    // should not compile
    // MutSharedBuffer assign_mut_shared_buffer3 = std::move(assign_const_shared_buffer3);
}

TEST_CASE("buffer const correctness MD", "[mem][sharedBuffer][correctness]")
{
    auto buffer0 = onHost::allocHost<int>(Vec{10, 10});
    requiresMutableBufferMd(buffer0);
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer0.data())>>);
    auto boundaryDir = alpaka::makeCoreBoundaryDirection<2u>(alpaka::Vec{1u, 2u}, alpaka::Vec{3u, 4u});

    SECTION("getSubView/getSubBuffer from mutable DataStorage object")
    {
        [[maybe_unused]] auto subBuffer0 = buffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = buffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = buffer0.getSubView(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = buffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = buffer0.getView();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = buffer0.getMdSpan();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);

        // BoundaryDirection subviews from mutable views must preserve writable element access.
        [[maybe_unused]] View boundaryView = buffer0.getView();
        [[maybe_unused]] auto boundarySubView = boundaryView.getSubView(boundaryDir);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(boundarySubView.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(boundarySubView[Vec{0, 0}])>>);
    }

    SECTION("getSubView/getSubBuffer from non-mutable DataStorage object (inner const)")
    {
        onHost::SharedBuffer innerConstBuffer0 = buffer0.getConstSharedBuffer();

        [[maybe_unused]] auto subBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = innerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = innerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = innerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = innerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);

        // BoundaryDirection subviews from inner-const views must stay read-only.
        [[maybe_unused]] View boundaryView = innerConstBuffer0.getView();
        [[maybe_unused]] auto boundarySubView = boundaryView.getSubView(boundaryDir);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(boundarySubView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(boundarySubView[Vec{0, 0}])>>);
    }

    SECTION("getSubView/getSubBuffer from non-mutable DataStorage object (outer const)")
    {
        onHost::SharedBuffer const outerConstBuffer0 = buffer0;
        requiresMutableBufferMd(outerConstBuffer0);

        [[maybe_unused]] auto subBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = outerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = outerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = outerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = outerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);

        // BoundaryDirection subviews from outer-const views must also stay read-only.
        [[maybe_unused]] View boundaryView = outerConstBuffer0.getView();
        [[maybe_unused]] auto boundarySubView = boundaryView.getSubView(boundaryDir);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(boundarySubView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(boundarySubView[Vec{0, 0}])>>);
    }
}

TEST_CASE("View::getConstView keeps host metadata and read-only access", "[mem][view][correctness][constView]")
{
    alignas(32) std::array<int, 2 * 3 * 4> storage{};
    for(std::size_t i = 0; i < storage.size(); ++i)
    {
        storage[i] = static_cast<int>(100 + i);
    }

    auto mutableView
        = alpaka::makeView(alpaka::api::host, storage.data(), alpaka::Vec{2u, 3u, 4u}, alpaka::Alignment<32>{});
    auto const constView = mutableView.getConstView();

    SECTION("compile-time const access stays locked down")
    {
        // `getConstView()` must turn a mutable host view into a read-only view without changing its view category.
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(constView)>,
                      View<alpaka::api::Host, int const, alpaka::Vec<uint32_t, 3>, alpaka::Alignment<32>>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(constView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(constView[alpaka::Vec{0u, 0u, 0u}])>>);

        auto const subView = constView.getSubView(alpaka::Vec{1u, 2u, 3u});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView[alpaka::Vec{0u, 0u, 0u}])>>);
        static_assert(std::is_same_v<decltype(constView.getAlignment()), alpaka::Alignment<32>>);
    }

    SECTION("runtime metadata and aliasing are preserved")
    {
        // Converting to a const view must preserve shape, layout, aliasing, and the original alignment contract.
        REQUIRE(constView.getExtents() == mutableView.getExtents());
        REQUIRE(constView.getPitches() == mutableView.getPitches());
        REQUIRE(constView.data() == mutableView.data());

        STATIC_REQUIRE(std::is_same_v<decltype(constView.getAlignment()), alpaka::Alignment<32>>);

        auto const sampleIdx = alpaka::Vec{1u, 2u, 3u};
        REQUIRE(&constView[sampleIdx] == &mutableView[sampleIdx]);
        REQUIRE(constView[sampleIdx] == mutableView[sampleIdx]);

        mutableView[sampleIdx] = 777;
        REQUIRE(constView[sampleIdx] == 777);
    }

    SECTION("subviews created from the const view stay read-only and alias the same storage")
    {
        // Follow-on subviews from the returned const view should keep the parent layout while exposing only const
        // access.
        auto const offset = alpaka::Vec{1u, 1u, 1u};
        auto const extents = alpaka::Vec{1u, 2u, 3u};
        auto const subView = constView.getSubView(offset, extents);

        REQUIRE(subView.getExtents() == extents);
        REQUIRE(subView.getPitches() == constView.getPitches());
        REQUIRE(subView.data() == &constView[offset]);
        STATIC_REQUIRE(std::is_same_v<decltype(subView.getAlignment()), alpaka::Alignment<>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView[alpaka::Vec{0u, 0u, 0u}])>>);

        for(auto idx : alpaka::IdxRange{extents})
        {
            REQUIRE(subView[idx] == mutableView[offset + idx]);
        }
    }
}

TEST_CASE(
    "alpaka::makeView(any) preserves host view shape alignment and inner constness",
    "[mem][view][correctness][makeView]")
{
    auto mutableBuffer = alpaka::onHost::allocHost<int>(alpaka::Vec{2u, 3u});
    auto mutableView = mutableBuffer.getView();
    auto const& outerConstBuffer = mutableBuffer;
    auto const& outerConstView = mutableView;

    alignas(32) std::array<int const, 2 * 3> innerConstStorage{0, 1, 2, 3, 4, 5};
    auto innerConstView
        = alpaka::makeView(alpaka::api::host, innerConstStorage.data(), alpaka::Vec{2u, 3u}, alpaka::Alignment<32>{});

    auto mutableBufferView = alpaka::makeView(mutableBuffer);
    auto outerConstBufferView = alpaka::makeView(outerConstBuffer);
    auto mutableRebuiltView = alpaka::makeView(mutableView);
    auto outerConstRebuiltView = alpaka::makeView(outerConstView);
    auto innerConstRebuiltView = alpaka::makeView(innerConstView);

    SECTION("compile-time rebuilding keeps host API metadata and inner constness")
    {
        // `makeView(any)` should rebuild host buffers and views without changing their API, shape, alignment, or
        // pointed-to constness.
        static_assert(std::same_as<decltype(mutableBufferView.getApi()), alpaka::api::Host>);
        static_assert(std::same_as<decltype(mutableRebuiltView.getApi()), alpaka::api::Host>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableBufferView.getExtents())>,
                      std::remove_cvref_t<decltype(mutableBuffer.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableRebuiltView.getExtents())>,
                      std::remove_cvref_t<decltype(mutableView.getExtents())>>);
        static_assert(
            std::same_as<decltype(mutableBufferView.getAlignment()), decltype(mutableBuffer.getAlignment())>);
        static_assert(std::same_as<decltype(mutableRebuiltView.getAlignment()), decltype(mutableView.getAlignment())>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableBufferView.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutableBufferView[alpaka::Vec{0u, 0u}])>>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableRebuiltView.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutableRebuiltView[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstBufferView.getExtents())>,
                      std::remove_cvref_t<decltype(outerConstBuffer.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstRebuiltView.getExtents())>,
                      std::remove_cvref_t<decltype(outerConstView.getExtents())>>);
        static_assert(
            std::same_as<decltype(outerConstBufferView.getAlignment()), decltype(outerConstBuffer.getAlignment())>);
        static_assert(
            std::same_as<decltype(outerConstRebuiltView.getAlignment()), decltype(outerConstView.getAlignment())>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstBufferView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(outerConstBufferView[alpaka::Vec{0u, 0u}])>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstRebuiltView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(outerConstRebuiltView[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstRebuiltView.getExtents())>,
                      std::remove_cvref_t<decltype(innerConstView.getExtents())>>);
        static_assert(
            std::same_as<decltype(innerConstRebuiltView.getAlignment()), decltype(innerConstView.getAlignment())>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(innerConstRebuiltView.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(innerConstRebuiltView[alpaka::Vec{0u, 0u}])>>);
    }
}

TEST_CASE(
    "alpaka::makeView(api, pointer, extents, pitches, alignment) preserves explicit-pitch host metadata and inner "
    "constness",
    "[mem][view][correctness][makeView]")
{
    auto const extents = alpaka::Vec{2u, 3u, 4u};
    auto const explicitPitches = alpaka::Vec{80u, 20u, 4u};
    alignas(64) std::array<int, 2 * 3 * 5 * 2> mutableStorage{};
    alignas(64) std::array<int const, 2 * 3 * 5 * 2> innerConstStorage{};

    auto mutableSeedView
        = alpaka::makeView(alpaka::api::host, mutableStorage.data(), extents, alpaka::Alignment<64>{});
    auto const& outerConstSeedView = mutableSeedView;
    auto innerConstSeedView
        = alpaka::makeView(alpaka::api::host, innerConstStorage.data(), extents, alpaka::Alignment<64>{});

    auto mutableExplicitPitchView
        = alpaka::makeView(mutableSeedView, mutableStorage.data(), extents, explicitPitches, alpaka::Alignment<64>{});
    auto const outerConstExplicitPitchView = alpaka::makeView(
        outerConstSeedView,
        mutableStorage.data(),
        extents,
        explicitPitches,
        alpaka::Alignment<64>{});
    auto innerConstExplicitPitchView = alpaka::makeView(
        innerConstSeedView,
        innerConstStorage.data(),
        extents,
        explicitPitches,
        alpaka::Alignment<64>{});

    SECTION("compile-time explicit pitches stay part of the returned host view contract")
    {
        // The explicit-pitch overload should keep the host API, vector types, alignment, and mutability encoded by the
        // caller instead of rebuilding a packed view.
        static_assert(std::same_as<decltype(mutableExplicitPitchView.getApi()), alpaka::api::Host>);
        static_assert(std::same_as<decltype(outerConstExplicitPitchView.getApi()), alpaka::api::Host>);
        static_assert(std::same_as<decltype(innerConstExplicitPitchView.getApi()), alpaka::api::Host>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableExplicitPitchView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableExplicitPitchView.getPitches())>,
                      std::remove_cvref_t<decltype(explicitPitches)>>);
        static_assert(std::same_as<decltype(mutableExplicitPitchView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableExplicitPitchView.data())>>);
        static_assert(
            !std::is_const_v<std::remove_reference_t<decltype(mutableExplicitPitchView[alpaka::Vec{0u, 0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstExplicitPitchView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstExplicitPitchView.getPitches())>,
                      std::remove_cvref_t<decltype(explicitPitches)>>);
        static_assert(std::same_as<decltype(outerConstExplicitPitchView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstExplicitPitchView.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(outerConstExplicitPitchView[alpaka::Vec{0u, 0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstExplicitPitchView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstExplicitPitchView.getPitches())>,
                      std::remove_cvref_t<decltype(explicitPitches)>>);
        static_assert(std::same_as<decltype(innerConstExplicitPitchView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(innerConstExplicitPitchView.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(innerConstExplicitPitchView[alpaka::Vec{0u, 0u, 0u}])>>);
    }
}

TEST_CASE(
    "alpaka::makeView(api, pointer, extents, alignment) preserves packed host metadata and inner constness",
    "[mem][view][correctness][makeView]")
{
    auto const extents = alpaka::Vec{2u, 3u, 4u};
    auto const packedPitches = alpaka::calculatePitchesFromExtents<int>(extents);
    alignas(64) std::array<int, 2 * 3 * 4> mutableStorage{};
    alignas(64) std::array<int const, 2 * 3 * 4> innerConstStorage{};

    auto mutablePackedView
        = alpaka::makeView(alpaka::api::host, mutableStorage.data(), extents, alpaka::Alignment<64>{});
    auto const outerConstPackedView
        = alpaka::makeView(alpaka::api::host, mutableStorage.data(), extents, alpaka::Alignment<64>{});
    auto innerConstPackedView
        = alpaka::makeView(alpaka::api::host, innerConstStorage.data(), extents, alpaka::Alignment<64>{});

    SECTION("compile-time packed pitches stay part of the returned host view contract")
    {
        // The packed overload should derive the host pitch type from the extents while preserving alignment and
        // pointed-to constness.
        static_assert(std::same_as<decltype(mutablePackedView.getApi()), alpaka::api::Host>);
        static_assert(std::same_as<decltype(outerConstPackedView.getApi()), alpaka::api::Host>);
        static_assert(std::same_as<decltype(innerConstPackedView.getApi()), alpaka::api::Host>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutablePackedView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutablePackedView.getPitches())>,
                      std::remove_cvref_t<decltype(packedPitches)>>);
        static_assert(std::same_as<decltype(mutablePackedView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutablePackedView.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutablePackedView[alpaka::Vec{0u, 0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstPackedView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstPackedView.getPitches())>,
                      std::remove_cvref_t<decltype(packedPitches)>>);
        static_assert(std::same_as<decltype(outerConstPackedView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstPackedView.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(outerConstPackedView[alpaka::Vec{0u, 0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstPackedView.getExtents())>,
                      std::remove_cvref_t<decltype(extents)>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstPackedView.getPitches())>,
                      std::remove_cvref_t<decltype(packedPitches)>>);
        static_assert(std::same_as<decltype(innerConstPackedView.getAlignment()), alpaka::Alignment<64>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(innerConstPackedView.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(innerConstPackedView[alpaka::Vec{0u, 0u, 0u}])>>);
    }
}

TEST_CASE(
    "alpaka::makeMdSpan(any) preserves host shape alignment and inner constness",
    "[mem][mdspan][correctness][makeMdSpan]")
{
    auto mutableBuffer = alpaka::onHost::allocHost<int>(alpaka::Vec{2u, 3u});
    auto mutableView = mutableBuffer.getView();
    auto const& outerConstBuffer = mutableBuffer;
    auto const& outerConstView = mutableView;

    alignas(32) std::array<int const, 2 * 3> innerConstStorage{0, 1, 2, 3, 4, 5};
    auto innerConstView
        = alpaka::makeView(alpaka::api::host, innerConstStorage.data(), alpaka::Vec{2u, 3u}, alpaka::Alignment<32>{});

    auto mutableBufferMdSpan = alpaka::makeMdSpan(mutableBuffer);
    auto mutableViewMdSpan = alpaka::makeMdSpan(mutableView);
    auto outerConstBufferMdSpan = alpaka::makeMdSpan(outerConstBuffer);
    auto outerConstViewMdSpan = alpaka::makeMdSpan(outerConstView);
    auto innerConstViewMdSpan = alpaka::makeMdSpan(innerConstView);

    SECTION("compile-time rebuilding keeps host layout metadata and inner constness")
    {
        // `makeMdSpan(any)` should rebuild host buffers and views without changing shape, pitches, alignment, or the
        // pointed-to constness encoded by the source object.
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableBufferMdSpan.getExtents())>,
                      std::remove_cvref_t<decltype(mutableBuffer.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableBufferMdSpan.getPitches())>,
                      std::remove_cvref_t<decltype(mutableBuffer.getPitches())>>);
        static_assert(
            std::same_as<decltype(mutableBufferMdSpan.getAlignment()), decltype(mutableBuffer.getAlignment())>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableBufferMdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutableBufferMdSpan[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableViewMdSpan.getExtents())>,
                      std::remove_cvref_t<decltype(mutableView.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableViewMdSpan.getPitches())>,
                      std::remove_cvref_t<decltype(mutableView.getPitches())>>);
        static_assert(std::same_as<decltype(mutableViewMdSpan.getAlignment()), decltype(mutableView.getAlignment())>);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableViewMdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutableViewMdSpan[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstBufferMdSpan.getExtents())>,
                      std::remove_cvref_t<decltype(outerConstBuffer.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstBufferMdSpan.getPitches())>,
                      std::remove_cvref_t<decltype(outerConstBuffer.getPitches())>>);
        static_assert(
            std::same_as<decltype(outerConstBufferMdSpan.getAlignment()), decltype(outerConstBuffer.getAlignment())>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstBufferMdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(outerConstBufferMdSpan[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstViewMdSpan.getExtents())>,
                      std::remove_cvref_t<decltype(outerConstView.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(outerConstViewMdSpan.getPitches())>,
                      std::remove_cvref_t<decltype(outerConstView.getPitches())>>);
        static_assert(
            std::same_as<decltype(outerConstViewMdSpan.getAlignment()), decltype(outerConstView.getAlignment())>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(outerConstViewMdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(outerConstViewMdSpan[alpaka::Vec{0u, 0u}])>>);

        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstViewMdSpan.getExtents())>,
                      std::remove_cvref_t<decltype(innerConstView.getExtents())>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(innerConstViewMdSpan.getPitches())>,
                      std::remove_cvref_t<decltype(innerConstView.getPitches())>>);
        static_assert(
            std::same_as<decltype(innerConstViewMdSpan.getAlignment()), decltype(innerConstView.getAlignment())>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(innerConstViewMdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(innerConstViewMdSpan[alpaka::Vec{0u, 0u}])>>);
    }
}

TEST_CASE("View::getMdSpan keeps host mutability boundaries", "[mem][view][mdspan][correctness]")
{
    alignas(32) std::array<int, 2 * 3> storage{};
    auto mutableView
        = alpaka::makeView(alpaka::api::host, storage.data(), alpaka::Vec{2u, 3u}, alpaka::Alignment<32>{});
    auto mutableMdSpan = mutableView.getMdSpan();
    auto const constMdSpanFromOuterConst = std::as_const(mutableView).getMdSpan();
    auto const constMdSpanFromConstView = mutableView.getConstView().getMdSpan();

    SECTION("compile-time mutability follows the originating host view")
    {
        // `getMdSpan()` must keep mutable access only for mutable host views and lock read-only paths otherwise.
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(mutableMdSpan)>,
                      MdSpan<int, alpaka::Vec<uint32_t, 2>, alpaka::Vec<uint32_t, 2>, alpaka::Alignment<32>>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(constMdSpanFromOuterConst)>,
                      MdSpan<int const, alpaka::Vec<uint32_t, 2>, alpaka::Vec<uint32_t, 2>, alpaka::Alignment<32>>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(constMdSpanFromConstView)>,
                      MdSpan<int const, alpaka::Vec<uint32_t, 2>, alpaka::Vec<uint32_t, 2>, alpaka::Alignment<32>>>);

        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutableMdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mutableMdSpan[alpaka::Vec{0u, 0u}])>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(constMdSpanFromOuterConst.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(constMdSpanFromOuterConst[alpaka::Vec{0u, 0u}])>>);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(constMdSpanFromConstView.data())>>);
        static_assert(
            std::is_const_v<std::remove_reference_t<decltype(constMdSpanFromConstView[alpaka::Vec{0u, 0u}])>>);

        auto const readOnlyMdSpan = mutableMdSpan.getConstMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(readOnlyMdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(readOnlyMdSpan[alpaka::Vec{0u, 0u}])>>);
        static_assert(std::is_same_v<decltype(readOnlyMdSpan.getAlignment()), alpaka::Alignment<32>>);
    }
}
