//
// Created by Greg Miller on 8/23/16.
//

#ifndef TEST_HEAP_MERGE_HPP
#define TEST_HEAP_MERGE_HPP


#include <cmath>
#include <tbb/tbb.h>
#include <immintrin.h>
#include "utility.hpp"

namespace sal {
namespace merge {

struct auto_block_partition
{
    size_t operator()(size_t arr_size)
    {
        //TODO
        return std::max((size_t)8192, arr_size / 8);
    }
};

template<size_t BlockSizeDivider>
struct simple_block_partition
{
    size_t operator()(size_t arr_size)
    {
        return arr_size / BlockSizeDivider;
    }
};

template<size_t BlockSize>
struct static_block_partition
{
    size_t operator()(size_t arr_size)
    {
        return BlockSize;
    }
};

struct default_merger {
    template<typename InputIterator, typename OutputIterator, typename Comparator>
    void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                    OutputIterator out, Comparator cmp) {
        std::merge(first1, last1, first2, last2, out, cmp);
    }
};

template<typename T, typename Invoker = parallel_invoker, typename BlockMerger = default_merger, typename BlockPartition = auto_block_partition>
class merger {
public:

    merger() = delete;

    template<typename Comparator = std::less<T>>
    static void merge(const T* src1, long long p1, long long r1,
                      long long p2, long long r2, T* dest, long long p3, Comparator cmp = Comparator());

    template<typename Comparator = std::less<T>>
    static void merge(const T* src1, long long p1, long long r1,
                      const T* src2, long long p2, long long r2,
                      T* dest, long long p3, Comparator cmp = Comparator());

    template<typename InputIterator, typename OutputIterator, typename Comparator = std::less<T>>
    static void merge(InputIterator first, InputIterator mid, InputIterator last, OutputIterator out, Comparator cmp = Comparator());

    template<typename InputIterator, typename OutputIterator, typename Comparator = std::less<T>>
    static void merge(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                      OutputIterator out, Comparator cmp = Comparator());


private:
    template<typename Comparator>
    static void _dac_merge(const T *t, long long p1, long long r1,
                    const T *t2, long long p2, long long r2,
                    T *a, long long p3, Comparator cmp, size_t block_size, BlockMerger block_merger, Invoker invoker = Invoker());

};


template<typename T, typename Invoker, typename BlockMerger, typename BlockPartition>
template<typename Comparator>
void merger<T, Invoker, BlockMerger, BlockPartition>::merge(const T* src1, long long p1, long long r1,
                                                   long long p2, long long r2, T* dest, long long p3, Comparator cmp)
{
    long long n1 = r1 - p1 + 1;
    long long n2 = r2 - p2 + 1;
    long long n12 = n1+n2;

    if(n12 == 0)
        return;

    size_t block_size = BlockPartition()(n12);

    _dac_merge(src1, p1, r1, src1, p2, r2, dest, p3, cmp, block_size, BlockMerger());
}

template<typename T, typename Invoker, typename BlockMerger, typename BlockPartition>
template<typename Comparator>
void merger<T, Invoker, BlockMerger, BlockPartition>::merge(const T* src1, long long p1, long long r1,
                                                   const T* src2, long long p2, long long r2,
                                                   T* dest, long long p3, Comparator cmp)
{
    long long n1 = r1 - p1 + 1;
    long long n2 = r2 - p2 + 1;
    size_t block_size = BlockPartition()(n1+n2);

    _dac_merge(src1, p1, r1, src2, p2, r2, dest, p3, cmp, block_size, BlockMerger());
}

template<typename T, typename Invoker, typename BlockMerger, typename BlockPartition>
template<typename InputIterator, typename OutputIterator, typename Comparator>
void merger<T, Invoker, BlockMerger, BlockPartition>::merge(InputIterator first, InputIterator mid, InputIterator last,
                                                            OutputIterator out, Comparator cmp)
{
    static_assert(std::is_same<T, typename std::iterator_traits<InputIterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<InputIterator>::value && is_random_access_iterator<OutputIterator>::value,
                  "Iterators must be of random-access iterator type");

    long long a_size = std::distance(first, mid);
    long long b_size = std::distance(mid, last);

    size_t block_size = BlockPartition()(a_size+b_size);

    long long p1 = 0, r1 = a_size-1 , p2 = 0, r2 = b_size-1 , p3 = 0;
    auto t = iterator2pointer(first);
    auto t2 = iterator2pointer(mid);
    auto outp = iterator2pointer(out);

    _dac_merge(t, p1, r1, t2, p2, r2, outp, p3, cmp, block_size, BlockMerger());
};

template<typename T, typename Invoker, typename BlockMerger, typename BlockPartition>
template<typename InputIterator, typename OutputIterator, typename Comparator>
void merger<T, Invoker, BlockMerger, BlockPartition>::merge(InputIterator first1, InputIterator last1, InputIterator first2,
                                                   InputIterator last2, OutputIterator out, Comparator cmp) {

    static_assert(std::is_same<T, typename std::iterator_traits<InputIterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<InputIterator>::value && is_random_access_iterator<OutputIterator>::value,
                  "Iterators must be of random-access iterator type");

    long long a_size = std::distance(first1, last1);
    long long b_size = std::distance(first2, last2);

    size_t block_size = BlockPartition()(a_size+b_size);

    long long p1 = 0, r1 = a_size-1 , p2 = 0, r2 = b_size-1 , p3 = 0;
    auto t = iterator2pointer(first1);
    auto t2 = iterator2pointer(first2);
    auto outp = iterator2pointer(out);

    _dac_merge(t, p1, r1, t2, p2, r2, outp, p3, cmp, block_size, BlockMerger());
}


namespace internal {

namespace kernel {

inline void merge_avx2_8x8_32bit(__m256i &vA, __m256i &vB,
                                 __m256i &vMin, __m256i &vMax);

inline void merge_256i(const int *a, const int *b, int *res);

}


template<typename T>
struct is_int32 {
    static constexpr bool value = std::is_same<T, int>::value ||
                                  std::is_same<T, unsigned int>::value;
};

template<typename T>
struct is_default_merger_type {
    static constexpr bool value = !is_int32<T>::value;
};

template<typename Iterator, typename Comparator>
struct is_simd_enabled_comparator {
    static constexpr bool value = std::is_same<typename std::less<typename std::iterator_traits<Iterator>::value_type>, Comparator>::value;
};

inline int
sequential_simd_merge(const int *first1, const int *last1, const int *first2, const int *last2, int *res) {
    size_t a_size = last1 - first1;
    size_t b_size = last2 - first2;
    auto min_size = std::min(a_size, b_size);
    auto max_size = std::max(a_size, b_size);
    auto rem = min_size % 8;

    int *pres = &res[0];
    const int *pa = first1;
    const int *pb = first2;


    size_t rem_size = min_size - rem;
    if (rem_size >= 8) {
        for (size_t i = 0; i < rem_size; i += 8) {
            internal::kernel::merge_256i(pa, pb, pres);

            pres += 16;

            pa += 8;
            pb += 8;
        }
    }

    if (rem || max_size - min_size || rem_size < 8) {
        std::merge(pa, last1, pb, last2, pres);
    }

    return 0;
}

}

struct simd_int_merger {
    template<typename InputIterator, typename OutputIterator, typename Comparator>
    void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                    OutputIterator out, Comparator cmp) {
        static_assert(internal::is_simd_enabled_comparator<InputIterator, Comparator>::value,
                      "simd_int_merger doesn't support custom comparators");

        internal::sequential_simd_merge(first1, last1, first2, last2, out);
    }
};


struct auto_merger {
    template<typename InputIterator, typename OutputIterator, typename Comparator>
    typename std::enable_if<
            internal::is_int32<typename std::iterator_traits<InputIterator>::value_type>::value
    && internal::is_simd_enabled_comparator<InputIterator, Comparator>::value
    >::type operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
               OutputIterator out, Comparator cmp) {
        simd_int_merger()(first1, last1, first2, last2, out, cmp);
    }


    template<typename InputIterator, typename OutputIterator, typename Comparator>
    typename std::enable_if<
            internal::is_default_merger_type<typename std::iterator_traits<InputIterator>::value_type>::value ||
            !internal::is_simd_enabled_comparator<InputIterator, Comparator>::value
    >::type operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
               OutputIterator out, Comparator cmp) {
        std::merge(first1, last1, first2, last2, out, cmp);
    }

};


template<typename Merger, typename Partitioner>
struct merger_settings
{
    using merger_type = Merger;
    using partitioner_type = Partitioner;
};

using default_merger_settings = merger_settings<auto_merger, auto_block_partition>;

}}

#include "merge.cpp"

#endif //TEST_HEAP_MERGE_HPP
