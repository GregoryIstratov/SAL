//
// Created by Greg Miller on 8/23/16.
//

#ifndef TEST_HEAP_MERGE_HPP
#define TEST_HEAP_MERGE_HPP


#include <tbb/tbb.h>
#include <immintrin.h>
#include "utility.hpp"

template<typename T>
struct default_merger
{
    using InputIterator = const T*;
    using OutputIterator = T*;

    void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2, OutputIterator out)
    {
        std::merge(first1, last1, first2, last2, out);
    }
};

// Merge two ranges of source array T[ p1 .. r1 ] and T[ p2 .. r2 ] into destination array A starting at index p3.
// From 3rd ed. of "Introduction to Algorithms" p. 798-802
// Listing 2
template< typename T, typename BlockMergeFun = default_merger<T>>
inline void merge_dac_common( const T* t, size_t p1, size_t r1, size_t p2, size_t r2, T* a, size_t p3, size_t threshold,
                              BlockMergeFun fun = BlockMergeFun())
{
    size_t length1 = r1 - p1 + 1;
    size_t length2 = r2 - p2 + 1;
    if ( length1 < length2 )
    {
        std::swap(      p1,      p2 );
        std::swap(      r1,      r2 );
        std::swap( length1, length2 );
    }
    if ( length1 == 0 ) return;
    if(length1+length2 <= threshold)
    {
        fun(&t[p1], &t[p1+length1], &t[p2], &t[p2+length2], &a[p3]);
    }
    else {
        size_t q1 = (p1 + r1) / 2;
        size_t q2 = binary_search(t[q1], t, p2, r2);
        size_t q3 = p3 + (q1 - p1) + (q2 - p2);
        a[q3] = t[q1];
        merge_dac_common(t, p1, q1 - 1, p2, q2 - 1, a, p3, threshold, fun);
        merge_dac_common(t, q1 + 1, r1, q2, r2, a, q3 + 1, threshold, fun);
    }
}

template< typename T, typename BlockMergeFun = default_merger<T>>
inline void parallel_merge_dac_common( const T* t, size_t p1, size_t r1, size_t p2, size_t r2, T* a, size_t p3, size_t threshold,
                              BlockMergeFun fun = BlockMergeFun())
{
    size_t length1 = r1 - p1 + 1;
    size_t length2 = r2 - p2 + 1;
    if ( length1 < length2 )
    {
        std::swap(      p1,      p2 );
        std::swap(      r1,      r2 );
        std::swap( length1, length2 );
    }
    if ( length1 == 0 ) return;
    if(length1+length2 <= threshold)
    {
        fun(&t[p1], &t[p1+length1], &t[p2], &t[p2+length2], &a[p3]);
    }
    else {
        size_t q1 = (p1 + r1) / 2;
        size_t q2 = binary_search(t[q1], t, p2, r2);
        size_t q3 = p3 + (q1 - p1) + (q2 - p2);
        a[q3] = t[q1];
        tbb::parallel_invoke(
                [&]{ parallel_merge_dac_common(t, p1, q1 - 1, p2, q2 - 1, a, p3, threshold, fun); },
                [&]{ parallel_merge_dac_common(t, q1 + 1, r1, q2, r2, a, q3 + 1, threshold, fun); }
        );
    }
}

template<typename T>
inline void merge_dac(const aligned_vector<T>& v, size_t mid, aligned_vector<T>& res, size_t block_dev = 8)
{
    size_t threshold = std::max((size_t)2, v.size() / block_dev);
    merge_dac_common(v.data(), 0, mid-1, mid, v.size()-1, res.data(), 0, threshold);
}

template<typename T>
inline void parallel_merge_dac(const aligned_vector<T>& v, size_t mid, aligned_vector<T>& res, size_t block_dev = 8)
{
    size_t threshold = std::max((size_t)2, v.size() / block_dev);
    parallel_merge_dac_common(v.data(), 0, mid-1, mid, v.size()-1, res.data(), 0, threshold);
}

namespace kernel {

    inline void merge_avx2_8x8_32bit(__m256i &vA, __m256i &vB, // input
                                            __m256i &vMin, __m256i &vMax) { // output
        __m256i vTmp;

        //pass 1
        vTmp = _mm256_min_epu32(vA, vB);
        vMax = _mm256_max_epu32(vB, vB);
        vTmp = _mm256_alignr_epi8(vTmp, vTmp, 4);

        //pass 2
        vMin = _mm256_min_epu32(vTmp, vMax);
        vMax = _mm256_max_epu32(vTmp, vMax);
        vTmp = _mm256_alignr_epi8(vMin, vMin, 4);

        //pass 3
        vMin = _mm256_min_epu32(vTmp, vMax);
        vMax = _mm256_max_epu32(vTmp, vMax);
        vTmp = _mm256_alignr_epi8(vMin, vMin, 4);

        //pass 4
        vMin = _mm256_min_epu32(vTmp, vMax);
        vMax = _mm256_max_epu32(vTmp, vMax);
        vMin = _mm256_alignr_epi8(vMin, vMin, 4);

        __m128i pm1 = _mm256_extractf128_si256(vMin, 0);
        __m128i pm2 = _mm256_extractf128_si256(vMax, 0);

        __m128i pmx1 = _mm256_extractf128_si256(vMin, 1);
        __m128i pmx2 = _mm256_extractf128_si256(vMax, 1);


        vMin = _mm256_set_m128i(pm2, pm1);
        vMax = _mm256_set_m128i(pmx2, pmx1);
    }


    inline void merge_256i(const int *a, const int *b, int *res) {
        __m256i m1 = _mm256_load_si256((__m256i *) a);
        __m256i m2 = _mm256_load_si256((__m256i *) b);
        __m256i vmin, vmax;

        merge_avx2_8x8_32bit(m1, m2, vmin, vmax);

        _mm256_store_si256((__m256i *) res, vmin);
        _mm256_store_si256((__m256i *) &res[8], vmax);
    }

}

inline int sequential_simd_merge(const int* a, size_t a_size, const int* b, size_t b_size, int* res)
{
    auto min_size = std::min(a_size, b_size);
    auto max_size = std::min(a_size, b_size);
    auto rem = min_size % 8;

    int i = 0;
    int* pres = &res[0];

    size_t rem_size = min_size - rem;
    for(;;)
    {
        if(i >= rem_size)
            break;


        const int* pa = &a[i];
        const int* pb = &b[i];

        kernel::merge_256i(pa, pb, pres);

        pres+=16;

        i += 8;
    }

    if(rem || max_size - min_size)
    {
        std::merge(&a[i], &a[a_size], &b[i], &b[b_size], pres);
    }

    return 0;
}

inline int sequential_simd_merge(const int* first1, const int* last1, const int* first2, const int* last2, int* res)
{
    size_t a_size = std::distance(first1, last1);
    size_t b_size = std::distance(first2, last2);
    return sequential_simd_merge(first1, a_size, first2, b_size, res);
}

struct simd_int_merger
{
    using InputIterator = const int*;
    using OutputIterator = int*;

    void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2, OutputIterator out)
    {
        sequential_simd_merge(first1, last1, first2, last2, out);
    }
};

inline void merge_simd_dac(const aligned_vector<int> &v, size_t mid, aligned_vector<int> &res, size_t block_dev = 8)
{
    size_t threshold = std::max((size_t)2, v.size() / block_dev);
    merge_dac_common<int, simd_int_merger>(v.data(), 0, mid-1, mid, v.size()-1, res.data(), 0, threshold);
}

inline void parallel_merge_simd_dac(const aligned_vector<int> &v, size_t mid, int *res, size_t block_div = 8)
{
    size_t threshold = std::max((size_t)2, v.size() / block_div);
    parallel_merge_dac_common<int, simd_int_merger>(v.data(), 0, mid-1, mid, v.size()-1, res, 0, threshold);
}

#endif //TEST_HEAP_MERGE_HPP
