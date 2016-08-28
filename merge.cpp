//
// Created by Greg Miller on 8/23/16.
//
#ifndef TEST_HEAP_MERGE_CPP
#define TEST_HEAP_MERGE_CPP
#include "merge.hpp"

namespace sal { namespace merge {

template<typename T, typename Invoker, typename BlockMerger, typename BlockPartition>
template<typename Comparator>
void merger<T, Invoker, BlockMerger, BlockPartition>::_dac_merge(const T *t, long long p1, long long r1,
                const T *t2, long long p2, long long r2,
                T *a, long long p3, Comparator cmp, size_t block_size, BlockMerger block_merger, Invoker invoker) {
    long long n1 = r1 - p1 + 1;
    long long n2 = r2 - p2 + 1;
    if (n1 < n2) {
        std::swap(t, t2);
        std::swap(p1, p2);
        std::swap(r1, r2);
        std::swap(n1, n2);
    }
    if (n1 == 0) return;

    if ((size_t)(n1 + n2) <= block_size) {
        block_merger(&t[p1], &t[p1 + n1], &t2[p2], &t2[p2 + n2], &a[p3], cmp);
    }
    else {

        long long q1 = (p1 + r1) / 2;
        long long q2 = binary_search(t[q1], t2, p2, r2);
        long long q3 = p3 + (q1 - p1) + (q2 - p2);
        a[q3] = t[q1];

        invoker(
        [&]{_dac_merge(t, p1, q1 - 1, t2, p2, q2 - 1, a, p3, cmp, block_size, block_merger, invoker);},
        [&]{_dac_merge(t, q1 + 1, r1, t2, q2, r2, a, q3 + 1, cmp, block_size, block_merger, invoker);}
        );
    }
}


/*
template<typename T, typename BlockMerger, typename BlockPartition>
void merger<T, BlockMerger, BlockPartition>::_parallel_dac_merge(const T *t, long long p1, long long r1,
                                const T *t2, long long p2, long long r2,
                                T *a, long long p3, size_t block_size, BlockMerger block_merger)
{
    long long n1 = r1 - p1 + 1;
    long long n2 = r2 - p2 + 1;
    if (n1 < n2) {
        std::swap(t, t2);
        std::swap(p1, p2);
        std::swap(r1, r2);
        std::swap(n1, n2);
    }
    if (n1 == 0) return;

    if (n1 + n2 <= block_size) {
        block_merger(&t[p1], &t[p1 + n1], &t2[p2], &t2[p2 + n2], &a[p3]);
    }
    else {

        long long q1 = (p1 + r1) / 2;
        long long q2 = binary_search(t[q1], t2, p2, r2);
        long long q3 = p3 + (q1 - p1) + (q2 - p2);
        a[q3] = t[q1];
        tbb::parallel_invoke(
                [&]{_dac_merge(t, p1, q1 - 1, t2, p2, q2 - 1, a, p3, block_size, block_merger);},
                [&]{_dac_merge(t, q1 + 1, r1, t2, q2, r2, a, q3 + 1, block_size, block_merger);}
        );
    }
}
*/

namespace internal {
namespace kernel
{
inline void merge_avx2_8x8_32bit(__m256i &vA, __m256i &vB, // input
                                 __m256i &vMin, __m256i &vMax) { // output
    __m256i vTmp;

    //TODO doesnt work properly
    //pass 1
    vTmp = _mm256_min_epu32(vA, vB);
    vMax = _mm256_max_epu32(vA, vB);
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

inline void reverse_merge_avx2_8x8_32bit(__m256i &vA, __m256i &vB, // input
                                 __m256i &vMin, __m256i &vMax) { // output
    __m256i vTmp;

    using utils::_print_register;

    _print_register(vA, "vA");
    _print_register(vB, "vB");

    //pass 1
    vTmp = _mm256_max_epu32(vA, vB);
    vMax = _mm256_min_epu32(vB, vB);

    _print_register(vTmp, "vMin");
    _print_register(vMax, "vMax");

    vTmp = _mm256_alignr_epi8(vTmp, vTmp, 4);

    _print_register(vTmp, "vMax");

    //pass 2
    vMin = _mm256_min_epu32(vTmp, vMax);
    vMax = _mm256_max_epu32(vTmp, vMax);

    _print_register(vMin, "vMin");
    _print_register(vMax, "vMax");

    vTmp = _mm256_alignr_epi8(vMax, vMax, 4);

    _print_register(vTmp, "vTmp");

    //pass 3
    vMin = _mm256_min_epu32(vTmp, vMax);
    vMax = _mm256_max_epu32(vTmp, vMax);

    _print_register(vMin, "vMin");
    _print_register(vMax, "vMax");

    vTmp = _mm256_alignr_epi8(vMax, vMax, 4);

    _print_register(vTmp, "vTmp");

    //pass 4
    vMin = _mm256_min_epu32(vTmp, vMax);
    vMax = _mm256_max_epu32(vTmp, vMax);

    _print_register(vMin, "vMin");
    _print_register(vMax, "vMax");

    vMax = _mm256_alignr_epi8(vMax, vMax, 4);

    _print_register(vMax, "vMax");

    __m128i pm1 = _mm256_extractf128_si256(vMin, 0);
    __m128i pm2 = _mm256_extractf128_si256(vMax, 0);

    __m128i pmx1 = _mm256_extractf128_si256(vMin, 1);
    __m128i pmx2 = _mm256_extractf128_si256(vMax, 1);


    vMin = _mm256_set_m128i(pm2, pm1);
    vMax = _mm256_set_m128i(pmx2, pmx1);
}


inline void reverse_merge_256i(const int *a, const int *b, int *res) {
    __m256i m1 = _mm256_load_si256((__m256i *) a);
    __m256i m2 = _mm256_load_si256((__m256i *) b);
    __m256i vmin, vmax;

    reverse_merge_avx2_8x8_32bit(m1, m2, vmin, vmax);

    _mm256_store_si256((__m256i *) res, vmin);
    _mm256_store_si256((__m256i *) &res[8], vmax);
}


inline void merge_4x4_32bit(__m128i &vA, __m128i &vB, // input 1 & 2
                            __m128i &vMin, __m128i &vMax) { // output
    __m128i vTmp; // temporary register
    vTmp = _mm_min_epu32(vA, vB);
    vMax = _mm_max_epu32(vA, vB);
    vTmp = _mm_alignr_epi8(vTmp, vTmp, 4);
    vMin = _mm_min_epu32(vTmp, vMax);
    vMax = _mm_max_epu32(vTmp, vMax);
    vTmp = _mm_alignr_epi8(vMin, vMin, 4);
    vMin = _mm_min_epu32(vTmp, vMax);
    vMax = _mm_max_epu32(vTmp, vMax);
    vTmp = _mm_alignr_epi8(vMin, vMin, 4);
    vMin = _mm_min_epu32(vTmp, vMax);
    vMax = _mm_max_epu32(vTmp, vMax);
    vMin = _mm_alignr_epi8(vMin, vMin, 4);
}
inline void merge_8x8_32bit(__m128i &vA0, __m128i &vA1, // input 1
        __m128i &vB0, __m128i &vB1, // input 2
        __m128i &vMin0, __m128i &vMin1, // output
        __m128i &vMax0, __m128i &vMax1) { // output
// 1st step
merge_4x4_32bit(vA1,vB1,vMin1,vMax1);
merge_4x4_32bit(vA0,vB0,vMin0,vMax0);
// 2nd step
merge_4x4_32bit(vMax0,vMin1,vMin1,vMax0);
}


inline void merge_8x8_128i(const int *a, const int *b, int *res) {

//    utils::print_array(a, 8, "a");
//    utils::print_array(b, 8, "b");

    __m128i a1 = _mm_loadu_si128((__m128i *) a);
    __m128i a2 = _mm_loadu_si128((__m128i *) &a[4]);
    __m128i b1 = _mm_loadu_si128((__m128i *) b);
    __m128i b2 = _mm_loadu_si128((__m128i *) &b[4]);
    __m128i vmin1, vmin2, vmax1, vmax2;

//    utils::_print_register128(a1, "a1");
//    utils::_print_register128(a2, "a2");
//    utils::_print_register128(b1, "b1");
//    utils::_print_register128(b2, "b2");
    merge_8x8_32bit(a1, a2, b1, b2, vmin1, vmin2, vmax1, vmax2);

//    utils::_print_register128(vmin1, "vmin1");
//    utils::_print_register128(vmin2, "vmin2");
//    utils::_print_register128(vmax1, "vmax1");
//    utils::_print_register128(vmax2, "vmax2");

    _mm_storeu_si128((__m128i *) res, vmin1);
    _mm_storeu_si128((__m128i *) &res[4], vmin2);
    _mm_storeu_si128((__m128i *) &res[8], vmax1);
    _mm_storeu_si128((__m128i *) &res[12], vmax2);

    //utils::print_array(res, 16, "res");
}



}

}}}

#endif