//
// Created by Greg Miller on 8/23/16.
//

#ifndef TEST_HEAP_MERGE_HPP
#define TEST_HEAP_MERGE_HPP


#include <tbb/tbb.h>
#include <immintrin.h>
#include "utility.hpp"

namespace merge {


    struct auto_block_partition
    {
        size_t operator()(size_t arr_size)
        {
            return arr_size / 8;
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
        template<typename InputIterator, typename OutputIterator>
        void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                        OutputIterator out) {
            std::merge(first1, last1, first2, last2, out);
        }
    };

    template<typename T, typename BlockMerger = default_merger, typename BlockPartition = auto_block_partition>
    class merger {
    public:

        merger() = delete;

        template<typename InputIterator, typename OutputIterator>
        static void merge(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                          OutputIterator out);

        template<typename InputIterator, typename OutputIterator>
        static void parallel_merge(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                                   OutputIterator out);


    private:
        template<typename InputIterator, typename OutputIterator>
        static void
        _merge_dac_common(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                          OutputIterator out, size_t threshold,
                          BlockMerger merger);

        template<typename InputIterator, typename OutputIterator>
        static void
        _parallel_merge_dac_common(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                                   OutputIterator out, size_t threshold,
                                   BlockMerger merger);


        template<typename InputIterator>
        static inline void print_seq(const char *label, InputIterator first, InputIterator last);

    };


    template<typename T, typename BlockMerger, typename BlockPartition>
    template<typename InputIterator, typename OutputIterator>
    void merger<T, BlockMerger, BlockPartition>::merge(InputIterator first1, InputIterator last1, InputIterator first2,
                                                  InputIterator last2,
                                                  OutputIterator out) {

        static_assert(std::is_same<T, typename std::iterator_traits<InputIterator>::value_type>::value,
                      "Specified type must be the same as type of iterators");

        size_t t_size = std::distance(first1, last1) + std::distance(first2, last2);

        size_t block_size = BlockPartition()(t_size);

        _merge_dac_common(first1, last1, first2, last2, out, block_size, BlockMerger());
    }

    template<typename T, typename BlockMerger, typename BlockPartition>
    template<typename InputIterator, typename OutputIterator>
    void merger<T, BlockMerger, BlockPartition>::_merge_dac_common(InputIterator first1, InputIterator last1,
                                                              InputIterator first2, InputIterator last2,
                                                              OutputIterator out, size_t threshold,
                                                              BlockMerger merger) {

        size_t length1 = last1 - first1;
        size_t length2 = last2 - first2;
        if (length1 < length2) {
            first2 = first1;
            last2 = last1;
            std::swap(length1, length2);
        }
        if (length1 == 0) return;
        if (length1 + length2 <= threshold) {
            merger(first1, last1, first2, last2, out);
        }
        else {
            InputIterator q1 = first1 + length1 / 2;
            InputIterator q2 = binary_search(*q1, first2, last2);
            OutputIterator q3 = out + (q1 - first1) + (q2 - first2);
            *q3 = *q1;
            _merge_dac_common(first1, q1, first2, q2, out, threshold, merger);
            _merge_dac_common(q1, last1, q2, last2, q3, threshold, merger);
        }
    }

    template<typename T, typename BlockMerger, typename BlockPartition>
    template<typename InputIterator, typename OutputIterator>
    void
    merger<T, BlockMerger, BlockPartition>::parallel_merge(InputIterator first1, InputIterator last1, InputIterator first2,
                                                      InputIterator last2,
                                                      OutputIterator out) {

        static_assert(std::is_same<T, typename std::iterator_traits<InputIterator>::value_type>::value,
                      "Specified type must be the same as type of iterators");

        size_t t_size = std::distance(first1, last1) + std::distance(first2, last2);

        size_t block_size = BlockPartition()(t_size);

        _parallel_merge_dac_common(first1, last1, first2, last2, out, block_size, BlockMerger());
    }

    template<typename T, typename BlockMerger, typename BlockPartition>
    template<typename InputIterator, typename OutputIterator>
    void merger<T, BlockMerger, BlockPartition>::_parallel_merge_dac_common(InputIterator first1, InputIterator last1,
                                                                       InputIterator first2, InputIterator last2,
                                                                       OutputIterator out, size_t threshold,
                                                                       BlockMerger merger) {
        size_t length1 = last1 - first1;
        size_t length2 = last2 - first2;
        if (length1 < length2) {
            first2 = first1;
            last2 = last1;
            std::swap(length1, length2);
        }
        if (length1 == 0) return;
        if (length1 + length2 <= threshold) {
            merger(first1, last1, first2, last2, out);
        }
        else {
            InputIterator q1 = first1 + length1 / 2;
            InputIterator q2 = binary_search(*q1, first2, last2);
            OutputIterator q3 = out + (q1 - first1) + (q2 - first2);
            *q3 = *q1;
            tbb::parallel_invoke(
                    [&] { _parallel_merge_dac_common(first1, q1, first2, q2, out, threshold, merger); },
                    [&] { _parallel_merge_dac_common(q1, last1, q2, last2, q3, threshold, merger); }
            );
        }
    }

    template<typename T, typename BlockMerger, typename BlockPartition>
    template<typename InputIterator>
    void merger<T, BlockMerger, BlockPartition>::print_seq(const char *label, InputIterator first, InputIterator last) {
        std::cout << label << " - ";
        while (first != last) {
            std::cout << *first << ", ";
            ++first;
        }

        std::cout << std::endl;
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
                kernel::merge_256i(pa, pb, pres);

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

    struct simd_int_merger {
        template<typename InputIterator, typename OutputIterator>
        void operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                        OutputIterator out) {
            sequential_simd_merge(first1, last1, first2, last2, out);
        }
    };


    template<typename T>
    struct is_int32 {
        static constexpr bool value = std::is_same<T, int>::value ||
                                      std::is_same<T, unsigned int>::value;
    };

    template<typename T>
    struct is_default_merger_type {
        static constexpr bool value = !is_int32<T>::value;
    };

    struct auto_merger {
        template<typename InputIterator, typename OutputIterator>
        typename std::enable_if<is_int32<typename std::iterator_traits<InputIterator>::value_type>::value>::type
        operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                   OutputIterator out) {
            simd_int_merger()(first1, last1, first2, last2, out);
        }


        template<typename InputIterator, typename OutputIterator>
        typename std::enable_if<is_default_merger_type<typename std::iterator_traits<InputIterator>::value_type>::value>::type
        operator()(InputIterator first1, InputIterator last1, InputIterator first2, InputIterator last2,
                   OutputIterator out) {
            std::merge(first1, last1, first2, last2, out);
        }

    };

}

#endif //TEST_HEAP_MERGE_HPP
