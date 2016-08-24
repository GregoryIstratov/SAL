//
// Created by Greg Miller on 8/23/16.
//

#ifndef TEST_HEAP_UTILITY_HPP
#define TEST_HEAP_UTILITY_HPP

#include <cmath>
#include <algorithm>
#include <type_traits>
#include "aligned_allocator.hpp"

template <typename T>
struct is_random_access_iterator : std::is_same<
        typename std::iterator_traits<T>::iterator_category
        , std::random_access_iterator_tag>
{};

template<typename T, typename InputIterator>
InputIterator binary_search(const T &value, InputIterator first, InputIterator last) {
    first = std::lower_bound(first, last, value);
    if (first != last) {
        return first;
    }

    return last;
}

// This version is borrowed from "Introduction to Algorithms" 3rd edition, p. 799.
template< class T >
inline size_t binary_search(T value, const T *a, size_t left, size_t right)
{
    size_t low  = left;
    size_t high = std::max( left, right + 1 );
    while( low < high )
    {
        size_t mid = ( low + high ) / 2;
        if ( value <= a[ mid ] ) high = mid;
        else                        low  = mid + 1; // because we compared to a[mid] and the value was larger than a[mid].
        // Thus, the next array element to the right from mid is the next possible
        // candidate for low, and a[mid] can not possibly be that candidate.
    }
    return high;
}

inline void _print_register(const __m256i& m, const char* label)
{
    int v[8] = {};
    _mm256_storeu_si256((__m256i*)v, m);
    std::cout<<label<<": { ";

    for(int i : v)
    {
        std::cout<<i<<", ";
    }

    std::cout<<" }"<<std::endl;
}

inline void _print_register128(const __m128i& m, const char* label)
{
    int v[4] = {};
    _mm_storeu_si128((__m128i*)v, m);
    std::cout<<label<<": { ";

    for(int i : v)
    {
        std::cout<<i<<", ";
    }

    std::cout<<" }"<<std::endl;
}


#endif //TEST_HEAP_UTILITY_HPP
