//
// Created by Greg Miller on 8/23/16.
//

#ifndef TEST_HEAP_UTILITY_HPP
#define TEST_HEAP_UTILITY_HPP

#include <cmath>
#include <algorithm>
#include <type_traits>
#include <tbb/tbb.h>
#include <future>
#include "aligned_allocator.hpp"

namespace sal { namespace utils {
struct parallel_invoker {
    template<typename ...Funs>
    void operator()(Funs... funs) {
        tbb::parallel_invoke(funs...);
    }
};

struct serial_invoker {
    template<typename ...Funs>
    void operator()(Funs... funs) {
        invoke(funs...);
    }

private:
    template<typename Fun>
    void invoke(Fun &fun) {
        fun();
    }

    template<typename Fun, typename ...Funs>
    void invoke(Fun &fun, Funs... funs) {
        fun();
        invoke(funs...);
    }

};

template<typename RandomAccessIterator>
typename std::iterator_traits<RandomAccessIterator>::value_type *
iterator2pointer(RandomAccessIterator it) {
    return &(*it);
}


template<typename T>
struct is_random_access_iterator : std::is_same<
        typename std::iterator_traits<T>::iterator_category, std::random_access_iterator_tag> {
};

template<typename T, typename InputIterator>
InputIterator binary_search(const T &x, InputIterator first, InputIterator last) {

    //if empty return first
    if (first == last)
        return first;


    if (x <= *first)
        return first;

    //if x > *first return q -> (first < q <= last) that *(q-1) < x

    //the value pointed by the iterator returned by this function cannot be equivalent to x, only greater, *q  > x
    auto q = std::upper_bound(first, last, x);
    if (q != last) {
        return q;
    }

    return last;
}


template<typename InputIterator>
void print_seq(const char *label, InputIterator first, InputIterator last) {
    std::cout << label << "|" << std::distance(first, last) << "|: ";
    std::copy(first, last,
              std::ostream_iterator<typename std::iterator_traits<InputIterator>::value_type>(std::cout, " "));
    std::cout << std::endl;
}

template<typename InputIterator>
void print_seq(const char *label, int height, InputIterator first, InputIterator last) {
    std::cout << "|---";
    for (int i = 0; i < height; ++i) {
        std::cout << "|---";
    }
    std::cout << label << "|" << std::distance(first, last) << "|: ";
    std::copy(first, last,
              std::ostream_iterator<typename std::iterator_traits<InputIterator>::value_type>(std::cout, " "));
    std::cout << std::endl;
}

template<typename T>
void print_seq(const char *label, int height, const T *a, long long l, long long r) {
    std::cout << "|---";
    for (int i = 0; i < height; ++i) {
        std::cout << "|---";
    }
    std::cout << label << "|" << r - l << "|: ";
    if (r - l + 1) for (long long i = l; i <= r; ++i) std::cout << a[i] << " ";
    std::cout << std::endl;
}

// This version is borrowed from "Introduction to Algorithms" 3rd edition, p. 799.
template<class T>
inline size_t binary_search(T value, const T *a, size_t left, size_t right) {
    size_t low = left;
    size_t high = std::max(left, right + 1);
    while (low < high) {
        size_t mid = (low + high) / 2;
        if (value <= a[mid]) high = mid;
        else low = mid + 1; // because we compared to a[mid] and the value was larger than a[mid].
        // Thus, the next array element to the right from mid is the next possible
        // candidate for low, and a[mid] can not possibly be that candidate.
    }
    return high;
}

inline void _print_register(const __m256i &m, const char *label) {
    int v[8] = {};
    _mm256_storeu_si256((__m256i *) v, m);
    std::cout << label << ": { ";

    for (int i : v) {
        std::cout << i << ", ";
    }

    std::cout << " }" << std::endl;
}

inline void _print_register128(const __m128i &m, const char *label) {
    int v[4] = {};
    _mm_storeu_si128((__m128i *) v, m);
    std::cout << label << ": { ";

    for (int i : v) {
        std::cout << i << ", ";
    }

    std::cout << " }" << std::endl;
}

}}

#endif //TEST_HEAP_UTILITY_HPP
