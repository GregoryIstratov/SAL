//
// Created by Greg Miller on 8/25/16.
//

#ifndef SAL_SORT_HPP
#define SAL_SORT_HPP

#include "merge.hpp"

//#define SORT_DEBUG_VERBOSE 1

namespace sal{ namespace sort {

using namespace merge;

template<typename T, size_t block_size = 8192, typename MergerSettings = default_merger_settings>
class sorter {
public:
    using merger_type = merger<T, typename MergerSettings::merger_type, typename MergerSettings::partitioner_type>;

    template<typename Iterator>
    static void parallel_merge_sort(Iterator first, Iterator last, Iterator out);

    template<typename Iterator>
    static void merge_sort(Iterator first, Iterator last, Iterator out);


private:
    template<typename Invoker>
    static void _merge_sort_common(T* src, size_t l, size_t r, T* dest, bool src2dest = true, Invoker invoker = Invoker(), int height = 0);

};


template<typename T, size_t block_size, typename MergerSettings>
template<typename Iterator>
void sorter<T, block_size, MergerSettings>::merge_sort(Iterator first, Iterator last, Iterator out)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    auto src = iterator2pointer(first);
    auto outp = iterator2pointer(out);

    _merge_sort_common(src, 0, n-1, outp, true, serial_invoker());
}

template<typename T, size_t block_size, typename MergerSettings>
template<typename Iterator>
void sorter<T, block_size, MergerSettings>::parallel_merge_sort(Iterator first, Iterator last, Iterator out)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    auto src = iterator2pointer(first);
    auto outp = iterator2pointer(out);

    _merge_sort_common(src, 0, n-1, outp, true, parallel_invoker());
}


template<typename T, size_t block_size, typename MergerSettings>
template<typename Invoker>
void sorter<T, block_size, MergerSettings>::_merge_sort_common(T* src, size_t l, size_t r, T* dest, bool src2dest, Invoker invoker, int height)
{
#ifdef SORT_DEBUG_VERBOSE
    print_seq("[Full Input] ", height, src, 0, 34);
    print_seq("[Out]", height, dest, 0, 34);
    print_seq("[Input] ", height, src, l, r);
#endif
    if(r == l)
    {
        if(src2dest)
            dest[l] = src[l];

        return;
    }

    if((r-l) <= block_size && !src2dest)
    {
        std::sort(src+l, src+r+1);
#ifdef SORT_DEBUG_VERBOSE
        print_seq("[Sort out] ", height, src, l, r);
#endif
        return;
    }

    size_t m = (r + l) / 2;

    invoker(
            [&]{_merge_sort_common(src, l, m, dest, !src2dest, invoker, height+1);},
            [&]{_merge_sort_common(src, m +1, r, dest, !src2dest, invoker, height+1);}
    );


    if(src2dest)
    {
#ifdef SORT_DEBUG_VERBOSE
        print_seq("[Merge Input A]", height, src , l, m);
        print_seq("[Merge Input B]", height, src , m+1, r);
#endif
        merger_type::parallel_merge(src, l, m, m+1, r, dest, l);

#ifdef SORT_DEBUG_VERBOSE
        print_seq("[Merge Out]", height, dest, 0, 34);
#endif
    }
    else
    {
#ifdef SORT_DEBUG_VERBOSE
        print_seq("[Merge Input A]", height, dest , l, m);
        print_seq("[Merge Input B]", height, dest , m+1, r);
#endif
        merger_type::parallel_merge(dest, l, m, m+1, r, src, l);

#ifdef SORT_DEBUG_VERBOSE
        print_seq("[Merge Out]", height, src, 0, 34);
#endif
    }
}

}}

#endif //SAL_SORT_HPP
