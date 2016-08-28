//
// Created by Greg Miller on 8/25/16.
//

#ifndef SAL_SORT_HPP
#define SAL_SORT_HPP

#include "merge.hpp"

//#define SORT_DEBUG_VERBOSE 1

namespace sal{ namespace sort {

using namespace merge;

namespace internal {

struct stable_block_sorter
{
    template<typename Iterator, typename Comparator>
    void operator()(Iterator first, Iterator last, Comparator cmp)
    {
        std::stable_sort(first, last, cmp);
    }
};


struct unstable_block_sorter
{
    template<typename Iterator, typename Comparator>
    void operator()(Iterator first, Iterator last, Comparator cmp)
    {
        std::sort(first, last, cmp);
    }
};

template<typename T, typename Comparator = std::less<T>>
inline void insertion_sort(T* a, size_t a_size, Comparator cmp = Comparator())
{
    for ( size_t i = 1; i < a_size; i++ )
    {
        if ( a[ i ] < a[ i - 1 ] )       // no need to do (j > 0) compare for the first iteration
        {
            T currentElement = a[ i ];
            a[ i ] = a[ i - 1 ];
            size_t j;
            for ( j = i - 1; j > 0 && cmp(currentElement, a[ j - 1 ]); j-- )
            {
                a[ j ] = a[ j - 1 ];
            }
            a[ j ] = currentElement;    // always necessary work/write
        }
        // Perform no work at all if the first comparison fails - i.e. never assign an element to itself!
    }
}

struct insertion_sorter
{
    template<typename Iterator, typename Comparator>
    void operator()(Iterator first, Iterator last, Comparator cmp)
    {
        size_t len = std::distance(first, last);
        auto a = utils::iterator2pointer(first);
        insertion_sort(a, len, cmp );
    }
};

}

template<typename T, typename Invoker = parallel_invoker, size_t block_size = 8192, typename MergerSettings = default_merger_settings>
class sorter {
public:
    using merger_type = merger<T, Invoker, typename MergerSettings::merger_type, typename MergerSettings::partitioner_type>;

    template<typename Iterator, typename Comparator = std::less<int>>
    static void merge_sort(Iterator first, Iterator last, Comparator cmp = Comparator());

    template<typename Iterator, typename Comparator = std::less<int>>
    static void merge_sort(Iterator first, Iterator last, Iterator out, Comparator cmp = Comparator());

    template<typename Iterator, typename Comparator = std::less<int>>
    static void stable_merge_sort(Iterator first, Iterator last, Iterator out, Comparator cmp = Comparator());

    template<typename Iterator, typename Comparator = std::less<int>>
    static void stable_merge_sort(Iterator first, Iterator last, Comparator cmp = Comparator());


private:
    template<typename BlockSorter, typename Comparator>
    static void _merge_sort_common(T* src, size_t l, size_t r, T* dest, bool src2dest, Comparator cmp,
                                   BlockSorter block_sorter = BlockSorter(), Invoker invoker = Invoker());

};

template<typename T, typename Invoker, size_t block_size, typename MergerSettings>
template<typename Iterator, typename Comparator>
void sorter<T, Invoker, block_size, MergerSettings>::merge_sort(Iterator first, Iterator last, Comparator cmp)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    if(n < 2)
        return;

    aligned_vector<T> tmp_buffer;
    tmp_buffer.resize(n);

    auto src = utils::iterator2pointer(first);

    _merge_sort_common(src, 0, n-1, tmp_buffer.data(), false, cmp, internal::unstable_block_sorter());
}

template<typename T, typename Invoker, size_t block_size, typename MergerSettings>
template<typename Iterator, typename Comparator>
void sorter<T, Invoker, block_size, MergerSettings>::merge_sort(Iterator first, Iterator last, Iterator out, Comparator cmp)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    auto src = utils::iterator2pointer(first);
    auto outp = utils::iterator2pointer(out);

    _merge_sort_common(src, 0, n-1, outp, true, cmp, internal::unstable_block_sorter());
}

template<typename T, typename Invoker, size_t block_size, typename MergerSettings>
template<typename Iterator, typename Comparator>
void sorter<T, Invoker, block_size, MergerSettings>::stable_merge_sort(Iterator first, Iterator last, Iterator out, Comparator cmp)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    auto src = utils::iterator2pointer(first);
    auto outp = utils::iterator2pointer(out);

    _merge_sort_common(src, 0, n-1, outp, true, cmp, internal::stable_block_sorter());
}

template<typename T, typename Invoker, size_t block_size, typename MergerSettings>
template<typename Iterator, typename Comparator>
void sorter<T, Invoker, block_size, MergerSettings>::stable_merge_sort(Iterator first, Iterator last, Comparator cmp)
{
    static_assert(std::is_same<T, typename std::iterator_traits<Iterator>::value_type>::value,
                  "Specified type must be the same as type of iterators");
    static_assert(is_random_access_iterator<Iterator>::value,
                  "Iterators must be of random-access iterator type");

    long long n = std::distance(first, last);
    if(n < 2)
        return;

    aligned_vector<T> tmp_buffer;
    tmp_buffer.resize(n);

    auto src = utils::iterator2pointer(first);

    _merge_sort_common(src, 0, n-1, tmp_buffer.data(), false, cmp, internal::stable_block_sorter());
}

template<typename T, typename Invoker, size_t block_size, typename MergerSettings>
template<typename BlockSorter, typename Comparator>
void sorter<T, Invoker, block_size, MergerSettings>::_merge_sort_common(T* src, size_t l, size_t r, T* dest, bool src2dest, Comparator cmp,
                                                                        BlockSorter block_sorter, Invoker invoker)
{

    if(r == l)
    {
        if(src2dest)
            dest[l] = src[l];

        return;
    }

    if((r-l) <= block_size && !src2dest)
    {
        block_sorter(src+l, src+r+1, cmp);
        return;
    }

    size_t m = (r + l) / 2;

    invoker(
            [&]{_merge_sort_common(src, l, m, dest, !src2dest, cmp, block_sorter, invoker);},
            [&]{_merge_sort_common(src, m +1, r, dest, !src2dest, cmp, block_sorter, invoker);}
    );

    if(src2dest)
        merger_type::merge(src, l, m, m+1, r, dest, l, cmp);
    else
        merger_type::merge(dest, l, m, m+1, r, src, l, cmp);


}

}}

#endif //SAL_SORT_HPP
