//
// Created by Greg Miller on 8/23/16.
//
#include <vector>
#include "aligned_allocator.hpp"

#ifndef TEST_HEAP_SHARED_VECTOR_HPP
#define TEST_HEAP_SHARED_VECTOR_HPP



template<typename T>
class partial_vector
{
    T* data_ = nullptr;
    size_t begin_ = 0;
    size_t end_ = 0;
public:
    partial_vector() {}
    partial_vector(aligned_vector<T>& vec, size_t begin, size_t end)
            : data_(vec.data()), begin_(begin), end_(end)
    {

    }

    partial_vector(T* vec, size_t begin, size_t end)
            : data_(vec), begin_(begin), end_(end)
    {

    }

    void set(aligned_vector<T>& vec, size_t begin, size_t end)
    {
        data_ = vec.data();
        begin_ = begin;
        end_ = end;
    }

    size_t size() const
    {
        return end_ - begin_ + 1;
    }

    T& operator[](size_t idx)
    {
        return data_[begin_ + idx];
    }

    const T& operator[](size_t idx) const
    {
        return data_[begin_ + idx];
    }


    T* begin()
    {
        return &data_[begin_];
    }

    const T* begin() const
    {
        return &data_[begin_];
    }

    T* end()
    {
        return (&data_[end_])+1;
    }

    const T* end() const
    {
        return (&data_[end_])+1;
    }

    T* data()
    {
        return &data_[begin_];
    }

    const T* data() const
    {
        return &data_[begin_];
    }

    template<typename InputIterator>
    static void create_solid_2partial(InputIterator first1, InputIterator last1,
                                    InputIterator first2, InputIterator last2,
                                    aligned_vector<T>& buffer, partial_vector& a, partial_vector& b);

};

template<typename T>
template<typename InputIterator>
void partial_vector<T>::create_solid_2partial(InputIterator first1, InputIterator last1,
                                           InputIterator first2, InputIterator last2,
                                           aligned_vector<T>& buffer, partial_vector& a, partial_vector& b)
{
    size_t a_size = std::distance(first1, last1);
    size_t b_size = std::distance(first2, last2);

    buffer.clear();
    buffer.shrink_to_fit();
    buffer.reserve(a_size + b_size);

    buffer.insert(buffer.end(), first1, last1);
    buffer.insert(buffer.end(), first2, last2);

    a = partial_vector(buffer, 0, a_size-1);
    b = partial_vector(buffer, a_size, buffer.size()-1);

}

#endif //TEST_HEAP_SHARED_VECTOR_HPP
