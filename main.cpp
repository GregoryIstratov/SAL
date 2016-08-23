#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iterator>
#include <tbb/tbb.h>
#include <immintrin.h>
#include "aligned_allocator.hpp"
#include "partial_vector.hpp"
#include "merge.hpp"



const size_t MEGABYTE = (1024 * 1024);

class Test {

    const size_t TEST_SIZE = 2048 * 2 * MEGABYTE / sizeof(int);
    const int N = 10;

    aligned_vector<int> buffer_;
    partial_vector<int> a_;
    partial_vector<int> b_;
    aligned_vector<int> res_;
    size_t mid_;
public:
    void inititalize()
    {
        make_merge_data_p(buffer_, a_, b_, mid_, TEST_SIZE);

        res_.resize(buffer_.size());
    }

    void make_merge_data(aligned_vector<int> &a, aligned_vector<int> &b, size_t size) {
        a.resize(size);
        b.resize(size);
        for (int i = 0, j = 1, k = 2; i < size; ++i, j += 2, k += 2) {
            a[i] = j;
            b[i] = k;
        }
    }

    void make_merge_data_p(aligned_vector<int> &buffer, partial_vector<int> &a, partial_vector<int> &b, size_t& mid, size_t size) {
        buffer.resize(size);
        mid = size / 2;
        for (size_t i = 0, j = 1, k = 2; i < mid; ++i, j += 2, k += 2) {
            buffer[i] = j;
            buffer[mid + i] = k;
        }

        a.set(buffer, 0, mid - 1);
        b.set(buffer, mid, size - 1);
    }


    void test_sequential_simd_merge() {
        res_.clear();

        auto tm_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i)
            sequential_simd_merge(a_.data(), a_.size(), b_.data(), b_.size(), res_.data());

        auto tm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = tm_end - tm_start;

        std::cout << "sequential_simd_merge has finished for " << elapsed.count() << " sec." << std::endl;
    }

    void test_std_merge() {
        res_.clear();


        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            std::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), std::back_inserter(res_));

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;
            total_elapsed += elapsed.count();
            res_.clear();
        }



        std::cout << "std::merge has finished for " << total_elapsed << " sec." << std::endl;
    }

    void test_dac_merge() {
        res_.clear();



        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merge_dac(buffer_, mid_, res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();

            res_.clear();
        }



        std::cout << "DAC Merge has finished for " << total_elapsed << " sec." << std::endl;
    }


    void test_simd_dac_merge() {
        res_.clear();




        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merge_simd_dac(buffer_, mid_, res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();
            res_.clear();
        }

        std::cout << "SIMD DAC Merge has finished for " << total_elapsed << " sec." << std::endl;
    }


    void test_parallel_simd_dac_merge() {
        res_.clear();


        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            parallel_merge_simd_dac(buffer_, mid_, res_.data());

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();
            res_.clear();
        }



        std::cout << "Parallel SIMD DAC Merge has finished for " << total_elapsed << " sec." << std::endl;
    }

};

int main() {

    Test t;
    t.inititalize();

    t.test_std_merge();
    t.test_sequential_simd_merge();
    t.test_dac_merge();
    t.test_simd_dac_merge();
    t.test_parallel_simd_dac_merge();


    return 0;
}