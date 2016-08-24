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

using namespace merge;

const size_t MEGABYTE = (1024 * 1024);

class Test {

    const size_t TEST_SIZE = 2048 * MEGABYTE / sizeof(int);
    const int N = 20;

    aligned_vector<int> buffer_;
    partial_vector<int> a_;
    partial_vector<int> b_;
    aligned_vector<int> _res_;
    int* res_;
    size_t mid_;
public:
    void inititalize()
    {
        make_merge_data_p(buffer_, a_, b_, mid_, TEST_SIZE);

        _res_.resize(buffer_.size());
        res_ = _res_.data();
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
        //res_.clear();

        auto tm_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i) {
            sequential_simd_merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);
        }

        auto tm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = tm_end - tm_start;

        std::cout << "sequential_simd_merge has finished for " << elapsed.count() << " sec." << std::endl;

        test_result();
    }

    void test_std_merge() {

        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            std::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;
            total_elapsed += elapsed.count();
        }



        std::cout << "std::merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }

    void test_dac_merge() {

        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merger<int, default_merger>::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();

        }



        std::cout << "DAC Merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }


    void test_simd_dac_merge() {


        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merger<int, simd_int_merger>::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();

        }

        std::cout << "SIMD DAC Merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }


    void test_parallel_simd_dac_merge() {

        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merger<int, simd_int_merger, simple_block_partition<64>>::parallel_merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();
        }

        std::cout << "Parallel SIMD DAC Merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }


    void test_parallel_dac_merge() {

        double total_elapsed = 0;
        for (int i = 0; i < N; ++i) {
            auto tm_start = std::chrono::high_resolution_clock::now();

            merger<int, default_merger>::parallel_merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();
        }

        std::cout << "Parallel DAC Merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }

    void test_result()
    {
        for(int j =0;j<buffer_.size();++j)
        {
            if(res_[j] != j+1)
            {
                std::cout<<"[test_result]: Consistency of resulting array is broken"<<std::endl;
                std::cout<<"[test_result]: result "<<res_[j]<<" != "<<j+1<<std::endl;
                abort();
            }
            res_[j] = 0;
        }
    }
};



int main() {

    Test t;
    t.inititalize();

    t.test_sequential_simd_merge();
    t.test_dac_merge();
    t.test_simd_dac_merge();
    t.test_parallel_simd_dac_merge();
    t.test_parallel_dac_merge();
    t.test_std_merge();

    return 0;
}