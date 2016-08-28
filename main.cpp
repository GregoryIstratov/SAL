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
#include "sort.hpp"
#include <random>

using namespace sal::merge;

using sal::utils::print_seq;

const size_t MEGABYTE = (1024 * 1024);
const size_t TEST_SIZE = 35;//1024 *2 * MEGABYTE / sizeof(int);
const int N = 20;

class Test {

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
        for (size_t i = 0, j = 1, k = 2; i < size; ++i, j += 2, k += 2) {
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
            sal::merge::internal::sequential_simd_merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);
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

            merger<int, serial_invoker, default_merger>::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

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

            merger<int, serial_invoker, simd_int_merger>::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

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

            merger<int, parallel_invoker, simd_int_merger, simple_block_partition<64>>
            ::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

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

            merger<int, parallel_invoker, default_merger>::merge(a_.begin(), a_.end(), b_.begin(), b_.end(), res_);

            auto tm_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = tm_end - tm_start;

            total_elapsed += elapsed.count();
        }

        std::cout << "Parallel DAC Merge has finished for " << total_elapsed << " sec." << std::endl;

        test_result();
    }

    void test_result()
    {
        for(long long j =0;j<(long long)buffer_.size();++j)
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


template<typename Iter>
bool check_sorted(Iter first, Iter last)
{
    typename std::iterator_traits<Iter>::value_type prev = *first;
    ++first;
    while(first != last)
    {
        if(prev > *first)
            return false;

        ++first;
    }

    return true;
}


void test_merger()
{
    aligned_vector<int> a = { -58, -72, -29, -29, 23, -8, -7, -5, 0, 6, -75, 85, };
    aligned_vector<int> b = {-28, 78, 15, -8, 55, -8, 15, 15, 78, 78, 100 };
    aligned_vector<int> c;
    c.resize(a.size()+b.size());
    std::sort(a.begin(),a.end());
    std::sort(b.begin(), b.end());

    aligned_vector<int> buffer;
    partial_vector<int> pa, pb;
    partial_vector<int>::create_solid_2partial(a.begin(),a.end(),b.begin(),b.end(), buffer, pa, pb);

    sal::merge::merger<int, serial_invoker, default_merger, static_block_partition<2>>
    ::merge(pa.begin(), pa.end(),pb.begin(),pb.end(),c.begin());


}

void test_binary_search()
{
    aligned_vector<int> a = { 1,3,5,5,7,7,9 };

    auto q = binary_search(6, a.begin(), a.end());

    a.insert(q, 6);

}

struct MyOutDebugger
{
    const aligned_vector<int>& v;

    MyOutDebugger(const aligned_vector<int>& _v)
            : v(_v)
    {}

    void operator()(int height)
    {
        print_seq("[OutputBuffer]", height, v.begin(), v.end());
    }
};

void test_index_merge()
{
    aligned_vector<int> a = { -72, -58, -49, -29, -11, -8, -7, -5, 0, 6, 16, 85, };
    aligned_vector<int> b = {-28, -13, -10, -8, 5, 8, 15, 16, 78, 96, 100 };
    aligned_vector<int> c;
    aligned_vector<int> out;
    out.resize(a.size()+b.size());
    std::sort(a.begin(),a.end());
    std::sort(b.begin(), b.end());

    c.insert(c.end(), a.begin(), a.end());
    c.insert(c.end(), b.begin(), b.end());

    auto mid = std::find(c.begin(), c.end(), -28);

    print_seq("[StartInput]", c.begin(), c.end());

   sal::merge::merger<int, serial_invoker, simd_int_merger, static_block_partition<4>>
   ::merge(c.begin(), mid, c.end(), out.begin());

    print_seq("[Resulting output]", out.begin(), out.end());
}

void test_index_merge2()
{
    aligned_vector<int> a = { -72, -58, -49, -29, -11, -8, -7, -5, 0, 6, 16, 85, };
    aligned_vector<int> b = {-28, -13, -10, -8, 5, 8, 15, 16, 78, 96, 100 };
    aligned_vector<int> out;
    out.resize(a.size()+b.size());
    std::sort(a.begin(),a.end());
    std::sort(b.begin(), b.end());


    print_seq("[StartInput A]", a.begin(), a.end());
    print_seq("[StartInput B]", b.begin(), b.end());

    sal::merge::merger<int, parallel_invoker, default_merger, static_block_partition<4>>
    ::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());

    print_seq("[Resulting output]", out.begin(), out.end());
}

void test_index_merge3()
{
    aligned_vector<int> a = { -72, -58, -49, -29, -11, -8, -7, -5, 0, 6, 16, 85, };
    aligned_vector<int> b = {-28, -13, -10, -8, 5, 8, 15, 16, 78, 96, 100 };
    aligned_vector<int> out;
    out.resize(a.size()+b.size());
    std::sort(a.begin(),a.end());
    std::sort(b.begin(), b.end());

    long long p1 = 0;
    long long r1 = a.size()-1;
    long long p2 = 0;
    long long r2 = b.size()-1;


    print_seq("[StartInput A]", a.begin(), a.end());
    print_seq("[StartInput B]", b.begin(), b.end());

    sal::merge::merger<int, serial_invoker, default_merger, static_block_partition<2>>
    ::merge(a.data(),p1,r1,p2,r2,out.data(), 0);

    print_seq("[Resulting output]", out.begin(), out.end());
}



void test_reverse_kernel()
{
    aligned_vector<int> a = { 15,13,11,9,7,5,3,1 };
    aligned_vector<int> b = { 14,12,10,8,6,4,2,0 };
    aligned_vector<int> c;
    c.resize(a.size()+b.size());

    internal::kernel::reverse_merge_256i(a.data(), b.data(), c.data());
}


class ApplyIntRandom
{
    int * const my_i;
public:

    void operator()( const tbb::blocked_range<size_t>& r ) const {
        std::random_device rd;
        std::mt19937 gen_(rd());
        //std::uniform_int_distribution<> dis(INT32_MIN, INT32_MAX);
        std::uniform_int_distribution<> dis(-100, 100);
        int *pi = my_i;
        for( size_t i=r.begin(); i!=r.end(); ++i )
            pi[i] = dis(gen_);
    }

    ApplyIntRandom(int* data)
            : my_i(data)
    {

    }
};

void parallel_randomize(int* a, size_t n ) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0,n), ApplyIntRandom(a));
}

int main() {

    std::cout<<"Initializing memory..."<<std::flush;

    auto tm_start = std::chrono::high_resolution_clock::now();
    aligned_vector<int> a;
    a.resize(TEST_SIZE);
    aligned_vector<int> res;
    res.resize(TEST_SIZE);

    auto tm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = tm_end - tm_start;

    std::cout<<"ok ("<<elapsed.count()<<" sec.)"<<std::endl;



    std::cout<<"Making testing data..."<<std::flush;
    tm_start = std::chrono::high_resolution_clock::now();

    parallel_randomize(a.data(), a.size());

    tm_end = std::chrono::high_resolution_clock::now();
    elapsed = tm_end - tm_start;

    std::cout<<"ok ("<<elapsed.count()<<" sec.)"<<std::endl;

    std::cout<<"Starting sorting the data..."<<std::flush;

    tm_start = std::chrono::high_resolution_clock::now();

    //simple way
    //sal::sort::sorter<int>::merge_sort(a.begin(), a.end(), res.begin());

    sal::sort::sorter<int, serial_invoker, 8192, merger_settings<simd_int_merger, static_block_partition<8192>>>
    ::merge_sort(a.begin(), a.end(), res.begin());


    //std::sort(a.begin(), a.end());

    tm_end = std::chrono::high_resolution_clock::now();
    elapsed = tm_end - tm_start;


    std::cout << "ok ( " << elapsed.count() << " sec.)" << std::endl;

    std::cout<<"Checking array for consistency..."<<std::flush;
    if(std::is_sorted(res.begin(), res.end()))
    {
        std::cout<<"ok"<<std::endl;
    }
    else
    {
        std::cout<<"fail"<<std::endl;
    }

    return 0;
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