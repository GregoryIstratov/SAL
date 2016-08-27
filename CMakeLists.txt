cmake_minimum_required(VERSION 3.5)
project(sal)

message("SAL - Superior Algorithm Library")

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse4.1 -mavx2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

set(SOURCE_FILES main.cpp aligned_allocator.hpp partial_vector.hpp merge.cpp merge.hpp utility.cpp utility.hpp sort.hpp)
add_executable(sal ${SOURCE_FILES})
target_link_libraries(sal tbb)