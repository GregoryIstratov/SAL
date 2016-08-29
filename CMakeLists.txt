cmake_minimum_required(VERSION 3.5)
project(sal)

message("SAL - Superior Algorithm Library")

include_directories(/usr/local/include)
link_directories(/usr/local/lib)



if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Compiler is Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Compiler is GNU GCC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("Compiler is Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    LINK_DIRECTORIES(/opt/intel/lib)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wpedantic")

add_definitions(-D_USE_AVX2_)
add_definitions(-D_USE_SSE4_)

set(SOURCE_FILES main.cpp aligned_allocator.hpp partial_vector.hpp merge.cpp merge.hpp utility.cpp utility.hpp sort.hpp)
add_executable(sal ${SOURCE_FILES})
target_link_libraries(sal tbb)

#set(LIB_SOURCE_FILES aligned_allocator.hpp partial_vector.hpp merge.cpp merge.hpp utils.cpp utils.hpp sort.hpp)
#add_library(sallib STATIC ${LIB_SOURCE_FILES})
#target_link_libraries(sallib tbb)