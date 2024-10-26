cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(pv021)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    add_definitions(-DRELEASE)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

file(GLOB "SOURCE" "./src/*.cpp")
file(GLOB "HEADER" "./src/*.h")

add_executable(network
    "${SOURCE}"
    "${HEADER}")
