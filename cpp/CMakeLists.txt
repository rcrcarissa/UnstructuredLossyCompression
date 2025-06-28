cmake_minimum_required(VERSION 3.10)
project(PTz CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR})



add_library(fileIO
    fileIO.cpp
)

add_executable(ptz
    main.cpp
)
target_link_libraries(ptz PUBLIC fileIO)
target_link_libraries(ptz PUBLIC zstd)
