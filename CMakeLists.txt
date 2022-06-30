cmake_minimum_required(VERSION 3.5)
project(Falconn++ CXX C)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -O3 -g")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (OpenMP_CXX_INCLUDE_DIR)
        include_directories("${OpenMP_CXX_INCLUDE_DIR}")
    endif()
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Boost 1.36.0 REQUIRED)

include_directories("src")
link_directories("src")
add_subdirectory("src")

add_executable(FALCONN++ "src/Falconn++.cpp")
if (OpenMP_FOUND)
	target_link_libraries(FALCONN++ ${OpenMP_CXX_LIBRARIES} Eigen3::Eigen falconn++)
endif()

add_executable(GLOVE "example/glove.cpp")
target_link_libraries(GLOVE ${OpenMP_CXX_LIBRARIES} Eigen3::Eigen falconn++)



