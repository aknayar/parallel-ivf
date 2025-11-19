cmake_minimum_required(VERSION 3.16)
set(CMAKE_POLICY_DEFAULT_CMP0148 OLD CACHE STRING "")
project(MyProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch dependencies
include(FetchContent)

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)

FetchContent_MakeAvailable(pybind11)

# Collect sources
add_library(my_project_lib
    src/IVFBase.cpp
    src/kmeans.cpp
    src/distances.cpp
    # Add more files here
)

# Enable position-independent code for shared library linking
set_target_properties(my_project_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(my_project_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Python bindings
pybind11_add_module(parallel_ivf src/python_bindings.cpp)
target_link_libraries(parallel_ivf PRIVATE my_project_lib)
