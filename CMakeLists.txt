cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Collect sources (or list them manually)
add_library(my_project_lib
    src/IVFBase.cpp
    src/kmeans.cpp
    src/distances.cpp
    # Add more files here
)

target_include_directories(my_project_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Grab all .cpp files in tests/
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
        PRIVATE
            my_project_lib
            gtest_main
    )

    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endforeach()

# Enable testing
enable_testing()
include(GoogleTest)

# Discover tests
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    gtest_discover_tests(${test_name})
endforeach()