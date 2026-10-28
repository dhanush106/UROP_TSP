cmake_minimum_required(VERSION 3.18)
project(TSP_XGBOOST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TSP_XGBOOST_WITH_HIGHS "Build with the open-source HiGHS TSP solver" ON)

if(TSP_XGBOOST_WITH_HIGHS)
    include(FetchContent)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_CXX ON CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(FAST_BUILD ON CACHE BOOL "" FORCE)
    set(ZLIB OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        highs
        GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
        GIT_TAG v1.10.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(highs)
endif()

add_executable(TSP_XGBOOST
    src/main.cpp
    src/graph.cpp
    src/features.cpp
    src/xgboost_api.cpp
    src/ml_pipeline.cpp
)

if(TSP_XGBOOST_WITH_HIGHS)
    target_sources(TSP_XGBOOST PRIVATE src/solver_highs.cpp)
    target_link_libraries(TSP_XGBOOST PRIVATE highs)
    target_compile_definitions(TSP_XGBOOST PRIVATE TSP_XGBOOST_WITH_HIGHS=1)
else()
    target_sources(TSP_XGBOOST PRIVATE src/solver_stub.cpp)
endif()

target_include_directories(TSP_XGBOOST PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
if(MSVC)
    target_compile_options(TSP_XGBOOST PRIVATE /W4 /permissive-)
else()
    target_compile_options(TSP_XGBOOST PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(TSP_XGBOOST PRIVATE ${CMAKE_DL_LIBS})
endif()

enable_testing()
add_executable(TSP_XGBOOST_TEST
    tests/test_core.cpp
    src/graph.cpp
    src/features.cpp
)
target_include_directories(TSP_XGBOOST_TEST PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME TSP_XGBOOST_TEST COMMAND TSP_XGBOOST_TEST)

if(TSP_XGBOOST_WITH_HIGHS)
    add_executable(TSP_XGBOOST_SOLVER_TEST
        tests/test_solver.cpp
        src/solver_highs.cpp
        src/features.cpp
        src/graph.cpp
    )
    target_include_directories(TSP_XGBOOST_SOLVER_TEST PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(TSP_XGBOOST_SOLVER_TEST PRIVATE highs)
    add_test(NAME TSP_XGBOOST_SOLVER_TEST COMMAND TSP_XGBOOST_SOLVER_TEST)
endif()

set_target_properties(TSP_XGBOOST PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
