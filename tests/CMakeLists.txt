# Test suite for PDX

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

# Helper: add a utility executable (not a test) linked against PDX
function(pdx_add_test_utility name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE PDX)
    target_compile_definitions(${name} PRIVATE CMAKE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endfunction()

# Helper: add a GoogleTest executable and register its tests with CTest
function(pdx_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE PDX gtest gtest_main)
    target_compile_definitions(${name} PRIVATE CMAKE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(${name})
endfunction()

# Utilities (not tests)
pdx_add_test_utility(generate_test_data.out generate_test_data.cpp)
pdx_add_test_utility(generate_test_ground_truth.out generate_test_ground_truth.cpp)

# Test executables
pdx_add_test(test_distance_computers.out test_distance_computers.cpp)
pdx_add_test(test_search.out test_search.cpp)
pdx_add_test(test_serialization.out test_serialization.cpp)
pdx_add_test(test_filtered_search.out test_filtered_search.cpp)
pdx_add_test(test_index_properties.out test_index_properties.cpp)
pdx_add_test(test_maintenance.out test_maintenance.cpp)
pdx_add_test(test_iterative_search.out test_iterative_search.cpp)

add_custom_target(tests
    DEPENDS
        test_distance_computers.out
        test_search.out
        test_serialization.out
        test_filtered_search.out
        test_index_properties.out
        test_maintenance.out
        test_iterative_search.out
)
