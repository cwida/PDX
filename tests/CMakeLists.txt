include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_COMMON_LIBS
    gtest
    gtest_main
    ${MKL_COMMON_LIBS}
    ${BLAS_LINK_LIBRARIES}
)

if (FFTW_FOUND)
    list(APPEND TEST_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

# Utilities (not tests)
add_executable(generate_test_data.out generate_test_data.cpp)
target_link_libraries(generate_test_data.out PRIVATE ${MKL_COMMON_LIBS} ${BLAS_LINK_LIBRARIES})

add_executable(generate_test_ground_truth.out generate_test_ground_truth.cpp)
target_link_libraries(generate_test_ground_truth.out PRIVATE ${MKL_COMMON_LIBS} ${BLAS_LINK_LIBRARIES})

# Test executables
add_executable(test_distance_computers.out test_distance_computers.cpp)
target_link_libraries(test_distance_computers.out PRIVATE ${TEST_COMMON_LIBS})

add_executable(test_search.out test_search.cpp)
target_link_libraries(test_search.out PRIVATE ${TEST_COMMON_LIBS})

add_executable(test_serialization.out test_serialization.cpp)
target_link_libraries(test_serialization.out PRIVATE ${TEST_COMMON_LIBS})

add_executable(test_filtered_search.out test_filtered_search.cpp)
target_link_libraries(test_filtered_search.out PRIVATE ${TEST_COMMON_LIBS})

add_executable(test_index_properties.out test_index_properties.cpp)
target_link_libraries(test_index_properties.out PRIVATE ${TEST_COMMON_LIBS})

include(GoogleTest)
gtest_discover_tests(test_distance_computers.out)
gtest_discover_tests(test_search.out)
gtest_discover_tests(test_serialization.out)
gtest_discover_tests(test_filtered_search.out)
gtest_discover_tests(test_index_properties.out)

add_custom_target(tests
    DEPENDS
        test_distance_computers.out
        test_search.out
        test_serialization.out
        test_filtered_search.out
        test_index_properties.out
)
