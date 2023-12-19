cmake_minimum_required(VERSION 3.0)
project(YourProjectName)
set(CMAKE_CXX_STANDARD 17)

# Define directories
set(LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Add your source files
add_executable(main ${SRC_DIR}/main.cpp)
add_executable(unitary ${SRC_DIR}/unitary.cpp)
add_executable(qpe ${SRC_DIR}/qpe.cpp)

# Include directories and libraries
target_include_directories(main PRIVATE ${LIB_DIR})
target_include_directories(unitary PRIVATE ${LIB_DIR})
target_include_directories(qpe PRIVATE ${LIB_DIR})

add_library(Encoding ${LIB_DIR}/Encoding.cpp ${LIB_DIR}/Encoding.h)
add_library(Circuit ${LIB_DIR}/Circuit.cpp ${LIB_DIR}/Circuit.h)
add_library(CppTest ${LIB_DIR}/CppTest.cpp ${LIB_DIR}/CppTest.h )

# Link External dependencies
find_package(qpp REQUIRED)

# include_directories(/usr/local/include/eigen3/)
add_subdirectory(matplotplusplus)
target_link_libraries(main PUBLIC matplot)

# link Encoding and Circuit
target_link_libraries(main PRIVATE Encoding)
target_link_libraries(main PRIVATE Circuit)
target_link_libraries(main PRIVATE CppTest)
target_link_libraries(main PUBLIC ${QPP_LINK_DEPS} libqpp)

target_link_libraries(unitary PRIVATE Encoding)
target_link_libraries(unitary PRIVATE Circuit)
target_link_libraries(unitary PRIVATE CppTest)
target_link_libraries(unitary PUBLIC ${QPP_LINK_DEPS} libqpp)

target_link_libraries(qpe PRIVATE Encoding)
target_link_libraries(qpe PRIVATE Circuit)
target_link_libraries(qpe PRIVATE CppTest)
target_link_libraries(qpe PUBLIC ${QPP_LINK_DEPS} libqpp)

target_link_libraries(Encoding PUBLIC ${QPP_LINK_DEPS} libqpp)
target_link_libraries(Circuit PUBLIC ${QPP_LINK_DEPS} libqpp)
target_link_libraries(CppTest PUBLIC ${QPP_LINK_DEPS} libqpp)

# link unitary with qpp 
