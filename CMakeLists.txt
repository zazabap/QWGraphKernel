cmake_minimum_required(VERSION 3.0)
project(YourProjectName)
set(CMAKE_CXX_STANDARD 17)

# Define directories
set(LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Add your source files
add_executable(main ${SRC_DIR}/main.cpp)

# Include directories and libraries
target_include_directories(main PRIVATE ${LIB_DIR})
add_library(Encoding ${LIB_DIR}/Encoding.cpp ${LIB_DIR}/Encoding.h)
target_link_libraries(main PRIVATE Encoding)

# Link any necessary dependencies
find_package(qpp REQUIRED)
include_directories(/usr/local/include/eigen3/)
add_subdirectory(matplotplusplus)
target_link_libraries(main PUBLIC matplot)
target_link_libraries(main PUBLIC ${QPP_LINK_DEPS} libqpp)

# include_directories(/opt/homebrew/opt/matplotplusplus/include)
# link_directories("/opt/homebrew/opt/matplotplusplus/lib")
# target_link_libraries(main PRIVATE /opt/homebrew/opt/matplotplusplus/lib)

# set(Matplotplusplus_DIR "/opt/homebrew/opt/matplotplusplus/lib/cmake/Matplot++")
# find_package(Matplot++ REQUIRED PATHS ${Matplotplusplus_DIR})
# target_link_libraries(main PRIVATE Matplot++)


#########################################################
# # only the old test
# cmake_minimum_required(VERSION 3.0)
# project(QppProject)
# set(CMAKE_CXX_STANDARD 17)
# find_package(qpp REQUIRED)

# # Add your source files
# add_executable(main src/qpptest.cpp)
# # Link any necessary libraries (e.g., Matplotplusplus)
# include_directories(/usr/local/include/eigen3/)
# target_link_libraries(main PUBLIC ${QPP_LINK_DEPS} libqpp)
#########################################################