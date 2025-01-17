# cmake_minimum_required(VERSION 3.10)
# project(VGG19)

# # Set the required C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")

# # Find required packages
# find_package(OpenCV REQUIRED)
# find_package(Torch REQUIRED)


# # Add executable and libraries
# add_executable(main main.cpp)
# add_library(vgg src/vgg.cpp)


# # Link libraries
# target_link_libraries(main 
#     PRIVATE 
#     "${TORCH_LIBRARIES}"
#     vgg
# )


# # Ensure CMake can find the Torch headers
# include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})
# include_directories(${OpenCV_INCLUDE_DIRS})

# # Set C++ standard for the target
# set_property(TARGET main PROPERTY CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)
project(VGG19)

# Set the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Add executable and libraries
add_executable(main main.cpp)
add_library(vgg src/vgg.cpp)
add_library(preprocess src/preprocess.cpp)

# Include directories for OpenCV and Torch
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(main 
    PRIVATE 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS} # Add OpenCV libraries
    vgg
    preprocess
)

# Ensure CMake can find the Torch headers
set_property(TARGET main PROPERTY CXX_STANDARD 17)
