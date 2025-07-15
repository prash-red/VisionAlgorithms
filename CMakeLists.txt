# This file specifies how the project should be built, using CMake.
# If you are unfamiliar with CMake, don't worry about all the details.
# The sections you might want to edit are marked as such, and
# the comments should hopefully make most of it clear.
#
# For many purposes, you may not need to change anything about this file.

cmake_minimum_required(VERSION 3.31)

# Set project name, version and laguages here. (change as needed)
# Version numbers are available by including "exampleConfig.h" in
# the source. See version.h.in for some more details.
project(VisionAlgorithms VERSION 0.1 LANGUAGES CXX)

# Options: Things you can set via commandline options to cmake (e.g. -DENABLE_LTO=[ON|OFF])
option(ENABLE_WARNINGS_SETTINGS "Allow target_set_warnings to add flags and defines.
                                 Set this to OFF if you want to provide your own warning parameters." OFF)
option(ENABLE_LTO "Enable link time optimization" ON)
option(ENABLE_DOCTESTS "Include tests in the library. Setting this to OFF will remove all doctest related code.
                        Tests in tests/*.cpp will still be enabled." ON)

option(ENABLE_CUDA "Enable CUDA" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
endif ()

# Include stuff. No change needed.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(ConfigSafeGuards)
include(Colors)
include(CTest)
include(Documentation)
include(LTO)
include(Misc)
include(Warnings)
if(ENABLE_CUDA)
    include(AutoDetectCudaArch)
endif ()
# Check for LTO support.
find_lto(CXX)

add_library(compiler_flags INTERFACE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(cxx_flags -Wall)
set(cuda_flags -Xcompiler=-Wall -Wno-deprecated-gpu-targets)
add_library(developer_flags INTERFACE)
target_compile_options(developer_flags
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:${cxx_flags}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:${cuda_flags}>
)

target_link_libraries(compiler_flags INTERFACE developer_flags)
# --------------------------------------------------------------------------------
#                                  Dependencies
# --------------------------------------------------------------------------------

find_package(OpenCV REQUIRED)
find_package(CLI11 REQUIRED)
find_package(Catch2 REQUIRED)

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif ()

# --------------------------------------------------------------------------------
#                         Locate files (change as needed).
# --------------------------------------------------------------------------------
set(SOURCES          # All .cpp files in src/
        src/algorithms/cpu_homographer.cpp
        src/algorithms/abstract_homographer.cpp
)
if(ENABLE_CUDA)
    list(APPEND SOURCES src/algorithms/cuda_homographer.cu)
endif()
set(TESTFILES        # All .cpp files in tests/
        tests/main.cpp
)
set(LIBRARY_NAME engine)  # Default name for the library built from src/*.cpp (change if you wish)

# --------------------------------------------------------------------------------
#                            Build! (Change as needed)
# --------------------------------------------------------------------------------
# Compile all sources into a library.
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

# Lib needs its header files, and users of the library must also see these (PUBLIC). (No change needed)
target_include_directories(${LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCV_LIBS} CLI11::CLI11)
if(ENABLE_CUDA)
    target_link_libraries(${LIBRARY_NAME} PUBLIC CUDA::cudart CUDA::cublas)
endif ()
# Set the compile options you want (change as needed).
target_set_warnings(${LIBRARY_NAME} ENABLE ALL AS_ERROR ALL DISABLE Annoying)
# target_compile_options(${LIBRARY_NAME} ... )  # For setting manually.

# Add an executable for the file app/main.cpp.
# If you add more executables, copy these lines accordingly.
add_executable(cv app/main.cpp)   # Name of exec. and location of file.
target_link_libraries(cv PRIVATE ${LIBRARY_NAME})  # Link the executable to library (if it uses it).
target_set_warnings(cv ENABLE ALL AS_ERROR ALL DISABLE Annoying) # Set warnings (if needed).
target_enable_lto(cv optimized)  # enable link-time-optimization if available for non-debug configurations
if(ENABLE_CUDA)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC USE_CUDA)
    target_compile_definitions(cv PRIVATE USE_CUDA)
endif()

# Set the properties you require, e.g. what C++ standard to use. Here applied to library and main (change as needed).
set_target_properties(
        ${LIBRARY_NAME} cv
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_SEPARABLE_COMPILATION ON
)

# Set up tests (see tests/CMakeLists.txt).
add_subdirectory(tests)
