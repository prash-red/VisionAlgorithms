# download CPM.cmake
file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
        ${CMAKE_BINARY_DIR}/CPM.cmake
        EXPECTED_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
)
include(${CMAKE_BINARY_DIR}/CPM.cmake)

# add dependencies here
CPMAddPackage(
        NAME CLI11
        GITHUB_REPOSITORY CLIUtils/CLI11
        GIT_TAG v2.5.0
)

if(NOT OpenCV_TAG)
    set(OpenCV_TAG "4.11.0")
endif()

find_package(OpenCV)

if(${OpenCV_FOUND})
    message(STATUS "OpenCV found: ${OpenCV_DIR}")
else()
    message(STATUS "OpenCV not found. Downloading and building from source...")
    set(OpenCV_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-install)
    set(CMAKE_PREFIX_PATH ${OpenCV_INSTALL_DIR})
    CPMAddPackage(
            NAME OpenCV
            GITHUB_REPOSITORY opencv/opencv
            GIT_TAG ${OpenCV_TAG}
            DOWNLOAD_ONLY
            OPTIONS
                "BUILD_SHARED_LIBS ON"
                "BUILD_JAVA OFF"
                "BUILD_opencv_python3 OFF"
                "WITH_FFMPEG OFF"
                "BUILD_PERF_TESTS OFF"
                "BUILD_TESTS OFF"
                "CMAKE_INSTALL_PREFIX ${OpenCV_INSTALL_DIR}"
    )
    set(OpenCV_DIR ${OpenCV_INSTALL_DIR}/lib/cmake/opencv4)
    set(OpenCV_LIBS opencv_calib3d opencv_core opencv_dnn opencv_features2d
        opencv_flann opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc
        opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_video
        opencv_videoio)
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INSTALL_DIR}/include/opencv4)
    add_custom_target(
            install_opencv ALL
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/_deps/opencv-build --target install
    )
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif ()