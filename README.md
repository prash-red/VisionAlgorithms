# VisionAlgorithms [![Ubuntu](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci.yml) [![Ubuntu](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci-with-cuda.yml/badge.svg)](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci-with-cuda.yml)

A CLI tool to run computer vision algorithms.  
Currently, only the homography algorithm is implemented.

## Features

- Run homography transformation on images using OpenCV.
- Specify source and destination points via command line or a coordinates file.
- Output the transformed image to a file or display it in a window.

## Build Instructions

1. ### Clone the repository:
   ```bash
   git clone https://github.com/prash-red/VisionAlgorithms.git
   cd VisionAlgorithms
   ```

2. ### Install dependencies:
   You can either install the required packages manually or use vcpkg for dependency management.

   **Manual installation:**
    - CMake (\>= 3.31)
    - OpenCV (\>= 4.11)
    - CLI11
    - Catch2
    - CUDA (optional, for GPU support)

   OpenCV is recommended to be built from source.

   **Using vcpkg:**
   See the official installation
   guide: [https://github.com/microsoft/vcpkg#quick-start](https://github.com/microsoft/vcpkg#quick-start)

3. ### Build the project:
   3.1 **For CPU**
   ```bash
   cmake -S ./ -B build
   cmake --build build   
   ```
   3.2 **For GPU** (requires CUDA)
    ```bash
    cmake -S ./ -B build -DENABLE_CUDA=ON
    cmake --build build
    ```
   3.3 **For CPU with vcpkg**
    ```bash
    cmake -S ./ -B build -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
    cmake --build build
    ```

## Usage

Run the homography algorithm:

```bash
./build/cv homography -f <image_path> -s <x0> <y0> <x1> <y1> <x2> <y2> <x3> <y3> -d <x0'> <y0'> <x1'> <y1'> <x2'> <y2'> <x3'> <y3'>
```

Or use a coordinates file:

```bash
./build/cv homography -f <image_path> -c <coords_file>
```

- `-f, --file` : Path to the input image (required)
- `-s, --source` : 4 source points (8 integers, separated by spaces)
- `-d, --destination` : 4 destination points (8 integers, separated by spaces)
- `-c, --coords-file` : File with 4 source and 4 destination coordinates (each line: `x,y`)
- `-o, --output-file` : Output image file (optional; if not set, displays the result)
- `--cuda` : Use CUDA for processing (optional; requires CUDA support)

**Example:**
An example image with coordinates has been provided in the test folder

```bash
./build/cv homography -f tests/test_data/homography/document.jpg -c tests/test_data/homography/coords.txt 
```
