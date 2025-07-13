# VisionAlgorithms [![Ubuntu](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci.yml) [![Ubuntu](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci-with-cuda.yml/badge.svg)](https://github.com/prash-red/VisionAlgorithms/actions/workflows/ci-with-cuda.yml)

A CLI tool to run computer vision algorithms.  
Currently, only the homography algorithm is implemented.

## Features

- Run homography transformation on images using OpenCV.
- Specify source and destination points via command line or a coordinates file.
- Output the transformed image to a file or display it in a window.

## Build Instructions

### Prerequisites

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh  # Linux/macOS
   # or .\bootstrap-vcpkg.bat  # Windows
   ```

2. **Set environment variable**:
   ```bash
   export VCPKG_ROOT=/path/to/vcpkg  # Linux/macOS
   # or set VCPKG_ROOT=C:\path\to\vcpkg  # Windows
   ```

### Build Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prash-red/VisionAlgorithms.git
   cd VisionAlgorithms
   ```

2. **Build the project**:
   
   **Option A: Using CMake presets (recommended)**:
   ```bash
   # For Release build
   cmake --preset default
   cmake --build build
   
   # For Debug build
   cmake --preset debug
   cmake --build build-debug
   
   # For CUDA build
   cmake --preset cuda
   cmake --build build-cuda
   ```
   
   **Option B: Manual CMake configuration**:
   ```bash
   # For CPU build
   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
   cmake --build build
   
   # For GPU build (requires CUDA)
   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -DENABLE_CUDA=ON -DVCPKG_MANIFEST_FEATURES=cuda
   cmake --build build
   ```

### Dependencies

Dependencies are automatically managed by vcpkg based on the `vcpkg.json` manifest:
- **OpenCV** (>= 4.5.0): Core computer vision library
- **CLI11** (>= 2.1.0): Command line interface library
- **Catch2** (>= 3.0.0): Testing framework

For CUDA builds, OpenCV with CUDA support is automatically included.

### Migration from CPM

This project previously used CPM (CPM.cmake) for dependency management. If you have been using the old system:

1. **Remove old build directories**: `rm -rf build/` to clear any CPM-based artifacts
2. **Install vcpkg**: Follow the vcpkg installation instructions above
3. **Use new build commands**: Use the vcpkg-based build instructions provided above

The old CPM-based system has been deprecated and `cmake/Dependencies.cmake` is no longer used.

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
