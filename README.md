# VisionAlgorithms

A CLI tool to run computer vision algorithms.  
Currently, only the homography algorithm is implemented.

## Features

- Run homography transformation on images using OpenCV.
- Specify source and destination points via command line or a coordinates file.
- Output the transformed image to a file or display it in a window.

## Build Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/prash-red/VisionAlgorithms.git
   cd VisionAlgorithms
   ```

2. **Install dependencies:**
    - CMake (>= 3.14)
    - OpenCV
    - CLI11

   OpenCV is recommended to built from source

3. **Build the project:**
   ```sh
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

Run the homography algorithm:

```sh
./build/cv homography -f <image_path> -s <x0> <y0> <x1> <y1> <x2> <y2> <x3> <y3> -d <x0'> <y0'> <x1'> <y1'> <x2'> <y2'> <x3'> <y3'>
```

Or use a coordinates file:

```sh
./build/cv homography -f <image_path> -c <coords_file>
```

- `-f, --file` : Path to the input image (required)
- `-s, --source` : 4 source points (8 integers, separated by spaces)
- `-d, --destination` : 4 destination points (8 integers, separated by spaces)
- `-c, --coords-file` : File with 4 source and 4 destination coordinates (each line: `x,y`)
- `-o, --output-file` : Output image file (optional; if not set, displays the result)

**Example:**
An example image with coordinates has been provided in the test folder

```sh
./build/cv homography -f tests/test_data/homography/document.jpg -c tests/test_data/homography/coords.txt 
```
