# Contrast Enhancer
Program that enhances contrast in images using decluttering. Written in C++ and glsl using Vulkan.

### Prerequisites
- C++11 or later
- Vulkan SDK (latest version recommended)
- CMake (3.10+)
- GPU with Vulkan 1.2+ support
- shader compiler:
    - glslang on mac
    - glslc on windows

### Building

To run this project, execute the following commands in the project directory:
#### Windows
```
cmake --preset windows
cmake --build .\build\
cd bin
.\vulkan_compute_boilerplate.exe "exampleImage.png"
```

#### MacOS
```
cmake --preset mac
cmake --build build
cd bin
./vulkan_compute_boilerplate "exampleImage.png"
```
The "exampleImage.png" must be in the images/ directory.

### Program constants
Constants like the histogram resolution, the size of the gauss kernel and the number of iterations can be set in the config.yaml file.
As the total number of values in the histogram grows cubically, the value of histogramBinCount should not surpass 256 for performance and memory reasons.


### LICENSE
MIT