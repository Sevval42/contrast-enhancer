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
.\vulkan_compute_boilerplate.exe
```

#### MacOS
```
cmake --preset mac
cmake --build build
cd bin
./vulkan_compute_boilerplate
```

### LICENSE
MIT