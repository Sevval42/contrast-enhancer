# Vulkan Compute Boilerplate
A Vulkan boilerplate that supports multiple **compute**shaders and iterative application.
Right now, uniform-, storage- and image-buffers are supported.

### Prerequisites
- C++11 or later
- Vulkan SDK (latest version recommended)
- CMake (3.10+)
- GPU with Vulkan 1.2+ support
- shader compiler:
    - glslang on mac
    - glslc on windows

### Project structure
The base of the project is the `vulkan_base.h` file, where all relevant structs and methods are defined. Implementations can be found in the according *.cpp files.

#### Program flow
The Program starts with initializing the application in `initApplication()`. This means:
- Creating a `VulkanContext`, that holds relevant Vulkan objects like the device, compute queue and command pool
- Setting up the descriptor sets for data transfers to the gpu. It's important to notice, that all shaders will use the same descriptor set with this setup
    - Descriptor set layouts can be added with `addDescriptorSetLayout(VulkanDescriptorSet, VkDescriptorType)`
    - After adding the layouts, `createDescriptorSet()` has to be called
- Filling the descriptor sets with buffers holding the data
    - Do this with the `addBufferAndData()` or `addImageAndData()` methods given by the `VulkanDescriptorSet` object. This will automatically use a stagingbuffer to load data into gpu memory
    - To access buffers from the CPU again, you will have to call the `getDataFromBufferWithStagingBuffer()` or `getDataFromImageWithStagingBuffer()` method
- Creating the `VulkanPipeline` which is used for shader execution.
    - The `createPipeline()` method will get the shader .spv filenames as a vector and the dispatch sizes for each shader.

The `runApplication()` method does one iteration through the compute shaders in the order they were added to the `VulkanPipeline`, using one `VkCommandBuffer`.  Therefore, this method is called in a for-loop inside the main() method of the program.

More detail about the implementation can be found in the example code of the main.cpp file. It uses three shaders, one storagebuffer, uniformbuffer and imagebuffer, and prints the storagebuffer into the console after each iteration.
One image is loaded ("images/image.png") and inverted. The output can be found in the bin directory.

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