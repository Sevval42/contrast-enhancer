#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include "vulkan/vulkan_core.h"
#include "vulkan_base/vulkan_base.h"
#include "setup.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RANDIMG true
#define INTEGRALS false
#define TRANSFORMATION true

int ITERATIONS = 50;

UniformData constants;

VulkanContext* context;

VulkanDescriptorSet* baseDescriptorSet;
VulkanPipeline basePipeline;
StageBuffers baseBuffers;

VulkanDescriptorSet* mainDescriptorSet;
VulkanPipeline mainPipeline;
StageBuffers mainBuffers;

float sigma = 1.0;

void initApplication(std::string imageFile) {

    const char* instanceExtensions[] = {
        #ifdef __APPLE__
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        #endif
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
    };
    uint32_t instanceExtensionsCount = ARRAY_COUNT(instanceExtensions);

    const char* deviceExtensions[] = {
        #ifdef __APPLE__
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
        #endif
    };
    uint32_t deviceExtensionsCount = ARRAY_COUNT(deviceExtensions);

    LOG("Get device");
    context = initVulkan(
        instanceExtensionsCount,
        instanceExtensions, 
        deviceExtensionsCount, 
        deviceExtensions
    );

    // TODO: Change loading of constants to yaml file
    constants = {
        128,
        3,
        0
    };

    #if RANDIMG
    const int size = 256;
    const int randchannels = 4;
    const int stride_in_bytes = size * randchannels;
    int minVal = 30;
    int maxVal = 225;

    // Seed RNG
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Allocate pixel buffer (unsigned char for 0–255 range)
    std::vector<unsigned char> randImg(size * size * randchannels);

    // Fill with random colors
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * randchannels;
            randImg[idx + 0] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 1] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 2] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 3] = 255;
        }
    }
    if (!stbi_write_png("../images/randomImage.png", size, size, randchannels, randImg.data(), stride_in_bytes)) {
        std::cerr << "Failed to write image!" << std::endl;
    }
    #endif


    int w,h,channels;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* pixels = stbi_loadf(imageFile.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }

    size_t numPixels = size_t(w) * h;
    mainBuffers.imageSize = numPixels * 4 * sizeof(float); // forcing 4 channels
    baseBuffers.imageSize = mainBuffers.imageSize;

    float maxC = -1.0f, minC = 1e6f;

    for (size_t p = 0; p < numPixels; ++p) {
        const float* pixel = pixels + p * 4;
        for (int c = 0; c < 3; ++c) {
            if (pixel[c] > maxC) maxC = pixel[c];
            if (pixel[c] < minC) minC = pixel[c];
        }
    }
    std::cout << "Range: [" << minC*255 << "," << maxC*255 << "]" << std::endl;

    float baseDensity = (float)(w*h) / pow(constants.binCount, 3);
    constants.baseDensity = baseDensity;
    std::cout << "Base density: " << baseDensity << std::endl;

    LOG("Creating descriptor set for base density");

    baseDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, baseDescriptorSet, &baseBuffers, pixels, w, h, sigma, &constants, NULL, true);

    LOG("Creating descriptor for main loop");
    mainDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, mainDescriptorSet, &mainBuffers, pixels, w, h, sigma, &constants, &baseBuffers.transformationBuffer, false);
    
    stbi_image_free(pixels);


    LOG("Creating pipelines");
    int groupsKernel = constants.binCount / 8+1;
    int groupsInt = constants.binCount / 1;


    std::vector<const char*> baseComputeShaders;
    baseComputeShaders.push_back("../shaders/kernelX.spv");
    baseComputeShaders.push_back("../shaders/kernelY.spv");
    baseComputeShaders.push_back("../shaders/kernelZ.spv");
    baseComputeShaders.push_back("../shaders/integralX.spv");
    baseComputeShaders.push_back("../shaders/integralY.spv");
    baseComputeShaders.push_back("../shaders/integralZ.spv");
    baseComputeShaders.push_back("../shaders/transformation.spv");

    std::vector<ivec3> baseDispatches = {
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
    };
    basePipeline = createPipeline(context, baseComputeShaders, baseDispatches, baseDescriptorSet);


    std::vector<const char*> computeShaders;
    computeShaders.push_back("../shaders/histogram.spv");
    computeShaders.push_back("../shaders/kernelX.spv");
    computeShaders.push_back("../shaders/kernelY.spv");
    computeShaders.push_back("../shaders/kernelZ.spv");
    computeShaders.push_back("../shaders/integralX.spv");
    computeShaders.push_back("../shaders/integralY.spv");
    computeShaders.push_back("../shaders/integralZ.spv");
    computeShaders.push_back("../shaders/transformation.spv");
    computeShaders.push_back("../shaders/enhanceContrast.spv");

    std::vector<ivec3> dispatches = {
        ivec3{(int)w/16+1, (int)h/16+1, 1},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{(int)w/16+1, (int)h/16+1, 1},
    };
    mainPipeline = createPipeline(context, computeShaders, dispatches, mainDescriptorSet);
}

void shutdownApplication() {
    vkDeviceWaitIdle(context->device);

    destroyStageBuffer(context, &baseBuffers);
    destroyPipeline(context, &basePipeline);
    destroyDescriptorSet(context, baseDescriptorSet);

    destroyStageBuffer(context, &mainBuffers);
    destroyPipeline(context, &mainPipeline);
    destroyDescriptorSet(context, mainDescriptorSet);
    
    exitVulkan(context);
}

void runApplication(VkCommandBuffer* commandBuffer, int iterations) {
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffer;

    for(int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (vkQueueSubmit(context->computeQueue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        }
        vkQueueWaitIdle(context->computeQueue.queue);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "Iteration took " << duration.count() << " ms" << std::endl;
    }

}


int main(int argc, char* argv[]) {
    std::string fileName = "../images/input.png";
    if(argc >= 3) {
        ITERATIONS = atoi(argv[2]);
    }
    if(argc >= 2) {
        fileName = std::string("../images/") + argv[1];
    }
    initApplication(fileName);

    // Start running the shaders

    VkCommandBuffer commandBuffer;
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = context->commandPool;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(context->device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffer");
        }
    }

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    // one iteration for base density transformation
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    setupCommandBuffer(&commandBuffer, baseDescriptorSet, &basePipeline);
    vkEndCommandBuffer(commandBuffer);
    
    runApplication(&commandBuffer, 1);
    
    vkResetCommandBuffer(commandBuffer, 0);

    // MAIN LOOP
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    setupCommandBuffer(&commandBuffer, mainDescriptorSet, &mainPipeline);
    vkEndCommandBuffer(commandBuffer);
    
    runApplication(&commandBuffer, ITERATIONS);
    
    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);

    LOG("Computing finished");


    std::vector<float> outputPixels(mainBuffers.imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, &mainBuffers.imageBuffer, outputPixels.data());

    std::vector<uint8_t> outputBytes(mainBuffers.imageBuffer.extent.width * mainBuffers.imageBuffer.extent.height * 4);
    for (size_t i = 0; i < outputPixels.size(); ++i) {
        float v = std::min(std::max(outputPixels[i], 0.0f), 1.0f);
        outputBytes[i] = static_cast<uint8_t>(v * 255.0f);
    }

    if(!stbi_write_png("imageOutput.png", static_cast<int>(mainBuffers.imageBuffer.extent.width), static_cast<int>(mainBuffers.imageBuffer.extent.height), 4, outputBytes.data(), mainBuffers.imageBuffer.extent.width*4)) {
        LOG_ERROR("Failed saving output image");
    }

    LOG("Loading histogram from gpu");
    uint32_t count = pow(constants.binCount, 3);
    std::vector<float> histogram(count);
    getDataFromBufferWithStagingBuffer(context, &mainBuffers.kernelBuffer, histogram.data(), sizeof(float) * count);

    LOG("Calculating 2D accumulated histogram");
    int histogram2DCount = pow(constants.binCount, 2);
    std::vector<float> histogram2D(histogram2DCount);
    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        int x = i % constants.binCount;
        int y = floor(i/constants.binCount);
        for(int z = 0; z < constants.binCount; ++z) {
            int getIndex = x + z * constants.binCount + y * constants.binCount * constants.binCount;
            histogram2D[i] += histogram[getIndex];
        }
    }

    float max = 0;
    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        if(histogram2D[i]>max) {
            max = histogram2D[i];
        }
    }

    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        histogram2D[i] = histogram2D[i] / max * 255.0f * 10;
    }

    std::vector<uint8_t> histogram2DBytes(histogram2DCount);
    for (size_t i = 0; i < histogram2DCount; ++i) {
        histogram2DBytes[i] = static_cast<uint8_t>(histogram2D[i]);
    }

    if(!stbi_write_png("histogram.png", constants.binCount, constants.binCount, 1, histogram2DBytes.data(), constants.binCount)) {
        LOG_ERROR("Failed saving histogram");
    }

    #if INTEGRALS

    for(int i = 0; i < mainBuffers.integralImages.size(); ++i) {
        std::vector<float> integral(count);
        getDataFromBufferWithStagingBuffer(context, &mainBuffers.integralImages[i], integral.data(), sizeof(float) * count);

        LOG("Calculating 2D accumulated histogram");
        int integral2DCount = pow(constants.binCount, 2);
        std::vector<float> integral2D(integral2DCount);
        for(uint32_t i = 0; i < integral2DCount; ++i) {
            int x = i % constants.binCount;
            int y = floor(i/constants.binCount);
            for(int z = 0; z < constants.binCount; ++z) {
                int getIndex = z + x * constants.binCount + y * constants.binCount * constants.binCount;
                integral2D[i] += integral[getIndex];
            }
        }

        float max = 0;
        for(uint32_t i = 0; i < integral2DCount; ++i) {
            if(integral2D[i]>max) {
                max = integral2D[i];
            }
        }

        for(uint32_t i = 0; i < integral2DCount; ++i) {
            integral2D[i] = integral2D[i] / max * 255.0f;
        }

        std::vector<uint8_t> integral2DBytes(integral2DCount);
        for (size_t i = 0; i < integral2DCount; ++i) {
            integral2DBytes[i] = static_cast<uint8_t>(integral2D[i]);
        }

        if(!stbi_write_png(("integral" + std::to_string(i) + ".png").c_str(), constants.binCount, constants.binCount, 1, integral2DBytes.data(), constants.binCount)) {
            LOG_ERROR("Failed saving histogram");
        }
    }

    #endif

    #if TRANSFORMATION
    LOG("Loading transformation");

    // read back the cube of vec4s (count = binCount^3)
    std::vector<float> transformation(count * 4);
    getDataFromBufferWithStagingBuffer(
        context, 
        &mainBuffers.transformationBuffer, 
        transformation.data(), 
        sizeof(float) * count * 4
    );

    std::vector<float> bTransformation(count * 4);
    getDataFromBufferWithStagingBuffer(
        context, 
        &baseBuffers.transformationBuffer, 
        bTransformation.data(), 
        sizeof(float) * count * 4
    );

    // parameters
    int binCount = constants.binCount;
    int tilesX = (int)std::ceil(std::sqrt((float)binCount)); // e.g. 16 if binCount=256
    int tilesY = (binCount + tilesX - 1) / tilesX;

    int sliceSize = binCount * binCount;    // one z-slice
    int outWidth  = binCount * tilesX;
    int outHeight = binCount * tilesY;

    // buffer for packed image
    std::vector<float> packed(outWidth * outHeight * 4, 0.0f);

    // pack the 3D cube into 2D tiled slices
    for (int z = 0; z < binCount; ++z) {
        int tileX = z % tilesX;
        int tileY = z / tilesX;

        for (int y = 0; y < binCount; ++y) {
            for (int x = 0; x < binCount; ++x) {
                int inIndex  = (x + y * binCount + z * binCount * binCount) * 4;
                int outX     = x + tileX * binCount;
                int outY     = y + tileY * binCount;
                int outIndex = (outX + outY * outWidth) * 4;

                packed[outIndex + 0] = transformation[inIndex + 0] - bTransformation[inIndex + 0];
                packed[outIndex + 1] = transformation[inIndex + 1] - bTransformation[inIndex + 1];
                packed[outIndex + 2] = transformation[inIndex + 2] - bTransformation[inIndex + 2];
                packed[outIndex + 3] = 1;

                if(x == binCount-1 && y == binCount-1 && z == binCount-1) {
                    std::cout << packed[outIndex + 0] << ", "  << packed[outIndex + 1] << ", "  << packed[outIndex + 2] << std::endl;
                }
            }
        }
    }

    LOG("Saving transformation.png");

    auto clampf = [](float v, float lo, float hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
    };

    std::vector<uint8_t> packed8(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        packed8[i] = (uint8_t)clampf(packed[i] * 255.0f, 0.0f, 255.0f);
    }

    if (!stbi_write_png("transformation.png", outWidth, outHeight, 4, packed8.data(), outWidth * 4)) {
        LOG_ERROR("Failed saving transformation");
    }
    #endif

   
    shutdownApplication();
    return 1;
}
