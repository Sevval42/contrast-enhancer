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
        0,
        0
    };

    int w,h,channels;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* pixels = stbi_loadf(imageFile.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }
        mainBuffers.imageSize = size_t(w) * h * 4 * sizeof(float); // forcing 4 channels
        baseBuffers.imageSize = mainBuffers.imageSize;

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
            int getIndex = z + x * constants.binCount + y * constants.binCount * constants.binCount;
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
   
    shutdownApplication();
    return 1;
}
