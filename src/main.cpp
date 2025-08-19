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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define ITERATIONS 10

VulkanContext* context;
VulkanDescriptorSet* descriptorSetInfo;
VulkanPipeline pipeline;

VulkanBuffer uniformsBuffer;
VulkanImage imageBuffer;
size_t imageSize;
VulkanBuffer histogramBuffer;
VulkanBuffer kernelBuffer;
VulkanBuffer kernelBufferTemp;
VulkanBuffer expLookUp;

// Integral images
std::vector<VulkanBuffer> integralImages(8);

VulkanBuffer transformationBuffer;



struct UniformData {
    uint32_t binCount = 128;
    uint32_t kernelRadius = 5;
}constants;

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

    LOG("Creating descriptor set");
    descriptorSetInfo = initDescriptorSet();

    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);   // Constants
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);    // Image
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Histogram
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // KernelBuffer
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Temp kernelBuffer
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // exp look up table
    for(int i = 0; i < integralImages.size(); ++i)
        addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // integral images
    
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Transformationbuffer
    createDescriptorSet(context, descriptorSetInfo);


    LOG("Filling descriptorsets with Buffers");
    descriptorSetInfo->addBufferAndData(context, 
        &uniformsBuffer, 
        &constants, 
        sizeof(constants), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    int w,h,channels;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* pixels = stbi_loadf(imageFile.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }
    imageSize = size_t(w) * h * 4 * sizeof(float); // forcing 4 channels

    descriptorSetInfo->addImageAndData(
        context, 
        &imageBuffer, pixels, imageSize,
        w, h, 1, 
        VK_FORMAT_R32G32B32A32_SFLOAT, //more precision while calculating the transformations
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    stbi_image_free(pixels);

    uint32_t binC = constants.binCount;
    descriptorSetInfo->addBufferAndData(context, 
        &histogramBuffer, 
        NULL, 
        sizeof(uint32_t) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSetInfo->addBufferAndData(context, 
        &kernelBuffer, 
        NULL, 
        sizeof(float) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSetInfo->addBufferAndData(context, 
        &kernelBufferTemp, 
        NULL, 
        sizeof(float) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    float lookUp[constants.kernelRadius * 2 + 1];
    int radius = (int)constants.kernelRadius;
    for(int i = -radius; i <= radius; ++i) {
        lookUp[i+radius] = std::exp(-(i * i) / (2.0f * sigma * sigma));
    }
    descriptorSetInfo->addBufferAndData(context, 
        &expLookUp, 
        lookUp, 
        sizeof(float) * (constants.kernelRadius*2+1), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    for(auto& intImg : integralImages) {
        descriptorSetInfo->addBufferAndData(context, 
            &intImg, 
            NULL, 
            sizeof(float) * binC * binC * binC, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    descriptorSetInfo->addBufferAndData(context, 
        &transformationBuffer, 
        NULL, 
        4*sizeof(float) * binC * binC * binC, // vec3 still need 16 bytes!!!
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    LOG("Load descriptor set");
    fillDescriptorSet(context, descriptorSetInfo);

    LOG("Creating pipeline");
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

    int groupsKernel = constants.binCount / 8+1;
    int groupsInt = constants.binCount / 1;
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
    pipeline = createPipeline(context, computeShaders, dispatches, descriptorSetInfo);
}

void shutdownApplication() {
    vkDeviceWaitIdle(context->device);

    destroyBuffer(context, &transformationBuffer);
    for(auto& intImg : integralImages) {
        destroyBuffer(context, &intImg);
    }
    destroyBuffer(context, &expLookUp);
    destroyBuffer(context, &kernelBufferTemp);
    destroyBuffer(context, &kernelBuffer);
    destroyBuffer(context, &histogramBuffer);
    destroyImage(context, &imageBuffer);
    destroyBuffer(context, &uniformsBuffer);
    destroyPipeline(context, &pipeline);
    destroyDescriptorSet(context, descriptorSetInfo);
    
    exitVulkan(context);
}

void recordCommandBuffer(VkCommandBuffer* commandBuffer, VulkanDescriptorSet* descriptorSet, VulkanPipeline* pipeline) {
    vkCmdBindDescriptorSets(
        *commandBuffer, 
        VK_PIPELINE_BIND_POINT_COMPUTE, 
        pipeline->pipelineLayout, 
        0, 
        1, 
        &descriptorSet->descriptorSet, 
        0, 
        0
    );

    for (int i = 0; i < pipeline->pipelines.size(); ++i) {
        vkCmdBindPipeline(
            *commandBuffer, 
            VK_PIPELINE_BIND_POINT_COMPUTE, 
            pipeline->pipelines[i]
        );

        ivec3 dispatchSize = pipeline->dispatchSizes[i];
        vkCmdDispatch(*commandBuffer, dispatchSize.x, dispatchSize.y, dispatchSize.z);
        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                *commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &barrier,
                0, nullptr,
                0, nullptr
            );
        }
    }
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
    if(argc == 2) {
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
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    recordCommandBuffer(&commandBuffer, descriptorSetInfo, &pipeline);
    vkEndCommandBuffer(commandBuffer);
    
    runApplication(&commandBuffer, ITERATIONS);
    
    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);

    LOG("Computing finished");

    std::vector<float> outputPixels(imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, &imageBuffer, outputPixels.data());

    std::vector<uint8_t> outputBytes(imageBuffer.extent.width * imageBuffer.extent.height * 4);
    for (size_t i = 0; i < outputPixels.size(); ++i) {
        float v = std::min(std::max(outputPixels[i], 0.0f), 1.0f);
        outputBytes[i] = static_cast<uint8_t>(v * 255.0f);
    }

    if(!stbi_write_png("imageOutput.png", static_cast<int>(imageBuffer.extent.width), static_cast<int>(imageBuffer.extent.height), 4, outputBytes.data(), imageBuffer.extent.width*4)) {
        LOG_ERROR("Failed saving output image");
    }

    LOG("Loading histogram from gpu");
    uint32_t count = pow(constants.binCount, 3);
    std::vector<float> histogram(count);
    getDataFromBufferWithStagingBuffer(context, &kernelBuffer, histogram.data(), sizeof(float) * count);

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
        histogram2D[i] = histogram2D[i] / max * 255.0f;
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
