#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include "vulkan/vulkan_core.h"
#include "vulkan_base/vulkan_base.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define ITERATIONS 1

VulkanContext* context;
VulkanDescriptorSet* descriptorSetInfo;
VulkanPipeline pipeline;

VulkanBuffer uniformsBuffer;
VulkanImage imageBuffer;
size_t imageSize;
VulkanBuffer histogramBuffer;

struct UniformData {
    uint32_t kernelRadius = 3;
    uint32_t binCount = 256;
}constants;

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

    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
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

    descriptorSetInfo->addBufferAndData(context, 
        &histogramBuffer, 
        NULL, 
        sizeof(uint) * 256 * 256 * 256, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    LOG("Load descriptor set");
    fillDescriptorSet(context, descriptorSetInfo);

    LOG("Creating pipeline");
    std::vector<const char*> computeShaders;
    computeShaders.push_back("../shaders/histogram.spv");
    std::vector<ivec3> dispatches = {
        ivec3{(int)w/16+1, (int)h/16+1, 1},
    };
    pipeline = createPipeline(context, computeShaders, dispatches, descriptorSetInfo);
}

void shutdownApplication() {
    vkDeviceWaitIdle(context->device);
    
    destroyBuffer(context, &histogramBuffer);
    destroyImage(context, &imageBuffer);
    destroyBuffer(context, &uniformsBuffer);
    destroyPipeline(context, &pipeline);
    destroyDescriptorSet(context, descriptorSetInfo);
    
    exitVulkan(context);
}

void runApplication() {
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

    vkCmdBindDescriptorSets(
        commandBuffer, 
        VK_PIPELINE_BIND_POINT_COMPUTE, 
        pipeline.pipelineLayout, 
        0, 
        1, 
        &descriptorSetInfo->descriptorSet, 
        0, 
        0
    );

    for (int i = 0; i < pipeline.pipelines.size(); ++i) {
        vkCmdBindPipeline(
            commandBuffer, 
            VK_PIPELINE_BIND_POINT_COMPUTE, 
            pipeline.pipelines[i]
        );

        ivec3 dispatchSize = pipeline.dispatchSizes[i];
        vkCmdDispatch(commandBuffer, dispatchSize.x, dispatchSize.y, dispatchSize.z);
        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &barrier,
                0, nullptr,
                0, nullptr
            );
        }
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(context->computeQueue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit compute command buffer!");
    }

    vkQueueWaitIdle(context->computeQueue.queue);

    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);
}


int main(int argc, char* argv[]) {
    std::string fileName = "../images/input.png";
    if(argc == 2) {
        fileName = std::string("../images/") + argv[1];
    }
    initApplication(fileName);

    for (int i = 0; i < ITERATIONS; ++i) {
        runApplication();
    }
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
    std::vector<uint> histogram(count);
    getDataFromBufferWithStagingBuffer(context, &histogramBuffer, histogram.data(), sizeof(int) * count);

    LOG("Calculating 2D accumulated histogram");
    int histogram2DCount = pow(constants.binCount, 2);
    std::vector<uint> histogram2D(histogram2DCount);
    for(uint i = 0; i < histogram2DCount; ++i) {
        int x = i % constants.binCount;
        int y = floor(i/constants.binCount);
        for(int z = 0; z < constants.binCount; ++z) {
            int getIndex = z + x * constants.binCount + y * constants.binCount * constants.binCount;
            histogram2D[i] += histogram[getIndex];
        }
    }

    uint max = 0;
    for(uint i = 0; i < histogram2DCount; ++i) {
        if(histogram2D[i]>max) {
            max = histogram2D[i];
        }
    }

    for(uint i = 0; i < histogram2DCount; ++i) {
        histogram2D[i] = (int)((float)histogram2D[i] / (float) max * 255);
    }

    std::vector<uint8_t> histogram2DBytes(histogram2DCount);
    for (size_t i = 0; i < histogram2DCount; ++i) {
        histogram2DBytes[i] = std::min(histogram2D[i], 255u);
    }

    if(!stbi_write_png("histogram.png", constants.binCount, constants.binCount, 1, histogram2DBytes.data(), constants.binCount)) {
        LOG_ERROR("Failed saving histogram");
    }
   
    shutdownApplication();
    return 1;
}
