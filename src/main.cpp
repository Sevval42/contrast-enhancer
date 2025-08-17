#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
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
    float offset = 4;
    uint32_t kernelRadius = 3;
    float defaultColor = 1;
}uniformData;

void initApplication() {

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
        &uniformData, 
        sizeof(uniformData), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    int w,h,channels;
    unsigned char* pixels = stbi_load("../images/image3.png", &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }
    imageSize = size_t(w) * h * 4;

    descriptorSetInfo->addImageAndData(
        context, 
        &imageBuffer, pixels, imageSize,
        w, h, 1, 
        VK_FORMAT_R8G8B8A8_UNORM, 
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
    computeShaders.push_back("../shaders/test1.spv");
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
    initApplication();

    for (int i = 0; i < ITERATIONS; ++i) {
        runApplication();
    }
    std::vector<uint8_t> outputPixels(imageSize);
    getDataFromImageWithStagingBuffer(context, &imageBuffer, outputPixels.data());

    if(!stbi_write_png("output.png", static_cast<int>(imageBuffer.extent.width), static_cast<int>(imageBuffer.extent.height), 4, outputPixels.data(), imageBuffer.extent.width*4)) {
        LOG_ERROR("Failed saving output image");
    }

    
    size_t count = 256*256*256;
    std::vector<uint> histogram(count);
    getDataFromBufferWithStagingBuffer(context, &histogramBuffer, histogram.data(), sizeof(int) * count);

    std::vector<uint> histogram2D(256*256);

    for(uint i = 0; i < 256*256; ++i) {
        int x = i % 256;
        int y = floor(i/256);
        for(int z = 0; z < 256; ++z) {
            int getIndex = x + z * 256 + y * 256 * 256;
            histogram2D[i] += histogram[getIndex];
        }
    }

    uint max = 0;
    for(uint i = 0; i < 256*256; ++i) {
        if(histogram2D[i]>max) {
            max = histogram2D[i];
        }
    }

    for(uint i = 0; i < 256*256; ++i) {
        histogram2D[i] = (int)((float)histogram2D[i] / (float) max * 255);
    }

    std::vector<uint8_t> histogram2DBytes(256*256);
    for (size_t i = 0; i < 256*256; ++i) {
        histogram2DBytes[i] = std::min(histogram2D[i], 255u);
    }
    stbi_write_png("output.png", 256, 256, 1, histogram2DBytes.data(), 256);

   
    shutdownApplication();
    return 1;
}
