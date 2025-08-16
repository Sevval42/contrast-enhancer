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
VulkanBuffer ioBuffer;
VulkanBuffer firstTempBuffer;
VulkanImage imageBuffer;
size_t imageSize;
float myData[] = {1, 2, 3, 4, 5};

struct UniformData {
    float offset = 4;
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

    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    addDescriptorSetLayout(descriptorSetInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    createDescriptorSet(context, descriptorSetInfo);


    LOG("Filling descriptorsets with Buffers");
    descriptorSetInfo->addBufferAndData(
        context, 
        &ioBuffer, 
        myData,
        sizeof(myData), 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT , 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSetInfo->addBufferAndData(
        context,
        &firstTempBuffer,
        &uniformData,
        sizeof(uniformData),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    int w,h,channels;
    unsigned char* pixels = stbi_load("../images/image.png", &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }
    imageSize = size_t(w) * h * channels;

    descriptorSetInfo->addImageAndData(
        context, 
        &imageBuffer, pixels, imageSize,
        w, h, 1, 
        VK_FORMAT_R8G8B8A8_UNORM, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    stbi_image_free(pixels);

    LOG("Load descriptor set");
    fillDescriptorSet(context, descriptorSetInfo);

    LOG("Creating pipeline");
    std::vector<const char*> computeShaders;
    computeShaders.push_back("../shaders/test1.spv");
    computeShaders.push_back("../shaders/test2.spv");
    computeShaders.push_back("../shaders/test3.spv");
    std::vector<ivec3> dispatches = {
        ivec3{5, 1, 1},
        ivec3{1, 1, 1},
        ivec3{(int)w/16+1, (int)h/16+1, 1},
    };
    pipeline = createPipeline(context, computeShaders, dispatches, descriptorSetInfo);
}

void shutdownApplication() {
    vkDeviceWaitIdle(context->device);
    
    destroyImage(context, &imageBuffer);
    destroyBuffer(context, &firstTempBuffer);
    destroyBuffer(context, &ioBuffer);
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

    float data[5];
    getDataFromBufferWithStagingBuffer(context, &ioBuffer, data, sizeof(myData));

    std::cout << "[" << data[0];
    for (int i = 1; i < 5; i++) {
        std::cout<< ", " << data[i];
    }
    std::cout << "]" << std::endl;

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

    shutdownApplication();
    return 1;
}
