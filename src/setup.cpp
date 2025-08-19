#include "setup.h"
#include "vulkan_base/vulkan_base.h"
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"

void setupDescriptorSet(
    VulkanContext* context,
    VulkanDescriptorSet* descriptorSet,
    StageBuffers* buffers,
    float* image, int w, int h, float sigma,
    UniformData* constants,
    float baseDensity
) {
    std::vector<float> baseHistogram((int)pow(constants->binCount, 3));
    
    for(int i = 0; i < baseHistogram.size(); ++i) {
        baseHistogram[i] = baseDensity;
    }

    buffers->integralImages = std::vector<VulkanBuffer>(8);

    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);   // Constants
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);    // Image
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Histogram
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // KernelBuffer
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Temp kernelBuffer
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // exp look up table
    for(int i = 0; i < buffers->integralImages.size(); ++i)
        addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // integral images
    
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // Transformationbuffer
    createDescriptorSet(context, descriptorSet);

    LOG("Filling descriptorsets with Buffers");
    descriptorSet->addBufferAndData(context, 
        &buffers->uniformsBuffer, 
        constants, 
        sizeof(UniformData), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    
    descriptorSet->addImageAndData(
        context, 
        &buffers->imageBuffer, image, buffers->imageSize,
        w, h, 1, 
        VK_FORMAT_R32G32B32A32_SFLOAT, //more precision while calculating the transformations
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    uint32_t binC = constants->binCount;
    descriptorSet->addBufferAndData(context, 
        &buffers->histogramBuffer, 
        NULL, 
        sizeof(uint32_t) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSet->addBufferAndData(context, 
        &buffers->kernelBuffer, 
        baseHistogram.data(), 
        sizeof(float) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSet->addBufferAndData(context, 
        &buffers->kernelBufferTemp, 
        NULL, 
        sizeof(float) * binC * binC * binC, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    float lookUp[constants->kernelRadius * 2 + 1];
    int radius = (int)constants->kernelRadius;
    for(int i = -radius; i <= radius; ++i) {
        lookUp[i+radius] = std::exp(-(i * i) / (2.0f * sigma * sigma));
    }
    descriptorSet->addBufferAndData(context, 
        &buffers->expLookUp, 
        lookUp, 
        sizeof(float) * (constants->kernelRadius*2+1), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    for(auto& intImg : buffers->integralImages) {
        descriptorSet->addBufferAndData(context, 
            &intImg, 
            NULL, 
            sizeof(float) * binC * binC * binC, 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    descriptorSet->addBufferAndData(context, 
        &buffers->transformationBuffer, 
        NULL, 
        4*sizeof(float) * binC * binC * binC, // vec3 still need 16 bytes!!!
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    LOG("Load descriptor set");
    fillDescriptorSet(context, descriptorSet);
}

void setupCommandBuffer(VkCommandBuffer *commandBuffer, VulkanDescriptorSet *descriptorSet, VulkanPipeline *pipeline) {
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

void destroyStageBuffer(VulkanContext* context, StageBuffers *stageBuffers) {
    destroyBuffer(context, &stageBuffers->transformationBuffer);
    for(auto& intImg : stageBuffers->integralImages) {
        destroyBuffer(context, &intImg);
    }
    destroyBuffer(context, &stageBuffers->expLookUp);
    destroyBuffer(context, &stageBuffers->kernelBufferTemp);
    destroyBuffer(context, &stageBuffers->kernelBuffer);
    destroyBuffer(context, &stageBuffers->histogramBuffer);
    destroyImage(context, &stageBuffers->imageBuffer);
    destroyBuffer(context, &stageBuffers->uniformsBuffer);
}