#include "setup.h"
#include "vulkan_base/vulkan_base.h"
#include <iostream>
#include <vulkan/vulkan_core.h>
#include "stb_image.h"
#include "stb_image_write.h"

void setupDescriptorSet(
    VulkanContext* context,
    VulkanDescriptorSet* descriptorSet,
    StageBuffers* buffers,
    float* image, int w, int h, float sigma,
    UniformData* constants,
    VulkanBuffer* baseTransformation,
    bool isBaseDensityRun
) {
    std::vector<float> baseHistogram((int)pow(constants->binCount, 3));

    // sizes:
    uint32_t binC = constants->binCount;
    uint32_t histogramSize = sizeof(uint32_t) * binC * binC * binC;
    uint32_t integralSize = sizeof(float) * binC * binC * binC;
    uint32_t transformationSize = 4 * sizeof(float) * (binC+1) * (binC+1) * (binC+1);
    
    for(int i = 0; i < baseHistogram.size(); ++i) {
        baseHistogram[i] = 0;
    }

    buffers->integralImages = std::vector<VulkanBuffer>(14); // 8 Corners + 6 Faces for a Cube
    buffers->tempIntegrals = std::vector<VulkanBuffer>(6); // One (big) temp Buffer for every face

    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);       // 0: Constant
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);        // 1: Image
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);       // 2: Histogram
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);       // 3: KernelBuffer
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);       // 4: exp look up table
    for(int i = 0; i < buffers->integralImages.size(); ++i)
        addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // 5-18: integral image

    for(int i = 0; i < buffers->tempIntegrals.size(); ++i)
        addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // 19-24: tempIntegrals
    
    addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);       // 25: Transformation buffer
    if(!isBaseDensityRun) {
        addDescriptorSetLayout(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);   // 26: Basedensity transformation buffer
    }

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

    descriptorSet->addBufferAndData(context, 
        &buffers->histogramBuffer, 
        NULL, 
        histogramSize, 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    descriptorSet->addBufferAndData(context, 
        &buffers->kernelBuffer, 
        baseHistogram.data(), 
        integralSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    size_t lookupCount = (int)pow(constants->kernelRadius * 2 + 1, 3);
    float lookUp[lookupCount];
    int radius = (int)constants->kernelRadius;
    int idx = 0;
    float sigma2 = 2*sigma*sigma;
    for (int z = -radius; z <= radius; ++z) {
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                float dist2 = float(x*x + y*y + z*z);
                lookUp[idx++] = std::exp(-dist2 / sigma2);
            }
        }
    }

    descriptorSet->addBufferAndData(context, 
        &buffers->expLookUp, 
        lookUp, 
        sizeof(float) * lookupCount, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    for(auto& intImg : buffers->integralImages) {
        descriptorSet->addBufferAndData(context, 
            &intImg, 
            NULL, 
            integralSize, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    for(auto& tempInt : buffers->tempIntegrals) {
        descriptorSet->addBufferAndData(context, 
            &tempInt, 
            NULL, 
            integralSize * 4, // 4 or 5 temp histograms for every face integral 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    descriptorSet->addBufferAndData(context, 
        &buffers->transformationBuffer, 
        NULL, 
        transformationSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    if(!isBaseDensityRun) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = baseTransformation->buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = transformationSize;

        VulkanDescriptorBufferInfo info{};
        info.bufferInfo = bufferInfo;
        info.type = VulkanDescriptorBufferInfo::Type::BUFFER;

        descriptorSet->buffers.push_back(info);
    }
    

    LOG("Load descriptor set");
    fillDescriptorSet(context, descriptorSet);
}

void setupCommandBuffer(VkCommandBuffer* commandBuffer, VulkanDescriptorSet* descriptorSet, VulkanPipeline* pipeline) {
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
    for(auto& tempInt : stageBuffers->tempIntegrals) {
        destroyBuffer(context, &tempInt);
    }
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