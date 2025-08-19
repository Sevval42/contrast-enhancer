#include "vulkan_base/vulkan_base.h"

struct UniformData {
    uint32_t binCount;
    uint32_t kernelRadius;
};

struct StageBuffers {
    VulkanBuffer uniformsBuffer;
    VulkanImage imageBuffer;
    size_t imageSize;
    VulkanBuffer histogramBuffer;
    VulkanBuffer kernelBuffer;
    VulkanBuffer kernelBufferTemp;
    VulkanBuffer expLookUp;

    // Integral images
    std::vector<VulkanBuffer> integralImages;

    VulkanBuffer transformationBuffer;
};

void destroyStageBuffer(VulkanContext* context, StageBuffers* stageBuffers);

void setupDescriptorSet(
    VulkanContext* context,
    VulkanDescriptorSet* descriptorSet,
    StageBuffers* buffers,
    float* image, int w, int h, float sigma,
    UniformData* constants,
    float baseDensity
);

void setupCommandBuffer(VkCommandBuffer* commandBuffer, VulkanDescriptorSet* descriptorSet, VulkanPipeline* pipeline);
