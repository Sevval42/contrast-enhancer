#include "vulkan_base/vulkan_base.h"

/*
    Helper methods for setting up the pipeline and running the shaders
*/

struct UniformData {
    uint32_t binCount;
    uint32_t kernelRadius;
    float baseDensity;
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

    // Temp buffer for calculating rotated integrals
    std::vector<VulkanBuffer> tempIntegrals;

    VulkanBuffer transformationBuffer;
    VulkanBuffer metric;
    VulkanImage gradient;
};

void destroyStageBuffer(VulkanContext* context, StageBuffers* stageBuffers);

/*
    The program is setup so each shader is in one vulkan pipeline object.
    To simplify everything, each vk_pipeline uses the same descriptor set.
*/
void setupDescriptorSet(
    VulkanContext* context,
    VulkanDescriptorSet* descriptorSet,
    StageBuffers* buffers,
    float* image, int w, int h, float sigma,
    UniformData* constants,
    VulkanBuffer* baseTransformation,
    bool isBaseDensity
);

/*
    This method fills the commandbuffer with the pipelines and data from the descriptorset.
    vkBeginCommandBuffer() and vkEndCommandBuffer() need to be called before and after this method separately.
*/
void setupCommandBuffer(VkCommandBuffer* commandBuffer, VulkanDescriptorSet* descriptorSet, VulkanPipeline* pipeline);
