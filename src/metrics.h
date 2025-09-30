#include "vulkan_base/vulkan_base.h"
#include <string>
#include <vector>

struct Result {
    std::string fileName;
    int lastIteration;
    std::vector<int> iterations;
    std::vector<float> mse;
    std::vector<float> gradient;
};

float calculateStandardDeviation(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount);

float calculateGradient(VulkanContext* context, VulkanImage* gradientImage, uint32_t imageSize);
