#include "vulkan_base/vulkan_base.h"
#include <vector>
#include <string>

void saveHistogramAsPng(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName);

void saveIntegralsAsPngs(VulkanContext* context, std::vector<VulkanBuffer>& integrals, uint32_t binCount);

void saveImageAsPng(VulkanContext* context, VulkanImage* image, uint32_t imageSize);

void saveTransformationAsPng(VulkanContext* context, VulkanBuffer* transformation, VulkanBuffer* baseTransformation, uint32_t binCount);

void printProgress(double percentage, int width, std::string bar, float avgTime);