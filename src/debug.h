#include "vulkan_base/vulkan_base.h"
#include <cstdint>
#include <vector>
#include <string>

/*
    Methods for debugging purposes
*/

void saveHistogramAsPng(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName, const char dimension);
void saveHistogramAsCsv(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName);
float calculateStandardDeviation(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount);

void saveIntegralsAsPngs(VulkanContext* context, std::vector<VulkanBuffer>& integrals, uint32_t binCount);
void saveRotatedIntegralAsCsv(VulkanContext* context, VulkanBuffer* integral, uint32_t binCount, const char* fileName, int direction);

void saveImageAsPng(VulkanContext* context, VulkanImage* image, uint32_t imageSize);
void saveDataAsCsv(VulkanContext* context, VulkanImage* image, uint32_t imageSize, const char* fileName, int skipFactor);

void saveTransformationAsPng(VulkanContext* context, VulkanBuffer* transformation, VulkanBuffer* baseTransformation, uint32_t binCount);
void saveTransformationAsCsv(VulkanContext* context, VulkanBuffer* transformation, VulkanBuffer* baseTransformation, uint32_t binCount, const char* fileName, int vectorCount);

void printProgress(double percentage, int width, std::string bar, float avgTime);
