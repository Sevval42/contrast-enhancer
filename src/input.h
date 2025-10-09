#include "vulkan_base/vulkan_base.h"
#include <vector>

#define STRIDE 2048

std::vector<float> loadImage(std::string fileName, int* width, int* height, int* channels);

std::vector<float> loadFits(
    const char* fileNameX,
    const char* fileNameY,
    const char* fileNameZ,
    int* width,
    int* height,
    float weightR = 1.0f,
    float weightG = 1.0f,
    float weightB = 1.0f
);

std::vector<float> loadCsv(const char* filename, int labelCount, int* width, int* height, int* channels);
void saveNDToCsv(const char* filename, int labelCount, VulkanContext* context, VulkanImage* image, uint32_t imageSize);