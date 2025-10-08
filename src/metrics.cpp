#include "metrics.h"

float calculateGradient(VulkanContext* context, VulkanImage* gradientImage, uint32_t imageSize) {
    std::vector<float> pixels(imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, gradientImage, pixels.data());

    float sum = 0;

    for(int i = 0; i < pixels.size(); ++i){
        if(i%4 == 3) continue;
        sum += pixels[i] / 3.0;
    }

    sum /= static_cast<float>(pixels.size());

    return sum;
}

