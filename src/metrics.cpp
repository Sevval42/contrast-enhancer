#include "metrics.h"

float calculateStandardDeviation(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount){    
    uint32_t count = pow(binCount, 3);
    std::vector<float> data(count);
    getDataFromBufferWithStagingBuffer(context, histogram, data.data(), sizeof(float) * count);

    float sum = 0.0f;
    for (float value : data) {
        sum += value;
    }
    float mean = sum / data.size();

    float variance = 0.0f;
    for (float value : data) {
        float diff = value - mean;
        variance += diff * diff;
    }
    variance /= (data.size() - 1);

    return std::sqrt(variance);
}

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

