#include <iostream>
#include "debug.h"
#include "vulkan_base/vulkan_base.h"
#include "stb_image.h"
#include "stb_image_write.h"

void saveHistogramAsPng(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName) {
    LOG("Loading histogram from gpu");
    uint32_t count = pow(binCount, 3);
    std::vector<float> data(count);
    getDataFromBufferWithStagingBuffer(context, histogram, data.data(), sizeof(float) * count);

    LOG("Calculating 2D accumulated histogram");
    int histogram2DCount = pow(binCount, 2);
    std::vector<float> histogram2D(histogram2DCount);
    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        int x = i % binCount;
        int y = floor(i/binCount);
        for(int z = 0; z < binCount; ++z) {
            int getIndex = z + x * binCount + y * binCount * binCount;
            histogram2D[i] += data[getIndex];
        }
    }

    float max = 0;
    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        if(histogram2D[i]>max) {
            max = histogram2D[i];
        }
    }

    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        histogram2D[i] = histogram2D[i] / max * 255.0f * 1;
    }

    std::vector<uint8_t> histogram2DBytes(histogram2DCount);
    for (size_t i = 0; i < histogram2DCount; ++i) {
        histogram2DBytes[i] = static_cast<uint8_t>(histogram2D[i]);
    }

    if(!stbi_write_png(fileName, binCount, binCount, 1, histogram2DBytes.data(), binCount)) {
        LOG_ERROR("Failed saving histogram");
    }
}

void saveIntegralsAsPngs(VulkanContext* context, std::vector<VulkanBuffer> integrals, uint32_t binCount) {
    size_t count = pow(binCount, 3);
    for(int i = 0; i < integrals.size(); ++i) {
        std::vector<float> integral(count);
        getDataFromBufferWithStagingBuffer(context, &integrals[i], integral.data(), sizeof(float) * count);

        LOG("Calculating 2D accumulated histogram");
        int integral2DCount = pow(binCount, 2);
        std::vector<float> integral2D(integral2DCount);
        for(uint32_t i = 0; i < integral2DCount; ++i) {
            int x = i % binCount;
            int y = floor(i/binCount);
            for(int z = 0; z < binCount; ++z) {
                int getIndex = z + x * binCount + y * binCount * binCount;
                integral2D[i] += integral[getIndex];
            }
        }

        float max = 0;
        for(uint32_t i = 0; i < integral2DCount; ++i) {
            if(integral2D[i]>max) {
                max = integral2D[i];
            }
        }

        for(uint32_t i = 0; i < integral2DCount; ++i) {
            integral2D[i] = integral2D[i] / max * 255.0f;
        }

        std::vector<uint8_t> integral2DBytes(integral2DCount);
        for (size_t i = 0; i < integral2DCount; ++i) {
            integral2DBytes[i] = static_cast<uint8_t>(integral2D[i]);
        }

        if(!stbi_write_png(("integral" + std::to_string(i) + ".png").c_str(), binCount, binCount, 1, integral2DBytes.data(), binCount)) {
            LOG_ERROR("Failed saving histogram");
        }
    }
}

void saveImageAsPng(VulkanContext* context, VulkanImage* image, uint32_t imageSize) {
    std::vector<float> outputPixels(imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, image, outputPixels.data());

    std::vector<uint8_t> outputBytes(image->extent.width * image->extent.height * 4);
    for (size_t i = 0; i < outputPixels.size(); ++i) {
        float v = std::min(std::max(outputPixels[i], 0.0f), 1.0f);
        outputBytes[i] = static_cast<uint8_t>(v * 255.0f);
    }

    if(
        !stbi_write_png(
        "imageOutput.png", 
        static_cast<int>(image->extent.width), static_cast<int>(image->extent.height), 4, 
        outputBytes.data(), image->extent.width*4)
    ) {
        LOG_ERROR("Failed saving output image");
    }

}

void saveTransformationAsPng(VulkanContext* context, VulkanBuffer* transformation, VulkanBuffer* baseTransformation, uint32_t binCount) {
    LOG("Loading transformation");
    size_t count = pow(binCount, 3);

    // read back the cube of vec4s (count = binCount^3)
    std::vector<float> transformationData(count * 4);
    getDataFromBufferWithStagingBuffer(
        context, 
        transformation, 
        transformationData.data(), 
        sizeof(float) * count * 4
    );

    std::vector<float> baseTransformationData(count * 4);
    getDataFromBufferWithStagingBuffer(
        context, 
        baseTransformation, 
        baseTransformationData.data(), 
        sizeof(float) * count * 4
    );

    // parameters
    int tilesX = (int)std::ceil(std::sqrt((float)binCount)); // e.g. 16 if binCount=256
    int tilesY = (binCount + tilesX - 1) / tilesX;

    int sliceSize = binCount * binCount;    // one z-slice
    int outWidth  = binCount * tilesX;
    int outHeight = binCount * tilesY;

    // buffer for packed image
    std::vector<float> packed(outWidth * outHeight * 4, 0.0f);

    // pack the 3D cube into 2D tiled slices
    for (int z = 0; z < binCount; ++z) {
        int tileX = z % tilesX;
        int tileY = z / tilesX;

        for (int y = 0; y < binCount; ++y) {
            for (int x = 0; x < binCount; ++x) {
                int inIndex  = (x + y * binCount + z * binCount * binCount) * 4;
                int outX     = x + tileX * binCount;
                int outY     = y + tileY * binCount;
                int outIndex = (outX + outY * outWidth) * 4;

                packed[outIndex + 0] = transformationData[inIndex + 0] - baseTransformationData[inIndex + 0];
                packed[outIndex + 1] = transformationData[inIndex + 1] - baseTransformationData[inIndex + 1];
                packed[outIndex + 2] = transformationData[inIndex + 2] - baseTransformationData[inIndex + 2];
                packed[outIndex + 3] = 1;

                if(x == binCount-1 && y == binCount-1 && z == binCount-1) {
                    std::cout << packed[outIndex + 0] << ", "  << packed[outIndex + 1] << ", "  << packed[outIndex + 2] << std::endl;
                }
            }
        }
    }

    auto clampf = [](float v, float lo, float hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
    };

    std::vector<uint8_t> packed8(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        packed8[i] = (uint8_t)clampf(packed[i] * 255.0f, 0.0f, 255.0f);
    }

    if (!stbi_write_png("transformation.png", outWidth, outHeight, 4, packed8.data(), outWidth * 4)) {
        LOG_ERROR("Failed saving transformation");
    }
}