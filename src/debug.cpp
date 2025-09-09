#include <iostream>
#include "debug.h"
#include "vulkan_base/vulkan_base.h"
#include "setup.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <sstream>
#include <fstream>

void saveHistogramAsPng(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName, const char dimension) {
    uint32_t count = pow(binCount, 3);
    std::vector<float> data(count);
    getDataFromBufferWithStagingBuffer(context, histogram, data.data(), sizeof(float) * count);

    int histogram2DCount = pow(binCount, 2);
    std::vector<float> histogram2D(histogram2DCount);
    for(uint32_t i = 0; i < histogram2DCount; ++i) {
        int x = i % binCount;
        int y = floor(i/binCount);
        for(int z = 0; z < binCount; ++z) {
            int getIndex = 0;
            switch(dimension) {
                case 'x': getIndex = z + x * binCount + y * binCount * binCount; break;
                case 'y': getIndex = x + z * binCount + y * binCount * binCount; break;
                case 'z': getIndex = x + y * binCount + z * binCount * binCount; break;
            }
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

inline std::string intToString(int value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

void saveIntegralsAsPngs(VulkanContext* context, std::vector<VulkanBuffer>& integrals, uint32_t binCount) {
    const size_t count3D = static_cast<size_t>(binCount) * binCount * binCount;
    const size_t count2D = static_cast<size_t>(binCount) * binCount;

    for (size_t bufIdx = 0; bufIdx < integrals.size(); ++bufIdx) {
        std::vector<float> integral(count3D);
        getDataFromBufferWithStagingBuffer(context, &integrals[bufIdx], integral.data(), sizeof(float) * count3D);

        // Accumulate 3D integral into 2D
        std::vector<float> integral2D(count2D, 0.0f);
        for (uint32_t y = 0; y < binCount; ++y) {
            for (uint32_t x = 0; x < binCount; ++x) {
                for (uint32_t z = 0; z < binCount; ++z) {
                    size_t index3D = z + x * binCount + y * binCount * binCount;
                    size_t index2D = x + y * binCount;
                    integral2D[index2D] += integral[index3D];
                }
            }
        }

        // Find max value for normalization
        float maxVal = 0.0f;
        for (float val : integral2D) {
            if (val > maxVal) maxVal = val;
        }
        if (maxVal == 0.0f) maxVal = 1.0f; // Avoid division by zero

        // Convert to 8-bit grayscale
        std::vector<uint8_t> integral2DBytes(count2D);
        for (size_t i = 0; i < count2D; ++i) {
            integral2DBytes[i] = static_cast<uint8_t>((integral2D[i] / maxVal) * 255.0f);
        }

        // Save PNG
        std::string filename = "integral" + intToString(static_cast<int>(bufIdx)) + ".png";
        if (!stbi_write_png(filename.c_str(), binCount, binCount, 1, integral2DBytes.data(), binCount)) {
            LOG_ERROR("Failed saving histogram: " + filename);
        }
    }
}

void saveImageAsPng(VulkanContext* context, VulkanImage* image, uint32_t imageSize) {
    std::vector<float> outputPixels(imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, image, outputPixels.data());

    std::vector<uint8_t> outputBytes(image->extent.width * image->extent.height * 4);

    float max = 0;
    for (size_t i = 0; i < outputPixels.size(); ++i) {
        float v = std::min(std::max(outputPixels[i], 0.0f), 1.0f);
        outputBytes[i] = static_cast<uint8_t>(v * 255.0f);
        if(outputPixels[i] > max && i % 4 == 0) max = outputPixels[i];
    }

    std::cout << "max red value: " << max << std::endl;

    if(
        !stbi_write_png(
        "imageOutput.png", 
        static_cast<int>(image->extent.width), static_cast<int>(image->extent.height), 4, 
        outputBytes.data(), image->extent.width*4)
    ) {
        LOG_ERROR("Failed saving output image");
    }

}

// Method written by chat-gpt
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

                if(x == binCount-2 && y == binCount-2 && z == binCount-2) {
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

void printProgress(double percentage, int width, std::string bar, float avgTime) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * width);
    int rpad = width - lpad;
    printf("\r [%.*s%*s] %3d%% (%.1f ms)", lpad, bar.c_str(), rpad, "", val, avgTime);
    fflush(stdout);
}



void saveHistogramAsCsv(VulkanContext* context, VulkanBuffer* histogram, uint32_t binCount, const char* fileName){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }
    file << "R,G,B,Count\n";
    
    uint32_t count = pow(binCount, 3);
    std::vector<float> data(count);
    getDataFromBufferWithStagingBuffer(context, histogram, data.data(), sizeof(float) * count);

    for(int z = 0; z < binCount; ++z){
        for(int y = 0; y < binCount; ++y){
            for(int x = 0; x < binCount; ++x){
                int i = x + y * binCount + z * binCount *binCount;
                file << x << "," << y << "," << z << "," << data[i] << "\n";
            }
        }
    }

    file.close();
}

void saveRotatedIntegralAsCsv(VulkanContext* context, VulkanBuffer* integral, uint32_t binCount, const char* fileName, int direction) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }
    file << "R,G,B,Count\n";
    
    uint32_t count = pow(binCount, 3);
    std::vector<float> data(count * 4);
    getDataFromBufferWithStagingBuffer(context, integral, data.data(), sizeof(float) * count * 4);

    for(int z = 0; z < binCount; ++z){
        for(int y = 0; y < binCount; ++y){
            for(int x = 0; x < binCount; ++x){
                int i = x + y * binCount + z * binCount * binCount + direction * binCount * binCount * binCount;
                file << x << "," << y << "," << z << "," << data[i] << "\n";
            }
        }
    }

    file.close();
}

void saveTransformationAsCsv(VulkanContext* context, VulkanBuffer* transformation, VulkanBuffer* baseTransformation, uint32_t binCount, const char* fileName, int vectorCount) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }
    file << "x,y,z,u,v,w\n";

    uint32_t count = pow(binCount+1, 3);
    std::vector<float> transf(count * 4);
    getDataFromBufferWithStagingBuffer(context, transformation, transf.data(), sizeof(float) * count * 4);

    std::vector<float> base_transf(count * 4);
    getDataFromBufferWithStagingBuffer(context, baseTransformation, base_transf.data(), sizeof(float) * count * 4);

    int n = binCount + 1;
    int step = (int)(binCount / vectorCount);
    for(int z = 0; z <= binCount; z += step){
        for(int y = 0; y <= binCount; y += step){
            for(int x = 0; x <= binCount; x += step){
                int index = (x + y * n + z * n * n) * 4;
                float dx = transf[index] - base_transf[index];
                float dy = transf[index+1] - base_transf[index+1];
                float dz = transf[index+2] - base_transf[index+2];
                float factor = 1;
                dx *= factor;
                dy *= factor;
                dz *= factor;
                file << x << "," << y << "," << z << "," << dx << "," << dy << "," << dz << "\n";
            }
        }
    }
    file.close();
}