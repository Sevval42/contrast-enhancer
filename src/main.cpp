#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include "vulkan/vulkan_core.h"
#include "vulkan_base/vulkan_base.h"
#include "debug.h"
#include "setup.h"
#include <yaml-cpp/yaml.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Constants for debugging purposes
#define RANDIMG true
#define INTEGRALS false
#define TRANSFORMATION false
#define VIDEO false
#define SAVECSV false
#define INVERT false

// Program constants
int ITERATIONS = 0;
UniformData constants;
float sigma = 1.0;

VulkanContext* context;

VulkanDescriptorSet* baseDescriptorSet;
VulkanPipeline basePipeline;
StageBuffers baseBuffers;

VulkanDescriptorSet* mainDescriptorSet;
VulkanPipeline mainPipeline;
StageBuffers mainBuffers;

void loadConstantsFromYaml(UniformData* constants) {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    constants->binCount = config["histogramBinCount"].as<int>();
    constants->kernelRadius = config["kernelRadius"].as<int>();
    constants->baseDensity = 0; // will be automatically set by the program
    ITERATIONS = config["iterations"].as<int>();
    sigma = config["gaussSigma"].as<float>();
}

void initApplication(std::string imageFile) {

    const char* instanceExtensions[] = {
        #ifdef __APPLE__
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        #endif
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
    };
    uint32_t instanceExtensionsCount = ARRAY_COUNT(instanceExtensions);

    const char* deviceExtensions[] = {
        #ifdef __APPLE__
        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
        #endif
    };
    uint32_t deviceExtensionsCount = ARRAY_COUNT(deviceExtensions);

    LOG("Get device");
    context = initVulkan(
        instanceExtensionsCount,
        instanceExtensions, 
        deviceExtensionsCount, 
        deviceExtensions
    );

    #if RANDIMG
    const int size = 256;
    const int randchannels = 4;
    const int stride_in_bytes = size * randchannels;
    int minVal = 100;
    int maxVal = 155;

    // Seed RNG
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Allocate pixel buffer (unsigned char for 0–255 range)
    std::vector<unsigned char> randImg(size * size * randchannels);

    // Fill with random colors
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * randchannels;
            randImg[idx + 0] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 1] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 2] = minVal + (std::rand() % (maxVal - minVal + 1));
            randImg[idx + 3] = 255;
        }
    }
    if (!stbi_write_png("../images/randomImage.png", size, size, randchannels, randImg.data(), stride_in_bytes)) {
        std::cerr << "Failed to write image!" << std::endl;
    }
    #endif


    int w,h,channels;
    stbi_ldr_to_hdr_gamma(1.0f);
    float* pixels = stbi_loadf(imageFile.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }

    size_t numPixels = size_t(w) * h;
    mainBuffers.imageSize = numPixels * 4 * sizeof(float); // forcing 4 channels
    baseBuffers.imageSize = mainBuffers.imageSize;

    float maxC = -1.0f, minC = 1e6f;

    float baseDensity = (float)(w*h) / pow(constants.binCount, 3);
    constants.baseDensity = baseDensity;

    LOG("Creating descriptor set for base density");

    baseDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, baseDescriptorSet, &baseBuffers, pixels, w, h, sigma, &constants, NULL, true);

    LOG("Creating descriptor for main loop");
    mainDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, mainDescriptorSet, &mainBuffers, pixels, w, h, sigma, &constants, &baseBuffers.transformationBuffer, false);
    
    stbi_image_free(pixels);


    LOG("Creating pipelines");
    int groupsKernel = constants.binCount / 8+1;
    int groupsInt = constants.binCount / 32 + 1;
    int transformationKernel = (constants.binCount+1) / 8 + 1;


    std::vector<const char*> baseComputeShaders;
    baseComputeShaders.push_back("../shaders/kernel.spv");
    baseComputeShaders.push_back("../shaders/integralX.spv");
    baseComputeShaders.push_back("../shaders/integralY.spv");
    baseComputeShaders.push_back("../shaders/integralZ.spv");
    baseComputeShaders.push_back("../shaders/integralD.spv");
    baseComputeShaders.push_back("../shaders/transformation.spv");

    std::vector<ivec3> baseDispatches = {
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{transformationKernel, transformationKernel, transformationKernel},
    };
    basePipeline = createPipeline(context, baseComputeShaders, baseDispatches, baseDescriptorSet);


    std::vector<const char*> computeShaders;
    computeShaders.push_back("../shaders/histogram.spv");
    computeShaders.push_back("../shaders/kernel.spv");
    computeShaders.push_back("../shaders/integralX.spv");
    computeShaders.push_back("../shaders/integralY.spv");
    computeShaders.push_back("../shaders/integralZ.spv");
    computeShaders.push_back("../shaders/integralD.spv");
    computeShaders.push_back("../shaders/transformation.spv");
    computeShaders.push_back("../shaders/enhanceContrast.spv");

    std::vector<ivec3> dispatches = {
        ivec3{(int)w/16+1, (int)h/16+1, 1},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{transformationKernel, transformationKernel, transformationKernel},
        ivec3{(int)w/16+1, (int)h/16+1, 1},
    };
    mainPipeline = createPipeline(context, computeShaders, dispatches, mainDescriptorSet);
}

void shutdownApplication() {
    vkDeviceWaitIdle(context->device);

    destroyStageBuffer(context, &baseBuffers);
    destroyPipeline(context, &basePipeline);
    destroyDescriptorSet(context, baseDescriptorSet);

    destroyStageBuffer(context, &mainBuffers);
    destroyPipeline(context, &mainPipeline);
    destroyDescriptorSet(context, mainDescriptorSet);
    
    exitVulkan(context);
}

void runApplication(VkCommandBuffer* commandBuffer, int iterations) {
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffer;

    const int barWidth = 50;
    float averageMillis = 0;
    std::string loadingBar = std::string(barWidth, '|');

    for(int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (vkQueueSubmit(context->computeQueue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        }
        vkQueueWaitIdle(context->computeQueue.queue);

        
        averageMillis += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        // Progress fraction
        float progress = static_cast<float>(i + 1) / iterations;
        int pos = static_cast<int>(barWidth * progress);

        // Draw progress bar
        printProgress(progress, barWidth, loadingBar, averageMillis/(i+1));

        #if VIDEO
        std::ostringstream oss;
        oss << "histograms/histogram" << std::setw(5) << std::setfill('0') << i+1 << ".png";
        saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, oss.str().c_str(), 'x');
        #endif
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    loadConstantsFromYaml(&constants);
    std::string fileName = "../images/input.png";
    if(argc >= 3) {
        ITERATIONS = atoi(argv[2]);
    }
    if(argc >= 2) {
        fileName = std::string("../images/") + argv[1];
    }
    initApplication(fileName);

    // Start running the shaders

    VkCommandBuffer commandBuffer;
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = context->commandPool;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(context->device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffer");
        }
    }

    
    std::cout << std::endl <<  "Running " << ITERATIONS << " iterations" << std::endl;

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    // one iteration for base density transformation
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    setupCommandBuffer(&commandBuffer, baseDescriptorSet, &basePipeline);
    vkEndCommandBuffer(commandBuffer);
    
    runApplication(&commandBuffer, 1);
    
    vkResetCommandBuffer(commandBuffer, 0);

    // MAIN LOOP
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    setupCommandBuffer(&commandBuffer, mainDescriptorSet, &mainPipeline);
    vkEndCommandBuffer(commandBuffer);
    
    runApplication(&commandBuffer, ITERATIONS);
    
    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);

    std::cout << std::endl;

    LOG("Computing finished");

    LOG("Save image as png");
    saveImageAsPng(context, &mainBuffers.imageBuffer, mainBuffers.imageSize);

    LOG("Save histogram as png");
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramx.png").c_str(), 'x');
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramy.png").c_str(), 'y');
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramz.png").c_str(), 'z');

    #if INTEGRALS
    LOG("Save integrals as pngs");
    saveIntegralsAsPngs(context, baseBuffers.integralImages, constants.binCount);
    #endif
    
    #if TRANSFORMATION
    LOG("Save transformation");
    saveTransformationAsPng(context, &mainBuffers.transformationBuffer, &baseBuffers.transformationBuffer, constants.binCount+1);
    #endif

    #if SAVECSV
    saveHistogramAsCsv(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("../plots/histogram.csv").c_str());
    for (int i = 0; i < mainBuffers.integralImages.size(); ++i) {
        std::string filename = "../plots/integral" + std::to_string(i) + ".csv";
        saveHistogramAsCsv(context, &mainBuffers.integralImages[i], constants.binCount, filename.c_str());
    }
    //saveRotatedIntegralAsCsv(context, &mainBuffers.tempIntegrals[0], constants.binCount, std::string("../plots/integral.csv").c_str(), 0);
    #endif
   
    shutdownApplication();
    return 1;
}
