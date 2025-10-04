#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include "vulkan/vulkan_core.h"
#include "vulkan_base/vulkan_base.h"
#include "debug.h"
#include "setup.h"
#include "input.h"
#include "metrics.h"
#include <yaml-cpp/yaml.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define AUTO_STOP false
#define JITTER true
#define USE_ROTATED_INTEGRALS false
#define MULTISPECTRAL_IMAGES true
#define DATA_ANALYSIS false

std::vector<int> keyIterations = {1,2,4,8,16,32,64,128};

// Constants for debugging purposes
#define RANDIMG true
#define UNIFORMIMG false
#define LINEIMG false
#define INTEGRALS false
#define TRANSFORMATION false
#define VIDEO false
#define SAVECSV false
#define INVERT false


// Program constants
int ITERATIONS = 0;
UniformData constants;
float sigma = 1.0;
float finalMse = 0;
float finalGradient = 0;
float jitterFactor = 0;
float backgroundDensityFactor = 1;

VulkanContext* context;

VulkanDescriptorSet* baseDescriptorSet;
VulkanPipeline basePipeline;
StageBuffers baseBuffers;

VulkanDescriptorSet* mainDescriptorSet;
VulkanPipeline mainPipeline;
StageBuffers mainBuffers;

Result result;

void loadConstantsFromYaml(UniformData* constants) {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    constants->binCount = config["histogramBinCount"].as<int>();
    constants->kernelRadius = config["kernelRadius"].as<int>();
    constants->baseDensity = 0; // will be automatically set by the program
    ITERATIONS = config["iterations"].as<int>();
    sigma = config["gaussSigma"].as<float>();
    jitterFactor = config["jitterFactor"].as<float>();
    backgroundDensityFactor = config["backgroundFactor"].as<float>();
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

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    #if RANDIMG
    const int size = 500;
    const int randchannels = 4;
    const int stride_in_bytes = size * randchannels;
    int minVal = 100;
    int maxVal = 155;

    std::vector<unsigned char> randImg(size * size * randchannels);

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

    #if UNIFORMIMG

    const int uChannels = 64;
    const int uw = 512;
    std::vector<unsigned char> uniformImg(uw * uw * 4);

    int index = 0;
    for(int z = 0; z < uChannels; z++) {
        for(int y = 0; y < uChannels; y++) {
            for(int x = 0; x < uChannels; x++) {
                uniformImg[index] = x*4;
                uniformImg[index+1] = y*4;
                uniformImg[index+2] = z*4;
                uniformImg[index+3] = 255;

                index+=4;
            }
        }
    }

    if (!stbi_write_png("../images/uniform.png", uw, uw, 4, uniformImg.data(), uw*4)) {
        std::cerr << "Failed to write image!" << std::endl;
    }

    #endif

    #if LINEIMG

    const int lw = 4048;
    std::vector<unsigned char> lineImg(lw * lw * 4);

    int index = 0;
    for(int i = 0; i < lineImg.size(); i+=4) {
        float d = ((float)i/lineImg.size()*255);
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        lineImg[i] = (int)fmax(0.0, fmin(255.0,(float)d + 20*(2*r-1)));
        lineImg[i+1] = (int)fmax(0.0, fmin(255.0,(float)d + 20*(2*r-1)));
        lineImg[i+2] = (int)fmax(0.0, fmin(255.0,(float)d + 20*(2*r-1)));
        lineImg[i+3] = 255;
    }

    if (!stbi_write_png("../images/lineTest.png", lw, lw, 4, lineImg.data(), lw*4)) {
        std::cerr << "Failed to write image!" << std::endl;
    }

    #endif

    int w,h,channels;
    #if MULTISPECTRAL_IMAGES
    std::vector<float> pixels = loadFits(
        "../images/multispectral/lagoon/hubble_lagoon_f673n.fits",
        "../images/multispectral/lagoon/hubble_lagoon_f656n.fits",
        "../images/multispectral/lagoon/hubble_lagoon_f502n.fits",
        &w, &h
    );

    /*std::vector<float> pixels = loadFits(
        "../images/multispectral/eagle/m16_f673n.fits",
        "../images/multispectral/eagle/m16_f657n.fits",
        "../images/multispectral/eagle/m16_f502n.fits",
        &w, &h
    );*/
    #else
    std::vector<float> pixels = loadImage(imageFile, &w, &h, &channels);
    #endif

    // add jitter to image
    #if JITTER
    for(int i = 0; i < w*h*4; i++) {
        if(i%4 == 3) continue; // skip alpha channel
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * jitterFactor;
        pixels[i] = fmax(0.0f, fmin(1.0f, pixels[i]+r));

        float scaleMin = 0;
        float scaleMax = 1;
        pixels[i] = pixels[i]*(scaleMax-scaleMin)+scaleMin;
    }
    #endif

    size_t numPixels = size_t(w) * h;
    mainBuffers.imageSize = numPixels * 4 * sizeof(float); // forcing 4 channels
    baseBuffers.imageSize = mainBuffers.imageSize;

    float baseDensity = (float)(w*h) / pow(constants.binCount, 3) * backgroundDensityFactor;
    constants.baseDensity = baseDensity;
    std::cout << "Basedensity: " << baseDensity << std::endl;

    LOG("Creating descriptor set for base density");

    baseDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, baseDescriptorSet, &baseBuffers, pixels.data(), w, h, sigma, &constants, NULL, true);

    LOG("Creating descriptor for main loop");
    mainDescriptorSet = initDescriptorSet();
    setupDescriptorSet(context, mainDescriptorSet, &mainBuffers, pixels.data(), w, h, sigma, &constants, &baseBuffers.transformationBuffer, false);

    LOG("Creating pipelines");
    int groupsKernel = constants.binCount / 8+1;
    int groupsInt = constants.binCount / 32 + 1;
    int transformationKernel = (constants.binCount+1) / 8 + 1;


    std::vector<const char*> baseComputeShaders;
    baseComputeShaders.push_back("../shaders/kernelX.spv");
    baseComputeShaders.push_back("../shaders/kernelY.spv");
    baseComputeShaders.push_back("../shaders/kernelZ.spv");
    baseComputeShaders.push_back("../shaders/integralX.spv");
    baseComputeShaders.push_back("../shaders/integralY.spv");
    baseComputeShaders.push_back("../shaders/integralZ.spv");
#if USE_ROTATED_INTEGRALS
    baseComputeShaders.push_back("../shaders/integralD.spv");
#endif
    baseComputeShaders.push_back("../shaders/transformation.spv");

    std::vector<ivec3> baseDispatches = {
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
#if USE_ROTATED_INTEGRALS
        ivec3{groupsInt, groupsInt, 1},
#endif
        ivec3{transformationKernel, transformationKernel, transformationKernel},
    };
    basePipeline = createPipeline(context, baseComputeShaders, baseDispatches, baseDescriptorSet);


    std::vector<const char*> computeShaders;
    computeShaders.push_back("../shaders/histogram.spv");
    computeShaders.push_back("../shaders/kernelX.spv");
    computeShaders.push_back("../shaders/kernelY.spv");
    computeShaders.push_back("../shaders/kernelZ.spv");
#if AUTO_STOP
    computeShaders.push_back("../shaders/metric.spv");
#endif
    computeShaders.push_back("../shaders/integralX.spv");
    computeShaders.push_back("../shaders/integralY.spv");
    computeShaders.push_back("../shaders/integralZ.spv");
#if USE_ROTATED_INTEGRALS
    computeShaders.push_back("../shaders/integralD.spv");
#endif
    computeShaders.push_back("../shaders/transformation.spv");
#if DATA_ANALYSIS
    computeShaders.push_back("../shaders/gradient.spv");
#endif
    computeShaders.push_back("../shaders/enhanceContrast.spv");

    std::vector<ivec3> dispatches = {
        ivec3{(int)w/16+1, (int)h/16+1, 1},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
        ivec3{groupsKernel, groupsKernel, groupsKernel},
#if AUTO_STOP
        ivec3{groupsInt, groupsInt, 1},
#endif
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
        ivec3{groupsInt, groupsInt, 1},
#if USE_ROTATED_INTEGRALS
        ivec3{groupsInt, groupsInt, 1},
#endif
        ivec3{transformationKernel, transformationKernel, transformationKernel},
        ivec3{(int)w/16+1, (int)h/16+1, 1},
#if DATA_ANALYSIS
        ivec3{(int)w/16+1, (int)h/16+1, 1},
#endif
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

// Runs the shaders for the given commandBuffer
void runApplication(VkCommandBuffer* commandBuffer, int iterations, bool isBaseRun) {
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffer;

    const int barWidth = 50;
    float averageMillis = 0;
    std::string loadingBar = std::string(barWidth, '|');

    std::ofstream file("../plots/standardDeviation_10.csv");
    if (!file.is_open()) {
        std::cerr << "Error opening file: standardDeviation.csv" << std::endl;
        return;
    }
    file << "iteration,x\n";

    float variance = 1e30;
    int i;
    for(i = 0; i < iterations; ++i) {
        // Run Shader iteration
        auto start = std::chrono::high_resolution_clock::now();
        if (vkQueueSubmit(context->computeQueue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        }
        vkQueueWaitIdle(context->computeQueue.queue);

        // If enabled, check for increasing variance of histogram data
        uint32_t count = constants.binCount * constants.binCount;
        std::vector<float> metricData(count);
        getDataFromBufferWithStagingBuffer(context, &mainBuffers.metric, metricData.data(), sizeof(float) * count);

        float newVariance = std::accumulate(metricData.begin(), metricData.end(), 0.0f);
        newVariance /= (float)pow(constants.binCount, 2);
        
        // time measurement and progress bar
        averageMillis += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        float progress = static_cast<float>(i + 1) / iterations;
        int pos = static_cast<int>(barWidth * progress);
        printProgress(progress, barWidth, loadingBar, averageMillis/(i+1));
        if(isBaseRun) {
            std::cout << std::endl;
            return;
        }

        #if VIDEO
        std::ostringstream oss;
        oss << "histograms/histogram" << std::setw(5) << std::setfill('0') << i+1 << ".png";
        saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, oss.str().c_str(), 'x');
        #endif

        file << i << "," << newVariance << "\n";

        #if DATA_ANALYSIS
        for(int j = 0; j < keyIterations.size(); ++j) {
            if(i == keyIterations[j]) { // histogram and gradient is the one from the previous iteration
                result.gradient.push_back(calculateGradient(context, &mainBuffers.gradient, mainBuffers.imageSize));
                result.mse.push_back(newVariance);
                break;
            }
        }
        #endif


        if(newVariance > variance) {
            variance = newVariance;
        #if false
            std::cout << std::endl << "Only did " << i+1 << " iterations because of increasing variance" << std::endl;
            break;
        #endif
        }
        variance = newVariance;
    }

    #if DATA_ANALYSIS
    result.lastIteration = i;
    finalGradient = calculateGradient(context, &mainBuffers.gradient, mainBuffers.imageSize);
    finalMse = variance;
    #endif

    #if AUTO_STOP
    file.close();
    #endif
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

    for(int i = 0; i < 64; ++i){
        keyIterations.push_back(i);
    }

    result = Result();
    result.fileName = fileName;
    result.iterations = keyIterations;

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
    
    runApplication(&commandBuffer, 1,true);
    
    vkResetCommandBuffer(commandBuffer, 0);

    // MAIN LOOP
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    setupCommandBuffer(&commandBuffer, mainDescriptorSet, &mainPipeline);
    vkEndCommandBuffer(commandBuffer);

    runApplication(&commandBuffer, ITERATIONS, false);
    
    vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);

    std::cout << std::endl;

    LOG("Computing finished");

    #if DATA_ANALYSIS
    LOG("Save analysis");
    // save analysis to files
    std::ofstream gradientFile("../plots/test_gradient.csv", std::ios::app);
    if (!gradientFile.is_open()) {
        std::cerr << "Error opening for analysis" << std::endl;
    }

    std::ofstream mseFile("../plots/test_mse.csv", std::ios::app);
    if (!mseFile.is_open()) {
        std::cerr << "Error opening for analysis" << std::endl;
    }

    gradientFile << fileName;
    mseFile << fileName;

    for(int i = 0; i < result.iterations.size(); ++i){
        if(i < result.gradient.size()) {
            gradientFile << "," << result.gradient[i];
            mseFile << "," << result.mse[i];
        } else {
            gradientFile << ",-1";
            mseFile << ",-1";
        }
    }

    gradientFile << "," << result.lastIteration;
    gradientFile << "," << finalGradient << "\n";
    mseFile << "," << result.lastIteration;
    mseFile << "," << finalMse << "\n";

    #else // Dont save the images during data analysis

    LOG("Save image as png");
    saveImageAsPng(context, &mainBuffers.imageBuffer, mainBuffers.imageSize);

    LOG("Save histogram as png");
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramx.png").c_str(), 'x');
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramy.png").c_str(), 'y');
    saveHistogramAsPng(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("histogramz.png").c_str(), 'z');
    #endif

    #if INTEGRALS
    LOG("Save integrals as pngs");
    saveIntegralsAsPngs(context, mainBuffers.integralImages, constants.binCount);
    #endif
    
    #if TRANSFORMATION
    LOG("Save transformation as png");
    saveTransformationAsPng(context, &mainBuffers.transformationBuffer, &baseBuffers.transformationBuffer, constants.binCount+1);
    #endif

    #if SAVECSV
    LOG("Save data to csv files");
    saveHistogramAsCsv(context, &mainBuffers.kernelBuffer, constants.binCount, std::string("../plots/histogram.csv").c_str());
    for (int i = 0; i < mainBuffers.integralImages.size(); ++i) {
        std::string filename = "../plots/integral" + std::to_string(i) + ".csv";
        saveHistogramAsCsv(context, &mainBuffers.integralImages[i], constants.binCount, filename.c_str());
    }
    saveRotatedIntegralAsCsv(context, &mainBuffers.tempIntegrals[0], constants.binCount, std::string("../plots/integral.csv").c_str(), 0);

    saveTransformationAsCsv(context, &mainBuffers.transformationBuffer, &baseBuffers.transformationBuffer, constants.binCount, "../plots/transformation.csv", constants.binCount);
    saveDataAsCsv(context, &mainBuffers.imageBuffer, mainBuffers.imageSize, "../plots/image.csv", 1);
    #endif
   
    shutdownApplication();
    return 1;
}
