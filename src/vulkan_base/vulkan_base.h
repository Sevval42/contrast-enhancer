#include "vulkan/vulkan_core.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>

#define ENABLE_LOGGING 1

#if ENABLE_LOGGING
    #define LOG(msg) std::cout << "[LOG] " << msg << std::endl
    #define LOG_WARN(msg) std::cout << "[WARN] " << msg << std::endl
    #define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#else
    #define LOG(msg)
    #define LOG_WARN(msg)
    #define LOG_ERROR(msg)
#endif

#define ARRAY_COUNT(array) (sizeof(array) / sizeof((array)[0]));

struct ivec3 {
    int x;
    int y;
    int z;
};

struct VulkanQueue {
    VkQueue queue;
    uint32_t familyIndex;
};

struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    VkDevice device;
    VulkanQueue computeQueue;
    VkCommandPool commandPool;
};

struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct VulkanImage {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkImageLayout currentLayout;
    VkExtent3D extent;
    size_t size;
};

struct VulkanDescriptorBufferInfo {
    VkDescriptorBufferInfo bufferInfo;
    VkDescriptorImageInfo imageInfo;
    enum class Type {
        BUFFER,
        IMAGE,
    } type;
};

struct VulkanDescriptorSet {
    uint32_t layoutCount;
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    VkDescriptorSetLayout descriptorSetLayout;
    std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCount;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    std::vector<VulkanDescriptorBufferInfo> buffers;
    void addBufferAndData(
        VulkanContext* context, 
        VulkanBuffer* buffer, 
        void* data, uint32_t size, 
        VkBufferUsageFlags usage, 
        VkMemoryPropertyFlags memoryProperties
    );

    void addImageAndData(
        VulkanContext* context, 
        VulkanImage* image, void* data, size_t size,
        uint32_t width, uint32_t height, uint32_t depth, 
        VkFormat format, 
        VkImageUsageFlags usage, 
        VkMemoryPropertyFlags memoryProperties
    );
};

struct VulkanPipeline {
    std::vector<VkPipeline> pipelines;
    std::vector<ivec3> dispatchSizes;
    VkPipelineLayout pipelineLayout;
};

// vulkan_device.cpp
VulkanContext* initVulkan(uint32_t extensionCount, const char** extensions, uint32_t deviceExtensionCount, const char** deviceExtensions);
void exitVulkan(VulkanContext* context);

// vulkan_helper.cpp
uint32_t findMemoryType(VulkanContext* context, uint32_t typeFilter, VkMemoryPropertyFlags memoryProperties);
VkCommandBuffer beginSingleTimeCommands(VulkanContext* context);
void endSingleTimeCommands(VulkanContext* context, VkCommandBuffer commandBuffer);

// vulkan_buffer.cpp
void createBuffer(VulkanContext* context, VulkanBuffer* buffer, uint32_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
void uploadDataToBufferWithStagingBuffer(VulkanContext* context, VulkanBuffer* buffer, void* data, size_t size);
void getDataFromBufferWithStagingBuffer(VulkanContext* context, VulkanBuffer* buffer, void* data, size_t size);
void destroyBuffer(VulkanContext* context, VulkanBuffer* buffer);

// vulkan_image.cpp
void createImage(VulkanContext* context, VulkanImage* image, size_t size, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties);
void uploadDataToImageWithStagingBuffer(VulkanContext* context, VulkanImage* image, void* data);
void transitionLayout(VulkanContext* context, VulkanImage* image, VkImageLayout newLayout, VkCommandBuffer commandBuffer);
void getDataFromImageWithStagingBuffer(VulkanContext* context, VulkanImage* image, void* data);
void destroyImage(VulkanContext* context, VulkanImage* image);

// vulkan_descriptor_set.cpp
VulkanDescriptorSet* initDescriptorSet();
void addDescriptorSetLayout(VulkanDescriptorSet* descriptorSet, VkDescriptorType descriptorType);
void createDescriptorSet(VulkanContext* context, VulkanDescriptorSet* descriptorSet);
void fillDescriptorSet(VulkanContext* context, VulkanDescriptorSet* descriptorSet);
void destroyDescriptorSet(VulkanContext* context, VulkanDescriptorSet* descriptorSet);

// vulkan_pipeline.cpp
VulkanPipeline createPipeline(VulkanContext* context, std::vector<const char*> computeShaderFilenames, std::vector<ivec3> dispatches, VulkanDescriptorSet* descriptorSet);
void destroyPipeline(VulkanContext* context, VulkanPipeline* pipeline);
