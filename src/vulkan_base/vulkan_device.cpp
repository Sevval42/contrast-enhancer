#include "vulkan/vulkan_core.h"
#include "vulkan_base.h"
#include <iostream>

#define DEBUGGING true

bool initVulkanInstance(VulkanContext* context, uint32_t extensionCount, const char**  extensions) {
    uint32_t layerPropertyCount;
    vkEnumerateInstanceLayerProperties(&layerPropertyCount, 0);
    VkLayerProperties* layerProperties = new VkLayerProperties[layerPropertyCount];
    vkEnumerateInstanceLayerProperties(&layerPropertyCount, layerProperties);

    /*std::cout << "Available VK_LayerProperties: " << std::endl;
    for(uint32_t i = 0; i < layerPropertyCount; ++i) {
        std::cout << layerProperties[i].layerName << std::endl;
        std::cout << layerProperties[i].description << std::endl;
    }*/
    delete[] layerProperties;

    const char* enabledLayers[] = {
    #if DEBUGGING
        "VK_LAYER_KHRONOS_validation",
    #endif
    };

    uint32_t extensionPropertyCount;
    vkEnumerateInstanceExtensionProperties(0, &extensionPropertyCount, 0);
    VkExtensionProperties* extensionProperties = new VkExtensionProperties[extensionPropertyCount];
    vkEnumerateInstanceExtensionProperties(0, &extensionPropertyCount, extensionProperties);

    /*std::cout << "Available VK_ExtensionProperties: " << std::endl;
    for(uint32_t i = 0; i < extensionPropertyCount; ++i) {
        std::cout << extensionProperties[i].extensionName << std::endl;
    }*/

    delete[] extensionProperties;

    VkApplicationInfo applicationInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};

    applicationInfo.pEngineName = "Vulkan Tutorial";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledLayerCount = ARRAY_COUNT(enabledLayers);
    createInfo.ppEnabledLayerNames = enabledLayers;
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = extensions;

    VkResult result = vkCreateInstance(&createInfo, 0, &context->instance);
    if(result != VK_SUCCESS) {
        std::cout << "Error creating Vulkan instance" << std::endl;
        return false;
    }
    return true;
}

bool selectPhysicalDevice(VulkanContext* context) {
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(context->instance, &numDevices, 0);
    if(numDevices == 0){
        std::cerr << "Could not find GPU with vulkan support" << std::endl;
        context->physicalDevice = 0;
        return false;
    }
    VkPhysicalDevice* physicalDevices = new VkPhysicalDevice[numDevices];
    vkEnumeratePhysicalDevices(context->instance, &numDevices, physicalDevices);
    std::cout << "Found " << numDevices << " GPU(s):" << std::endl;
    for(uint32_t i = 0; i < numDevices; ++i) {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(physicalDevices[i], &properties);
        std::cout << "  (" << i << ") " << properties.deviceName << std::endl;
    }

    context->physicalDevice = physicalDevices[0];
    vkGetPhysicalDeviceProperties(context->physicalDevice, &context->physicalDeviceProperties);
    std::cout << "Selected GPU: " << context->physicalDeviceProperties.deviceName << std::endl;

    delete[] physicalDevices;
    return true;
}

bool createLogicalDevice(VulkanContext* context, uint32_t deviceExtensionCount, const char** deviceExtensions) {
    uint32_t numQueueFamilies = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice, &numQueueFamilies, 0);
    VkQueueFamilyProperties* queueFamilies = new VkQueueFamilyProperties[numQueueFamilies];
    vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice, &numQueueFamilies, queueFamilies);

    uint32_t computeQueueIndex = 0;
    for(uint32_t i = 0; i < numQueueFamilies; ++i) {
        VkQueueFamilyProperties queueFamily = queueFamilies[i];
        if(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if(queueFamily.queueCount > 0) {
                computeQueueIndex = i;
                break;
            }
        }
    }

    float priorities[] = {1.0f};
    VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = computeQueueIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = priorities;
    
    VkPhysicalDeviceFeatures enabledFeatures = {};

    VkDeviceCreateInfo createInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledExtensionCount = deviceExtensionCount;
    createInfo.ppEnabledExtensionNames = deviceExtensions;
    createInfo.pEnabledFeatures = &enabledFeatures;
    
    VkResult result = vkCreateDevice(context->physicalDevice, &createInfo, 0, &context->device);
    if(result != VK_SUCCESS){
        std::cout << "Failed to create vulkan logical device" << std::endl;
        return false;
    }

    context->computeQueue.familyIndex = computeQueueIndex;
    vkGetDeviceQueue(context->device, computeQueueIndex, 0, &context->computeQueue.queue);

    return true;
}

VulkanContext* initVulkan(uint32_t extensionCount, const char** extensions, uint32_t deviceExtensionCount, const char** deviceExtensions) {
    VulkanContext* context = new VulkanContext;

    if(!initVulkanInstance(context, extensionCount, extensions)){
        return 0;
    }

    if(!selectPhysicalDevice(context)) {
        return 0;
    }

    if(!createLogicalDevice(context, deviceExtensionCount, deviceExtensions)) {
        return 0;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = context->computeQueue.familyIndex;
    if (vkCreateCommandPool(context->device, &poolInfo, nullptr, &context->commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    return context;
}

void exitVulkan(VulkanContext* context) {
    vkDestroyCommandPool(context->device, context->commandPool, 0);
    vkDeviceWaitIdle(context->device);
    vkDestroyDevice(context->device, 0);
    vkDestroyInstance(context->instance, 0);
}