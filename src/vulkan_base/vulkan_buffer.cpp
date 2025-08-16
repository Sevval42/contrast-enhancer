#include "vulkan/vulkan_core.h"
#include "vulkan_base.h"
#include <cstring>

void createBuffer(VulkanContext* context, VulkanBuffer* buffer, uint32_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size = size;
    createInfo.usage = usage;
    vkCreateBuffer(context->device, &createInfo, 0, &buffer->buffer);

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(context->device, buffer->buffer, &memoryRequirements);
    uint32_t memoryIndex = findMemoryType(context, memoryRequirements.memoryTypeBits, memoryProperties);

    VkMemoryAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryIndex;

    vkAllocateMemory(context->device, &allocateInfo, 0, &buffer->memory);
    vkBindBufferMemory(context->device, buffer->buffer, buffer->memory, 0);
}

void copyBuffer(VulkanContext* context, VulkanBuffer* srcBuffer, VulkanBuffer* dstBuffer, size_t size) {

    VkCommandBuffer commandBuffer = beginSingleTimeCommands(context);

    VkBufferCopy copyRegion = {0, 0, size};
    vkCmdCopyBuffer(commandBuffer, srcBuffer->buffer, dstBuffer->buffer, 1, &copyRegion);

    endSingleTimeCommands(context, commandBuffer);
}

void uploadDataToBufferWithStagingBuffer(VulkanContext* context, VulkanBuffer* buffer, void* data, size_t size) {
    VulkanBuffer stagingBuffer;
    createBuffer(
        context, 
        &stagingBuffer, size, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* mapped;
    vkMapMemory(context->device, stagingBuffer.memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(context->device, stagingBuffer.memory);

    copyBuffer(context, &stagingBuffer, buffer, size);

    vkDestroyBuffer(context->device, stagingBuffer.buffer, 0);
    vkFreeMemory(context->device, stagingBuffer.memory, 0);
}

void getDataFromBufferWithStagingBuffer(VulkanContext* context, VulkanBuffer* buffer, void* data, size_t size) {
    VulkanBuffer stagingBuffer;
    createBuffer(
        context, 
        &stagingBuffer, 
        size, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* mapped;
    

    copyBuffer(context, buffer, &stagingBuffer, size);

    vkMapMemory(context->device, stagingBuffer.memory, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(context->device, stagingBuffer.memory);

    vkDestroyBuffer(context->device, stagingBuffer.buffer, 0);
    vkFreeMemory(context->device, stagingBuffer.memory, 0);
}

void destroyBuffer(VulkanContext* context, VulkanBuffer* buffer) {
    vkDestroyBuffer(context->device, buffer->buffer, 0);
    vkFreeMemory(context->device, buffer->memory, 0);
}
