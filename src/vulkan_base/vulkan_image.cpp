#include "vulkan/vulkan_core.h"
#include "vulkan_base.h"
#include <stdexcept>

void copyBufferToImage(VulkanContext* context, VulkanBuffer* buffer, VulkanImage* image) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(context);
    transitionLayout(context, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, commandBuffer);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = image->extent;

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer->buffer,
        image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );
    
    transitionLayout(context, image, VK_IMAGE_LAYOUT_GENERAL, commandBuffer);
    endSingleTimeCommands(context, commandBuffer);
}

void copyImageToBuffer(VulkanContext* context, VulkanImage* image, VulkanBuffer* buffer) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(context);
    transitionLayout(context, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, commandBuffer);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = image->extent;

    vkCmdCopyImageToBuffer(
        commandBuffer,
        image->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        buffer->buffer,
        1,
        &region
    );

    transitionLayout(context, image, VK_IMAGE_LAYOUT_GENERAL, commandBuffer);
    endSingleTimeCommands(context, commandBuffer);
}


void createImage(VulkanContext* context, VulkanImage* image, size_t size, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image->extent = imageInfo.extent;
    image->size = size;

    if (vkCreateImage(context->device, &imageInfo, nullptr, &image->image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(context->device, image->image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(context, memRequirements.memoryTypeBits, memoryProperties);

    if (vkAllocateMemory(context->device, &allocInfo, nullptr, &image->memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(context->device, image->image, image->memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image->image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(context->device, &viewInfo, nullptr, &image->view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view!");
    }

    image->currentLayout = imageInfo.initialLayout;
}

void uploadDataToImageWithStagingBuffer(VulkanContext* context, VulkanImage* image, void* data) {
    VulkanBuffer stagingBuffer;
    createBuffer(
        context, 
        &stagingBuffer, image->size, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* mapped;
    vkMapMemory(context->device, stagingBuffer.memory, 0, image->size, 0, &mapped);
    memcpy(mapped, data, image->size);
    vkUnmapMemory(context->device, stagingBuffer.memory);

    copyBufferToImage(context, &stagingBuffer, image);

    vkDestroyBuffer(context->device, stagingBuffer.buffer, 0);
    vkFreeMemory(context->device, stagingBuffer.memory, 0);
}

void transitionLayout(VulkanContext* context, VulkanImage* image, VkImageLayout newLayout, VkCommandBuffer commandBuffer) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = image->currentLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image->image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (image->currentLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (image->currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (image->currentLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (image->currentLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
    image->currentLayout = newLayout;
}

void getDataFromImageWithStagingBuffer(VulkanContext* context, VulkanImage* image, void* data) {
    VulkanBuffer stagingBuffer;
    createBuffer(
        context,
        &stagingBuffer,
        image->size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    copyImageToBuffer(context, image, &stagingBuffer);

    void* mapped;
    vkMapMemory(context->device, stagingBuffer.memory, 0, image->size, 0, &mapped);
    memcpy(data, mapped, image->size);
    vkUnmapMemory(context->device, stagingBuffer.memory);

    vkDestroyBuffer(context->device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(context->device, stagingBuffer.memory, nullptr);
}

void destroyImage(VulkanContext* context, VulkanImage* image) {
    vkDestroyImageView(context->device, image->view, 0);
    vkDestroyImage(context->device, image->image, 0);
    vkFreeMemory(context->device, image->memory, 0);
}
