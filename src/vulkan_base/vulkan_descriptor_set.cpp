#include "vulkan/vulkan_core.h"
#include "vulkan_base.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

VulkanDescriptorSet* initDescriptorSet() {
    VulkanDescriptorSet* descriptorSet = new VulkanDescriptorSet;
    descriptorSet->layoutCount = 0;
    descriptorSet->descriptorSetLayoutBindings = {};
    return descriptorSet;
}

void addDescriptorSetLayout(VulkanDescriptorSet* descriptorSet, VkDescriptorType descriptorType) {
    VkDescriptorSetLayoutBinding layoutBinding;
    layoutBinding.binding = descriptorSet->layoutCount++;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.pImmutableSamplers = nullptr;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptorSet->descriptorTypeCount[descriptorType]++;

    descriptorSet->descriptorSetLayoutBindings.push_back(layoutBinding);
}

void createDescriptorSet(VulkanContext* context, VulkanDescriptorSet* descriptorSet) {
    {
        VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        createInfo.bindingCount = descriptorSet->layoutCount;
        createInfo.pBindings = descriptorSet->descriptorSetLayoutBindings.data();
        if (vkCreateDescriptorSetLayout(context->device, &createInfo, 0, &descriptorSet->descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for(auto typeCount : descriptorSet->descriptorTypeCount){
        poolSizes.push_back({typeCount.first, typeCount.second}); 
        // descriptorCount Needs to be multiplied by the count of Descriptorsets (in my case, its 1 anyways)
    }

    {
        VkDescriptorPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        createInfo.poolSizeCount = descriptorSet->descriptorTypeCount.size();
        createInfo.pPoolSizes = poolSizes.data();
        createInfo.maxSets = 1; // Is the number of descriptorsets, for now its only 1 as every compute shader will use the same data
        if(vkCreateDescriptorPool(context->device, &createInfo, 0, &descriptorSet->descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor pool!");
        }
    }

    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorSet->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSet->descriptorSetLayout;
    
    if(vkAllocateDescriptorSets(context->device, &allocInfo, &descriptorSet->descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate compute descriptorsets!");
    }
}

void fillDescriptorSet(VulkanContext *context, VulkanDescriptorSet *descriptorSet) {
    if(descriptorSet->layoutCount != descriptorSet->buffers.size()) {
        throw std::runtime_error("incorrect count of given buffers, must match the DescriptorSetLayoutBindings!");
    }

    std::vector<VkWriteDescriptorSet> writes(descriptorSet->layoutCount);

    for (size_t i = 0; i < descriptorSet->layoutCount; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descriptorSet->descriptorSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = descriptorSet->descriptorSetLayoutBindings[i].descriptorType;
        writes[i].descriptorCount = 1;

        switch (descriptorSet->descriptorSetLayoutBindings[i].descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                if (descriptorSet->buffers[i].type != VulkanDescriptorBufferInfo::Type::BUFFER) {
                    throw std::runtime_error("Invalid descriptor type while filling descriptorset with data");
                }
                writes[i].pBufferInfo = &descriptorSet->buffers[i].bufferInfo;
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                if (descriptorSet->buffers[i].type != VulkanDescriptorBufferInfo::Type::IMAGE) {
                        throw std::runtime_error("Invalid descriptor type while filling descriptorset with data");
                }
                writes[i].pImageInfo = &descriptorSet->buffers[i].imageInfo;
                break;
            default:
                throw std::runtime_error("Invalid descriptor type while filling descriptorset with data");
                break;
        }
    }
    vkUpdateDescriptorSets(context->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, 0);
}


void destroyDescriptorSet(VulkanContext *context, VulkanDescriptorSet *descriptorSet) {
    vkDestroyDescriptorPool(context->device, descriptorSet->descriptorPool, 0);
    vkDestroyDescriptorSetLayout(context->device, descriptorSet->descriptorSetLayout, 0);
}

void VulkanDescriptorSet::addBufferAndData(VulkanContext* context, VulkanBuffer* buffer, void* data, uint32_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties) {
    createBuffer(
        context, 
        buffer, 
        size, 
        usage, 
        memoryProperties
    );
    if (data != NULL) {
        uploadDataToBufferWithStagingBuffer(context, buffer, data, size);
    }

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buffer->buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = size;

    VulkanDescriptorBufferInfo info{};
    info.bufferInfo = bufferInfo;
    info.type = VulkanDescriptorBufferInfo::Type::BUFFER;

    this->buffers.push_back(info);
}

void VulkanDescriptorSet::addImageAndData(
    VulkanContext* context, 
    VulkanImage* image, void* data, size_t size,
    uint32_t width, uint32_t height, uint32_t depth, 
    VkFormat format, 
    VkImageUsageFlags usage, 
    VkMemoryPropertyFlags memoryProperties
) {
    createImage(
        context,
        image, size,
        width, height, depth,
        format,
        usage,
        memoryProperties
    );

    if (data != NULL) {
        uploadDataToImageWithStagingBuffer(context, image, data);
    }

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageLayout = image->currentLayout;
    imageInfo.imageView = image->view;

    VulkanDescriptorBufferInfo info{};
    info.imageInfo = imageInfo;
    info.type = VulkanDescriptorBufferInfo::Type::IMAGE;

    this->buffers.push_back(info);
}
