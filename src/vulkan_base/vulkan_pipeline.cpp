#include "vulkan/vulkan_core.h"
#include "vulkan_base.h"
#include <iostream>
#include <cassert>

VkShaderModule createShaderModule(VulkanContext* context, const char* shaderFilename) {
    VkShaderModule result;

    FILE* file = fopen(shaderFilename, "rb");
    if(!file) {
        std::cerr << "File " << shaderFilename << " not found!" << std::endl;
        return result;
    }
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    assert((fileSize & 0x03) == 0);
    uint8_t* buffer = new uint8_t[fileSize];
    fread(buffer, 1, fileSize, file);

    VkShaderModuleCreateInfo createInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = fileSize;
    createInfo.pCode = (uint32_t*)buffer;
    vkCreateShaderModule(context->device, &createInfo, 0, &result);

    delete[] buffer;
    fclose(file);

    return result;
}



VulkanPipeline createPipeline(VulkanContext* context, std::vector<const char*> computeShaderFilenames, std::vector<ivec3> dispatches, VulkanDescriptorSet* descriptorSet) {
    VkPipelineLayout pipelineLayout;
    {
        VkPipelineLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &descriptorSet->descriptorSetLayout;
        vkCreatePipelineLayout(context->device, &createInfo, 0, &pipelineLayout);
    }

    
    std::vector<VkPipeline> pipelines;
    for (auto fileName : computeShaderFilenames) {
        VkPipeline pipeline;

        {
            VkShaderModule computeShaderModule = createShaderModule(context, fileName);
        
            VkPipelineShaderStageCreateInfo shaderStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStage.module = computeShaderModule;
            shaderStage.pName = "main";

            VkComputePipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            createInfo.layout = pipelineLayout;
            createInfo.stage = shaderStage;
            vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &createInfo, 0, &pipeline);
            pipelines.push_back(pipeline);

            vkDestroyShaderModule(context->device, computeShaderModule, 0);
        }
    }

    VulkanPipeline result = {};
    result.pipelines = pipelines;
    result.pipelineLayout = pipelineLayout;
    result.dispatchSizes = dispatches;

    return result;
}

void destroyPipeline(VulkanContext* context, VulkanPipeline* pipeline) {
    for(auto vkPipeline : pipeline->pipelines){
        vkDestroyPipeline(context->device, vkPipeline, 0);
    }
    vkDestroyPipelineLayout(context->device, pipeline->pipelineLayout, 0);
}