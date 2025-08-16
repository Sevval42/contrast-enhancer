#version 450

layout(set = 0, binding = 0) buffer InputOutputBuffer {
    float data[];
} bufferData;

layout(set = 0, binding = 1) uniform UniformBuffer {
    float offset;
} uniformBuffer;

layout(local_size_x = 1) in;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    bufferData.data[idx] *= 5;
}
