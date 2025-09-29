#include <vector>

std::vector<float> loadImage(std::string fileName, int* width, int* height, int* channels);

std::vector<float> loadFits(
    const char* fileNameX,
    const char* fileNameY,
    const char* fileNameZ,
    int* width,
    int* height,
    float weightR = 1.0f,
    float weightG = 1.0f,
    float weightB = 1.0f
);