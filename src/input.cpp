#include "input.h"
#ifdef HAVE_CFITSIO
#include <fitsio.h>
#endif
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::vector<float> loadImage(std::string fileName, int* width, int* height, int* channels) {
    stbi_ldr_to_hdr_gamma(1.0f);
    float* pixels = stbi_loadf(fileName.c_str(), width, height, channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load image");
    }
    size_t size = (*width) * (*height) * 4;
    std::vector<float> image(pixels, pixels + size);
    stbi_image_free(pixels);
    return image;
}

float computePercentile(const std::vector<float>& data, float percentile, float ignoreValue) {
    std::vector<float> valid;
    valid.reserve(data.size());
    for (float v : data)
        if (v != ignoreValue && !std::isnan(v)) valid.push_back(v);
    if (valid.empty()) return 0.0f;
    std::sort(valid.begin(), valid.end());
    double idx = percentile * (valid.size() - 1);
    size_t i0 = static_cast<size_t>(std::floor(idx));
    size_t i1 = std::min(i0 + 1, valid.size() - 1);
    double frac = idx - i0;
    return valid[i0] * (1.0 - frac) + valid[i1] * frac;
}

#if HAVE_CFITSIO
// written with chat-gpt
// Loads single FITS file
std::vector<float> loadFit(const char* fileName, int* width, int* height) {
    int status = 0;
    fitsfile *fptr;

    if (fits_open_file(&fptr, fileName, READONLY, &status)) {
        fits_report_error(stderr, status);
        throw std::runtime_error("Failed to open FITS file");
    }

    int n_hdus = 0;
    fits_get_num_hdus(fptr, &n_hdus, &status);
    if (status) { fits_report_error(stderr, status); throw std::runtime_error("Failed to get HDUs"); }

    int bitpix, naxis;
    long naxes[3] = {1,1,1};
    std::vector<float> image;
    bool found = false;

    for (int hdu = 1; hdu <= n_hdus; ++hdu) {
        int hdutype;
        fits_movabs_hdu(fptr, hdu, &hdutype, &status);
        if (status) break;
        if (hdutype == IMAGE_HDU) {
            fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
            if (status) break;
            if (naxis == 2) {
                *width  = static_cast<int>(naxes[0]);
                *height = static_cast<int>(naxes[1]);

                image.resize((*width) * (*height));
                long fpixel[2] = {1,1};
                if (fits_read_pix(fptr, TFLOAT, fpixel, (*width) * (*height),
                                  nullptr, image.data(), nullptr, &status)) {
                    fits_report_error(stderr, status);
                    throw std::runtime_error("Failed to read pixels");
                }
                found = true;
                break;
            }
        }
    }

    fits_close_file(fptr, &status);

    if (!found) throw std::runtime_error("No 2D image found in FITS file");

    return image;
}

// written with chat-gpt
// Loads 3 FITS files and combines into RGBA with per-channel stretch and weights
std::vector<float> loadFits(
    const char* fileNameR,
    const char* fileNameG,
    const char* fileNameB,
    int* width,
    int* height,
    float weightR,
    float weightG,
    float weightB
) {
    int wx, wy, wz, hx, hy, hz;
    std::vector<float> channelR = loadFit(fileNameR, &wx, &hx);
    std::vector<float> channelG = loadFit(fileNameG, &wy, &hy);
    std::vector<float> channelB = loadFit(fileNameB, &wz, &hz);

    if (wx != wy || wy != wz || hx != hy || hy != hz)
        throw std::runtime_error("FITS files have different sizes");

    *width = wx;
    *height = hx;

    float ignoreValue = channelR[0];

    float pR_low = computePercentile(channelR, 0.005f, ignoreValue);
    float pR_high = computePercentile(channelR, 0.995f, ignoreValue);
    if (pR_high == pR_low) pR_high = pR_low + 1e-6f;

    float pG_low = computePercentile(channelG, 0.005f, ignoreValue);
    float pG_high = computePercentile(channelG, 0.995f, ignoreValue);
    if (pG_high == pG_low) pG_high = pG_low + 1e-6f;

    float pB_low = computePercentile(channelB, 0.005f, ignoreValue);
    float pB_high = computePercentile(channelB, 0.995f, ignoreValue);
    if (pB_high == pB_low) pB_high = pB_low + 1e-6f;

    auto stretch = [](float v, float p_low, float p_high, float weight, float ignoreValue) -> float {
        if (v == ignoreValue || std::isnan(v)) return 0.0f;
        float nv = (v - p_low) / (p_high - p_low);
        nv = fmin(1.0, fmax(nv, 0.0f));
        return weight * std::asinh(1.0f * nv) / std::asinh(1.0f);
    };

    std::vector<float> image(wx * hx * 4);
    for (size_t i = 0; i < channelR.size(); ++i) {
        size_t idx = i * 4;
        image[idx]   = stretch(channelR[i], pR_low, pR_high, weightR, ignoreValue);
        image[idx+1] = stretch(channelG[i], pG_low, pG_high, weightG, ignoreValue);
        image[idx+2] = stretch(channelB[i], pB_low, pB_high, weightB, ignoreValue);
        image[idx+3] = (channelR[i] == ignoreValue) ? 0.0f : 1.0f;
    }

    return image;
}
#else 
std::vector<float> loadFits(
    const char* fileNameR,
    const char* fileNameG,
    const char* fileNameB,
    int* width,
    int* height,
    float weightR,
    float weightG,
    float weightB
) {
    throw std::runtime_error("FITS loading is only supported on macOS (CFITSIO not available)");
}
#endif
// loads csv files for the image data. They need the layout: r,g,b,label
std::vector<float> loadCsv(const char* filename, int labelCount, int* width, int* height, int* channels) {
    *channels = 4;

    std::ifstream file((std::string(filename)));

    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: ");
    }

    std::vector<float> data;
    std::string line;

    std::getline(file, line);

    int count = 0;
    int numPixels = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                if(count%4 != 3){
                    data.push_back(std::stof(cell));
                }else{
                    data.push_back((std::stof(cell) + 1) / ((float)labelCount+1));
                }
                count++;
            } catch (const std::invalid_argument&) {
                continue;
            }
        }
        numPixels++;
    }

    file.close();

    *width = STRIDE;
    *height = (numPixels + (*width) - 1) / (*width);

    size_t totalPixels = (*width) * (*height);
    data.resize(totalPixels * (*channels), 0.0f);

    for(int i = count; i < *width*(*height); i++){
        data.push_back(0);
    }

    return data;
}

void saveNDToCsv(const char* filename, int labelCount, VulkanContext* context, VulkanImage* image, uint32_t imageSize) {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for writing");

    file << "X,Y,Z,label\n";

    std::vector<float> data(imageSize/sizeof(float));
    getDataFromImageWithStagingBuffer(context, image, data.data());
    int channels = 4;
    size_t numPixels = image->extent.width * image->extent.height;
    for (size_t i = 0; i < numPixels; ++i) {
        size_t idx = i * channels;
        float r = data[idx];
        float g = data[idx + 1];
        float b = data[idx + 2];

        float alpha = data[idx + 3];
        if(alpha == 0) continue;
        int label = static_cast<int>(alpha * (labelCount + 1) + 0.5f) - 1;
        
        file << r << "," << g << "," << b << "," << label << "\n";
    }

    file.close();
}
