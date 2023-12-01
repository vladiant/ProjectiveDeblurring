// -----------------------------------------------------
// bitmap.h
//
// header file for MS bitmap format
// -----------------------------------------------------

#pragma once

#include <stdio.h>
#include <string.h>

#include <cstdint>
#include <vector>

// global I/O routines
std::vector<uint8_t> readBMP(const char* fname, int& width, int& height);
void readBMP(const char* fname, std::vector<float>& fImg, int& width,
             int& height);
void readBMP(const char* fname, std::vector<float>& fImgR,
             std::vector<float>& fImgG, std::vector<float>& fImgB, int& width,
             int& height);
void writeBMP(const char* iname, int width, int height, uint8_t* data);
void writeBMP(const char* iname, int width, int height,
              const std::vector<float>& data);
void writeBMP(const char* iname, int width, int height,
              const std::vector<float>& dataR, const std::vector<float>& dataG,
              const std::vector<float>& dataB);
