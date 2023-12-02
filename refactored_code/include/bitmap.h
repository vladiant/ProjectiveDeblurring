// -----------------------------------------------------
// bitmap.h
//
// header file for MS bitmap format
// -----------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// global I/O routines
std::vector<uint8_t> readBMP(const std::string& fname, int& width, int& height);
void readBMP(const std::string& fname, std::vector<float>& fImg, int& width,
             int& height);
void readBMP(const std::string& fname, std::vector<float>& fImgR,
             std::vector<float>& fImgG, std::vector<float>& fImgB, int& width,
             int& height);
void writeBMP(const std::string& iname, int width, int height, uint8_t* data);
void writeBMP(const std::string& iname, int width, int height,
              const std::vector<float>& data);
void writeBMP(const std::string& iname, int width, int height,
              const std::vector<float>& dataR, const std::vector<float>& dataG,
              const std::vector<float>& dataB);
