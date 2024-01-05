// ---------------------------------------------------------------------------
// bitmap.cpp
//
// handle MS bitmap I/O. For portability, we don't use the data structure
// defined in Windows.h. However, there is some strange thing, the size of our
// structure is different from what it should be though we define it in the
// same way as MS did. So, there is a hack, we use the hardcoded constant, 14,
// instead of the sizeof to calculate the size of the structure.  You are not
// supposed to worry about this part. However, I will appreciate if you find
// out the reason and let me know. Thanks.
// ---------------------------------------------------------------------------

#include "bitmap.h"

#include <cstring>
#include <fstream>
#include <iostream>

namespace {
constexpr auto BMP_BI_RGB = 0L;

using BMP_WORD = uint16_t;
using BMP_DWORD = uint32_t;
using BMP_LONG = int32_t;

struct BMP_BITMAPFILEHEADER {
  BMP_WORD bfType;
  BMP_DWORD bfSize;
  BMP_WORD bfReserved1;
  BMP_WORD bfReserved2;
  BMP_DWORD bfOffBits;
};

struct BMP_BITMAPINFOHEADER {
  BMP_DWORD biSize;
  BMP_LONG biWidth;
  BMP_LONG biHeight;
  BMP_WORD biPlanes;
  BMP_WORD biBitCount;
  BMP_DWORD biCompression;
  BMP_DWORD biSizeImage;
  BMP_LONG biXPelsPerMeter;
  BMP_LONG biYPelsPerMeter;
  BMP_DWORD biClrUsed;
  BMP_DWORD biClrImportant;
};
}  // namespace

template <class T>
void swapBytes(T* val) {
  static int typeSize;
  static char *start, *end;
  typeSize = sizeof(T);

  start = (char*)val;
  end = start + typeSize - 1;

  while (start < end) {
    *start ^= *end ^= *start ^= *end;
    start++;
    end--;
  }
}

// Bitmap data returned is (R,G,B) tuples in row-major order.
std::vector<uint8_t> readBMP(const std::string& fname, int& width,
                             int& height) {
  std::fstream file(fname, std::fstream::in | std::fstream::binary);
  BMP_DWORD pos = 0;

  if (!file) return {};

  //	I am doing file.read(reinterpret_cast<char*>(&bmfh),
  // sizeof(BMP_BITMAPFILEHEADER)) in a
  // safe way. :}
  BMP_BITMAPFILEHEADER bmfh{};
  file.read(reinterpret_cast<char*>(&(bmfh.bfType)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfSize)), 4);
  file.read(reinterpret_cast<char*>(&(bmfh.bfReserved1)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfReserved2)), 2);
  file.read(reinterpret_cast<char*>(&(bmfh.bfOffBits)), 4);

  pos = bmfh.bfOffBits;

  BMP_BITMAPINFOHEADER bmih{};
  file.read(reinterpret_cast<char*>(&bmih), sizeof(BMP_BITMAPINFOHEADER));

  // error checking
  if (bmfh.bfType != 0x4d42) {  // "BM" actually
    return {};
  }
  if (bmih.biBitCount != 24) return {};
  /*
          if ( bmih.biCompression != BMP_BI_RGB ) {
                  return {};
          }
  */
  file.seekg(pos, std::fstream::beg);

  width = bmih.biWidth;
  height = bmih.biHeight;

  std::cout << "readBMP1 " << width << " " << height << '\n';

  int padWidth = width * 3;
  int pad = 0;
  if (padWidth % 4 != 0) {
    pad = 4 - (padWidth % 4);
    padWidth += pad;
  }
  int bytes = height * padWidth;

  std::vector<uint8_t> data(bytes);

  file.read(reinterpret_cast<char*>(data.data()), bytes);

  if (!file) {
    return {};
  }

  // shuffle bitmap data such that it is (R,G,B) tuples in row-major order
  int i = 0, j = 0;
  j = 0;
  uint8_t temp = 0;
  uint8_t* in = nullptr;
  uint8_t* out = nullptr;

  in = data.data();
  out = data.data();

  for (j = 0; j < height; ++j) {
    for (i = 0; i < width; ++i) {
      out[1] = in[1];
      temp = in[2];
      out[2] = in[0];
      out[0] = temp;

      in += 3;
      out += 3;
    }
    in += pad;
  }

  return data;
}

void readBMP(const std::string& fname, std::vector<float>& fImg, int& width,
             int& height) {
  auto Img = readBMP(fname, width, height);
  fImg.resize(3 * width * height);
  auto pfImg = fImg.data();
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      pfImg[3 * index] = Img[3 * index] / 255.0f;
      pfImg[3 * index + 1] = Img[3 * index + 1] / 255.0f;
      pfImg[3 * index + 2] = Img[3 * index + 2] / 255.0f;
    }
  }

  std::cout << "readBMP " << width << " " << height << '\n';
}

void readBMP(const std::string& fname, std::vector<float>& fImgR,
             std::vector<float>& fImgG, std::vector<float>& fImgB, int& width,
             int& height) {
  std::cout << "readBMP fname " << fname << '\n';
  auto Img = readBMP(fname, width, height);
  fImgR.resize(width * height);
  fImgG.resize(width * height);
  fImgB.resize(width * height);
  auto* pfImgR = fImgR.data();
  auto* pfImgG = fImgG.data();
  auto* pfImgB = fImgB.data();
  int x = 0, y = 0, index = 0;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      pfImgR[index] = Img[3 * index] / 255.0f;
      pfImgG[index] = Img[3 * index + 1] / 255.0f;
      pfImgB[index] = Img[3 * index + 2] / 255.0f;
    }
  }

  std::cout << "readBMP " << width << " " << height << '\n';
}

void writeBMP(const std::string& iname, int width, int height,
              const std::vector<uint8_t>& data) {
  int bytes = 0, pad = 0;
  bytes = width * 3;
  pad = (bytes % 4) ? 4 - (bytes % 4) : 0;
  bytes += pad;
  bytes *= height;

  BMP_BITMAPFILEHEADER bmfh{};
  bmfh.bfType = 0x4d42;  // "BM"
  bmfh.bfSize =
      sizeof(BMP_BITMAPFILEHEADER) + sizeof(BMP_BITMAPINFOHEADER) + bytes;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = /*hack sizeof(BMP_BITMAPFILEHEADER)=14, sizeof doesn't
                      work?*/
      14 + sizeof(BMP_BITMAPINFOHEADER);

  BMP_BITMAPINFOHEADER bmih{};
  bmih.biSize = sizeof(BMP_BITMAPINFOHEADER);
  bmih.biWidth = width;
  bmih.biHeight = height;
  bmih.biPlanes = 1;
  bmih.biBitCount = 24;
  bmih.biCompression = BMP_BI_RGB;
  bmih.biSizeImage = 0;
  bmih.biXPelsPerMeter = (int)(100 / 2.54 * 72);
  bmih.biYPelsPerMeter = (int)(100 / 2.54 * 72);
  bmih.biClrUsed = 0;
  bmih.biClrImportant = 0;

  std::fstream outFile(iname, std::fstream::out | std::fstream::binary);

  //	outFile.write(reinterpret_cast<const char*>(&bmfh),
  // sizeof(BMP_BITMAPFILEHEADER));
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfType)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfSize)), 4);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfReserved1)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfReserved2)), 2);
  outFile.write(reinterpret_cast<const char*>(&(bmfh.bfOffBits)), 4);

  outFile.write(reinterpret_cast<const char*>(&bmih),
                sizeof(BMP_BITMAPINFOHEADER));

  bytes /= height;
  std::vector<uint8_t> scanline(bytes);
  for (int j = 0; j < height; ++j) {
    memcpy(scanline.data(), &data[j * 3 * width], 3 * width);
    for (int i = 0; i < width; ++i) {
      uint8_t temp = scanline[i * 3];
      scanline[i * 3] = scanline[i * 3 + 2];
      scanline[i * 3 + 2] = temp;
    }
    outFile.write(reinterpret_cast<const char*>(scanline.data()), bytes);
  }
}

void writeBMP(const std::string& iname, int width, int height,
              const std::vector<float>& data) {
  int x = 0, y = 0, index = 0;
  std::vector<uint8_t> Img(3 * width * height);
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (data[3 * index] < 0)
        Img[3 * index] = 0;
      else if (data[3 * index] > 1)
        Img[3 * index] = 255;
      else
        Img[3 * index] = (uint8_t)(data[3 * index] * 255.0f);
      if (data[3 * index + 1] < 0)
        Img[3 * index + 1] = 0;
      else if (data[3 * index + 1] > 1)
        Img[3 * index + 1] = 255;
      else
        Img[3 * index + 1] = (uint8_t)(data[3 * index + 1] * 255.0f);
      if (data[3 * index + 2] < 0)
        Img[3 * index + 2] = 0;
      else if (data[3 * index + 2] > 1)
        Img[3 * index + 2] = 255;
      else
        Img[3 * index + 2] = (uint8_t)(data[3 * index + 2] * 255.0f);
    }
  }
  writeBMP(iname, width, height, Img);
}
void writeBMP(const std::string& iname, int width, int height,
              const std::vector<float>& dataR, const std::vector<float>& dataG,
              const std::vector<float>& dataB) {
  int x = 0, y = 0, index = 0;
  std::vector<uint8_t> Img(3 * width * height);
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      if (dataR[index] < 0)
        Img[3 * index] = 0;
      else if (dataR[index] > 1)
        Img[3 * index] = 255;
      else
        Img[3 * index] = (uint8_t)(dataR[index] * 255.0f);
      if (dataG[index] < 0)
        Img[3 * index + 1] = 0;
      else if (dataG[index] > 1)
        Img[3 * index + 1] = 255;
      else
        Img[3 * index + 1] = (uint8_t)(dataG[index] * 255.0f);
      if (dataB[index] < 0)
        Img[3 * index + 2] = 0;
      else if (dataB[index] > 1)
        Img[3 * index + 2] = 255;
      else
        Img[3 * index + 2] = (uint8_t)(dataB[index] * 255.0f);
    }
  }
  writeBMP(iname, width, height, Img);
}