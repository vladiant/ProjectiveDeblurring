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

#include <iostream>

namespace {
constexpr auto BMP_BI_RGB = 0L;

typedef uint16_t BMP_WORD;
typedef uint32_t BMP_DWORD;
typedef int32_t BMP_LONG;

typedef struct {
  BMP_WORD bfType;
  BMP_DWORD bfSize;
  BMP_WORD bfReserved1;
  BMP_WORD bfReserved2;
  BMP_DWORD bfOffBits;
} BMP_BITMAPFILEHEADER;

typedef struct {
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
} BMP_BITMAPINFOHEADER;
}  // namespace

BMP_BITMAPFILEHEADER bmfh;
BMP_BITMAPINFOHEADER bmih;

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
std::vector<uint8_t> readBMP(const char* fname, int& width, int& height) {
  FILE* file;
  BMP_DWORD pos;

  if ((file = fopen(fname, "rb")) == NULL) return {};

  //	I am doing fread( &bmfh, sizeof(BMP_BITMAPFILEHEADER), 1, file ) in a
  // safe way. :}
  fread(&(bmfh.bfType), 2, 1, file);
  fread(&(bmfh.bfSize), 4, 1, file);
  fread(&(bmfh.bfReserved1), 2, 1, file);
  fread(&(bmfh.bfReserved2), 2, 1, file);
  fread(&(bmfh.bfOffBits), 4, 1, file);

  pos = bmfh.bfOffBits;

  fread(&bmih, sizeof(BMP_BITMAPINFOHEADER), 1, file);

  // error checking
  if (bmfh.bfType != 0x4d42) {  // "BM" actually
    return {};
  }
  if (bmih.biBitCount != 24) return {};
  /*
          if ( bmih.biCompression != BMP_BI_RGB ) {
                  return NULL;
          }
  */
  fseek(file, pos, SEEK_SET);

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

  int result = fread(data.data(), bytes, 1, file);

  if (!result) {
    return {};
  }

  fclose(file);

  // shuffle bitmap data such that it is (R,G,B) tuples in row-major order
  int i, j;
  j = 0;
  uint8_t temp;
  uint8_t* in;
  uint8_t* out;

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

void readBMP(const char* fname, std::vector<float>& fImg, int& width,
             int& height) {
  auto Img = readBMP(fname, width, height);
  fImg.resize(3 * width * height);
  auto pfImg = fImg.data();
  int x, y, index;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      pfImg[3 * index] = Img[3 * index] / 255.0f;
      pfImg[3 * index + 1] = Img[3 * index + 1] / 255.0f;
      pfImg[3 * index + 2] = Img[3 * index + 2] / 255.0f;
    }
  }

  std::cout << "readBMP " << width << " " << height << '\n';
}

void readBMP(const char* fname, std::vector<float>& fImgR,
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
  int x, y, index;
  for (y = 0, index = 0; y < height; y++) {
    for (x = 0; x < width; x++, index++) {
      pfImgR[index] = Img[3 * index] / 255.0f;
      pfImgG[index] = Img[3 * index + 1] / 255.0f;
      pfImgB[index] = Img[3 * index + 2] / 255.0f;
    }
  }

  std::cout << "readBMP " << width << " " << height << '\n';
}

void writeBMP(const char* iname, int width, int height,
              const std::vector<uint8_t>& data) {
  int bytes, pad;
  bytes = width * 3;
  pad = (bytes % 4) ? 4 - (bytes % 4) : 0;
  bytes += pad;
  bytes *= height;

  bmfh.bfType = 0x4d42;  // "BM"
  bmfh.bfSize =
      sizeof(BMP_BITMAPFILEHEADER) + sizeof(BMP_BITMAPINFOHEADER) + bytes;
  bmfh.bfReserved1 = 0;
  bmfh.bfReserved2 = 0;
  bmfh.bfOffBits = /*hack sizeof(BMP_BITMAPFILEHEADER)=14, sizeof doesn't
                      work?*/
      14 + sizeof(BMP_BITMAPINFOHEADER);

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

  FILE* outFile = fopen(iname, "wb");

  //	fwrite(&bmfh, sizeof(BMP_BITMAPFILEHEADER), 1, outFile);
  fwrite(&(bmfh.bfType), 2, 1, outFile);
  fwrite(&(bmfh.bfSize), 4, 1, outFile);
  fwrite(&(bmfh.bfReserved1), 2, 1, outFile);
  fwrite(&(bmfh.bfReserved2), 2, 1, outFile);
  fwrite(&(bmfh.bfOffBits), 4, 1, outFile);

  fwrite(&bmih, sizeof(BMP_BITMAPINFOHEADER), 1, outFile);

  bytes /= height;
  std::vector<uint8_t> scanline(bytes);
  for (int j = 0; j < height; ++j) {
    memcpy(scanline.data(), &data[j * 3 * width], bytes);
    for (int i = 0; i < width; ++i) {
      uint8_t temp = scanline[i * 3];
      scanline[i * 3] = scanline[i * 3 + 2];
      scanline[i * 3 + 2] = temp;
    }
    fwrite(scanline.data(), bytes, 1, outFile);
  }

  fclose(outFile);
}

void writeBMP(const char* iname, int width, int height,
              const std::vector<float>& data) {
  int x, y, index;
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
void writeBMP(const char* iname, int width, int height,
              const std::vector<float>& dataR, const std::vector<float>& dataG,
              const std::vector<float>& dataB) {
  int x, y, index;
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