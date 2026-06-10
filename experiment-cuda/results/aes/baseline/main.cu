#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
// ---- INLINED: SDKBitMap.h (from /home/WillFu/parallel/final/HeCBench/src/include/SDKBitMap.h) ----
/**********************************************************************
  Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  �   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  �   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/
#ifndef SDKBITMAP_H_
#define SDKBITMAP_H_

/**
 * Headers
 */
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

static const short bitMapID = 19778;


/**
 * fixme this needs to be moved to common types header?
 */
#pragma pack(push,1)

#ifdef _OPENMP
/**
 * uchar4
 * struct implements a vector of chars
 */
typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;
#endif

/**
 * ColorPelette of type uchar4
 */
typedef uchar4 ColorPalette;

/**
 * struct Bitmap header
 */
typedef struct
{
  short id;
  int size;
  short reserved1;
  short reserved2;
  int offset;
} BitMapHeader;


/**
 * struct Bitmap info header
 */
typedef struct
{
  int sizeInfo;
  int width;
  int height;
  short planes;
  short bitsPerPixel;
  unsigned compression;
  unsigned imageSize;
  int xPelsPerMeter;
  int yPelsPerMeter;
  int clrUsed;
  int clrImportant;
} BitMapInfoHeader;

/**
 *class Bitmap used to load a bitmap image from a file.
 */
class SDKBitMap : public BitMapHeader, public BitMapInfoHeader
{
  private:
    uchar4 * pixels_;               /**< Pixel Data */
    int numColors_;                 /**< Number of colors */
    ColorPalette * colors_;         /**< Color Data */
    bool isLoaded_;                 /**< If Bitmap loaded */
    void releaseResources(void)     /**< Release Resources */
    {
      if (pixels_ != NULL)
      {
        delete[] pixels_;
      }
      if (colors_ != NULL)
      {
        delete[] colors_;
      }
      pixels_    = NULL;
      colors_    = NULL;
      isLoaded_  = false;
    }
    int colorIndex(uchar4 color)    /**< get a color index */
    {
      for (int i = 0; i < numColors_; i++)
      {
#if defined(SYCL_LANGUAGE_VERSION)
        if (colors_[i].x() == color.x() && colors_[i].y() == color.y() &&
            colors_[i].z() == color.z() && colors_[i].w() == color.w())
#else
        if (colors_[i].x == color.x && colors_[i].y == color.y &&
            colors_[i].z == color.z && colors_[i].w == color.w)
#endif
        {
          return i;
        }
      }
      return SDK_SUCCESS;
    }
  public:

    /**
     * brief Default constructor
     */
    SDKBitMap() :
      pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) {}

    /**
     * brief Constructor
     * Tries to load bitmap image from filename provided.
     *
     * @param filename pointer to null terminated string that is the path and
     * filename to the bitmap image to be loaded.
     *
     * In the case of an error, e.g. the bitmap file could not be loaded for
     * some reason, then a following call to isLoaded will return false.
     */
    SDKBitMap(const char * filename) :
      pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) 
    {
      load(filename);
    }

    /**
     * Copy constructor
     *
     * @param rhs is the bitmap to be copied (cloned).
     */
    SDKBitMap(const SDKBitMap& rhs)
    {
      *this = rhs;
    }

    /**
     * Destructor
     */
    ~SDKBitMap()
    {
      releaseResources();
    }

    /**
     * Assignment operator
     * @param rhs is the bitmap to be assigned (cloned).
     */
    SDKBitMap& operator=(const SDKBitMap& rhs)
    {
      if (this == &rhs)
      {
        return *this;
      }
      // header
      id             = rhs.id;
      size           = rhs.size;
      reserved1      = rhs.reserved1;
      reserved2      = rhs.reserved2;
      offset         = rhs.offset;
      // header info
      sizeInfo       = rhs.sizeInfo;
      width          = rhs.width;
      height         = rhs.height;
      planes         = rhs.planes;
      bitsPerPixel   = rhs.bitsPerPixel;
      compression    = rhs.compression;
      imageSize      = rhs.imageSize;
      xPelsPerMeter  = rhs.xPelsPerMeter;
      yPelsPerMeter  = rhs.yPelsPerMeter;
      clrUsed        = rhs.clrUsed;
      clrImportant   = rhs.clrImportant;
      numColors_     = rhs.numColors_;
      isLoaded_      = rhs.isLoaded_;
      pixels_        = NULL;
      colors_        = NULL;
      if (isLoaded_)
      {
        if (rhs.colors_ != NULL)
        {
          colors_ = new ColorPalette[numColors_];
          if (colors_ == NULL)
          {
            isLoaded_ = false;
            return *this;
          }
          memcpy(colors_, rhs.colors_, numColors_ * sizeof(ColorPalette));
        }
        pixels_ = new uchar4[width * height];
        if (pixels_ == NULL)
        {
          delete[] colors_;
          colors_   = NULL;
          isLoaded_ = false;
          return *this;
        }
        memcpy(pixels_, rhs.pixels_, width * height * sizeof(uchar4));
      }
      return *this;
    }

    /**
     * Load Bitmap image
     *
     * @param filename is a pointer to a null terminated string that is the
     * path and filename name to the the bitmap file to be loaded.
     *
     * @return In the case of an error, e.g. the bitmap file could not be loaded for
     * some reason, then a following call to isLoaded will return false.
     */
    void
      load(const char * filename)
      {
        size_t val;
        // Release any existing resources
        releaseResources();
        // Open BMP file
        FILE * fd = fopen(filename, "rb");
        // FILE *fd;
        //fopen_s(&fd, filename, "rb");
        // Opened OK
        if (fd != NULL)
        {
          // Read header
          val = fread((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
#ifdef DEBUG
          printf("%d %d\n", size, offset);
#endif
          // Failed to read header
          if (val != 1) 
          {
            fclose(fd);
            return;
          }
          // Confirm that we have a bitmap file
          if (id != bitMapID)
          {
            fclose(fd);
            return;
          }
          // Read map info header
          val = fread((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
#ifdef DEBUG
          printf("%d, %d, %d %d %d, %u %u %d %d %d %d\n",
                  sizeInfo,
                  width,
                  height,
                  planes,
                  bitsPerPixel,
                  compression,
                  imageSize,
                  xPelsPerMeter,
                  yPelsPerMeter,
                  clrUsed,
                  clrImportant);
#endif
          // Failed to read map info header
          if (val != 1) 
          {
            fclose(fd);
            return;
          }

          // No support for compressed images
          if (compression)
          {
            fclose(fd);
            return;
          }
          // Support only 8 or 24 bits images
          if (bitsPerPixel != 8 && bitsPerPixel != 24)
          {
            fclose(fd);
            return;
          }
          int sizeBuffer = size - offset;
          if (width * height * (bitsPerPixel / 8) != sizeBuffer) {
            printf("This is not a valid bitmap file.\n");
            fclose(fd);
            return;
          }
          // Store number of colors
          numColors_ = 1 << bitsPerPixel;
          //load the palate for 8 bits per pixel
          if(bitsPerPixel == 8)
          {
            colors_ = new ColorPalette[numColors_];
            if (colors_ == NULL)
            {
              fclose(fd);
              return;
            }
            val  = fread(
                (char *)colors_,
                numColors_ * sizeof(ColorPalette),
                1,
                fd);

            // Failed to read colors
            if (val != 1) 
            {
              fclose(fd);
              return;
            }
          }
          // Allocate buffer to hold all pixels
          unsigned char * tmpPixels = new unsigned char[sizeBuffer];
          if (tmpPixels == NULL)
          {
            delete colors_;
            colors_ = NULL;
            fclose(fd);
            return;
          }
          // Read pixels from file, including any padding
          val = fread(tmpPixels, sizeBuffer * sizeof(unsigned char), 1, fd);
          // Failed to read pixel data
          if (val != 1) 
          {
            delete colors_;
            colors_ = NULL;
            delete[] tmpPixels;
            fclose(fd);
            return;
          }
          // Allocate image
          pixels_ = new uchar4[width * height];
          if (pixels_ == NULL)
          {
            delete colors_;
            colors_ = NULL;
            delete[] tmpPixels;
            fclose(fd);
            return;
          }
          // Set image, including w component (white)
          memset(pixels_, 0xff, width * height * sizeof(uchar4));
          unsigned int index = 0;
          for(int y = 0; y < height; y++)
          {
            for(int x = 0; x < width; x++)
            {
              // Read RGB values
              if (bitsPerPixel == 8)
              {
                pixels_[(y * width + x)] = colors_[tmpPixels[index++]];
              }
              else   // 24 bit
              {
#if defined(SYCL_LANGUAGE_VERSION)
                pixels_[(y * width + x)].z() = tmpPixels[index++];
                pixels_[(y * width + x)].y() = tmpPixels[index++];
                pixels_[(y * width + x)].x() = tmpPixels[index++];
#else
                pixels_[(y * width + x)].z = tmpPixels[index++];
                pixels_[(y * width + x)].y = tmpPixels[index++];
                pixels_[(y * width + x)].x = tmpPixels[index++];
#endif
              }
            }
            // Handle padding
            for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
            {
              index++;
            }
          }
          // Loaded file so we can close the file.
          fclose(fd);
          delete[] tmpPixels;
          // Loaded file so record this fact
          isLoaded_  = true;
        }
        else 
        {
          fprintf(stderr, "Failed to load file %s\n", filename);
        }
      }

    /**
     * Write Bitmap image
     *
     * @param filename is a pointer to a null terminated string that is the
     * path and filename name to the the bitmap file to be written.
     *
     * @return In the case that the bitmap is written true is returned. In
     * the case that a bitmap image is not already loaded or the write fails
     * for some reason false is returned.
     */
    bool
      write(const char * filename)
      {
        if (!isLoaded_)
        {
          return false;
        }
        // Open BMP file
        FILE * fd = fopen(filename, "wb");
        //FILE * fd;
        //fopen_s(&fd, filename, "wb");
        // Opened OK
        if (fd != NULL)
        {
          // Write header
          fwrite((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
          // Failed to write header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          // Write map info header
          fwrite((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
          // Failed to write map info header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          // Write palate for 8 bits per pixel
          if(bitsPerPixel == 8)
          {
            fwrite(
                (char *)colors_,
                numColors_ * sizeof(ColorPalette),
                1,
                fd);
            // Failed to write colors
            if (ferror(fd))
            {
              fclose(fd);
              return false;
            }
          }
          for(int y = 0; y < height; y++)
          {
            for(int x = 0; x < width; x++)
            {
              // Read RGB values
              if (bitsPerPixel == 8)
              {
                fputc(
                    colorIndex(
                      pixels_[(y * width + x)]),
                    fd);
              }
              else   // 24 bit
              {
#if defined(SYCL_LANGUAGE_VERSION)
                fputc(pixels_[(y * width + x)].z(), fd);
                fputc(pixels_[(y * width + x)].y(), fd);
                fputc(pixels_[(y * width + x)].x(), fd);
#else
                fputc(pixels_[(y * width + x)].z, fd);
                fputc(pixels_[(y * width + x)].y, fd);
                fputc(pixels_[(y * width + x)].x, fd);
#endif
                if (ferror(fd))
                {
                  fclose(fd);
                  return false;
                }
              }
            }
            // Add padding
            for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
            {
              fputc(0, fd);
            }
          }
          return true;
        }
        return false;
      }

    bool
      write(const char * filename, int width, int height, unsigned int *ptr)
      {
        // Open BMP file
        FILE * fd = fopen(filename, "wb");
        int alignSize  = width * 4;
        alignSize ^= 0x03;
        alignSize ++;
        alignSize &= 0x03;
        int rowLength = width * 4 + alignSize;
        // Opened OK
        if (fd != NULL)
        {
          BitMapHeader *bitMapHeader = new BitMapHeader;
          bitMapHeader->id = bitMapID;
          bitMapHeader->offset = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader);
          bitMapHeader->reserved1 = 0x0000;
          bitMapHeader->reserved2 = 0x0000;
          bitMapHeader->size = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader) + rowLength
            * height;
          // Write header
          fwrite(bitMapHeader, sizeof(BitMapHeader), 1, fd);
          // Failed to write header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          BitMapInfoHeader *bitMapInfoHeader = new BitMapInfoHeader;
          bitMapInfoHeader->bitsPerPixel = 32;
          bitMapInfoHeader->clrImportant = 0;
          bitMapInfoHeader->clrUsed = 0;
          bitMapInfoHeader->compression = 0;
          bitMapInfoHeader->height = height;
          bitMapInfoHeader->imageSize = rowLength * height;
          bitMapInfoHeader->planes = 1;
          bitMapInfoHeader->sizeInfo = sizeof(BitMapInfoHeader);
          bitMapInfoHeader->width = width;
          bitMapInfoHeader->xPelsPerMeter = 0;
          bitMapInfoHeader->yPelsPerMeter = 0;
          // Write map info header
          fwrite(bitMapInfoHeader, sizeof(BitMapInfoHeader), 1, fd);
          // Failed to write map info header
          if (ferror(fd))
          {
            fclose(fd);
            return false;
          }
          unsigned char buffer[4];
          int x, y;
          for (y = 0; y < height; y++)
          {
            for (x = 0; x < width; x++, ptr++)
            {
              if( 4 != fwrite(ptr, 1, 4, fd))
              {
                fclose(fd);
                return false;
              }
            }
            memset( buffer, 0x00, 4 );
            fwrite( buffer, 1, alignSize, fd );
          }
          fclose( fd );
          return true;
        }
        return false;
      }

    /**
     * Get image width
     *
     * @return If a bitmap image has been successfully loaded, then the width
     * image is returned, otherwise -1;
     */
    int
      getWidth(void) const
      {
        if (isLoaded_)
        {
          return width;
        }
        else
        {
          return -1;
        }
      }


    /**
     * Get Number of Channels
     *
     * @return the number of channels used in image, otherwise -1.
     */
    int getNumChannels()
    {
      if (isLoaded_)
      {
        return bitsPerPixel / 8;
      }
      else
      {
        return SDK_FAILURE;
      }
    }

    /**
     * Get image height
     *
     * @return If a bitmap image has been successfully loaded, then the height
     * image is returned, otherwise -1.
     */
    int
      getHeight(void) const
      {
        if (isLoaded_)
        {
          return height;
        }
        else
        {
          return -1;
        }
      }

    /**
     * Get image width
     *
     * @return If a bitmap image has been successfully loaded, then returns
     * a pointer to image's pixels, otherwise NULL.
     */
    uchar4 * getPixels(void) const
    {
      return pixels_;
    }

    /**
     * Is an image currently loaded
     *
     * @return If a bitmap image has been successfully loaded, then returns
     * true, otherwise if an image could not be loaded or an image has yet
     * to be loaded false is returned.
     */
    bool
      isLoaded(void) const
      {
        return isLoaded_;
      }

};
#pragma pack(pop)
#endif //CL_BITMAP

// ---- END INLINED: SDKBitMap.h ----

// ---- INLINED: aes.h (from /home/WillFu/parallel/final/HeCBench/src/aes-cuda/aes.h) ----

#ifndef AESENCRYPTDECRYPT_H_
#define AESENCRYPTDECRYPT_H_

typedef unsigned char uchar;

uchar sbox[256] = 
{  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76 //0
 , 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0 //1
 , 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15 //2
 , 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75 //3
 , 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84 //4
 , 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf //5
 , 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8 //6
 , 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2 //7
 , 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73 //8
 , 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb //9
 , 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79 //A
 , 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08 //B
 , 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a //C
 , 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e //D
 , 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf //E
 , 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};//F
//0      1    2      3     4    5     6     7      8    9     A      B    C     D     E     F


uchar rsbox[256] =
{ 0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb
, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb
, 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e
, 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25
, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92
, 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84
, 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06
, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b
, 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73
, 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e
, 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b
, 0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4
, 0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f
, 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef
, 0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61
, 0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};       


uchar Rcon[255] = 
{ 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a
, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39
, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a
, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8
, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef
, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc
, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b
, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3
, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94
, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35
, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f
, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04
, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63
, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd
, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb      };


#endif

// ---- END INLINED: aes.h ----

// ---- INLINED: kernels.cu (from /home/WillFu/parallel/final/HeCBench/src/aes-cuda/kernels.cu) ----
inline __device__ uchar4 operator^(uchar4 a, uchar4 b)
{
  return make_uchar4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ void operator^=(uchar4 &a, const uchar4 b)
{
  a.x ^= b.x;
  a.y ^= b.y;
  a.z ^= b.z;
  a.w ^= b.w;
}

__host__ __device__
uchar galoisMultiplication(uchar a, uchar b)
{
  uchar p = 0; 
  for(unsigned int i=0; i < 8; ++i)
  {
    if((b&1) == 1)
    {
      p^=a;
    }
    uchar hiBitSet = (a & 0x80);
    a <<= 1;
    if(hiBitSet == 0x80)
    {
      a ^= 0x1b;
    }
    b >>= 1;
  }
  return p;
}

inline __device__
uchar4 sboxRead(const uchar * SBox, uchar4 block)
{
  return make_uchar4(SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]);
}

__device__
uchar4 mixColumns(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
  unsigned int bw = 4;

  uchar x, y, z, w;

  x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
  y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
  z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
  w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);

  for(unsigned int k=1; k< 4; ++k)
  {
    x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
    y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
    z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
    w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
  }

  return make_uchar4(x, y, z, w);
}

__device__
uchar4 shiftRows(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    //r.xyzw() = r.yzwx();
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = make_uchar4(y,z,w,x);
  }
  return r;
}

__global__
void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                const uint     width , 
                const uint     rounds )

{
  __shared__ uchar4 block0[4];
  __shared__ uchar4 block1[4];

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  //unsigned int localIdx = threadIdx.x;
  unsigned int localIdy = threadIdx.y;

  unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
  unsigned int localIndex  = localIdy;

  uchar4 galiosCoeff[4];
  galiosCoeff[0] = make_uchar4(2, 0, 0, 0);
  galiosCoeff[1] = make_uchar4(3, 0, 0, 0);
  galiosCoeff[2] = make_uchar4(1, 0, 0, 0);
  galiosCoeff[3] = make_uchar4(1, 0, 0, 0);

  block0[localIndex]  = input[globalIndex];

  block0[localIndex] ^= roundKey[localIndex];

  for(unsigned int r=1; r < rounds; ++r)
  {
    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

    __syncthreads();
    block1[localIndex]  = mixColumns(block0, galiosCoeff, localIndex); 

    __syncthreads();
    block0[localIndex] = block1[localIndex]^roundKey[r*4 + localIndex];
  }
  block0[localIndex] = sboxRead(SBox, block0[localIndex]);

  block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

  output[globalIndex] =  block0[localIndex]^roundKey[(rounds)*4 + localIndex];
}

__device__
uchar4 shiftRowsInv(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    // r = r.wxyz();
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = make_uchar4(w,x,y,z);
  }
  return r;
}

__global__
void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                const  uint    width , 
                const  uint    rounds)

{
  __shared__ uchar4 block0[4];
  __shared__ uchar4 block1[4];

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  //unsigned int localIdx = threadIdx.x;
  unsigned int localIdy = threadIdx.y;

  unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
  unsigned int localIndex  = localIdy;

  uchar4 galiosCoeff[4];
  galiosCoeff[0] = make_uchar4(14, 0, 0, 0);
  galiosCoeff[1] = make_uchar4(11, 0, 0, 0);
  galiosCoeff[2] = make_uchar4(13, 0, 0, 0);
  galiosCoeff[3] = make_uchar4( 9, 0, 0, 0);

  block0[localIndex]  = input[globalIndex];

  block0[localIndex] ^= roundKey[4*rounds + localIndex];

  for(unsigned int r=rounds -1 ; r > 0; --r)
  {
    block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    __syncthreads();
    block1[localIndex] = block0[localIndex]^roundKey[r*4 + localIndex];

    __syncthreads();
    block0[localIndex]  = mixColumns(block1, galiosCoeff, localIndex); 
  }  

  block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

  block0[localIndex] = sboxRead(SBox, block0[localIndex]);

  output[globalIndex] =  block0[localIndex]^roundKey[localIndex];
}

// ---- END INLINED: kernels.cu ----

// ---- INLINED: reference.cu (from /home/WillFu/parallel/final/HeCBench/src/aes-cuda/reference.cu) ----
uchar getRconValue(unsigned int num)
{
  return Rcon[num];
}

uchar getSBoxValue(unsigned int num)
{
  return sbox[num];
}

uchar getSBoxInvert(unsigned int num)
{
  return rsbox[num];
}

void mixColumn(uchar *column)
{
  uchar cpy[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    cpy[i] = column[i];
  }
  column[0] = galoisMultiplication(cpy[0], 2)^
    galoisMultiplication(cpy[3], 1)^
    galoisMultiplication(cpy[2], 1)^
    galoisMultiplication(cpy[1], 3);

  column[1] = galoisMultiplication(cpy[1], 2)^
    galoisMultiplication(cpy[0], 1)^
    galoisMultiplication(cpy[3], 1)^
    galoisMultiplication(cpy[2], 3);

  column[2] = galoisMultiplication(cpy[2], 2)^
    galoisMultiplication(cpy[1], 1)^
    galoisMultiplication(cpy[0], 1)^
    galoisMultiplication(cpy[3], 3);

  column[3] = galoisMultiplication(cpy[3], 2)^
    galoisMultiplication(cpy[2], 1)^
    galoisMultiplication(cpy[1], 1)^
    galoisMultiplication(cpy[0], 3);
}

void mixColumnInv(uchar *column)
{
  uchar cpy[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    cpy[i] = column[i];
  }
  column[0] = galoisMultiplication(cpy[0], 14 )^
    galoisMultiplication(cpy[3], 9 )^
    galoisMultiplication(cpy[2], 13)^
    galoisMultiplication(cpy[1], 11);

  column[1] = galoisMultiplication(cpy[1], 14 )^
    galoisMultiplication(cpy[0], 9 )^
    galoisMultiplication(cpy[3], 13)^
    galoisMultiplication(cpy[2], 11);

  column[2] = galoisMultiplication(cpy[2], 14 )^
    galoisMultiplication(cpy[1], 9 )^
    galoisMultiplication(cpy[0], 13)^
    galoisMultiplication(cpy[3], 11);

  column[3] = galoisMultiplication(cpy[3], 14 )^
    galoisMultiplication(cpy[2], 9 )^
    galoisMultiplication(cpy[1], 13)^
    galoisMultiplication(cpy[0], 11);
}

void mixColumns(uchar * state, bool inverse)
{
  uchar column[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    for(unsigned int j=0; j < 4; ++j)
    {
      column[j] = state[j*4 + i];
    }

    if(inverse)
    {
      mixColumnInv(column);
    }
    else
    {
      mixColumn(column);
    }

    for(unsigned int j=0; j < 4; ++j)
    {
      state[j*4 + i] = column[j];
    }
  }
}

void subBytes(uchar * state, bool inverse, unsigned int keySize)
{
  for(unsigned int i=0; i < keySize; ++i)
  {
    state[i] = inverse ? getSBoxInvert(state[i]): getSBoxValue(state[i]);
  }
}

void shiftRow(uchar *state, uchar nbr)
{
  for(unsigned int i=0; i < nbr; ++i)
  {
    uchar tmp = state[0];
    for(unsigned int j = 0; j < 3; ++j)
    {
      state[j] = state[j+1];
    }
    state[3] = tmp;
  }
}

void shiftRowInv(uchar *state, uchar nbr)
{
  for(unsigned int i=0; i < nbr; ++i)
  {
    uchar tmp = state[3];
    for(unsigned int j = 3; j > 0; --j)
    {
      state[j] = state[j-1];
    }
    state[0] = tmp;
  }
}

__host__
void shiftRows(uchar * state, bool inverse)
{
  for(unsigned int i=0; i < 4; ++i)
  {
    if(inverse)
      shiftRowInv(state + i*4, i);
    else
      shiftRow(state + i*4, i);
  }
}

void addRoundKey(uchar * state,
    uchar * rKey,
    unsigned int keySize)
{
  for(unsigned int i=0; i < keySize; ++i)
  {
    state[i] = state[i] ^ rKey[i];
  }
}


void aesRound(uchar * state, uchar * rKey, 
    bool decrypt, unsigned int keySize)
{
  subBytes(state, decrypt, keySize);
  shiftRows(state, decrypt);
  mixColumns(state, decrypt);
  addRoundKey(state, rKey, keySize);
}

void aesMain(uchar * state, uchar * rKey, unsigned int rounds, 
             bool decrypt, unsigned int keySize)
{
  addRoundKey(state, rKey, keySize);
  for(unsigned int i=1; i < rounds; ++i)
  {
    aesRound(state, rKey + keySize*i, decrypt, keySize);
  } 
  subBytes(state, decrypt, keySize);
  shiftRows(state, decrypt);
  addRoundKey(state, rKey + keySize*rounds, keySize);
}

void aesRoundInv(uchar * state, uchar * rKey, 
                 bool decrypt, unsigned int keySize)
{
  shiftRows(state, decrypt);
  subBytes(state, decrypt, keySize);
  addRoundKey(state, rKey, keySize);
  mixColumns(state, decrypt);
}

void aesMainInv(uchar * state, uchar * rKey, unsigned int rounds,
                  bool decrypt, unsigned int keySize)
{
  addRoundKey(state, rKey + keySize*rounds, keySize);
  for(unsigned int i=rounds-1; i > 0; --i)
  {
    aesRoundInv(state, rKey + keySize*i, decrypt, keySize);
  } 
  shiftRows(state, decrypt);
  subBytes(state, decrypt, keySize);
  addRoundKey(state, rKey, keySize);
}

/**
 *
 *
 */
void reference(uchar * output       ,
               uchar * input        ,
               uchar * rKey         ,
               unsigned int explandedKeySize,
               unsigned int width           ,
               unsigned int height          ,
               bool inverse,
               unsigned int rounds          ,
               unsigned int keySize         )
{
  uchar block[16];

  for(unsigned int blocky = 0; blocky < height/4; ++blocky)
    for(unsigned int blockx= 0; blockx < width/4; ++blockx)
    { 
      for(unsigned int i=0; i < 4; ++i)
      {
        for(unsigned int j=0; j < 4; ++j)
        {
          unsigned int index  = (((blocky * width/4) + blockx) * keySize )+ (i*4 + j);
          block[i*4 + j] = input[index];
        }
      }

      if(inverse)
        aesMainInv(block, rKey, rounds, inverse, keySize);
      else
        aesMain(block, rKey, rounds, inverse, keySize);

      for(unsigned int i=0; i <4 ; ++i)
      {
        for(unsigned int j=0; j <4; ++j)
        {
          unsigned int index  = (((blocky * width/4) + blockx) * keySize )+ (i*4 + j);
          output[index] = block[i*4 + j];
        } 
      }
    }
}



// ---- END INLINED: reference.cu ----

// ---- INLINED: utils.cu (from /home/WillFu/parallel/final/HeCBench/src/aes-cuda/utils.cu) ----
//
// utilities called by functions in main.cpp
//
void convertColorToGray(const uchar4 *pixels, 
                        uchar *gray,
                        const int height,
                        const int width)
{
  for(int i=0; i< height; ++i)
    for(int j=0; j<width; ++j)
    {
      unsigned int index = i*width + j;
      // gray = (0.3*R + 0.59*G + 0.11*B)
      gray[index] = (uchar) (pixels[index].x * 0.3f  + 
                             pixels[index].y * 0.59f + 
                             pixels[index].z * 0.11f );
    }
}

void convertGrayToGray(const uchar4 *pixels, 
                       uchar *gray,
                       const int height,
                       const int width)
{
  for(int i=0; i< height; ++i)
    for(int j=0; j<width; ++j)
    {
      unsigned int index = i*width + j;
      gray[index] = pixels[index].x;
    }
}

void createRoundKey(uchar * eKey, uchar * rKey)
{
  for(unsigned int i=0; i < 4; ++i)
    for(unsigned int j=0; j < 4; ++j)
    {
      rKey[i+ j*4] = eKey[i*4 + j];
    }
}

void rotate(uchar * word)
{
  uchar c = word[0];
  for(unsigned int i=0; i<3; ++i)
    word[i] = word[i+1];
  word[3] = c;
}

void core(uchar * word, unsigned int iter)
{
  rotate(word);

  for(unsigned int i=0; i < 4; ++i)
  {
    word[i] = getSBoxValue(word[i]);
  }    

  word[0] = word[0]^getRconValue(iter);
}


void keyExpansion(uchar * key, uchar * expandedKey,
                  unsigned int keySize, unsigned int explandedKeySize)
{
  unsigned int currentSize   = 0;
  unsigned int rConIteration = 1;
  uchar temp[4] = {0};

  for(unsigned int i=0; i < keySize; ++i)
  {
    expandedKey[i] = key[i];
  }

  currentSize += keySize;

  while(currentSize < explandedKeySize)
  {
    for(unsigned int i=0; i < 4; ++i)
    {
      temp[i] = expandedKey[(currentSize - 4) + i];
    }

    if(currentSize%keySize == 0)
    {
      core(temp, rConIteration++);
    }

    //XXX: add extra SBOX here if the keySize is 32 Bytes

    for(unsigned int i=0; i < 4; ++i)
    {
      expandedKey[currentSize] = expandedKey[currentSize - keySize]^temp[i];
      currentSize++;
    }
  }
}

/**
 * fill array with random values
 */
template<typename T>
int fillRandom(
    T * arrayPtr,
    const int width,
    const int height,
    const T rangeMin,
    const T rangeMax,
    unsigned int seed=123)
{
  if(!arrayPtr)
  {
    std::cerr << "Cannot fill array. NULL pointer.\n";
    return 1;
  }
  if(!seed)
  {
    seed = (unsigned int)time(NULL);
  }
  srand(seed);
  double range = double(rangeMax - rangeMin) + 1.0;
  /* random initialisation of input */
  for(int i = 0; i < height; i++)
    for(int j = 0; j < width; j++)
    {
      int index = i*width + j;
      arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
    }
  return 0;
}

// ---- END INLINED: utils.cu ----


int main(int argc, char * argv[])
{
  if (argc != 4) {
    printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    printf("0=encrypt, 1=decrypt\n");
    return 1;
  }

  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;

  const int iterations = atoi(argv[1]);
  const bool decrypt = atoi(argv[2]);
  const char* filePath = argv[3];

  SDKBitMap image;
  image.load(filePath);
  const int width  = image.getWidth();
  const int height = image.getHeight();

  /* check condition for the bitmap to be initialized */
  if (width <= 0 || height <= 0) return 1;

  std::cout << "Image width and height: " 
            << width << " " << height << std::endl;

  uchar4 *pixels = image.getPixels();

  unsigned int sizeBytes = width*height*sizeof(uchar);
  uchar *input = (uchar*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);

  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits

  unsigned int keySizeBytes = keySize*sizeof(uchar);

  uchar *key = (uchar*)malloc(keySizeBytes);

  fillRandom<uchar>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;

  unsigned int explandedKeySizeBytes = explandedKeySize*sizeof(uchar);

  uchar *expandedKey = (uchar*)malloc(explandedKeySizeBytes);
  uchar *roundKey    = (uchar*)malloc(explandedKeySizeBytes);

  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i)
  {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  uchar* output = (uchar*)malloc(sizeBytes);

  uchar *inputBuffer;
  cudaMalloc((void**)&inputBuffer, sizeBytes);
  cudaMemcpy(inputBuffer, input, sizeBytes, cudaMemcpyHostToDevice);

  uchar *outputBuffer;
  cudaMalloc((void**)&outputBuffer, sizeBytes);

  uchar *rKeyBuffer;
  cudaMalloc((void**)&rKeyBuffer, explandedKeySizeBytes);
  cudaMemcpy(rKeyBuffer, roundKey, explandedKeySizeBytes, cudaMemcpyHostToDevice);

  uchar *sBoxBuffer;
  cudaMalloc((void**)&sBoxBuffer, 256);
  cudaMemcpy(sBoxBuffer, sbox, 256, cudaMemcpyHostToDevice);

  uchar *rsBoxBuffer;
  cudaMalloc((void**)&rsBoxBuffer, 256);
  cudaMemcpy(rsBoxBuffer, rsbox, 256, cudaMemcpyHostToDevice);

  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  dim3 grid (width/4, height/4);
  dim3 block (1, 4);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    if (decrypt) 
      AESDecrypt<<< grid, block >>>(
        (uchar4*)outputBuffer,
        (uchar4*)inputBuffer,
        (uchar4*)rKeyBuffer,
        rsBoxBuffer,
        width, rounds);
    else
      AESEncrypt<<< grid, block >>>(
        (uchar4*)outputBuffer,
        (uchar4*)inputBuffer,
        (uchar4*)rKeyBuffer,
        sBoxBuffer,
        width, rounds);

  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  cudaMemcpy(output, outputBuffer, width * height, cudaMemcpyDeviceToHost);

  // Verify
  uchar *verificationOutput = (uchar *) malloc(sizeBytes);

  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);

  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, sizeBytes) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  /* release program resources (input memory etc.) */
  cudaFree(inputBuffer);
  cudaFree(outputBuffer);
  cudaFree(rKeyBuffer);
  cudaFree(sBoxBuffer);
  cudaFree(rsBoxBuffer);

  if(input) free(input);

  if(key) free(key);

  if(expandedKey) free(expandedKey);

  if(roundKey) free(roundKey);

  if(output) free(output);

  if(verificationOutput) free(verificationOutput);

  return 0;
}

