/*
Leo DaGraca
CS5330
1/17/2024
This file will store all image manipulation functions
*/

//libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


using namespace std;
using namespace cv;

/*
Alternative greyscale live video function
Parameters: Takes in a src Mat object (video) and the output will be a greyscale version of the input
Returns 0 on success
*/
int greyscale(cv::Mat &src, cv::Mat &dst){
  //ensure that the src image is 8-bit unsigned integer with 3 channels
  CV_Assert(src.type() == CV_8UC3);
  //create dst image of same size of src image with 1 channel
  dst.create(src.size(), CV_8UC1);

  //access pixels within images and modify them
  //Pixels are as BGR values in memory; ex: pixel(0,0) -> [200, 200, 200]
  for(int i = 0; i < src.rows; i++){
    //pointers to pixel
    uchar* srcPtr = src.ptr<uchar>(i);
    uchar* dstPtr = dst.ptr<uchar>(i);

    //access cols and change col[j] value to represent greyscale
    for(int j = 0; j < src.cols; j++){
      uchar blueChange = 255 - srcPtr[0]; //subtracting blue value from 255

      dstPtr[j] = blueChange; //applying blue change to B
      dstPtr[j + 1] = blueChange; //applying blue change to G
      dstPtr[j + 2] = blueChange; //applying blue change to R
      
      //move to the next group of channels
      srcPtr += 3;
    }
  }
  return 0;
}

/*
Sepia tone live video filter function
Parameters: Takes in a src Mat object (video) and the output will be a sepia version of the input
Returns 0 on success
*/
int sepia(cv::Mat &src, cv::Mat &dst){
  CV_Assert(src.type() == CV_8UC3);
  dst.create(src.size(), src.type());

  for(int i = 0; i < src.rows; i++){
    uchar* srcPtr = src.ptr<uchar>(i); //stores a ptr to each row in memory
    uchar* dstPtr = dst.ptr<uchar>(i);

    for(int j = 0; j < src.cols; j++){
      
      //access each color channel
      float blue = srcPtr[3 * j]; //mult by 3 at each col to get exact index for color channel
      float green = srcPtr[3 * j + 1];//move by 1 to access green channel
      float red = srcPtr[3 * j + 2]; //move by 2 to access red channel

      //apply sepia coeffecient change to get new color values
      float newRed = (0.393 * red) + (0.769 * green) + (0.189 * blue);
      float newGreen = (0.349 * red) + (0.686 * green) + (0.168 * blue);
      float newBlue = (0.272 * red) + (0.534 * green) + (0.131 * blue);

      //keep the values of the pixels within the range of 0 - 255 using the saturate_cast method
      newRed = saturate_cast<uchar>(newRed);
      newGreen = saturate_cast<uchar>(newGreen);
      newBlue = saturate_cast<uchar>(newBlue);

      //apply sepia color change to the respective color channel in dst image
      dstPtr[3 * j] = newBlue;
      dstPtr[3 * j + 1] = newGreen;
      dstPtr[3 * j + 2] = newRed;
    }
  }
  return 0;
}

/*
5 X 5 blur filter function (naive approach)
Parameters: Takes in a src Mat object (video) and the output will be a blurred version of the input
Returns 0 on success

1 2 4 2 1 
2 4 8 4 2 
4 8 16 8 4 
2 4 8 4 2 
1 2 4 2 1
*/
int blur5x5_1(cv::Mat &src, cv::Mat &dst){
  src.copyTo(dst); //allocate dst Mat object

  //start at center pixel for 5x5 -> (2,2)
  for(int i = 2; i < src.rows - 2; i++){
    for(int j = 2; j < src.cols - 2; j++){
      for(int k = 0; k < src.channels(); k++){
        int sum = src.at<cv::Vec3b>(i - 2, j - 2)[k] + 2 * src.at<cv::Vec3b>(i - 2, j - 1)[k] + 4 * src.at<cv::Vec3b>(i - 2, j)[k] 
                  + 2 * src.at<cv::Vec3b>(i - 2, j + 1)[k] + src.at<cv::Vec3b>(i - 2, j + 2)[k]  //row 1
                  + 2 * src.at<cv::Vec3b>(i - 1, j - 2)[k] + 4 * src.at<cv::Vec3b>(i - 1, j - 1)[k] + 8 * src.at<cv::Vec3b>(i - 1, j)[k]
                  + 4 * src.at<cv::Vec3b>(i - 1, j + 1)[k] + 2 * src.at<cv::Vec3b>(i - 1, j + 2)[k] //row 2
                  + 4 * src.at<cv::Vec3b>(i, j - 2)[k] + 8 * src.at<cv::Vec3b>(i, j - 1)[k] + 16 * src.at<cv::Vec3b>(i, j)[k]
                  + 8 * src.at<cv::Vec3b>(i, j + 1)[k] + 4 * src.at<cv::Vec3b>(i, j + 2)[k] //row 3
                  + 2 * src.at<cv::Vec3b>(i + 1, j - 2)[k] + 4 * src.at<cv::Vec3b>(i + 1, j - 1)[k] + 8 * src.at<cv::Vec3b>(i + 1, j)[k]
                  + 4 * src.at<cv::Vec3b>(i + 1, j + 1)[k] + 2 * src.at<cv::Vec3b>(i + 1, j + 2)[k] //row 4
                  + src.at<cv::Vec3b>(i + 2, j - 2)[k] + 2 * src.at<cv::Vec3b>(i + 2, j - 1)[k] + 4 * src.at<cv::Vec3b>(i + 2, j)[k]
                  + 2 * src.at<cv::Vec3b>(i + 2, j + 1)[k] + src.at<cv::Vec3b>(i + 2, j + 2)[k];
        //normalize back to range of [0, 255]
        sum /= 100;
        dst.at<cv::Vec3b>(i, j)[k] = sum;
      }
    }
  }
  return 0;
}

/*
5 X 5 blur filter function (improved approach)
Parameters: Takes in a src Mat object (video) and the output will be a blurred version of the input
Returns 0 on success

1 2 4 2 1 
2 4 8 4 2 
4 8 16 8 4 
2 4 8 4 2 
1 2 4 2 1
*/
int blur5x5_2(cv::Mat &src, cv::Mat &dst){
  src.copyTo(dst); //allocate dst Mat object
  //store results after first pass
  Mat hResults = src.clone();

  //1 x 5 horizontal pass
  for(int i = 0; i < src.cols; i++){
    cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *hResultsPtr = hResults.ptr<cv::Vec3b>(i);
    for(int j = 2; j < src.cols - 2; j++){
      for(int k = 0; k < src.channels(); k++){
        int sum = srcptr[j - 2][k] + 2 * srcptr[j - 1][k] + 4 * srcptr[j][k] + 2 * srcptr[j + 1][k] + srcptr[j + 2][k];
        sum /= 10;
        //store horizontal results
        hResultsPtr[j][k] = sum;
      }
    }
  }
  
  //1 x 5 vertical pass
  for(int j = 2; j < src.cols - 2; j++){ //start by accessing the cols
    for(int i = 2; i < src.rows - 2; i++){
      cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);
      for(int k = 0; k < src.channels(); k++){
        int sum = hResults.ptr<cv::Vec3b>(i - 2)[j][k] + 2 * hResults.ptr<cv::Vec3b>(i - 1)[j][k] + 4 * hResults.ptr<cv::Vec3b>(i)[j][k] + 2 * hResults.ptr<cv::Vec3b>(i + 1)[j][k] + hResults.ptr<cv::Vec3b>(i + 2)[j][k];
        sum /= 10;
        dstptr[j][k] = sum;
      }
    }
  }
  return 0;
}

/*
3 X 3 Sobel filter function (vertical)
Parameters: Takes in a src Mat object (video) and the output will be a sobel version of the input
Returns 0 on success

Y - direction:
-1 0 1
-2 0 2
-1 0 1
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst){
  //set dst image as the same size of src and type 16SC3 
  dst.create(src.size(), CV_16SC3);

  //start at center pixel(1, 1)
  //skip outer row and cols
  for (int i = 1; i < src.rows - 1; i++){
    cv::Vec3b *ptrup = src.ptr<cv::Vec3b>(i - 1); //pointer to row i - 1
    cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i); //pointer to row i
    cv::Vec3b *ptrdown = src.ptr<cv::Vec3b>(i + 1); //pointer to row i + 1
    cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);
    for(int j = 1; j < src.cols - 1; j++){
      for(int k = 0; k < src.channels(); k++){
        int sum = -1 * ptrup[j - 1][k] + 0 * ptrup[j][k] + ptrup[j + 1][k] 
                  + -2 * ptrmd[j - 1][k] + 0 * ptrmd[j][k] + 2 * ptrmd[j + 1][k]
                  + -1 * ptrdown[j - 1][k] + 0 * ptrdown[j][k] + ptrdown[j + 1][k];
        
        dstptr[j][k] = static_cast<short>(sum);
      }
    }
  }
  return 0;
}


/*
3 X 3 Sobel filter function (horizontal)
Parameters: Takes in a src Mat object (video) and the output will be a sobel version of the input
Returns 0 on success

X - direction:
-1 -2 -1
 0  0  0
 1  2  1
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
  //set dst image as the same size of src and type 16SC3 
  dst.create(src.size(), CV_16SC3);

  //start at center pixel(1, 1)
  //skip outer row and cols
  for (int i = 1; i < src.rows - 1; i++){
    cv::Vec3b *ptrup = src.ptr<cv::Vec3b>(i - 1); //pointer to row i - 1
    cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i); //pointer to row i
    cv::Vec3b *ptrdown = src.ptr<cv::Vec3b>(i + 1); //pointer to row i + 1
    cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);
    for(int j = 1; j < src.cols - 1; j++){
      for(int k = 0; k < src.channels(); k++){
        int sum = -1 * ptrup[j - 1][k] + -2 * ptrup[j][k] + -1 * ptrup[j + 1][k] 
                  + 0 * ptrmd[j - 1][k] + 0 * ptrmd[j][k] + 0 * ptrmd[j + 1][k]
                  + ptrdown[j - 1][k] + 2 * ptrdown[j][k] + ptrdown[j + 1][k];
        
        dstptr[j][k] = static_cast<short>(sum);
      }
    }
  }
  return 0;
}

/*
Gradient magnitude image from X and Y Sobel images function
Parameters: Takes in a X and Y sobel image along with a dst that will be transformed
Returns 0 on success

Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
  //set the input images as 3-channel signed short images
  CV_Assert(sx.type() == CV_16SC3);
  CV_Assert(sy.type() == CV_16SC3);
  //create the output image as a uchar suitable for display
  dst.create(sx.size(), CV_8UC3);
  

  //find the actual maximum magnitude
  float maxMagnitude = 0;
  for (int i = 0; i < sx.rows; i++) {
      for(int j = 0; j < sx.cols; j++) {
          for(int k = 0; k < sx.channels(); k++) {
              float magnitude = sqrt(sx.at<cv::Vec3s>(i, j)[k] * sx.at<cv::Vec3s>(i, j)[k] + 
                                    sy.at<cv::Vec3s>(i, j)[k] * sy.at<cv::Vec3s>(i, j)[k]);
              if (magnitude > maxMagnitude) {
                  maxMagnitude = magnitude;
              }
          }
      }
  }
  //access pixels
  for (int i = 0; i < sx.rows; i++){
    cv::Vec3s *sxPtr = sx.ptr<cv::Vec3s>(i);//row for the sx image
    cv::Vec3s *syPtr = sy.ptr<cv::Vec3s>(i);//row for the sy image
    cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);
    for(int j = 0; j < sy.cols; j++){
      for(int k = 0; k < sx.channels(); k++){
        //calculate magnitude using euclidean formula
        float magnitude = sqrt(sxPtr[j][k] * sxPtr[j][k] + syPtr[j][k] * syPtr[j][k]);
        //clamping values
        magnitude = (magnitude / maxMagnitude) * 255.0f;
        dstptr[j][k] = static_cast<uchar>(magnitude);
      }
    }
  }
  return 0;
}

/*
Function that blurs and quantizes a color image
Parameters: Takes in a src Mat object (video) and the output will be a blurred version of the input
            The levels determines how many levels the image will be quantized
Returns 0 on success
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels){
  //first blur the image
  blur5x5_2(src, dst);

  //initialize bucket size
  int b = 255 / levels;
  //access blurred image pixels
  for(int i = 0; i < dst.rows; i++){
    cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);//access row i
    for(int j = 0; j < dst.cols; j++){
      for(int k = 0; k < dst.channels(); k++){
        int preQuant = (dstPtr[j][k]) / b; //get value at channel prior to quantizing
        int postQuant = static_cast<int>(preQuant) * b; //calculate quant value

        dstPtr[j][k] = static_cast<uchar>(postQuant); //apply quant value to the color channel
      }
    }
  }
  return 0;
}

/*
Brightness or contrast adjuster filter function
Parameters: Takes in a src Mat object (video) and the output will be either an adjusted brightness/contrast version of the input
The user will be prompted to enter alpha and beta values that control the contrast and brightness of the dst image
Returns 0 on success

f(x) - input image
g(x) - output image
α - gain (contrast control)
β - bias (brightness control)

Gain and bias parameters are what control the contrast and brightness
g(i,j)= α⋅f(i,j)+β
*/
int brightness_contrast(cv::Mat &src, cv::Mat &dst, float alpha, int beta){
  src.copyTo(dst); //allocate dst mat object
  //access pixels
  for(int i = 0; i < src.rows; i++){
    for(int j = 0; j < src.cols; j++){
      for(int k = 0; k < src.channels(); k++){
        //calculate brightness and contrast control
        dst.at<cv::Vec3b>(i, j)[k] = saturate_cast<uchar>(alpha * src.at<cv::Vec3b>(i, j)[k] + beta);
      }
    }
  }
  return 0;
}

/*
Embossing filter from X and Y Sobel images function
Parameters: Takes in a X and Y sobel image along with a dst that will be transformed
Returns 0 on success

*/
int emboss(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
  //set the input images as 3-channel signed short images
  CV_Assert(sx.type() == CV_16SC3);
  CV_Assert(sy.type() == CV_16SC3);
  //create the output image as a uchar suitable for display
  dst.create(sx.size(), CV_8UC3);

  //directions
  float xDir = 0.7071;
  float yDir = 0.7071;

  //access pixels
  for(int i = 0; i < sx.rows; i++){
    cv::Vec3s *sxPtr = sx.ptr<cv::Vec3s>(i);//row for the sx image
    cv::Vec3s *syPtr = sy.ptr<cv::Vec3s>(i);//row for the sy image
    cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);
    for(int j = 0; j < sx.cols; j++){
      for(int k = 0; k < sx.channels(); k++){
        //calculate dot product with the x and y directions
        int dotProduct = static_cast<int>(sxPtr[j][k] * xDir + syPtr[j][k] * yDir);
        //Add 128 to the dot product to set baseline color to mid-grey for embossing effect
        dstptr[j][k] = saturate_cast<uchar>(dotProduct + 128);
      }
    }
  }
  return 0;
}

/*
Blur filter that blurs everything but the face
Parameters: Takes in an existing Mat frame and a standard vector of cv::Rect rectangles indicating where faces were found
Returns 0 on success

Function uses a mask to store the face, then applies the blur on the entire image, 
and finally copies the face back to the blurred image.
*/
int excludeFaceBlur(cv::Mat &frame, std::vector<cv::Rect> &faces){
  //mask to cover up face and prevent it from being blurred
  Mat mask = cv::Mat::zeros(frame.size(), frame.type()); //set mask to zeros and same size/type as src

  cv::Scalar wcolor(255, 255, 255); //set color to white

  //set variables
  int minWidth = 50;
  float scale = 1.0;

  for(int i = 0;i < faces.size(); i++){
    if(faces[i].width > minWidth) {
      cv::Rect face(faces[i]);
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle(mask, face, wcolor, cv::FILLED); //change thickness so the rect is filled
    }
  }
  //blur entire frame
  Mat blurredFrame;
  blur5x5_2(frame, blurredFrame); //apply blur

  //copy just the face from the original frame to the blurred frame
  frame.copyTo(blurredFrame, mask);
  //set the original frame to the updated blurred image with face showing
  frame = blurredFrame;

  return 0;

}