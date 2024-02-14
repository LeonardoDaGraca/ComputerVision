/*
Leo DaGraca
CS5330
1/31/2024
This program's intention is to store the different functions
that will compute the features on an image.
*/

//Libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"

using namespace std;
using namespace cv;


//calculates the sum of squared differences between two vectors
float SSD(std::vector<float> &vect1, std::vector<float> &vect2){
  float sum = 0;
  for(size_t i = 0; i < vect1.size(); i++){
    float diff = vect1[i] - vect2[i];
    sum += diff * diff;
  }
  return sum;
}

//distance metric for two histogram vectors
float histIntersection(std::vector<float> &histA, std::vector<float> &histB){
  float intersection = 0;
  for(size_t i = 0; i < histA.size(); i++){
    intersection += std::min(histA[i], histB[i]);
  }
  return intersection;
}


/*
Baseline Matching:
Use the 7x7 square in the middle of the image as a feature vector. 
Use sum-of-squared-difference as the distance metric. 
Make sure that comparing the image with itself results in a distance of 0.

Parameter: takes in a src mat image

1 2 4 2 1 0 1
2 4 8 4 2 0 1
4 8 1 8 4 0 1
2 4 8 X 2 0 1
1 2 4 2 1 0 1
2 4 8 4 2 0 1
1 2 4 2 1 0 1

*/

std::vector<float> baselineMatch(cv::Mat &src){
  //find the center point of the image
  int midPtWidth = src.cols / 2;
  int midPtHeight = src.rows / 2;

  //vector to hold the extracted feature
  std::vector<float> feature;

  //access middle pixel using 7x7 square -> (3, 3)
  for(int i = midPtHeight - 3; i <= midPtHeight + 3; i++){
    for(int j = midPtWidth - 3; j <= midPtWidth + 3; j++){
      //obtain pixel values
      cv::Vec3b values = src.at<cv::Vec3b>(i, j);
      for (int k = 0; k < src.channels(); k++){
        //add pixel values to the vector
        feature.push_back(values[k]); //adding all color channel
      }
    }
  }
  return feature;
}

/*
Histogram matching:
Paramater: takes in a src mat image

Creates a 3D color histogram from src image and 
places data from histogram to a feature vector
*/

std::vector<float> histMatch(cv::Mat &src){
  const int histsize = 8; //bins for hist

  int histArr[] = {histsize, histsize, histsize};

  //init 3D histogram
  cv::Mat hist;
  hist = cv::Mat::zeros(3, histArr, CV_32FC1);
  
  //vector to hold the extracted feature
  std::vector<float> feature;

  for(int i = 0; i < src.rows; i++){
    cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
    for(int j = 0; j < src.cols; j++){
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      float divisor = R + G + B;
      divisor = divisor > 0.0 ? divisor : 1.0;
      float r = R / divisor;
      float g = G / divisor;
      float b = B / divisor;

      //compute indexes
      int rindex = (int)(r * (histsize - 1) + 0.5);
      int gindex = (int)(g * (histsize - 1) + 0.5);
      int bindex = (int)(b * (histsize - 1) + 0.5);

      //increment histogram
      hist.at<float>(rindex, gindex, bindex)++;
    }
  }
  //normalize the histogram by the number of pixels
  hist /= (src.rows * src.cols);

  //convert histogram to a feature vector
  feature.reserve(histsize * histsize * histsize); //declare space for 1D array
  for(int i = 0; i < histsize; i++){
    for(int j = 0; j < histsize; j++){
      for(int k = 0; k < histsize; k++){
      feature.push_back(hist.at<float>(i, j, k));
      }
    }
  }

  return feature;
}

/*
Center Histogram matching:
Paramater: takes in a src mat image

Creates a 2D color histogram from the center src image as a feature vector.
*/

std::vector<float> multiHistMatch(cv::Mat &src){
  //divide src image into bottom and top half
  cv::Mat top = src(cv::Rect(0, 0, src.cols, src.rows / 2)); //region for top
  cv::Mat bottom = src(cv::Rect(0, src.rows / 2, src.cols, src.rows / 2)); //region for bottom

  //histograms for both halfs
  std::vector<float>topHist = histMatch(top);
  std::vector<float>bottomHist = histMatch(bottom);

  //combine histograms into a feature vector
  std::vector<float> feature;
  feature.insert(feature.end(), topHist.begin(), topHist.end());
  feature.insert(feature.end(), bottomHist.begin(), bottomHist.end());

  return feature;
}


/*
Texture and Color histogram match
Paramater: takes in a src mat image

Combines a whole image histogram feature vector and
a histogram of gradient magnitudes into one feature vector.
*/
std::vector<float> txtColorHistMatch(cv::Mat &src){
  //vector for color histogram
  std::vector<float> colorHist = histMatch(src);

  //texture histogram
  cv::Mat sx, sy, mag;
  sobelX3x3(src, sx);
  sobelY3x3(src, sy);
  magnitude(sx, sy, mag);
  std::vector<float> magHist = histMatch(mag); //vector for texture histogram

  //combine histograms
  std::vector<float> feature;
  feature.insert(feature.end(), colorHist.begin(), colorHist.end());
  feature.insert(feature.end(), magHist.begin(), magHist.end());

  return feature;
}


/*
Local Binary Pattern Match
Paramater: takes in a src mat image

Calculates LBP from grayscale image using LBP algorithm.
Then creates a histogram with those LBP values which is used to
be turned into a feature vector.
*/
std::vector<float> lbpMatch(cv::Mat &src){
  //convert image to grayscale
  Mat grey;
  cv::cvtColor(src, grey, COLOR_BGR2GRAY);
  //reduce image size to deal with borders
  Mat lbpImg = cv::Mat::zeros(grey.rows - 2, grey.cols - 2, CV_8UC1);

  //calculate LBP
  for(int i = 1; i < grey.rows - 1; i++){//ignore border pixels
    for(int j = 1; j < grey.cols - 1; j++){
      unsigned char center = grey.at<unsigned char>(i, j); //get center pixel
      unsigned char code = 0; //store binary pattern
      //compare center pixel with 8 neighbors
      code |= (grey.at<unsigned char>(i - 1, j - 1) >= center) << 7;
      code |= (grey.at<unsigned char>(i - 1, j) >= center) << 6;
      code |= (grey.at<unsigned char>(i - 1, j + 1) >= center) << 5;
      code |= (grey.at<unsigned char>(i, j + 1) >= center) << 4;
      code |= (grey.at<unsigned char>(i + 1, j + 1) >= center) << 3;
      code |= (grey.at<unsigned char>(i + 1, j) >= center) << 2;
      code |= (grey.at<unsigned char>(i + 1, j - 1) >= center) << 1;
      code |= (grey.at<unsigned char>(i, j - 1) >= center) << 0;

      lbpImg.at<unsigned char>(i - 1, j - 1) = code; //assign bit values to pixel
    }
  }

  //building histogram
  int histsize = 8;
  float range[] = {0, 256};
  const float* histrange[] = {range};
  Mat hist;

  cv::calcHist(&lbpImg, 1, 0, Mat(), hist, 1, &histsize, histrange, true, false);
  //normalize hist
  hist /= (lbpImg.rows * lbpImg.cols);

  //convert hist to feature vector
  std::vector<float> feature;
  feature.reserve(histsize);
  for(int i = 0; i < histsize; i++){
    feature.push_back(hist.at<float>(i));
  }

  return feature;
}

/*
Histograms of Gabor filter responses Match
Paramater: takes in a src mat image

Calculates Gabor filter responses from grayscale image.
Then creates a histogram with those values which is used to
be turned into a feature vector.
*/
std::vector<float>gaborMatch(cv::Mat &src){
  //convert image to greyscale
  Mat grey;
  cv::cvtColor(src, grey, COLOR_BGR2GRAY);

  //parameters for gabor filter
  int kernelSize = 31;
  float sigma = 4.0;
  float theta = 0;
  float lambda = 10.0;
  float gamma = 0.5;
  float psi = CV_PI * 0.5; //phase offset
  int nFilters = 8; //num of filters for different orientations

  std::vector<Mat> gaborResponses;
  //apply gabor filter for different orientations
  for(int i = 0; i < nFilters; i++){
    theta = i * CV_PI / nFilters;
    Mat kernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, theta, lambda, gamma, psi, CV_32F);
    Mat response;
    filter2D(grey, response, CV_32F, kernel);
    gaborResponses.push_back(response);
  }

  //for each gabor response calculate the hist and build feature vector
  std::vector<float> feature;
  for(auto& response : gaborResponses){
    //calculate histogram
    int histsize = 8;
    float range[] = {0, 256}; 
    const float* histrange = {range};
    Mat hist;
    calcHist(&response, 1, 0, Mat(), hist, 1, &histsize, &histrange, true, false);

    //normalize hist
    hist /= (response.rows * response.cols);

    //append response to feature vector
    for(int i = 0; i < histsize; i++){
      feature.push_back(hist.at<float>(i));
    }
  }
  return feature;
}