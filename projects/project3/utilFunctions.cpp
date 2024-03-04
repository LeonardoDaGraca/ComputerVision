/*
Leo DaGraca
CS5330
2/14/2024
Utility functions for 2D object recognition
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "filter.h"

using namespace std;
using namespace cv;


/*
Greyscale live video function
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
      uchar blue = srcPtr[j * 3];
      uchar green = srcPtr[j * 3 + 1];
      uchar red = srcPtr[j * 3 + 2];

      //average method
      uchar grey = (blue + green + red) / 3;

      //assign greyscale
      dstPtr[j] = grey;
    }
  }
  return 0;
}


/*
Threshold Function
Parameters: source Mat image, destination Mat image, threshold val, max value

The function converts the src image to greyscale and then uses the greyscale
image to set the threshold values onto the dst image
*/
void threshold(cv::Mat &src, cv::Mat &dst, int thresh, int maxval){
  cv::Mat grey;
  greyscale(src, grey);
  //cv::cvtColor(src, grey, COLOR_BGR2GRAY); //convert src to grey

  dst = cv::Mat(grey.size(), grey.type()); //set dst image to same size, type as grey

  for(int i = 0; i < grey.rows; i++){
    uchar *greyptr = grey.ptr<uchar>(i);
    uchar *dstptr = dst.ptr<uchar>(i);
    for(int j = 0; j < grey.cols; j++){
      dstptr[j] = (uchar)(greyptr[j] > thresh ? 0 : maxval);
    }
  }
}


/*
Dilation Cleanup Function
Parameters: source Mat image, destination Mat image, int kernel size

This function cleans up a thresholded image using dilation.
*/
cv::Mat customDilate(cv::Mat &src, cv::Mat &dst, int kernelSize){
  //init kernel
  cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8U);
  //prep dst image
  dst = cv::Mat::zeros(src.size(), src.type());

  //padding size
  int pad = kernelSize / 2;

  //iterate through src image pixels
  for(int i = 0; i < src.rows; i++){
    for(int j = 0; j < src.cols; j++){

      //apply kernel
      for(int k = -pad; k <= pad; k++){
        for(int z = -pad; z <= pad; z++){
          int row = i + k; //neighbor row index
          int col = j + z; // neighbor col index

          //check bounds
          if(k >= 0 && row < src.rows && col >= 0 && col < src.cols){
            if(src.at<uchar>(row, col) == 255){
              dst.at<uchar>(i, j) = 255;
              goto setDone; //exit kernel loop if foreground pixel is found
            }
          }
        }
      }
      setDone: continue;
    }
  }
  //return output
  return dst;
}

/*
Compute Features Function
Paramaters: region map, region id, output mat for display

The function computes features for each major region.
It displays the oriented bounding box around the region,
the axis of least central moment, the Percent filled,
and the bounding box height/width ratio.
*/
void computeFeat(cv::Mat &regionMap, int regionId, cv::Mat &output){
  //calculate moments
  cv::Moments m = moments(regionMap == regionId);

  //points
  cv::Point2f mean(m.m10 / m.m00, m.m01 / m.m00); //mean point (ux, uy)

  //axis of least central moment
  float theta = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);
  //eigenvectors
  cv::Point2f e1(cos(theta), sin(theta));
  cv::Point2f e2(-sin(theta), cos(theta));

  //init min/max projections on eigenvectors
  float minProjE1 = FLT_MAX;
  float maxProjE1 = -FLT_MAX;
  float minProjE2 = FLT_MAX;
  float maxProjE2 = -FLT_MAX;

  //iterate through region map to find projections
  for(int y = 0; y < regionMap.rows; y++){
    for(int x = 0; x < regionMap.cols; x++){
      if(regionMap.at<int>(y, x) == regionId){
        cv::Point2f point(x, y);
        float projE1 = (point.x - mean.x) * e1.x + (point.y - mean.y) * e1.y; //projection on e1
        float projE2 = (point.x - mean.x) * e2.x + (point.y - mean.y) * e2.y; //projection on e2

        //update projections
        minProjE1 = std::min(minProjE1, projE1);
        maxProjE1 = std::max(maxProjE1, projE1);
        minProjE2 = std::min(minProjE2, projE2);
        maxProjE2 = std::max(maxProjE2, projE2);
      }
    }
  }
  //calculate corners
  std::vector<cv::Point2f> corners(4);
  corners[0] = mean + maxProjE1 * e1 + maxProjE2 * e2; //top right
  corners[1] = mean + minProjE1 * e1 + maxProjE2 * e2; //top left
  corners[2] = mean + minProjE1 * e1 + minProjE2 * e2; //bottom left
  corners[3] = mean + maxProjE1 * e1 + minProjE2 * e2; //bottom right

  //draw the oriented bounding box
  for(int i = 0; i < 4; i++){
    cv::line(output, corners[i], corners[(i + 1) % 4], Scalar(255, 0, 0), 2);
  }

  //display axis of least central moment
  float axisLength = std::max(maxProjE1 - minProjE1, maxProjE2 - minProjE2) * 0.5;
  cv::Point2f p1 = mean;
  cv::Point2f p2 = mean + axisLength * e1; //direction of e1
  cv::Point2f p3 = mean + axisLength * e2; //direction of e1

  //draw axis
  cv::line(output, p1, p2, Scalar(0, 255, 0), 2);
  cv::line(output, p1, p3, Scalar(0, 255, 0), 2);

  //area of region
  float regionArea = static_cast<float>(cv::countNonZero(regionMap == regionId));

  //bounding box dimensions
  float lengthE1 = maxProjE1 - minProjE1;
  float lengthE2 = maxProjE2 - minProjE2;
  float boxArea = lengthE1 * lengthE2;

  //percent filled
  float percentFilled = (regionArea / boxArea) * 100;

  //height/width ratio
  float ratio = lengthE2 / lengthE1;

  //display features
  std::ostringstream textStream;
  textStream << "Filled: " << std::fixed << std::setprecision(2) << percentFilled << "%, HW Ratio: " << ratio;
  cv::putText(output, textStream.str(), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

/*
Collect Features Function
Paramaters: region map, region id

This function collects features from a specific region label
and stores those features in a vector.
*/
std::vector<float>collectFeat(cv::Mat &regionMap, int regionId){
  //init feature vector
  std::vector<float>features;

  //calculate moments
  cv::Moments m = moments(regionMap == regionId);

  //points
  cv::Point2f mean(m.m10 / m.m00, m.m01 / m.m00); //mean point (ux, uy)

  //axis of least central moment
  float theta = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);
  //eigenvectors
  cv::Point2f e1(cos(theta), sin(theta));
  cv::Point2f e2(-sin(theta), cos(theta));

  //init min/max projections on eigenvectors
  float minProjE1 = FLT_MAX;
  float maxProjE1 = -FLT_MAX;
  float minProjE2 = FLT_MAX;
  float maxProjE2 = -FLT_MAX;

  //iterate through region map to find projections
  for(int y = 0; y < regionMap.rows; y++){
    for(int x = 0; x < regionMap.cols; x++){
      if(regionMap.at<int>(y, x) == regionId){
        cv::Point2f point(x, y);
        float projE1 = (point.x - mean.x) * e1.x + (point.y - mean.y) * e1.y; //projection on e1
        float projE2 = (point.x - mean.x) * e2.x + (point.y - mean.y) * e2.y; //projection on e2

        //update projections
        minProjE1 = std::min(minProjE1, projE1);
        maxProjE1 = std::max(maxProjE1, projE1);
        minProjE2 = std::min(minProjE2, projE2);
        maxProjE2 = std::max(maxProjE2, projE2);
      }
    }
  }
  //calculate corners
  std::vector<cv::Point2f> corners(4);
  corners[0] = mean + maxProjE1 * e1 + maxProjE2 * e2; //top right
  corners[1] = mean + minProjE1 * e1 + maxProjE2 * e2; //top left
  corners[2] = mean + minProjE1 * e1 + minProjE2 * e2; //bottom left
  corners[3] = mean + maxProjE1 * e1 + minProjE2 * e2; //bottom right

  //area of region
  float regionArea = static_cast<float>(cv::countNonZero(regionMap == regionId));

  //bounding box dimensions
  float lengthE1 = maxProjE1 - minProjE1;
  float lengthE2 = maxProjE2 - minProjE2;
  float boxArea = lengthE1 * lengthE2;

  //percent filled
  float percentFilled = (regionArea / boxArea) * 100;

  //height/width ratio
  float ratio = lengthE2 / lengthE1;

  features.push_back(percentFilled);
  features.push_back(ratio);

  return features;
}

/*
Function to calculate mean
*/
float calculateMean(std::vector<float> &vect){
  float sum = 0;
  for(float val: vect){
    sum += val;
  }
  return sum / vect.size();
}

/*
Function to calculate std deviation
*/
float calculateDev(std::vector<float> &vect, float mean){
  float sum = 0;
  for(int i = 0; i < vect.size(); i++){
    sum += (vect[i] - mean) * (vect[i] - mean);
  }
  sum /= vect.size();
  float std = sqrt(sum);
  return std;
}

//calculates the scaled euclidean distance between two vectors
float scaledEuclidean(std::vector<float> &vect1, std::vector<float> &vect2){
  float mean = calculateMean(vect1);
  float stdDev = calculateDev(vect1, mean);

  float sum = 0;
  for(size_t i = 0; i < vect1.size(); i++){
    float diff = (vect1[i] - vect2[i]) / stdDev;
    sum += diff * diff;
  }
  return sum;
}

/*
Classify Object using Euclidean distance
Returns best match label
*/
std::string classifyObj(std::vector<float> &newFeat, std::vector<std::vector<float>> &knownFeat, std::vector<char*> &labels){
  float minDistance = FLT_MAX;
  std::string matchLabel = "Unkown"; //store as unknown to begin

  //iterate through known features and check if there is a match
  for(size_t i = 0; i < knownFeat.size(); i++){
    float distance = scaledEuclidean(newFeat, knownFeat[i]);
    if(distance < minDistance){
      minDistance = distance;
      matchLabel = labels[i];
    }
  }
  return matchLabel;
}


/*
  Function courtesy of Bruce Maxwell

  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 128;
  cv::Mat padImg;
  cv::Mat blob;
	
  cv::Mat roiImg = src( bbox );
  int top = bbox.height > 128 ? 10 : (128 - bbox.height)/2 + 10;
  int left = bbox.width > 128 ? 10 : (128 - bbox.width)/2 + 10;
  int bottom = top;
  int right = left;
	
  cv::copyMakeBorder( roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0  );
  cv::resize( padImg, padImg, cv::Size( 128, 128 ) );

  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) / 0.5, // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  128,   // subtract mean prior to scaling
			  false, // input is a single channel image
			  true,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!/fc1/Gemm" );

  if(debug) {
    cv::imshow( "pad image", padImg );
    std::cout << embedding << std::endl;
    cv::waitKey(0);
  }

  return(0);
}

//calculates the sum of squared differences between two vectors
float SSD(std::vector<float> &vect1, std::vector<float> &vect2){
  float sum = 0;
  for(size_t i = 0; i < vect1.size(); i++){
    float diff = vect1[i] - vect2[i];
    sum += diff * diff;
  }
  return sum;
}

/*
calculates the sum of squared differences between two vectors and
returns the matching label
*/
std::string classifyObjSSD(std::vector<float> &newFeat, std::vector<std::vector<float>> &knownFeat, std::vector<char*> &labels){
  float minDistance = FLT_MAX;
  std::string matchLabel = "Unkown"; //store as unknown to begin

  //iterate through known features and check if there is a match
  for(size_t i = 0; i < knownFeat.size(); i++){
    float distance = SSD(newFeat, knownFeat[i]);
    if(distance < minDistance){
      minDistance = distance;
      matchLabel = labels[i];
    }
  }
  return matchLabel;
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