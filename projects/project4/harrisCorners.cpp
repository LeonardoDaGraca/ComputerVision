/*
Leo DaGraca
CS5330
3/15/2024
This program's intention is to open a video channel, 
detect the harris corners from the image
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

/*
Main function to detect harris corners
*/
int main(int argc, char *argv[]){
  //initialize VideoCapture obj 
  cv::VideoCapture cap(0);

  //camera check
  if(!cap.isOpened()){
    cerr << "Could not connect to the camera" << endl;
    return -1;
  }

  while(true){
    cv::Mat frame;
    //capture frame by frame
    cap >> frame;
    //break if the frame is empty
    if(frame.empty()){
      break;
    }

    //set key
    char key = waitKey(10);

    //command to close window
    if(key == 'q' || key == 'Q' || key == 27){
      break;
    }

    //convert src Mat to grey
    cv::Mat grey;
    cv::cvtColor(frame, grey, COLOR_BGR2GRAY);

    //apply harris corners
    cv::Mat output = Mat::zeros(frame.size(), CV_32FC1);
    cv::cornerHarris(grey, output, 2, 3, 0.04);

    //normalize image
    cv::Mat output_norm;
    // cv::Mat output_norm_scaled;
    cv::normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    // cv::convertScaleAbs(output_norm, output_norm_scaled);
    
    int thresh = 120;
    for(int i = 0; i < output_norm.rows; i++){
      for(int j = 0; j < output_norm.cols; j++){
        if((int) output_norm.at<float>(i, j) > thresh){
          cv::circle(frame, cv::Point(j, i), 5, Scalar(0, 255, 0), 2, 8, 0);
        }  
      }
    }
    imshow("Harris Corners", frame);
  }
  cap.release();
  destroyAllWindows();

  return 0;
}