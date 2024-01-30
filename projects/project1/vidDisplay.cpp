/*
Leo DaGraca
CS5330
1/11/2024
This program's intention is to open a video channel, create a window, 
and then loop, capturing a new frame and displaying it each time through the loop
*/

//libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"


using namespace std;
using namespace cv;

//Filters
enum Filters{
  None,
  Greyscale,
  AltGreyscale,
  Sepia,
  Blur_1,
  Blur_2,
  Sobel_x,
  Sobel_y,
  Magnitude,
  Blur_Quant,
  FaceDetect,
  BrightContr,
  Embossing,
  ExcludeFace,
  VidSequence,
};

//Global variables for trackbar
const int alphaSliderMax = 300;
const int betaSliderMax = 100;
int alphaSlider;
int betaSlider;
float alpha = 1.0;
int beta = 0;


/*
Trackbar callback function for brightness/contrast control
Contrast slider will have [1.0 - 3.0] range, slider will start at 100
Brightness slider has a range of [0 - 100], slider will start at 50
*/
static void on_trackbar(int, void* action){
  //create two different trackbars for each action
  if(action == &alphaSlider){
    alpha = (float) alphaSlider / 100; //set slider value
  }
  else if(action == &betaSlider){
    beta = betaSlider - 50; //set slider value
  }
}

/*
Main function will create a window and start a live video stream from the computer's camera
It will display the video frame by frame
The user can exit the program by selecting either 'q' or 'esc'
The user can save a frame as an image by pressing 's' on the keyboard
The user can toggle in and out of filter modes by pressing specific keys on the keyboard
*/
int main(){
  //initialize VideoCapture object and set the input to 0 to connect to the camera
  VideoCapture cap(0);

  //camera check
  if(!cap.isOpened()){
    cout << "Could not connect to the camera" << endl;
    return -1;
  }

  // //frame properties
  // int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  // int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  // cout << "Frame properties: " << "width: " << frameWidth << " height: " << frameHeight << endl;

  //init empty filter
  Filters mode = None;
  
  //identify the window
  namedWindow("Video", 1);

  while(true){
    Mat frame;
    //capture frame by frame
    cap >> frame;
    //break if frame is empty
    if(frame.empty()){
      break;
    }

    char key = waitKey(10);

    //close the window
    if(key == 'q' || key == 'Q' || key == 27){
      break;
    }

    //save frame image using the image write function
    if(key == 's'){
      imwrite("capimage.jpg", frame);
    }

    //greyscale control, user can toggle into and out of greyscale by selecting 'g'
    if(key == 'g'){
      mode = (mode == Greyscale) ? None : Greyscale;
    }
    //alternative greyscale control, user can toggle into and out of greyscale by selecting 'h'
    if(key == 'h'){
      mode = (mode == AltGreyscale) ? None : AltGreyscale;
    }
    //sepia control, user can toggle into and out of sepia by selecting 't'
    if(key == 't'){
      mode = (mode == Sepia) ? None : Sepia;
    }
    //blur filter 1 (naive), user can toggle into and out of blur by selecting 'u'
    if(key == 'u'){
      mode = (mode == Blur_1) ? None : Blur_1;
    }
    //blur filter 2, user can toggle into and out of blur by selecting 'b'
    if(key == 'b'){
      mode = (mode == Blur_2) ? None : Blur_2;
    }
    //sobel filter x, user can toggle into and out of sobel by selecting 'x'
    if(key == 'x'){
      mode = (mode == Sobel_x) ? None : Sobel_x;
    }
    //sobel filter y, user can toggle into and out of sobel by selecting 'y'
    if(key == 'y'){
      mode = (mode == Sobel_y) ? None : Sobel_y;
    }
    //gradient magnitude filter, user can toggle into and out of sobel by selecting 'm'
    if(key == 'm'){
      mode = (mode == Magnitude) ? None : Magnitude;
    }
    //blur quantize filter, user can toggle into and out of blur by selecting 'l'
    if(key == 'l'){
      mode = (mode == Blur_Quant) ? None : Blur_Quant;
    }
    //face detect filter, user can toggle into and out of detecting faces by selecting 'f'
    if(key == 'f'){
      mode = (mode == FaceDetect) ? None : FaceDetect;
    }
    //brightness/contrast control, user can toggle into and out of control by selecting 'z'
    if(key == 'z'){
      mode = (mode == BrightContr) ? None : BrightContr;
      if(mode == BrightContr){
        namedWindow("Brightness/Contrast", 1);
        //trackbar for Contrast
        alphaSlider = static_cast<int>(alpha * 100);
        createTrackbar("Contrast", "Brightness/Contrast", &alphaSlider, alphaSliderMax, on_trackbar, &alphaSlider);
        //trackbar for Brightness
        betaSlider = beta + 50;
        createTrackbar("Brightness", "Brightness/Contrast", &betaSlider, betaSliderMax, on_trackbar, &betaSlider);
      }
      else{
        destroyWindow("Brightness/Contrast");
      }
    }
    //embossing filter, user can toggle into and out of the filter by selecting 'e'
    if(key == 'e'){
      mode = (mode == Embossing) ? None : Embossing;
    }
    //blur the image outside of found faces filter, user can toggle into and out of filter by selecting 'j'
    if(key == 'j'){
      mode = (mode == ExcludeFace) ? None : ExcludeFace;
    }

    //using switch statements to call functions respective to the filter mode
    switch(mode){
      case Greyscale: {
        Mat greyscale;
        cvtColor(frame, greyscale, COLOR_BGR2GRAY); //conversion to greyscale
        imshow("Video", greyscale);
        break;
      }
      case AltGreyscale: {
        Mat altGreyscale;
        greyscale(frame, altGreyscale);
        imshow("Video", altGreyscale);
        break;
      }
      case Sepia: {
        Mat sepiaTone;
        sepia(frame, sepiaTone);
        imshow("Video", sepiaTone);
        break;
      }
      case Blur_1: {
        Mat blurFrame;
        blur5x5_1(frame, blurFrame);
        imshow("Video", blurFrame);
        break;
      }
      case Blur_2: {
        Mat blurFrame_2;
        blur5x5_2(frame, blurFrame_2);
        imshow("Video", blurFrame_2);
        break;
      }
      case Sobel_x: {
        Mat sobelFrame_x;
        Mat display;
        sobelX3x3(frame, sobelFrame_x);
        convertScaleAbs(sobelFrame_x, display);
        imshow("Video", display);
        break;
      }
      case Sobel_y: {
        Mat sobelFrame_y;
        Mat display;
        sobelY3x3(frame, sobelFrame_y);
        convertScaleAbs(sobelFrame_y, display);
        imshow("Video", display);
        break;
      }
      case Magnitude: {
        Mat sx, sy, display;
        sobelX3x3(frame, sx);
        sobelY3x3(frame, sy);
        magnitude(sx, sy, display);
        imshow("Video", display);
        break;
      }
      case Blur_Quant: {
        Mat display;
        blurQuantize(frame, display, 10);
        imshow("Video", display);
        break;
      }
      case FaceDetect: {
        Mat grey; //grey mat object to be passed into detectFaces
        vector<Rect> faces;
        Rect last(0, 0, 0, 0);
        cvtColor(frame, grey, COLOR_BGR2GRAY, 0);
        detectFaces(grey, faces);
        drawBoxes(frame, faces);

        //add smoothing by averaging last two detections
        if(faces.size() > 0) {
          last.x = (faces[0].x + last.x)/2;
          last.y = (faces[0].y + last.y)/2;
          last.width = (faces[0].width + last.width)/2;
          last.height = (faces[0].height + last.height)/2;
        }
        //display face detection
        imshow("Video", frame);
      }
      case BrightContr: {
        Mat display;
        brightness_contrast(frame, display, alpha, beta);
        imshow("Video", display);
        break;
      }
      case Embossing: {
        Mat sx, sy, display;
        sobelX3x3(frame, sx);
        sobelY3x3(frame, sy);
        emboss(sx, sy, display);
        imshow("Video", display);
        break;
      }
      case ExcludeFace: {
        Mat grey; //grey mat object to be passed into detectFaces
        vector<Rect> faces;
        Rect last(0, 0, 0, 0);
        cvtColor(frame, grey, COLOR_BGR2GRAY, 0);
        detectFaces(grey, faces);
        drawBoxes(frame, faces);

        //add smoothing by averaging last two detections
        if(faces.size() > 0) {
          last.x = (faces[0].x + last.x)/2;
          last.y = (faces[0].y + last.y)/2;
          last.width = (faces[0].width + last.width)/2;
          last.height = (faces[0].height + last.height)/2;
        }
        excludeFaceBlur(frame, faces);
        imshow("Video", frame);
        break;
      }
      case VidSequence: {
        video.write(frame);
        break;
      }
      default: {
        imshow("Video", frame);
      }
    }
  }
  //release video capture
  cap.release();
  //close all windows
  destroyAllWindows();

  return 0;
}
