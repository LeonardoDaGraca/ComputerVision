/*
Leo DaGraca
CS5330
3/05/2024
This program's intention is to open a video channel, 
detect and extract target corners, select calibration images,
and calibrate the camera
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;


/*
Main function that will complete the camera 
calibration tasks
*/
int main(int argc, char *argv[]){
  //initialize VideoCapture obj 
  cv::VideoCapture cap(0);

  //camera check
  if(!cap.isOpened()){
    cerr << "Could not connect to the camera" << endl;
    return -1;
  }

  //locations for calibration
  std::vector<std::vector<cv::Vec3f>> point_list;
  std::vector<std::vector<cv::Point2f>> corner_list;
  int selected_frames = 0; //frames for calibration

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
    

    //detect and extract target corners
    cv::Size pattern(8, 5); //num of interior corners for 9 x 6 chessboard
    cv::Mat grey;
    cv::cvtColor(frame, grey, COLOR_BGR2GRAY);
    std::vector<Point2f> corner_set;

    //set detection check
    bool foundPattern = cv::findChessboardCorners(grey, pattern, corner_set, 
                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 
                          + CALIB_CB_FAST_CHECK);
    
    if(foundPattern){
      cv::cornerSubPix(grey, corner_set, Size(11, 11), Size(-1, -1),
          TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      
      cv::drawChessboardCorners(frame, pattern, Mat(corner_set), foundPattern);
    
      cout << "Number of corners: " << corner_set.size() << endl;

      if(!corner_set.empty()){
        cout << "corner coordinates: ("  << corner_set[0].x << ", "<< corner_set[0].y << ")" << endl;
      } else{
        cout << "No corners found" << endl;
      }
      //Select Calibration Images
      if(key == 's' || key == 'S'){
        selected_frames++; //increment frames
        //save corners
        corner_list.push_back(corner_set);

        //create and save 3D points
        std::vector<cv::Vec3f> point_set;
        for(int i = 0; i < pattern.height; i++){
          for(int j = 0; j < pattern.width; j++){
            point_set.push_back(cv::Vec3f(j, -i, 0.0f));
          }
        }

        //check 3D points with number of corners
        if(point_set.size() == corner_set.size()){
          point_list.push_back(point_set);

           cout << "3D points for the selected frame: " << endl;
           for(auto &point: point_set){
            cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")" << endl;
           } 

        } else{
          cerr << "Number of 3D points does not match detected corners" << endl;
        }
       

        cout << "Selected frame for calibration. Total selected frames: " << selected_frames << endl;
      }
    }

    /* Calibrate the camera
        initializing in this format : 
        [1, 0, frame.cols/2]
        [0, 1, frame.rows/2]
        [0, 0, 1           ] 
    */
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
    camera_matrix.at<double>(0, 0) = 1; //fx
    camera_matrix.at<double>(1, 1) = 1; //fy
    camera_matrix.at<double>(0, 2) = frame.cols / 2.0; //cx
    camera_matrix.at<double>(1, 2) = frame.rows / 2.0; //cy

    //distortion coeffecients
    std::vector<double> dist_coeffs;

    //translation and rotation vectors
    std::vector<cv::Mat> rot_vec;
    std::vector<cv::Mat> trans_vec;

    if(selected_frames >= 5) {
      cout << "Enough frames selected for calibration." << endl;

      double rms = cv::calibrateCamera(point_list, corner_list, frame.size(), camera_matrix, dist_coeffs, rot_vec, trans_vec, CALIB_FIX_ASPECT_RATIO);
      //print or check the results
      cout << "RMS error: " << rms << endl;
      cout << "Camera matrix: " << camera_matrix << endl;
      cout << "Distortion coefficients: ";
      for(size_t i = 0; i < dist_coeffs.size(); i++){
        cout << dist_coeffs[i];
        if(i < dist_coeffs.size() - 1){
          cout << ", ";
        }
      }
      cout << endl;

      //option for user to save intrinsic parameters to a file
      cout << "Press 'w' to write camera matrix and distortion coefficients to file" << endl;
      if(key == 'w'){
        cv::FileStorage fs ("intrinsic_parameters.yml", cv::FileStorage::WRITE);
        fs << "camera_matrix" << camera_matrix << "distortion_coefficients" << dist_coeffs;
        fs.release();
        cout << "File saved" << endl;
      }
    }
    imshow("Video", frame);
  }

  cap.release();
  destroyAllWindows();

  return 0;
}