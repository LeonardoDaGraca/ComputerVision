/*
Leo DaGraca
CS5330
3/13/2024
This program's intention is to open a video channel, 
calculate the current position of the camera,
and display the virtual object.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>


using namespace std;
using namespace cv;

/*
Main function that displays translation and rotation vector data
and displays the virtual object
*/
int main(int argc, char *argv[]){
  //initialize VideoCapture obj 
  cv::VideoCapture cap(0);

  //camera check
  if(!cap.isOpened()){
    cerr << "Could not connect to the camera" << endl;
    return -1;
  }

  //read camera calibration parameters
  cv::FileStorage fs("intrinsic_parameters.yml", cv::FileStorage::READ);

  if(!fs.isOpened()){
    cerr << "Failed to open file" << endl;
    return -1;
  }

  cv::Mat camera_matrix;
  std::vector<double> dist_coeffs;

  //read data
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> dist_coeffs;
  fs.release();
  
  //display values
  cout << "Camera Matrix: " << endl << camera_matrix << endl;
  cout << "Distortion Coefficients: " << endl;
  for(double coeff : dist_coeffs){
    cout << coeff << " ";
  }
  cout << endl;


  //target calculation
  std::vector<Point3f> object_points;
  std::vector<Point2f> corner_set;
  cv:: Size pattern(8, 5);

  //object points for chessboard
  for(int i = 0; i < pattern.height; i++){
    for(int j = 0; j < pattern.width; j++){
      object_points.push_back(cv::Point3f(j, -i, 0.0f));
    }
  }

  //skyscrapers base and height
  std::vector<cv::Point3f> sky_bases = {
    {2, -2, 0}, {4, -4, 0}, {7, -3, 0}, {9, -1, 0}, {11, -5, 0}, 
    {0, -1, 0}, {5, -2, 0}, {-2, -1, 0}, {13, 4, 0}, {-1, 10, 0},
    {3, 10, 0}
  };
  std::vector<float> heights = {
    4.0, 3.0, 5.0, 4.5, 6.0, 
    8.0, 10.0, 6.0, 7.0, 10.0,
    6.0
  };

  while(true){
    cv::Mat frame;
    //capture frame by frame
    cap >> frame;
    //break if the frame is empty
    if(frame.empty()){
      break;
    }

    //set detection check
    bool foundPattern = cv::findChessboardCorners(frame, pattern, corner_set, 
                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 
                          + CALIB_CB_FAST_CHECK);
    
    if(foundPattern){
      cv::Mat grey;
      cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(grey, corner_set, Size(11, 11), Size(-1, -1),
          TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      
      //draw corners
      // cv::drawChessboardCorners(frame, pattern, corner_set, foundPattern);

      //parameters for solvePnP
      cv::Mat r_vec;
      cv::Mat t_vec;
      bool solved = cv::solvePnP(object_points, corner_set, camera_matrix, dist_coeffs, r_vec, t_vec);

      if(solved){
        cout << "Rotation Vector:\n" << r_vec << endl;
        cout << "Translation Vector:\n" << t_vec << endl;

        //cover target image
        std::vector<cv::Point3f> board_corners;
        cv::Point3f top_left = {-4, 4, 0};
        cv::Point3f top_right = {12, 4, 0};
        cv::Point3f bottom_left = {-4, -10, 0};
        cv::Point3f bottom_right = {12, -10, 0};
        
        std::vector<cv::Point3f> board_corners_3d = {
          top_left, top_right, bottom_left, bottom_right
        };

        std::vector<cv::Point2f> board_corners_2d;
        cv::projectPoints(board_corners_3d, r_vec, t_vec, camera_matrix, dist_coeffs, board_corners_2d);

        std::vector<cv::Point> rect_points(4);
        rect_points[0] = board_corners_2d[0];
        rect_points[1] = board_corners_2d[2];
        rect_points[2] = board_corners_2d[3];
        rect_points[3] = board_corners_2d[1];
        
        cv::fillConvexPoly(frame, rect_points, cv::Scalar(143, 188, 143));

        //Project outside corners and draw axes
        cv::projectPoints(object_points, r_vec, t_vec, camera_matrix, dist_coeffs, corner_set);

        // for(auto& point: corner_set){
        //   cv::circle(frame, point, 5, cv::Scalar(0, 0, 255), -1);
        // }

        //axes
        std::vector<cv::Point3f> axis_points = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}};
        std::vector<cv::Point2f> projected_axis_points;
        cv::projectPoints(axis_points, r_vec, t_vec, camera_matrix, dist_coeffs, projected_axis_points);
        // cv::line(frame, projected_axis_points[0], projected_axis_points[1], cv::Scalar(255, 0, 0), 2);
        // cv::line(frame, projected_axis_points[0], projected_axis_points[2], cv::Scalar(0, 255, 0), 2);
        // cv::line(frame, projected_axis_points[0], projected_axis_points[3], cv::Scalar(0, 0, 255), 2);
        
        

        //skyscrapers display
        for(size_t i = 0; i < sky_bases.size(); i++){
          std::vector<cv::Point3f> building_points = {
            sky_bases[i],
            {sky_bases[i].x + 1, sky_bases[i].y, 0},
            {sky_bases[i].x + 1, sky_bases[i].y - 1, 0},
            {sky_bases[i].x, sky_bases[i].y - 1, 0},
            sky_bases[i] + cv::Point3f(0, 0, -heights[i]),
            {sky_bases[i].x + 1, sky_bases[i].y, -heights[i]},
            {sky_bases[i].x + 1, sky_bases[i].y - 1, -heights[i]},
            {sky_bases[i].x, sky_bases[i].y - 1, -heights[i]}
          };

          std::vector<cv::Point2f> projected_points;
          cv::projectPoints(building_points, r_vec, t_vec, camera_matrix, dist_coeffs, projected_points);

          //filling sides of skyscrapers using polygons
          std::vector<std::vector<cv::Point>> sides = {
            {projected_points[0], projected_points[1], projected_points[5], projected_points[4]},
            {projected_points[1], projected_points[2], projected_points[6], projected_points[5]},
            {projected_points[2], projected_points[3], projected_points[7], projected_points[6]},
            {projected_points[3], projected_points[0], projected_points[4], projected_points[7]}
          };

          cv::Scalar color(170, 170, 170);
          for(auto &side : sides){
            cv::fillConvexPoly(frame, side, color);
          }

          //fill top of skyscraper
          std::vector<cv::Point> top = {
            projected_points[4], projected_points[5], projected_points[6], projected_points[7]
          };
          cv::fillConvexPoly(frame, top, color - cv::Scalar(20, 20, 20));
        }
        //clouds
        cv::circle(frame, cv::Point(1200, 300), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1200, 350), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1250, 325), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);

        cv::circle(frame, cv::Point(1200, 500), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1200, 550), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1250, 525), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);

        cv::circle(frame, cv::Point(1100, 400), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1100, 450), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        cv::circle(frame, cv::Point(1150, 425), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);

        // cv::circle(frame, cv::Point(1200, 400), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        // cv::circle(frame, cv::Point(1200, 450), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
        // cv::circle(frame, cv::Point(1150, 425), 50, cv::Scalar(79, 79, 47), CV_FILLED, 8, 0);
      }
    }

    //set key
    char key = waitKey(10);

    //command to close window
    if(key == 'q' || key == 'Q' || key == 27){
      break;
    }
    imshow("Video", frame);
  }

  cap.release();
  destroyAllWindows();

  return 0;
}