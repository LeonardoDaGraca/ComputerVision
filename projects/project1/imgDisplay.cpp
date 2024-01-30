/*
Leo DaGraca
CS5330
1/9/2024
This program's intention is to read an image from a file and display it
*/

//libraries
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

/*
Main function that takes in an image and then displays it using command line arguments
The image is stored as a Mat object after it is read in using the imread function
The imshow function displays the read image
The waitkey() function waits for either the q or esc key to be pressed for the window to close
*/
int main(int argc, char *argv[]){
  char filename[256]; //string for the file name

  //check for two command line arguments
  if(argc < 2) {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable


  Mat image = imread(filename, IMREAD_COLOR); //allocating input image as a Mat image
  imshow("Display Window", image); //displaying the image
  
  while(true){
    char key = waitKey(0);
    if (key == 'q' || key == 'Q' || key == 27){
      break;
    }
  }
  return 0;
}
