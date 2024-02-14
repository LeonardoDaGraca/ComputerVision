/*
Leo DaGraca
CS5330
2/01/2024
This program's intention is to write an image's feature vector to a file.
Each image's feature vector will be stored in a csv file.
*/

//Files and libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "csv_util.h"
#include "features.h"


using namespace std;
using namespace cv;

//Features
enum Features{
  None,
  Base,
  Histogram,
  Multi,
  Texture,
  LBP,
  Gabor,
};

/*
Main function that takes in a directory of images 
and writes the feature vector for each image to a file.
*/

int main(int argc, char *argv[]){
  //set up variables to take store directory data
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;

  //make sure there are sufficient arguments
  if(argc < 3){
    cerr << "usage: " << argv[0] << " <directory path> <output csv>\n";
    return -1;
  }

  //get directory path
  strcpy(dirname, argv[1]);
  cout << "Processing directory " << dirname << "\n";

  //open directory
  dirp = opendir(dirname);
  if(dirp == NULL){
    cerr << "Directory: " << dirname << " cannot be opened\n";
    return -1;
  }

  fp = fopen(argv[2], "w");
  if(fp == NULL){
    cerr << "Csv output file could not be opened\n";
    return -1;
  }

  std::cout << "Starting to process images..." << std::endl;

  //init command line selection to choose mode
  cout << "Select a feature option to export onto the output csv file: ";
  char selection;
  cin >> selection;

  //init empty feature
  Features mode = None;
  switch(selection){
    case 'b':
      mode = Base;
      break;
    case 'h':
      mode = Histogram;
      break;
    case 'm':
      mode = Multi;
      break;
    case 't':
      mode = Texture;
      break;
    case 'l':
      mode = LBP;
      break;
    case 'g':
      mode = Gabor;
      break;
    default:
      cerr << "Invalid selection" << endl;
      closedir(dirp);
      return -1;
  }


  //loop over all image files in directory
  while((dp = readdir(dirp)) != NULL){
    //check for image files
    if(strstr(dp->d_name, ".jpg") ||
       strstr(dp->d_name, ".png") ||
       strstr(dp->d_name, ".ppm") ||
       strstr(dp->d_name, ".tif")){
        
        std::string imagePath = string(dirname) + "/" + string(dp->d_name); //get the image path
        Mat image = imread(imagePath); //store the current image being read as a Mat object
        if(image.empty()){
          cerr << "Current image " << imagePath << " is empty and cannot be opened\n";
          continue;
        }
        
        vector<float> features;
        switch(mode){
          case Base: {
            features = baselineMatch(image);
            break;
          }
          case Histogram: {
            features = histMatch(image);
            break;
          }
          case Multi: {
            features = multiHistMatch(image);
            break;
          }
          case Texture: {
            features = txtColorHistMatch(image);
            break;
          }
          case LBP: {
            features = lbpMatch(image);
            break;
          }
          case Gabor: {
            features = gaborMatch(image);
            break;
          }
          default: {
            cerr << "Invalid selection" << endl;
            return -1;
          }
        }
       append_image_data_csv(argv[2], dp->d_name, features);
       }
  }
  closedir(dirp);
  return 0;
}