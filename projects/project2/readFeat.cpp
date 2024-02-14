/*
Leo DaGraca
CS5330
2/01/2024
This program's intention is to read an image's feature vector from a csv file.
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
#include <filesystem>

using namespace std;
using namespace cv;

//Features
enum Features{
  None,
  Base,
  Histogram,
  Multi,
  Texture,
  ResNet,
  LBP,
  Gabor,
};


/*
Function to extract filename from path
*/
std::string extractFileName(std::string &path){
  //find last pos of '/' or '\\'
  size_t pos = path.find_last_of("/\\");
  if(pos != std::string::npos){
    return path.substr(pos + 1); //return substring of the next pos
  }
  else{
    return path;
  }
}



/*
Main function that takes in image path and feature vector file.
It then computes the features for the target image, 
reads the feature vector file, and identifies the top N matches.
*/
int main(int argc, char *argv[]){
  //set up variables to process csv file
  std::vector<char *> filenames;
  std::vector<std::vector<float>> data;

  //make sure there are sufficient arguments
  if(argc < 3){
    cerr << "usage: " << argv[0] << " <target image path> <input csv>\n";
    return -1;
  }

  //read the target image
  std::string targetPath = argv[1];
  Mat targetImg = imread(targetPath);
  if(targetImg.empty()){
    cerr << "Target image " << targetPath << "cannot be opened";
    return -1;
  }

  //read in csv file
  int readResult = read_image_data_csv(argv[2], filenames, data);
  if(readResult != 0){
    cerr << "Failed to read CSV file\n";
    return -1;
  }

  //init empty feature
  Features mode = None;

  //init command line selection to choose mode
  cout << "Select a matching option based on the input csv file: ";
  char selection;
  selection = getchar();

  if(selection == 'b'){
    mode = Base;
  }
  if(selection == 'h'){
    mode = Histogram;
  }
  if(selection == 'm'){
    mode = Multi;
  }
  if(selection == 't'){
    mode = Texture;
  }
  if(selection == 'r'){
    mode = ResNet;
  }
  if(selection == 'l'){
    mode = LBP;
  }
  if(selection == 'g'){
    mode = Gabor;
  }

  switch(mode){
    case Base: {
      //get target features
      std::vector<float> targetFeat = baselineMatch(targetImg);
      //calculate SSD between target image and other images
      std::vector<std::pair<int, float>> values; //using pair to combine image index with SSD
      for(size_t i = 0; i < data.size(); i++){
        float ssd = SSD(targetFeat, data[i]);
        values.push_back(std::make_pair(i, ssd));
      }
      //sort the values using lambda expressions in ascending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second < b.second; //compare SSD values
      });
      //find the top 3 matches
      int N = 3;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; i <= N && i < values.size(); i++){
        if(values[i].second == 0){
          continue;
        }
        cout <<"Match " << i << ": " << filenames[values[i].first] << " with SSD: " << values[i].second << endl; 
      }
      break;
    }
    case Histogram: {
      std::vector<float> targetFeat = histMatch(targetImg);
      std::vector<std::pair<int, float>> values; //using pair to combine image index with intersection
      for(size_t i = 0; i < data.size(); i++){
        float intersection = histIntersection(targetFeat, data[i]);
        values.push_back(std::make_pair(i, intersection));
      }
      //sort the values using lambda expressions in descending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second > b.second; //compare intersection values a
      });

      //find the top 3 matches
      int N = 3;
      int count = 0;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; i <= N && i < values.size(); i++){
        std::string currFile = filenames[values[i].first];
        if(values[i].second >= 0.99){
          continue;
        }
        cout <<"Match " << count + 1 << ": " << filenames[values[i].first] << " with intersection: " << values[i].second << endl;
        count++; 
      }
      break;
    }
    case Multi: {
      std::vector<float> targetFeat = multiHistMatch(targetImg);
      std::vector<std::pair<int, float>> values; //using pair to combine image index with intersection
      for(size_t i = 0; i < data.size(); i++){
        float intersection = histIntersection(targetFeat, data[i]);
        values.push_back(std::make_pair(i, intersection));
      }
      //sort the values using lambda expressions in descending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second > b.second; //compare intersection values a
      });

      //find the top 3 matches
      int N = 3;
      int count = 0;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; count < N && i < values.size(); i++){
        std::string currFile = filenames[values[i].first];
        if(values[i].second >= 1.99){
          continue;
        }
        cout <<"Match " << count + 1 << ": " << filenames[values[i].first] << " with intersection: " << values[i].second << endl;
        count++; 
      }
      break;
    }
    case Texture: {
      std::vector<float> targetFeat = txtColorHistMatch(targetImg);
      std::vector<std::pair<int, float>> values; //using pair to combine image index with intersection
      for(size_t i = 0; i < data.size(); i++){
        float intersection = histIntersection(targetFeat, data[i]);
        values.push_back(std::make_pair(i, intersection));
      }
      //sort the values using lambda expressions in descending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second > b.second; //compare intersection values a
      });

      //find the top 3 matches
      int N = 3;
      int count = 0;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; count < N && i < values.size(); i++){
        std::string currFile = filenames[values[i].first];
        // if(values[i].second >= 1.99){
        //   continue;
        // }
        cout <<"Match " << count + 1 << ": " << filenames[values[i].first] << " with intersection: " << values[i].second << endl;
        count++; 
      }
      break;
    }
    case ResNet: {
      namespace fs = std::__fs::filesystem;
      //get the filename from the target img path
      fs::path targetPathObj(targetPath); //convert path string to filesystem path object
      std::string targetFilename = targetPathObj.filename(); //get the filename as string


      //find index of target img using an iterator
      auto it = std::find_if(filenames.begin(), filenames.end(), [&targetFilename](const char* filename){
        //convert C-style string to std::string for comparison
        std::string fname(filename);
        return fname == targetFilename; 
      });
      if(it == filenames.end()){
        cerr << "Target image "  << targetPath << " not found in CSV file\n";
        return -1;
      }
      size_t index = std::distance(filenames.begin(), it);

      //get feature vector for the target image
      std::vector<float> targetFeat = data[index];
      std::vector<std::pair<int, float>> values; //using pair to combine image index with SSD
      for(size_t i = 0; i < data.size(); i++){
        if(i == index) continue;
        float ssd = SSD(targetFeat, data[i]);
        values.push_back(std::make_pair(i, ssd));
      }
      //sort the values using lambda expressions in ascending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second < b.second; //compare SSD values
      });
      //find the top 3 matches
      int N = 3;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; i < N && i < values.size(); i++){
        if(values[i].second == 0){
          continue;
        }
        cout <<"Match " << i + 1 << ": " << filenames[values[i].first] << " with SSD: " << values[i].second << endl; 
      }
      break;
    }
    case LBP: {
      std::vector<float> targetFeat = lbpMatch(targetImg);
      std::vector<std::pair<int, float>> values; //using pair to combine image index with intersection
      for(size_t i = 0; i < data.size(); i++){
        float intersection = histIntersection(targetFeat, data[i]);
        values.push_back(std::make_pair(i, intersection));
      }
      //sort the values using lambda expressions in descending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second > b.second; //compare intersection values a
      });

      std::string targetFileName = extractFileName(targetPath);
      //find the top 5 matches
      int N = 5;
      int count = 0;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; count < N && i < values.size(); i++){
        std::string currFile = filenames[values[i].first];
        if(currFile == targetFileName){
          continue;
        }
        cout <<"Match " << count + 1 << ": " << filenames[values[i].first] << " with intersection: " << values[i].second << endl;
        count++; 
      }

      //find the lowest 5 matches
      cout << "Lowest " << N << " matches for image " << targetPath << endl;
      count = 0; //reset counter

      //reverse iteration
      for(auto it = values.rbegin(); count < N && it != values.rend(); it++){
        std::string currFile = filenames[it ->first];
        if(currFile == targetFileName){
          continue;
        }
        cout << "Match " << count + 1 << ": " << currFile << " with intersection: " << it->second << endl;
        count++;
      }
      break;
    }
    case Gabor: {
      std::vector<float> targetFeat = gaborMatch(targetImg);
      std::vector<std::pair<int, float>> values; //using pair to combine image index with intersection
      for(size_t i = 0; i < data.size(); i++){
        float intersection = histIntersection(targetFeat, data[i]);
        values.push_back(std::make_pair(i, intersection));
      }
      //sort the values using lambda expressions in descending order
      std::sort(values.begin(), values.end(), [](const std::pair<int, float>&a, const std::pair<int, float>&b){
        return a.second > b.second; //compare intersection values a
      });

      std::string targetFileName = extractFileName(targetPath);
      //find the top 5 matches
      int N = 5;
      int count = 0;
      cout << "Top " << N << " matches for image " << targetPath << endl;
      for(int i = 0; count < N && i < values.size(); i++){
        std::string currFile = filenames[values[i].first];
        if(currFile == targetFileName){
          continue;
        }
        cout <<"Match " << count + 1 << ": " << filenames[values[i].first] << " with intersection: " << values[i].second << endl;
        count++; 
      }

      //find the lowest 5 matches
      cout << "Lowest " << N << " matches for image " << targetPath << endl;
      count = 0; //reset counter

      //reverse iteration
      for(auto it = values.rbegin(); count < N && it != values.rend(); it++){
        std::string currFile = filenames[it ->first];
        if(currFile == targetFileName){
          continue;
        }
        cout << "Match " << count + 1 << ": " << currFile << " with intersection: " << it->second << endl;
        count++;
      }
      break;
    }
    default: {
      cerr << "Invalid selection" << endl;
      return -1;
    }
  }
  return 0;
}