# Project2
This is the readme for project2. The purpose of this project is to continue the process of learning how to manipulate and analyze images at a pixel level. 

- Project Contributors: Leonardo DaGraca
- Title: Project 2: Content-Based Image Retrieval
- OS: Mac OS
- IDE: Visual Studio
- Time Travel: 0 days

## Project Details
For this assignment there are two programs:
- The writeFeat.cpp program writes the features to a csv file and has the following usage: executable <directory path> <output csv>. The path of the directory with the images is given along with the csv the user wants to write the features. The user is then asked to select what feature function they want to apply on the images.
- The readFeat.cpp program reads the features from the csv file and compares it with a target image and has the following usage: executable <target image path> <input csv>. The path of the target image is given along with the csv that has the features from the image directory. The user is then asked to select what feature function they want to apply on the target image and it will return the top N matches.

The list of feature functions and the key to apply the feature:
  - Base -- 'b'
  - Histogram -- 'h'
  - Multi Histogram -- 'm'
  - Texture -- 't'
  - ResNet -- 'r'
  - LBP -- 'l'
  - Gabor -- 'g'

## Notes
For the extension (Gabor Filter), I created a smaller directory of images that consisted of similar images to the blue bin and different images to the blue bin.