# Project1
This is the readme for project1. The purpose of this project is to familiarize with C/C++, the OpenCV package, and the mechanics of opening, capturing, manipulating, and writing images.   

- Project Contributors: Leonardo DaGraca
- Title: Project 1: Video-special effects
- OS: Mac OS
- IDE: Visual Studio
- Time Travel: 0 days

## Project Details
For this assignment the main program is located in vidDisplay.cpp. The implementation of each filter function can be found in filter.cpp. Users can toggle in and out of a filter by selecting specific keys.

The list of filter functions and the key to apply the filter:
  - Greyscale -- 'g'
  - AltGreyscale -- 'h'
  - Sepia -- 't'
  - Blur_1 -- 'u'
  - Blur_2 -- 'b'
  - Sobel_x -- 'x'
  - Sobel_y -- 'y'
  - Magnitude -- 'm'
  - Blur_Quant -- 'l'
  - FaceDetect -- 'f'
  - BrightContr -- 'z'
  - Embossing -- 'e'
  - ExcludeFace -- 'j'

## Notes
- The user can exit the program by selecting either 'q' or 'esc'
- The user can save a frame as an image by pressing 's' on the keyboard
- For the Brightness/Control filter, the user must interact with the command line and enter in alpha and beta values for the desired filter to be captured.
