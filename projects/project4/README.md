# Project 4
This is the readme for project 4. The goal for the project is learning how to calibrate a camera and then use the calibration to generate virtual objects in a scene. The end result is a program that can detect a target and then place a virtual object in the scene relative to the target that moves and orients itself correctly given motion of the camera or target.



- Project Contributors: Leonardo DaGraca
- Title: Project 4: Calibration and Augmented Reality
- OS: Mac OS
- IDE: Visual Studio
- Time Travel: 1 days

## Project Details
For this assignment there is 3 programs:
- The calibration.cpp program starts a video stream. It then detects and extracts target corners. The user then can place the target image in different calibration frames. For each desired frame, the user should select 's' to save that frame. Once 5 frames are saved, the console will alert the user. The user can then write the intrinsic parameters to a file by selecting 'w'.
- The main.cpp program calculates the current position of the camera,
and display the virtual object.
- The harrisCorners.cpp program detects the harris corners from the image on the input video.

## Notes
Extensions:
1. I made the target image not look like the target image anymore.
2. I created a creative virtual object/scene. My scene consisted of 3D skyscrapers with clouds surrounding them.
