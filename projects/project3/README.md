# Project 3
This is the readme for project 3. The goal is to have the computer identify a specified set of objects placed on a white surface in a translation, scale, and rotation invariant manner from a camera looking straight down. The computer should be able to recognize single objects placed in the image and show the position and category of the object in an output image. If provided a video sequence, it should be able to do this in real time.



- Project Contributors: Leonardo DaGraca
- Title: Project 3: Real-time 2-D Object Recognition
- OS: Mac OS
- IDE: Visual Studio
- Time Travel: 1 days

## Project Details
For this assignment there is one main program ran by just the executable:
- The objRecognition.cpp program starts a video stream. Once the video stream has started the user should select 't' to enter the threshold control. Here there will be two additional windows that will open (cleanup and output object). While in threshold control the user can do the following:
  - 'n' -- collected features using baseline method
  - 'c' -- classify new objects using scaled euclidean distance
  - 'x' -- collect embedding using pre-trained dnn
  - 'd' -- classify new objects based on embeddings and SSD matching

## Notes
All the utility functions that were used in the main program can be found in
utilFunctions.cpp.

For the extension (Dilation task), I implemented the function in utilFunctions.cpp and the process takes place in objRecognition.cpp.