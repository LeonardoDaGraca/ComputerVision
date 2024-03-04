/*
Leo DaGraca
CS5330
2/14/2024
This program's intention is to open a video channel, create a window, 
and then loop, capturing a new frame and displaying it each time through the loop
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "utilFunctions.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

//Filters
enum Filters{
  None,
  Thresh,
};

//set up confusion matrix
int confusionMatrix[5][5] = {0};

/*
Main function will create a window and start a live video stream from the computer's camera
It will display the video frame by frame
*/
int main(int argc, char *argv[]){
  //initialize VideoCapture obj 
  cv::VideoCapture cap(0);

  //camera check
  if(!cap.isOpened()){
    cerr << "Could not connect to the camera" << endl;
    return -1;
  }

  Filters mode = None;

  //video window
  cv::namedWindow("Video", 1);

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

    //thresh control
    if(key == 't'){
      mode = (mode == Thresh) ? None : Thresh;
    }

    switch(mode){
      case Thresh: {
        Mat display, blur, grey;
        cv::GaussianBlur(frame, blur, cv::Size(3, 3), 0); //blur original frame
        //blur5x5_2(frame, blur);
        greyscale(blur, grey);

        //sample pixel values
        int B = 4;
        std::vector<float> data;
        for(int i = 0; i < grey.rows - B; i += B){
          for(int j = 0; j < grey.cols - B; j += B){
            int jx = rand() % B;
            int jy = rand() % B;
            data.push_back(static_cast<float>(grey.at<uchar>(i + jy, j + jx)));
          }
        }

        //make sure data is in right format for kmeans
        cv::Mat samples = cv::Mat(data.size(), 1, CV_32F, data.data());

        //apply kmeans
        Mat labels, centers;
        int K = 2;
        cv::kmeans(samples, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.01),
                  3, cv::KMEANS_PP_CENTERS, centers);
        //dynamic threshold
        float dynamicThresh = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2;
        threshold(blur, display, dynamicThresh, 255);

        //adding cleanup using dilation
        cv::Mat erosion, dilation;
        customDilate(display, dilation, 3);
        imshow("Cleaned Image", dilation);

        //segment the image into regions
        Mat ccLabels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(dilation, ccLabels, stats, centroids);

        //center of image
        cv::Point imageCenter(dilation.cols / 2, dilation.rows / 2);

        //filter out regions that are too small and touching edges
        int minThreshold = 500;
        vector<int>validLabels;
        vector<float>centroidDistances;


        //filter regions and store valid labels
        for(int i = 1; i < nLabels; i++){
          int area = stats.at<int>(i, CC_STAT_AREA);
          int left = stats.at<int>(i, CC_STAT_LEFT);
          int top = stats.at<int>(i, CC_STAT_TOP);
          int width = stats.at<int>(i, CC_STAT_WIDTH);
          int height = stats.at<int>(i, CC_STAT_HEIGHT);

          //check for edge touching
          bool touchesEdge = left <= 0 || top <= 0 || (left + width) >= dilation.cols || (top + height) >= dilation.rows;

          if(area > minThreshold && !touchesEdge){
            validLabels.push_back(i);

            //get distance of centroid
            cv::Point centroid(centroids.at<float>(i, 0), centroids.at<float>(i, 1));
            float distance = cv::norm(centroid - imageCenter);
            centroidDistances.push_back(distance);
          }
        }
        //center most region
        if(!centroidDistances.empty()){
          auto minIt = min_element(centroidDistances.begin(), centroidDistances.end());
          int labelIndex = std::distance(centroidDistances.begin(), minIt);
          int centerLabel = validLabels[labelIndex];

          validLabels = {centerLabel};
        }

        //output image
        Mat output = Mat::zeros(dilation.size(), CV_8UC3);
        cv::Mat netMat = Mat::zeros(output.size(), CV_8UC1);

        //give color to each valid label
        for(int i = 0; i < validLabels.size(); i++){
          int label = validLabels[i];
          //mask for current label
          Mat mask = ccLabels == label;
          //set color to region in output img
          output.setTo(Scalar(0, 0, 255), mask);
          netMat.setTo(Scalar(255), mask);
        }

        for(int i = 0; i < validLabels.size(); i++){
          int label = validLabels[i];
          computeFeat(ccLabels, label, output);
        }
        imshow("Object", output);

        /*
        Collect features
        */
        if(key == 'n' || key == 'N'){
          //get region id and store features
          for(int i = 0; i < validLabels.size(); i++){
            int label = validLabels[i];
            std::vector<float>features = collectFeat(ccLabels, label);

            //take user input
            char input[20];
            char filename[] = "../features.csv";
            cout << "Enter label/name for current object " << (i + 1) << ": ";
            cin >> input;

            // Print features to console
            cout << "Features for " << input << ": ";
            for (auto feature : features) {
                cout << feature << " ";
            }
            cout << endl;

            //append features and input to csv
            append_image_data_csv(filename, input, features, 0);
          }
        }
        /*
        Classify new objects using scaled euclidean distance
        */
        if(key == 'c' || key == 'C'){
          //static matrix
          static int confusionMatrix[5][5] = {0};
          //set up variables to process csv file
          char csv[] = "../features.csv";
          std::vector<char *>filenames;
          std::vector<std::vector<float>>data;

          //read in csv file
          int readResult = read_image_data_csv(csv, filenames, data, 0);
          if(readResult != 0){
            cerr << "Failed to read CSV file\n";
            return -1;
          }

          //map labels to indices
          std::map<std::string, int> labelToIndex = {
            {"orange", 0},
            {"pen", 1},
            {"key", 2},
            {"usb", 3},
            {"teaPacket", 4}
          };

          //extract features from current object
          for(int i = 0; i < validLabels.size(); i++){
            int rLabel = validLabels[i];
            std::vector<float> currentFeat = collectFeat(ccLabels, rLabel);

            //prompt user for true label
            std::string trueLabel;
            cout << "Enter true label for the current object: ";
            cin >> trueLabel;

            int trueIndex = labelToIndex[trueLabel];

            //classify current object
            std::string predLabel = classifyObj(currentFeat, data, filenames);
            int predIndex = labelToIndex[predLabel];

            confusionMatrix[trueIndex][predIndex]++;
            //display label
            cv::putText(output, predLabel, imageCenter, cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 2);
          }
         
          imshow("Classification", output);

          cout << "Confusion Matrix:" << endl;
          for(int i = 0; i < 5; i++){
            for(int j = 0; j < 5; j++){
                cout << confusionMatrix[i][j] << " ";
              }
            cout << endl;
          }
        }
        /*
        Second classification method using pre-trained dnn
        */
        //train mode
        if(key == 'x' || key == 'X'){
          //load pre-trained dnn
          cv::dnn::Net net = cv::dnn::readNetFromONNX("/Users/leodagraca/Desktop/CS5330/projects/project3/or2d-normmodel-007.onnx");
          char filename[] = "../dnn_features.csv"; //store embeddings and labels
        
          for(int i = 0; i < validLabels.size(); i++){
            int label = validLabels[i];
            cv::Rect bbox = cv::Rect(0, 0, netMat.cols, netMat.rows);
            cv::Mat embedding;
            int result = getEmbedding(netMat, embedding, bbox, net, 0);
            if(result != 0){
              cerr << "Failed to get embedding";
              continue;
            }

            //add embedding to db
            std::vector<float>embeddingVect;
            for(int i = 0; i < embedding.rows; i++){
              for(int j = 0; j < embedding.cols; j++){
                embeddingVect.push_back(embedding.at<float>(i, j));
              }
            }

            //user input for label
            char input[20];
            cout << "Enter label/name for current object " << (i + 1) << ": ";
            cin >> input;

            // Append embedding and label to CSV
            append_image_data_csv(filename, input, embeddingVect, 0);
          }
        }

        //classify mode
        if(key == 'd' || key == 'D'){
          //load pre-trained dnn
          cv::dnn::Net net = cv::dnn::readNetFromONNX("/Users/leodagraca/Desktop/CS5330/projects/project3/or2d-normmodel-007.onnx");
          
          char csv[] = "../dnn_features.csv";
          std::vector<char *>labels;
          std::vector<std::vector<float>>embeddings;

          //read embeddings and labels from csv
          int readResult = read_image_data_csv(csv, labels, embeddings, 1);
          if(readResult != 0){
            cerr << "Failed to read CSV file\n";
            return -1;
          }

          //create instance of confusion matrix
          static int confusionMatrix[5][5] = {0};

          //map labels to indices
          std::map<std::string, int> labelToIndex = {
            {"orange", 0},
            {"pen", 1},
            {"key", 2},
            {"usb", 3},
            {"teaPacket", 4}
          };

          for(int i = 0; i < validLabels.size(); i++){
            int label = validLabels[i];
            cv::Mat newEmbedding;
            cv::Rect bbox = cv::Rect(0, 0, output.cols, output.rows);
            getEmbedding(netMat, newEmbedding, bbox, net, 0);

            //get embedding to vector for current obj
            std::vector<float> newEmbeddingVector;
            for(int i = 0; i < newEmbedding.rows; i++){
              for(int j = 0; j < newEmbedding.cols; j++){
                newEmbeddingVector.push_back(newEmbedding.at<float>(i, j));
              }
            }

            //classify new object based on embeddings
            std::string predLabel = classifyObjSSD(newEmbeddingVector, embeddings, labels);

            //prompt user for true label
            std::string trueLabel;
            cout << "Enter true label for the current object: ";
            cin >> trueLabel;

            //update confusion matrix
            int trueIndex = labelToIndex[trueLabel];
            int predIndex = labelToIndex[predLabel];
            confusionMatrix[trueIndex][predIndex]++;


            //display predicted label
            cout << "Predicted label: " << predLabel << endl;
          }
          //display confusion matrix
          cout << "Confusion Matrix:" << endl;
          for(int i = 0; i < 5; i++){
            for(int j = 0; j < 5; j++){
              cout << confusionMatrix[i][j] << " ";
            }
            cout << endl;
          }
        }
        break;
      }
      default: {
        //show original video
        imshow("Video", frame);
        break;
      }
    }
  }
  cap.release();
  destroyAllWindows();

  return 0;
}