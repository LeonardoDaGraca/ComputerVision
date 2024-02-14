/*
Leo DaGraca
CS5330
1/31/2024
Header file to hold all feature vector functions
*/

#ifndef FEATURES_H
#define FEATURES_H

std::vector<float> baselineMatch(cv::Mat &src);
std::vector<float> histMatch(cv::Mat &src);
float SSD(std::vector<float> &vect1, std::vector<float> &vect2);
float histIntersection(std::vector<float> &histA, std::vector<float> &histB);
std::vector<float> multiHistMatch(cv::Mat &src);
std::vector<float> txtColorHistMatch(cv::Mat &src);
std::vector<float> lbpMatch(cv::Mat &src);
std::vector<float>gaborMatch(cv::Mat &src);
#endif

