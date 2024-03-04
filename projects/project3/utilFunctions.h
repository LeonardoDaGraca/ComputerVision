/*
Leo DaGraca
CS5330
2/14/2024
Utility header file
*/

#ifndef UTILFUNCTIONS_H
#define UTILFUNCTIONS_H

void threshold(cv::Mat &src, cv::Mat &dst, int thresh, int maxval);
void computeFeat(cv::Mat &regionMap, int regionId, cv::Mat &output);
std::vector<float>collectFeat(cv::Mat &regionMap, int regionId);
float scaledEuclidean(std::vector<float> &vect1, std::vector<float> &vect2);
std::string classifyObj(std::vector<float> &newFeat, std::vector<std::vector<float>> &knownFeat, std::vector<char*> &labels);
cv::Mat customDilate(cv::Mat &src, cv::Mat &dst, int kernelSize);
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug);
float SSD(std::vector<float> &vect1, std::vector<float> &vect2);
int greyscale(cv::Mat &src, cv::Mat &dst);
std::string classifyObjSSD(std::vector<float> &newFeat, std::vector<std::vector<float>> &knownFeat, std::vector<char*> &labels);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

#endif

