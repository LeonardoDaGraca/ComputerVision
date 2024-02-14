/*
Leo DaGraca
CS5330
1/17/2024
Filter header file
*/

#ifndef FILTER_H
#define FILTER_H

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int brightness_contrast(cv::Mat &src, cv::Mat &dst, float alpha, int beta);
int emboss(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int excludeFaceBlur(cv::Mat &frame, std::vector<cv::Rect> &faces);

#endif
