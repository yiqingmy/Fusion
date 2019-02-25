#pragma once
#ifndef Octave_H
#define Octave_H
#include <opencv2/opencv.hpp>
#include<iostream>
//#include<opencv2/xfeatures2d.hpp>
namespace sc {
	void Octave(const cv::Mat&base,std::vector<cv::Mat>&scale_maps, std::vector<cv::Mat>&feature_maps,cv::Mat& max_map,const int layers, const int Octave_num,const int sigma);
	
}
#endif // !Octave_H

