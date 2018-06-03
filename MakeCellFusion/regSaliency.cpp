#include "stdafx.h"
#include"regSaliency.h"
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace xfeatures2d;
void mr::MisRegImg::GetTheKeypoints(const cv::Mat&image, std::vector<cv::KeyPoint>&keypoints, std::vector<cv::DetectedPoint>& detectedpoints, cv::Mat &descriptors)const {
	//To get the detected points and keypoints by the DoG pyramid.
	cv::Mat mask=Mat::ones(image.rows,image.cols,image.type());
	
	Ptr<xfeatures2d::SIFT>detector = xfeatures2d::SIFT::create();
	detector->detectAndCompute(image,mask, keypoints,descriptors);
}
void mr::MisRegImg::GetSaliencyMap(const std::vector<cv::DetectedPoint>& detectedpoints, cv::Mat&saliency_map) const{

}
void mr::MisRegImg::MatchImg(const std::vector<cv::KeyPoint>&f_keypoints, const std::vector<cv::KeyPoint>&s_keypoints, const cv::Mat&f_saliency_map, const cv::Mat&s_saliency_map, cv::Mat&warp_img)const {

}