#include "stdafx.h"
#include"Octave.h"
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void sc::Octave(const Mat& base, std::vector<cv::Mat>& scale_maps, std::vector<cv::Mat>&feature_maps, cv::Mat& max_map,int layers, const int Octave_num,const int sigma) {
	
	if (Octave_num==0) {
		
		Mat sharp;
		Laplacian(base,sharp,base.depth());
		Mat feature_map;
		cv::GaussianBlur(abs(sharp),feature_map,Size(3,3),0,0);
		max_map = feature_map;
	}
	else {
		std::vector<vector<double>>Octave_sigmas;
		double kth;
		//Compute the sigma for each layer

		kth = std::pow(2., 1. / layers);
		//Several Octaves
		for (int oth = 0; oth < Octave_num; oth++) {
			std::vector<double>sigmas(layers+1);
			sigmas[0] = sigma;
			for (int lth = 1; lth < (layers+1); lth++) {//To establish the DoG, two more layers attached
				double sigma_up = std::pow(kth, (double)(lth - 1))*sigma;
				double sigma_down = sigma_up*kth;
				sigmas[lth] = std::sqrt(sigma_down*sigma_down - sigma_up*sigma_up);
			}
			Octave_sigmas.push_back(sigmas);
		}

		Mat src = base;
		std::vector<cv::Mat>max_maps;
		//Extract the blur maps in different scale
		//Several Octaves
		for (int onth = 0; onth < Octave_num; onth++) {

			std::vector<cv::Mat>DoGs;

			vector<double>single_sigmas;
			single_sigmas = Octave_sigmas[onth];
			Mat next_img;
			Mat dst_p;
			cv::GaussianBlur(src, dst_p, Size(), single_sigmas[0], single_sigmas[0]);
			for (int ssth = 1; ssth < single_sigmas.size(); ssth++) {//To blur each Octave with the different value of sigma
				Mat dst_b;
				cv::GaussianBlur(dst_p, dst_b, Size(), single_sigmas[ssth], single_sigmas[ssth]);
				if (ssth== (single_sigmas.size()-1)) {
					next_img = dst_b;
				}
				//********************Construct DOG pyramid****************************************
				Mat sub_img,bblur1;
				subtract(dst_p,dst_b,sub_img,noArray(),CV_32F);
				cv::GaussianBlur(cv::abs(sub_img), bblur1, Size(3, 3), 0, 0, BORDER_REFLECT);
				DoGs.push_back(bblur1);			
				scale_maps.push_back(dst_p);
				scale_maps.push_back(dst_b);
				dst_p = dst_b;
			}//Finish a single Octave

			//Choose the max_map of this Octave layer
			cv::Mat max_value = cv::Mat::zeros(DoGs[0].rows, DoGs[0].cols, DoGs[0].type());
			
			//***************choose the lap layers********************
			for (int mth = 0; mth < DoGs.size(); mth++) {
				max_value = cv::max(max_value, DoGs[mth]);
			}
	
			max_maps.push_back(max_value);
			Mat src_copy = next_img;
			cv::resize(src_copy, src, Size(src_copy.cols / 2, src_copy.rows / 2));

		}//Finish all of the Octaves
		int original_cols = max_maps[0].cols;
		int original_rows = max_maps[0].rows;
		vector<Mat>sized_maps;
		for (int mmth = 0; mmth < max_maps.size(); mmth++) {
			Mat sized_map;
			cv::resize(max_maps[mmth], sized_map, Size(original_cols, original_rows));
			sized_maps.push_back(sized_map);
		}
		//To choose the max value of all the max_maps from the different Octaves
		cv::Mat max_feature_map = cv::Mat::zeros(original_rows, original_cols, max_maps[0].type());
		for (int smth = 0; smth < sized_maps.size(); smth++) {
			max_feature_map = cv::max(max_feature_map, sized_maps[smth]);
		}
		max_map = max_feature_map;
	}//End of else with Octave !=0
}


