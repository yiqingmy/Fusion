#include "stdafx.h"
#include"Octave.h"
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
void sc::Octave(const Mat& base, std::vector<cv::Mat>& scale_maps, std::vector<cv::Mat>&feature_maps, cv::Mat& max_map,int layers, const int Octave_num,const int sigma) {
	
	if (Octave_num==0) {//similar to the gff
		
		Mat sharp;
		Laplacian(base,sharp,base.depth());
		Mat feature_map;
		GaussianBlur(sharp,feature_map,Size(3,3),0,0);
		max_map = feature_map;
	}
	else {
		std::vector<vector<double>>Octave_sigmas;

		//*********Record each sigma for each layer****************
		std::vector<double>layer_sigmas(layers);
		std::vector<std::vector<double>>Octave_layer_sigmas;
		//*********************************************************
		double kth;
		//Compute the sigma for each layer

		kth = std::pow(2., 1. / layers);
		//Several Octaves
		for (int oth = 0; oth < Octave_num; oth++) {
			std::vector<double>sigmas(layers+1);
			if (oth == 0) {
				sigmas[0] = sigma;
			}
			else {
				//sigmas[0] = pow(2, oth)*sigma;
				sigmas[0] = sigma;
			}

			for (int lth = 1; lth < layers+1; lth++) {

				double sigma_up = std::pow(kth, (double)(lth - 1))*sigma;
				//layer_sigmas[lth]=sigma_up;
				double sigma_down = sigma_up*kth;
				sigmas[lth] = std::sqrt(sigma_down*sigma_down - sigma_up*sigma_up);
			}
			Octave_sigmas.push_back(sigmas);
			//Octave_layer_sigmas.push_back(layer_sigmas);
		}

		Mat src = base;
		std::vector<cv::Mat>max_maps;
		//Extract the blur maps in different scale
		//Several Octaves
		for (int onth = 0; onth < Octave_num; onth++) {

			std::vector<cv::Mat>laps;

			vector<double>single_sigmas;
			single_sigmas = Octave_sigmas[onth];
			cout << "single_sigmas size:" << single_sigmas.size() << endl;
			//*******************************************
			//vector<double>single_layer_sigmas;
			//single_layer_sigmas = Octave_layer_sigmas[onth];
			for (int ssth = 0; ssth < single_sigmas.size(); ssth++) {//To blur each Octave with the different value of sigma
				Mat dst;

				GaussianBlur(src, dst, Size(), single_sigmas[ssth], single_sigmas[ssth]);
				//******************Test**********************
				Mat lap1;
				//Laplacian(dst, lap1, dst.depth());
				//Mat lap1;
				/*Mat kernel = (Mat_<float>(3, 3) <<
					0, 1, 0,
					1, -4, 1,
					0, 1, 0);*/
				Mat kernel_s = (Mat_<float>(3, 3) <<
					1, 1, 1,
					1, -8, 1,
					1, 1, 1);
				//*************************************
				double lay_sig = pow(single_sigmas[ssth],2);
				Mat kernel = kernel_s*lay_sig;
				//cv::filter2D(dst, lap1, CV_32F, kernel, Point(-1, -1), 0, BORDER_REFLECT);
				cv::filter2D(dst, lap1, CV_32F, kernel_s, Point(-1, -1), 0, BORDER_REFLECT);
				if (ssth!= single_sigmas.size()-1) {
					Mat bblur1;
					GaussianBlur(cv::abs(lap1), bblur1, Size(3, 3), 0, 0, BORDER_REFLECT);
					//GaussianBlur(cv::abs(lap1), bblur1, Size(41, 41), 0, 0, BORDER_REFLECT);
					//********************************************
					//sprintf_s(name, "H:\\Project\\PCellFusion\\fusion\\%dfusion.jpg", j);
					//cv::imwrite(name, dst);
					laps.push_back(bblur1);
					//laps.push_back(cv::abs(lap1));
				}
				
				scale_maps.push_back(dst);
				src = dst;
				//Mat bblur1;
				//GaussianBlur(cv::abs(lap1), bblur1, Size(3, 3), 0, 0, BORDER_REFLECT);
				////GaussianBlur(cv::abs(lap1), bblur1, Size(41, 41), 0, 0, BORDER_REFLECT);
				////********************************************
				////sprintf_s(name, "H:\\Project\\PCellFusion\\fusion\\%dfusion.jpg", j);
				////cv::imwrite(name, dst);
				//laps.push_back(bblur1);
				////laps.push_back(cv::abs(lap1));
				//
				//scale_maps.push_back(dst);
				//src = dst;
			}//Finish a single Octave
			//Choose the max_map of this Octave layer
			cv::Mat max_value = cv::Mat::zeros(laps[0].rows, laps[0].cols, laps[0].type());
			//cv::Mat max_value = cv::Mat::zeros(scale_maps[0].rows, scale_maps[0].cols, scale_maps[0].type());
			//***************choose the lap layers********************
			for (int mth = 0; mth < laps.size(); mth++) {
				/*cout << "max_value size:"<<max_value.size() << endl;
				cout << "lap size:" << laps[mth].size() << endl;*/
				max_value = cv::max(max_value, laps[mth]);
			}

			//**********choose the blur layers************************
			//for (int i = 0; i<scale_maps.size(); i++) {
			//	max_value = cv::max(max_value, scale_maps[i]);
			//}
			//Mat lap;
			//Mat kernel = (Mat_<float>(3, 3) <<
			//	1, 1, 1,
			//	1, -8, 1,
			//	1, 1, 1);
			//cv::filter2D(max_value, lap, CV_32F, kernel, Point(-1, -1), 0, BORDER_REFLECT);
			//Mat bblur;
			//GaussianBlur(cv::abs(lap), bblur, Size(41,41), 0, 0, BORDER_REFLECT);
			//Laplacian(max_value,lap,CV_32FC1);
			//max_map = bblur;

			 //Constuct the DOG of one Octave
			//for (int k = 0; k < scale_maps.size() - 1; k++) {
			//	Mat diff = scale_maps[k + 1] - scale_maps[k];
			//	feature_maps.push_back(diff);
			//}
			max_maps.push_back(max_value);
			Mat src_copy = src.clone();//The last image of the prior Octave layer
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

//To extract the max value of all the pixels in the Octave of one image 
//void sc::ExtractMax(const std::vector<cv::Mat>&feature_maps, cv::Mat& max_map) {
//	cv::Mat max_value = cv::Mat::zeros(feature_maps[0].rows, feature_maps[0].cols, feature_maps[0].type());
//	for (int i = 0; i<feature_maps.size(); i++) {
//		max_value = cv::max(max_value, feature_maps[i]);
//	}
//	max_map = max_value;
//}