#include"stdafx.h"
#include <iostream>
#include <string>
#include<fstream>
#include <vector>
#include <sstream>
#include <Windows.h>
#include <io.h>
#include<time.h>
#include <direct.h>
#include <opencv2/opencv.hpp>
#include"guidedFilter.h"
#include"Octave.h"
#include"regSaliency.h"


using namespace std;
using namespace cv;


//Read all the images under the same folder
bool FindOrginalImages(const std::string& folder, std::vector<std::string>& validImageFiles, const std::string& format) {
	WIN32_FIND_DATAA findFileData;
	std::string path = folder + "/*." + format;
	HANDLE hFind = FindFirstFileA((LPCSTR)path.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return false;

	do {
		std::string fileName = findFileData.cFileName;
		std::string ext = fileName.substr(fileName.find_last_of('.') + 1, fileName.length());

		if (ext == format) {
			std::string tmpFileName = folder + '\\' + fileName;
			validImageFiles.push_back(tmpFileName);
		}

	} while (FindNextFileA(hFind, &findFileData));

	return true;
}

static void help()
{
	cout << "\n This program demonstrates TCT Image Fusion \n"
		"Usage: \n"
		"  ./TCTImageFusion.exe image_folder image_ext\n";
}

void extract_maps(const std::vector<cv::Mat>&images,std::vector<cv::Mat>&grays) {
	for (int idx = 0; idx<images.size(); idx++) {
		Mat ggray;
		if (images[idx].channels() == 3) {
			cvtColor(images[idx], ggray, CV_RGB2GRAY);
		}
		else {
			ggray = images[idx];
		}
		grays.push_back(ggray);
	}
}

void GetOriginalMask(const std::vector<cv::Mat>&saliency_layers, std::vector<cv::Mat>&original_masks) {
	//Get the original mask of each images
	cv::Mat saliency_max = cv::Mat::zeros(saliency_layers[0].rows, saliency_layers[0].cols, saliency_layers[0].type());
	for (int sdx = 0; sdx<saliency_layers.size(); sdx++) {

		saliency_max = cv::max(saliency_max, saliency_layers[sdx]);
	}
	
	std::vector<cv::Mat>mask_maps;
	for (int mdx = 0; mdx<saliency_layers.size(); mdx++) {
		Mat mask;
		mask = (saliency_layers[mdx] == saliency_max);
		mask_maps.push_back(mask);
	}

	Mat sum_mask = Mat::zeros(mask_maps[0].rows, mask_maps[0].cols, mask_maps[0].type());
	for (int idx = 0; idx<mask_maps.size(); idx++) {

		Mat mask_map = mask_maps[idx].mul(1. / 255);
		sum_mask = sum_mask + mask_map;
	}

	vector<Point>over_mask;//save the location of over value
	for (int k = 0; k<sum_mask.rows; k++) {
		for (int m = 0; m<sum_mask.cols; m++) {
			if (int(sum_mask.at<uchar>(k, m))>1) {
				Point f;
				f.x = m;
				f.y = k;
				over_mask.push_back(f);
			}
		}
	}

		for (int t = 0; t<over_mask.size(); t++) {//For each point
			Point p = over_mask[t];
			for (int w = 0; w<mask_maps.size(); w++) {//For each image
				if (int(mask_maps[w].at<uchar>(p.y, p.x)) == 255) {
					for (int r = w + 1; r<mask_maps.size(); r++) {
						mask_maps[r].at<uchar>(p.y, p.x) = 0;
					}
					break;
				}
			}
		}
		//for (int v = 0; v < mask_maps.size();v++) {
		//	cvtColor(mask_maps[v], mask_maps[v], CV_GRAY2RGB);
		//	mask_maps[v].convertTo(mask_maps[v], CV_32FC3);
		//}
	original_masks = mask_maps;

}

void GuidOptimize(const std::vector<cv::Mat>&images, const string type, const std::vector<cv::Mat>&init_weights, std::vector<cv::Mat>&weights, const int r, const double eps) {
	if (images.size() == 0) {
		return;
	}
	Mat weight;
	for (int i = 0; i<images.size(); i++) {
		Mat filtered, simg;
		images[i].convertTo(simg, CV_32FC3, 1. / 255);
		if (images[i].channels() == 3) {
			std::vector<Mat>schannels;
			split(simg, schannels);
			Mat filtered_b = guidedFilter(schannels[0], init_weights[i], r, eps);
			Mat filtered_g = guidedFilter(schannels[1], init_weights[i], r, eps);
			Mat filtered_r = guidedFilter(schannels[2], init_weights[i], r, eps);
			filtered = filtered_b + filtered_g + filtered_r;
			//filtered = guidedFilter(images[i], init_weights[i], r, eps);
		}
		else {
			filtered = guidedFilter(simg, init_weights[i], r, eps);
		}
		if (type=="color") {
			cvtColor(filtered, weight, CV_GRAY2RGB);
		}
		else {
			weight = filtered;
		}
		
		weight.convertTo(weight, CV_32FC3);
		weights.push_back(weight);
	}

}

void WeightedCombine(const std::vector<cv::Mat>&images, const std::vector<cv::Mat>&weights, cv::Mat& wcimage) {

	Mat sum_image;
	Mat sum_weights;
	if (images[0].channels() == 3) {
		sum_image = Mat::zeros(images[0].rows, images[0].cols, CV_32FC3);
		sum_weights = Mat::zeros(images[0].rows, images[0].cols, CV_32FC3);
	}
	else {
		sum_image = Mat::zeros(images[0].rows, images[0].cols, CV_32F);
		sum_weights = Mat::zeros(images[0].rows, images[0].cols, CV_32F);
	}

	if (images.size() != weights.size()) {
		std::cout << "check the vector sizse of weights";
		return;
	}
	for (int i = 0; i<images.size(); i++) {

		sum_image = sum_image + images[i].mul(weights[i]);

	}

	for (int wdx = 0; wdx<weights.size(); wdx++) {
		sum_weights = sum_weights + weights[wdx];
	}
	wcimage = (sum_image.mul(1./ sum_weights));

}

cv::Mat GFF(const std::vector<cv::Mat>&images,const string type, const int octave_nums, const int layers,const int r_base = 2, const double eps_base =2, const int r_detail = 2, const double eps_detail = 0.1, const int fsize =3, const int sigma = 5) {

	std::vector<cv::Mat> gray_layers;
	std::vector<cv::Mat> base_layers;
	std::vector<cv::Mat> detail_layers;
	std::vector<cv::Mat> saliency_layers;
	std::vector<cv::Mat> original_masks;
	std::vector<cv::Mat> fine_base_masks;
	std::vector<cv::Mat> fine_detail_masks;
	Mat multibase, multidetail, multisum;
	//extract_maps(images, gray_layers,base_layers, detail_layers, saliency_layers);
	extract_maps(images, gray_layers);
	std::vector<cv::Mat> scale_maps;
	std::vector<cv::Mat> feature_maps;
	std::vector<cv::Mat>max_maps;
	std::vector<std::vector<cv::Mat>>images_scale_maps;
	std::vector<std::vector<cv::Mat>>images_features_maps;
	for (int kth = 0; kth < gray_layers.size();kth++) {
		Mat max_map;
		//sc::Octave(gray_layers[kth], scale_maps,feature_maps, max_map,layers,3,1);
		sc::Octave(gray_layers[kth], scale_maps, feature_maps, max_map, layers, octave_nums, 1);
		max_maps.push_back(max_map);
	}
	GetOriginalMask(max_maps, original_masks);
	//********Test the filter with color or gray*******************
	//GuidOptimize(images, type, original_masks, fine_base_masks, r_base, eps_base);
	GuidOptimize(gray_layers, type, original_masks, fine_base_masks, r_base, eps_base);
	//cout << fine_base_masks[0] << endl;
	//imwrite("C:\\Users\\mao_y\\Desktop\\gff.jpg", fine_base_masks);
	//*************************************
	//use the original image to get the final fused image
	WeightedCombine(images, fine_base_masks, multibase);
	
	multibase.convertTo(multibase, CV_8UC3);
	//*******************************************
	return multibase;

}


int main(int argc, char* argv[]) {
	std::vector<std::string> imageFiles;
	if (argc < 3)
	{
		help();
		return -1;
	}
	std::vector<std::string> files_names;
	std::vector<cv::Mat>images;
	std::string img_fold(argv[1]);
	std::string img_format(argv[2]);	
	FindOrginalImages(img_fold, files_names, img_format);
	
	for (int i = 0; i<files_names.size(); i++) {
		cout << "reading " << files_names[i] << endl;
		Mat img_src = cv::imread(files_names[i], CV_LOAD_IMAGE_COLOR);
		//Mat img_src= cv::imread(files_names[i],0);//for gray image.
		Mat img;
		img_src.convertTo(img, CV_32FC3);//for the color image
		//images.push_back(img);
		images.push_back(img_src);//For the registration.
	}
	//imshow("atest",images[0]);
	//**************Test for the image registration**************
	//vector<Mat>matches;
	////double T1 = (double)cv::getTickCount();
	//sc::MatchImg(images,matches);
	//double T2 = (double)cv::getTickCount();
	//cout << "Total time:" << ((double(T2 - T1)) / cv::getTickFrequency()) << endl;
	/*double T1 = (double)cv::getTickCount();*/
	//Mat result;
	//result = GFF(images, "gray",5,3);
	//imwrite("C:\\Users\\mao_y\\Desktop\\Database\\MultilmodalImages\\a\\c01_O5_S3fused.jpg", result);
	/*double T2 = (double)cv::getTickCount();
	cout << "Total time:" << ((double(T2 - T1)) / cv::getTickFrequency()) << endl;
	imwrite("H:\\Dissertation\\new_fusionPapers\\Data\\TestImage\\c05\\P-DoGc05-O1-L2.jpg", result);*/
	/*result = GFF(images, "color", 3, 5);
	imwrite("C:\\Users\\mao_y\\Desktop\\o3_L5fused.jpg", result);*/
	//****************************************************
	//for (int nth = 0; nth < 9;nth++) {
	//	if (nth==0) {
	//		Mat result = GFF(images, "color", nth, 0);
	//		char tag_file_dst[100];
	//		sprintf_s(tag_file_dst, "C:\\Users\\mao_y\\Desktop\\TestResult\\T3\\T3_DoG_O%d_L%d.jpg", 0, 0);
	//		imwrite(tag_file_dst, result);
	//	}
	//	else {
	//		for (int yth = 1; yth < 9;yth++) {
	//			Mat result = GFF(images, "color", nth,yth);
	//			char tag_file_dst[100];
	//			sprintf_s(tag_file_dst, "C:\\Users\\mao_y\\Desktop\\TestResult\\T3\\T3_DoG_O%d_L%d.jpg", nth,yth);
	//			imwrite(tag_file_dst, result);
	//		}
	//	}
	////
	//}

	//********************Test for the detectedpoints***********************************
	std::vector<cv::KeyPoint> keypoints;
	std::vector<DetectedPoint> detectedpoints;
	cv::Mat descriptor;
	mr::MisRegImg Timg(images[0]);
	cout << "hello" << endl;
	Timg.GetTheKeypoints(Timg.g_img,keypoints,detectedpoints,descriptor);
	cout << detectedpoints[0].val<<" "<<detectedpoints[0].dp<<" "<<detectedpoints[0].layer_num<<" "<<detectedpoints[0].octave_num << endl;
	system("pause");
	return 0;
}
