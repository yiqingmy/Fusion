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

void extract_maps(const std::vector<cv::Mat>&images,std::vector<cv::Mat>&grays,std::vector<cv::Mat>&base_layers, std::vector<cv::Mat>&detail_layers, std::vector<cv::Mat>&saliency_layers, const int fsize = 31, const int sigma = 5) {
	for (int idx = 0; idx<images.size(); idx++) {
		Mat ggray;
		if (images[idx].channels() == 3) {
			cvtColor(images[idx], ggray, CV_RGB2GRAY);
		}
		else {
			ggray = images[idx];
		}
		//*****display the gray images***
		char g_name[100];
		sprintf_s(g_name,"C:\\Users\\mao_y\\Desktop\\gray_images\\%dgimg.jpg",idx);
		cv::imwrite(g_name, ggray);
		//*******************************
		grays.push_back(ggray);
		Mat kernel = (Mat_<float>(3,3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);
		Mat base, detail;
		GaussianBlur(images[idx], base, Size(3, 3), 0);
		detail = images[idx] - base;
		base.convertTo(base, CV_32FC3);
		detail.convertTo(detail, CV_32FC3);
		base_layers.push_back(base);
		detail_layers.push_back(detail);


		Mat gray, saliency, saliency_map;
		if (images[idx].channels() >= 3) {

			cvtColor(images[idx], gray, CV_RGB2GRAY);
		}
		else {
			Mat t_map;
			images[idx].convertTo(t_map, CV_32F);
			gray = t_map - 0.0001;
		}
		
		cv::filter2D(gray,saliency,CV_32F,kernel,Point(-1,-1),0,BORDER_REFLECT);

		GaussianBlur(cv::abs(saliency), saliency_map, Size(41, 41), 0,0,BORDER_REFLECT);
		char name[100];
		sprintf_s(name,"H:\\Project\\PCellFusion\\saliency\\%dth.jpg",idx);
		imwrite(name,saliency_map);
		saliency_map.convertTo(saliency, CV_32F);
		saliency_layers.push_back(saliency_map);


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

cv::Mat GFF(const std::vector<cv::Mat>&images,const string type, const int r_base = 2, const double eps_base =2, const int r_detail = 2, const double eps_detail = 0.1, const int fsize =3, const int sigma = 5) {

	std::vector<cv::Mat> gray_layers;
	std::vector<cv::Mat> base_layers;
	std::vector<cv::Mat> detail_layers;
	std::vector<cv::Mat> saliency_layers;
	std::vector<cv::Mat> original_masks;
	std::vector<cv::Mat> fine_base_masks;
	std::vector<cv::Mat> fine_detail_masks;
	Mat multibase, multidetail, multisum;

	//double t1 = (double)cv::getTickCount();

	//double time1 = (double)cv::getTickCount();
	extract_maps(images, gray_layers,base_layers, detail_layers, saliency_layers);
	//double time2 = (double)cv::getTickCount();
	//cout << "extract map ending" << endl;
	//std::cout << " Time1 in seconds: " <<
	//((double(time2-time1)) / cv::getTickFrequency() )<<endl;

	//double time3 = (double)cv::getTickCount();

	//*******************************************************
	
	std::vector<cv::Mat> scale_maps;
	std::vector<cv::Mat> feature_maps;
	std::vector<cv::Mat>max_maps;
	std::vector<std::vector<cv::Mat>>images_scale_maps;
	std::vector<std::vector<cv::Mat>>images_features_maps;
	int layers =5;
	for (int kth = 0; kth < gray_layers.size();kth++) {
		Mat max_map;
		sc::Octave(gray_layers[kth], scale_maps,feature_maps, max_map,layers,3,1);
		char name[100];
		sprintf_s(name,"C:\\Users\\mao_y\\Desktop\\saliency_images\\%dsimg.jpg",kth);
		cv::imwrite(name,max_map*255);
		max_maps.push_back(max_map);
		images_scale_maps.push_back(scale_maps);
		images_features_maps.push_back(feature_maps);
	}
	
	GetOriginalMask(max_maps, original_masks);
	//***************Extend the channels if the original images are colorful**
	vector<Mat>c_omasks;
	//*************Resize the image for the mask************
	vector<Mat>rmc_omasks;
	//******************************************************
	//********************Resize the mask*******************************
		for (int om_index = 0; om_index < original_masks.size(); om_index++) {
			Mat rmc_omask;
			cv::resize(original_masks[om_index], rmc_omask, Size(original_masks[om_index].cols * 2, original_masks[om_index].rows * 2));
			rmc_omasks.push_back(rmc_omask);
		}
		//******************************************************************
		vector<Mat>resize_mc_omasks;
	if (images[0].channels()==3) {
		for (int om_index = 0; om_index < original_masks.size();om_index++) {
			Mat c_omask;
			Mat mc_omask;
			cvtColor(original_masks[om_index], c_omask,CV_GRAY2RGB);
			c_omask.convertTo(c_omask,CV_32F);
			c_omasks.push_back(c_omask);
			//Resize the image for mask
			cvtColor(rmc_omasks[om_index], mc_omask, CV_GRAY2RGB);
			mc_omask.convertTo(mc_omask, CV_32F);
			resize_mc_omasks.push_back(mc_omask);
		}		
	}

		//for () {
		//
		//}
	
	//************************************************************************
	////***************display the original mask*****************
	for (int mi = 0; mi < original_masks.size(); mi++) {
		char name[100];
		sprintf_s(name, "C:\\Users\\mao_y\\Desktop\\o_mask\\%dmask.jpg", mi);
		Mat mul_image = original_masks[mi] * 255;
		cv::imwrite(name, mul_image);
	}
	////*******************************************************
	//********************************************************
	//GetOriginalMask(saliency_layers, original_masks);
	//double time4 = (double)cv::getTickCount();
	//cout << "GetOriginalMask ending" << endl;
	//std::cout << " Time2 in seconds: " <<
	//((double(time4-time3)) / cv::getTickFrequency() )<<endl;

	//double time5 = (double)cv::getTickCount();	

	//*********************Without guided filter*********************************
	//GuidOptimize(images, type, original_masks, fine_base_masks, r_base, eps_base);
	//***************************************************************************
	//********Resize the image for mask**********
	vector<Mat>n_images;
	vector<Mat>n_gray_images;
	for (int r = 0; r < images.size(); r++) {
		Mat n_image;
		Mat n_gray_layer;
		cv::resize(images[r], n_image, Size(images[r].cols * 2, images[r].rows * 2));
		cv::resize(gray_layers[r], n_gray_layer, Size(gray_layers[r].cols * 2, gray_layers[r].rows * 2));
		n_images.push_back(n_image);
		n_gray_images.push_back(n_gray_layer);
	}
	GuidOptimize(n_images, type, rmc_omasks, fine_base_masks, r_base, eps_base);
	//*************
	//GuidOptimize(gray_layers, type, original_masks, fine_base_masks, r_base, eps_base);
	//////******************display the filtered mask*************************************
	//for (int bm = 0; bm < fine_base_masks.size(); bm++) {
	//	char name[100];
	//	sprintf_s(name, "C:\\Users\\mao_y\\Desktop\\f_mask\\%df_mask.jpg", bm);
	//	Mat mul_image = fine_base_masks[bm] * 255;
	//	cv::imwrite(name, mul_image);
	//}
	//////*******************************************************
	//GuidOptimize(gray_layers, original_masks, fine_detail_masks, r_detail, eps_detail,"color");
	//double time6 = (double)cv::getTickCount();
	//cout << "GuidOptimize ending" << endl;
	//std::cout << " Time3 in seconds: " <<
	//((double(time6-time5)) / cv::getTickFrequency() )<<endl;

	//double time7 = (double)cv::getTickCount();
	//*************************************
	//use the original image to get the final fused image
	//************Use the original mask*******************

	WeightedCombine(n_images, fine_base_masks, multibase);
	//WeightedCombine(images, fine_base_masks, multibase);
	//WeightedCombine(images, c_omasks, multibase);
	multibase.convertTo(multibase, CV_8UC3);
	//*******************************************
	//WeightedCombine(base_layers, fine_base_masks, multibase);
	//WeightedCombine(detail_layers, fine_detail_masks, multidetail);
	//double time8 = (double)cv::getTickCount();
	//cout << "WeightedCombine ending" << endl;

	/*multisum = multibase + multidetail;
	multisum.convertTo(multisum, CV_8UC3);*/
	//std::cout << " Time4 in seconds: " <<
	//((double(time8-time7)) / cv::getTickFrequency() )<<endl;
	
	//double t2 = (double)cv::getTickCount();

	//std::cout << " Total time in seconds: " <<
	//((double(t2-t1)) / cv::getTickFrequency() )<<endl;
	return multibase;
	//return multisum;
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
	//****************************************************************************
	//double T1 = (double)cv::getTickCount();
	////To fuse two images
	//Mat image1 = imread("C:\\Users\\mao_y\\Desktop\\evaluation\\data\\source\\a02\\a02_1.tif");
	////cout << "channel:" << image1.channels() << endl;
	//Mat img1;
	////image1.convertTo(img1, CV_32F);
	//cvtColor(image1,img1,CV_RGB2GRAY);
	//img1.convertTo(img1, CV_32F);
	//images.push_back(img1);
	//Mat image2 = imread("C:\\Users\\mao_y\\Desktop\\evaluation\\data\\source\\a02\\a02_2.tif");
	//Mat img2;
	////image2.convertTo(img2, CV_32F);
	//cvtColor(image2, img2, CV_RGB2GRAY);
	//img2.convertTo(img2, CV_32F);
	//images.push_back(img2);
	//Mat result = GFF(images,"gray");
	//double T2 = (double)cv::getTickCount();
	//cout<<"Total time:"<< ((double(T2 - T1)) / cv::getTickFrequency()) << endl;
	//imwrite("C:\\Users\\mao_y\\Desktop\\proposedR\\3Obooks1.jpg", result);
	//******************************************************************************
	FindOrginalImages(img_fold, files_names, img_format);
	for (int i = 0; i<files_names.size(); i++) {
		cout << "reading " << files_names[i] << endl;
		Mat img_src = cv::imread(files_names[i], CV_LOAD_IMAGE_COLOR);
		Mat img;
		img_src.convertTo(img, CV_32FC3);//for the color image
		//cvtColor(img_src,img,CV_RGB2GRAY);%for the gray image
		//img.convertTo(img,CV_32FC3);
		////*************************Resize the images(down-sampled)********************************
		Mat down_img;
		cv::resize(img, down_img, Size(img.cols / 2, img.rows / 2));
		images.push_back(down_img);
		//**************************************************************************
		//images.push_back(img);
	}
	Mat result = GFF(images,"color");
	//*****************Resize the images(up-sampled)********************************
	//Mat up_img;
	//cv::resize(result, up_img, Size(result.cols * 2, result.rows * 2));
	//imwrite("C:\\Users\\mao_y\\Desktop\\2t_color_gff_fusion.jpg", up_img);
	//*******************************************************************************
	imwrite("C:\\Users\\mao_y\\Desktop\\mask_resize_colorgff_fused.jpg",result);
	//std::string result_filename = img_fold + "\\" + "3O_fused.jpg";
	//std::string result_filename = img_fold + "\\" + "3O_5L_GCOLOR_fused.jpg";
	//cv::imwrite(result_filename, result);
	//imwrite("C:\\Users\\mao_y\\Desktop\\resized_nogff_fused.jpg", result);
	system("pause");
	return 0;
}