// cellfusion.cpp : 定义控制台应用程序的入口点。
//

#include"stdafx.h"
#include <iostream>
#include <string>
#include<fstream>
#include <vector>
#include <sstream>
#include <Windows.h>
#include <io.h>
#include <direct.h>
#include <opencv2/opencv.hpp>
#include"guidedFilter.h"

using namespace std;
using namespace cv;
//void getFiles(std::string path, std::vector<std::string>& files,const string format)
//{
//	long   hFile = 0;
//	struct _finddata_t fileinfo;
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("/*" + format).c_str(), &fileinfo)) != -1)
//	{
//		do
//		{
//			if ((fileinfo.attrib &  _A_SUBDIR))
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//					getFiles(p.assign(path).append("/").append(fileinfo.name), files, format);
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("/").append(fileinfo.name));
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//	//for(int i=0;i<files.size();i++){
//	//	cout<<files[i]<<endl;
//	//}
//}

bool FindOrginalImages(const std::string& folder, std::vector<std::string>& validImageFiles, const std::string& Ext)
{
	WIN32_FIND_DATAA findFileData;
	std::string path = folder + "/*." + Ext;
	HANDLE hFind = FindFirstFileA((LPCSTR)path.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return false;

	do
	{
		std::string fileName = findFileData.cFileName;
		std::string ext = fileName.substr(fileName.find_last_of('.') + 1, fileName.length());

		if (ext == Ext)
		{
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

void normalize(std::vector<cv::Mat>&images) {
	if (images.size() == 0) {
		return;
	}
	cv::Mat images_sum = cv::Mat::zeros(images[0].rows, images[0].cols, images[0].type());
	for (int i = 0; i<images.size(); i++) {
		images_sum = images_sum + images[i];
	}
	for (int i = 0; i<images.size(); i++) {
		images[i] = images[i].mul(1 / images_sum);
	}
}

//void GetSaliencyMap(const std::vector<cv::Mat>&images, std::vector<cv::Mat>& saliency_maps) {
//
//	for (int i = 0; i<images.size(); i++) {
//		Mat lmap, smap;// , gray;
//					   //if (images[i].channels()>=3) {
//					   //	cvtColor(images[i],gray,CV_RGB2GRAY);
//					   //}
//					   //else {
//					   //	gray = images[i]-0.0001;
//					   //}
//					   //gray.convertTo(gray, CV_32F, 1.0 / 255);
//		Laplacian(images[i], lmap, CV_32F);
//		GaussianBlur(cv::abs(lmap), smap, Size(5, 5), 0);
//		saliency_maps.push_back(smap);//change the size from (5,5) to (3,3)
//	}
//
//}

void GetMaskMap(const std::vector<cv::Mat>& saliency_maps, std::vector<cv::Mat>& mask_maps) {
	cv::Mat saliency_max = cv::Mat::zeros(saliency_maps[0].rows, saliency_maps[0].cols, CV_32F);

	for (int i = 0; i<saliency_maps.size(); i++) {
		saliency_max = cv::max(saliency_max, saliency_maps[i]);
	}
	//cout << saliency_max << endl;
	for (int j = 0; j<saliency_maps.size(); j++) {
		Mat mask;
		mask = (saliency_maps[j] >= saliency_max);//change from ">=" to "=="											  //mask = mask.mul(1. / 255)											  //cout << mask << endl;
		mask_maps.push_back(mask);
	}
}

void GuidOptimize(const std::vector<cv::Mat>&images, const std::vector<cv::Mat>&init_weights, std::vector<cv::Mat>&weights, const int r, const double eps) {
	if (images.size() == 0) {
		return;
	}
	weights.clear();
	for (int i = 0; i<images.size(); i++) {
		Mat filtered;

		//images[i].convertTo(timg,images[i].type(),1./255);
		filtered = guidedFilter(images[i], init_weights[i], r, eps);
		filtered.convertTo(filtered, CV_32F, 1.0 / 255);
		//filtered.convertTo(filtered, CV_32F, 1.0 / 255);
		//if(images[i].channels()==3){
		//	cvtColor(filtered,filtered,CV_GRAY2RGB);
		//}
		weights.push_back(filtered);
	}

	normalize(weights);

}

void WeightedCombine(const std::vector<cv::Mat>&images, const std::vector<cv::Mat>&weights, cv::Mat& wcimage) {

	Mat sum_image;
	if (images[0].channels() == 3) {
		sum_image = Mat::zeros(images[0].rows, images[0].cols, CV_32FC3);
	}
	else {
		sum_image = Mat::zeros(images[0].rows, images[0].cols, CV_32F);
	}

	if (images.size() != weights.size()) {
		std::cout << "check the vector sizse of weights";
		return;
	}
	for (int i = 0; i<images.size(); i++) {
		//Mat m_image;
		//m_image = images[i].mul(weights[i]);
		sum_image = sum_image + images[i].mul(weights[i]);

	}
	wcimage = sum_image;
}


int main(int argc, char* argv[]) {
	//std::vector<std::string> imageFiles;
	//if (argc < 3) {
	//	help();
	//	return -1;
	//}

	//std::string img_fold(argv[1]);
	//std::string img_ext(argv[2]);

	std::string img_fold = "H:\\Project\\PCellFusion\\Data\\1-200";
	std::string img_ext = "bmp";
	std::vector<std::string> files_names;
	FindOrginalImages(img_fold, files_names, img_ext);
	int r1 = 5, r2 = 5;
	//int r1 = 15, r2 = 7;
	double eps1 = 2, eps2 = 0.1;

	//Read the images from files
	std::vector<cv::Mat>images;
	std::vector<cv::Mat>grays;
	std::vector<cv::Mat>base_images;
	std::vector<cv::Mat>detail_images;
	std::vector<cv::Mat>saliency_maps;
	
	/////////////////
	cout << "runinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn" << endl;
	//////////////////
	//To pre-operate the readed images[gray,base,detail images]
	//for(auto filename : files_names ){
	for (int idx = 0; idx<files_names.size(); idx++) {
		//std::string ext = filename.substr(filename.find_last_of('.') + 1);
		//if (ext != img_ext) continue;//Judge the suffix of the file
		cout << "reading " << files_names[idx] << endl;
		const cv::Mat image = cv::imread(files_names[idx], CV_LOAD_IMAGE_COLOR);
		//image.convertTo(image,CV_32FC3,1./255);
		Mat gray, saliency_map;
		cvtColor(image, gray, CV_RGB2GRAY);
		//cout << gray << endl;
		gray.convertTo(gray, CV_32F, 1.0 / 255);
		Laplacian(gray, saliency_map, CV_32F);
		GaussianBlur(cv::abs(saliency_map), saliency_map, Size(5, 5), 0);
		//decompose the image into two scales,one for base,the other for detail
		Mat base, detail;
		cout << image.channels() << endl;
		if (image.channels() == 3) {

			base = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
			detail = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
		}
		else {
			base = cv::Mat::zeros(image.rows, image.cols, CV_32F);
			detail = cv::Mat::zeros(image.rows, image.cols, CV_32F);
		}

		//boxFilter(image, base, image.depth(), Size(31, 31));
		GaussianBlur(image, base, Size(15, 15), 0);
		addWeighted(image, 1.0, base, -1.0, 0, detail);
		base.convertTo(base, CV_32FC3, 1. / 255);//Make the guided image to be three channels
												 //detail=image-base;
		detail.convertTo(detail, CV_32FC3, 1. / 255);//Make the guided image to be three channels



		base_images.push_back(base);
		detail_images.push_back(detail);
		images.push_back(image);
		grays.push_back(gray);
		saliency_maps.push_back(saliency_map);
		////////////////
		cout << "ssssssssssssssssssssssss" << endl;
		////////////////
	}//end for 


	double start = (double)cv::getTickCount();
	//Step1:Get the saliency maps of each image
	//std::vector<cv::Mat>saliency_maps;
	//GetSaliencyMap(grays, saliency_maps);
	cout << "saliency" << saliency_maps[0].at<float>(5, 5) << "deee" << saliency_maps[3].at<float>(5, 5) << endl;
	cout << "QQQQQQQQQQQQQQQQQq" << endl;

	//
	cout << "Step1:Get the saliency maps of each image FINISHED" << endl;
	//
	//Step2:Get the weight mask of each image
	std::vector<cv::Mat>mask_maps;
	GetMaskMap(saliency_maps, mask_maps);
	//
	cout << "Step2:Get the weight mask of each image FINISHED" << endl;
	//
	//Step3:Use the guided filter to the mask_maps
	std::vector<cv::Mat> weight_maps_bases, weight_maps_details;
	std::vector<cv::Mat> weight_bases;
	std::vector<cv::Mat> weight_details;


	GuidOptimize(images, mask_maps, weight_maps_bases, r1, eps1);
	if (images[0].channels() == 3) {
		
		for (int bdx = 0; bdx<images.size(); bdx++) {
			Mat weight_base;
			cvtColor(weight_maps_bases[bdx], weight_base, CV_GRAY2RGB);
			weight_bases.push_back(weight_base);
		}
	}
	else {
		weight_bases = weight_maps_bases;
	}

	GuidOptimize(images, mask_maps, weight_maps_details, r2, eps2);
	if (images[0].channels() == 3) {
		
		for (int ddx = 0; ddx<images.size(); ddx++) {
			Mat weight_detail;
			cvtColor(weight_maps_details[ddx], weight_detail, CV_GRAY2RGB);
			weight_details.push_back(weight_detail);
		}
	}
	else {
		weight_details = weight_maps_details;
	}
	//
	cout << "Step3:Use the guided filter to the mask_maps FINISHED" << endl;
	//

	//Step4:Use the weight maps to the original images and combine all the images into a fused map
	//cout << "base_images size:" << base_images[0].type() << endl;//<<base_images[0].cols<<endl;
	//cout << "weight_bases size:" << weight_bases[0].type() << endl;//rows<<weight_bases[0].cols<<endl;
	//cout << "detail_images size:" << detail_images[0].type() << endl;//rows<<detail_images[0].cols<<endl;
	//cout << "weight_details size:" << weight_details[0].type() << endl;//rows<<weight_details[0].cols<<endl;
	Mat base_wcimage, detail_wcimage, fused_image;
	WeightedCombine(base_images, weight_bases, base_wcimage);
	WeightedCombine(detail_images, weight_details, detail_wcimage);
	fused_image = base_wcimage + detail_wcimage;
	fused_image.convertTo(fused_image, CV_8UC3, 255);
	cout << "ddddd:" << fused_image.rows << fused_image.cols << endl;
	//
	cout << "Step4:Use the weight maps to the original images and combine all the images into a fused map FINISHED" << endl;
	//

	double end = (double)cv::getTickCount();

	cout << "Time in seconds:" << (end - start) / cv::getTickFrequency() << endl;
	//Step5:Save the fused image
	//std::string result_filename = "H:\\Project\\PCellFusion\\Data\\5-250\\gff.jpg";
	//std::string result_filename=files_names[0].substr(0,files_names[0].find_last_of('.'))+"gff.jpg";
	std::string result_filename = img_fold + "\\" + "gff.jpg";
	cv::imwrite(result_filename, fused_image);
	system("pause");
	return 0;
}