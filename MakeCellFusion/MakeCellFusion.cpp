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

void extract_maps(const std::vector<cv::Mat>&images, std::vector<cv::Mat>&gray_layers,std::vector<cv::Mat>&base_layers, std::vector<cv::Mat>&detail_layers, std::vector<cv::Mat>&saliency_layers, const int fsize = 31, const int sigma = 5) {
	for (int idx = 0; idx<images.size(); idx++) {
		//Mat ggray;//************************************
		//cvtColor(images[idx],ggray,CV_RGB2GRAY);
		//gray_layers.push_back(ggray);
		//Get the base & detail maps
		//*****************************8
		Mat kernel = (Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);
		//******************************
		Mat base, detail;
		//cv::filter2D(images[idx], detail, CV_32F, kernel, Point(-1, -1), 0, BORDER_REFLECT);
		//Mat bbase, ddetail;
		GaussianBlur(images[idx], base, Size(3, 3), 0);
		//blur(ggray, bbase, Size(fsize, fsize));
		//blur(images[idx], base, Size(fsize, fsize));
		detail = images[idx] - base;

		//cvtColor(bbase,base,CV_GRAY2RGB);
		//cvtColor(ddetail, detail, CV_GRAY2RGB);

		base.convertTo(base, CV_32FC3);
		detail.convertTo(detail, CV_32FC3);
		base_layers.push_back(base);
		detail_layers.push_back(detail);

		//Get the saliency maps
		Mat  gray,saliency, saliency_map;
		//if (images[idx].channels() >= 3) {

		//	cvtColor(images[idx], gray, CV_RGB2GRAY);
		//}
		//else {
		//	Mat t_map;
		//	images[idx].convertTo(t_map, CV_32F);
		//	gray = t_map - 0.0001;
		//}
		/*Laplacian(gray, saliency, CV_32F);
		GaussianBlur(cv::abs(saliency), saliency_map, Size(sigma, sigma), 0);*/
		//************************88

		cv::filter2D(gray, saliency, CV_32F, kernel, Point(-1, -1), 0, BORDER_REFLECT);
		//**************************
		//Laplacian(gray, saliency, CV_32F,1,1,0,BORDER_REFLECT);
		GaussianBlur(cv::abs(saliency), saliency_map, Size(41, 41), 0, 0, BORDER_REFLECT);
		saliency_map.convertTo(saliency, CV_32F);
		saliency_layers.push_back(saliency_map);


	}
}

void GetOriginalMask(const std::vector<cv::Mat>&saliency_layers, std::vector<cv::Mat>&original_masks) {
	//Get the original mask of each images
	cv::Mat saliency_max = cv::Mat::zeros(saliency_layers[0].rows, saliency_layers[0].cols, saliency_layers[0].type());

	//Get the saliency_max which means the max value of each position
	//srand((unsigned)time(NULL));
	for (int sdx = 0; sdx<saliency_layers.size(); sdx++) {
		//Mat s = saliency_layers[sdx];
		//s.convertTo(saliency_layers[sdx],CV_32FC);
		//float mm= (rand() / float(RAND_MAX));
		//cout << "The value of mm:" << mm << endl;
		//Mat b=Mat(saliency_layers[sdx].rows, saliency_layers[sdx].cols, CV_32FC1,Scalar(mm));
		//Mat c = saliency_layers[sdx]+b;
		//cout << "Rand value:" <<c << endl;
		//saliency_max = cv::max(saliency_max, c);
		saliency_max = cv::max(saliency_max, saliency_layers[sdx]);
	}

	//cout << "saliency max value:"<<saliency_max << endl;
	//cout << "saliency max value:" << saliency_max.at<float>(1,0) << endl;
	//cout << "saliency  value:" << saliency_layers[0].at<float>(0, 0) << endl;
	//Get the mask of each salilency map which belongs to 0 or 255
	std::vector<cv::Mat>mask_maps;
	for (int mdx = 0; mdx<saliency_layers.size(); mdx++) {
		Mat mask;
		mask = (saliency_layers[mdx] == saliency_max);
		mask_maps.push_back(mask);
	}
	//cout << "mask_maps value:" << mask_maps[0] << endl;
	//*************************8
	//saliency_max.convertTo(saliency_max, CV_32FC1);
	//cout << "saliency max value:" << saliency_max.at<float>(0, 0) << endl;
	//Mat qq = saliency_layers[0];
	//qq.convertTo(qq, CV_32FC1);
	//cout << "saliency  value:" << qq.at<float>(0, 0) << endl;
	//********************************8
	//cout << "mask value:" <<int( mask_maps[0].at<uchar>(0,0)) << endl;
	//cout << "mask value:" << mask_maps[1] << endl;
	////Find the max pixel of each map
	Mat sum_mask = Mat::zeros(mask_maps[0].rows, mask_maps[0].cols, mask_maps[0].type());
	//cout << "mask data type:" << mask_maps[0].type() << endl;
	for (int idx = 0; idx<mask_maps.size(); idx++) {
		//mask_maps[idx]=mask_maps[idx].mul(1./255);
		//sum_mask=sum_mask+mask_maps[idx];
		Mat mask_map = mask_maps[idx].mul(1. / 255);
		sum_mask = sum_mask + mask_map;
	}
	//cout << "sum_mask value:" << sum_mask << endl;
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
	//cout << "over point:" << over_mask[0] << endl;
	//for (int jdx = 0; jdx<saliency_layers.size(); jdx++) {//For each image
	for (int t = 0; t<over_mask.size(); t++) {//For each point
		Point p = over_mask[t];
		for (int w = 0; w<mask_maps.size(); w++) {//For each image
			if (int(mask_maps[w].at<uchar>(p.y, p.x)) == 255) {
				for (int r = w + 1; r<mask_maps.size(); r++) {
					mask_maps[r].at<uchar>(p.y, p.x) = 0;
				}
				//mask_maps[w].at<uchar>(r.y, r.x) = 0;
				break;
			}
			//mask_maps[w].at<uchar>(r.y, r.x) = 0;
		}
		//if (int(mask_maps[jdx].at<uchar>(r.y, r.x)) == 255){
		//if (mask_maps[jdx].at<uchar>(r.y, r.x) == 255) {
		//for (int w = jdx + 1; w<saliency_layers.size(); w++) {
		//	mask_maps[w].at<uchar>(r.y, r.x) = 0;
		//}
	}
	//}
	//}
	//Mat sumall = Mat::zeros(mask_maps[0].rows, mask_maps[0].cols, mask_maps[0].type());
	//for (int kk = 0; kk < mask_maps.size();kk++) {
	//	;
	//	sumall = sumall + mask_maps[kk].mul(1. / 255);
	//}
	//cout<<"mask all::::" << sumall << endl;
	//Test**********************************************************************
	//Mat sumall=Mat::zeros(mask_maps[0].rows,mask_maps[0].cols,mask_maps[0].type());
	//int num=0;
	//int rnum;
	//std::vector<int>tags;
	//	for(int r=0;r<mask_maps[0].rows;r++){
	//		for(int c=0;c<mask_maps[0].cols;c++){
	//			for(int b=0;b<saliency_layers.size();b++){
	//				if(mask_maps[b].at<uchar>(r,c)==255){
	//					num++;
	//				}				
	//			}
	//			if(num==1){
	//				rnum=1;
	//			}
	//			else{
	//				rnum=0;
	//			}
	//			tags.push_back(rnum);
	//			num=0;
	//		}
	//	}
	//	int sum=0;
	//	for(int m=0;m<tags.size();m++){
	//		sum=sum+tags[m];
	//	}
	//	cout<<sum<<endl;
	//	if(sum==tags.size()){
	//		cout<<"right"<<endl;
	//	}
	//	else{
	//		cout<<"error"<<endl;
	//	}
	//*******************************************************************************	



	//imwrite("D:\\cell\\data\\1-250\\combine.jpg",sumall);
	//cout<<abs(mask_maps[mask_maps.size()-1])<<endl;

	original_masks = mask_maps;
	//cout << "original mask channel:" << original_masks[0].channels() << endl;
	//cout<<(original_masks==mask_maps)<<endl;
}

void GuidOptimize(const std::vector<cv::Mat>&images, const std::vector<cv::Mat>&init_weights, std::vector<cv::Mat>&weights, const int r, const double eps) {
	if (images.size() == 0) {
		return;
	}
	//weights.clear();
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
		//cout << "filter channels:" << filtered.channels() << endl;
		Mat weight;
		//if (images[i].channels() == 3) {
			//cout <<"filter channels:" <<filtered.channels() << endl;
			cvtColor(filtered, weight, CV_GRAY2RGB);
			weight.convertTo(weight, CV_32FC3);
		//}
	/*	else {
			weight = filtered;
			weight.convertTo(weight, CV_32F);
		}*/

		weights.push_back(weight);
	}

	//normalize(weights);

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
		//Mat m_image;
		//m_image = images[i].mul(weights[i]);
		sum_image = sum_image + images[i].mul(weights[i]);

	}

	for (int wdx = 0; wdx<weights.size(); wdx++) {
		sum_weights = sum_weights + weights[wdx];
	}
	wcimage = (sum_image.mul(1. / sum_weights));
	//wcimage = sum_image;
}

cv::Mat GFF(const std::vector<cv::Mat>&images, const int r_base = 2, const double eps_base = 2, const int r_detail = 2, const double eps_detail = 0.1, const int fsize = 3, const int sigma = 5) {
	std::vector<cv::Mat> gray_layers;
	std::vector<cv::Mat> base_layers;
	std::vector<cv::Mat> detail_layers;
	std::vector<cv::Mat> saliency_layers;
	std::vector<cv::Mat> original_masks;
	std::vector<cv::Mat> fine_base_masks;
	std::vector<cv::Mat> fine_detail_masks;
	Mat multibase, multidetail, multisum;

	double time1 = (double)cv::getTickCount();

	double t1 = (double)cv::getTickCount();

	extract_maps(images, gray_layers,base_layers, detail_layers, saliency_layers);

	double t2 = (double)cv::getTickCount();

	cout << "extract map ending" << endl;
	std::cout << " Time1 in seconds: " << ((double(t2 - t1)) / cv::getTickFrequency()) << endl;
		
	//*********************************
	//std::string fpath = "C:\\Users\\mao_y\\Desktop\\saliency_layer";
	//saliency_layers.clear();
	//std::vector<string> files;
	//FindOrginalImages(fpath, files, "jpg");
	//for (int bb = 0; bb < files.size();bb++) {
	//	cout << files[bb] << endl;
	//	Mat mimg = imread(files[bb], 0);
	//	//mimg.convertTo(mimg,CV_32FC1);
	//	saliency_layers.push_back(mimg);
	//}
	//cout << "saliency value:" << saliency_layers[0] << endl;
	//cout << "tsaliency channel:" << saliency_layers[0].channels() << endl;
	//**********************************
	double t3 = (double)cv::getTickCount();
	GetOriginalMask(saliency_layers, original_masks);
	double t4 = (double)cv::getTickCount();
	
	//for (int ndx = 0; ndx < original_masks.size();ndx++) {
	//	cout << "The pixels value:" << int(original_masks[ndx].at<uchar>(1, 0)) << endl;
	//}
	//
	cout << "GetOriginalMask ending" << endl;
	std::cout << " Time2 in seconds: " << ((double(t4 - t3)) / cv::getTickFrequency()) << endl;
	//vector<Mat>tmaps;
	//for(int m=0;m<original_masks.size();m++){
	//	Mat tmap;
	//	cvtColor(original_masks[m],tmap,CV_GRAY2BGR);
	//	tmap.convertTo(tmap,CV_32FC3);
	//	tmaps.push_back(tmap);
	//}
	//***************************************************************
	//std::string mpath = "C:\\Users\\mao_y\\Desktop\\mask_layer";
	//original_masks.clear();
	//std::vector<string> mfiles;
	//FindOrginalImages(mpath, mfiles, "jpg");
	//for (int bb = 0; bb < mfiles.size();bb++) {
	//	cout << mfiles[bb] << endl;
	//	Mat mimg = imread(mfiles[bb],0);
	//	original_masks.push_back(mimg);
	//}



	//****************************************************************
	double t5 = (double)cv::getTickCount();
	//************************88

	GuidOptimize(gray_layers, original_masks, fine_base_masks, r_base, eps_base);
	GuidOptimize(gray_layers, original_masks, fine_detail_masks, r_detail, eps_detail);
	//*******************************
	/*GuidOptimize(images, original_masks, fine_base_masks, r_base, eps_base);
	GuidOptimize(images, original_masks, fine_detail_masks, r_detail, eps_detail);*/
	double t6 = (double)cv::getTickCount();
	cout << "GuidOptimize ending" << endl;
	std::cout << " Time3 in seconds: " << ((double(t6 - t5)) / cv::getTickFrequency()) << endl;
	//vector<Mat>mimages;
	//for(int n=0;n<images.size();n++){
	//	Mat mimage;
	//	images[n].convertTo(mimage,CV_32FC3);
	//	mimages.push_back(mimage);
	//}

	//WeightedCombine(mimages,tmaps,multibase);
	//******************************************************
	//base_layers.clear();
	//detail_layers.clear();
	//std::string bpath = "C:\\Users\\mao_y\\Desktop\\base_layer";
	//std::vector<string> base_files;
	//FindOrginalImages(bpath, base_files, "jpg");
	//for (int ff = 0; ff < base_files.size(); ff++) {
	//	cout << base_files[ff] << endl;
	//	Mat bimg = imread(base_files[ff],1);
	//	bimg.convertTo(bimg,CV_32FC3);
	//	base_layers.push_back(bimg);
	//}
	//std::string dpath = "C:\\Users\\mao_y\\Desktop\\detail_layer";
	//std::vector<string> detail_files;
	//FindOrginalImages(dpath, detail_files, "jpg");
	//for (int dd = 0; dd < detail_files.size(); dd++) {
	//	cout << detail_files[dd] << endl;
	//	Mat dimg = imread(detail_files[dd],1);
	//	dimg.convertTo(dimg,CV_32FC3);
	//	detail_layers.push_back(dimg);
	//}
	//******************************************************
	//cout << "base_layers size:" <<base_layers.size()<< endl;
	//cout << "fine_base_masks size:" << fine_base_masks.size() << endl;
	//cout << "detail_layers size:" << detail_layers .size()<<endl;
	//cout << "fine_detail_masks size:" << fine_detail_masks.size() << endl;
	////**********************************************************
	//std::string wpath = "C:\\Users\\mao_y\\Desktop\\base_weight";
	//fine_base_masks.clear();
	//std::vector<string> wfiles;
	//FindOrginalImages(wpath, wfiles, "jpg");
	//for (int ww = 0; ww < wfiles.size(); ww++) {
	//	cout << wfiles[ww] << endl;
	//	Mat wimg = imread(wfiles[ww], 0);
	//	cvtColor(wimg, wimg, CV_GRAY2RGB);
	//	wimg.convertTo(wimg, CV_32FC3);
	//	fine_base_masks.push_back(wimg);
	//}
	////**********************************************************
	double t7 = (double)cv::getTickCount();
	//**********************8
	//vector<Mat>fbms;
	//vector<Mat>fdms;
	//for (int r = 0; r < images.size();r++) {
	//	Mat fbm, fdm;
	//	cvtColor(fine_base_masks[r], fbm, CV_GRAY2RGB);
	//	cvtColor(fine_detail_masks[r], fdm, CV_GRAY2RGB);
	//	fbms.push_back(fbm);
	//	fdms.push_back(fdm);
	//}
	//
	
	
	
	/*
	WeightedCombine(base_layers, fbms, multibase);
	WeightedCombine(detail_layers, fdms, multidetail);*/
	//******************************
	WeightedCombine(base_layers, fine_base_masks, multibase);
	WeightedCombine(detail_layers, fine_detail_masks, multidetail);
	
	//WeightedCombine(detail_layers, fine_detail_masks, multidetail);
	cout << "WeightedCombine ending" << endl;

	multisum = multibase + multidetail;
	//cout << "The value:"<<multisum.at<Vec3b>(5, 5) << endl;
	multisum.convertTo(multisum, CV_8UC3);

	double t8 = (double)cv::getTickCount();
	std::cout << " Time4 in seconds: " << ((double(t8 - t7)) / cv::getTickFrequency()) << endl;

	double time2 = (double)cv::getTickCount();
	std::cout << " Total time in seconds: " <<
		((double(time2 - time1)) / cv::getTickFrequency()) << endl;

	char path[100];
	for (int kdx = 0; kdx < images.size(); kdx++) {
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\bbase_layer\\%d.jpg", kdx);
		imwrite(path, base_layers[kdx]);
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\ddetail_layer\\%d.jpg", kdx);
		imwrite(path, detail_layers[kdx]);
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\ssaliency_layer\\%d.jpg", kdx);
		imwrite(path, saliency_layers[kdx]);
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\mmask_layer\\%d.jpg", kdx);
		imwrite(path, original_masks[kdx]);
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\bbase_weight\\%d.jpg", kdx);
		imwrite(path, fine_base_masks[kdx]);
		sprintf_s(path, "C:\\Users\\mao_y\\Desktop\\ddetail_weight\\%d.jpg", kdx);
		imwrite(path, fine_detail_masks[kdx]);
	}

	return multisum;
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
	//std::string img_fold(argv[1]);
	//std::string img_format(argv[2]);
	std::string img_fold = "H:\\Project\\PCellFusion\\Data\\2-250";
	std::string img_format = "bmp";
	FindOrginalImages(img_fold, files_names, img_format);
	for (int i = 0; i<files_names.size(); i++) {
		cout << "reading " << files_names[i] << endl;
		const Mat img = cv::imread(files_names[i], CV_LOAD_IMAGE_COLOR);

		images.push_back(img);
	}
	Mat result = GFF(images);
	std::string result_filename = img_fold + "\\" + "nowfused.jpg";
	cv::imwrite(result_filename, result);
	system("pause");
	return 0;
}