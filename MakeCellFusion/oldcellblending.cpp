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

//#define  TCT_DEBUG

using namespace std;
using namespace cv;
//void getFiles(std::string path, std::vector<std::string>& files) {
//	long   hFile = 0;
//	struct _finddata_t fileinfo;
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
//	{
//		do
//		{
//			if ((fileinfo.attrib &  _A_SUBDIR))
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//					getFiles(p.assign(path).append("/").append(fileinfo.name), files);
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("/").append(fileinfo.name));
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}
//void getFiles(std::string path, std::vector<std::string>& flist) {
//	HANDLE file;
//	WIN32_FIND_DATA fileData;
//	char line[1024];
//	wchar_t fn[1000];
//	mbstowcs(fn,(const char*)path);
//	file = FindFirstFile(fn,&fileData);
//	FindNextFile(file,&fileData);
//	while (FindNextFile(file,&fileData)) {
//		wcstombs(line,(const wcahr_t*)fileData.cFileName,259);
//		flist.push_back(line);
//	}
//	return flist;
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

//// Normalize the images in to [0, 1]
void normalize(std::vector<cv::Mat>& images)
{
	if (images.size() == 0) return;

	cv::Mat image_sum = cv::Mat::zeros(images[0].rows, images[0].cols, images[0].type());
	for (int i = 0; i < images.size(); i++)
		image_sum = image_sum + images[i];
	for (int i = 0; i < images.size(); i++)
		images[i] = images[i].mul(1 / image_sum);
}

void GuidOptimize(const std::vector<cv::Mat>&images, const std::vector<cv::Mat>&init_weights, std::vector<cv::Mat>& weights, int r, double eps)
{
	if (images.size() == 0) return;

	weights.clear();
	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat filterd;
		filterd = guidedFilter(images[i], init_weights[i], r, eps);
		//cv::ximgproc::guidedFilter(images[i], init_weights[i], filterd, r, eps);
		filterd.convertTo(filterd, CV_32F, 1.0 / 255);
		weights.push_back(filterd);
	}

	normalize(weights);
}

int main(int argc, char* argv[])
{
	std::vector<std::string> imageFiles;
	if (argc < 3)
	{
		help();
		return -1;
	}

	double t = (double)cv::getTickCount();
	std::string img_fold = "H:\\Project\\PCellFusion\\Data\\4-250";
	std::string img_ext = "bmp";
	std::vector<std::string> files_names;
	FindOrginalImages(img_fold, files_names, img_ext);

	int r1 = 15, r2 = 7;
	double eps1 = 0.3, eps2 = 0.000001;

	//1. first read images and calculate saliency maps
	std::vector<cv::Mat> images, grays, saliency_maps, init_weights, weight_maps;
#ifdef TCT_DEBUG	
	double t1 = (double)cv::getTickCount();
#endif
	for (auto filename : files_names)
	{
		std::string ext = filename.substr(filename.find_last_of('.') + 1);
		if (ext != img_ext) continue;
		std::cout << "reading " << filename << std::endl;
		const cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

		//SALIENCY MAPS
		cv::Mat gray, saliency_map;
		cvtColor(image, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_32F, 1.0 / 255);
		Laplacian(gray, saliency_map, CV_32F);
		GaussianBlur(cv::abs(saliency_map), saliency_map, Size(5, 5), 0);

		grays.push_back(gray);
		images.push_back(image);
		saliency_maps.push_back(saliency_map);
	}
	if (images.size() < 2)
	{
		std::cout << "At least 2 images are needed." << std::endl;
		return -1;
	}
#ifdef TCT_DEBUG	
	std::cout << " Time for read images and calculate saliency maps in seconds: " <<
		((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
#endif

	//2. Construct the initial weight maps
#ifdef TCT_DEBUG	
	t1 = (double)cv::getTickCount();
#endif

	Mat saliency_max = Mat::zeros(saliency_maps[0].rows, saliency_maps[0].cols, saliency_maps[0].type());
	for (int i = 0; i < saliency_maps.size(); i++) {
		saliency_max = cv::max(saliency_max, saliency_maps[i]);
	}
	for (int i = 0; i < saliency_maps.size(); i++)
	{
		cv::Mat P = (saliency_maps[i] >= saliency_max);
		init_weights.push_back(P);
	}
#ifdef TCT_DEBUG	
	std::cout << " Time for construct the initial weight maps in seconds: " <<
		((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
#endif

	//3.  Weight Optimization with Guided Filtering
#ifdef TCT_DEBUG	
	t1 = (double)cv::getTickCount();
#endif
	std::vector<cv::Mat> weight_maps_base, weight_maps_detail;
	GuidOptimize(images, init_weights, weight_maps_base, r1, eps1);
	GuidOptimize(images, init_weights, weight_maps_detail, r2, eps2);
#ifdef TCT_DEBUG
	std::cout << " Time for Weight Optimization in seconds: " <<
		((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
#endif

	//4. Two Scale Decomposition and Fusion
#ifdef TCT_DEBUG	
	t1 = (double)cv::getTickCount();
#endif
	cv::Mat im_base, im_detail, im_fused;
	if (images[0].channels() == 3)
	{
		im_base = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC3);
		im_detail = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC3);
	}
	else
	{
		im_base = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32F);
		im_detail = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32F);
	}

	for (int i = 0; i < images.size(); i++)
	{
		cv::Mat image = images[i];

		//BASE LAYERS AND DETAIL LAYERS
		cv::Mat base, detail, im_float;
		GaussianBlur(image, base, Size(15, 15), 0);
		addWeighted(image, 1.0, base, -1.0, 0, detail);
		base.convertTo(base, CV_32FC3, 1. / 255);
		detail.convertTo(detail, CV_32FC3, 1. / 255);

		cv::Mat weight_base, weight_detail;
		if (image.channels() == 3)
		{
			cvtColor(weight_maps_base[i], weight_base, CV_GRAY2RGB);
			cvtColor(weight_maps_detail[i], weight_detail, CV_GRAY2RGB);
		}
		else if (image.channels() == 1)
		{
			weight_base = weight_maps_base[i];
			weight_detail = weight_maps_detail[i];
		}
		else
			return -1;

		im_base = im_base + base.mul(weight_base);
		im_detail = im_detail + detail.mul(weight_detail);
	}
	im_fused = im_base + im_detail;
	im_fused.convertTo(im_fused, CV_8UC3, 255);
#ifdef TCT_DEBUG	
	std::cout << " Time for two scale decomposition and fusion in seconds: " <<
		((double)cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
#endif

	std::cout << " Time in seconds: " <<
		((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;

	std::string result_filename = files_names[0].substr(0, files_names[0].find_last_of('.'));
	result_filename = result_filename + "mmmmm_fused.jpg";
	cv::imwrite(result_filename, im_fused);
	//cv::namedWindow("filterd", WINDOW_NORMAL);
	//cv::imshow("filterd", im_fused);
	//cv::waitKey();
	system("pause");
	return 0;
}