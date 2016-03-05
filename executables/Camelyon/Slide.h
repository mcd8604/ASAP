#include <string>
#include <vector>
#include "MultiResolutionImage.h"
#include "AnnotationList.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#pragma once
class Slide
{
public:
	Slide();
	Slide(std::string dataFile);
	~Slide();
	void loadFromDataFile(std::string dataFile);
	void saveToDataFile(std::string dataFile);
	cv::Mat getFeatures();
	cv::Mat getGroundTruth();
	std::vector<cv::Rect> Slide::getTissueTiles();
	void setFeatures(cv::Mat features);
	void setGroundTruth(cv::Mat groundTruth);
	void setTissueTiles(std::vector<cv::Rect> tissueTiles);
private:
	std::string mName;
	cv::Mat mFeatures;
	cv::Mat mGroundTruth;
	std::vector<cv::Rect> mTissueTiles;
};

