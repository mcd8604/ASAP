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
	Slide(MultiResolutionImage *image);
	~Slide();
	void setAnnotationList(std::shared_ptr<AnnotationList> annoList);
	void setGroundTruth(MultiResolutionImage *groundTruth);
	void Slide::classifyTissueTiles(int sampleLevel, cv::Size nativeTileSize);
	std::vector<cv::Rect> Slide::getTissueTiles(int level);
	cv::Mat constructFeatures(std::vector<cv::Ptr<cv::Feature2D>> featureDetectors, std::vector<cv::Rect> tiles, int level);
	cv::Mat constructFeatures(const std::vector<cv::Rect> tiles, const int level, std::vector<std::string> *featureNames_out);
	cv::Mat getGroundTruth(std::vector<cv::Rect> tiles, int level);
	void Slide::rforest(const cv::Mat groundTruth, const cv::Mat features);
private:
	MultiResolutionImage *mImage;
	MultiResolutionImage *mGroundTruth;
	std::shared_ptr<AnnotationList> mAnnoList;
	std::vector<cv::Rect> mNativeTissueTiles;
};

