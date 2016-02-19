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
	std::vector<cv::Rect> Slide::getTissueTiles(int sampleLevel, cv::Size targetTileSize);
	cv::Mat constructFeatures(std::vector<cv::Ptr<cv::Feature2D>> featureDetectors, std::vector<cv::Rect> tiles, int level);
	cv::Mat getGroundTruth(std::vector<cv::Rect> tiles);
private:
	MultiResolutionImage *mImage;
	std::shared_ptr<AnnotationList> mAnnoList;
};

