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
	~Slide();
	void setAnnotationList(std::shared_ptr<AnnotationList> annoList);
	std::vector<cv::Rect> Slide::getTissueTiles(MultiResolutionImage *image, int level, cv::Size nativeTileSize);
private:
	std::shared_ptr<AnnotationList> mAnnoList;
};

