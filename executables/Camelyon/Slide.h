#include <string>
#include <vector>
#include "Segment.h"
#include "MultiResolutionImage.h"
#include "AnnotationList.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

enum SuperpixelType {
	RECT,
	SLIC
};

#pragma once
class Slide
{
public:
	Slide(MultiResolutionImage *image, SuperpixelType sType);
	~Slide();

	void constructFeatures(cv::Ptr<cv::Feature2D> featureDetector);

	/*
	*	Prints a CSV file with a header.
	*	Each row contains the segment id and feature vector.
	*		Ex:
	*			SegmentID, Feature1, Feature2, ..., FeatureN
	*	It is assumed that each segment has a unique ID and 
	*	contains an equal number of features.
	*/
	bool segFeatsToCSV(std::string filePath);
	void setAnnotationList(std::shared_ptr<AnnotationList> annoList);
	void evaluatePredictions();
private:
	std::string id;
	std::vector<Segment> segments;
	MultiResolutionImage *mImage;
	SuperpixelType mSType;
	std::shared_ptr<AnnotationList> mAnnoList;
};

