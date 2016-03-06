/*******************************************************
Inputs: directory path of Images and Ground Truths
Outputs: FSFs

assumptions: All of images, ground truth pairs have same name
minus the suffix.
********************************************************/

#pragma once

#include <string>
#include <vector>

#include "config/pathology_config.h"
#include "io/multiresolutionimageinterface/MultiResolutionImage.h"
#include "annotation/AnnotationService.h"
#include "annotation/AnnotationList.h"
#include "opencv2/core.hpp"
#include "FeatureStrategy.h"

class FeatureConstructor
{
public:
	FeatureConstructor();
	FeatureConstructor(std::string filePath, std::string imageName, std::vector<FeatureStrategy> strategies);
	~FeatureConstructor();

	void run();
private:
	int mTissueClassLevel;
	int mFeatureConstrLevel;
	cv::Size mNativeTileSize;
	std::vector<FeatureStrategy> mStrategies;

	// Think about relocating these..
	std::string mfilePath;
	std::string mImageName;
	MultiResolutionImage *mImage;
	MultiResolutionImage *mGroundTruthImage;
	std::shared_ptr<AnnotationList> mAnnoList;
	std::vector<cv::Rect> mNativeTissueTiles;
	std::vector<std::string> mFeatureNames;
	cv::Mat mFeatures;
	cv::Mat mGroundTruthMat;
	std::vector<cv::Rect> getTissueTiles(int level);

	// process
	void loadImage(std::string filePath, std::string imageName);
	void classifyTissueTiles();
	void processGroundTruth();
	void constructFeatures();
};