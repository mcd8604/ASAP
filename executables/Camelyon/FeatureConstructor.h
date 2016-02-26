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
#include "opencv2/features2d.hpp"

class FeatureConstructor
{
public:
	FeatureConstructor();
	FeatureConstructor(std::string filePath);
	~FeatureConstructor();

	void run();
private:
	void loadImage(std::string filePath);
	void saveFeatureSegmentFile(std::string filePath);

	std::string mfilePath;

	MultiResolutionImage *mImage;
	MultiResolutionImage *mGroundTruthImage;
	int mTissueClassLevel;
	int mFeatureConstrLevel;
	cv::Size mNativeTileSize;
	std::shared_ptr<AnnotationList> mAnnoList;
	std::vector<cv::Rect> mNativeTissueTiles;
	std::vector<std::string> mFeatureNames;
	cv::Mat mFeatures;
	cv::Mat mGroundTruthMat;
	void preProcess();
	void classifyTissueTiles();
	void processGroundTruth();
	void constructFeatures();
	std::vector<cv::Rect> getTissueTiles(int level);
	// TODO refactor into Strategy Pattern
	void calcFeatures(int tileIdx, int modeIdx, cv::Mat *m);
	void setFeatureNames(std::string mode);
	void setFeatureNames();
};