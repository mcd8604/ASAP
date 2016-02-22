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
	//TODO Create Param object?
	Slide(MultiResolutionImage *image, MultiResolutionImage *groundTruth, int tissueClassLevel = 8, int featureConstrLevel = 4, cv::Size nativeTileSize = cv::Size(512, 512));
	Slide(MultiResolutionImage *image, std::shared_ptr<AnnotationList> groundTruth, int tissueClassLevel = 8, int featureConstrLevel = 4, cv::Size nativeTileSize = cv::Size(512, 512));
	~Slide();
	std::vector<cv::Rect> Slide::getTissueTiles(int level);
	void Slide::outputFeaturesCSV(std::string filePath);
	void Slide::rfTrain(const std::string outputFile);
	cv::Mat Slide::rfTest(const std::string rfModelFile, const std::string outputFile);
	cv::Mat Slide::rfPredict(const std::string rfModelFile, const std::string outputFile);
private:
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
	void Slide::preProcess();
	void Slide::classifyTissueTiles();
	void Slide::processGroundTruth();
	void Slide::constructFeatures();
};

