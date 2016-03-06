#include <vector>
#include <fstream>

#include "FeatureConstructor.h"
#include "Slide.h"
#include "MultiResolutionImageReader.h"
#include "annotation/Annotation.h"
#include "annotation/AnnotationList.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef ::Point AnnoPoint;

FeatureConstructor::FeatureConstructor()
	: mfilePath("")
{
}

FeatureConstructor::FeatureConstructor(std::string filePath, std::string imageName, std::vector<FeatureStrategy*> strategies)
	: mfilePath(filePath), mImageName(imageName), mStrategies(strategies),
	mTissueClassLevel(8),
	mFeatureConstrLevel(0),
	mNativeTileSize(cv::Size(512, 512))
{
}

FeatureConstructor::~FeatureConstructor()
{
}

void FeatureConstructor::loadImage(std::string filePath, std::string imageName) {
	MultiResolutionImageReader reader;
	mImage = reader.open(filePath + imageName + ".tif");
	mGroundTruthImage = reader.open(filePath + imageName + "_Mask.tif");
}

void FeatureConstructor::run() {
	loadImage(mfilePath, mImageName);
	Slide slide;
	classifyTissueTiles();
	slide.setTissueTiles(mNativeTissueTiles);
	// TODO generate superpixels on each tile at native resolution (use constructor params)
	processGroundTruth();
	slide.setGroundTruth(mGroundTruthMat);
	constructFeatures();
	slide.setFeatures(mFeatures);
	slide.saveToDataFile(mfilePath + mImageName + ".yaml");
}

void FeatureConstructor::classifyTissueTiles() {
	mNativeTissueTiles.clear();

	//TODO 
	//assert targetTileSize is rectangular
	//assert targetTileSize is power of 2
	//assert targetTileSize is max 512 per side

	//CV_Assert(mTissueClassLevel < mImage->getNumberOfLevels());
	std::vector<unsigned long long, std::allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(mTissueClassLevel);
	Patch<double> levelPatch = mImage->getPatch<double>(0, 0, levelDim[0], levelDim[1], mTissueClassLevel);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<double> *filter = new ColorDeconvolutionFilter<double>();
	// TODO adjust filter levels - minor non-tissue artifacts are present
	//filter->setGlobalDensityThreshold(0.25);
	Patch<double> hemaPatch;
	filter->filter(levelPatch, hemaPatch);
	cv::Mat hema = patchToMat(hemaPatch);
	delete filter;

	double downsample = mImage->getLevelDownsample(mTissueClassLevel);
	int w = mNativeTileSize.width / downsample;
	int h = mNativeTileSize.height / downsample;
	int numTilesX = levelDim[0] / w;
	int numTilesY = levelDim[1] / h;

	for (int y = 0; y < numTilesY; y++) {
		for (int x = 0; x < numTilesX; x++) {
			cv::Rect r(x * w, y * h, w, h);
			// Classify Foreground/Background
			// The hema patch has binary values 
			// (ok, apparently not..)
			// TODO investigate ColorDeconFilter
			// so this yields the percent of foreground pixels
			cv::Scalar sc = sum(hema(r));
			double c1 = sc[0];
			double ar = r.area();
			double percentForeground = c1 / ar;
			if (percentForeground > 0.1) {
				mNativeTissueTiles.push_back(cv::Rect(cv::Point(x * mNativeTileSize.width, y * mNativeTileSize.height), mNativeTileSize));
			}
		}
	}
}

void FeatureConstructor::processGroundTruth()
{
	int numTiles = mNativeTissueTiles.size();
	if (mGroundTruthImage) {
		mGroundTruthMat = cv::Mat(numTiles, 1, CV_32S);
		//int topLevel = mGroundTruthImage->getNumberOfLevels() - 1;
		//double d = mImage->getLevelDownsample(topLevel);

		for (int i = 0; i < numTiles; i++) {
			cv::Rect r = mNativeTissueTiles[i];
			Patch<uchar> p = mGroundTruthImage->getPatch<uchar>(r.x, r.y, r.width, r.height, 0);
			cv::Mat m = patchToMat(p);
			cv::Scalar bSum = sum(m);
			mGroundTruthMat.at<int>(i, 0) = bSum[0] == 0 ? 0 : 1;
			// bSum[0] / r.area();
		}
	}
	else if (mAnnoList) {
		// TODO process annotations
		//mGroundTruthMat = cv::Mat(numTiles, 1, CV_32S);
	}
}

std::vector<cv::Rect> FeatureConstructor::getTissueTiles(int level) {
	std::vector<cv::Rect> tissueTiles;
	if (level == 0) {
		tissueTiles = mNativeTissueTiles;
	}
	else {
		tissueTiles.reserve(mNativeTissueTiles.size());
		double d = mImage->getLevelDownsample(level);
		for (cv::Rect r : mNativeTissueTiles) {
			tissueTiles.push_back(cv::Rect(r.x / d, r.y / d, r.width / d, r.height / d));
		}
	}
	return tissueTiles;
}

void FeatureConstructor::constructFeatures() {
	const std::vector<cv::Rect> tiles = getTissueTiles(mFeatureConstrLevel);
	int numTiles = tiles.size();
	mFeatureNames.clear();
	for (FeatureStrategy *s : mStrategies) {
		std::vector<std::string> featureNames = s->getFeatureNames();
		mFeatureNames.insert(mFeatureNames.end(), featureNames.begin(), featureNames.end());
	}
	int numFeatures = mFeatureNames.size();
	mFeatures = cv::Mat(numTiles, numFeatures, CV_32FC1);
	for (int i = 0; i < numTiles; i++) {
		cv::Rect r = tiles[i];
		Patch<double> tilePatch = mImage->getPatch<double>(r.x, r.y, r.width, r.height, mFeatureConstrLevel);
		// Note: This vector/Mat usage can probably be optimized for better performance
		std::vector<float> featureVector;
		for (FeatureStrategy *s : mStrategies) {
			std::vector<float> features = s->constructFeatures(tilePatch);
			featureVector.insert(featureVector.end(), features.begin(), features.end());
		}
		for (int f = 0; f < numFeatures; f++)
			mFeatures.at<float>(i, f) = featureVector.at(f);
	}
}