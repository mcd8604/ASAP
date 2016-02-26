#include <vector>
#include <fstream>

#include "FeatureConstructor.h"
#include "annotation/Annotation.h"
#include "annotation/AnnotationList.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/NucleiDetectionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ml.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef ::Point AnnoPoint;

FeatureConstructor::FeatureConstructor() 
	: mfilePath("")
{
}

FeatureConstructor::FeatureConstructor(std::string filePath) 
	: mfilePath(filePath)
{
}

FeatureConstructor::~FeatureConstructor()
{
}

void FeatureConstructor::run() {
	classifyTissueTiles();
	// TODO generate superpixels on each tile at native resolution (use constructor params)
	processGroundTruth();
	constructFeatures();
}

void FeatureConstructor::classifyTissueTiles() {
	mNativeTissueTiles.clear();

	//TODO 
	//assert targetTileSize is rectangular
	//assert targetTileSize is power of 2
	//assert targetTileSize is max 512 per side

	//CV_Assert(mTissueClassLevel < mImage->getNumberOfLevels());
	std::vector<unsigned long long, std::allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(mTissueClassLevel);
	Patch<uchar> levelPatch = mImage->getPatch<uchar>(0, 0, levelDim[0], levelDim[1], mTissueClassLevel);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
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

// TODO Implement Strategy pattern for dynamic calculations
void FeatureConstructor::calcFeatures(int tileIdx, int modeIdx, cv::Mat *m) {
	int numFeatures = 3;
	// Feature calculations
	cv::Scalar sSum, sMean, sStdDev;
	sSum = sum(*m);
	meanStdDev(*m, sMean, sStdDev);
	// Storage
	int featureStartIdx = modeIdx * numFeatures;
	mFeatures.at<float>(tileIdx, featureStartIdx) = sSum[0];
	mFeatures.at<float>(tileIdx, featureStartIdx + 1) = sMean[0];
	mFeatures.at<float>(tileIdx, featureStartIdx + 2) = sStdDev[0];
}

void FeatureConstructor::setFeatureNames(std::string mode) {
	mFeatureNames.push_back(mode + "_sum");
	mFeatureNames.push_back(mode + "_mean");
	mFeatureNames.push_back(mode + "_stdDev");
}

void FeatureConstructor::setFeatureNames() {
	mFeatureNames.clear();
	setFeatureNames("hema");
	setFeatureNames("eos");
	setFeatureNames("zero");
	mFeatureNames.push_back("nuclei");
}

void FeatureConstructor::constructFeatures() {
	const std::vector<cv::Rect> tiles = getTissueTiles(mFeatureConstrLevel);
	int numTiles = tiles.size();
	setFeatureNames();
	int numFeatures = mFeatureNames.size();
	mFeatures = cv::Mat(numTiles, numFeatures, CV_32FC1);
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	//NucleiDetectionFilter<double> *ndf = new NucleiDetectionFilter<double>();
	for (int i = 0; i < numTiles; i++) {
		cv::Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, mFeatureConstrLevel);
		Patch<double> hemaPatch, eosPatch, zeroPatch;
		filter->setOutputStain(0);
		filter->filter(p, hemaPatch);
		filter->setOutputStain(1);
		filter->filter(p, eosPatch);
		filter->setOutputStain(2);
		filter->filter(p, zeroPatch);
		cv::Mat hemaMat = patchToMat(hemaPatch);
		calcFeatures(i, 0, &hemaMat);
		calcFeatures(i, 1, &patchToMat(eosPatch));
		calcFeatures(i, 2, &patchToMat(zeroPatch));
		std::vector<Point> nuclei;
		//ndf->filter(hemaPatch, nuclei);
		// THIS IS ERRORING HARD IN DEBUG! - issue is memory de/allocation across DLL
		//mFeatures.at<float>(i, 9) = ndf->getNumberOfDetectedNuclei();
		std::vector<std::vector<cv::Point> > contours;
		cv::Mat t = hemaMat >= .99;
		std::cout << t.type();
		cv::findContours(t, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
		mFeatures.at<float>(i, 9) = contours.size();
	}
	//delete filter;
}

// FeatureSegmentFile - file containing segment data with feature vectors and possibly ground truth
void FeatureConstructor::saveFeatureSegmentFile(std::string filePath) {
	cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
	// TODO parameterize image name/id
	fs << "imageName" << "someImageName";
	fs << "tissueTiles" << mNativeTissueTiles;
	// TODO segments instead of tiles
	fs << "features" << mFeatures;
	if (!mGroundTruthMat.empty()) {
		fs << "groundTruth" << mGroundTruthMat;
	}
	fs.release();
}