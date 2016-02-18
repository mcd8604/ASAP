#include <vector>
#include "Slide.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "Annotation.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv::ml;

Slide::Slide(MultiResolutionImage * image)
{
	mImage = image;
}

Slide::~Slide()
{
}

void Slide::setAnnotationList(shared_ptr<AnnotationList> annoList)
{
	mAnnoList = annoList;
}

// Uses the given image and level to classify tissue tiles and returns them as rectangles of size nativeTileSize 
vector<cv::Rect> Slide::getTissueTiles(int sampleLevel, cv::Size targetTileSize) {
	using namespace cv;
	vector<Rect> tissueTiles;

	CV_Assert(sampleLevel < mImage->getNumberOfLevels());
	vector<unsigned long long, allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(sampleLevel);
	Patch<uchar> levelPatch = mImage->getPatch<uchar>(0, 0, levelDim[0], levelDim[1], sampleLevel);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	// TODO adjust filter levels - minor non-tissue artifacts are present
	//filter->setGlobalDensityThreshold(0.25);
	Patch<double> hemaPatch;
	filter->filter(levelPatch, hemaPatch);
	Mat hema = patchToMat(hemaPatch);
	delete filter;

	double downsample = mImage->getLevelDownsample(sampleLevel);
	int w = targetTileSize.width / downsample;
	int h = targetTileSize.height / downsample;
	int numTilesX = levelDim[0] / w;
	int numTilesY = levelDim[1] / h;

	for (int y = 0; y < numTilesY; y++) {
		for (int x = 0; x < numTilesX; x++) {
			Rect r(x * w, y * h, w, h);
			// Classify Foreground/Background
			// The hema patch has binary values 
			// (ok, apparently not..)
			// TODO investigate ColorDeconFilter
			// so this yields the percent of foreground pixels
			Scalar sc = sum(hema(r));
			double c1 = sc[0];
			double ar = r.area();
			double percentForeground = c1 / ar;
			if (percentForeground > 0.1) {
				tissueTiles.push_back(Rect(cv::Point(x * targetTileSize.width, y * targetTileSize.height), targetTileSize));
			}
		}
	}
	return tissueTiles;
}

//#define TEST_FC_VIEW_TILES 0;
cv::Mat Slide::constructFeatures(vector<cv::Ptr<cv::Feature2D>> featureDetectors, vector<cv::Rect> tiles, int level) {
	using namespace cv;
	Mat m;
	int numTiles = tiles.size();
	int numFeatureDetectors = featureDetectors.size();
	// TODO parameterize statistics
	Mat features(numTiles, 6, CV_32F);
	vector<KeyPoint> keyPoints;

	for (int i = 0; i < numTiles; i++) {
		Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, level);
		m = patchToMat(p);
		Scalar bSum = sum(m);
		features.at<float>(i, 0) = bSum[0];
		features.at<float>(i, 1) = bSum[1];
		features.at<float>(i, 2) = bSum[2];
		Scalar bMean = mean(m);
		features.at<float>(i, 3) = bMean[0];
		features.at<float>(i, 4) = bMean[1];
		features.at<float>(i, 5) = bMean[2];
	}

	/*for (int f = 0; f < featureDetectors.size(); f++) {
	Ptr<Feature2D> fd = featureDetectors[f];
	//fd->clear();
	keyPoints.clear();
	fd->detect(m, keyPoints);

	// TODO parameterize statistics
	float count = 0;
	float sumSize = 0;
	for (KeyPoint keyPoint : keyPoints) {
	count++;
	sumSize += keyPoint.size;
	}
	float meanSize = sumSize / count;

	int statOffsetIndex = numStats * f;
	features.at<float>(i, 0 + statOffsetIndex) = count;
	features.at<float>(i, 1 + statOffsetIndex) = sumSize;
	features.at<float>(i, 2 + statOffsetIndex) = meanSize;

	//fd->detectAndCompute(m, noArray(), keyPoints, descriptors);
	//features.push_back(descriptors);

	#if TEST_FC_VIEW_TILES
	drawKeypoints(m, keyPoints, m);
	imshow("test", m);
	//imshow("descriptors", descriptors);
	waitKey(0);
	#endif
	// TODO feature reduction
	}*/
	return features;
}

cv::Mat Slide::generateGroundTruth(std::vector<cv::Rect> tiles)
{
	int numTiles = tiles.size();
	cv::Mat groundTruthMat(numTiles, 1, CV_32S);
	for (int i = 0; i < numTiles; i++) {
		cv::Rect tile = tiles[i];
		int groundTruth = 0;
		for (shared_ptr<Annotation> annoPtr : mAnnoList->getAnnotations()) {
			// TODO change from center point containment to polygonal intersections
			Point p = annoPtr->getCenter();
			if(tile.contains(cv::Point(p.getX(), p.getY()))) {
				groundTruth = 1;
				break;
			}
		}
		groundTruthMat.at<int>(i, 0) = groundTruth;
	}
	return groundTruthMat;
}

