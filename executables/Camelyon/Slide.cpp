#include <vector>
#include "Slide.h"
#include "Annotation.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "Annotation.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ml.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

using namespace std;
using namespace cv;
using namespace ml;

typedef ::Point AnnoPoint;

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

// Uses the given image and level to classify tissue tiles and returns them as rectangles of size targetTileSize 
void Slide::classifyTissueTiles(int sampleLevel, Size targetTileSize) {
	using namespace cv;
	mNativeTissueTiles.clear();

	//TODO 
	//assert targetTileSize is rectangular
	//assert targetTileSize is power of 2
	//assert targetTileSize is max 512 per side

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
				mNativeTissueTiles.push_back(Rect(cv::Point(x * targetTileSize.width, y * targetTileSize.height), targetTileSize));
			}
		}
	}
}

vector<Rect> Slide::getTissueTiles(int level) {
	vector<Rect> tissueTiles;
	if (level == 0) {
		tissueTiles = mNativeTissueTiles;
	}
	else {
		tissueTiles.reserve(mNativeTissueTiles.size());
		double d = mImage->getLevelDownsample(level);
		for (Rect r : mNativeTissueTiles) {
			tissueTiles.push_back(Rect(r.x / d, r.y / d, r.width / d, r.height / d));
		}
	}
	return tissueTiles;
}

void calcFeatures(int tileIdx, int modeIdx, Mat *m, Mat *features_out) {
	int numFeatures = 3;
	// Feature calculations
	Scalar sSum, sMean, sStdDev;
	sSum = sum(*m);
	meanStdDev(*m, sMean, sStdDev);
	// Storage
	int featureStartIdx = modeIdx * numFeatures;
	features_out->at<float>(tileIdx, featureStartIdx) = sSum[0];
	features_out->at<float>(tileIdx, featureStartIdx + 1) = sMean[0];
	features_out->at<float>(tileIdx, featureStartIdx + 2) = sStdDev[0];
}

void setFeatureNames(string mode, vector<string> *featureNames_out) {
	featureNames_out->push_back(mode + "_sum");
	featureNames_out->push_back(mode + "_mean");
	featureNames_out->push_back(mode + "_stdDev");
}

void setFeatureNames(vector<string> *featureNames_out) {
	featureNames_out->clear();
	setFeatureNames("hema", featureNames_out);
	setFeatureNames("eos", featureNames_out);
	setFeatureNames("zero", featureNames_out);
}

Mat Slide::constructFeatures(const vector<Rect> tiles, const int level, vector<string> *featureNames_out) {
	int numTiles = tiles.size();
	setFeatureNames(featureNames_out);
	int numFeatures = featureNames_out->size();
	Mat features_out(numTiles, numFeatures, CV_32FC1);
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	for (int i = 0; i < numTiles; i++) {
		Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, level);
		Patch<double> hemaPatch, eosPatch, zeroPatch;
		filter->setOutputStain(0);
		filter->filter(p, hemaPatch);
		filter->setOutputStain(1);
		filter->filter(p, eosPatch);
		filter->setOutputStain(2);
		filter->filter(p, zeroPatch);
		calcFeatures(i, 0, &patchToMat(hemaPatch), &features_out);
		calcFeatures(i, 1, &patchToMat(eosPatch), &features_out);
		calcFeatures(i, 2, &patchToMat(zeroPatch), &features_out);
	}
	delete filter;
	return features_out;
}

Mat Slide::constructFeatures(vector<Ptr<Feature2D>> featureDetectors, vector<Rect> tiles, int level) {
	int numTiles = tiles.size();
	Mat features(numTiles, 2, CV_32F);
	// TODO parameterize statistics
	/*int numStats = 3;
	vector<KeyPoint> keyPoints;
	Mat features(numTiles, numStats * numFeatureDetectors, CV_32F);
	for (int i = 0; i < numTiles; i++) {
		Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, level);
		Mat m = patchToMat(p);
		for (int f = 0; f < featureDetectors.size(); f++) {
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
			float meanSize = 0;
			if(count > 0) meanSize = sumSize / count;

			int statOffsetIndex = numStats * f;
			features.at<float>(i, 0 + statOffsetIndex) = count;
			if (f == 0 && count > 10) {
				int yay = 1;
			}
			features.at<float>(i, 1 + statOffsetIndex) = sumSize;
			features.at<float>(i, 2 + statOffsetIndex) = meanSize;

			//fd->detectAndCompute(m, noArray(), keyPoints, descriptors);
			//features.push_back(descriptors);

			//#if TEST_FC_VIEW_TILES
			//drawKeypoints(m, keyPoints, m);
			//imshow("test", m);
			////imshow("descriptors", descriptors);
			//waitKey(0);
			//#endif
			//// TODO feature reduction
			//}
		}
	}*/
	return features;
}

void Slide::setGroundTruth(MultiResolutionImage *groundTruth) {
	mGroundTruth = groundTruth;
}

Mat Slide::getGroundTruth(vector<Rect> tiles, int level)
{
	int numTiles = mNativeTissueTiles.size();
	cv::Mat groundTruthMat(numTiles, 1, CV_32F);
	int topLevel = mGroundTruth->getNumberOfLevels() - 1;
	double d = mImage->getLevelDownsample(topLevel);

	for (int i = 0; i < numTiles; i++) {
		Rect r = mNativeTissueTiles[i];
		Patch<uchar> p = mGroundTruth->getPatch<uchar>(r.x, r.y, r.width, r.height, 0);
		Mat m = patchToMat(p);
		Scalar bSum = sum(m);
		groundTruthMat.at<float>(i, 0) = bSum[0] == 0 ? 0 : 1;// bSum[0] / r.area();
	}
	/*for (int i = 0; i < numTiles; i++) {
		cv::Rect tile = tiles[i];
		int groundTruth = 0;
		for (shared_ptr<Annotation> annoPtr : mAnnoList->getAnnotations()) {
			if (groundTruth == 1) break;
			// TODO change from center point containment to polygonal intersections
			AnnoPoint p = annoPtr->getCenter();
			if (tile.contains(cv::Point(p.getX(), p.getY()))) {
				groundTruth = 1;
				break;
			}
			vector<AnnoPoint> pts = annoPtr->getCoordinates();
			for (AnnoPoint pt : pts) {
				if (tile.contains(cv::Point(pt.getX() / d, pt.getY() / d))) {
					groundTruth = 1;
					break;
				}
			}
		}
		groundTruthMat.at<int>(i, 0) = groundTruth;
	}*/
	return groundTruthMat;
}

void Slide::rforest(const Mat groundTruth, const Mat features) {
	Ptr<TrainData> trainData = TrainData::create(features, SampleTypes::ROW_SAMPLE, groundTruth);
	trainData->setTrainTestSplitRatio(0.8);
	std::cout << "Test/Train: " << trainData->getNTestSamples() << "/" << trainData->getNTrainSamples() << "\n";
	Scalar s = sum(groundTruth);

	Ptr<RTrees> rf = RTrees::create();
	rf->setCalculateVarImportance(true);
	rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10, 0));
	bool ok = rf->train(trainData);
	if (!ok)
	{
		printf("Training failed\n");
	}
	else
	{
		printf("train error: %f\n", rf->calcError(trainData, false, noArray()));
		printf("test error: %f\n\n", rf->calcError(trainData, true, noArray()));
	}
	rf->save("rf.txt");

	// Print variable importance
	Mat var_importance = rf->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}
}