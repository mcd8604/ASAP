#include <vector>
#include <fstream>

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

Slide::Slide(MultiResolutionImage * image, MultiResolutionImage *groundTruthImage, int tissueClassLevel, int featureConstrLevel, Size nativeTileSize) :
	mImage(image),
	mGroundTruthImage(groundTruthImage),
	mTissueClassLevel(tissueClassLevel),
	mFeatureConstrLevel(featureConstrLevel),
	mNativeTileSize(nativeTileSize)
{
	preProcess();
}

Slide::Slide(MultiResolutionImage * image, shared_ptr<AnnotationList> groundTruth, int tissueClassLevel, int featureConstrLevel, Size nativeTileSize) :
	mImage(image),
	mAnnoList(groundTruth),
	mTissueClassLevel(tissueClassLevel),
	mFeatureConstrLevel(featureConstrLevel),
	mNativeTileSize(nativeTileSize)
{
	preProcess();
}

Slide::~Slide()
{
}

/// Pre-processing 
/// - Tissue Classification
/// - Superpixel Segmentation
/// - Process Ground Truth
/// - Feature construction
void Slide::preProcess() {
	classifyTissueTiles();
	// TODO generate superpixels on each tile at native resolution (use constructor params)
	processGroundTruth();
	constructFeatures();
}

// Uses the given image and level to classify tissue tiles and returns them as rectangles of size targetTileSize 
void Slide::classifyTissueTiles() {
	mNativeTissueTiles.clear();

	//TODO 
	//assert targetTileSize is rectangular
	//assert targetTileSize is power of 2
	//assert targetTileSize is max 512 per side

	CV_Assert(mTissueClassLevel < mImage->getNumberOfLevels());
	vector<unsigned long long, allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(mTissueClassLevel);
	Patch<uchar> levelPatch = mImage->getPatch<uchar>(0, 0, levelDim[0], levelDim[1], mTissueClassLevel);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	// TODO adjust filter levels - minor non-tissue artifacts are present
	//filter->setGlobalDensityThreshold(0.25);
	Patch<double> hemaPatch;
	filter->filter(levelPatch, hemaPatch);
	Mat hema = patchToMat(hemaPatch);
	delete filter;

	double downsample = mImage->getLevelDownsample(mTissueClassLevel);
	int w = mNativeTileSize.width / downsample;
	int h = mNativeTileSize.height / downsample;
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
				mNativeTissueTiles.push_back(Rect(cv::Point(x * mNativeTileSize.width, y * mNativeTileSize.height), mNativeTileSize));
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

// TODO Implement Strategy pattern for dynamic calculations
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

void Slide::outputFeaturesCSV(string filePath) {
	std::ofstream csv(filePath);
	for (int f = 0; f < mFeatureNames.size(); f++) {
		if (f > 0) csv << ",";
		csv << mFeatureNames.at(f);
	}
	csv << ",ground_truth";
	for (int x = 0; x < mFeatures.rows; x++) {
		csv << "\n";
		for (int y = 0; y < mFeatures.cols; y++) {
			if (y > 0) csv << ",";
			csv << mFeatures.at<float>(x, y);
		}
		csv << "," << mGroundTruthMat.at<float>(x, 0);
	}
	csv.close();
}

void Slide::constructFeatures() {
	const vector<Rect> tiles = getTissueTiles(mFeatureConstrLevel);
	int numTiles = tiles.size();
	setFeatureNames(&mFeatureNames);
	int numFeatures = mFeatureNames.size();
	mFeatures = Mat(numTiles, numFeatures, CV_32FC1);
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	for (int i = 0; i < numTiles; i++) {
		Rect r = tiles[i];
		Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, mFeatureConstrLevel);
		Patch<double> hemaPatch, eosPatch, zeroPatch;
		filter->setOutputStain(0);
		filter->filter(p, hemaPatch);
		filter->setOutputStain(1);
		filter->filter(p, eosPatch);
		filter->setOutputStain(2);
		filter->filter(p, zeroPatch);
		calcFeatures(i, 0, &patchToMat(hemaPatch), &mFeatures);
		calcFeatures(i, 1, &patchToMat(eosPatch), &mFeatures);
		calcFeatures(i, 2, &patchToMat(zeroPatch), &mFeatures);
	}
	delete filter;
}

void Slide::processGroundTruth()
{
	int numTiles = mNativeTissueTiles.size();
	mGroundTruthMat = Mat(numTiles, 1, CV_32S);
	if(mGroundTruthImage) {
		//int topLevel = mGroundTruthImage->getNumberOfLevels() - 1;
		//double d = mImage->getLevelDownsample(topLevel);

		for (int i = 0; i < numTiles; i++) {
			Rect r = mNativeTissueTiles[i];
			Patch<uchar> p = mGroundTruthImage->getPatch<uchar>(r.x, r.y, r.width, r.height, 0);
			Mat m = patchToMat(p);
			Scalar bSum = sum(m);
			mGroundTruthMat.at<int>(i, 0) = bSum[0] == 0 ? 0 : 1;// bSum[0] / r.area();
		}
	}
}

void Slide::rfTrain(const string outputFile) {
	Ptr<TrainData> trainData = TrainData::create(mFeatures, SampleTypes::ROW_SAMPLE, mGroundTruthMat);
	trainData->setTrainTestSplitRatio(0.8);
	std::cout << "Test/Train: " << trainData->getNTestSamples() << "/" << trainData->getNTrainSamples() << "\n";
	Scalar s = sum(mGroundTruthMat);

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
	rf->save(outputFile);

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

Mat Slide::rfTest(const string rfModelFile, const string outputFile) {
	// TODO assert model file
	Ptr<RTrees> rf = RTrees::load<RTrees>(rfModelFile);
	Ptr<TrainData> trainData = TrainData::create(mFeatures, SampleTypes::ROW_SAMPLE, mGroundTruthMat);
	Mat resp;
	float errorCalc = rf->calcError(trainData, false, resp);
	FileStorage fs(outputFile, FileStorage::Mode::WRITE);
	printf("RF Test error: %f\n", errorCalc);
	fs << "TestResult" << resp;
	fs.release();

	// TEST
	for (int i = 0; i < mNativeTissueTiles.size(); i++) {
		Rect r = mNativeTissueTiles[i];
		float gt = mGroundTruthMat.at<float>(i, 0);
		float test = resp.at<float>(i, 0);
		if(gt != 0 && test != 0) {
			Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, 0);
			Mat m = patchToMat(p);
			//groundTruthImage.
			imshow(to_string(r.x) + "," + to_string(r.y), m);
			//Scalar bSum = sum(m);
			//groundTruthMat.at<float>(i, 0) = bSum[0] == 0 ? 0 : 1;// bSum[0] / r.area();
			waitKey();
		}
	}


	return resp;
}

Mat Slide::rfPredict(const string rfModelFile, const string outputFile) {
	// TODO assert model file
	Ptr<RTrees> rf = RTrees::load<RTrees>(rfModelFile);
	Mat results;
	rf->predict(mFeatures, results);
	FileStorage fs(outputFile, FileStorage::Mode::WRITE);
	fs << "PredictionResult" << results;
	fs.release();
	return results;
}