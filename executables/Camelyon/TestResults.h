#include "opencv2\core.hpp"

#pragma once
class TestResults
{
public:
	TestResults();
	TestResults(cv::Mat_<float> predictions, cv::Mat_<float> groundTruth);
	~TestResults();
	//cv::Mat getPredictionsMatrix() { return mPredictionsMatrix; }
	//cv::Mat getGroundTruthMatrix() { return mGroundTruthMatrix; }
	//cv::Mat getConfusionMatrix() { return mConfusionMatrix; }
	cv::Mat plotROC(const int numPoints);
private:
	double getAccuracy();
	// Fall-out (False positive rate)
	double getFallout();
	// Miss Rate (False negative rate)
	double getMissRate();
	double getPrecision();
	double getPrevalence();
	// Sensitivity (True positive rate)
	double getSensitivity();
	// Specificity (True negative rate)
	double getSpecificity();
	// Reciever Operating Characteristic (Fall-out vs Sensitivity)
	cv::Point_<double> getROCPoint();

	double mThreshold = 0.5;
	int mTruePositives;
	int mFalsePositives;
	int mTrueNegatives;
	int mFalseNegatives;
	int mPopulationSize;
	cv::Mat_<float> mPredictionsMatrix;
	cv::Mat_<float> mGroundTruthMatrix;
	cv::Mat_<float> mConfusionMatrix;
	void TestResults::calculateConfusionMatrix();
};

