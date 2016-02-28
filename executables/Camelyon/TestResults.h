#include "opencv2\core.hpp"

#pragma once
class TestResults
{
public:
	TestResults(cv::Mat testResults);
	TestResults(cv::Mat testResults, cv::Mat groundTruth);
	~TestResults();
	cv::Mat getResultsMatrix() { return mResultsMatrix; }
	cv::Mat getGroundTruthMatrix() { return mGroundTruthMatrix; }
	cv::Mat getConfusionMatrix() { return mConfusionMatrix; }
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
private:
	int mTruePositives;
	int mFalsePositives;
	int mTrueNegatives;
	int mFalseNegatives;
	int mPopulationSize;
	cv::Mat mResultsMatrix;
	cv::Mat mGroundTruthMatrix;
	cv::Mat mConfusionMatrix;
	void TestResults::calculateConfusionMatrix();
};

