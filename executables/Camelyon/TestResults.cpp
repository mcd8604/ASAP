#include "TestResults.h"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"

TestResults::TestResults(cv::Mat_<float> predictions, cv::Mat_<float> groundTruth) :
	mTruePositives(0),
	mFalsePositives(0),
	mTrueNegatives(0),
	mFalseNegatives(0),
	mPopulationSize(0)
{
	mPredictionsMatrix = predictions;
	mGroundTruthMatrix = groundTruth;
}

TestResults::~TestResults()
{
}

void TestResults::plotROC(const int numPoints) {
	cv::Size size(512, 512);
	cv::Mat plot = cv:: Mat::zeros(size, CV_8UC1);
	for (float i = 1; i < numPoints; ++i) {
		//float t = i / 512;
		mThreshold = i / numPoints;
		calculateConfusionMatrix();
		cv::Point_<double> p = getROCPoint();
		cv::Point pScaled(p.x * size.width, size.height - (p.y * size.height));
		cv::circle(plot, pScaled, 3, cv::Scalar(255), 0);
	}
	cv::imshow("ROC", plot);
	cv::waitKey(0);
}

void TestResults::calculateConfusionMatrix() {
	mPopulationSize = mGroundTruthMatrix.rows;
	// Assume these are single-column matrices with equal number of rows
	//CV_ASSERT(populationSize == testResults.rows && groundTruth.cols == 1 && testResults.cols == 1);
	// assumes values are float (regression) between 0.0 and 1.0
	cv::Mat thresholdedTruths;
	cv::threshold(mGroundTruthMatrix, thresholdedTruths, 0.5, 1.0, cv::THRESH_BINARY);
	cv::Mat thresholdedPredictions;
	cv::threshold(mPredictionsMatrix, thresholdedPredictions, mThreshold, 1.0, cv::THRESH_BINARY);
	mTruePositives = cv::sum((thresholdedTruths == 1) & (thresholdedPredictions == 1))[0];
	mTrueNegatives = cv::sum((thresholdedTruths == 0) & (thresholdedPredictions == 0))[0];
	mFalsePositives = cv::sum((thresholdedTruths == 0) & (thresholdedPredictions == 1))[0];
	mFalseNegatives = cv::sum((thresholdedTruths == 1) & (thresholdedPredictions == 0))[0];
	/*for (int i = 0; i < mPopulationSize; i++) {
		bool actual = mGroundTruthMatrix.at<float>(i) / mThreshold;
		bool prediction = mPredictionsMatrix.at<float>(i) / 0.5;
		if (actual) 
			if (prediction)
				mTrueNegatives++;
			else
				mFalsePositives++;
		else 
			if (prediction)
				mFalseNegatives++;
			else
				mTruePositives++;
	}*/
	mConfusionMatrix = cv::Mat(2, 2, CV_32SC1);
	mConfusionMatrix.at<int>(0, 0) = mTruePositives;
	mConfusionMatrix.at<int>(0, 1) = mFalsePositives;
	mConfusionMatrix.at<int>(1, 0) = mFalseNegatives;
	mConfusionMatrix.at<int>(1, 1) = mTrueNegatives;
}

double TestResults::getAccuracy() {
	double accuracy = 0;
	if (mPopulationSize > 0)
		accuracy = (mTruePositives + mTrueNegatives) / double(mPopulationSize);
	return accuracy;
}

double TestResults::getFallout() {
	double fallout = 0;
	double conditionNegative = mTrueNegatives + mFalsePositives;
	if (conditionNegative > 0)
		fallout = mFalsePositives / conditionNegative;
	return fallout;
}

double TestResults::getMissRate()	{
	double missRate = 0;
	double conditionPositive = mTruePositives + mFalseNegatives;
	if (conditionPositive > 0)
		missRate = mFalseNegatives / conditionPositive;
	return missRate;
}

double TestResults::getPrecision() {
	double precision = 0;
	double predictedPositive = mTruePositives + mFalsePositives;
	if (predictedPositive > 0)
		precision = mTruePositives / predictedPositive;
	return precision;
}

double TestResults::getPrevalence()	{
	double prevalence = 0;
	if (mPopulationSize > 0)
		prevalence = (mTruePositives + mFalseNegatives) / double(mPopulationSize);
	return prevalence;
}

double TestResults::getSensitivity() {
	double sensitivity = 0;
	double conditionPositive = mTruePositives + mFalseNegatives;
	if (conditionPositive > 0)
		sensitivity = mTruePositives / conditionPositive;
	return sensitivity;
}

double TestResults::getSpecificity() {
	double specificity = 0;
	double conditionNegative = mTrueNegatives + mFalsePositives;
	if (conditionNegative > 0)
		specificity = mTrueNegatives / conditionNegative;
	return specificity;
}

cv::Point_<double> TestResults::getROCPoint() {
	return cv::Point_<double>(getFallout(), getSensitivity());
}