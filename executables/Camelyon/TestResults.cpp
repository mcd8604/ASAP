#include "TestResults.h"


TestResults::TestResults(cv::Mat testResults) :
	mTruePositives(0),
	mFalsePositives(0),
	mTrueNegatives(0),
	mFalseNegatives(0),
	mPopulationSize(0)
{
	mResultsMatrix = testResults;
}

TestResults::TestResults(cv::Mat testResults, cv::Mat groundTruth) :
	mTruePositives(0),
	mFalsePositives(0),
	mTrueNegatives(0),
	mFalseNegatives(0),
	mPopulationSize(0)
{
	mResultsMatrix = testResults;
	mGroundTruthMatrix = groundTruth;
	calculateConfusionMatrix();
}

TestResults::~TestResults()
{
}

void TestResults::calculateConfusionMatrix() {
	int populationSize = mGroundTruthMatrix.rows;
	// Assume these are single-column matrices with equal number of rows
	//CV_ASSERT(populationSize == testResults.rows && groundTruth.cols == 1 && testResults.cols == 1);
	//TODO: get type from groundTruth as int (classification) or float (regression)
	for (int i = 0; i < populationSize; i++) {
		int actual = mGroundTruthMatrix.at<int>(i);
		int prediction = mResultsMatrix.at<int>(i);
		// Assume values are only ever 0 or 1
		if (actual == prediction)
			actual == 0 ? mTrueNegatives++ : mTruePositives++;
		else
			actual == 0 ? mFalsePositives++ : mFalseNegatives++;
	}
	mConfusionMatrix = cv::Mat(2, 2, CV_8UC1);
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
	double conditionNegative = mTrueNegatives + mFalseNegatives;
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
	double conditionNegative = mTrueNegatives + mFalseNegatives;
	if (conditionNegative > 0)
		specificity = mTrueNegatives / conditionNegative;
	return specificity;
}

cv::Point_<double> TestResults::getROCPoint() {
	return cv::Point_<double>(getFallout(), getSensitivity());
}