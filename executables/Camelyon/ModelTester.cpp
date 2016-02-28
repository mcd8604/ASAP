#include "ModelTester.h"
#include "opencv2/core.hpp"

ModelTester::ModelTester(){}

ModelTester::~ModelTester(){}

// For now, assume we are always loading an RTrees model
void ModelTester::loadModel(const std::string modelFile) {
	mModel = cv::ml::RTrees::load<cv::ml::RTrees>(modelFile);
}

TestResults ModelTester::Test(const std::string outputFile) {
	// TODO: pass in slide object, get features and ground truth
	cv::Mat features;
	cv::Mat groundTruth;

	// TODO: assert model file
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::SampleTypes::ROW_SAMPLE, groundTruth);
	cv::Mat resp;
	float errorCalc = mModel->calcError(trainData, false, resp);
	TestResults testResults(resp, groundTruth);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "Error" << errorCalc;
	fs << "ConfusionMatrix" << testResults.getConfusionMatrix();
	fs << "ROCPoint" << testResults.getROCPoint();
	fs << "ResultMatrix" << resp;
	fs.release();

	return testResults;
}

TestResults ModelTester::Predict(const std::string outputFile) {
	// TODO: pass in slide object, get features
	cv::Mat features;
	cv::Mat results;
	mModel->predict(features, results);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "PredictionResult" << results;
	fs.release();
	return TestResults(results);
}