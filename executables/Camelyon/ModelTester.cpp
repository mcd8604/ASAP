#include "ModelTester.h"
#include "opencv2/core.hpp"

ModelTester::ModelTester(){}

ModelTester::~ModelTester(){}

TestResults ModelTester::Test(const std::string rfModelFile, const std::string outputFile) {
	// TODO: pass in slide object, get features and ground truth
	cv::Mat features;
	cv::Mat groundTruth;

	// TODO: assert model file
	cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::load<cv::ml::RTrees>(rfModelFile);
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::SampleTypes::ROW_SAMPLE, groundTruth);
	cv::Mat resp;
	float errorCalc = rf->calcError(trainData, false, resp);
	TestResults testResults(resp, groundTruth);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "Error" << errorCalc;
	fs << "ConfusionMatrix" << testResults.getConfusionMatrix();
	fs << "ROCPoint" << testResults.getROCPoint();
	fs << "ResultMatrix" << resp;
	fs.release();

	return testResults;
}

TestResults ModelTester::Predict(const std::string rfModelFile, const std::string outputFile) {
	// TODO: pass in slide object, get features
	cv::Mat features;

	// TODO assert model file
	cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::load<cv::ml::RTrees>(rfModelFile);
	cv::Mat results;
	rf->predict(features, results);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "PredictionResult" << results;
	fs.release();
	return TestResults(results);
}