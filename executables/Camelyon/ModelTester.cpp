#include "ModelTester.h"
#include "SlideLoader.h"
#include "opencv2/core.hpp"

ModelTester::ModelTester(){}

ModelTester::~ModelTester(){}

void ModelTester::loadRFModel(const std::string modelFile) {
	mModel = cv::ml::StatModel::load<cv::ml::RTrees>(modelFile);
}

void ModelTester::loadSVMModel(const std::string modelFile) {
	mModel = cv::ml::StatModel::load<cv::ml::SVM>(modelFile);
}

TestResults ModelTester::Test(const std::string slideDir) {
	std::vector<std::string> slideNames;
	std::vector<Slide> slides;
	SlideLoader::loadSlides(slideDir, slides, slideNames);

	cv::Mat totalPredictions;
	cv::Mat totalGroundTruth;
	for (int i = 0; i < slides.size(); ++i) {
		Slide slide = slides[i];
		cv::Mat predictions = Test(slide, slideDir + "/" + slideNames[i] + "_testResult.yaml");
		cv::Mat groundTruth = slide.getGroundTruth();
		totalPredictions.push_back(predictions);
		totalGroundTruth.push_back(groundTruth);
	}

	TestResults testResults(totalPredictions, totalGroundTruth);
	return testResults;
}

cv::Mat ModelTester::Test(Slide slide, const std::string outputFile) {
	cv::Mat features = slide.getFeatures();
	cv::Mat results = cv::Mat::zeros(features.rows, 1, CV_32F);
	for (int i = 0; i < features.rows; i++)	
		results.at<float>(i) = mModel->predict(features.row(i));
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "ResultVector" << results;
	return results;
}

cv::Mat ModelTester::Predict(Slide slide, const std::string outputFile) {
	cv::Mat features = slide.getFeatures();
	cv::Mat results;
	mModel->predict(features, results);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "PredictionResult" << results;
	fs.release();
	return results;
}