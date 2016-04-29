#include "ModelTester.h"
#include "SlideLoader.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

ModelTester::ModelTester(){}

ModelTester::~ModelTester(){}

void ModelTester::loadRFModel(const std::string modelFile) {
	mModel = cv::ml::StatModel::load<cv::ml::RTrees>(modelFile);
}

void ModelTester::loadSVMModel(const std::string modelFile) {
	mModel = cv::ml::StatModel::load<cv::ml::SVM>(modelFile);
}

cv::Vec3f thresholdGetColor(const float value) {
	cv::Vec3f color;
	if (value > 0.5) {
		//red
		color = cv::Vec3f(0, 0, value);
	}
	else if (value > 0.25) {
		//yellow
		color = cv::Vec3f(value, 1, 1);
	}
	else {
		//green
		color = cv::Vec3f(0, 1, value);
	}
	return color;
}

cv::Mat ModelTester::renderHeatMap(const std::vector<cv::Rect> segments, const cv::Mat &predictions) {
	// TODO: parameterize size
	cv::Mat heatMap = cv::Mat::zeros(cv::Size(191, 432), CV_32FC3);
	for (int i = 0; i < segments.size(); ++i) {
		cv::Rect segment = segments[i];
		float prediction = predictions.at<float>(i);
		cv::Point p = segment.tl() / 512;
		heatMap.at<cv::Vec3f>(p) = thresholdGetColor(prediction);
	}
	cv::Mat outImg;
	heatMap *= 255;
	heatMap.convertTo(outImg, CV_8UC3);
	return heatMap;
}

TestResults ModelTester::Test(const std::string slideDir, const std::string outputDir) {
	std::vector<std::string> slideNames;
	std::vector<Slide> slides;
	SlideLoader::loadSlides(slideDir, slides, slideNames);

	cv::Mat totalPredictions;
	cv::Mat totalGroundTruth;
	for (int i = 0; i < slides.size(); ++i) {
		Slide slide = slides[i];
		cv::Mat predictions = Test(slide, outputDir + "/" + slideNames[i] + "_predictions.yaml");
		cv::Mat groundTruth = slide.getGroundTruth();
		std::string groundTruthLoc = outputDir + "/" + slideNames[i] + "_groundTruth.bmp";
		cv::imwrite(groundTruthLoc, renderHeatMap(slide.getTissueTiles(), groundTruth));
		std::string heatMapLoc = outputDir + "/" + slideNames[i] + "_heatMap.bmp";
		cv::imwrite(heatMapLoc, renderHeatMap(slide.getTissueTiles(), predictions));
		totalPredictions.push_back(predictions);
		totalGroundTruth.push_back(groundTruth);
	}

	TestResults testResults(totalPredictions, totalGroundTruth);
	cv::imwrite(outputDir + "/ROC.bmp", testResults.plotROC(1000));
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