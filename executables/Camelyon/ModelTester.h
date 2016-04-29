/*******************************************************
Inputs: directory with Models , FSFs
Outputs: Test Results

assumptions: FSFs have no ground truths also assume a model
follows naming convention model_* every other file is FSF
********************************************************/


#pragma once

#include <string>
#include "opencv2\ml.hpp"
#include "TestResults.h"
#include "Slide.h"

class ModelTester {

public:
	ModelTester();
	ModelTester(std::string filePath);
	~ModelTester();

	void ModelTester::loadSVMModel(const std::string modelFile);
	void ModelTester::loadRFModel(const std::string modelFile);
	TestResults Test(const std::string slideDir, const std::string outputDir);
	cv::Mat Test(Slide slide, const std::string outputFile);
	cv::Mat Predict(Slide slide, const std::string outputFile);
private:
	cv::Ptr<cv::ml::StatModel> mModel;
	cv::Mat renderHeatMap(const std::vector<cv::Rect> segments, const cv::Mat &predictions);
};