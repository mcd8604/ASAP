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

class ModelTester {

public:
	ModelTester();
	ModelTester(std::string filePath);
	~ModelTester();

	void loadModel(const std::string modelFile);
	TestResults Test(const std::string outputFile);
	TestResults Predict(const std::string outputFile);
private:
	cv::Ptr<cv::ml::StatModel> mModel;
};