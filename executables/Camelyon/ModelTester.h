/*******************************************************
Inputs: directory with Models , FSFs
Outputs: Test Results

assumptions: FSFs have no ground truths also assume a model
follows naming convention model_* every other file is FSF
********************************************************/


#pragma once

#include <string>
#include "opencv2\ml.hpp"

class ModelTester {

public:
	ModelTester();
	ModelTester(std::string filePath);
	~ModelTester();

	cv::Mat Test(const std::string rfModelFile, const std::string outputFile);
	cv::Mat Predict(const std::string rfModelFile, const std::string outputFile);

};