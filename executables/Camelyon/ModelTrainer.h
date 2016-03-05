/*******************************************************
Inputs: directory path of selector model and FSFs
Outputs: Model

assumptions: 
********************************************************/

#pragma once

#include <string>
#include <vector>
#include "opencv2\ml.hpp"
#include "Slide.h"

class ModelTrainer{

public:
	ModelTrainer();
	ModelTrainer(std::string dirPath);
	~ModelTrainer();
	void trainRF(const std::string outputFilePath, int nTrees, int maxDepth);
	void trainModel(cv::Ptr<cv::ml::StatModel> model);
private:
	std::string mDirPath;
	cv::Ptr<cv::ml::TrainData> loadTrainingData();
	std::vector<Slide> loadSlides();
};