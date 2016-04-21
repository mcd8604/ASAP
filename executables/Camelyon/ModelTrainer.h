/*******************************************************
Inputs: directory path of selector model and FSFs
Outputs: Model

assumptions: 
********************************************************/

#pragma once

#include <string>
#include <vector>
#include "opencv2\ml.hpp"
//#include "alglib\dataanalysis.h"
#include "Slide.h"

class ModelTrainer{

public:
	ModelTrainer();
	ModelTrainer(std::string dirPath);
	~ModelTrainer();
	void trainRF(const std::string outputFilePath, int nTrees, int maxDepth);
	void trainSVM(const std::string outputFilePath);
	void trainModel(cv::Ptr<cv::ml::StatModel> model);
	//void trainRF_alglib(const std::string outputFilePath, int nTrees, bool isClassifier);
private:
	std::string mDirPath;
	cv::Ptr<cv::ml::TrainData> loadTrainingData();
	//alglib::real_2d_array loadTrainingData_alglib(bool isClassifier);
	std::vector<Slide> loadSlides();
};