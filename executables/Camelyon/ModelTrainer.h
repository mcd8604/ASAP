/*******************************************************
Inputs: directory path of selector model and FSFs
Outputs: Model

assumptions: 
********************************************************/

#pragma once

#include <string>

class ModelTrainer{

public:
	ModelTrainer();
	ModelTrainer(std::string filePath);
	~ModelTrainer();

	void TrainModel(const std::string outputFile);

};