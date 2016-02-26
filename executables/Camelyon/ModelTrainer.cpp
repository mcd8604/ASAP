#include "ModelTrainer.h"

ModelTrainer::ModelTrainer(){}

ModelTrainer::~ModelTrainer(){}


void ModelTrainer::TrainModel(const std::string outputFile) {
	/*cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(mFeatures, cv::ml::SampleTypes::ROW_SAMPLE, mGroundTruthMat);
	trainData->setTrainTestSplitRatio(0.8);
	std::cout << "Test/Train: " << trainData->getNTestSamples() << "/" << trainData->getNTrainSamples() << "\n";
	cv::Scalar s = sum(mGroundTruthMat);

	cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::create();
	rf->setCalculateVarImportance(true);
	rf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 0));
	bool ok = rf->train(trainData);
	if (!ok)
	{
		printf("Training failed\n");
	}
	else
	{
		printf("train error: %f\n", rf->calcError(trainData, false, cv::noArray()));
		printf("test error: %f\n\n", rf->calcError(trainData, true, cv::noArray()));
	}
	rf->save(outputFile);

	// Print variable importance
	cv::Mat var_importance = rf->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}*/
}