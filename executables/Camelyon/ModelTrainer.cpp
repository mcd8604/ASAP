#include "ModelTrainer.h"
#include "SlideLoader.h"
#include "boost\filesystem.hpp"
#include "boost\algorithm\string.hpp"

using namespace std;
using namespace boost;
using namespace boost::filesystem;

ModelTrainer::ModelTrainer(){}

ModelTrainer::ModelTrainer(string dirPath) : mDirPath(dirPath) { }

ModelTrainer::~ModelTrainer(){}

/*
* Loads all Slide files in mDirPath and combines the data into a single TrainData.
* Assumes all Slide files contain the same order of features.
*/
cv::Ptr<cv::ml::TrainData> ModelTrainer::loadTrainingData() {
	cv::Ptr<cv::ml::TrainData> trainData;
	vector<Slide> slides;
	vector<string> slideNames;
	SlideLoader::loadSlides(mDirPath, slides, slideNames);
	int numSlides = slides.size();
	if (numSlides > 0) {
		int rows = 0;
		int cols = 0;
		for (Slide slide : slides) {
			rows += slide.getFeatures().rows;
			cols = slide.getFeatures().cols;
		}
		if (rows > 0 && cols > 0) {
			cv::Mat features = cv::Mat(rows, cols, CV_32F);
			uchar *fPtr = features.ptr();
			cv::Mat groundTruth = cv::Mat(rows, 1, CV_32F);
			uchar *gtPtr = groundTruth.ptr();
			for (Slide slide : slides) {
				cv::Mat slideFeatures = slide.getFeatures();
				const uchar* fsPtr = slideFeatures.datastart;
				while (fsPtr != slideFeatures.dataend) {
					*fPtr = *fsPtr;
					fPtr++;
					fsPtr++;
				}
				cv::Mat slideGroundTruth = slide.getGroundTruth();
				const uchar* gtsPtr = slideGroundTruth.datastart;
				while (gtsPtr != slideGroundTruth.dataend) {
					*gtPtr = *gtsPtr;
					gtPtr++;
					gtsPtr++;
				}
			}
			trainData = cv::ml::TrainData::create(features, cv::ml::SampleTypes::ROW_SAMPLE, groundTruth);
		}
	}
	return trainData;
}

/*
* Training a random forest
*/
void ModelTrainer::trainRF(const std::string outputFilePath, int nTrees, int maxDepth) {
	cv::Ptr<cv::ml::TrainData> trainData = loadTrainingData();
	cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::create();
	rf->setCalculateVarImportance(true);
	rf->setMaxDepth(maxDepth);
	rf->setRegressionAccuracy(0.0001);
	rf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, nTrees, 0.0001));
	if(rf->train(trainData))
		rf->save(outputFilePath);
}

void ModelTrainer::trainSVM(const std::string outputFilePath) {
	cv::Ptr<cv::ml::TrainData> trainData = loadTrainingData();
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	svm->setType(cv::ml::SVM::Types::EPS_SVR);
	svm->setP(0.01);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT, 1000, 1e-6));
	if(svm->train(trainData))
		svm->save(outputFilePath);
}

/*
* Train any given model if it is not already trained
*/
void ModelTrainer::trainModel(cv::Ptr<cv::ml::StatModel> model) {
	if (model && !model->isTrained()) {
		cv::Ptr<cv::ml::TrainData> trainData = loadTrainingData();
		model->train(trainData);
	}
}