#include "ModelTrainer.h"
#include "boost\filesystem.hpp"
#include "boost\algorithm\string.hpp"

using namespace std;
using namespace boost;
using namespace boost::filesystem;

ModelTrainer::ModelTrainer(){}

ModelTrainer::ModelTrainer(string dirPath) : mDirPath(dirPath) { }

ModelTrainer::~ModelTrainer(){}

/*
* Load all files in mDirPath with the extension .xml or .yaml as Slide objects
*/
vector<Slide> ModelTrainer::loadSlides() {
	vector<Slide> slides;
	path dir(mDirPath);
	try {
		if (exists(dir) && is_directory(dir)) {
			for (directory_entry& entry : directory_iterator(dir)) {
				path file = entry.path();
				if (is_regular_file(file)) {
					string ext = to_upper_copy(file.extension().generic_string());
					if (ext == ".YAML" || ext == ".XML") {
						string imgPath = file.generic_string();
						Slide s(imgPath);
						slides.push_back(s);
					}
				}
			}
		}
	}
	catch (const filesystem_error& ex) {
		cerr << ex.what() << '\n';
	}
	return slides;
}

/*
* Loads all Slide files in mDirPath and combines the data into a single TrainData.
* Assumes all Slide files contain the same order of features.
*/
cv::Ptr<cv::ml::TrainData> ModelTrainer::loadTrainingData() {
	cv::Ptr<cv::ml::TrainData> trainData;
	vector<Slide> slides = loadSlides();
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
			cv::Mat groundTruth = cv::Mat(rows, 1, CV_32S);
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
	rf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, nTrees, 0));
	rf->train(trainData);
	rf->save(outputFilePath);
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