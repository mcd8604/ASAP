#include "ModelTester.h"
#include "opencv2/core.hpp"

ModelTester::ModelTester(){}

ModelTester::~ModelTester(){}



cv::Mat ModelTester::Test(const std::string rfModelFile, const std::string outputFile) {
	// TODO assert model file
	/*cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::load<cv::ml::RTrees>(rfModelFile);
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(mFeatures, cv::ml::SampleTypes::ROW_SAMPLE, mGroundTruthMat);
	cv::Mat resp;
	float errorCalc = rf->calcError(trainData, false, resp);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	printf("RF Test error: %f\n", errorCalc);
	fs << "TestResult" << resp;
	fs.release();

	// TEST
	for (int i = 0; i < mNativeTissueTiles.size(); i++) {
		cv::Rect r = mNativeTissueTiles[i];
		float gt = mGroundTruthMat.at<float>(i, 0);
		float test = resp.at<float>(i, 0);
		if (gt != 0 && test != 0) {
			Patch<uchar> p = mImage->getPatch<uchar>(r.x, r.y, r.width, r.height, 0);
			cv::Mat m = patchToMat(p);
			//groundTruthImage.
			cv::imshow(std::to_string(r.x) + "," + std::to_string(r.y), m);
			//Scalar bSum = sum(m);
			//groundTruthMat.at<float>(i, 0) = bSum[0] == 0 ? 0 : 1;// bSum[0] / r.area();
			cv::waitKey();
		}
	}

	*/
	return cv::Mat();//resp;
}

cv::Mat ModelTester::Predict(const std::string rfModelFile, const std::string outputFile) {
	// TODO assert model file
	/*cv::Ptr<cv::ml::RTrees> rf = cv::ml::RTrees::load<cv::ml::RTrees>(rfModelFile);
	cv::Mat results;
	rf->predict(mFeatures, results);
	cv::FileStorage fs(outputFile, cv::FileStorage::Mode::WRITE);
	fs << "PredictionResult" << results;
	fs.release();*/
	return cv::Mat();//results;
}