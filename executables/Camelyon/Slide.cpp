#include <vector>
#include <fstream>

#include "Slide.h"

Slide::Slide() {}

Slide::Slide(std::string dataFile)
{
	loadFromDataFile(dataFile);
}

Slide::~Slide() { }

void Slide::loadFromDataFile(std::string dataFile) {
	cv::FileStorage fs(dataFile, cv::FileStorage::Mode::READ);
	//fs["imageName"] >> mName;
	fs["tissueTiles"] >> mTissueTiles;
	fs["features"] >> mFeatures;
	fs["groundTruth"] >> mGroundTruth;
	fs.release();
}

void Slide::saveToDataFile(std::string dataFile) {
	cv::FileStorage fs(dataFile, cv::FileStorage::WRITE);
	fs << "tissueTiles" << mTissueTiles;
	fs << "features" << mFeatures;
	//if (!mGroundTruthMat.empty()) {
	fs << "groundTruth" << mGroundTruth;
	//}
	fs.release();
}

cv::Mat Slide::getFeatures() {
	return mFeatures;
}

cv::Mat Slide::getGroundTruth() {
	return mGroundTruth;
}

std::vector<cv::Rect> Slide::getTissueTiles() {
	return mTissueTiles;
}

void Slide::setFeatures(cv::Mat features) {
	mFeatures = features;
}

void Slide::setGroundTruth(cv::Mat groundTruth) {
	mGroundTruth = groundTruth;
}

void Slide::setTissueTiles(std::vector<cv::Rect> tissueTiles) {
	mTissueTiles = tissueTiles;
}