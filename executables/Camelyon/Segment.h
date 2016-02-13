#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#pragma once
class Segment
{
public:
	Segment(int id);
	~Segment();
	void setContour(std::vector<cv::Point> contour);
	void constructFeatures(cv::Mat m, cv::Ptr<cv::Feature2D> featureDetector);
	int getId() { return mId; };
	std::vector<double> getFeatures() { return features; };
private:
	int mId;
	int mDepth;
	std::vector<cv::Point> mContour;
	Segment *mParent;
	std::vector<Segment> mChildren;
	std::vector<double> features;
	bool mGroundTruth;
	double mPrediction;
};

