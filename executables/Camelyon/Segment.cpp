#include "Segment.h"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

Segment::Segment(int id)
{
	mId = id;
}

Segment::~Segment()
{
}

void Segment::setContour(std::vector<cv::Point> contour)
{
	mContour = contour;
}

// (move this to wrapper class)
void Segment::constructFeatures(Mat m, Ptr<Feature2D> featureDetector) {
	vector<KeyPoint> keyPoints;
	featureDetector->detect(m, keyPoints);
	// temporary - for now just throw the number of keyPoints as a "feature"
	features.push_back(keyPoints.size());
	// TODO - create wrapper class: FeatureExtractor
	//		- runs a number of (configurable) feature detectors and is in charge of mapping
	//			the various results into a linear vector of (named) feature data
	//		- performs feature selection
}
