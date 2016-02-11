#include <vector>
#include "opencv2/imgproc/imgproc.hpp"

#pragma once
class Segment
{
public:
	Segment();
	~Segment();
private:
	int id;
	int depth;
	std::vector<cv::Point> contour;
	Segment *parent;
	std::vector<Segment> children;
	bool groundTruth;
	double prediction;
};

