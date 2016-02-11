#include <string>
#include <vector>
#include "Segment.h"

#pragma once
class Slide
{
public:
	Slide();
	~Slide();

	/*
	*	Prints a CSV file with a header.
	*	Each row contains the segment id and feature vector.
	*		Ex:
	*			SegmentID, Feature1, Feature2, ..., FeatureN
	*	It is assumed that each segment has a unique ID and 
	*	contains an equal number of features.
	*/
	bool segFeatsToCSV(std::string filePath);

	void setPredictions();
private:
	std::string id;
	std::vector<Segment> segments;
};

