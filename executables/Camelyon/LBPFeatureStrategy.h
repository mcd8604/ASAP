#include <vector>
#include <string>
#include "FeatureStrategy.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"

#pragma once
class LBPFeatureStrategy : public FeatureStrategy
{
public:
	LBPFeatureStrategy() : mNumBins(256) { init(); }
	LBPFeatureStrategy(unsigned int numBins) : mNumBins(numBins) { init(); }
	~LBPFeatureStrategy() { delete mFilter; }
	int getNumFeatures() { return 256; }
	std::vector<std::string> getFeatureNames() { return mFeatureNames; }
	std::vector<float> constructFeatures(Patch<double> tilePatch);
private:
	unsigned int mNumBins;
	std::vector<std::string> mFeatureNames;
	ColorDeconvolutionFilter<double> *mFilter;
	void init();
};