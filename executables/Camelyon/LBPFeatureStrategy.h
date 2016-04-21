#include <vector>
#include <string>
#include "FeatureStrategy.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"

#pragma once

// This strategy is based on the following paper:
//
// Ojala, T.; Pietikainen, M.; Maenpaa, T., 
// "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," 
// Pattern Analysis and Machine Intelligence, IEEE Transactions on, 
// vol.24, no.7, pp.971,987, Jul 2002

class LBPFeatureStrategy : public FeatureStrategy
{
public:
	LBPFeatureStrategy() : mIsUniform(true) { init(); }
	LBPFeatureStrategy(bool isUniform) : mIsUniform(isUniform) { init(); }
	~LBPFeatureStrategy() { delete mFilter; }
	int getNumFeatures() { return mIsUniform ? 10 : 256; }
	std::vector<std::string> getFeatureNames() { return mFeatureNames; }
	std::vector<float> constructFeatures(Patch<double> &tilePatch);
private:
	bool mIsUniform;
	std::vector<std::string> mFeatureNames;
	ColorDeconvolutionFilter<double> *mFilter;
	void init();
	unsigned char getLBPUniformVal(const bool lbp[8]);
};