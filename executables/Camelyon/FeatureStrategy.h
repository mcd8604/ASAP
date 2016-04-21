#include <vector>
#include <string>
#include "core/Patch.hpp"

#pragma once
class FeatureStrategy
{
public:
	virtual int getNumFeatures() = 0;
	virtual std::vector<std::string> getFeatureNames() = 0;
	virtual std::vector<float> constructFeatures(Patch<double> &tilePatch) = 0;
};