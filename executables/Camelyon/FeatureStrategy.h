#include <vector>
#include <string>
#include "core/Patch.hpp"

#pragma once
class FeatureStrategy
{
public:
	virtual int getNumFeatures();
	virtual std::vector<std::string> getFeatureNames();
	virtual std::vector<float> constructFeatures(Patch<double> tilePatch);
};