#include <string>
#include <vector>
#include "Slide.h"

#pragma once
class SlideLoader
{
public:
	SlideLoader();
	static void loadSlides(const std::string dirPath, std::vector<Slide> &slides, std::vector<std::string> &slideNames);
};

