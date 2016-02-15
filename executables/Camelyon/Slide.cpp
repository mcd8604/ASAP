#include <vector>
#include "Slide.h"
#include "imgproc/generic/ColorDeconvolutionFilter.h"
#include "imgproc/opencv/DIAGPathologyOpenCVBridge.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;

Slide::Slide()
{
}

Slide::~Slide()
{
}

void Slide::setAnnotationList(shared_ptr<AnnotationList> annoList)
{
	mAnnoList = annoList;
}

vector<Rect> Slide::getTissueTiles(MultiResolutionImage * mImage, int level, Size nativeTileSize) {
	vector<Rect> tissueTiles;

	CV_Assert(level < mImage->getNumberOfLevels());
	vector<unsigned long long, allocator<unsigned long long>> levelDim = mImage->getLevelDimensions(level);
	Patch<uchar> levelPatch = mImage->getPatch<uchar>(0, 0, levelDim[0], levelDim[1], level);

	// filter will convert level patch to thresholded density map for Hematoxylin stain
	ColorDeconvolutionFilter<uchar> *filter = new ColorDeconvolutionFilter<uchar>();
	// TODO adjust filter levels - minor non-tissue artifacts are present
	//filter->setGlobalDensityThreshold(0.25);
	Patch<double> hemaPatch;
	filter->filter(levelPatch, hemaPatch);
	Mat hema = patchToMat(hemaPatch);
	delete filter;

	double downsample = mImage->getLevelDownsample(level);
	int w = nativeTileSize.width / downsample;
	int h = nativeTileSize.height / downsample;
	int numTilesX = levelDim[0] / w;
	int numTilesY = levelDim[1] / h;

	for (int y = 0; y < numTilesY; y++) {
		for (int x = 0; x < numTilesX; x++) {
			Rect r(x * w, y * h, w, h);
			// Classify Foreground/Background
			// The hema patch has binary values 
			// (ok, apparently not..)
			// TODO investigate ColorDeconFilter
			// so this yields the percent of foreground pixels
			Scalar sc = sum(hema(r));
			double c1 = sc[0];
			double ar = r.area();
			double percentForeground = c1 / ar;
			if (percentForeground > 0.1) {
				tissueTiles.push_back(Rect(Point(x * nativeTileSize.width, y * nativeTileSize.height), nativeTileSize));
			}
		}
	}
	return tissueTiles;
}